from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from apps.agentic_rag.app.grading import (
    CorrectiveQueryRewriter,
    EvidenceGrade,
    EvidenceGrader,
    HeuristicEvidenceGrader,
    KeywordFallbackQueryRewriter,
)
from apps.agentic_rag.app.planning import (
    HeuristicQueryPlanner,
    QueryPlan,
    QueryPlanner,
)
from apps.agentic_rag.app.schemas import (
    AgentTrace,
    AnswerResponse,
    AskRequest,
    Citation,
    EvidenceReview,
    PlannedStep,
    RetrievedChunk,
    ToolCall,
)
from apps.agentic_rag.app.tools import (
    RetrievalTool,
    RetrievedEvidence,
    merge_evidence,
)
from apps.hybrid_rag.app.generation import (
    ABSTENTION_MESSAGE,
    AnswerGenerator,
    build_context_sources,
    format_context,
)
from apps.hybrid_rag.app.retrieval import ScoredDocument


@dataclass(frozen=True)
class RetrievalAttempt:
    attempt: int
    query: str
    plan: QueryPlan
    evidence_groups: tuple[tuple[RetrievedEvidence, ...], ...]
    evidence: tuple[RetrievedEvidence, ...]
    grade: EvidenceGrade
    corrective_query: str | None = None


class AgenticRAGService:
    def __init__(
        self,
        tools: Sequence[RetrievalTool],
        generator: AnswerGenerator,
        *,
        planner: QueryPlanner | None = None,
        evidence_grader: EvidenceGrader | None = None,
        query_rewriter: CorrectiveQueryRewriter | None = None,
        default_top_k: int = 5,
        max_correction_attempts: int = 1,
    ) -> None:
        if default_top_k <= 0:
            raise ValueError(
                "default_top_k must be greater than 0"
            )

        if max_correction_attempts < 0:
            raise ValueError(
                "max_correction_attempts must not be negative"
            )

        if not tools:
            raise ValueError("tools must not be empty")

        self.tools = {
            tool.name: tool
            for tool in tools
        }
        self.generator = generator
        self.planner = planner or HeuristicQueryPlanner()
        self.evidence_grader = evidence_grader or HeuristicEvidenceGrader()
        self.query_rewriter = query_rewriter or KeywordFallbackQueryRewriter()
        self.default_top_k = default_top_k
        self.max_correction_attempts = max_correction_attempts

    def ask(
        self,
        request: AskRequest,
    ) -> AnswerResponse:
        top_k = (
            request.top_k
            or self.default_top_k
        )
        attempts = self._retrieve_with_correction(
            request.query,
            top_k=top_k,
        )
        final_attempt = attempts[-1]
        evidence = final_attempt.evidence
        retrieved_chunks = to_retrieved_chunks(evidence)
        trace = build_agent_trace(attempts)

        if not evidence or not final_attempt.grade.is_sufficient:
            return AnswerResponse(
                query=request.query,
                answer=ABSTENTION_MESSAGE,
                citations=[],
                retrieved_chunks=retrieved_chunks,
                agent_trace=trace,
            )

        scored_documents = [
            ScoredDocument(
                document=item.document,
                score=item.score,
            )
            for item in evidence
        ]
        context = format_context(
            build_context_sources(scored_documents)
        )
        answer = self.generator.generate(
            request.query,
            context,
        )
        citations = extract_citations(
            answer,
            retrieved_chunks,
        )

        return AnswerResponse(
            query=request.query,
            answer=answer,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            agent_trace=trace,
        )

    def _retrieve_with_correction(
        self,
        query: str,
        *,
        top_k: int,
    ) -> tuple[RetrievalAttempt, ...]:
        attempts: list[RetrievalAttempt] = []
        attempted_query = query

        for attempt_number in range(
            1,
            self.max_correction_attempts + 2,
        ):
            plan = self.planner.plan(
                attempted_query,
                tuple(self.tools),
                top_k=top_k,
            )
            evidence_groups = self._execute_plan(plan)
            evidence = merge_evidence(
                evidence_groups,
                limit=top_k,
            )
            grade = self.evidence_grader.grade(
                query,
                attempted_query,
                evidence,
            )
            corrective_query = None

            if (
                not grade.is_sufficient
                and attempt_number <= self.max_correction_attempts
            ):
                rewritten_query = self.query_rewriter.rewrite(
                    query,
                    attempted_query,
                    grade,
                ).strip()

                if (
                    rewritten_query
                    and rewritten_query.lower() != attempted_query.lower()
                ):
                    corrective_query = rewritten_query

            attempts.append(
                RetrievalAttempt(
                    attempt=attempt_number,
                    query=attempted_query,
                    plan=plan,
                    evidence_groups=evidence_groups,
                    evidence=evidence,
                    grade=grade,
                    corrective_query=corrective_query,
                )
            )

            if grade.is_sufficient or corrective_query is None:
                break

            attempted_query = corrective_query

        return tuple(attempts)

    def _execute_plan(
        self,
        plan: QueryPlan,
    ) -> tuple[tuple[RetrievedEvidence, ...], ...]:
        evidence_groups: list[tuple[RetrievedEvidence, ...]] = []

        for step in plan.steps:
            tool = self.tools.get(step.tool_name)

            if tool is None:
                raise ValueError(
                    f"Planner selected unknown tool: {step.tool_name}"
                )

            evidence_groups.append(
                tool.search(
                    step.query,
                    top_k=step.top_k,
                )
            )

        return tuple(evidence_groups)


def build_agent_trace(
    attempts: Sequence[RetrievalAttempt],
) -> AgentTrace:
    return AgentTrace(
        original_query=attempts[0].plan.original_query,
        planned_steps=[
            PlannedStep(
                tool_name=step.tool_name,
                query=step.query,
                top_k=step.top_k,
                reason=step.reason,
            )
            for attempt in attempts
            for step in attempt.plan.steps
        ],
        tool_calls=[
            ToolCall(
                tool_name=step.tool_name,
                query=step.query,
                top_k=step.top_k,
                retrieved_chunk_ids=[
                    evidence.chunk_id
                    for evidence in evidence_group
                ],
            )
            for attempt in attempts
            for step, evidence_group in zip(
                attempt.plan.steps,
                attempt.evidence_groups,
                strict=True,
            )
        ],
        evidence_reviews=[
            EvidenceReview(
                attempt=attempt.attempt,
                query=attempt.query,
                is_sufficient=attempt.grade.is_sufficient,
                score=attempt.grade.score,
                matched_terms=list(attempt.grade.matched_terms),
                missing_terms=list(attempt.grade.missing_terms),
                reason=attempt.grade.reason,
                corrective_query=attempt.corrective_query,
            )
            for attempt in attempts
        ],
    )


def to_retrieved_chunks(
    evidence: Sequence[RetrievedEvidence],
) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            rank=rank,
            score=item.score,
            chunk_id=item.chunk_id,
            document_id=str(
                item.document.metadata.get(
                    "document_id",
                    "unknown",
                )
            ),
            source=str(
                item.document.metadata.get(
                    "source",
                    "unknown",
                )
            ),
            content=item.document.page_content,
            metadata=dict(item.document.metadata),
            tool_name=item.tool_name,
            tool_rank=item.tool_rank,
        )
        for rank, item in enumerate(
            evidence,
            start=1,
        )
    ]


def extract_citations(
    answer: str,
    retrieved_chunks: Sequence[RetrievedChunk],
) -> list[Citation]:
    referenced_indexes = sorted(
        {
            int(value)
            for value in re.findall(
                r"\[(\d+)\]",
                answer,
            )
        }
    )

    citations: list[Citation] = []

    for index in referenced_indexes:
        if not 1 <= index <= len(retrieved_chunks):
            continue

        chunk = retrieved_chunks[index - 1]

        citations.append(
            Citation(
                index=index,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                source=chunk.source,
            )
        )

    return citations

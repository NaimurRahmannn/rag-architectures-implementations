from __future__ import annotations

import re
from collections.abc import Sequence

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


class AgenticRAGService:
    def __init__(
        self,
        tools: Sequence[RetrievalTool],
        generator: AnswerGenerator,
        *,
        planner: QueryPlanner | None = None,
        default_top_k: int = 5,
    ) -> None:
        if default_top_k <= 0:
            raise ValueError(
                "default_top_k must be greater than 0"
            )

        if not tools:
            raise ValueError("tools must not be empty")

        self.tools = {
            tool.name: tool
            for tool in tools
        }
        self.generator = generator
        self.planner = planner or HeuristicQueryPlanner()
        self.default_top_k = default_top_k

    def ask(
        self,
        request: AskRequest,
    ) -> AnswerResponse:
        top_k = (
            request.top_k
            or self.default_top_k
        )
        plan = self.planner.plan(
            request.query,
            tuple(self.tools),
            top_k=top_k,
        )
        evidence_groups = self._execute_plan(plan)
        evidence = merge_evidence(
            evidence_groups,
            limit=top_k,
        )
        retrieved_chunks = to_retrieved_chunks(evidence)
        trace = build_agent_trace(
            plan,
            evidence_groups,
        )

        if not evidence:
            return AnswerResponse(
                query=request.query,
                answer=ABSTENTION_MESSAGE,
                citations=[],
                retrieved_chunks=[],
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
    plan: QueryPlan,
    evidence_groups: Sequence[Sequence[RetrievedEvidence]],
) -> AgentTrace:
    return AgentTrace(
        original_query=plan.original_query,
        planned_steps=[
            PlannedStep(
                tool_name=step.tool_name,
                query=step.query,
                top_k=step.top_k,
            )
            for step in plan.steps
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
            for step, evidence_group in zip(
                plan.steps,
                evidence_groups,
                strict=True,
            )
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

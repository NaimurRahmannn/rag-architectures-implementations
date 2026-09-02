import pytest
from langchain_core.documents import Document

from apps.agentic_rag.app.planning import QueryPlan, RetrievalStep
from apps.agentic_rag.app.schemas import AskRequest
from apps.agentic_rag.app.service import AgenticRAGService
from apps.agentic_rag.app.tools import RetrievalTool
from apps.hybrid_rag.app.generation import ABSTENTION_MESSAGE
from apps.hybrid_rag.app.retrieval import ScoredDocument


class StaticPlanner:
    def __init__(self, steps: tuple[RetrievalStep, ...]) -> None:
        self.steps = steps
        self.calls: list[tuple[str, tuple[str, ...], int]] = []

    def plan(
        self,
        query: str,
        available_tools: tuple[str, ...],
        *,
        top_k: int,
    ) -> QueryPlan:
        self.calls.append((query, available_tools, top_k))
        return QueryPlan(original_query=query, steps=self.steps)


class StaticRetriever:
    def __init__(self, results: list[ScoredDocument]) -> None:
        self.results = results

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[ScoredDocument]:
        return self.results[:top_k]


class RecordingGenerator:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def generate(self, query: str, context: str) -> str:
        self.calls.append((query, context))
        return self.answer


def make_result(
    chunk_id: str,
    score: float,
    content: str,
) -> ScoredDocument:
    return ScoredDocument(
        document=Document(
            page_content=content,
            metadata={
                "chunk_id": chunk_id,
                "document_id": "authentication-guide",
                "source": "authentication_guide.md",
            },
        ),
        score=score,
    )


def test_service_executes_plan_generates_answer_citations_and_trace() -> None:
    shared = make_result("chunk-shared", 0.8, "Recovery codes are single use.")
    dense_result = make_result("chunk-dense", 0.9, "Store codes securely.")
    bm25_result = make_result("chunk-bm25", 12.0, "Rotate compromised codes.")
    planner = StaticPlanner(
        (
            RetrievalStep("dense", "recovery codes", 2),
            RetrievalStep("bm25", "AUTH-42", 2),
        )
    )
    generator = RecordingGenerator("Use secure storage [1] and rotate codes [3].")
    service = AgenticRAGService(
        tools=(
            RetrievalTool("dense", StaticRetriever([dense_result, shared])),
            RetrievalTool("bm25", StaticRetriever([shared, bm25_result])),
        ),
        generator=generator,
        planner=planner,
        default_top_k=3,
    )

    response = service.ask(AskRequest(query="How should I handle recovery codes?"))

    assert [chunk.chunk_id for chunk in response.retrieved_chunks] == [
        "chunk-dense",
        "chunk-shared",
        "chunk-bm25",
    ]
    assert [citation.chunk_id for citation in response.citations] == [
        "chunk-dense",
        "chunk-bm25",
    ]
    assert planner.calls == [
        ("How should I handle recovery codes?", ("dense", "bm25"), 3)
    ]
    assert len(response.agent_trace.planned_steps) == 2
    assert response.agent_trace.evidence_reviews[0].is_sufficient is True
    assert response.agent_trace.tool_calls[0].retrieved_chunk_ids == [
        "chunk-dense",
        "chunk-shared",
    ]
    assert response.agent_trace.tool_calls[1].retrieved_chunk_ids == [
        "chunk-shared",
        "chunk-bm25",
    ]
    assert generator.calls[0][0] == "How should I handle recovery codes?"
    assert "[1]" in generator.calls[0][1]
    assert "Chunk ID: chunk-dense" in generator.calls[0][1]
    assert "Store codes securely." in generator.calls[0][1]


def test_service_abstains_without_calling_generator_when_no_evidence() -> None:
    generator = RecordingGenerator("must not be returned")
    service = AgenticRAGService(
        tools=(RetrievalTool("dense", StaticRetriever([])),),
        generator=generator,
        planner=StaticPlanner((RetrievalStep("dense", "missing", 5),)),
    )

    response = service.ask(AskRequest(query="missing"))

    assert response.answer == ABSTENTION_MESSAGE
    assert response.citations == []
    assert response.retrieved_chunks == []
    assert generator.calls == []
    assert len(response.agent_trace.tool_calls) == 1


def test_service_ignores_unknown_and_duplicate_citation_numbers() -> None:
    generator = RecordingGenerator("Supported [1], repeated [1], invalid [99].")
    service = AgenticRAGService(
        tools=(
            RetrievalTool(
                "bm25",
                StaticRetriever([make_result("chunk-1", 1.0, "Evidence")]),
            ),
        ),
        generator=generator,
        planner=StaticPlanner((RetrievalStep("bm25", "query", 1),)),
    )

    response = service.ask(AskRequest(query="Evidence"))

    assert [citation.index for citation in response.citations] == [1]


def test_service_retries_with_corrective_query_when_evidence_is_weak() -> None:
    class QueryAwareRetriever:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        def search(
            self,
            query: str,
            top_k: int = 5,
        ) -> list[ScoredDocument]:
            self.calls.append((query, top_k))

            if query == "rotate recovery codes":
                return [
                    make_result(
                        "chunk-1",
                        0.9,
                        "Rotate recovery codes after compromise.",
                    )
                ]

            return []

    retriever = QueryAwareRetriever()
    generator = RecordingGenerator("Rotate recovery codes after compromise [1].")
    service = AgenticRAGService(
        tools=(RetrievalTool("dense", retriever),),
        generator=generator,
    )

    response = service.ask(
        AskRequest(query="How should I rotate recovery codes?")
    )

    assert response.answer == "Rotate recovery codes after compromise [1]."
    assert [call[0] for call in retriever.calls] == [
        "How should I rotate recovery codes?",
        "rotate recovery codes",
    ]
    assert [
        review.is_sufficient
        for review in response.agent_trace.evidence_reviews
    ] == [False, True]
    assert response.agent_trace.evidence_reviews[0].corrective_query == (
        "rotate recovery codes"
    )


def test_service_abstains_after_exhausting_corrective_retry() -> None:
    generator = RecordingGenerator("must not be returned")
    service = AgenticRAGService(
        tools=(
            RetrievalTool(
                "dense",
                StaticRetriever(
                    [make_result("chunk-1", 0.2, "Password reset policy.")]
                ),
            ),
        ),
        generator=generator,
    )

    response = service.ask(AskRequest(query="How should I configure MFA?"))

    assert response.answer == ABSTENTION_MESSAGE
    assert response.citations == []
    assert generator.calls == []
    assert len(response.agent_trace.evidence_reviews) == 2
    assert response.agent_trace.evidence_reviews[-1].is_sufficient is False


def test_service_rejects_unknown_planner_tool() -> None:
    service = AgenticRAGService(
        tools=(RetrievalTool("bm25", StaticRetriever([])),),
        generator=RecordingGenerator("answer"),
        planner=StaticPlanner((RetrievalStep("missing", "query", 1),)),
    )

    with pytest.raises(ValueError, match="unknown tool"):
        service.ask(AskRequest(query="query"))


def test_service_validates_required_tools_and_default_top_k() -> None:
    generator = RecordingGenerator("answer")

    with pytest.raises(ValueError, match="tools"):
        AgenticRAGService(tools=(), generator=generator)

    with pytest.raises(ValueError, match="default_top_k"):
        AgenticRAGService(
            tools=(RetrievalTool("bm25", StaticRetriever([])),),
            generator=generator,
            default_top_k=0,
        )

    with pytest.raises(ValueError, match="max_correction_attempts"):
        AgenticRAGService(
            tools=(RetrievalTool("bm25", StaticRetriever([])),),
            generator=generator,
            max_correction_attempts=-1,
        )

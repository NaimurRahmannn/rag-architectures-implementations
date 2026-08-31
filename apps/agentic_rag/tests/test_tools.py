import pytest
from langchain_core.documents import Document

from apps.agentic_rag.app.tools import RetrievalTool, merge_evidence
from apps.hybrid_rag.app.retrieval import ScoredDocument


class StaticRetriever:
    def __init__(self, results: list[ScoredDocument]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[ScoredDocument]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


def make_result(chunk_id: str, score: float) -> ScoredDocument:
    return ScoredDocument(
        document=Document(
            page_content=f"Content for {chunk_id}",
            metadata={
                "chunk_id": chunk_id,
                "document_id": "guide",
                "source": "guide.md",
            },
        ),
        score=score,
    )


def test_retrieval_tool_adds_tool_provenance_and_rank() -> None:
    retriever = StaticRetriever(
        [make_result("chunk-1", 0.9), make_result("chunk-2", 0.7)]
    )
    tool = RetrievalTool("dense", retriever)

    evidence = tool.search("reset password", top_k=2)

    assert retriever.calls == [("reset password", 2)]
    assert [item.chunk_id for item in evidence] == ["chunk-1", "chunk-2"]
    assert [item.tool_rank for item in evidence] == [1, 2]
    assert {item.tool_name for item in evidence} == {"dense"}


def test_merge_evidence_deduplicates_in_first_seen_order() -> None:
    dense = RetrievalTool(
        "dense",
        StaticRetriever(
            [make_result("chunk-1", 0.9), make_result("chunk-2", 0.8)]
        ),
    ).search("query", top_k=2)
    bm25 = RetrievalTool(
        "bm25",
        StaticRetriever(
            [make_result("chunk-2", 10.0), make_result("chunk-3", 8.0)]
        ),
    ).search("query", top_k=2)

    merged = merge_evidence((dense, bm25), limit=3)

    assert [item.chunk_id for item in merged] == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]
    assert [item.tool_name for item in merged] == ["dense", "dense", "bm25"]


def test_tool_and_merger_reject_non_positive_limits() -> None:
    tool = RetrievalTool("bm25", StaticRetriever([]))

    with pytest.raises(ValueError, match="top_k"):
        tool.search("query", top_k=0)

    with pytest.raises(ValueError, match="limit"):
        merge_evidence((), limit=0)


def test_tool_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="tool name"):
        RetrievalTool("  ", StaticRetriever([]))


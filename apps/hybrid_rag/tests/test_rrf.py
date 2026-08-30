import pytest
from langchain_core.documents import Document

from apps.hybrid_rag.app.fusion import (
    RRFRetriever,
    reciprocal_rank_fusion,
)
from apps.hybrid_rag.app.retrieval import (
    ParallelHybridRetriever,
    ScoredDocument,
)


def make_document(
    chunk_id: str,
) -> Document:
    return Document(
        page_content=f"Content for {chunk_id}",
        metadata={
            "chunk_id": chunk_id,
        },
    )


class FixedRetriever:
    def __init__(
        self,
        results: list[ScoredDocument],
    ) -> None:
        self.results = results

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[ScoredDocument]:
        return self.results[:top_k]


def test_rrf_combines_rankings() -> None:
    a = make_document("A")
    b = make_document("B")
    c = make_document("C")
    d = make_document("D")

    bm25_results = [
        ScoredDocument(
            document=a,
            score=10.0,
        ),
        ScoredDocument(
            document=c,
            score=8.0,
        ),
        ScoredDocument(
            document=b,
            score=6.0,
        ),
    ]

    dense_results = [
        ScoredDocument(
            document=b,
            score=0.95,
        ),
        ScoredDocument(
            document=a,
            score=0.91,
        ),
        ScoredDocument(
            document=d,
            score=0.80,
        ),
    ]

    fused = reciprocal_rank_fusion(
        bm25_results=bm25_results,
        dense_results=dense_results,
        rank_constant=60,
    )

    assert [
        result.chunk_id
        for result in fused
    ] == [
        "A",
        "B",
        "C",
        "D",
    ]

    assert fused[0].bm25_rank == 1
    assert fused[0].dense_rank == 2

    assert fused[0].score == pytest.approx(
        (1 / 61)
        + (1 / 62)
    )


def test_rrf_retriever_uses_common_retriever_interface() -> None:
    a = make_document("A")
    b = make_document("B")
    c = make_document("C")

    bm25 = FixedRetriever(
        [
            ScoredDocument(
                document=a,
                score=5.8,
            ),
            ScoredDocument(
                document=b,
                score=3.2,
            ),
        ]
    )

    dense = FixedRetriever(
        [
            ScoredDocument(
                document=b,
                score=0.95,
            ),
            ScoredDocument(
                document=a,
                score=0.90,
            ),
            ScoredDocument(
                document=c,
                score=0.80,
            ),
        ]
    )

    parallel = ParallelHybridRetriever(
        bm25_retriever=bm25,
        dense_retriever=dense,
    )

    retriever = RRFRetriever(
        parallel,
        rank_constant=60,
        candidate_k=3,
    )

    results = retriever.search(
        "test query",
        top_k=2,
    )

    assert len(results) == 2
    assert isinstance(
        results[0],
        ScoredDocument,
    )

    assert (
        results[0]
        .document
        .metadata["chunk_id"]
        == "A"
    )
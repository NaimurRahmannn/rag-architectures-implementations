from langchain_core.documents import Document

from apps.hybrid_rag.app.bm25 import BM25Index
from apps.hybrid_rag.app.retrieval import (
    BM25Retriever,
    ParallelHybridRetriever,
    ScoredDocument,
    collect_candidates,
)


class FixedRetriever:
    def __init__(
        self,
        results: list[ScoredDocument],
    ) -> None:
        self.results = results
        self.queries: list[str] = []

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[ScoredDocument]:
        self.queries.append(query)
        return self.results[:top_k]


def test_bm25_retriever_adapts_bm25_results_to_common_interface() -> None:
    documents = [
        Document(
            page_content="AUTH-401 missing credentials",
            metadata={"chunk_id": "chunk-001"},
        ),
        Document(
            page_content="AUTH-403 missing permission",
            metadata={"chunk_id": "chunk-002"},
        ),
    ]

    retriever = BM25Retriever(
        BM25Index(documents)
    )

    results = retriever.search(
        "AUTH-401",
        top_k=2,
    )

    assert isinstance(results[0], ScoredDocument)
    assert results[0].document.metadata["chunk_id"] == "chunk-001"
    assert results[0].score > 0


def test_collect_candidates_deduplicates_bm25_and_dense_results() -> None:
    shared_document = Document(
        page_content="Authorization header",
        metadata={"chunk_id": "chunk-006"},
    )
    dense_only_document = Document(
        page_content="Token expiration",
        metadata={"chunk_id": "chunk-004"},
    )

    candidates = collect_candidates(
        bm25_results=[
            ScoredDocument(
                document=shared_document,
                score=1.2,
            ),
        ],
        dense_results=[
            ScoredDocument(
                document=shared_document,
                score=0.9,
            ),
            ScoredDocument(
                document=dense_only_document,
                score=0.7,
            ),
        ],
    )

    assert len(candidates) == 2
    assert candidates[0].chunk_id == "chunk-006"
    assert candidates[0].sources == ("bm25", "dense")
    assert candidates[0].bm25_score == 1.2
    assert candidates[0].dense_score == 0.9
    assert candidates[1].chunk_id == "chunk-004"
    assert candidates[1].sources == ("dense",)


def test_parallel_hybrid_retriever_runs_both_retrievers() -> None:
    bm25_document = Document(
        page_content="AUTH-401",
        metadata={"chunk_id": "chunk-001"},
    )
    dense_document = Document(
        page_content="Authorization header",
        metadata={"chunk_id": "chunk-006"},
    )
    bm25_retriever = FixedRetriever(
        [
            ScoredDocument(
                document=bm25_document,
                score=2.0,
            )
        ]
    )
    dense_retriever = FixedRetriever(
        [
            ScoredDocument(
                document=dense_document,
                score=0.8,
            )
        ]
    )

    retriever = ParallelHybridRetriever(
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
    )

    result = retriever.retrieve(
        "How should auth work?",
        top_k=5,
    )

    assert bm25_retriever.queries == ["How should auth work?"]
    assert dense_retriever.queries == ["How should auth work?"]
    assert result.bm25_results[0].document.metadata["chunk_id"] == "chunk-001"
    assert result.dense_results[0].document.metadata["chunk_id"] == "chunk-006"
    assert {
        candidate.chunk_id
        for candidate in result.candidates
    } == {"chunk-001", "chunk-006"}

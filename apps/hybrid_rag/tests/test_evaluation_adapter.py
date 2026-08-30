from langchain_core.documents import Document

from apps.hybrid_rag.app.evaluation import EvaluationExample
from apps.hybrid_rag.app.retrieval import ScoredDocument
from apps.hybrid_rag.scripts.evaluate_retrieval import (
    build_retrieval_baselines,
    build_rrf_retriever,
    evaluate_baselines,
    evaluate_retriever,
    format_rank,
    retrieve_chunk_ids,
)


class FixedRetriever:
    def __init__(
        self,
        results: list[ScoredDocument],
    ) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[ScoredDocument]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


class KeywordEmbeddingModel:
    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [
            self._embed(text)
            for text in texts
        ]

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        return self._embed(text)

    def _embed(
        self,
        text: str,
    ) -> list[float]:
        normalized = text.lower()

        if "authorization" in normalized:
            return [1.0, 0.0]

        if "expired" in normalized:
            return [0.0, 1.0]

        return [0.0, 0.0]


class FixedScoreReranker:
    def __init__(
        self,
        scores: list[float],
    ) -> None:
        self.scores = scores

    def score(
        self,
        query: str,
        documents: list[Document],
    ) -> list[float]:
        return self.scores[: len(documents)]


def test_retrieve_chunk_ids_adapts_any_retriever_result_to_ranked_ids() -> None:
    retriever = FixedRetriever(
        [
            ScoredDocument(
                document=Document(
                    page_content="First",
                    metadata={"chunk_id": "chunk-001"},
                ),
                score=2.0,
            ),
            ScoredDocument(
                document=Document(
                    page_content="Second",
                    metadata={"chunk_id": "chunk-002"},
                ),
                score=1.0,
            ),
        ]
    )

    chunk_ids = retrieve_chunk_ids(
        retriever,
        "example query",
        top_k=1,
    )

    assert chunk_ids == ["chunk-001"]
    assert retriever.calls == [("example query", 1)]


def test_evaluate_retriever_keeps_evaluation_retriever_independent() -> None:
    retriever = FixedRetriever(
        [
            ScoredDocument(
                document=Document(
                    page_content="Wrong first",
                    metadata={"chunk_id": "chunk-001"},
                ),
                score=0.9,
            ),
            ScoredDocument(
                document=Document(
                    page_content="Relevant second",
                    metadata={"chunk_id": "chunk-003"},
                ),
                score=0.8,
            ),
        ]
    )
    examples = [
        EvaluationExample(
            query="Which chunk matters?",
            relevant_chunk_ids=("chunk-003",),
        )
    ]

    summary = evaluate_retriever(
        retriever,
        examples,
        top_k=2,
        k_values=(1, 2),
    )

    assert retriever.calls == [("Which chunk matters?", 2)]
    assert summary.recall_at_k[1] == 0.0
    assert summary.recall_at_k[2] == 1.0
    assert summary.mean_reciprocal_rank == 0.5
    assert summary.queries[0].retrieved_chunk_ids == (
        "chunk-001",
        "chunk-003",
    )


def test_evaluate_baselines_compares_retrievers_with_same_examples() -> None:
    bm25_retriever = FixedRetriever(
        [
            ScoredDocument(
                document=Document(
                    page_content="Relevant",
                    metadata={"chunk_id": "chunk-003"},
                ),
                score=2.0,
            ),
        ]
    )
    dense_retriever = FixedRetriever(
        [
            ScoredDocument(
                document=Document(
                    page_content="Wrong",
                    metadata={"chunk_id": "chunk-001"},
                ),
                score=0.9,
            ),
            ScoredDocument(
                document=Document(
                    page_content="Relevant",
                    metadata={"chunk_id": "chunk-003"},
                ),
                score=0.8,
            ),
        ]
    )
    examples = [
        EvaluationExample(
            query="Which chunk matters?",
            relevant_chunk_ids=("chunk-003",),
        )
    ]

    baselines = evaluate_baselines(
        (
            ("BM25", bm25_retriever),
            ("Dense", dense_retriever),
        ),
        examples,
        top_k=2,
        k_values=(1, 2),
    )

    assert [baseline.name for baseline in baselines] == ["BM25", "Dense"]
    assert baselines[0].summary.recall_at_k[1] == 1.0
    assert baselines[1].summary.recall_at_k[1] == 0.0
    assert bm25_retriever.calls == [("Which chunk matters?", 2)]
    assert dense_retriever.calls == [("Which chunk matters?", 2)]


def test_build_rrf_retriever_adapts_fused_results_for_evaluation() -> None:
    bm25_retriever = FixedRetriever(
        [
            ScoredDocument(
                document=Document(
                    page_content="BM25 first",
                    metadata={"chunk_id": "chunk-001"},
                ),
                score=2.0,
            ),
            ScoredDocument(
                document=Document(
                    page_content="Shared",
                    metadata={"chunk_id": "chunk-003"},
                ),
                score=1.0,
            ),
        ]
    )
    dense_retriever = FixedRetriever(
        [
            ScoredDocument(
                document=Document(
                    page_content="Shared",
                    metadata={"chunk_id": "chunk-003"},
                ),
                score=0.9,
            ),
            ScoredDocument(
                document=Document(
                    page_content="Dense second",
                    metadata={"chunk_id": "chunk-002"},
                ),
                score=0.8,
            ),
        ]
    )
    rrf_retriever = build_rrf_retriever(
        bm25_retriever,
        dense_retriever,
        candidate_k=2,
    )

    chunk_ids = retrieve_chunk_ids(
        rrf_retriever,
        "shared query",
        top_k=2,
    )

    assert chunk_ids == ["chunk-003", "chunk-001"]
    assert bm25_retriever.calls == [("shared query", 2)]
    assert dense_retriever.calls == [("shared query", 2)]


def test_build_retrieval_baselines_includes_rrf() -> None:
    chunks = [
        Document(
            page_content="Authorization header",
            metadata={"chunk_id": "chunk-006"},
        ),
        Document(
            page_content="Expired token",
            metadata={"chunk_id": "chunk-004"},
        ),
    ]

    baselines = build_retrieval_baselines(
        chunks,
        embeddings=KeywordEmbeddingModel(),
    )

    assert [
        name
        for name, _retriever in baselines
    ] == ["BM25", "Dense", "RRF"]


def test_build_retrieval_baselines_can_include_reranker() -> None:
    chunks = [
        Document(
            page_content="Authorization header",
            metadata={"chunk_id": "chunk-006"},
        ),
        Document(
            page_content="Expired token",
            metadata={"chunk_id": "chunk-004"},
        ),
    ]

    baselines = build_retrieval_baselines(
        chunks,
        embeddings=KeywordEmbeddingModel(),
        reranker=FixedScoreReranker(
            [0.2, 0.9]
        ),
    )

    assert [
        name
        for name, _retriever in baselines
    ] == ["BM25", "Dense", "RRF", "Rerank"]


def test_format_rank_returns_first_relevant_rank() -> None:
    rank = format_rank(
        relevant_chunk_ids=("chunk-003",),
        retrieved_chunk_ids=("chunk-001", "chunk-003"),
    )

    assert rank == "2"


def test_format_rank_reports_missing_relevant_chunk() -> None:
    rank = format_rank(
        relevant_chunk_ids=("chunk-999",),
        retrieved_chunk_ids=("chunk-001", "chunk-003"),
    )

    assert rank == "not found"

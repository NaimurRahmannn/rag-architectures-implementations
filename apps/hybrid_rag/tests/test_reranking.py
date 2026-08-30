import pytest
from langchain_core.documents import Document

from apps.hybrid_rag.app.evaluation import EvaluationExample
from apps.hybrid_rag.app.reranking import (
    DEFAULT_CROSS_ENCODER_MODEL,
    CrossEncoderReranker,
    RerankingRetriever,
    rerank_documents,
)
from apps.hybrid_rag.app.retrieval import ScoredDocument
from apps.hybrid_rag.scripts.evaluate_retrieval import evaluate_retriever


def make_document(
    chunk_id: str,
    content: str | None = None,
) -> Document:
    return Document(
        page_content=content or f"Content for {chunk_id}",
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
        self.calls: list[tuple[str, int]] = []

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[ScoredDocument]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


class KeywordReranker:
    def score(
        self,
        query: str,
        documents: list[Document],
    ) -> list[float]:
        query_terms = set(query.lower().split())

        return [
            float(
                len(
                    query_terms
                    & set(document.page_content.lower().split())
                )
            )
            for document in documents
        ]


class FixedScoreReranker:
    def __init__(
        self,
        scores: list[float],
    ) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def score(
        self,
        query: str,
        documents: list[Document],
    ) -> list[float]:
        self.calls.append(
            (
                query,
                [
                    str(document.metadata["chunk_id"])
                    for document in documents
                ],
            )
        )
        return self.scores


class FakeCrossEncoder:
    def __init__(
        self,
        scores: list[float],
    ) -> None:
        self.scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def predict(
        self,
        sentence_pairs: list[tuple[str, str]],
    ) -> list[float]:
        self.calls.append(sentence_pairs)
        return self.scores


def test_rerank_documents_orders_candidates_by_reranker_score() -> None:
    first = make_document(
        "chunk-001",
        "authorization header",
    )
    second = make_document(
        "chunk-002",
        "expired token",
    )
    third = make_document(
        "chunk-003",
        "send token in authorization header",
    )

    reranked = rerank_documents(
        query="authorization header token",
        results=[
            ScoredDocument(
                document=first,
                score=0.9,
            ),
            ScoredDocument(
                document=second,
                score=0.8,
            ),
            ScoredDocument(
                document=third,
                score=0.7,
            ),
        ],
        reranker=KeywordReranker(),
    )

    assert [
        result.document.metadata["chunk_id"]
        for result in reranked
    ] == [
        "chunk-003",
        "chunk-001",
        "chunk-002",
    ]
    assert reranked[0].rerank_score == 3.0
    assert reranked[0].original_score == 0.7
    assert reranked[0].original_rank == 3


def test_rerank_documents_preserves_original_rank_for_score_ties() -> None:
    first = make_document("chunk-001")
    second = make_document("chunk-002")

    reranked = rerank_documents(
        query="query",
        results=[
            ScoredDocument(
                document=first,
                score=0.9,
            ),
            ScoredDocument(
                document=second,
                score=0.8,
            ),
        ],
        reranker=FixedScoreReranker(
            [1.0, 1.0]
        ),
    )

    assert [
        result.document.metadata["chunk_id"]
        for result in reranked
    ] == ["chunk-001", "chunk-002"]


def test_rerank_documents_validates_score_count() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Reranker returned a different number of scores "
            "than candidate documents."
        ),
    ):
        rerank_documents(
            query="query",
            results=[
                ScoredDocument(
                    document=make_document("chunk-001"),
                    score=0.9,
                )
            ],
            reranker=FixedScoreReranker([]),
        )


def test_reranking_retriever_uses_common_retriever_interface() -> None:
    base_retriever = FixedRetriever(
        [
            ScoredDocument(
                document=make_document("chunk-001"),
                score=0.9,
            ),
            ScoredDocument(
                document=make_document("chunk-002"),
                score=0.8,
            ),
            ScoredDocument(
                document=make_document("chunk-003"),
                score=0.7,
            ),
        ]
    )
    reranker = FixedScoreReranker(
        [0.1, 0.9, 0.2]
    )
    retriever = RerankingRetriever(
        base_retriever,
        reranker,
        candidate_k=3,
    )

    results = retriever.search(
        "query",
        top_k=2,
    )

    assert base_retriever.calls == [("query", 3)]
    assert reranker.calls == [
        (
            "query",
            [
                "chunk-001",
                "chunk-002",
                "chunk-003",
            ],
        )
    ]
    assert [
        result.document.metadata["chunk_id"]
        for result in results
    ] == ["chunk-002", "chunk-003"]
    assert results[0].score == 0.9


def test_cross_encoder_reranker_scores_query_document_pairs() -> None:
    model = FakeCrossEncoder(
        [0.2, 0.9]
    )
    reranker = CrossEncoderReranker(
        model=model,
    )

    scores = reranker.score(
        "where should the token be sent?",
        [
            make_document(
                "chunk-001",
                "Access tokens expire.",
            ),
            make_document(
                "chunk-006",
                "Use the Authorization header.",
            ),
        ],
    )

    assert reranker.model_name == DEFAULT_CROSS_ENCODER_MODEL
    assert scores == [0.2, 0.9]
    assert model.calls == [
        [
            (
                "where should the token be sent?",
                "Access tokens expire.",
            ),
            (
                "where should the token be sent?",
                "Use the Authorization header.",
            ),
        ]
    ]


def test_cross_encoder_reranker_can_drive_reranking_retriever() -> None:
    retriever = RerankingRetriever(
        FixedRetriever(
            [
                ScoredDocument(
                    document=make_document(
                        "chunk-001",
                        "Access tokens expire.",
                    ),
                    score=0.9,
                ),
                ScoredDocument(
                    document=make_document(
                        "chunk-006",
                        "Use the Authorization header.",
                    ),
                    score=0.8,
                ),
            ]
        ),
        CrossEncoderReranker(
            model=FakeCrossEncoder(
                [0.1, 0.95]
            ),
        ),
        candidate_k=2,
    )

    results = retriever.search(
        "where should the token be sent?",
        top_k=1,
    )

    assert results[0].document.metadata["chunk_id"] == "chunk-006"
    assert results[0].score == 0.95


def test_reranking_retriever_can_be_evaluated_by_ranked_chunk_ids() -> None:
    retriever = RerankingRetriever(
        FixedRetriever(
            [
                ScoredDocument(
                    document=make_document("chunk-001"),
                    score=0.9,
                ),
                ScoredDocument(
                    document=make_document("chunk-003"),
                    score=0.8,
                ),
            ]
        ),
        FixedScoreReranker(
            [0.1, 0.9]
        ),
        candidate_k=2,
    )

    summary = evaluate_retriever(
        retriever,
        [
            EvaluationExample(
                query="query",
                relevant_chunk_ids=("chunk-003",),
            )
        ],
        top_k=2,
        k_values=(1, 2),
    )

    assert summary.recall_at_k[1] == 1.0
    assert summary.recall_at_k[2] == 1.0
    assert summary.mean_reciprocal_rank == 1.0


def test_reranking_retriever_returns_details() -> None:
    retriever = RerankingRetriever(
        FixedRetriever(
            [
                ScoredDocument(
                    document=make_document("chunk-001"),
                    score=2.0,
                ),
                ScoredDocument(
                    document=make_document("chunk-002"),
                    score=1.0,
                ),
            ]
        ),
        FixedScoreReranker(
            [0.3, 0.8]
        ),
        candidate_k=2,
    )

    results = retriever.search_with_details(
        "query",
        top_k=1,
    )

    assert results[0].document.metadata["chunk_id"] == "chunk-002"
    assert results[0].rerank_score == 0.8
    assert results[0].original_score == 1.0
    assert results[0].original_rank == 2


def test_reranking_retriever_validates_candidate_k() -> None:
    with pytest.raises(
        ValueError,
        match="candidate_k must be greater than 0",
    ):
        RerankingRetriever(
            FixedRetriever([]),
            FixedScoreReranker([]),
            candidate_k=0,
        )


def test_reranking_retriever_validates_top_k() -> None:
    retriever = RerankingRetriever(
        FixedRetriever([]),
        FixedScoreReranker([]),
    )

    with pytest.raises(
        ValueError,
        match="top_k must be greater than 0",
    ):
        retriever.search(
            "query",
            top_k=0,
        )

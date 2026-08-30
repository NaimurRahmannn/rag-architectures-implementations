from langchain_core.documents import Document

from apps.hybrid_rag.app.evaluation import EvaluationExample
from apps.hybrid_rag.app.retrieval import ScoredDocument
from apps.hybrid_rag.scripts.evaluate_retrieval import (
    evaluate_retriever,
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

from apps.hybrid_rag.app.evaluation import (
    EvaluationExample,
    evaluate_dataset,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_scores_fraction_of_relevant_chunks_found() -> None:
    score = recall_at_k(
        relevant_chunk_ids=["chunk-001", "chunk-003"],
        retrieved_chunk_ids=["chunk-001", "chunk-002", "chunk-003"],
        k=2,
    )

    assert score == 0.5


def test_reciprocal_rank_scores_first_relevant_result_position() -> None:
    score = reciprocal_rank(
        relevant_chunk_ids=["chunk-003"],
        retrieved_chunk_ids=["chunk-001", "chunk-002", "chunk-003"],
    )

    assert score == 1 / 3


def test_reciprocal_rank_returns_zero_when_no_relevant_result_found() -> None:
    score = reciprocal_rank(
        relevant_chunk_ids=["chunk-999"],
        retrieved_chunk_ids=["chunk-001", "chunk-002", "chunk-003"],
    )

    assert score == 0.0


def test_evaluate_dataset_uses_any_ranked_chunk_id_retriever() -> None:
    examples = [
        EvaluationExample(
            query="first query",
            relevant_chunk_ids=("chunk-001",),
        ),
        EvaluationExample(
            query="second query",
            relevant_chunk_ids=("chunk-003",),
        ),
    ]

    ranked_results = {
        "first query": ["chunk-001", "chunk-002", "chunk-003"],
        "second query": ["chunk-001", "chunk-002", "chunk-003"],
    }

    summary = evaluate_dataset(
        examples,
        retrieve=lambda query: ranked_results[query],
        k_values=(1, 3),
    )

    assert summary.recall_at_k[1] == 0.5
    assert summary.recall_at_k[3] == 1.0
    assert summary.mean_reciprocal_rank == (1 + 1 / 3) / 2
    assert summary.queries[0].retrieved_chunk_ids == (
        "chunk-001",
        "chunk-002",
        "chunk-003",
    )

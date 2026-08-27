import pytest

from apps.baseline_rag.evaluation.metrics import (
    average_metrics,
    evaluate_ranking,
)


def test_retrieval_metrics() -> None:
    metrics = evaluate_ranking(
        retrieved_ids=[
            "irrelevant",
            "relevant",
            "another",
        ],
        relevant_ids={"relevant"},
        k=3,
    )

    assert metrics.recall_at_k == 1.0
    assert metrics.precision_at_k == pytest.approx(
        1 / 3
    )
    assert metrics.reciprocal_rank == 0.5
    assert metrics.ndcg_at_k == pytest.approx(
        0.630929,
        abs=0.000001,
    )


def test_average_metrics() -> None:
    first = evaluate_ranking(
        retrieved_ids=["relevant"],
        relevant_ids={"relevant"},
        k=1,
    )

    second = evaluate_ranking(
        retrieved_ids=["irrelevant"],
        relevant_ids={"relevant"},
        k=1,
    )

    average = average_metrics(
        [first, second]
    )

    assert average.recall_at_k == 0.5
    assert average.precision_at_k == 0.5
    assert average.reciprocal_rank == 0.5
    assert average.ndcg_at_k == 0.5
from dataclasses import dataclass
from math import log2


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float


def evaluate_ranking(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> RetrievalMetrics:
    if k <= 0:
        raise ValueError(
            "k must be greater than zero"
        )

    if not relevant_ids:
        raise ValueError(
            "relevant_ids cannot be empty"
        )

    top_k = retrieved_ids[:k]
    relevant_retrieved = (
        set(top_k) & relevant_ids
    )

    recall = (
        len(relevant_retrieved)
        / len(relevant_ids)
    )

    precision = (
        len(relevant_retrieved)
        / k
    )

    reciprocal_rank = 0.0

    for rank, document_id in enumerate(
        top_k,
        start=1,
    ):
        if document_id in relevant_ids:
            reciprocal_rank = 1.0 / rank
            break

    dcg = sum(
        1.0 / log2(rank + 1)
        for rank, document_id in enumerate(
            top_k,
            start=1,
        )
        if document_id in relevant_ids
    )

    ideal_relevant_count = min(
        len(relevant_ids),
        k,
    )

    ideal_dcg = sum(
        1.0 / log2(rank + 1)
        for rank in range(
            1,
            ideal_relevant_count + 1,
        )
    )

    ndcg = (
        dcg / ideal_dcg
        if ideal_dcg > 0
        else 0.0
    )

    return RetrievalMetrics(
        recall_at_k=recall,
        precision_at_k=precision,
        reciprocal_rank=reciprocal_rank,
        ndcg_at_k=ndcg,
    )


def average_metrics(
    results: list[RetrievalMetrics],
) -> RetrievalMetrics:
    if not results:
        raise ValueError(
            "results cannot be empty"
        )

    count = len(results)

    return RetrievalMetrics(
        recall_at_k=sum(
            result.recall_at_k
            for result in results
        )
        / count,
        precision_at_k=sum(
            result.precision_at_k
            for result in results
        )
        / count,
        reciprocal_rank=sum(
            result.reciprocal_rank
            for result in results
        )
        / count,
        ndcg_at_k=sum(
            result.ndcg_at_k
            for result in results
        )
        / count,
    )
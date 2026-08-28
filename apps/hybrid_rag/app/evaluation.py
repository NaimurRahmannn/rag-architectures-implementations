from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationExample:
    query: str
    relevant_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class QueryEvaluation:
    query: str
    relevant_chunk_ids: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]
    recall_at_k: dict[int, float]
    reciprocal_rank: float


@dataclass(frozen=True)
class EvaluationSummary:
    recall_at_k: dict[int, float]
    mean_reciprocal_rank: float
    queries: tuple[QueryEvaluation, ...]


Retriever = Callable[[str], Sequence[str]]


def recall_at_k(
    relevant_chunk_ids: Sequence[str],
    retrieved_chunk_ids: Sequence[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than 0")

    relevant = set(relevant_chunk_ids)

    if not relevant:
        return 0.0

    retrieved = set(retrieved_chunk_ids[:k])

    return len(relevant & retrieved) / len(relevant)


def reciprocal_rank(
    relevant_chunk_ids: Sequence[str],
    retrieved_chunk_ids: Sequence[str],
) -> float:
    relevant = set(relevant_chunk_ids)

    if not relevant:
        return 0.0

    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant:
            return 1 / rank

    return 0.0


def evaluate_dataset(
    examples: Sequence[EvaluationExample],
    retrieve: Retriever,
    *,
    k_values: Sequence[int] = (1, 3, 5),
) -> EvaluationSummary:
    if not examples:
        raise ValueError("examples must not be empty")

    if not k_values:
        raise ValueError("k_values must not be empty")

    query_evaluations: list[QueryEvaluation] = []

    for example in examples:
        retrieved_chunk_ids = tuple(retrieve(example.query))
        recall_scores = {
            k: recall_at_k(
                example.relevant_chunk_ids,
                retrieved_chunk_ids,
                k,
            )
            for k in k_values
        }

        query_evaluations.append(
            QueryEvaluation(
                query=example.query,
                relevant_chunk_ids=example.relevant_chunk_ids,
                retrieved_chunk_ids=retrieved_chunk_ids,
                recall_at_k=recall_scores,
                reciprocal_rank=reciprocal_rank(
                    example.relevant_chunk_ids,
                    retrieved_chunk_ids,
                ),
            )
        )

    query_count = len(query_evaluations)

    return EvaluationSummary(
        recall_at_k={
            k: sum(
                evaluation.recall_at_k[k]
                for evaluation in query_evaluations
            )
            / query_count
            for k in k_values
        },
        mean_reciprocal_rank=sum(
            evaluation.reciprocal_rank
            for evaluation in query_evaluations
        )
        / query_count,
        queries=tuple(query_evaluations),
    )

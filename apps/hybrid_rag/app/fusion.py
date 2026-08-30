from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.documents import Document

from apps.hybrid_rag.app.retrieval import (
    ParallelHybridRetriever,
    ScoredDocument,
    get_chunk_id,
)


@dataclass(frozen=True)
class RRFDocument:
    document: Document
    chunk_id: str
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None


def rrf_contribution(
    rank: int,
    *,
    rank_constant: int = 60,
) -> float:
    if rank <= 0:
        raise ValueError("rank must be greater than 0")

    if rank_constant < 0:
        raise ValueError(
            "rank_constant must be greater than or equal to 0"
        )

    return 1.0 / (rank_constant + rank)


def reciprocal_rank_fusion(
    *,
    bm25_results: Sequence[ScoredDocument],
    dense_results: Sequence[ScoredDocument],
    rank_constant: int = 60,
) -> tuple[RRFDocument, ...]:
    if rank_constant < 0:
        raise ValueError(
            "rank_constant must be greater than or equal to 0"
        )

    documents: dict[str, Document] = {}
    scores: dict[str, float] = {}

    bm25_ranks: dict[str, int] = {}
    dense_ranks: dict[str, int] = {}

    for source, results, rank_store in (
        ("bm25", bm25_results, bm25_ranks),
        ("dense", dense_results, dense_ranks),
    ):
        seen_chunk_ids: set[str] = set()

        for rank, result in enumerate(
            results,
            start=1,
        ):
            chunk_id = get_chunk_id(
                result.document
            )

            if chunk_id in seen_chunk_ids:
                raise ValueError(
                    f"{source} ranking contains duplicate "
                    f"chunk_id: {chunk_id}"
                )

            seen_chunk_ids.add(chunk_id)

            documents.setdefault(
                chunk_id,
                result.document,
            )

            rank_store[chunk_id] = rank

            contribution = rrf_contribution(
                rank,
                rank_constant=rank_constant,
            )

            scores[chunk_id] = (
                scores.get(chunk_id, 0.0)
                + contribution
            )

    fused = [
        RRFDocument(
            document=document,
            chunk_id=chunk_id,
            score=scores[chunk_id],
            bm25_rank=bm25_ranks.get(chunk_id),
            dense_rank=dense_ranks.get(chunk_id),
        )
        for chunk_id, document
        in documents.items()
    ]

    fused.sort(
        key=lambda item: (
            -item.score,
            _best_rank(item),
            item.chunk_id,
        )
    )

    return tuple(fused)


class RRFRetriever:
    def __init__(
        self,
        parallel_retriever: ParallelHybridRetriever,
        *,
        rank_constant: int = 60,
        candidate_k: int = 20,
    ) -> None:
        if rank_constant < 0:
            raise ValueError(
                "rank_constant must be greater than "
                "or equal to 0"
            )

        if candidate_k <= 0:
            raise ValueError(
                "candidate_k must be greater than 0"
            )

        self.parallel_retriever = parallel_retriever
        self.rank_constant = rank_constant
        self.candidate_k = candidate_k

    def search_with_details(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RRFDocument]:
        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        retrieval_k = max(
            top_k,
            self.candidate_k,
        )

        parallel_result = (
            self.parallel_retriever.retrieve(
                query,
                top_k=retrieval_k,
            )
        )

        fused = reciprocal_rank_fusion(
            bm25_results=(
                parallel_result.bm25_results
            ),
            dense_results=(
                parallel_result.dense_results
            ),
            rank_constant=self.rank_constant,
        )

        return list(
            fused[:top_k]
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[ScoredDocument]:
        results = self.search_with_details(
            query,
            top_k=top_k,
        )

        return [
            ScoredDocument(
                document=result.document,
                score=result.score,
            )
            for result in results
        ]


def _best_rank(
    result: RRFDocument,
) -> int:
    ranks = [
        rank
        for rank in (
            result.bm25_rank,
            result.dense_rank,
        )
        if rank is not None
    ]

    return min(ranks)
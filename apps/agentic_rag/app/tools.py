from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.documents import Document

from apps.hybrid_rag.app.retrieval import (
    Retriever,
    get_chunk_id,
)


@dataclass(frozen=True)
class RetrievedEvidence:
    document: Document
    score: float
    chunk_id: str
    tool_name: str
    query: str
    tool_rank: int


class RetrievalTool:
    def __init__(
        self,
        name: str,
        retriever: Retriever,
    ) -> None:
        value = name.strip()

        if not value:
            raise ValueError("tool name must not be empty")

        self.name = value
        self.retriever = retriever

    def search(
        self,
        query: str,
        *,
        top_k: int,
    ) -> tuple[RetrievedEvidence, ...]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        results = self.retriever.search(
            query,
            top_k=top_k,
        )

        return tuple(
            RetrievedEvidence(
                document=result.document,
                score=result.score,
                chunk_id=get_chunk_id(result.document),
                tool_name=self.name,
                query=query,
                tool_rank=rank,
            )
            for rank, result in enumerate(
                results,
                start=1,
            )
        )


def merge_evidence(
    evidence_groups: Sequence[Sequence[RetrievedEvidence]],
    *,
    limit: int,
) -> tuple[RetrievedEvidence, ...]:
    if limit <= 0:
        raise ValueError("limit must be greater than 0")

    merged: list[RetrievedEvidence] = []
    seen_chunk_ids: set[str] = set()

    for evidence_group in evidence_groups:
        for evidence in evidence_group:
            if evidence.chunk_id in seen_chunk_ids:
                continue

            seen_chunk_ids.add(evidence.chunk_id)
            merged.append(evidence)

            if len(merged) == limit:
                return tuple(merged)

    return tuple(merged)

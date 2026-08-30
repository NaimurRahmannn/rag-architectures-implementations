from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from langchain_core.documents import Document

from apps.hybrid_rag.app.retrieval import (
    Retriever,
    ScoredDocument,
)

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


class Reranker(Protocol):
    def score(
        self,
        query: str,
        documents: Sequence[Document],
    ) -> Sequence[float]:
        """Return one relevance score per candidate document."""


class CrossEncoderModel(Protocol):
    def predict(
        self,
        sentence_pairs: Sequence[tuple[str, str]],
    ) -> Sequence[float]:
        """Return one relevance score per query-document pair."""


@dataclass(frozen=True)
class RerankedDocument:
    document: Document
    rerank_score: float
    original_score: float
    original_rank: int


def rerank_documents(
    *,
    query: str,
    results: Sequence[ScoredDocument],
    reranker: Reranker,
) -> tuple[RerankedDocument, ...]:
    if not results:
        return ()

    documents = [
        result.document
        for result in results
    ]
    rerank_scores = tuple(
        float(score)
        for score in reranker.score(
            query,
            documents,
        )
    )

    if len(rerank_scores) != len(results):
        raise ValueError(
            "Reranker returned a different number of scores "
            "than candidate documents."
        )

    reranked = [
        RerankedDocument(
            document=result.document,
            rerank_score=rerank_scores[index],
            original_score=result.score,
            original_rank=index + 1,
        )
        for index, result in enumerate(results)
    ]

    reranked.sort(
        key=lambda result: (
            -result.rerank_score,
            result.original_rank,
        )
    )

    return tuple(reranked)


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        *,
        model: CrossEncoderModel | None = None,
    ) -> None:
        self.model_name = model_name
        self.model = model or self._load_model(model_name)

    def score(
        self,
        query: str,
        documents: Sequence[Document],
    ) -> list[float]:
        sentence_pairs = [
            (
                query,
                document.page_content,
            )
            for document in documents
        ]

        if not sentence_pairs:
            return []

        return [
            float(score)
            for score in self.model.predict(sentence_pairs)
        ]

    @staticmethod
    def _load_model(
        model_name: str,
    ) -> CrossEncoderModel:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(model_name)


class RerankingRetriever:
    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        *,
        candidate_k: int = 20,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError(
                "candidate_k must be greater than 0"
            )

        self.retriever = retriever
        self.reranker = reranker
        self.candidate_k = candidate_k

    def search_with_details(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RerankedDocument]:
        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        candidates = self.retriever.search(
            query,
            top_k=max(
                top_k,
                self.candidate_k,
            ),
        )
        reranked = rerank_documents(
            query=query,
            results=candidates,
            reranker=self.reranker,
        )

        return list(
            reranked[:top_k]
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
                score=result.rerank_score,
            )
            for result in results
        ]

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
from typing import Protocol

from langchain_core.documents import Document


@dataclass(frozen=True)
class ScoredDocument:
    document: Document
    score: float


class EmbeddingModel(Protocol):
    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> Sequence[Sequence[float]]:
        """Embed multiple document texts."""

    def embed_query(
        self,
        text: str,
    ) -> Sequence[float]:
        """Embed one query text."""


@dataclass(frozen=True)
class DenseDocumentVector:
    document: Document
    vector: tuple[float, ...]


class InMemoryDenseIndex:
    def __init__(
        self,
        documents: Sequence[Document],
        embeddings: EmbeddingModel,
    ) -> None:
        if not documents:
            raise ValueError("Dense index requires at least one document.")

        self.documents = list(documents)
        self.embeddings = embeddings

        document_vectors = embeddings.embed_documents(
            [
                document.page_content
                for document in self.documents
            ]
        )

        if len(document_vectors) != len(self.documents):
            raise ValueError(
                "Embedding model returned a different number of vectors "
                "than documents."
            )

        self._vectors = [
            DenseDocumentVector(
                document=document,
                vector=_validate_vector(vector),
            )
            for document, vector in zip(
                self.documents,
                document_vectors,
                strict=True,
            )
        ]

        dimensions = {
            len(item.vector)
            for item in self._vectors
        }

        if len(dimensions) != 1:
            raise ValueError("All document vectors must have the same dimension.")

        self.dimension = dimensions.pop()

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[ScoredDocument]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_vector = _validate_vector(
            self.embeddings.embed_query(query)
        )

        if len(query_vector) != self.dimension:
            raise ValueError("Query vector dimension does not match the index.")

        results = [
            ScoredDocument(
                document=item.document,
                score=cosine_similarity(
                    query_vector,
                    item.vector,
                ),
            )
            for item in self._vectors
        ]

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return results[:top_k]


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    left_vector = _validate_vector(left)
    right_vector = _validate_vector(right)

    if len(left_vector) != len(right_vector):
        raise ValueError("Vectors must have the same dimension.")

    left_norm = _norm(left_vector)
    right_norm = _norm(right_vector)

    if left_norm == 0 or right_norm == 0:
        return 0.0

    dot_product = sum(
        left_value * right_value
        for left_value, right_value in zip(
            left_vector,
            right_vector,
            strict=True,
        )
    )

    return dot_product / (left_norm * right_norm)


def _validate_vector(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)

    if not values:
        raise ValueError("Vector must not be empty.")

    return values


def _norm(vector: Sequence[float]) -> float:
    return sqrt(
        sum(
            value * value
            for value in vector
        )
    )

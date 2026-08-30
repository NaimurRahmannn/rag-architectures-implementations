from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import sqrt
from typing import Protocol

from langchain_core.documents import Document

from apps.hybrid_rag.app.bm25 import BM25Index


@dataclass(frozen=True)
class ScoredDocument:
    document: Document
    score: float


class Retriever(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[ScoredDocument]:
        """Return documents ordered by relevance."""


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


@dataclass(frozen=True)
class HybridCandidate:
    document: Document
    chunk_id: str
    sources: tuple[str, ...]
    bm25_score: float | None = None
    dense_score: float | None = None


@dataclass(frozen=True)
class ParallelHybridResult:
    bm25_results: tuple[ScoredDocument, ...]
    dense_results: tuple[ScoredDocument, ...]
    candidates: tuple[HybridCandidate, ...]


class BM25Retriever:
    def __init__(
        self,
        index: BM25Index,
    ) -> None:
        self.index = index

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[ScoredDocument]:
        return [
            ScoredDocument(
                document=result.document,
                score=result.score,
            )
            for result in self.index.search(
                query,
                top_k=top_k,
            )
        ]


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


class DenseRetriever:
    def __init__(
        self,
        index: InMemoryDenseIndex,
    ) -> None:
        self.index = index

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[ScoredDocument]:
        return self.index.search(
            query,
            top_k=top_k,
        )


class ParallelHybridRetriever:
    def __init__(
        self,
        bm25_retriever: Retriever,
        dense_retriever: Retriever,
    ) -> None:
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> ParallelHybridResult:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        with ThreadPoolExecutor(max_workers=2) as executor:
            bm25_future = executor.submit(
                self.bm25_retriever.search,
                query,
                top_k,
            )
            dense_future = executor.submit(
                self.dense_retriever.search,
                query,
                top_k,
            )

            bm25_results = tuple(bm25_future.result())
            dense_results = tuple(dense_future.result())

        return ParallelHybridResult(
            bm25_results=bm25_results,
            dense_results=dense_results,
            candidates=collect_candidates(
                bm25_results=bm25_results,
                dense_results=dense_results,
            ),
        )


def collect_candidates(
    *,
    bm25_results: Sequence[ScoredDocument],
    dense_results: Sequence[ScoredDocument],
) -> tuple[HybridCandidate, ...]:
    candidates: dict[str, HybridCandidate] = {}

    for source, results in (
        ("bm25", bm25_results),
        ("dense", dense_results),
    ):
        for result in results:
            chunk_id = get_chunk_id(result.document)
            current = candidates.get(chunk_id)

            if current is None:
                candidates[chunk_id] = HybridCandidate(
                    document=result.document,
                    chunk_id=chunk_id,
                    sources=(source,),
                    bm25_score=result.score if source == "bm25" else None,
                    dense_score=result.score if source == "dense" else None,
                )
                continue

            sources = current.sources

            if source not in sources:
                sources = (*sources, source)

            candidates[chunk_id] = HybridCandidate(
                document=current.document,
                chunk_id=current.chunk_id,
                sources=sources,
                bm25_score=(
                    result.score
                    if source == "bm25"
                    else current.bm25_score
                ),
                dense_score=(
                    result.score
                    if source == "dense"
                    else current.dense_score
                ),
            )

    return tuple(candidates.values())


def get_chunk_id(document: Document) -> str:
    chunk_id = document.metadata.get("chunk_id")

    if chunk_id is None:
        raise ValueError("Retrieved document is missing chunk_id metadata.")

    value = str(chunk_id).strip()

    if not value:
        raise ValueError("Retrieved document has an empty chunk_id.")

    return value


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

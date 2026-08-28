from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log
from typing import Sequence

from langchain_core.documents import Document

from .tokenizer import tokenize


@dataclass(frozen=True)
class BM25Result:
    document: Document
    score: float


@dataclass(frozen=True)
class DocumentStatistics:
    document: Document
    tokens: tuple[str, ...]
    term_frequencies: dict[str, int]
    length: int


class BM25Index:
    """
    Stores the corpus statistics required by BM25.
    """

    def __init__(
        self,
        documents: Sequence[Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 < 0:
            raise ValueError("k1 must be >= 0.")

        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1.")

        if not documents:
            raise ValueError("BM25 index requires at least one document.")

        self.k1 = k1
        self.b = b

        self.documents = list(documents)

        self._statistics = [
            self._build_document_statistics(document)
            for document in self.documents
        ]

        self.document_count = len(self.documents)

        self.document_frequencies = self._build_document_frequencies()

        self.average_document_length = (
            sum(stat.length for stat in self._statistics)
            / self.document_count
        )

    @staticmethod
    def _build_document_statistics(
        document: Document,
    ) -> DocumentStatistics:
        tokens = tokenize(document.page_content)

        term_frequencies = Counter(tokens)

        return DocumentStatistics(
            document=document,
            tokens=tuple(tokens),
            term_frequencies=dict(term_frequencies),
            length=len(tokens),
        )

    def _build_document_frequencies(self) -> dict[str, int]:
        frequencies: Counter[str] = Counter()

        for statistics in self._statistics:
            frequencies.update(statistics.term_frequencies.keys())

        return dict(frequencies)

    def idf(self, term: str) -> float:
        """
        Calculate BM25's IDF value for a term.
        """
        normalized_term = term.lower()

        document_frequency = self.document_frequencies.get(
            normalized_term,
            0,
        )

        if document_frequency == 0:
            return 0.0

        n = self.document_count

        return log(
            1
            + (
                (n - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
        )

    def statistics(self) -> list[DocumentStatistics]:
        """
        Return per-document statistics for inspection/testing.
        """
        return list(self._statistics)

    def search(self, query: str, top_k: int = 5) -> list[BM25Result]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        results: list[BM25Result] = []

        for statistics in self._statistics:
            score = self._score_document(
                query_tokens,
                statistics,
            )

            results.append(
                BM25Result(
                    document=statistics.document,
                    score=score,
                )
            )

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return results[:top_k]

    def _score_document(
        self,
        query_tokens: list[str],
        statistics: DocumentStatistics,
    ) -> float:
        score = 0.0

        for term in query_tokens:
            term_frequency = statistics.term_frequencies.get(
                term,
                0,
            )

            if term_frequency == 0:
                continue

            term_idf = self.idf(term)

            denominator = (
                term_frequency
                + self.k1
                * (
                    1
                    - self.b
                    + self.b
                    * (
                        statistics.length
                        / self.average_document_length
                    )
                )
            )

            term_score = (
                term_idf
                * (
                    term_frequency
                    * (self.k1 + 1)
                )
                / denominator
            )

            score += term_score

        return score

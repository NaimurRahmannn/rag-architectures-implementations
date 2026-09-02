from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from apps.agentic_rag.app.planning import IDENTIFIER_PATTERN
from apps.agentic_rag.app.tools import RetrievedEvidence

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?")

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "should",
    "the",
    "to",
    "what",
    "when",
    "where",
    "why",
    "with",
}


@dataclass(frozen=True)
class EvidenceGrade:
    is_sufficient: bool
    score: float
    matched_terms: tuple[str, ...]
    missing_terms: tuple[str, ...]
    reason: str


class EvidenceGrader(Protocol):
    def grade(
        self,
        original_query: str,
        attempted_query: str,
        evidence: Sequence[RetrievedEvidence],
    ) -> EvidenceGrade:
        """Judge whether retrieved evidence is strong enough to answer."""


class CorrectiveQueryRewriter(Protocol):
    def rewrite(
        self,
        original_query: str,
        attempted_query: str,
        grade: EvidenceGrade,
    ) -> str:
        """Create a fallback retrieval query after weak evidence."""


class HeuristicEvidenceGrader:
    def __init__(
        self,
        *,
        min_overlap_ratio: float = 0.35,
        require_identifier_match: bool = True,
    ) -> None:
        if not 0 <= min_overlap_ratio <= 1:
            raise ValueError("min_overlap_ratio must be between 0 and 1")

        self.min_overlap_ratio = min_overlap_ratio
        self.require_identifier_match = require_identifier_match

    def grade(
        self,
        original_query: str,
        attempted_query: str,
        evidence: Sequence[RetrievedEvidence],
    ) -> EvidenceGrade:
        if not evidence:
            return EvidenceGrade(
                is_sufficient=False,
                score=0.0,
                matched_terms=(),
                missing_terms=tuple(query_terms(original_query)),
                reason="No evidence was retrieved.",
            )

        terms = tuple(query_terms(original_query))
        searchable_text = "\n".join(
            evidence_text(item)
            for item in evidence
        ).lower()
        searchable_terms = set(query_terms(searchable_text))
        matched_terms = tuple(
            term
            for term in terms
            if term.lower() in searchable_terms
        )
        missing_terms = tuple(
            term
            for term in terms
            if term not in matched_terms
        )
        score = (
            len(matched_terms) / len(terms)
            if terms
            else 1.0
        )

        missing_identifiers = tuple(
            identifier
            for identifier in extract_identifiers(original_query)
            if identifier.lower() not in searchable_text
        )
        identifier_ok = (
            not self.require_identifier_match
            or not missing_identifiers
        )
        is_sufficient = (
            score >= self.min_overlap_ratio
            and identifier_ok
        )

        reason = (
            "Evidence covers enough query terms."
            if is_sufficient
            else insufficient_reason(
                score=score,
                min_overlap_ratio=self.min_overlap_ratio,
                missing_identifiers=missing_identifiers,
            )
        )

        return EvidenceGrade(
            is_sufficient=is_sufficient,
            score=score,
            matched_terms=matched_terms,
            missing_terms=missing_terms,
            reason=reason,
        )


class KeywordFallbackQueryRewriter:
    def rewrite(
        self,
        original_query: str,
        attempted_query: str,
        grade: EvidenceGrade,
    ) -> str:
        del grade

        without_identifiers = IDENTIFIER_PATTERN.sub(
            " ",
            original_query,
        )
        keyword_query = " ".join(
            query_terms(without_identifiers)
        )

        if keyword_query and keyword_query.lower() != attempted_query.lower():
            return keyword_query

        original_keywords = " ".join(
            query_terms(original_query)
        )

        if (
            original_keywords
            and original_keywords.lower() != attempted_query.lower()
        ):
            return original_keywords

        return attempted_query


def query_terms(
    query: str,
) -> tuple[str, ...]:
    terms = []
    seen_terms: set[str] = set()

    for token in TOKEN_PATTERN.findall(query.lower()):
        if token in STOP_WORDS:
            continue

        if token in seen_terms:
            continue

        seen_terms.add(token)
        terms.append(token)

    return tuple(terms)


def extract_identifiers(
    query: str,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            match.group(0)
            for match in IDENTIFIER_PATTERN.finditer(query)
        )
    )


def evidence_text(
    evidence: RetrievedEvidence,
) -> str:
    metadata = " ".join(
        str(value)
        for value in evidence.document.metadata.values()
    )

    return f"{evidence.document.page_content}\n{metadata}"


def insufficient_reason(
    *,
    score: float,
    min_overlap_ratio: float,
    missing_identifiers: Sequence[str],
) -> str:
    if missing_identifiers:
        return (
            "Evidence does not mention required identifier(s): "
            + ", ".join(missing_identifiers)
            + "."
        )

    return (
        "Evidence term overlap is too low "
        f"({score:.2f} < {min_overlap_ratio:.2f})."
    )

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

IDENTIFIER_PATTERN = re.compile(r"\b[A-Z]+-\d+\b")
QUESTION_WORDS = {
    "how",
    "why",
    "should",
    "can",
}


@dataclass(frozen=True)
class RetrievalStep:
    tool_name: str
    query: str
    top_k: int


@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    steps: tuple[RetrievalStep, ...]


class QueryPlanner(Protocol):
    def plan(
        self,
        query: str,
        available_tools: Sequence[str],
        *,
        top_k: int,
    ) -> QueryPlan:
        """Create retrieval steps for a user query."""


class HeuristicQueryPlanner:
    def plan(
        self,
        query: str,
        available_tools: Sequence[str],
        *,
        top_k: int,
    ) -> QueryPlan:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        normalized_tools = tuple(dict.fromkeys(available_tools))

        if not normalized_tools:
            raise ValueError("available_tools must not be empty")

        subqueries = split_query(query)
        preferred_tools = choose_tools(
            query,
            normalized_tools,
        )

        return QueryPlan(
            original_query=query,
            steps=tuple(
                RetrievalStep(
                    tool_name=tool_name,
                    query=subquery,
                    top_k=top_k,
                )
                for subquery in subqueries
                for tool_name in preferred_tools
            ),
        )


def split_query(
    query: str,
) -> tuple[str, ...]:
    parts = [
        part.strip()
        for part in re.split(
            r"\s+and\s+",
            query,
            flags=re.IGNORECASE,
        )
        if part.strip()
    ]

    if len(parts) <= 1:
        return (query,)

    return tuple(parts)


def choose_tools(
    query: str,
    available_tools: Sequence[str],
) -> tuple[str, ...]:
    available = tuple(dict.fromkeys(available_tools))

    if not available:
        raise ValueError("available_tools must not be empty")

    if has_identifier(query):
        preferred = (
            "bm25",
            "hybrid",
            "dense",
            "rerank",
        )
    elif is_semantic_question(query):
        preferred = (
            "dense",
            "hybrid",
            "rerank",
            "bm25",
        )
    else:
        preferred = (
            "hybrid",
            "bm25",
            "dense",
            "rerank",
        )

    selected = tuple(
        tool_name
        for tool_name in preferred
        if tool_name in available
    )

    if selected:
        return selected

    return (available[0],)


def has_identifier(
    query: str,
) -> bool:
    return IDENTIFIER_PATTERN.search(query) is not None


def is_semantic_question(
    query: str,
) -> bool:
    tokens = {
        token.strip("?:!.,;").lower()
        for token in query.split()
    }

    return bool(tokens & QUESTION_WORDS)

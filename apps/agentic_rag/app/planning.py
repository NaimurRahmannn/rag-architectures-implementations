from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

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
    reason: str | None = None


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


class PlannerLLM(Protocol):
    def generate(
        self,
        prompt: str,
    ) -> str:
        """Return a structured planning response as JSON text."""


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
                    reason="Selected by deterministic query heuristics.",
                )
                for subquery in subqueries
                for tool_name in preferred_tools
            ),
        )


class LLMDynamicQueryPlanner:
    def __init__(
        self,
        llm: PlannerLLM,
        *,
        fallback_planner: QueryPlanner | None = None,
        max_steps: int = 6,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be greater than 0")

        self.llm = llm
        self.fallback_planner = fallback_planner or HeuristicQueryPlanner()
        self.max_steps = max_steps

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

        prompt = build_dynamic_planner_prompt(
            query=query,
            available_tools=normalized_tools,
            top_k=top_k,
            max_steps=self.max_steps,
        )

        try:
            response = self.llm.generate(prompt)
            steps = parse_dynamic_plan_response(
                response,
                available_tools=normalized_tools,
                default_top_k=top_k,
                max_steps=self.max_steps,
            )
        except ValueError:
            return self.fallback_planner.plan(
                query,
                normalized_tools,
                top_k=top_k,
            )

        return QueryPlan(
            original_query=query,
            steps=steps,
        )


class GeminiPlannerLLM:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
    ) -> None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        self._client = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0,
        )

    def generate(
        self,
        prompt: str,
    ) -> str:
        response = self._client.invoke(prompt)
        content = response.content

        if isinstance(content, str):
            return content

        return str(content)


def build_dynamic_planner_prompt(
    *,
    query: str,
    available_tools: Sequence[str],
    top_k: int,
    max_steps: int,
) -> str:
    tools = ", ".join(available_tools)

    return f"""
You are a query planner for an agentic RAG system.

Create a retrieval plan for the user's question using only the available tools.
Choose the smallest useful set of retrieval calls. Decompose compound questions
when separate evidence is needed.

Available tools:
- bm25: best for exact keywords, identifiers, codes, filenames, and terms.
- dense: best for semantic questions and paraphrases.
- hybrid: best default when both lexical and semantic signals matter.
- rerank: best for precision after broad retrieval when available.

Rules:
1. Use only these available tool names: {tools}.
2. Return at most {max_steps} steps.
3. Every step must include tool_name, query, top_k, and reason.
4. top_k must be an integer from 1 to {top_k}.
5. Return JSON only. Do not use markdown.

JSON shape:
{{
  "steps": [
    {{
      "tool_name": "hybrid",
      "query": "focused retrieval query",
      "top_k": {top_k},
      "reason": "why this tool and query are useful"
    }}
  ]
}}

User question:
{query}
""".strip()


def parse_dynamic_plan_response(
    response: str,
    *,
    available_tools: Sequence[str],
    default_top_k: int,
    max_steps: int,
) -> tuple[RetrievalStep, ...]:
    payload = json.loads(extract_json_object(response))

    if not isinstance(payload, dict):
        raise ValueError("planner response must be a JSON object")

    raw_steps = payload.get("steps")

    if not isinstance(raw_steps, list):
        raise ValueError("planner response must include a steps list")

    available = set(available_tools)
    steps: list[RetrievalStep] = []

    for raw_step in raw_steps[:max_steps]:
        if not isinstance(raw_step, dict):
            continue

        step = parse_dynamic_plan_step(
            raw_step,
            available_tools=available,
            default_top_k=default_top_k,
        )

        if step is not None:
            steps.append(step)

    if not steps:
        raise ValueError("planner response did not contain valid steps")

    return tuple(steps)


def parse_dynamic_plan_step(
    raw_step: dict[str, Any],
    *,
    available_tools: set[str],
    default_top_k: int,
) -> RetrievalStep | None:
    tool_name = normalized_text(raw_step.get("tool_name")).lower()
    query = normalized_text(raw_step.get("query"))

    if tool_name not in available_tools or not query:
        return None

    raw_top_k = raw_step.get("top_k", default_top_k)

    try:
        top_k = int(raw_top_k)
    except (TypeError, ValueError):
        top_k = default_top_k

    top_k = min(
        max(top_k, 1),
        default_top_k,
    )
    reason = normalized_text(raw_step.get("reason")) or None

    return RetrievalStep(
        tool_name=tool_name,
        query=query,
        top_k=top_k,
        reason=reason,
    )


def extract_json_object(
    response: str,
) -> str:
    value = response.strip()

    if value.startswith("```"):
        value = strip_markdown_fence(value)

    if value.startswith("{") and value.endswith("}"):
        return value

    start = value.find("{")
    end = value.rfind("}")

    if start == -1 or end == -1 or start >= end:
        raise ValueError("planner response did not contain a JSON object")

    return value[start : end + 1]


def strip_markdown_fence(
    response: str,
) -> str:
    lines = response.splitlines()

    if len(lines) >= 3 and lines[0].startswith("```") and lines[-1] == "```":
        return "\n".join(lines[1:-1]).strip()

    return response


def normalized_text(
    value: object,
) -> str:
    if value is None:
        return ""

    return str(value).strip()


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

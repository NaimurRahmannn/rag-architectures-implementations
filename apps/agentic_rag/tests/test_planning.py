import pytest

from apps.agentic_rag.app.planning import (
    HeuristicQueryPlanner,
    LLMDynamicQueryPlanner,
    choose_tools,
    has_identifier,
    is_semantic_question,
    parse_dynamic_plan_response,
    split_query,
)


class StaticPlannerLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def test_split_query_creates_subqueries_for_compound_question() -> None:
    assert split_query("Find AUTH-42 and explain password resets") == (
        "Find AUTH-42",
        "explain password resets",
    )


def test_split_query_preserves_a_single_query() -> None:
    assert split_query("How do password resets work?") == (
        "How do password resets work?",
    )


def test_identifier_query_prioritizes_lexical_retrieval() -> None:
    assert has_identifier("What happened to AUTH-42?") is True
    assert choose_tools(
        "What happened to AUTH-42?",
        ("dense", "bm25", "hybrid"),
    ) == ("bm25", "hybrid", "dense")


def test_semantic_question_prioritizes_dense_retrieval() -> None:
    query = "Why should users rotate recovery codes?"

    assert is_semantic_question(query) is True
    assert choose_tools(
        query,
        ("bm25", "dense", "hybrid"),
    ) == ("dense", "hybrid", "bm25")


def test_planner_creates_one_step_per_subquery_and_tool() -> None:
    plan = HeuristicQueryPlanner().plan(
        "Find AUTH-42 and explain password resets",
        ("bm25", "dense"),
        top_k=3,
    )

    assert plan.original_query == "Find AUTH-42 and explain password resets"
    assert [
        (step.tool_name, step.query, step.top_k)
        for step in plan.steps
    ] == [
        ("bm25", "Find AUTH-42", 3),
        ("dense", "Find AUTH-42", 3),
        ("bm25", "explain password resets", 3),
        ("dense", "explain password resets", 3),
    ]
    assert all(step.reason is not None for step in plan.steps)


def test_llm_dynamic_planner_uses_structured_tool_selection() -> None:
    llm = StaticPlannerLLM(
        """
        {
          "steps": [
            {
              "tool_name": "hybrid",
              "query": "recovery code rotation",
              "top_k": 2,
              "reason": "Hybrid balances terms and semantic intent."
            },
            {
              "tool_name": "bm25",
              "query": "AUTH-42",
              "top_k": 5,
              "reason": "Identifier lookup needs exact matching."
            }
          ]
        }
        """
    )

    plan = LLMDynamicQueryPlanner(llm).plan(
        "How should I rotate codes for AUTH-42?",
        ("bm25", "dense", "hybrid"),
        top_k=3,
    )

    assert plan.original_query == "How should I rotate codes for AUTH-42?"
    assert [
        (step.tool_name, step.query, step.top_k, step.reason)
        for step in plan.steps
    ] == [
        (
            "hybrid",
            "recovery code rotation",
            2,
            "Hybrid balances terms and semantic intent.",
        ),
        (
            "bm25",
            "AUTH-42",
            3,
            "Identifier lookup needs exact matching.",
        ),
    ]
    assert "bm25, dense, hybrid" in llm.prompts[0]


def test_dynamic_plan_parser_filters_invalid_steps() -> None:
    steps = parse_dynamic_plan_response(
        """
        ```json
        {
          "steps": [
            {"tool_name": "web", "query": "external", "top_k": 2},
            {"tool_name": "dense", "query": "valid query", "top_k": "bad"}
          ]
        }
        ```
        """,
        available_tools=("dense",),
        default_top_k=4,
        max_steps=3,
    )

    assert [
        (step.tool_name, step.query, step.top_k)
        for step in steps
    ] == [("dense", "valid query", 4)]


def test_llm_dynamic_planner_falls_back_to_heuristics_on_bad_output() -> None:
    plan = LLMDynamicQueryPlanner(
        StaticPlannerLLM("not-json")
    ).plan(
        "Why rotate recovery codes?",
        ("bm25", "dense", "hybrid"),
        top_k=2,
    )

    assert [
        step.tool_name
        for step in plan.steps
    ] == ["dense", "hybrid", "bm25"]


def test_planner_rejects_invalid_configuration() -> None:
    planner = HeuristicQueryPlanner()

    with pytest.raises(ValueError, match="top_k"):
        planner.plan("query", ("bm25",), top_k=0)

    with pytest.raises(ValueError, match="available_tools"):
        planner.plan("query", (), top_k=1)

    with pytest.raises(ValueError, match="max_steps"):
        LLMDynamicQueryPlanner(StaticPlannerLLM("{}"), max_steps=0)

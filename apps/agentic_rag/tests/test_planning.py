import pytest

from apps.agentic_rag.app.planning import (
    HeuristicQueryPlanner,
    choose_tools,
    has_identifier,
    is_semantic_question,
    split_query,
)


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


def test_planner_rejects_invalid_configuration() -> None:
    planner = HeuristicQueryPlanner()

    with pytest.raises(ValueError, match="top_k"):
        planner.plan("query", ("bm25",), top_k=0)

    with pytest.raises(ValueError, match="available_tools"):
        planner.plan("query", (), top_k=1)


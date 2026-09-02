from __future__ import annotations

import argparse
from collections.abc import Sequence

from langchain_core.documents import Document

from apps.agentic_rag.app.planning import (
    GeminiPlannerLLM,
    HeuristicQueryPlanner,
    LLMDynamicQueryPlanner,
    QueryPlanner,
)
from apps.agentic_rag.app.schemas import AnswerResponse, AskRequest
from apps.agentic_rag.app.service import AgenticRAGService
from apps.agentic_rag.app.tools import RetrievalTool
from apps.baseline_rag.app.settings import Settings
from apps.hybrid_rag.app.generation import GeminiAnswerGenerator
from apps.hybrid_rag.app.reranking import Reranker
from apps.hybrid_rag.app.retrieval import EmbeddingModel, Retriever
from apps.hybrid_rag.scripts.evaluate_retrieval import (
    build_bm25_retriever,
    build_cross_encoder_reranker,
    build_dense_retriever,
    build_gemini_embeddings,
    build_reranking_retriever,
    build_rrf_retriever,
)
from apps.hybrid_rag.scripts.inspect_chunks import (
    chunk_documents,
    load_documents,
)

DEFAULT_TOOLS = "bm25,dense,hybrid"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a grounded Agentic RAG answer."
    )
    parser.add_argument(
        "query",
        help="Question to answer from the configured document chunks.",
    )
    parser.add_argument(
        "--tools",
        default=DEFAULT_TOOLS,
        help=(
            "Comma-separated retrieval tools available to the planner. "
            "Choices: bm25,dense,hybrid,rerank."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of final evidence chunks to send to the answer generator.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="Number of candidates to gather before RRF/reranking.",
    )
    parser.add_argument(
        "--max-correction-attempts",
        type=int,
        default=1,
        help="Number of corrective retrieval retries after weak evidence.",
    )
    parser.add_argument(
        "--planner",
        choices=("heuristic", "llm"),
        default="heuristic",
        help="Planner strategy to use for structured tool selection.",
    )
    return parser


def parse_tool_names(
    raw_tools: str,
) -> tuple[str, ...]:
    tool_names = tuple(
        dict.fromkeys(
            tool.strip().lower()
            for tool in raw_tools.split(",")
            if tool.strip()
        )
    )

    if not tool_names:
        raise ValueError("At least one tool must be configured.")

    invalid_tools = set(tool_names) - {
        "bm25",
        "dense",
        "hybrid",
        "rerank",
    }

    if invalid_tools:
        raise ValueError(
            "Unknown tools: "
            + ", ".join(sorted(invalid_tools))
        )

    return tool_names


def build_retrievers(
    chunks: Sequence[Document],
    *,
    embeddings: EmbeddingModel,
    candidate_k: int = 20,
    reranker: Reranker | None = None,
) -> dict[str, Retriever]:
    bm25_retriever = build_bm25_retriever(chunks)
    dense_retriever = build_dense_retriever(
        chunks,
        embeddings=embeddings,
    )
    hybrid_retriever = build_rrf_retriever(
        bm25_retriever,
        dense_retriever,
        candidate_k=candidate_k,
    )

    retrievers: dict[str, Retriever] = {
        "bm25": bm25_retriever,
        "dense": dense_retriever,
        "hybrid": hybrid_retriever,
    }

    if reranker is not None:
        retrievers["rerank"] = build_reranking_retriever(
            hybrid_retriever,
            reranker,
            candidate_k=candidate_k,
        )

    return retrievers


def build_tools(
    tool_names: Sequence[str],
    retrievers: dict[str, Retriever],
) -> tuple[RetrievalTool, ...]:
    return tuple(
        RetrievalTool(
            name=tool_name,
            retriever=retrievers[tool_name],
        )
        for tool_name in tool_names
    )


def build_planner(
    planner_name: str,
    settings: Settings,
) -> QueryPlanner:
    if planner_name == "heuristic":
        return HeuristicQueryPlanner()

    if planner_name == "llm":
        return LLMDynamicQueryPlanner(
            GeminiPlannerLLM(
                api_key=settings.require_google_api_key(),
                model=settings.gemini_chat_model,
            )
        )

    raise ValueError(f"Unknown planner: {planner_name}")


def build_service(
    tools: Sequence[RetrievalTool],
    settings: Settings,
    *,
    top_k: int,
    max_correction_attempts: int = 1,
    planner: QueryPlanner | None = None,
) -> AgenticRAGService:
    api_key = settings.require_google_api_key()

    return AgenticRAGService(
        tools=tools,
        generator=GeminiAnswerGenerator(
            api_key=api_key,
            model=settings.gemini_chat_model,
        ),
        planner=planner,
        default_top_k=top_k,
        max_correction_attempts=max_correction_attempts,
    )


def format_answer_response(
    response: AnswerResponse,
) -> str:
    lines = [
        "ANSWER",
        "======",
        response.answer,
        "",
        "AGENT TRACE",
        "===========",
    ]

    for step in response.agent_trace.planned_steps:
        lines.append(
            f"- {step.tool_name}: {step.query} (top_k={step.top_k})"
        )

        if step.reason is not None:
            lines.append(f"  reason={step.reason}")

    lines.extend(
        [
            "",
            "EVIDENCE REVIEWS",
            "================",
        ]
    )

    for review in response.agent_trace.evidence_reviews:
        lines.append(
            f"- attempt={review.attempt} "
            f"query={review.query!r} "
            f"sufficient={review.is_sufficient} "
            f"score={review.score:.2f}"
        )
        lines.append(f"  reason={review.reason}")

        if review.corrective_query is not None:
            lines.append(
                f"  corrective_query={review.corrective_query!r}"
            )

    lines.extend(
        [
            "",
            "CITATIONS",
            "=========",
        ]
    )

    if response.citations:
        for citation in response.citations:
            lines.append(
                f"[{citation.index}] {citation.chunk_id} ({citation.source})"
            )
    else:
        lines.append("None")

    lines.extend(
        [
            "",
            "RETRIEVED CHUNKS",
            "================",
        ]
    )

    if response.retrieved_chunks:
        for chunk in response.retrieved_chunks:
            lines.append(
                
                    f"{chunk.rank}. {chunk.chunk_id} "
                    f"score={chunk.score:.4f} "
                    f"tool={chunk.tool_name}:{chunk.tool_rank}"
                
            )
    else:
        lines.append("None")

    return "\n".join(lines)


def main() -> None:
    args = build_argument_parser().parse_args()

    if args.top_k <= 0:
        raise ValueError("top-k must be greater than 0")

    if args.candidate_k <= 0:
        raise ValueError("candidate-k must be greater than 0")

    if args.max_correction_attempts < 0:
        raise ValueError("max-correction-attempts must not be negative")

    tool_names = parse_tool_names(args.tools)
    documents = load_documents()
    chunks = chunk_documents(documents)
    settings = Settings()
    retrievers = build_retrievers(
        chunks,
        embeddings=build_gemini_embeddings(),
        candidate_k=args.candidate_k,
        reranker=(
            build_cross_encoder_reranker()
            if "rerank" in tool_names
            else None
        ),
    )
    service = build_service(
        build_tools(
            tool_names,
            retrievers,
        ),
        settings,
        top_k=args.top_k,
        max_correction_attempts=args.max_correction_attempts,
        planner=build_planner(args.planner, settings),
    )
    response = service.ask(
        AskRequest(
            query=args.query,
            top_k=args.top_k,
        )
    )

    print(format_answer_response(response))


if __name__ == "__main__":
    main()

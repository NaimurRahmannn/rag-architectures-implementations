from __future__ import annotations

import argparse
from collections.abc import Sequence

from langchain_core.documents import Document

from apps.baseline_rag.app.settings import Settings
from apps.hybrid_rag.app.generation import GeminiAnswerGenerator
from apps.hybrid_rag.app.retrieval import (
    EmbeddingModel,
    Retriever,
)
from apps.hybrid_rag.app.schemas import (
    AnswerResponse,
    AskRequest,
)
from apps.hybrid_rag.app.service import HybridRAGService
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

RetrieverName = str


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a grounded Hybrid RAG answer."
    )
    parser.add_argument(
        "query",
        help="Question to answer from the configured document chunks.",
    )
    parser.add_argument(
        "--retriever",
        choices=("bm25", "dense", "rrf", "rerank"),
        default="rerank",
        help=(
            "Retrieval strategy to use. "
            "Default: rerank, which reranks RRF candidates."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of final chunks to send to the answer generator.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="Number of candidates to gather before RRF/reranking.",
    )
    return parser


def build_retriever(
    name: RetrieverName,
    chunks: Sequence[Document],
    *,
    embeddings: EmbeddingModel | None = None,
    candidate_k: int = 20,
) -> Retriever:
    if name == "bm25":
        return build_bm25_retriever(chunks)

    if embeddings is None:
        embeddings = build_gemini_embeddings()

    bm25_retriever = build_bm25_retriever(chunks)
    dense_retriever = build_dense_retriever(
        chunks,
        embeddings=embeddings,
    )

    if name == "dense":
        return dense_retriever

    rrf_retriever = build_rrf_retriever(
        bm25_retriever,
        dense_retriever,
        candidate_k=candidate_k,
    )

    if name == "rrf":
        return rrf_retriever

    if name == "rerank":
        return build_reranking_retriever(
            rrf_retriever,
            build_cross_encoder_reranker(),
            candidate_k=candidate_k,
        )

    raise ValueError(f"Unknown retriever: {name}")


def build_service(
    retriever: Retriever,
    settings: Settings,
    *,
    top_k: int,
) -> HybridRAGService:
    api_key = settings.require_google_api_key()

    return HybridRAGService(
        retriever=retriever,
        generator=GeminiAnswerGenerator(
            api_key=api_key,
            model=settings.gemini_chat_model,
        ),
        default_top_k=top_k,
    )


def format_answer_response(
    response: AnswerResponse,
) -> str:
    lines = [
        "ANSWER",
        "======",
        response.answer,
        "",
        "CITATIONS",
        "=========",
    ]

    if response.citations:
        for citation in response.citations:
            lines.append(
                
                    f"[{citation.index}] "
                    f"{citation.chunk_id} "
                    f"({citation.source})"
                
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
                    f"source={chunk.source}"
                
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

    documents = load_documents()
    chunks = chunk_documents(documents)
    settings = Settings()
    retriever = build_retriever(
        args.retriever,
        chunks,
        candidate_k=args.candidate_k,
    )
    service = build_service(
        retriever,
        settings,
        top_k=args.top_k,
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

from __future__ import annotations

import re
from collections.abc import Sequence

from apps.hybrid_rag.app.generation import (
    ABSTENTION_MESSAGE,
    AnswerGenerator,
    ContextSource,
    build_context_sources,
    format_context,
)
from apps.hybrid_rag.app.retrieval import (
    Retriever,
)
from apps.hybrid_rag.app.schemas import (
    AnswerResponse,
    AskRequest,
    Citation,
    RetrievedChunk,
)


class HybridRAGService:
    def __init__(
        self,
        retriever: Retriever,
        generator: AnswerGenerator,
        *,
        default_top_k: int = 5,
    ) -> None:
        if default_top_k <= 0:
            raise ValueError(
                "default_top_k must be greater than 0"
            )

        self.retriever = retriever
        self.generator = generator
        self.default_top_k = default_top_k

    def ask(
        self,
        request: AskRequest,
    ) -> AnswerResponse:
        top_k = (
            request.top_k
            or self.default_top_k
        )
        results = self.retriever.search(
            request.query,
            top_k=top_k,
        )
        context_sources = build_context_sources(results)
        retrieved_chunks = to_retrieved_chunks(
            context_sources
        )

        if not results:
            return AnswerResponse(
                query=request.query,
                answer=ABSTENTION_MESSAGE,
                citations=[],
                retrieved_chunks=[],
            )

        answer = self.generator.generate(
            request.query,
            format_context(context_sources),
        )
        citations = extract_citations(
            answer,
            retrieved_chunks,
        )

        return AnswerResponse(
            query=request.query,
            answer=answer,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
        )


def to_retrieved_chunks(
    sources: Sequence[ContextSource],
) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            rank=source.rank,
            score=source.score,
            chunk_id=source.chunk_id,
            document_id=source.document_id,
            source=source.source,
            content=source.document.page_content,
            metadata=dict(source.document.metadata),
        )
        for source in sources
    ]


def extract_citations(
    answer: str,
    retrieved_chunks: Sequence[RetrievedChunk],
) -> list[Citation]:
    referenced_indexes = sorted(
        {
            int(value)
            for value in re.findall(
                r"\[(\d+)\]",
                answer,
            )
        }
    )

    citations: list[Citation] = []

    for index in referenced_indexes:
        if not 1 <= index <= len(retrieved_chunks):
            continue

        chunk = retrieved_chunks[index - 1]

        citations.append(
            Citation(
                index=index,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                source=chunk.source,
            )
        )

    return citations

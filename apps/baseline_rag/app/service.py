import re

from apps.baseline_rag.app.chunking import (
    BaselineChunker,
)
from apps.baseline_rag.app.generation import (
    AnswerGenerator,
)
from apps.baseline_rag.app.retrieval import (
    RetrievalRepository,
    ScoredDocument,
)
from apps.baseline_rag.app.schemas import (
    AnswerResponse,
    AskRequest,
    Citation,
    DocumentInput,
    IngestResponse,
    RetrievedChunk,
)


ABSTENTION_MESSAGE = (
    "I don't have enough information "
    "in the provided context."
)


class BaselineRAGService:
    def __init__(
        self,
        chunker: BaselineChunker,
        repository: RetrievalRepository,
        generator: AnswerGenerator,
        default_top_k: int,
    ) -> None:
        self._chunker = chunker
        self._repository = repository
        self._generator = generator
        self._default_top_k = default_top_k

    def ingest(
        self,
        documents: list[DocumentInput],
    ) -> IngestResponse:
        chunks = self._chunker.split(documents)
        self._repository.add(chunks)

        return IngestResponse(
            documents_received=len(documents),
            chunks_indexed=len(chunks),
        )

    def ask(
        self,
        request: AskRequest,
    ) -> AnswerResponse:
        top_k = (
            request.top_k
            or self._default_top_k
        )

        scored_documents = self._repository.search(
            query=request.query,
            limit=top_k,
        )

        retrieved_chunks = (
            self._to_retrieved_chunks(
                scored_documents
            )
        )

        if not scored_documents:
            return AnswerResponse(
                query=request.query,
                answer=ABSTENTION_MESSAGE,
                citations=[],
                retrieved_chunks=[],
            )

        documents = [
            item.document
            for item in scored_documents
        ]

        answer = self._generator.generate(
            query=request.query,
            documents=documents,
        )

        citations = self._extract_citations(
            answer=answer,
            retrieved_chunks=retrieved_chunks,
        )

        return AnswerResponse(
            query=request.query,
            answer=answer,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
        )

    @staticmethod
    def _to_retrieved_chunks(
        scored_documents: list[ScoredDocument],
    ) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []

        for rank, item in enumerate(
            scored_documents,
            start=1,
        ):
            metadata = dict(
                item.document.metadata
            )

            chunks.append(
                RetrievedChunk(
                    rank=rank,
                    score=item.score,
                    chunk_id=str(
                        metadata.get(
                            "chunk_id",
                            "unknown",
                        )
                    ),
                    document_id=str(
                        metadata.get(
                            "document_id",
                            "unknown",
                        )
                    ),
                    source=str(
                        metadata.get(
                            "source",
                            "unknown",
                        )
                    ),
                    content=item.document.page_content,
                    metadata=metadata,
                )
            )

        return chunks

    @staticmethod
    def _extract_citations(
        answer: str,
        retrieved_chunks: list[RetrievedChunk],
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
            if not 1 <= index <= len(
                retrieved_chunks
            ):
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
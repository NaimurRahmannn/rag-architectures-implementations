from collections.abc import Sequence

from langchain_core.documents import Document

from apps.hybrid_rag.app.schemas import (
    AnswerResponse,
    Citation,
    RetrievedChunk,
)
from apps.hybrid_rag.scripts.generate_answer import (
    build_argument_parser,
    build_retriever,
    format_answer_response,
)


class KeywordEmbeddingModel:
    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        return [
            self._embed(text)
            for text in texts
        ]

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        return self._embed(text)

    def _embed(
        self,
        text: str,
    ) -> list[float]:
        normalized = text.lower()

        if "authorization" in normalized:
            return [1.0, 0.0]

        if "expired" in normalized:
            return [0.0, 1.0]

        return [0.0, 0.0]


def make_chunks() -> list[Document]:
    return [
        Document(
            page_content="Use the Authorization header.",
            metadata={
                "chunk_id": "chunk-006",
                "source": "authentication_guide.md",
                "document_id": "authentication_guide",
            },
        ),
        Document(
            page_content="Access tokens expire.",
            metadata={
                "chunk_id": "chunk-004",
                "source": "authentication_guide.md",
                "document_id": "authentication_guide",
            },
        ),
    ]


def test_argument_parser_defaults_to_rerank() -> None:
    args = build_argument_parser().parse_args(
        ["How should I send a token?"]
    )

    assert args.query == "How should I send a token?"
    assert args.retriever == "rerank"
    assert args.top_k == 5
    assert args.candidate_k == 20


def test_build_retriever_can_build_bm25_without_embeddings() -> None:
    retriever = build_retriever(
        "bm25",
        make_chunks(),
    )

    results = retriever.search(
        "Authorization",
        top_k=1,
    )

    assert results[0].document.metadata["chunk_id"] == "chunk-006"


def test_build_retriever_can_build_dense_with_injected_embeddings() -> None:
    retriever = build_retriever(
        "dense",
        make_chunks(),
        embeddings=KeywordEmbeddingModel(),
    )

    results = retriever.search(
        "Authorization",
        top_k=1,
    )

    assert results[0].document.metadata["chunk_id"] == "chunk-006"


def test_build_retriever_can_build_rrf_with_injected_embeddings() -> None:
    retriever = build_retriever(
        "rrf",
        make_chunks(),
        embeddings=KeywordEmbeddingModel(),
        candidate_k=2,
    )

    results = retriever.search(
        "Authorization",
        top_k=1,
    )

    assert results[0].document.metadata["chunk_id"] == "chunk-006"


def test_format_answer_response_prints_answer_citations_and_chunks() -> None:
    output = format_answer_response(
        AnswerResponse(
            query="How should I send a token?",
            answer="Use the Authorization header [1].",
            citations=[
                Citation(
                    index=1,
                    chunk_id="chunk-006",
                    document_id="authentication_guide",
                    source="authentication_guide.md",
                )
            ],
            retrieved_chunks=[
                RetrievedChunk(
                    rank=1,
                    score=0.95,
                    chunk_id="chunk-006",
                    document_id="authentication_guide",
                    source="authentication_guide.md",
                    content="Use the Authorization header.",
                    metadata={},
                )
            ],
        )
    )

    assert "ANSWER" in output
    assert "Use the Authorization header [1]." in output
    assert "[1] chunk-006 (authentication_guide.md)" in output
    assert "1. chunk-006 score=0.9500 source=authentication_guide.md" in output

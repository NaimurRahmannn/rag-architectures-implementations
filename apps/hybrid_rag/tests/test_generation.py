from langchain_core.documents import Document

from apps.hybrid_rag.app.generation import (
    ABSTENTION_MESSAGE,
    build_answer_prompt,
    build_context_sources,
    format_context,
)
from apps.hybrid_rag.app.retrieval import ScoredDocument
from apps.hybrid_rag.app.schemas import AskRequest
from apps.hybrid_rag.app.service import (
    HybridRAGService,
    extract_citations,
    to_retrieved_chunks,
)


def make_document(
    chunk_id: str,
    content: str,
    *,
    document_id: str = "authentication_guide",
    source: str = "authentication_guide.md",
    section_title: str | None = None,
) -> Document:
    metadata = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "source": source,
    }

    if section_title is not None:
        metadata["section_title"] = section_title

    return Document(
        page_content=content,
        metadata=metadata,
    )


class FixedRetriever:
    def __init__(
        self,
        results: list[ScoredDocument],
    ) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[ScoredDocument]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


class FakeAnswerGenerator:
    def __init__(
        self,
        answer: str,
    ) -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def generate(
        self,
        query: str,
        context: str,
    ) -> str:
        self.calls.append((query, context))
        return self.answer


def test_context_construction_numbers_sources_and_preserves_metadata() -> None:
    sources = build_context_sources(
        [
            ScoredDocument(
                document=make_document(
                    "chunk-006",
                    "Use the Authorization header.",
                    section_title="Authorization Header",
                ),
                score=0.91,
            )
        ]
    )

    assert sources[0].index == 1
    assert sources[0].rank == 1
    assert sources[0].score == 0.91
    assert sources[0].chunk_id == "chunk-006"
    assert sources[0].document_id == "authentication_guide"
    assert sources[0].source == "authentication_guide.md"
    assert sources[0].section_title == "Authorization Header"


def test_format_context_builds_numbered_source_blocks() -> None:
    sources = build_context_sources(
        [
            ScoredDocument(
                document=make_document(
                    "chunk-006",
                    "Use the Authorization header.",
                    section_title="Authorization Header",
                ),
                score=0.91,
            )
        ]
    )

    context = format_context(sources)

    assert "[1]" in context
    assert "Source: authentication_guide.md" in context
    assert "Document ID: authentication_guide" in context
    assert "Chunk ID: chunk-006" in context
    assert "Section: Authorization Header" in context
    assert "Content:\nUse the Authorization header." in context


def test_prompt_construction_includes_question_and_context() -> None:
    prompt = build_answer_prompt()
    messages = prompt.format_messages(
        question="How should an access token be sent?",
        context="[1]\nContent:\nUse the Authorization header.",
    )

    assert "grounded question-answering assistant" in messages[0].content
    assert "How should an access token be sent?" in messages[1].content
    assert "Use the Authorization header." in messages[1].content


def test_service_retrieves_constructs_context_generates_and_maps_citations() -> None:
    document = make_document(
        "chunk-006",
        "Protected API requests use the Authorization header.",
    )
    retriever = FixedRetriever(
        [
            ScoredDocument(
                document=document,
                score=0.95,
            )
        ]
    )
    generator = FakeAnswerGenerator(
        "Send the token in the Authorization header [1]."
    )
    service = HybridRAGService(
        retriever,
        generator,
        default_top_k=5,
    )

    response = service.ask(
        AskRequest(
            query="How should an access token be sent?",
            top_k=3,
        )
    )

    assert retriever.calls == [
        (
            "How should an access token be sent?",
            3,
        )
    ]
    assert generator.calls[0][0] == "How should an access token be sent?"
    assert "Chunk ID: chunk-006" in generator.calls[0][1]
    assert response.answer == "Send the token in the Authorization header [1]."
    assert response.retrieved_chunks[0].chunk_id == "chunk-006"
    assert response.citations[0].index == 1
    assert response.citations[0].chunk_id == "chunk-006"
    assert response.citations[0].source == "authentication_guide.md"


def test_service_abstains_without_calling_llm_when_retrieval_is_empty() -> None:
    retriever = FixedRetriever([])
    generator = FakeAnswerGenerator("This should not be used.")
    service = HybridRAGService(
        retriever,
        generator,
    )

    response = service.ask(
        AskRequest(
            query="What is the billing policy?"
        )
    )

    assert response.answer == ABSTENTION_MESSAGE
    assert response.citations == []
    assert response.retrieved_chunks == []
    assert generator.calls == []


def test_extract_citations_ignores_invalid_references() -> None:
    sources = build_context_sources(
        [
            ScoredDocument(
                document=make_document(
                    "chunk-001",
                    "AUTH-401 means credentials are missing.",
                ),
                score=1.0,
            )
        ]
    )
    retrieved_chunks = to_retrieved_chunks(sources)

    citations = extract_citations(
        "Supported claim [1]. Invalid claim [9].",
        retrieved_chunks,
    )

    assert [
        citation.index
        for citation in citations
    ] == [1]
    assert citations[0].chunk_id == "chunk-001"

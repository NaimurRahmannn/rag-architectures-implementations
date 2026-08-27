from fastapi.testclient import TestClient
from langchain_core.documents import Document

from apps.baseline_rag.app.chunking import (
    BaselineChunker,
)
from apps.baseline_rag.app.main import create_app
from apps.baseline_rag.app.retrieval import (
    ScoredDocument,
)
from apps.baseline_rag.app.schemas import (
    AskRequest,
    DocumentInput,
)
from apps.baseline_rag.app.service import (
    ABSTENTION_MESSAGE,
    BaselineRAGService,
)


class FakeRepository:
    def __init__(self) -> None:
        self.documents: list[Document] = []

    def add(
        self,
        documents: list[Document],
    ) -> None:
        self.documents.extend(documents)

    def search(
        self,
        query: str,
        limit: int,
    ) -> list[ScoredDocument]:
        del query

        return [
            ScoredDocument(
                document=document,
                score=1.0 / rank,
            )
            for rank, document in enumerate(
                self.documents[:limit],
                start=1,
            )
        ]


class FakeGenerator:
    def __init__(
        self,
        answer: str = (
            "Employees receive "
            "20 leave days [1]."
        ),
    ) -> None:
        self.answer = answer
        self.calls = 0

    def generate(
        self,
        query: str,
        documents: list[Document],
    ) -> str:
        del query
        del documents

        self.calls += 1
        return self.answer


def build_test_service(
    repository: FakeRepository | None = None,
    generator: FakeGenerator | None = None,
) -> BaselineRAGService:
    return BaselineRAGService(
        chunker=BaselineChunker(
            chunk_size=100,
            chunk_overlap=20,
        ),
        repository=(
            repository
            or FakeRepository()
        ),
        generator=(
            generator
            or FakeGenerator()
        ),
        default_top_k=4,
    )


def test_chunker_creates_deterministic_ids() -> None:
    chunker = BaselineChunker(
        chunk_size=25,
        chunk_overlap=5,
    )

    document = DocumentInput(
        id="policy-1",
        source="policy.md",
        content=(
            "Annual leave is twenty days. "
            "Requests require approval."
        ),
    )

    first = chunker.split([document])
    second = chunker.split([document])

    assert len(first) > 1

    assert [
        chunk.metadata["chunk_id"]
        for chunk in first
    ] == [
        chunk.metadata["chunk_id"]
        for chunk in second
    ]


def test_service_maps_citations() -> None:
    service = build_test_service()

    service.ingest(
        [
            DocumentInput(
                id="leave-policy",
                source="leave_policy.md",
                content=(
                    "Employees receive "
                    "20 days of annual leave."
                ),
            )
        ]
    )

    response = service.ask(
        AskRequest(
            query="How many leave days?"
        )
    )

    assert response.citations[0].index == 1
    assert (
        response.citations[0].document_id
        == "leave-policy"
    )


def test_service_abstains_when_retrieval_is_empty() -> None:
    generator = FakeGenerator()

    service = build_test_service(
        generator=generator
    )

    response = service.ask(
        AskRequest(
            query="What is the refund policy?"
        )
    )

    assert response.answer == ABSTENTION_MESSAGE
    assert response.citations == []
    assert generator.calls == 0


def test_invalid_citation_is_ignored() -> None:
    generator = FakeGenerator(
        answer=(
            "Supported information [1]. "
            "Invalid citation [9]."
        )
    )

    service = build_test_service(
        generator=generator
    )

    service.ingest(
        [
            DocumentInput(
                id="document-1",
                source="document.md",
                content="Supported information.",
            )
        ]
    )

    response = service.ask(
        AskRequest(query="What is supported?")
    )

    assert [
        citation.index
        for citation in response.citations
    ] == [1]


def test_fastapi_round_trip() -> None:
    app = create_app(
        build_test_service()
    )

    client = TestClient(app)

    ingest_response = client.post(
        "/documents",
        json={
            "documents": [
                {
                    "id": "leave-policy",
                    "source": "leave_policy.md",
                    "content": (
                        "Employees receive "
                        "20 days of annual leave."
                    ),
                }
            ]
        },
    )

    answer_response = client.post(
        "/ask",
        json={
            "query": "How many leave days?",
            "top_k": 4,
        },
    )

    assert ingest_response.status_code == 200
    assert answer_response.status_code == 200

    assert (
        answer_response.json()["citations"][0][
            "source"
        ]
        == "leave_policy.md"
    )
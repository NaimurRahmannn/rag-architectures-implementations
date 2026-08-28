from collections.abc import Sequence

from langchain_core.documents import Document

from apps.hybrid_rag.app.retrieval import (
    InMemoryDenseIndex,
    cosine_similarity,
)


class ToyEmbeddingModel:
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

        if (
            "authorization" in normalized
            or "bearer" in normalized
            or "sent to the api" in normalized
        ):
            return [1.0, 0.0, 0.0]

        if (
            "expire" in normalized
            or "expired" in normalized
            or "refresh" in normalized
        ):
            return [0.0, 1.0, 0.0]

        if (
            "permission" in normalized
            or "forbidden" in normalized
        ):
            return [0.0, 0.0, 1.0]

        return [0.0, 0.0, 0.0]


def test_cosine_similarity_scores_identical_vectors_highest() -> None:
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert cosine_similarity([1, 0], [0, 1]) == 0.0


def test_dense_index_returns_semantically_matching_document_first() -> None:
    documents = [
        Document(
            page_content=(
                "Access tokens expire after a fixed lifetime. "
                "Clients should refresh the token."
            ),
            metadata={"chunk_id": "chunk-004"},
        ),
        Document(
            page_content=(
                "Protected API requests must include the access token "
                "using the Authorization header. "
                "Authorization: Bearer <access_token>"
            ),
            metadata={"chunk_id": "chunk-006"},
        ),
        Document(
            page_content=(
                "AUTH-403 indicates that the user does not have permission."
            ),
            metadata={"chunk_id": "chunk-002"},
        ),
    ]

    index = InMemoryDenseIndex(
        documents,
        embeddings=ToyEmbeddingModel(),
    )

    results = index.search(
        "How should an access token be sent to the API?",
        top_k=3,
    )

    assert results[0].document.metadata["chunk_id"] == "chunk-006"
    assert results[0].score > results[1].score


def test_dense_index_validates_query_vector_dimension() -> None:
    class BadQueryEmbeddingModel(ToyEmbeddingModel):
        def embed_query(
            self,
            text: str,
        ) -> list[float]:
            return [1.0, 0.0]

    documents = [
        Document(page_content="Authorization header"),
    ]

    index = InMemoryDenseIndex(
        documents,
        embeddings=BadQueryEmbeddingModel(),
    )

    try:
        index.search("Authorization header")
    except ValueError as error:
        assert str(error) == "Query vector dimension does not match the index."
    else:
        raise AssertionError("Expected ValueError")

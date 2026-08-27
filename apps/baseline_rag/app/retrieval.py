from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
)
from langchain_qdrant import (
    QdrantVectorStore,
    RetrievalMode,
)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from apps.baseline_rag.app.settings import Settings


@dataclass(frozen=True)
class ScoredDocument:
    document: Document
    score: float


class RetrievalRepository(Protocol):
    def add(
        self,
        documents: list[Document],
    ) -> None:
        """Index documents."""

    def search(
        self,
        query: str,
        limit: int,
    ) -> list[ScoredDocument]:
        """Return documents ordered by relevance."""


class QdrantDenseRepository:
    def __init__(
        self,
        vector_store: QdrantVectorStore,
    ) -> None:
        self._vector_store = vector_store

    def add(
        self,
        documents: list[Document],
    ) -> None:
        if not documents:
            return

        ids = [
            str(document.metadata["chunk_id"])
            for document in documents
        ]

        self._vector_store.add_documents(
            documents=documents,
            ids=ids,
        )

    def search(
        self,
        query: str,
        limit: int,
    ) -> list[ScoredDocument]:
        results = (
            self._vector_store.similarity_search_with_score(
                query,
                k=limit,
            )
        )

        return [
            ScoredDocument(
                document=document,
                score=float(score),
            )
            for document, score in results
        ]


def build_qdrant_repository(
    settings: Settings,
    client: QdrantClient | None = None,
) -> QdrantDenseRepository:
    api_key = settings.require_google_api_key()

    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=api_key,
    )

    if client is None:
        qdrant_path = Path(settings.qdrant_path)
        qdrant_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        client = QdrantClient(
            path=str(qdrant_path)
        )

    if not client.collection_exists(
        settings.qdrant_collection
    ):
        vector_size = len(
            embeddings.embed_query(
                "vector dimension probe"
            )
        )

        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
        retrieval_mode=RetrievalMode.DENSE,
    )

    return QdrantDenseRepository(vector_store)
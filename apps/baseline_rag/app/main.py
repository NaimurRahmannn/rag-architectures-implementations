from functools import lru_cache

from fastapi import FastAPI, HTTPException

from apps.baseline_rag.app.chunking import (
    BaselineChunker,
)
from apps.baseline_rag.app.generation import (
    GeminiAnswerGenerator,
)
from apps.baseline_rag.app.retrieval import (
    build_qdrant_repository,
)
from apps.baseline_rag.app.schemas import (
    AnswerResponse,
    AskRequest,
    IngestRequest,
    IngestResponse,
)
from apps.baseline_rag.app.service import (
    BaselineRAGService,
)
from apps.baseline_rag.app.settings import (
    ConfigurationError,
    Settings,
)


@lru_cache(maxsize=1)
def build_default_service() -> BaselineRAGService:
    settings = Settings()
    api_key = settings.require_google_api_key()

    return BaselineRAGService(
        chunker=BaselineChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
        repository=build_qdrant_repository(
            settings
        ),
        generator=GeminiAnswerGenerator(
            api_key=api_key,
            model=settings.gemini_chat_model,
        ),
        default_top_k=settings.default_top_k,
    )


def create_app(
    service: BaselineRAGService | None = None,
) -> FastAPI:
    app = FastAPI(
        title=(
            "Baseline RAG with "
            "Gemini and LangChain"
        ),
        version="0.1.0",
    )

    def get_service() -> BaselineRAGService:
        if service is not None:
            return service

        return build_default_service()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "architecture": "baseline_rag",
        }

    @app.post(
        "/documents",
        response_model=IngestResponse,
    )
    def ingest_documents(
        request: IngestRequest,
    ) -> IngestResponse:
        try:
            return get_service().ingest(
                request.documents
            )
        except ConfigurationError as exc:
            raise HTTPException(
                status_code=503,
                detail=str(exc),
            ) from exc

    @app.post(
        "/ask",
        response_model=AnswerResponse,
    )
    def ask_question(
        request: AskRequest,
    ) -> AnswerResponse:
        try:
            return get_service().ask(request)
        except ConfigurationError as exc:
            raise HTTPException(
                status_code=503,
                detail=str(exc),
            ) from exc

    return app


app = create_app()
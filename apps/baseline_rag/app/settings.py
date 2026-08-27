from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SHARED_ENV_FILE = REPOSITORY_ROOT / ".env"


class ConfigurationError(RuntimeError):
    """Raised when required application configuration is missing."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=SHARED_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    google_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
        ),
    )

    gemini_chat_model: str = Field(
        default="gemini-3.7-flash",
        validation_alias="GEMINI_CHAT_MODEL",
    )

    gemini_embedding_model: str = Field(
        default="gemini-embedding-001",
        validation_alias="GEMINI_EMBEDDING_MODEL",
    )

    qdrant_path: Path = Field(
        default=Path(".data/qdrant"),
        validation_alias="QDRANT_PATH",
    )

    qdrant_collection: str = Field(
        default="baseline_rag_documents",
        validation_alias="QDRANT_COLLECTION",
    )

    chunk_size: int = Field(
        default=1000,
        gt=0,
        validation_alias="CHUNK_SIZE",
    )

    chunk_overlap: int = Field(
        default=200,
        ge=0,
        validation_alias="CHUNK_OVERLAP",
    )

    default_top_k: int = Field(
        default=4,
        ge=1,
        le=20,
        validation_alias="DEFAULT_TOP_K",
    )

    @model_validator(mode="after")
    def validate_chunk_configuration(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "CHUNK_OVERLAP must be smaller than CHUNK_SIZE"
            )

        return self

    def require_google_api_key(self) -> str:
        if self.google_api_key is None:
            raise ConfigurationError(
                "GOOGLE_API_KEY or GEMINI_API_KEY is missing. "
                f"Configure it in the shared environment file: {SHARED_ENV_FILE}"
            )

        return self.google_api_key.get_secret_value()

from typing import Any

from pydantic import BaseModel, Field


class DocumentInput(BaseModel):
    id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[DocumentInput] = Field(min_length=1)


class IngestResponse(BaseModel):
    documents_received: int
    chunks_indexed: int


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


class RetrievedChunk(BaseModel):
    rank: int
    score: float
    chunk_id: str
    document_id: str
    source: str
    content: str
    metadata: dict[str, Any]


class Citation(BaseModel):
    index: int
    chunk_id: str
    document_id: str
    source: str


class AnswerResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
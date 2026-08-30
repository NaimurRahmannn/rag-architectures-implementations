from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


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

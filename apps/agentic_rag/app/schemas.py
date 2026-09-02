from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


class PlannedStep(BaseModel):
    tool_name: str
    query: str
    top_k: int


class ToolCall(BaseModel):
    tool_name: str
    query: str
    top_k: int
    retrieved_chunk_ids: list[str]


class EvidenceReview(BaseModel):
    attempt: int
    query: str
    is_sufficient: bool
    score: float
    matched_terms: list[str]
    missing_terms: list[str]
    reason: str
    corrective_query: str | None = None


class AgentTrace(BaseModel):
    original_query: str
    planned_steps: list[PlannedStep]
    tool_calls: list[ToolCall]
    evidence_reviews: list[EvidenceReview] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    rank: int
    score: float
    chunk_id: str
    document_id: str
    source: str
    content: str
    metadata: dict[str, Any]
    tool_name: str
    tool_rank: int


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
    agent_trace: AgentTrace

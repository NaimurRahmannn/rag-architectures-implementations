import json
from pathlib import Path

from pydantic import BaseModel, Field

from apps.baseline_rag.app.schemas import (
    DocumentInput,
)


class RetrievalEvaluationCase(BaseModel):
    id: str = Field(min_length=1)
    query: str = Field(min_length=1)

    relevant_document_ids: set[str] = Field(
        min_length=1
    )


class RetrievalEvaluationDataset(BaseModel):
    documents: list[DocumentInput] = Field(
        min_length=1
    )

    cases: list[RetrievalEvaluationCase] = Field(
        min_length=1
    )


def load_evaluation_dataset(
    path: Path,
) -> RetrievalEvaluationDataset:
    raw_data = json.loads(
        path.read_text(encoding="utf-8")
    )

    return RetrievalEvaluationDataset.model_validate(
        raw_data
    )
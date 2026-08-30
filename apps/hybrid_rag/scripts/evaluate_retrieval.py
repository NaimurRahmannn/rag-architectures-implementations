from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from apps.hybrid_rag.app.bm25 import BM25Index
from apps.hybrid_rag.app.evaluation import (
    EvaluationExample,
    EvaluationSummary,
    evaluate_dataset,
)
from apps.hybrid_rag.app.retrieval import (
    BM25Retriever,
    Retriever,
    get_chunk_id,
)
from apps.hybrid_rag.scripts.inspect_chunks import (
    chunk_documents,
    load_documents,
)

EVAL_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "bm25_eval.json"
)


def load_evaluation_examples(
    path: Path = EVAL_FIXTURE_PATH,
) -> list[EvaluationExample]:
    data = json.loads(path.read_text(encoding="utf-8"))

    return [
        EvaluationExample(
            query=str(item["query"]),
            relevant_chunk_ids=tuple(
                str(chunk_id)
                for chunk_id in item["relevant_chunk_ids"]
            ),
        )
        for item in data
    ]


def retrieve_chunk_ids(
    retriever: Retriever,
    query: str,
    top_k: int,
) -> list[str]:
    results = retriever.search(
        query,
        top_k=top_k,
    )

    return [
        get_chunk_id(result.document)
        for result in results
    ]


def evaluate_retriever(
    retriever: Retriever,
    examples: Sequence[EvaluationExample],
    *,
    top_k: int = 5,
    k_values: Sequence[int] = (1, 3, 5),
) -> EvaluationSummary:
    return evaluate_dataset(
        examples,
        retrieve=lambda query: retrieve_chunk_ids(
            retriever,
            query,
            top_k=top_k,
        ),
        k_values=k_values,
    )


def print_summary(summary: EvaluationSummary) -> None:
    print("Recall:")

    for k, score in summary.recall_at_k.items():
        print(f"  @{k}: {score:.3f}")

    print(f"MRR: {summary.mean_reciprocal_rank:.3f}")

    for query in summary.queries:
        print()
        print("QUERY:", query.query)
        print("RELEVANT:", ", ".join(query.relevant_chunk_ids))
        print("RETRIEVED:", ", ".join(query.retrieved_chunk_ids))
        print(f"RR: {query.reciprocal_rank:.3f}")


def main() -> None:
    documents = load_documents()
    chunks = chunk_documents(documents)
    retriever = BM25Retriever(
        BM25Index(chunks)
    )

    summary = evaluate_retriever(
        retriever,
        load_evaluation_examples(),
        top_k=5,
        k_values=(1, 3, 5),
    )

    print_summary(summary)


if __name__ == "__main__":
    main()

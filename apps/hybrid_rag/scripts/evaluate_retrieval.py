from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from apps.hybrid_rag.app.bm25 import BM25Index
from apps.hybrid_rag.app.evaluation import (
    EvaluationExample,
    EvaluationSummary,
    evaluate_dataset,
)
from apps.hybrid_rag.app.retrieval import (
    BM25Retriever,
    DenseRetriever,
    EmbeddingModel,
    InMemoryDenseIndex,
    Retriever,
    get_chunk_id,
)
from apps.hybrid_rag.scripts.inspect_chunks import (
    chunk_documents,
    load_documents,
)


@dataclass(frozen=True)
class BaselineEvaluation:
    name: str
    summary: EvaluationSummary


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


def build_bm25_retriever(
    chunks: Sequence[Document],
) -> BM25Retriever:
    return BM25Retriever(
        BM25Index(chunks)
    )


def build_dense_retriever(
    chunks: Sequence[Document],
    embeddings: EmbeddingModel,
) -> DenseRetriever:
    return DenseRetriever(
        InMemoryDenseIndex(
            chunks,
            embeddings=embeddings,
        )
    )


def build_gemini_embeddings() -> EmbeddingModel:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    from apps.baseline_rag.app.settings import Settings

    settings = Settings()
    api_key = settings.require_google_api_key()

    return GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=api_key,
    )


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


def evaluate_baselines(
    baselines: Sequence[tuple[str, Retriever]],
    examples: Sequence[EvaluationExample],
    *,
    top_k: int = 5,
    k_values: Sequence[int] = (1, 3, 5),
) -> tuple[BaselineEvaluation, ...]:
    return tuple(
        BaselineEvaluation(
            name=name,
            summary=evaluate_retriever(
                retriever,
                examples,
                top_k=top_k,
                k_values=k_values,
            ),
        )
        for name, retriever in baselines
    )


def format_rank(
    relevant_chunk_ids: Sequence[str],
    retrieved_chunk_ids: Sequence[str],
) -> str:
    relevant = set(relevant_chunk_ids)

    for rank, chunk_id in enumerate(
        retrieved_chunk_ids,
        start=1,
    ):
        if chunk_id in relevant:
            return str(rank)

    return "not found"


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


def print_baseline_comparison(
    baselines: Sequence[BaselineEvaluation],
    *,
    k_values: Sequence[int],
) -> None:
    name_width = max(
        8,
        *(
            len(baseline.name)
            for baseline in baselines
        ),
    )
    metric_width = 12

    print("RETRIEVAL BASELINES")
    print("===================")
    print()
    print(
        f"{'Metric':<{metric_width}}"
        + "".join(
            f"{baseline.name:>{name_width + 2}}"
            for baseline in baselines
        )
    )
    print(
        "-" * (
            metric_width
            + len(baselines) * (name_width + 2)
        )
    )

    for k in k_values:
        print(
            f"{f'Recall@{k}':<{metric_width}}"
            + "".join(
                f"{baseline.summary.recall_at_k[k]:>{name_width + 2}.3f}"
                for baseline in baselines
            )
        )

    print(
        f"{'MRR':<{metric_width}}"
        + "".join(
            f"{baseline.summary.mean_reciprocal_rank:>{name_width + 2}.3f}"
            for baseline in baselines
        )
    )


def print_query_comparison(
    baselines: Sequence[BaselineEvaluation],
) -> None:
    if not baselines:
        return

    queries = baselines[0].summary.queries

    for query_index, query in enumerate(queries):
        print()
        print("Query:")
        print(query.query)
        print()
        print("Relevant:")
        print(", ".join(query.relevant_chunk_ids))

        for baseline in baselines:
            baseline_query = baseline.summary.queries[query_index]
            rank = format_rank(
                query.relevant_chunk_ids,
                baseline_query.retrieved_chunk_ids,
            )

            print()
            print(baseline.name)

            if baseline_query.retrieved_chunk_ids:
                for result_rank, chunk_id in enumerate(
                    baseline_query.retrieved_chunk_ids,
                    start=1,
                ):
                    print(f"{result_rank}. {chunk_id}")
            else:
                print("No results")

            print(f"{baseline.name} rank: {rank}")


def main() -> None:
    documents = load_documents()
    chunks = chunk_documents(documents)
    examples = load_evaluation_examples()
    k_values = (1, 3, 5)

    bm25_retriever = build_bm25_retriever(chunks)
    dense_retriever = build_dense_retriever(
        chunks,
        embeddings=build_gemini_embeddings(),
    )

    baselines = evaluate_baselines(
        (
            ("BM25", bm25_retriever),
            ("Dense", dense_retriever),
        ),
        examples,
        top_k=5,
        k_values=k_values,
    )

    print_baseline_comparison(
        baselines,
        k_values=k_values,
    )
    print_query_comparison(baselines)


if __name__ == "__main__":
    main()

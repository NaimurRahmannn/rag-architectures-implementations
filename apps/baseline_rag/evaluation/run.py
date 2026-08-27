from pathlib import Path

from qdrant_client import QdrantClient

from apps.baseline_rag.app.chunking import (
    BaselineChunker,
)
from apps.baseline_rag.app.retrieval import (
    build_qdrant_repository,
)
from apps.baseline_rag.app.settings import Settings
from apps.baseline_rag.evaluation.dataset import (
    load_evaluation_dataset,
)
from apps.baseline_rag.evaluation.metrics import (
    RetrievalMetrics,
    average_metrics,
    evaluate_ranking,
)


def unique_document_ids(
    document_ids: list[str],
) -> list[str]:
    return list(dict.fromkeys(document_ids))


def run_evaluation() -> None:
    dataset_path = Path(__file__).with_name(
        "dataset.json"
    )

    dataset = load_evaluation_dataset(
        dataset_path
    )

    settings = Settings()

    chunker = BaselineChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    repository = build_qdrant_repository(
        settings=settings,
        client=QdrantClient(":memory:"),
    )

    chunks = chunker.split(dataset.documents)
    repository.add(chunks)

    k = settings.default_top_k
    all_results: list[RetrievalMetrics] = []

    for case in dataset.cases:
        retrieved = repository.search(
            query=case.query,
            limit=k,
        )

        retrieved_document_ids = (
            unique_document_ids(
                [
                    str(
                        item.document.metadata[
                            "document_id"
                        ]
                    )
                    for item in retrieved
                ]
            )
        )

        metrics = evaluate_ranking(
            retrieved_ids=retrieved_document_ids,
            relevant_ids=case.relevant_document_ids,
            k=k,
        )

        all_results.append(metrics)

        print(
            f"{case.id}: "
            f"Recall@{k}={metrics.recall_at_k:.3f}, "
            f"Precision@{k}={metrics.precision_at_k:.3f}, "
            f"MRR={metrics.reciprocal_rank:.3f}, "
            f"nDCG@{k}={metrics.ndcg_at_k:.3f}"
        )

    average = average_metrics(all_results)

    print("\nAverage retrieval performance")
    print(
        f"Recall@{k}: "
        f"{average.recall_at_k:.3f}"
    )
    print(
        f"Precision@{k}: "
        f"{average.precision_at_k:.3f}"
    )
    print(
        f"MRR: "
        f"{average.reciprocal_rank:.3f}"
    )
    print(
        f"nDCG@{k}: "
        f"{average.ndcg_at_k:.3f}"
    )


if __name__ == "__main__":
    run_evaluation()
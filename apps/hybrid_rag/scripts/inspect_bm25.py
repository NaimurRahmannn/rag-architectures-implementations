from __future__ import annotations

import json
from pathlib import Path

from apps.hybrid_rag.app.bm25 import BM25Index
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


def load_eval_queries(path: Path = EVAL_FIXTURE_PATH) -> list[dict[str, list[str] | str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def print_index_statistics(index: BM25Index) -> None:
    print("Document count:")
    print(index.document_count)

    print("\nAverage document length:")
    print(index.average_document_length)

    print("\nDocument statistics:")

    for stat in index.statistics():
        print()
        print("Chunk:", stat.document.metadata["chunk_id"])
        print("Section:", stat.document.metadata["section_title"])
        print("Tokens:", stat.tokens)
        print("Length:", stat.length)
        print("Term frequencies:", stat.term_frequencies)

    print("\nDocument frequencies:")

    for term, frequency in sorted(index.document_frequencies.items()):
        print(f"{term}: {frequency}")


def print_query_results(index: BM25Index) -> None:
    eval_queries = load_eval_queries()

    for item in eval_queries:
        query = str(item["query"])
        relevant_chunk_ids = item["relevant_chunk_ids"]

        print()
        print("=" * 60)
        print("QUERY:", query)
        print("RELEVANT:", ", ".join(relevant_chunk_ids))
        print("=" * 60)

        results = index.search(query, top_k=3)

        for rank, result in enumerate(results, start=1):
            chunk_id = result.document.metadata["chunk_id"]
            section_title = result.document.metadata["section_title"]
            marker = "✓" if chunk_id in relevant_chunk_ids else " "

            print(
                f"{marker} {rank}. {chunk_id} "
                f"({section_title}) "
                f"score={result.score:.4f}"
            )


def main() -> None:
    documents = load_documents()
    chunks = chunk_documents(documents)
    index = BM25Index(chunks)

    print_index_statistics(index)
    print_query_results(index)


if __name__ == "__main__":
    main()

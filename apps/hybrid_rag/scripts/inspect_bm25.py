from langchain_core.documents import Document

from apps.hybrid_rag.app.bm25 import BM25Index


documents = [
    Document(
        page_content=(
            "AUTH-401 means the authentication token "
            "is missing or expired."
        ),
        metadata={"chunk_id": "chunk-001"},
    ),
    Document(
        page_content=(
            "AUTH-403 means the authenticated user "
            "does not have permission."
        ),
        metadata={"chunk_id": "chunk-002"},
    ),
    Document(
        page_content=(
            "Authentication tokens are used to protect "
            "API requests."
        ),
        metadata={"chunk_id": "chunk-003"},
    ),
]


index = BM25Index(documents)

print("Document count:")
print(index.document_count)

print("\nAverage document length:")
print(index.average_document_length)

print("\nDocument statistics:")

for stat in index.statistics():
    print()
    print("Chunk:", stat.document.metadata["chunk_id"])
    print("Tokens:", stat.tokens)
    print("Length:", stat.length)
    print("Term frequencies:", stat.term_frequencies)

print("\nDocument frequencies:")

for term, frequency in sorted(
    index.document_frequencies.items()
):
    print(f"{term}: {frequency}")

print("\nIDF:")

for term in [
    "auth-401",
    "auth-403",
    "authentication",
    "permission",
]:
    print(f"{term}: {index.idf(term):.4f}")
    
queries = [
    "AUTH-401",
    "AUTH-403",
    "authentication token expired",
    "permission resource",
    "quantum mechanics",
]


for query in queries:
    print()
    print("=" * 60)
    print("QUERY:", query)
    print("=" * 60)

    results = index.search(query, top_k=3)

    for rank, result in enumerate(results, start=1):
        chunk_id = result.document.metadata["chunk_id"]

        print(
            f"{rank}. {chunk_id} "
            f"score={result.score:.4f}"
        )
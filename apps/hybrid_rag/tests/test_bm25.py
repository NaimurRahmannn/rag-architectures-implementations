from langchain_core.documents import Document

from apps.hybrid_rag.app.bm25 import BM25Index


def build_documents() -> list[Document]:
    return [
        Document(
            page_content=(
                "AUTH-401 means the authentication token "
                "is missing or expired."
            ),
            metadata={
                "chunk_id": "chunk-001",
            },
        ),
        Document(
            page_content=(
                "AUTH-403 means the authenticated user "
                "does not have permission."
            ),
            metadata={
                "chunk_id": "chunk-002",
            },
        ),
        Document(
            page_content=(
                "Authentication tokens are used to protect "
                "API requests."
            ),
            metadata={
                "chunk_id": "chunk-003",
            },
        ),
    ]


def test_index_stores_document_count() -> None:
    index = BM25Index(build_documents())

    assert index.document_count == 3


def test_index_calculates_document_lengths() -> None:
    index = BM25Index(build_documents())

    statistics = index.statistics()

    assert all(stat.length > 0 for stat in statistics)


def test_index_calculates_average_document_length() -> None:
    index = BM25Index(build_documents())

    lengths = [
        stat.length
        for stat in index.statistics()
    ]

    expected = sum(lengths) / len(lengths)

    assert index.average_document_length == expected


def test_document_frequency_counts_documents_not_occurrences() -> None:
    documents = [
        Document(
            page_content="authentication authentication authentication",
        ),
        Document(
            page_content="authentication token",
        ),
        Document(
            page_content="database error",
        ),
    ]

    index = BM25Index(documents)

    assert index.document_frequencies["authentication"] == 2


def test_rare_term_has_higher_idf_than_common_term() -> None:
    index = BM25Index(build_documents())

    rare_term_idf = index.idf("auth-401")
    common_term_idf = index.idf("authentication")

    assert rare_term_idf > common_term_idf


def test_unknown_term_has_zero_idf() -> None:
    index = BM25Index(build_documents())

    assert index.idf("does-not-exist") == 0.0
    
def test_search_returns_matching_document_first() -> None:
    index = BM25Index(build_documents())

    results = index.search("AUTH-401", top_k=3)

    assert results[0].document.metadata["chunk_id"] == "chunk-001"
    assert results[0].score > 0
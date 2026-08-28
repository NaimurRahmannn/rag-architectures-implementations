from langchain_core.documents import Document

from apps.hybrid_rag.app.bm25 import BM25Index


def test_term_frequency_increases_score() -> None:
    documents = [
        Document(
            page_content="authentication",
            metadata={"chunk_id": "doc-1"},
        ),
        Document(
            page_content="authentication authentication",
            metadata={"chunk_id": "doc-2"},
        ),
        Document(
            page_content=(
                "authentication authentication authentication"
            ),
            metadata={"chunk_id": "doc-3"},
        ),
    ]

    index = BM25Index(documents)

    results = index.search(
        "authentication",
        top_k=3,
    )

    assert results[0].document.metadata["chunk_id"] == "doc-3"
    assert results[1].document.metadata["chunk_id"] == "doc-2"
    assert results[2].document.metadata["chunk_id"] == "doc-1"
    assert results[0].score > results[1].score > results[2].score


def test_rare_terms_receive_stronger_idf_signal() -> None:
    documents = [
        Document(
            page_content="authentication token",
            metadata={"chunk_id": "doc-1"},
        ),
        Document(
            page_content="authentication password",
            metadata={"chunk_id": "doc-2"},
        ),
        Document(
            page_content="authentication session",
            metadata={"chunk_id": "doc-3"},
        ),
        Document(
            page_content="authentication timeout",
            metadata={"chunk_id": "doc-4"},
        ),
    ]

    index = BM25Index(documents)

    assert index.idf("token") > index.idf("authentication")
def test_exact_identifier_is_strong_lexical_signal() -> None:
    documents = [
        Document(
            page_content=(
                "Authentication requests may return AUTH-401 "
                "when credentials are invalid."
            ),
            metadata={"chunk_id": "auth-401"},
        ),
        Document(
            page_content=(
                "Authentication requests may return AUTH-403 "
                "when permission is denied."
            ),
            metadata={"chunk_id": "auth-403"},
        ),
        Document(
            page_content=(
                "Authentication requests may return AUTH-404 "
                "when the resource does not exist."
            ),
            metadata={"chunk_id": "auth-404"},
        ),
    ]

    index = BM25Index(documents)

    results = index.search(
        "AUTH-401",
        top_k=3,
    )

    assert results[0].document.metadata["chunk_id"] == "auth-401"

def test_multiple_query_terms_accumulate() -> None:
    documents = [
        Document(
            page_content="authentication token expired",
            metadata={"chunk_id": "doc-1"},
        ),
        Document(
            page_content="authentication password",
            metadata={"chunk_id": "doc-2"},
        ),
        Document(
            page_content="database connection timeout",
            metadata={"chunk_id": "doc-3"},
        ),
    ]

    index = BM25Index(documents)

    results = index.search(
        "authentication token expired",
        top_k=3,
    )

    assert results[0].document.metadata["chunk_id"] == "doc-1"

def test_document_length_affects_score() -> None:
    documents = [
        Document(
            page_content="authentication token expired",
            metadata={"chunk_id": "short"},
        ),
        Document(
            page_content=(
                "authentication token expired "
                + "unrelated " * 100
            ),
            metadata={"chunk_id": "long"},
        ),
    ]

    index = BM25Index(documents)

    results = index.search(
        "authentication token expired",
        top_k=2,
    )

    for result in results:
        print(
            result.document.metadata["chunk_id"],
            result.score,
        )
def test_bm25_has_lexical_limitation() -> None:
    documents = [
        Document(
            page_content=(
                "User credentials have expired and "
                "must be refreshed."
            ),
            metadata={"chunk_id": "doc-1"},
        ),
        Document(
            page_content=(
                "The database connection timed out."
            ),
            metadata={"chunk_id": "doc-2"},
        ),
    ]

    index = BM25Index(documents)

    results = index.search(
        "employee cannot access account",
        top_k=2,
    )

    print(results)
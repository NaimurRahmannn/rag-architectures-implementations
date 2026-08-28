from apps.hybrid_rag.app.tokenizer import tokenize


def test_tokenizer_normalizes_case() -> None:
    assert tokenize("AUTHENTICATION Error") == [
        "authentication",
        "error",
    ]


def test_tokenizer_preserves_technical_identifiers() -> None:
    assert tokenize(
        "AUTH-401 RedisConnectionError PG-08006"
    ) == [
        "auth-401",
        "redisconnectionerror",
        "pg-08006",
    ]


def test_tokenizer_is_deterministic() -> None:
    text = "AUTH-401 authentication token"

    assert tokenize(text) == tokenize(text)
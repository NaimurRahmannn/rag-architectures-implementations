import pytest
from langchain_core.documents import Document

from apps.agentic_rag.app.schemas import (
    AgentTrace,
    AnswerResponse,
    Citation,
    PlannedStep,
    RetrievedChunk,
    ToolCall,
)
from apps.agentic_rag.scripts.generate_answer import (
    build_argument_parser,
    build_retrievers,
    build_tools,
    format_answer_response,
    parse_tool_names,
)


class KeywordEmbeddingModel:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        normalized = text.lower()
        return [
            float("password" in normalized),
            float("recovery" in normalized),
        ]


def make_chunks() -> list[Document]:
    return [
        Document(
            page_content="Password reset instructions",
            metadata={"chunk_id": "chunk-1"},
        ),
        Document(
            page_content="Recovery code instructions",
            metadata={"chunk_id": "chunk-2"},
        ),
    ]


def test_argument_parser_exposes_agentic_retrieval_options() -> None:
    args = build_argument_parser().parse_args(["How do resets work?"])

    assert args.query == "How do resets work?"
    assert args.tools == "bm25,dense,hybrid"
    assert args.top_k == 5
    assert args.candidate_k == 20


def test_parse_tool_names_normalizes_deduplicates_and_validates() -> None:
    assert parse_tool_names(" BM25,dense,bm25 ") == ("bm25", "dense")

    with pytest.raises(ValueError, match="At least one"):
        parse_tool_names(" , ")

    with pytest.raises(ValueError, match="Unknown tools"):
        parse_tool_names("bm25,magic")


def test_build_retrievers_and_tools_support_each_base_strategy() -> None:
    retrievers = build_retrievers(
        make_chunks(),
        embeddings=KeywordEmbeddingModel(),
        candidate_k=2,
    )
    tools = build_tools(("bm25", "dense", "hybrid"), retrievers)

    assert set(retrievers) == {"bm25", "dense", "hybrid"}
    assert [tool.name for tool in tools] == ["bm25", "dense", "hybrid"]
    assert all(tool.search("password", top_k=1) for tool in tools)


def test_build_tools_rejects_an_unbuilt_retriever() -> None:
    with pytest.raises(KeyError):
        build_tools(("rerank",), {})


def test_format_answer_response_includes_trace_citations_and_chunks() -> None:
    response = AnswerResponse(
        query="How do resets work?",
        answer="Follow the documented process [1].",
        citations=[
            Citation(
                index=1,
                chunk_id="chunk-1",
                document_id="guide",
                source="guide.md",
            )
        ],
        retrieved_chunks=[
            RetrievedChunk(
                rank=1,
                score=0.9,
                chunk_id="chunk-1",
                document_id="guide",
                source="guide.md",
                content="Reset instructions",
                metadata={"chunk_id": "chunk-1"},
                tool_name="dense",
                tool_rank=1,
            )
        ],
        agent_trace=AgentTrace(
            original_query="How do resets work?",
            planned_steps=[
                PlannedStep(
                    tool_name="dense",
                    query="How do resets work?",
                    top_k=5,
                )
            ],
            tool_calls=[
                ToolCall(
                    tool_name="dense",
                    query="How do resets work?",
                    top_k=5,
                    retrieved_chunk_ids=["chunk-1"],
                )
            ],
        ),
    )

    output = format_answer_response(response)

    assert "Follow the documented process [1]." in output
    assert "dense: How do resets work? (top_k=5)" in output
    assert "[1] chunk-1 (guide.md)" in output
    assert "tool=dense:1" in output

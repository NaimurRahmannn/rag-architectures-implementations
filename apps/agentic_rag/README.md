# Agentic RAG

This folder contains a learning-oriented Agentic Retrieval-Augmented Generation
pipeline. It builds on the retrievers and generation components in
`apps/hybrid_rag` and adds a planning and tool-execution layer.

The implementation can:

- decompose a compound question into retrieval subqueries;
- choose retrieval tools according to query characteristics;
- run BM25, dense, hybrid RRF, and optional cross-encoder reranking;
- merge and deduplicate evidence from multiple tool calls;
- generate an answer grounded in numbered source chunks;
- extract source citations from the generated answer; and
- return an execution trace containing the plan and tool results.

## Architecture

```text
                         User query
                             |
                             v
                    HeuristicQueryPlanner
                    - split compound query
                    - select retrieval tools
                             |
                             v
              +--------------+--------------+
              |              |              |
             BM25           Dense       Hybrid RRF
              |              |              |
              +--------------+--------------+
                             |
                    optional reranker
                             |
                             v
                 merge and deduplicate evidence
                             |
                             v
                numbered context construction
                             |
                             v
                   Gemini answer generation
                             |
                             v
              answer + citations + agent trace
```

The planner is intentionally deterministic. It demonstrates the orchestration
boundary without requiring another LLM call:

- queries containing identifiers such as `AUTH-42` prioritize BM25;
- semantic questions beginning with words such as `how` or `why` prioritize
  dense retrieval;
- other queries prioritize hybrid retrieval; and
- compound queries joined by `and` are split into separate retrieval steps.

All configured matching tools may be used. The final evidence list keeps the
first occurrence of each chunk and is limited by `top_k`.

## Folder Structure

```text
apps/agentic_rag/
|-- app/
|   |-- planning.py   # query plans, decomposition, and tool selection
|   |-- schemas.py    # request, response, citation, and trace models
|   |-- service.py    # agent orchestration and grounded generation
|   `-- tools.py      # retriever adapters and evidence merging
|-- scripts/
|   `-- generate_answer.py  # command-line entry point
`-- tests/
    |-- test_generate_answer_script.py
    |-- test_planning.py
    |-- test_service.py
    `-- test_tools.py
```

## Setup

Run commands from the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

For macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

Configure a Gemini API key in the repository-level `.env` file:

```dotenv
GOOGLE_API_KEY=your-api-key
```

`GEMINI_API_KEY` is also accepted. Optional model overrides are:

```dotenv
GEMINI_CHAT_MODEL=gemini-3.7-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
```

Do not commit `.env` or API keys.

## Generate an Answer

The CLI currently indexes the authentication guide fixture from the hybrid RAG
app, plans retrieval steps, and prints the answer, agent trace, citations, and
retrieved chunks.

```powershell
.\.venv\Scripts\python.exe -m apps.agentic_rag.scripts.generate_answer `
  "How does authentication work?"
```

Choose the tools exposed to the planner:

```powershell
.\.venv\Scripts\python.exe -m apps.agentic_rag.scripts.generate_answer `
  "How should I rotate recovery codes?" `
  --tools bm25,dense,hybrid `
  --top-k 5 `
  --candidate-k 20
```

Enable cross-encoder reranking by including `rerank`:

```powershell
.\.venv\Scripts\python.exe -m apps.agentic_rag.scripts.generate_answer `
  "How should I secure recovery codes?" `
  --tools hybrid,rerank
```

The reranker uses the cross-encoder configured by the hybrid RAG experiment
builder. The first use may download model files through `sentence-transformers`.

## Run Tests

Run the agentic tests only:

```powershell
.\.venv\Scripts\python.exe -m pytest apps\agentic_rag\tests
```

Run all architecture tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Run lint for this app:

```powershell
.\.venv\Scripts\python.exe -m ruff check apps\agentic_rag
```

The agentic suite uses deterministic fake retrievers and generators, so tests do
not call Gemini or download a cross-encoder model.

## Main Data Flow

1. `AgenticRAGService` sends the query and available tool names to the planner.
2. The planner returns ordered `RetrievalStep` objects.
3. Each `RetrievalTool` adapts a retriever result into `RetrievedEvidence`.
4. Evidence is deduplicated by `chunk_id` and converted into numbered context.
5. The answer generator receives only the original query and retrieved context.
6. Citation markers such as `[1]` are resolved to chunk and source metadata.
7. The response includes the answer, citations, retrieved chunks, and full trace.

## Current Scope

This is an educational first agentic layer, not a fully autonomous production
agent. It does not yet include:

- LLM-based planning or structured tool calling;
- retrieval-quality grading and corrective retries;
- query rewriting based on weak evidence;
- web or external knowledge tools;
- conversational memory or durable state;
- token-budget-aware context compression; or
- production observability, persistence, and access controls.

Those capabilities can be added later without changing the central boundary:
retrievers remain tools, and the orchestration service owns planning, execution,
generation, citations, and traceability.

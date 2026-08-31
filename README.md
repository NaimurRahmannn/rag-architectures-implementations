# RAG Architecture Implementations

This repository is a hands-on learning project for building Retrieval-Augmented
Generation systems one architectural layer at a time. It contains three
separate implementations that progress from a conventional dense RAG pipeline
to hybrid retrieval and then to agent-controlled retrieval orchestration.

The project favors explicit, testable components over framework-heavy magic.
Each stage keeps retrieval, evaluation, context construction, generation, and
citations behind clear interfaces so they can evolve independently.

## Learning Path

```text
Baseline RAG
Dense retrieval + grounded generation
        |
        v
Hybrid RAG
BM25 + dense + RRF + cross-encoder reranking
        |
        v
Agentic RAG
Query planning + retrieval tools + execution trace
```

| Architecture | Retrieval | Orchestration | Primary interface |
| --- | --- | --- | --- |
| [Baseline RAG](apps/baseline_rag/) | Gemini embeddings and Qdrant dense search | Fixed retrieve-then-generate flow | FastAPI |
| [Hybrid RAG](apps/hybrid_rag/) | BM25, dense, Reciprocal Rank Fusion, and cross-encoder reranking | Configurable retrieval pipeline | Python scripts |
| [Agentic RAG](apps/agentic_rag/) | BM25, dense, hybrid RRF, and optional reranking tools | Heuristic planning, query decomposition, and tool execution | Python script |

## Architecture Overview

### 1. Baseline RAG

The baseline establishes the complete RAG loop with the smallest useful set of
components:

```text
documents -> chunking -> Gemini embeddings -> Qdrant
                                                |
query -> dense retrieval -> numbered context -> Gemini -> citations
```

It includes:

- document ingestion and recursive text chunking;
- Gemini embeddings and a local persistent Qdrant collection;
- dense similarity retrieval;
- grounded prompt and answer generation;
- numbered source citations and retrieved-chunk metadata;
- FastAPI endpoints for ingestion and questions; and
- retrieval evaluation with Recall, Precision, MRR, and nDCG.

### 2. Hybrid RAG

The hybrid implementation focuses on retrieval quality and experimentation:

```text
                     +-> BM25 --------+
query -> chunks -----|                +-> RRF -> cross-encoder -> context
                     +-> dense -------+
```

It adds:

- BM25 lexical retrieval for exact terms and identifiers;
- in-memory dense retrieval using Gemini embeddings;
- parallel candidate collection;
- Reciprocal Rank Fusion (RRF);
- cross-encoder reranking with `sentence-transformers`;
- a retriever-independent evaluation engine; and
- the same grounded generation and citation boundary for every retriever.

The evaluator receives ranked chunk IDs only. It does not know whether rankings
came from BM25, dense retrieval, RRF, or reranking.

### 3. Agentic RAG

The agentic implementation adds a decision layer above the retrievers:

```text
query -> planner -> retrieval steps -> named tools -> merged evidence
                                                          |
                                                          v
                                    answer + citations + execution trace
```

It adds:

- deterministic query decomposition;
- query-aware retrieval-tool ordering;
- named adapters for BM25, dense, hybrid, and reranked retrieval;
- evidence deduplication across tool calls;
- an inspectable plan and tool-call trace; and
- grounded answer generation with source attribution.

The current planner is deliberately heuristic rather than LLM-driven. This
makes the orchestration behavior deterministic and creates a clean starting
point for corrective retrieval, evidence grading, and structured LLM tool calls.
See the [Agentic RAG README](apps/agentic_rag/README.md) for its full design and
current scope.

## Repository Structure

```text
apps/
|-- baseline_rag/
|   |-- app/          # FastAPI, chunking, dense retrieval, and generation
|   |-- evaluation/   # dataset and retrieval metrics
|   `-- tests/
|-- hybrid_rag/
|   |-- app/          # BM25, dense retrieval, fusion, reranking, and generation
|   |-- scripts/      # inspection, evaluation, and answer generation
|   `-- tests/
`-- agentic_rag/
    |-- app/          # planning, tools, schemas, and orchestration service
    |-- scripts/      # agentic answer-generation CLI
    `-- tests/
```

Shared dependencies and test configuration live in `pyproject.toml`. The apps
reuse stable lower-level components while preserving separate architectural
entry points.

## Requirements

- Python 3.11 or newer
- A Gemini API key for real embedding and generation calls
- Internet access on the first cross-encoder run so model weights can download

Qdrant runs locally through its Python client; no separate Qdrant server is
required for the baseline configuration.

## Setup

Run these commands from the repository root.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

### macOS or Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Create a repository-level `.env` file:

```dotenv
GOOGLE_API_KEY=your-api-key
```

`GEMINI_API_KEY` is accepted as an alternative. Optional settings include:

```dotenv
GEMINI_CHAT_MODEL=gemini-3.7-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
QDRANT_PATH=.data/qdrant
QDRANT_COLLECTION=baseline_rag_documents
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
DEFAULT_TOP_K=4
```

The `.env` and `.data` paths are ignored by Git. Never commit API keys.

## Run Baseline RAG

Start the FastAPI development server:

```powershell
python -m uvicorn apps.baseline_rag.app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation.

Ingest a document from PowerShell:

```powershell
$body = @{
  documents = @(
    @{
      id = "authentication-guide"
      source = "authentication_guide.md"
      content = "Users can reset passwords with a verified recovery email."
      metadata = @{ category = "security" }
    }
  )
} | ConvertTo-Json -Depth 4

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/documents `
  -ContentType "application/json" `
  -Body $body
```

Ask a question:

```powershell
$body = @{
  query = "How can a user reset a password?"
  top_k = 4
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/ask `
  -ContentType "application/json" `
  -Body $body
```

Run the baseline retrieval evaluation:

```powershell
python -m apps.baseline_rag.evaluation.run
```

## Run Hybrid RAG

Compare BM25, dense, RRF, and reranked retrieval on the evaluation fixture:

```powershell
python -m apps.hybrid_rag.scripts.evaluate_retrieval
```

Generate an answer using the default RRF plus cross-encoder pipeline:

```powershell
python -m apps.hybrid_rag.scripts.generate_answer `
  "How should recovery codes be stored?"
```

Select a specific retrieval strategy:

```powershell
python -m apps.hybrid_rag.scripts.generate_answer `
  "What does AUTH-42 describe?" `
  --retriever bm25 `
  --top-k 5
```

Valid retrievers are `bm25`, `dense`, `rrf`, and `rerank`. The default is
`rerank`. The first reranking run may download
`cross-encoder/ms-marco-MiniLM-L6-v2`.

Useful inspection scripts:

```powershell
python -m apps.hybrid_rag.scripts.inspect_chunks
python -m apps.hybrid_rag.scripts.inspect_bm25
```

## Run Agentic RAG

Generate an answer with BM25, dense, and hybrid tools available to the planner:

```powershell
python -m apps.agentic_rag.scripts.generate_answer `
  "How does authentication work?"
```

Choose tools and candidate limits explicitly:

```powershell
python -m apps.agentic_rag.scripts.generate_answer `
  "Find AUTH-42 and explain recovery codes" `
  --tools bm25,dense,hybrid,rerank `
  --top-k 5 `
  --candidate-k 20
```

The output contains the generated answer, selected retrieval steps, citations,
and final retrieved chunks.

## Tests and Quality Checks

Run every test suite:

```powershell
python -m pytest
```

Run one architecture suite:

```powershell
python -m pytest apps\baseline_rag\tests
python -m pytest apps\hybrid_rag\tests
python -m pytest apps\agentic_rag\tests
```

Lint an individual implementation:

```powershell
python -m ruff check apps\agentic_rag
```

Tests use deterministic fakes where practical, keeping retrieval and service
orchestration checks independent from external APIs and model downloads.

## Design Principles

- Retrieval results cross architecture boundaries as ranked documents or chunk
  IDs, not retriever-specific internals.
- Evaluation remains independent from BM25, dense, RRF, and reranking classes.
- Context is numbered before generation so citations can be resolved back to
  document and chunk metadata.
- External models are wrapped behind small protocols and can be replaced by
  deterministic test doubles.
- Each architecture is runnable on its own while reusing proven components from
  the previous learning phase.

## Next Learning Steps

Natural extensions to the agentic phase include evidence-quality grading,
corrective retrieval, query rewriting, structured LLM tool calling, context
compression, conversational memory, and end-to-end observability.

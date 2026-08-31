# Baseline RAG

This folder contains the foundational Retrieval-Augmented Generation
implementation in this repository. It demonstrates the complete path from
document ingestion to dense retrieval, grounded Gemini generation, and source
citations through a FastAPI service.

See the [repository overview](../../README.md) to compare this implementation
with Hybrid RAG and Agentic RAG.

## Architecture

```text
Document ingestion
      |
      v
Recursive text chunking
      |
      v
Gemini embeddings -> local Qdrant collection
                           |
User query -> dense search-+
                           |
                           v
                 numbered source context
                           |
                           v
                  Gemini generation
                           |
                           v
          answer + citations + retrieved chunks
```

The baseline keeps the orchestration fixed: retrieve the top matching chunks,
format them as numbered sources, generate an answer from those sources, and map
citation markers such as `[1]` back to chunk metadata.

## Features

- FastAPI endpoints for document ingestion and question answering
- Recursive character chunking with configurable overlap
- Stable UUID-based chunk IDs
- Gemini document and query embeddings
- Persistent local Qdrant dense-vector storage
- Grounded prompts that treat retrieved content as untrusted data
- Answer abstention when retrieval returns no evidence
- Citation extraction and source attribution
- Retrieval evaluation with Recall, Precision, MRR, and nDCG

## Folder Structure

```text
apps/baseline_rag/
|-- app/
|   |-- chunking.py    # document splitting and stable chunk IDs
|   |-- generation.py  # grounded Gemini prompt and generation
|   |-- main.py        # FastAPI application and endpoints
|   |-- retrieval.py   # Gemini embeddings and Qdrant repository
|   |-- schemas.py     # API request and response models
|   |-- service.py     # ingestion and question-answering workflow
|   `-- settings.py    # shared .env configuration
|-- evaluation/
|   |-- dataset.json   # retrieval evaluation examples
|   |-- dataset.py     # dataset loading and validation
|   |-- metrics.py     # retrieval metric calculations
|   `-- run.py         # evaluation entry point
`-- tests/
```

## Setup

Run commands from the repository root. Install the project if you have not
already done so:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Add a Gemini API key to the repository-level `.env` file:

```dotenv
GOOGLE_API_KEY=your-api-key
```

`GEMINI_API_KEY` is also accepted. The baseline supports these optional values:

```dotenv
GEMINI_CHAT_MODEL=gemini-3.7-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
QDRANT_PATH=.data/qdrant
QDRANT_COLLECTION=baseline_rag_documents
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
DEFAULT_TOP_K=4
```

The `.env` and `.data` paths are ignored by Git.

## Run the API

Start the development server:

```powershell
python -m uvicorn apps.baseline_rag.app.main:app --reload
```

Available routes:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check service availability |
| `POST` | `/documents` | Chunk, embed, and index documents |
| `POST` | `/ask` | Retrieve context and generate an answer |

Open `http://127.0.0.1:8000/docs` for interactive request schemas and testing.

### Ingest Documents

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

### Ask a Question

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

The response contains `answer`, `citations`, and `retrieved_chunks`. Each
citation points to the source, document ID, and chunk ID used by the answer.

## Evaluation

Run the built-in retrieval dataset against an in-memory Qdrant client:

```powershell
python -m apps.baseline_rag.evaluation.run
```

The report prints per-query and average Recall@K, Precision@K, reciprocal rank,
and nDCG@K. The baseline evaluation measures ranked document IDs, while the
hybrid evaluator measures ranked chunk IDs.

## Tests

```powershell
python -m pytest apps\baseline_rag\tests
```

The service tests use fake repositories and generators, so they do not require
Gemini or persistent Qdrant data.

## Current Scope

This baseline intentionally does not include lexical retrieval, rank fusion,
cross-encoder reranking, query planning, or retrieval correction. It also has no
relevance threshold, document update/delete API, authentication, or production
observability. Those concerns are outside this first architecture and provide
the motivation for the later implementations.

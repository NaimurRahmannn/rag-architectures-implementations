# Hybrid RAG

This folder explores retrieval quality beyond a dense-only baseline. It combines
lexical and semantic retrieval, fuses their rankings with Reciprocal Rank Fusion
(RRF), optionally reranks candidates with a cross-encoder, and sends the final
ranked chunks through a shared grounded-generation service.

See the [repository overview](../../README.md) to compare this implementation
with Baseline RAG and Agentic RAG.

## Architecture

```text
                         user query
                             |
                +------------+------------+
                |                         |
                v                         v
          BM25 retrieval            dense retrieval
                |                         |
                +------------+------------+
                             |
                             v
                  Reciprocal Rank Fusion
                             |
                             v
                   cross-encoder reranker
                             |
                             v
                 numbered context + Gemini
                             |
                             v
              answer + citations + ranked chunks
```

Every retrieval strategy implements the same `Retriever.search()` boundary and
returns ranked `ScoredDocument` objects. Generation therefore works with BM25,
dense, RRF, or reranked results without knowing how the ranking was produced.

## Retrieval Stages

### BM25

BM25 provides lexical matching for exact words, identifiers, and uncommon
terms. The local tokenizer preserves useful hyphenated and underscored tokens.

### Dense Retrieval

The in-memory dense index embeds every chunk and compares the query using cosine
similarity. Production embedding calls use Gemini, while tests use deterministic
embedding doubles.

### Reciprocal Rank Fusion

BM25 and dense retrieval run in parallel. RRF combines their ranks without
trying to normalize incomparable lexical and cosine scores:

```text
score(document) = sum(1 / (rank_constant + rank))
```

The default rank constant is `60`. A chunk appearing in both rankings receives
a contribution from each retriever.

### Cross-Encoder Reranking

The optional reranker scores each `(query, chunk)` pair jointly and reorders the
RRF candidates. The default model is:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

This stage is more expensive than first-stage retrieval, so it operates on a
limited candidate set rather than the entire corpus.

## Folder Structure

```text
apps/hybrid_rag/
|-- app/
|   |-- bm25.py         # BM25 index and scoring
|   |-- evaluation.py   # retriever-independent metrics engine
|   |-- fusion.py       # RRF calculation and retriever
|   |-- generation.py   # context formatting and Gemini generation
|   |-- reranking.py    # generic and cross-encoder reranking
|   |-- retrieval.py    # retriever protocols, dense index, parallel retrieval
|   |-- schemas.py      # answer, citation, and chunk models
|   |-- service.py      # retrieval-to-generation workflow
|   `-- tokenizer.py    # lexical tokenization
|-- scripts/
|   |-- evaluate_retrieval.py  # compare retrieval baselines
|   |-- generate_answer.py     # generate with a selected retriever
|   |-- inspect_bm25.py        # inspect lexical rankings
|   `-- inspect_chunks.py      # inspect fixture chunking
`-- tests/
    `-- fixtures/              # authentication guide and evaluation labels
```

## Setup

Run commands from the repository root. Install the shared project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Add a Gemini API key to the repository-level `.env` file:

```dotenv
GOOGLE_API_KEY=your-api-key
```

`GEMINI_API_KEY` is accepted as an alternative. Embedding and chat models can be
overridden with `GEMINI_EMBEDDING_MODEL` and `GEMINI_CHAT_MODEL`.

## Inspect the Corpus

The scripts use `tests/fixtures/authentication_guide.md` as a small reproducible
learning corpus.

Inspect generated chunks:

```powershell
python -m apps.hybrid_rag.scripts.inspect_chunks
```

Inspect BM25 behavior against the labeled queries:

```powershell
python -m apps.hybrid_rag.scripts.inspect_bm25
```

## Compare Retrievers

Run the retrieval experiment:

```powershell
python -m apps.hybrid_rag.scripts.evaluate_retrieval
```

The script builds BM25, dense, RRF, and cross-encoder pipelines and reports
Recall@1, Recall@3, Recall@5, and Mean Reciprocal Rank. It also prints each
retriever's ranking for every query.

The evaluation engine in `app/evaluation.py` receives only:

```python
retrieve: Callable[[str], Sequence[str]]
```

The experiment-layer adapter converts any retriever output into ranked chunk
IDs. This keeps evaluation independent from BM25, dense retrieval, RRF, and
reranking implementations.

## Generate an Answer

The default generation pipeline retrieves RRF candidates and reranks them with
the cross-encoder:

```powershell
python -m apps.hybrid_rag.scripts.generate_answer `
  "How should recovery codes be stored?"
```

Choose a retrieval strategy explicitly:

```powershell
python -m apps.hybrid_rag.scripts.generate_answer `
  "What does AUTH-42 describe?" `
  --retriever bm25 `
  --top-k 5 `
  --candidate-k 20
```

Supported values for `--retriever` are:

| Value | Pipeline |
| --- | --- |
| `bm25` | Lexical retrieval only |
| `dense` | Dense cosine retrieval only |
| `rrf` | Parallel BM25 and dense retrieval followed by RRF |
| `rerank` | RRF candidates followed by cross-encoder reranking |

The first `rerank` run may download model weights through
`sentence-transformers`.

## Tests

Run the hybrid suite:

```powershell
python -m pytest apps\hybrid_rag\tests
```

The tests cover tokenizer behavior, BM25 scoring, dense similarity, parallel
retrieval, RRF, cross-encoder adapters, retriever-independent evaluation,
generation, citations, and CLI construction. Model-dependent components accept
test doubles, so the suite does not require network calls or model downloads.

## Current Scope

This implementation uses an in-memory dense index and a fixture corpus for
experiments. It does not provide an ingestion API, persistent hybrid index,
distributed retrieval, learned fusion, relevance thresholds, or production
monitoring. Retrieval strategy selection is configured by the caller; the
Agentic RAG implementation adds a planning and tool-execution layer above these
retrievers.

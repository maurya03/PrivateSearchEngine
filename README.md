# PrivateSearch — Local Information Retrieval Engine

PrivateSearch is a **local-first search engine** built to make information-retrieval fundamentals explicit. Documents, indexes and queries stay local; there is no hosted search dependency.

## Why this project exists

This is the retrieval foundation for the portfolio's later RAG and autonomous-agent projects. It demonstrates how a search system works before adding embeddings or an LLM: ingestion, tokenization, inverted indexes, BM25 ranking, phrase matching, filtering and evaluation.

## Architecture

```text
Documents
   │
   ▼
Parser / normalizer ──► SHA-256 content hash
   │
   ▼
Tokenization + positions
   │
   ▼
SQLite inverted index ──► postings(term, doc, field, freq, positions)
   │
   ▼
Candidate retrieval
   │
   ▼
BM25 + field boosts + phrase boosts
   │
   ▼
Ranked results + snippets
   │
   ├── CLI
   └── FastAPI

Evaluation: Precision@K / Recall@K / MRR
Benchmark: query throughput + latency
```

## Features

- SQLite-backed inverted index with explicit postings and token positions.
- BM25 ranking with configurable `k1` and `b`.
- Separate title/body fields and title boosting.
- Phrase queries such as `"write ahead logging"`.
- Negative terms such as `database -recovery`.
- HTML, Markdown and text ingestion.
- Incremental/idempotent indexing using SHA-256 content hashes.
- Deterministic tie-breaking for reproducible rankings.
- Context snippets around query terms.
- HTTP API with FastAPI.
- Offline evaluation with Precision@K, Recall@K and MRR.
- Simple benchmark reporting QPS and average latency.
- Unit/integration tests and GitHub Actions CI.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[test]

privatesearch index ./docs
privatesearch search "write ahead logging" -k 5
privatesearch benchmark "database recovery" -n 1000
privatesearch evaluate eval.json

uvicorn privatesearch.api:app --reload
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Search model

For each query term, PrivateSearch computes BM25 using document frequency, term frequency and document length normalization. Title occurrences receive a higher field weight because titles are stronger relevance signals. Phrase matches receive an additional configurable boost.

The engine intentionally uses an inverted index rather than scanning every document at query time. SQLite provides durable local storage while keeping the index schema inspectable.

## Evaluation

`eval.json` contains a tiny relevance judgment set. Real retrieval systems need a larger, representative dataset; the included metrics are a framework for measuring changes rather than evidence of production-scale search quality.

```text
Precision@K — how many returned results are relevant
Recall@K    — how much of the known relevant set was retrieved
MRR         — how early the first relevant result appears
```

## Deliberate scope

This is a retrieval-engineering project, not Elasticsearch/OpenSearch. It intentionally does not implement distributed indexing, stemming/lemmatization, typo tolerance, vector search, learning-to-rank, sharding or replicated storage. Those are natural extensions after the fundamentals are understood.

The next project, **EngineeringRAG**, will add hybrid lexical + semantic retrieval, chunking, embeddings, reranking, citation grounding and retrieval evaluation on top of these principles.

## What this demonstrates

- Information retrieval fundamentals
- Inverted indexes and positional postings
- BM25 ranking
- Search query processing
- Incremental indexing and content hashing
- Retrieval evaluation
- Latency/throughput measurement
- Python engineering, SQLite and FastAPI
- Testable architecture and CI

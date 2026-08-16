# ShopEasy — Customer Support RAG Tutorial

A hands-on walkthrough of Retrieval-Augmented Generation (RAG) for a real-world
use case: a **customer-support assistant for ShopEasy**, a multi-category
e-commerce platform. When a support ticket arrives, the assistant searches the
company's internal knowledge base to diagnose the issue and suggest the next
step.

The project progresses across three notebooks — from a basic pipeline, to
metadata-filtered retrieval, to hybrid (semantic + keyword) search — and ships
with a Streamlit app that lets you browse the knowledge base and chat with the
assistant using any of the three strategies.

---

## The knowledge base

All notebooks and the app share one dataset: [`shopeasy_knowledge_base.json`](shopeasy_knowledge_base.json) —
**51 internal support documents**. Each entry has a `content` string and a
structured `metadata` object.

**Document types (`doc_type`)**

| Type          | Count | Description |
|---------------|-------|-------------|
| `past_ticket` | 18    | Historical resolved tickets with investigation notes |
| `runbook`     | 10    | Step-by-step troubleshooting guides |
| `product_doc` | 10    | Feature reference, policies, configuration guides |
| `faq`         | 7     | Common questions and answers |
| `bug_report`  | 6     | Active and resolved bugs with workarounds |

**Product areas (`product_area`):** `orders` (14), `payments` (13), `returns` (9),
`account` (9), `shipping` (6).

**Other metadata fields:** `priority` (P0–P3), `platform` (`web`/`mobile`/`all`),
`customer_tier` (`regular`/`plus`/`all`), `status` (`active`/`resolved`), and a
human-readable `title`.

---

## The notebooks

Run them **in order** — Notebook 1 indexes the data into Pinecone, and Notebooks
2 and 3 reuse that same index (namespace `shopeasy-basic-rag`) without
re-indexing.

### 1. [`1_rag_pipeline.ipynb`](1_rag_pipeline.ipynb) — Basic RAG pipeline

Builds the end-to-end pipeline from scratch:

- **Load** the knowledge base and wrap each entry in a LangChain `Document`.
- **Chunk** with `RecursiveCharacterTextSplitter` (500 chars, 100 overlap).
- **Index** into Pinecone using OpenAI `text-embedding-3-small` embeddings at 512 dims.
- **Retrieve** the top-k chunks for a customer ticket via semantic similarity search.
- **Generate** an answer with `gpt-4.1-mini`, including citations from doc metadata.

Teaching point: plain semantic search works for natural language but casts a wide
net — a shipping question also pulls in returns and refund docs because the
language overlaps.

### 2. [`2_metadata_filtering.ipynb`](2_metadata_filtering.ipynb) — Metadata filtering

Connects to the **same** index and makes retrieval precise by filtering on metadata
*before* the semantic search runs:

- **Manual filters** — scope to an exact slice, e.g. `doc_type=bug_report` +
  `product_area=payments` to surface only known checkout bugs. Uses Pinecone
  operators (`$eq`, `$in`, `$ne`, `$gt`, …).
- **LLM-classified filters** — an LLM with **structured output** (Pydantic) reads
  the raw ticket, infers `product_area` (and optionally `doc_type` /
  `customer_tier`), and the filter is assembled automatically.

Teaching point: metadata adds *structure* on top of semantic search, removing the
cross-area noise from Notebook 1.

### 3. [`3_hybrid_rag.ipynb`](3_hybrid_rag.ipynb) — Hybrid RAG (semantic + keyword)

Combines two retrievers for the best of both worlds:

- **Vector retriever** — reuses the Pinecone vectors (semantic similarity).
- **BM25 retriever** — in-memory keyword/lexical search built from the source JSON.
- **Fusion** — LangChain's `EnsembleRetriever` merges both ranked lists with
  weighted **Reciprocal Rank Fusion (RRF)**.

Three worked examples show when each signal wins:

1. An **exact identifier** (`ORD-97654`) — keyword search wins.
2. **Plain natural language** with vocabulary that doesn't match the KB — semantic search wins.
3. A **mixed query** (a "stuck refund" concept + order id `ORD-91045`) — hybrid wins. 🎯

The notebook closes with a survey of other fusion/reranking strategies
(cross-encoder rerankers like Cohere Rerank, BGE).

> Diagrams for each step live in [`images/`](images/) and are referenced inline in the notebooks.

---

## The Company KB Viewer app

[`company_kb_viewer.py`](company_kb_viewer.py) is a Streamlit app with two tabs:

- **Knowledge Base** — a read-only, master/detail browser for support agents.
  Filter and full-text search the 51 docs on the left; read the full article on
  the right.
- **Support Assistant** — a RAG chat agent that answers customer questions and
  lets you choose the retrieval strategy live:
  - 🔍 **Basic semantic** — plain vector similarity (top 5).
  - 🏷️ **Metadata filtering** — an LLM infers product area / doc type / tier and scopes the search.
  - 🔀 **Hybrid (vector + BM25)** — fuses semantic and keyword search via RRF.

All three strategies reuse the **existing Pinecone index** populated by the
notebooks (namespace `shopeasy-basic-rag`) — nothing is re-indexed.

---

## Setup

### Prerequisites

- **Python 3.12+**
- An **OpenAI** API key (embeddings + chat) and a **Pinecone** account/index.
- [`uv`](https://docs.astral.sh/uv/) (recommended) for dependency management.

### 1. Install dependencies

Using `uv` (reads [`pyproject.toml`](pyproject.toml) / `uv.lock`):

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=your-index-name
```

> Create a Pinecone index configured for **512-dimensional** vectors with
> **cosine** similarity (matching `text-embedding-3-small` at 512 dims as used by
> the notebooks). `.env` is loaded automatically by both the notebooks and the app.

---

## Running

### Notebooks

Run them in order — **Notebook 1 first**, since it creates and populates the
Pinecone index that Notebooks 2 and 3 (and the app) depend on.

```bash
uv run jupyter lab        # or: jupyter notebook
```

Then open `1_rag_pipeline.ipynb` → `2_metadata_filtering.ipynb` →
`3_hybrid_rag.ipynb`. (If using `uv`, select the project's `.venv` kernel.)

### KB Viewer app

Once Notebook 1 has populated the index:

```bash
uv run streamlit run company_kb_viewer.py
```

Streamlit opens the app in your browser. The **Knowledge Base** tab works with no
API keys; the **Support Assistant** tab needs `OPENAI_API_KEY`,
`PINECONE_API_KEY`, and `PINECONE_INDEX_NAME`.

---

## How it all fits together

```
shopeasy_knowledge_base.json
        │
        ▼
1_rag_pipeline.ipynb  ──(indexes 512-dim vectors)──►  Pinecone
        │                                          namespace: shopeasy-basic-rag
        │                                                 ▲   ▲   ▲
        ▼                                                 │   │   │
2_metadata_filtering.ipynb ───────────reuses index────────┘   │   │
3_hybrid_rag.ipynb ───────────────────reuses index────────────┘   │
company_kb_viewer.py (Streamlit) ──────reuses index────────────────┘
```

Each notebook builds on the same dataset and scenario, so the improvements —
less noise, more precision — are easy to see and measure.

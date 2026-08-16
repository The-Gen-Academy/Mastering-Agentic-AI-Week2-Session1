"""ShopEasy Knowledge Base viewer + Support Assistant.

Two tabs:
  * Knowledge Base — read-only internal reference for support agents. Two-pane
    master/detail layout: filter + search on the left, full article on the right.
  * Support Assistant — a RAG chat agent that answers customer questions, letting
    the agent pick which retrieval strategy to use (basic semantic, metadata
    filtering, or hybrid). All strategies reuse the existing Pinecone index
    (namespace ``shopeasy-basic-rag``) populated by the notebooks — nothing is
    re-indexed here.

Run with:
    streamlit run company_kb_viewer.py

The Support Assistant needs OPENAI_API_KEY, PINECONE_API_KEY, and
PINECONE_INDEX_NAME in the environment (a local .env is loaded automatically).
"""

import json
import os
from pathlib import Path

import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional; env vars may already be set
    pass

KB_PATH = Path(__file__).parent / "shopeasy_knowledge_base.json"

# ---- RAG strategy definitions (Support Assistant tab) -------------------
# Each strategy reuses the EXISTING Pinecone vectors in the "shopeasy-basic-rag"
# namespace — we only change how we retrieve, never re-index.
NAMESPACE = "shopeasy-basic-rag"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 512  # must match how the notebooks indexed the vectors
CHAT_MODEL = "gpt-4.1-mini"

RAG_STRATEGIES = {
    "basic": {
        "label": "🔍 Basic semantic",
        "help": "Plain vector similarity search (top 5). Great for natural language; "
        "casts a wide net and can mix in adjacent product areas.",
    },
    "metadata": {
        "label": "🏷️ Metadata filtering",
        "help": "An LLM reads the ticket, infers product area / doc type / tier, and "
        "scopes the semantic search to that slice before retrieving.",
    },
    "hybrid": {
        "label": "🔀 Hybrid (vector + BM25)",
        "help": "Fuses semantic search with keyword (BM25) search via Reciprocal Rank "
        "Fusion. Best when tickets mix prose with exact IDs (order #, SKU, bug code).",
    },
}

RAG_PROMPT = """You are a customer-support assistant for ShopEasy, an e-commerce platform.
Use the following pieces of retrieved internal knowledge-base context to help resolve the customer's issue.
If the context doesn't contain the answer, say you don't have that information rather than guessing.
Be concise and practical: state the likely cause and the next step the agent should take.

Context:
{context}

Customer issue: {question}

Support guidance:"""

# Metadata fields we expose as sidebar filters, in display order.
FILTER_FIELDS = [
    ("doc_type", "Doc type"),
    ("product_area", "Product area"),
    ("priority", "Priority"),
    ("platform", "Platform"),
    ("customer_tier", "Customer tier"),
    ("status", "Status"),
]

# Muted priority colors — read as status, not decoration.
PRIORITY_COLOR = {
    "P0": "#c0392b",  # deep red
    "P1": "#e74c3c",  # red
    "P2": "#e67e22",  # orange
    "P3": "#7f8c8d",  # gray
}
PRIORITY_DOT = {"P0": "🔴", "P1": "🔴", "P2": "🟠", "P3": "⚪"}

DOC_TYPE_ICON = {
    "runbook": "📘",
    "past_ticket": "🎫",
    "product_doc": "📄",
    "bug_report": "🐞",
    "faq": "❓",
}

# Sort key so P0 floats to the top of the list.
PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}

ACCENT = "#0aa3a3"  # ShopEasy teal


@st.cache_data
def load_docs():
    """Load the KB and attach a stable id to each doc for selection."""
    with open(KB_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    docs = []
    for i, item in enumerate(raw):
        meta = item.get("metadata", {})
        docs.append(
            {
                "id": i,
                "content": item.get("content", ""),
                "title": meta.get("title", "(untitled)"),
                "doc_type": meta.get("doc_type", ""),
                "product_area": meta.get("product_area", ""),
                "priority": meta.get("priority", ""),
                "platform": meta.get("platform", ""),
                "customer_tier": meta.get("customer_tier", ""),
                "status": meta.get("status", ""),
            }
        )
    return docs


def unique_values(docs, field):
    """Sorted distinct values for a metadata field."""
    return sorted({d[field] for d in docs if d[field]})


def snippet(content, title, limit=75):
    """First meaningful body line that actually adds info beyond the title.

    Many docs open with a line that just restates the title (e.g.
    "Bug CD-8567 — <title>"), which is pure noise in the list. Skip any line
    that overlaps the title in either direction.
    """
    t = title.strip().lower()
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if low == t or t in low or low in t:
            continue
        return line[:limit] + ("…" if len(line) > limit else "")
    return ""


def inject_css():
    st.markdown(
        f"""
        <style>
            .block-container {{ padding-top: 2rem; }}

            .kb-title {{ font-size: 1.9rem; font-weight: 700; margin-bottom: 0; }}
            .kb-sub {{ color: #9aa0a6; font-size: 0.9rem; margin-top: 0.2rem; }}

            /* ---- Search results: Google-style left-aligned list ----
               Streamlit 1.58 tags each widget container with `st-key-<key>`.
               Our result buttons use keys like `card_0`, so target that prefix
               instead of relying on column position. */
            div[class*="st-key-card_"] .stButton > button {{
                width: 100%;
                background: transparent;
                border: none;
                border-radius: 0;
                border-bottom: 1px solid rgba(255,255,255,0.07);
                border-left: 3px solid transparent;
                box-shadow: none;
                padding: 0.85rem 0.4rem 0.95rem 0.9rem;
                margin: 0;
                /* Defeat Streamlit's default centered flex layout. Streamlit
                   sets the button to a centering flex container with enough
                   weight that plain overrides lose (our color rules win, the
                   layout ones don't), so these need !important. align-items is
                   the one that actually pins a short, shrink-wrapped title to
                   the left edge — text-align alone can't. */
                display: flex !important;
                flex-direction: column !important;
                align-items: flex-start !important;
                justify-content: flex-start !important;
                text-align: left !important;
                transition: background 0.1s ease-in-out, border-color 0.1s ease-in-out;
            }}
            div[class*="st-key-card_"] .stButton > button:hover {{
                background: rgba(255,255,255,0.03);
                color: inherit;
            }}
            div[class*="st-key-card_"] .stButton > button:focus:not(:active) {{
                border-left: 3px solid {ACCENT};
                background: rgba(10,163,163,0.06);
                box-shadow: none;
                color: inherit;
            }}
            /* Force the label/markdown container to fill width and align left */
            div[class*="st-key-card_"] .stButton > button > div,
            div[class*="st-key-card_"] .stButton [data-testid="stMarkdownContainer"] {{
                width: 100% !important;
                text-align: left !important;
            }}
            /* The label lines render as separate <p> tags inside the button */
            div[class*="st-key-card_"] .stButton p {{
                margin: 0 !important;
                text-align: left !important;
            }}
            /* Line 1 — meta breadcrumb */
            div[class*="st-key-card_"] .stButton p:first-child {{
                font-size: 0.72rem;
                color: #9aa0a6;
                letter-spacing: 0.02em;
                margin-bottom: 0.18rem !important;
            }}
            /* Line 2 — title (the "link") */
            div[class*="st-key-card_"] .stButton p:nth-child(2) {{
                font-size: 1.05rem;
                line-height: 1.3;
                color: #6cb6ff;
                font-weight: 500;
            }}
            div[class*="st-key-card_"] .stButton > button:hover p:nth-child(2) {{
                text-decoration: underline;
            }}
            /* Line 3 — snippet */
            div[class*="st-key-card_"] .stButton p:nth-child(3) {{
                font-size: 0.85rem;
                line-height: 1.4;
                color: #bdc1c6;
                margin-top: 0.2rem !important;
            }}

            .chip {{
                display: inline-block;
                font-size: 0.7rem;
                font-weight: 600;
                padding: 0.1rem 0.5rem;
                border-radius: 999px;
                margin-right: 0.3rem;
                background: #f1f5f9;
                color: #475569;
            }}
            .detail-card {{
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
                padding: 1.5rem 1.8rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }}
            .detail-title {{ font-size: 1.4rem; font-weight: 700; margin: 0.6rem 0 1rem; color: #111827; }}
            .detail-body {{ white-space: pre-wrap; line-height: 1.6; color: #1f2937; font-size: 0.95rem; }}
            .empty-state {{
                text-align: center;
                color: #9ca3af;
                padding: 4rem 1rem;
                border: 1.5px dashed #e5e7eb;
                border-radius: 12px;
            }}
            .count-pill {{ color: #6b7280; font-size: 0.85rem; margin-bottom: 0.6rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def priority_badge(priority):
    color = PRIORITY_COLOR.get(priority, "#7f8c8d")
    return (
        f'<span class="chip" style="background:{color}1a;color:{color};'
        f'border:1px solid {color}55;">{priority}</span>'
    )


# ============================================================================
# Support Assistant — RAG backend
#
# These helpers build the retrieval strategies on top of the EXISTING Pinecone
# index. Heavy imports live inside the functions so the Knowledge Base tab keeps
# working even if the RAG dependencies / API keys aren't available. Resources are
# cached with @st.cache_resource so we connect to Pinecone and build BM25 once.
# ============================================================================

REQUIRED_ENV = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]


def missing_env():
    """Return the list of required env vars that aren't set."""
    return [v for v in REQUIRED_ENV if not os.environ.get(v)]


@st.cache_resource
def get_embeddings():
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=EMBED_MODEL, dimensions=EMBED_DIMS)


@st.cache_resource
def get_vector_store():
    """Connect to the existing Pinecone index + namespace (no indexing)."""
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
    return PineconeVectorStore(
        index=index, embedding=get_embeddings(), namespace=NAMESPACE
    )


@st.cache_resource
def get_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=CHAT_MODEL)


@st.cache_resource
def get_bm25_retriever():
    """Build an in-memory BM25 retriever from the source KB, chunked exactly as
    the notebooks chunked it so BM25 and the vector index search the same units."""
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    with open(KB_PATH, encoding="utf-8") as f:
        kb = json.load(f)
    raw_docs = [
        Document(page_content=e["content"], metadata=e["metadata"]) for e in kb
    ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, add_start_index=True
    )
    retriever = BM25Retriever.from_documents(splitter.split_documents(raw_docs))
    retriever.k = 3
    return retriever


@st.cache_resource
def get_ensemble_retriever():
    """Vector (k=3) + BM25 (k=3) fused with weighted Reciprocal Rank Fusion."""
    from langchain_classic.retrievers import EnsembleRetriever

    vector_retriever = get_vector_store().as_retriever(search_kwargs={"k": 3})
    return EnsembleRetriever(
        retrievers=[vector_retriever, get_bm25_retriever()],
        weights=[0.5, 0.5],
    )


@st.cache_resource
def get_ticket_classifier():
    """LLM with structured output that infers a metadata filter from a ticket."""
    from typing import Optional

    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field

    class TicketFilter(BaseModel):
        """Structured metadata filter inferred from a customer support ticket."""

        product_area: str = Field(
            description="One of: payments, returns, shipping, orders, account"
        )
        doc_type: Optional[str] = Field(
            default=None,
            description="Optionally one of: runbook, past_ticket, product_doc, "
            "bug_report, faq. Set only if the ticket clearly targets a doc type "
            "(e.g. asking about a known bug -> bug_report).",
        )
        customer_tier: Optional[str] = Field(
            default=None,
            description="Optionally one of: plus, regular. Set only if the "
            "customer's tier is explicitly mentioned.",
        )

    structured_llm = ChatOpenAI(model=CHAT_MODEL, temperature=0).with_structured_output(
        TicketFilter
    )
    prompt = PromptTemplate.from_template(
        "You classify ShopEasy customer support tickets so we can filter the "
        "knowledge base.\nRead the ticket and return the metadata that best "
        "scopes the search.\n\nTicket: {question}\n"
    )
    return structured_llm, prompt


def build_pinecone_filter(ticket_filter):
    """Turn the classified fields into a Pinecone metadata filter, skipping empty
    ones. customer_tier matches both the tier AND "all" so tier-specific queries
    don't drop the general docs that also apply."""
    pinecone_filter = {}
    for field, value in ticket_filter.model_dump().items():
        if not value:
            continue
        if field == "customer_tier":
            pinecone_filter[field] = {"$in": [value, "all"]}
        else:
            pinecone_filter[field] = {"$eq": value}
    return pinecone_filter


def retrieve(question, strategy):
    """Run the chosen strategy. Returns (docs, debug_info)."""
    if strategy == "basic":
        retriever = get_vector_store().as_retriever(
            search_kwargs={"k": 5}, search_type="similarity"
        )
        return retriever.invoke(question), None

    if strategy == "metadata":
        structured_llm, prompt = get_ticket_classifier()
        ticket_filter = structured_llm.invoke(prompt.invoke({"question": question}))
        pinecone_filter = build_pinecone_filter(ticket_filter)
        retriever = get_vector_store().as_retriever(
            search_kwargs={"k": 5, "filter": pinecone_filter},
            search_type="similarity",
        )
        debug = {"classified": ticket_filter.model_dump(), "filter": pinecone_filter}
        return retriever.invoke(question), debug

    if strategy == "hybrid":
        return get_ensemble_retriever().invoke(question), None

    raise ValueError(f"Unknown strategy: {strategy}")


def answer_question(question, strategy):
    """Retrieve with the chosen strategy and generate a grounded answer.

    Returns (answer_text, sources, debug_info) where sources is a deduped list
    of (title, doc_type, product_area)."""
    from langchain_core.prompts import PromptTemplate

    docs, debug = retrieve(question, strategy)
    context = "\n\n".join(d.page_content for d in docs)
    prompt = PromptTemplate.from_template(RAG_PROMPT).invoke(
        {"question": question, "context": context}
    )
    response = get_llm().invoke(prompt)

    seen, sources = set(), []
    for d in docs:
        m = d.metadata
        key = m.get("title", "")
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            (key, m.get("doc_type", ""), m.get("product_area", ""))
        )
    return response.content, sources, debug


def render_support_assistant():
    """The Support Assistant chat tab."""
    st.markdown(
        '<div class="kb-sub" style="margin-bottom:0.8rem;">Ask a customer question and the '
        "assistant answers from the ShopEasy knowledge base. Pick the retrieval strategy "
        "below — each one queries the same Pinecone index a different way.</div>",
        unsafe_allow_html=True,
    )

    missing = missing_env()
    if missing:
        st.warning(
            "The Support Assistant needs these environment variables set "
            f"(in your `.env` or shell): **{', '.join(missing)}**.\n\n"
            "The Knowledge Base tab still works without them."
        )
        return

    top = st.columns([3, 1])
    with top[0]:
        strategy = st.radio(
            "Retrieval strategy",
            options=list(RAG_STRATEGIES.keys()),
            format_func=lambda k: RAG_STRATEGIES[k]["label"],
            horizontal=True,
            key="rag_strategy",
        )
    with top[1]:
        st.write("")  # spacer to align the button with the radio
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()
    st.caption(RAG_STRATEGIES[strategy]["help"])

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Replay history.
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("debug"):
                with st.expander("🏷️ Inferred metadata filter"):
                    st.json(msg["debug"])
            if msg.get("sources"):
                with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                    for title, doc_type, area in msg["sources"]:
                        icon = DOC_TYPE_ICON.get(doc_type, "📄")
                        st.markdown(f"- {icon} **{title}** · `{doc_type}` · `{area}`")

    if prompt := st.chat_input("Describe the customer's issue…"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            label = RAG_STRATEGIES[strategy]["label"]
            with st.spinner(f"Retrieving with {label}…"):
                try:
                    answer, sources, debug = answer_question(prompt, strategy)
                except Exception as e:  # surface config/runtime errors in-chat
                    answer, sources, debug = (
                        f"⚠️ Something went wrong while answering: `{e}`",
                        [],
                        None,
                    )
            st.markdown(answer)
            if debug:
                with st.expander("🏷️ Inferred metadata filter"):
                    st.json(debug)
            if sources:
                with st.expander(f"📚 Sources ({len(sources)})"):
                    for title, doc_type, area in sources:
                        icon = DOC_TYPE_ICON.get(doc_type, "📄")
                        st.markdown(f"- {icon} **{title}** · `{doc_type}` · `{area}`")

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "debug": debug,
            }
        )


def render_kb_browser(docs):
    """The Knowledge Base browser tab (filter + search + master/detail)."""
    if "selected_id" not in st.session_state:
        st.session_state.selected_id = None

    query = st.text_input(
        "Search", placeholder="🔍 Search titles & content...", label_visibility="collapsed"
    )

    # ---- Sidebar filters ------------------------------------------------
    with st.sidebar:
        st.markdown("### Filters")
        if st.button("Reset filters", use_container_width=True):
            for field, _ in FILTER_FIELDS:
                st.session_state.pop(f"filter_{field}", None)
            st.rerun()

        selected = {}
        for field, label in FILTER_FIELDS:
            options = unique_values(docs, field)
            chosen = st.multiselect(
                label,
                options,
                default=[],
                placeholder="All",
                key=f"filter_{field}",
            )
            # Empty box means no restriction on this field.
            selected[field] = set(chosen) if chosen else set(options)

    # ---- Apply filters + search ----------------------------------------
    q = query.strip().lower()
    results = []
    for d in docs:
        if any(d[field] not in selected[field] for field, _ in FILTER_FIELDS):
            continue
        if q and q not in d["title"].lower() and q not in d["content"].lower():
            continue
        results.append(d)

    results.sort(key=lambda d: (PRIORITY_ORDER.get(d["priority"], 9), d["title"]))

    # ---- Two-pane master/detail ----------------------------------------
    list_col, detail_col = st.columns([1, 1.3], gap="large")

    with list_col:
        st.markdown(
            f'<div class="count-pill">Showing <b>{len(results)}</b> of {len(docs)}</div>',
            unsafe_allow_html=True,
        )
        if not results:
            st.info("No articles match your filters or search.")
        for d in results:
            icon = DOC_TYPE_ICON.get(d["doc_type"], "📄")
            dot = PRIORITY_DOT.get(d["priority"], "⚪")
            # Google-style breadcrumb meta line: priority · type · area
            crumbs = [f"{dot} {d['priority']}", f"{icon} {d['doc_type'].replace('_', ' ')}"]
            if d["product_area"]:
                crumbs.append(d["product_area"])
            label = "  ·  ".join(crumbs) + f"\n\n{d['title']}"
            sub = snippet(d["content"], d["title"])
            if sub:
                label += f"\n\n{sub}"
            if st.button(label, key=f"card_{d['id']}", use_container_width=True):
                st.session_state.selected_id = d["id"]

    with detail_col:
        doc = next((d for d in docs if d["id"] == st.session_state.selected_id), None)
        if doc is None:
            st.markdown(
                '<div class="empty-state">'
                '<div style="font-size:2.5rem;">📄</div>'
                "<div style='margin-top:0.5rem;'>Select a document from the list to read it here</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            badges = priority_badge(doc["priority"])
            for field in ["status", "doc_type", "product_area", "platform", "customer_tier"]:
                val = doc[field]
                if val:
                    show = f"{val} tier" if field == "customer_tier" else val
                    badges += f'<span class="chip">{show}</span>'
            # Drop the first content line if it duplicates the title.
            body = doc["content"]
            first, _, rest = body.partition("\n")
            if first.strip() == doc["title"].strip():
                body = rest.lstrip("\n")
            st.markdown(
                f'<div class="detail-card">{badges}'
                f'<div class="detail-title">{doc["title"]}</div>'
                f'<div class="detail-body">{body}</div></div>',
                unsafe_allow_html=True,
            )


def main():
    st.set_page_config(page_title="ShopEasy KB", page_icon="🛒", layout="wide")
    inject_css()

    docs = load_docs()

    # ---- Shared header --------------------------------------------------
    st.markdown('<div class="kb-title">🛒 ShopEasy Knowledge Base</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="kb-sub">Internal support reference · {len(docs)} articles</div>',
        unsafe_allow_html=True,
    )

    tab_browse, tab_assistant = st.tabs(["📚 Knowledge Base", "💬 Support Assistant"])
    with tab_browse:
        render_kb_browser(docs)
    with tab_assistant:
        render_support_assistant()


if __name__ == "__main__":
    main()

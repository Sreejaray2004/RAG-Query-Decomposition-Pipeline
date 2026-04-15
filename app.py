import streamlit as st
from langchain_huggingface import HuggingFaceEndpointEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.messages import SystemMessage, HumanMessage
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from dotenv import load_dotenv
from pypdf import PdfReader
import os
import json
import time
import uuid
import hashlib
import numpy as np

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
QDRANT_PATH = "./qdrant_storage"
COLLECTION_NAME = "rag_chunks"

# ─────────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Query Pipeline",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Query Decomposition Pipeline")
st.markdown("Upload one or more **PDFs**, enter a **complex query**, and get a smart answer powered by AI.")
st.divider()

# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("**Model Info**")
    st.caption("🔹 Embeddings: all-MiniLM-L6-v2")
    st.caption("🔹 LLM: Qwen2.5-7B-Instruct")
    st.caption("🔹 Chunking: Semantic (topic-wise)")
    st.caption("🔹 Vector Store: Qdrant (on-disk)")
    st.divider()
    st.markdown("**Storage**")
    st.caption(f"📁 Qdrant path: `{QDRANT_PATH}`")

# ─────────────────────────────────────────────
#  Cached Model Loaders
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    embedding_model = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
    )
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="text-generation",
        max_new_tokens=512,
    )
    chat_model = ChatHuggingFace(llm=llm)
    return embedding_model, chat_model

# ─────────────────────────────────────────────
#  Qdrant Client (on-disk, persistent)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_qdrant_client():
    return QdrantClient(path=QDRANT_PATH)

def reset_collection(client, vector_size=384):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )



def store_chunks_in_qdrant(client, chunks, embeddings, source_name):
    """Store chunks with source filename tracked in payload."""
    points = []
    
    for i in range(len(chunks)):
        # 1. Create an MD5 hash of the chunk's text
        text_hash = hashlib.md5(chunks[i].encode('utf-8')).hexdigest()
        
        # 2. Convert that hash into a valid UUID format
        deterministic_id = str(uuid.UUID(text_hash))

        points.append(
            PointStruct(
                id=deterministic_id,     # <-- Replaced uuid.uuid4() with our text-based ID
                vector=embeddings[i],
                payload={
                    "text": chunks[i],
                    "chunk_index": i,
                    "source": source_name 
                }
            )
        )
        
    client.upsert(collection_name=COLLECTION_NAME, points=points)

    
def search_qdrant(client, query_vector, top_k=4):
    """
    Fetch extra results and deduplicate by the ACTUAL TEXT, 
    ensuring identical chunks are never shown twice.
    """
    raw_hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k * 3,          # Fetch extra to survive the text deduplication
    ).points

    seen_texts = set()
    unique_hits = []
    
    for hit in raw_hits:
        # Strip whitespace/newlines to catch identical chunks with hidden formatting
        chunk_text = hit.payload.get("text", "").strip()
        
        if chunk_text not in seen_texts:
            seen_texts.add(chunk_text)
            unique_hits.append(hit)
            
        if len(unique_hits) == top_k:
            break

    return unique_hits
# ─────────────────────────────────────────────
#  PDF Text Extraction
# ─────────────────────────────────────────────
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text.strip()

# ─────────────────────────────────────────────
#  Semantic Chunking
# ─────────────────────────────────────────────
def semantic_chunk(text, embedding_model):
    chunker = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85
    )
    docs = chunker.create_documents([text])
    return [doc.page_content for doc in docs]

# ─────────────────────────────────────────────
#  Retry Embedding
# ─────────────────────────────────────────────
def embed_with_retry(texts, embedding_model, retries=5, delay=5):
    """Embed a list of document texts (batch mode)."""
    for attempt in range(retries):
        try:
            return embedding_model.embed_documents(texts)
        except Exception as e:
            if attempt < retries - 1:
                st.warning(f"⚠️ Attempt {attempt+1} failed. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                raise RuntimeError("❌ Embedding failed after all retries.") from e

def embed_query_with_retry(query, embedding_model, retries=5, delay=5):
    """
    Embed a single query string using embed_query() — NOT embed_documents().
    This is the correct method for query-time encoding and avoids
    identical scores caused by using batch-document mode for queries.
    """
    for attempt in range(retries):
        try:
            return embedding_model.embed_query(query)   # ← key fix
        except Exception as e:
            if attempt < retries - 1:
                st.warning(f"⚠️ Query embed attempt {attempt+1} failed. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                raise RuntimeError("❌ Query embedding failed after all retries.") from e

# ─────────────────────────────────────────────
#  Query Decomposition
# ─────────────────────────────────────────────
def decompose_query(complex_query, chat_model):
    messages = [
        SystemMessage(content="""You are a query decomposition assistant.
Break the user's complex query into 3-5 simpler, self-contained sub-queries.
Respond ONLY with a valid JSON array of strings. No explanation, no markdown.
Example: ["sub-query 1", "sub-query 2", "sub-query 3"]"""),
        HumanMessage(content=f"Complex Query: {complex_query}")
    ]
    response = chat_model.invoke(messages)
    raw = response.content.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

# ─────────────────────────────────────────────
#  Retrieve Best Chunks from Qdrant
# ─────────────────────────────────────────────
def retrieve_best_chunks(sub_queries, embedding_model, qdrant_client):
    results = []
    all_top_scores = []

    for query in sub_queries:
        # Use embed_query — not embed_documents — for query vectors
        query_vector = embed_query_with_retry(query, embedding_model)
        hits = search_qdrant(qdrant_client, query_vector, top_k=2)

        top    = hits[0] if len(hits) > 0 else None
        second = hits[1] if len(hits) > 1 else None

        top_score = float(top.score) if top else 0.0
        all_top_scores.append(top_score)

        results.append({
            "sub_query":     query,
            "best_chunk":    top.payload["text"]    if top    else "No result",
            "best_source":   top.payload.get("source", "unknown")    if top    else "—",
            "score":         top_score,
            "second_chunk":  second.payload["text"] if second else "No result",
            "second_source": second.payload.get("source", "unknown") if second else "—",
            "second_score":  float(second.score)    if second else 0.0,
        })

    aggregate_score = float(np.mean(all_top_scores)) if all_top_scores else 0.0
    return results, aggregate_score

# ─────────────────────────────────────────────
#  Final Answer Assembly
# ─────────────────────────────────────────────
def assemble_final_answer(complex_query, retrieved_results, chat_model):
    context = "\n\n".join([
        f"Sub-Query: {r['sub_query']}\nSource: {r['best_source']}\nRelevant Info: {r['best_chunk']}"
        for r in retrieved_results
    ])
    messages = [
        SystemMessage(content="""You are a helpful assistant.
Using the provided context from multiple sub-queries,
write a single, smooth, and coherent final answer to the original complex query.
Do not mention sub-queries in your answer. Be concise and informative."""),
        HumanMessage(content=f"Original Query: {complex_query}\n\nContext:\n{context}\n\nProvide a comprehensive and fluent final answer.")
    ]
    response = chat_model.invoke(messages)
    return response.content.strip()

# ─────────────────────────────────────────────
#  Main UI
# ─────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📄 Upload PDFs")
    # ← accept_multiple_files=True allows multiple PDFs
    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
        for uf in uploaded_files:
            with st.expander(f"📋 Preview: {uf.name}"):
                preview_text = extract_text_from_pdf(uf)
                st.text(preview_text[:1000] + "..." if len(preview_text) > 1000 else preview_text)

with col2:
    st.subheader("💬 Your Query")
    user_query = st.text_area(
        "Enter your complex query",
        placeholder="e.g. How is AI used in healthcare and what are the ethical concerns?",
        height=150
    )

st.divider()

# ─────────────────────────────────────────────
#  Run Button
# ─────────────────────────────────────────────
run = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

if run:
    if not HF_TOKEN:
        st.error("❌ HUGGINGFACEHUB_API_TOKEN not found. Set it in your .env file.")
    elif not uploaded_files:
        st.error("❌ Please upload at least one PDF file.")
    elif not user_query.strip():
        st.error("❌ Please enter a query.")
    else:
        # Load models
        with st.spinner("⚙️ Loading models..."):
            embedding_model, chat_model = load_models()

        # ── Process each PDF separately, store all into same collection ──
        qdrant_client = get_qdrant_client()
        collection_initialized = False
        all_chunk_counts = {}

        for uf in uploaded_files:
            st.markdown(f"---\n### 📄 Processing: `{uf.name}`")

            with st.spinner(f"Extracting text from {uf.name}..."):
                raw_text = extract_text_from_pdf(uf)
                st.info(f"📃 Extracted **{len(raw_text.split())} words**")

            with st.spinner(f"Semantic chunking {uf.name}..."):
                chunks = semantic_chunk(raw_text, embedding_model)

            st.success(f"✅ **{len(chunks)} chunks** created from `{uf.name}`")
            all_chunk_counts[uf.name] = len(chunks)

            with st.expander(f"📋 View chunks from {uf.name}"):
                for i, chunk in enumerate(chunks):
                    st.markdown(f"**Chunk {i+1}** ({len(chunk.split())} words)")
                    st.caption(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                    st.divider()

            with st.spinner(f"💾 Embedding & storing chunks from {uf.name}..."):
                chunk_embeddings = embed_with_retry(chunks, embedding_model)

                # Create collection once (using first PDF's vector size)
                if not collection_initialized:
                    vector_size = len(chunk_embeddings[0])
                    reset_collection(qdrant_client, vector_size=vector_size)
                    collection_initialized = True

                # Store with source filename in payload
                store_chunks_in_qdrant(qdrant_client, chunks, chunk_embeddings, uf.name)

            st.success(f"✅ Stored in Qdrant with source tag `{uf.name}`")

        # Summary
        st.divider()
        total_chunks = sum(all_chunk_counts.values())
        st.info(f"📦 Total chunks stored across all PDFs: **{total_chunks}** "
                f"from **{len(uploaded_files)}** file(s)")

        # Decompose query
        with st.spinner("🔍 Decomposing query into sub-queries..."):
            sub_queries = decompose_query(user_query, chat_model)

        st.subheader("🔍 Sub-Queries Generated")
        for i, q in enumerate(sub_queries, 1):
            st.markdown(f"**{i}.** {q}")

        st.divider()

        # Retrieve chunks via Qdrant
        with st.spinner("📊 Querying Qdrant for best matching chunks..."):
            retrieved, aggregate_score = retrieve_best_chunks(
                sub_queries, embedding_model, qdrant_client
            )

        st.subheader("📊 Cosine Similarity Results")
        for r in retrieved:
            with st.expander(f"🔸 {r['sub_query']}"):
                col_a, col_b = st.columns([1, 3])
                with col_a:
                    st.metric("Top Score", f"{r['score']:.4f}")
                    st.caption(f"📄 `{r['best_source']}`")
                with col_b:
                    st.caption(r['best_chunk'][:400])

                st.divider()

                col_c, col_d = st.columns([1, 3])
                with col_c:
                    st.metric("2nd Score", f"{r['second_score']:.4f}")
                    st.caption(f"📄 `{r['second_source']}`")
                with col_d:
                    st.caption(r['second_chunk'][:400])

        # ── Aggregate Score ──────────────────────────────
        st.divider()
        st.subheader("📈 Aggregate Retrieval Score")
        agg_col1, agg_col2 = st.columns([1, 3])
        with agg_col1:
            st.metric(
                label="Average Cosine Score",
                value=f"{aggregate_score:.4f}",
                help="Mean of top-1 cosine similarity scores across all sub-queries"
            )
        with agg_col2:
            individual_scores = [r['score'] for r in retrieved]
            score_breakdown = " | ".join(
                [f"Q{i+1}: {s:.4f}" for i, s in enumerate(individual_scores)]
            )
            st.caption(f"Per sub-query top scores → {score_breakdown}")
            if aggregate_score >= 0.75:
                st.success("✅ High relevance — retrieved chunks closely match the query.")
            elif aggregate_score >= 0.50:
                st.warning("⚠️ Moderate relevance — some chunks may be partially relevant.")
            else:
                st.error("❌ Low relevance — document may not cover this topic well.")

        st.divider()

        # Final answer
        with st.spinner("✨ Assembling final answer..."):
            final_answer = assemble_final_answer(user_query, retrieved, chat_model)

        st.subheader("✨ Final Answer")
        st.success(final_answer)
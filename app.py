import streamlit as st

from config.settings import (
    HF_TOKEN,
    QDRANT_PATH,
)

from pipeline.rag_pipeline import run_pipeline

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="RAG Query Pipeline",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 RAG Query Decomposition Pipeline")
st.markdown(
    "Upload one or more **PDFs**, enter a **complex query**, and get a smart answer powered by AI."
)
st.divider()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
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

# --------------------------------------------------
# Main Layout
# --------------------------------------------------
col1, col2 = st.columns([1, 1])

# --------------------------------------------------
# Upload PDFs
# --------------------------------------------------
with col1:

    st.subheader("📄 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

# --------------------------------------------------
# Query
# --------------------------------------------------
with col2:

    st.subheader("💬 Your Query")

    user_query = st.text_area(
        "Enter your complex query",
        placeholder="e.g. How is AI used in healthcare and what are the ethical concerns?",
        height=150,
    )

st.divider()

# --------------------------------------------------
# Run Button
# --------------------------------------------------
run = st.button(
    "🚀 Run Pipeline",
    type="primary",
    use_container_width=True,
)

# --------------------------------------------------
# Run Pipeline
# --------------------------------------------------
if run:

    if not HF_TOKEN:

        st.error(
            "❌ HUGGINGFACEHUB_API_TOKEN not found. Set it in your .env file."
        )

    elif not uploaded_files:

        st.error(
            "❌ Please upload at least one PDF file."
        )

    elif not user_query.strip():

        st.error(
            "❌ Please enter a query."
        )

    else:

        with st.spinner("🚀 Running RAG Pipeline..."):

            # Call your pipeline (single call only — calling this twice
            # re-reads the same UploadedFile objects, which are already
            # exhausted after the first read and can produce empty results)
            result = run_pipeline(uploaded_files, user_query)

        # --------------------------------------------------
        # Document Information
        # --------------------------------------------------
        st.subheader("📄 Document Processing")

        for doc in result["documents"]:

            st.markdown(f"### 📄 `{doc['filename']}`")

            st.info(
                f"📃 Extracted **{doc['word_count']} words**"
            )

            st.success(
                f"✅ **{doc['chunk_count']} chunks** created"
            )

            with st.expander(
                f"📋 Preview: {doc['filename']}"
            ):

                preview = doc["raw_text"]

                if len(preview) > 1000:
                    preview = preview[:1000] + "..."

                st.text(preview)

            with st.expander(
                f"📑 View Chunks ({doc['filename']})"
            ):

                for i, chunk in enumerate(doc["chunks"], start=1):

                    st.markdown(
                        f"**Chunk {i}** ({len(chunk.split())} words)"
                    )

                    st.caption(
                        chunk[:300] + "..."
                        if len(chunk) > 300
                        else chunk
                    )

                    st.divider()

        st.divider()

        # --------------------------------------------------
        # Summary
        # --------------------------------------------------
        st.info(
            f"📦 Total chunks stored across all PDFs: "
            f"**{result['total_chunks']}** "
            f"from **{len(uploaded_files)}** file(s)"
        )

        # --------------------------------------------------
        # Sub Queries
        # --------------------------------------------------
        st.subheader("🔍 Sub-Queries Generated")

        for i, query in enumerate(
            result["sub_queries"],
            start=1,
        ):

            st.markdown(f"**{i}.** {query}")

        st.divider()

        # --------------------------------------------------
        # Retrieval Results
        # --------------------------------------------------

        st.subheader("📊 Cosine Similarity Results")

        for r in result["retrieved"]:

            with st.expander(
                f"🔸 {r['sub_query']}"
            ):

                col_a, col_b = st.columns([1, 3])

                with col_a:

                    st.metric(
                        "Top Score",
                        f"{r['score']:.4f}",
                    )

                    st.caption(
                        f"📄 `{r['best_source']}`"
                    )

                with col_b:

                    st.caption(
                        r["best_chunk"][:400]
                    )

                st.divider()

                col_c, col_d = st.columns([1, 3])

                with col_c:

                    st.metric(
                        "2nd Score",
                        f"{r['second_score']:.4f}",
                    )

                    st.caption(
                        f"📄 `{r['second_source']}`"
                    )

                with col_d:

                    st.caption(
                        r["second_chunk"][:400]
                    )

        # --------------------------------------------------
        # Aggregate Score
        # --------------------------------------------------
        st.divider()

        st.subheader("📈 Aggregate Retrieval Score")

        aggregate_score = result["aggregate_score"]

        agg_col1, agg_col2 = st.columns([1, 3])

        with agg_col1:

            st.metric(
                "Average Cosine Score",
                f"{aggregate_score:.4f}",
                help="Mean of top-1 cosine similarity scores across all sub-queries",
            )

        with agg_col2:

            scores = [
                r["score"]
                for r in result["retrieved"]
            ]

            breakdown = " | ".join(
                [
                    f"Q{i+1}: {s:.4f}"
                    for i, s in enumerate(scores)
                ]
            )

            st.caption(
                f"Per sub-query top scores → {breakdown}"
            )

            if aggregate_score >= 0.75:

                st.success(
                    "✅ High relevance — retrieved chunks closely match the query."
                )

            elif aggregate_score >= 0.50:

                st.warning(
                    "⚠️ Moderate relevance — some chunks may be partially relevant."
                )

            else:

                st.error(
                    "❌ Low relevance — document may not cover this topic well."
                )

        st.divider()

        # --------------------------------------------------
        # Final Answer
        # --------------------------------------------------
        st.subheader("✨ Final Answer")

        st.success(
            result["final_answer"]
        )
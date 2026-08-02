from models.loader import load_models

from document.pdf_loader import extract_text_from_pdf
from document.chunking import semantic_chunk

from models.embeddings import embed_with_retry

from database.qdrant_manager import (
    get_qdrant_client,
    reset_collection,
    store_chunks_in_qdrant,
)

from pipeline.decomposition import decompose_query
from pipeline.retrieval import retrieve_best_chunks
from pipeline.answer_generator import assemble_final_answer



def run_pipeline(uploaded_files, user_query):
    """
    Complete RAG workflow.

    Parameters
    ----------
    uploaded_files : list
        Uploaded PDF files from Streamlit

    user_query : str
        User's complex query

    Returns
    -------
    dict
        Everything required by the UI.
    """

    # -----------------------------
    # Load Models
    # -----------------------------
    embedding_model, chat_model = load_models()

    # -----------------------------
    # Qdrant
    # -----------------------------
    qdrant_client = get_qdrant_client()

    collection_initialized = False

    all_chunk_counts = {}

    document_details = []

    # -----------------------------
    # Process Documents
    # -----------------------------
    for uf in uploaded_files:

        # Extract Text
        raw_text = extract_text_from_pdf(uf)

        # Semantic Chunking
        chunks = semantic_chunk(
            raw_text,
            embedding_model,
        )

        # Embeddings
        chunk_embeddings = embed_with_retry(
            chunks,
            embedding_model,
        )

        # Initialize Collection
        if not collection_initialized:

            vector_size = len(chunk_embeddings[0])

            reset_collection(
                qdrant_client,
                vector_size,
            )

            collection_initialized = True

        # Store
        store_chunks_in_qdrant(
            qdrant_client,
            chunks,
            chunk_embeddings,
            uf.name,
        )

        # Store UI Information
        all_chunk_counts[uf.name] = len(chunks)

        document_details.append(
            {
                "filename": uf.name,
                "raw_text": raw_text,
                "word_count": len(raw_text.split()),
                "chunks": chunks,
                "chunk_count": len(chunks),
            }
        )

    # -----------------------------
    # Query Decomposition
    # -----------------------------
    sub_queries = decompose_query(
        user_query,
        chat_model,
    )

    # -----------------------------
    # Retrieval
    # -----------------------------
    retrieved, aggregate_score = retrieve_best_chunks(
        sub_queries,
        embedding_model,
        qdrant_client,
    )

    # -----------------------------
    # Final Answer
    # -----------------------------
    final_answer = assemble_final_answer(
        user_query,
        retrieved,
        chat_model,
    )

    # -----------------------------
    # Return Everything
    # -----------------------------
    return {

        "documents": document_details,

        "sub_queries": sub_queries,

        "retrieved": retrieved,

        "aggregate_score": aggregate_score,

        "final_answer": final_answer,

        "chunk_counts": all_chunk_counts,

        "total_chunks": sum(all_chunk_counts.values()),
    }
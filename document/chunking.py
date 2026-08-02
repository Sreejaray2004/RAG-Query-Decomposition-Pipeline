from langchain_experimental.text_splitter import SemanticChunker


def semantic_chunk(text, embedding_model):
    """
    Split text into semantic chunks.
    """

    chunker = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85,
    )

    docs = chunker.create_documents([text])

    return [doc.page_content for doc in docs]
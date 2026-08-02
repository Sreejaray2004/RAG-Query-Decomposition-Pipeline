import hashlib
import uuid

import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from config.settings import (
    COLLECTION_NAME,
    QDRANT_PATH,
)


# --------------------------------------------------
# Qdrant Client
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_qdrant_client():
    """
    Returns a persistent on-disk Qdrant client.
    """
    return QdrantClient(path=QDRANT_PATH)


# --------------------------------------------------
# Reset Collection
# --------------------------------------------------

def reset_collection(client, vector_size=384):

    existing = [
        c.name
        for c in client.get_collections().collections
    ]

    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )


# --------------------------------------------------
# Store Chunks
# --------------------------------------------------

def store_chunks_in_qdrant(
    client,
    chunks,
    embeddings,
    source_name,
):

    points = []

    for i in range(len(chunks)):

        text_hash = hashlib.md5(
            chunks[i].encode("utf-8")
        ).hexdigest()

        deterministic_id = str(uuid.UUID(text_hash))

        points.append(
            PointStruct(
                id=deterministic_id,
                vector=embeddings[i],
                payload={
                    "text": chunks[i],
                    "chunk_index": i,
                    "source": source_name,
                },
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )


# --------------------------------------------------
# Search
# --------------------------------------------------

def search_qdrant(
    client,
    query_vector,
    top_k=4,
):

    raw_hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k * 3,
    ).points

    seen_texts = set()

    unique_hits = []

    for hit in raw_hits:

        chunk_text = hit.payload.get(
            "text",
            "",
        ).strip()

        if chunk_text not in seen_texts:

            seen_texts.add(chunk_text)

            unique_hits.append(hit)

        if len(unique_hits) == top_k:
            break

    return unique_hits
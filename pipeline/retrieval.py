import numpy as np

from database.qdrant_manager import search_qdrant
from models.embeddings import embed_query_with_retry


def retrieve_best_chunks(
    sub_queries,
    embedding_model,
    qdrant_client,
):

    results = []

    all_top_scores = []

    for query in sub_queries:

        query_vector = embed_query_with_retry(
            query,
            embedding_model,
        )

        hits = search_qdrant(
            qdrant_client,
            query_vector,
            top_k=2,
        )

        top = hits[0] if len(hits) > 0 else None
        second = hits[1] if len(hits) > 1 else None

        top_score = float(top.score) if top else 0.0

        all_top_scores.append(top_score)

        results.append(
            {
                "sub_query": query,

                "best_chunk":
                    top.payload["text"]
                    if top else "No result",

                "best_source":
                    top.payload.get("source", "unknown")
                    if top else "—",

                "score": top_score,

                "second_chunk":
                    second.payload["text"]
                    if second else "No result",

                "second_source":
                    second.payload.get("source", "unknown")
                    if second else "—",

                "second_score":
                    float(second.score)
                    if second else 0.0,
            }
        )

    aggregate_score = (
        float(np.mean(all_top_scores))
        if all_top_scores
        else 0.0
    )

    return results, aggregate_score
import time
import streamlit as st


def embed_with_retry(texts, embedding_model, retries=5, delay=5):
    """
    Embed a list of document texts (batch mode).
    """
    for attempt in range(retries):
        try:
            return embedding_model.embed_documents(texts)

        except Exception as e:
            if attempt < retries - 1:
                st.warning(
                    f"⚠️ Attempt {attempt + 1} failed. Retrying in {delay}s..."
                )
                time.sleep(delay)
                delay *= 2

            else:
                raise RuntimeError(
                    "❌ Embedding failed after all retries."
                ) from e


def embed_query_with_retry(query, embedding_model, retries=5, delay=5):
    """
    Embed a single query string using embed_query().
    """

    for attempt in range(retries):
        try:
            return embedding_model.embed_query(query)

        except Exception as e:

            if attempt < retries - 1:

                st.warning(
                    f"⚠️ Query embed attempt {attempt + 1} failed. Retrying in {delay}s..."
                )

                time.sleep(delay)
                delay *= 2

            else:
                raise RuntimeError(
                    "❌ Query embedding failed after all retries."
                ) from e
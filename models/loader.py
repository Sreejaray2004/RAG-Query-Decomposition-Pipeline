import streamlit as st

from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)

from config.settings import (
    EMBEDDING_MODEL_NAME,
    LLM_REPO_ID,
    MAX_NEW_TOKENS,
)


@st.cache_resource(show_spinner=False)
def load_models():

    embedding_model = HuggingFaceEmbeddings(
        model=EMBEDDING_MODEL_NAME,
    )

    llm = HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID,
        task="text-generation",
        provider="featherless-ai",
        max_new_tokens=MAX_NEW_TOKENS,
    )

    chat_model = ChatHuggingFace(llm=llm)

    return embedding_model, chat_model
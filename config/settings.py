import os
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------
# Environment Variables
# --------------------------------------------------
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

# --------------------------------------------------
# Qdrant Configuration
# --------------------------------------------------
QDRANT_PATH = "./qdrant_storage"
COLLECTION_NAME = "rag_chunks"

# --------------------------------------------------
# Embedding Model
# --------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --------------------------------------------------
# LLM Configuration
# --------------------------------------------------
LLM_REPO_ID = "Qwen/Qwen2.5-7B-Instruct-1M"
MAX_NEW_TOKENS = 512
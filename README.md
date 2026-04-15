#  🤖 RAG Query Decomposition Pipeline

A powerful, local-first Retrieval-Augmented Generation (RAG) application built with **Streamlit**, **LangChain**, and **Qdrant**. 

Instead of searching for a complex query directly, this pipeline uses an LLM to **decompose** your complex question into smaller, targeted sub-queries. It then retrieves the most relevant semantic chunks for each sub-query from your uploaded PDFs and synthesizes a comprehensive, highly accurate final answer.

## ✨ Features

* **📄 Multi-PDF Processing:** Upload and extract text from multiple PDF documents simultaneously.
* **🧠 Semantic Chunking:** Intelligently splits text based on topic boundaries (using `SemanticChunker`) rather than arbitrary character counts.
* **🗄️ Local Vector Database:** Uses an on-disk instance of **Qdrant** for persistent, fast, and private vector storage.
* **🔍 Query Decomposition:** Uses **Qwen2.5-7B-Instruct** to break down complex user prompts into 3-5 manageable sub-queries.
* **🎯 Precision Retrieval:** Uses Hugging Face embeddings (`all-MiniLM-L6-v2`) to find the best context, applying deduplication to ensure diverse information.
* **✨ Synthesized Answers:** Generates a fluent, coherent final response using the combined context of all sub-queries.

## 🛠️ Tech Stack

* **UI Framework:** [Streamlit](https://streamlit.io/)
* **Orchestration:** [LangChain](https://python.langchain.com/)
* **Embeddings:** Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`)
* **LLM:** Hugging Face Serverless Inference API (`Qwen/Qwen2.5-7B-Instruct`)
* **Vector Store:** [Qdrant](https://qdrant.tech/) (Local Client)
* **Document Parsing:** PyPDF

## 🚀 Getting Started

### Prerequisites
* Python 3.8 or higher
* A free [Hugging Face account and Access Token](https://huggingface.co/settings/tokens)

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/rag-decomposition-pipeline.git](https://github.com/yourusername/rag-decomposition-pipeline.git)
   cd rag-decomposition-pipeline
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
3. **Set up your environment variables:
Create a .env file in the root directory of the project and add your Hugging Face API token:**
   ```bash
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
4. **Start the Streamlit server:**
   ```bash
   streamlit run app.py
## 💡 How to Use
* **📄 Upload PDFs:** Use the sidebar or main panel to upload one or more PDF files. The app will extract the text, create semantic chunks, and store them in the local Qdrant database.
* **📄 Enter a Query:** Type a multi-part or complex question in the text area (e.g., "How does the provided document define machine learning, and what are the primary ethical concerns mentioned regarding its use in healthcare?").
* **📄 Run Pipeline:** Click the "Run Pipeline" button.
* **📄 View Process:** Expand the UI sections to see the extracted chunks, the decomposed sub-queries, the cosine similarity scores for the retrieved text, and the final LLM-generated answer.
## 📂 Project Structure
   
     ├── app.py                # Main Streamlit application file
     ├── requirements.txt      # Python dependencies
     ├── .env                  # Environment variables (create this!)
     ├── README.md             # Project documentation
     └── qdrant_storage/       # Auto-generated local vector database folder
## ⚠️ Important Notes
* **📄 API Limits::** This project relies on the free Hugging Face Serverless Inference API. If you process massive PDFs or make rapid, repeated requests, you may hit rate limits. For heavy production use, consider switching to local inference (e.g., Ollama) or a paid API provider.
* **📄 Storage:** The qdrant_storage folder will be created locally in your project directory to persist your document embeddings between sessions.

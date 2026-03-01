# RAG Knowledge Assistant

A local Retrieval-Augmented Generation (RAG) system built using:

- SentenceTransformers for embeddings
- ChromaDB for vector storage
- Ollama (Llama3) for local LLM inference
- Streamlit for UI

## Features
- PDF/TXT document ingestion
- Semantic chunking with overlap
- Vector similarity search (Top-K retrieval)
- Grounded answer generation with citations
- CLI and Streamlit interface

## How It Works
1. Documents are chunked
2. Chunks are embedded and stored in ChromaDB
3. User query is embedded
4. Top-K similar chunks are retrieved
5. LLM generates answer grounded in retrieved context

## Run Locally

```bash
pip install -r requirements.txt
python ingest.py
streamlit run streamlit_app.py
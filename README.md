 RAG Knowledge Assistant (Local LLM + Vector Search)

A production-style Retrieval-Augmented Generation (RAG) system built using:

- SentenceTransformers (all-MiniLM-L6-v2) for embeddings
- ChromaDB for persistent vector storage
- Ollama (Llama3) for local LLM inference
- Streamlit for interactive UI
- Rich CLI for terminal interface

---

 Architecture Overview

1. Document Ingestion
   - Supports PDF / TXT / Markdown
   - Text extraction using pypdf
   - Structured format: {"source": "...", "text": "..."}

2. Semantic Chunking
   - Sliding window chunking (800 char size, 120 overlap)
   - Preserves context continuity across chunks

3. Embedding Generation
   - SentenceTransformers (all-MiniLM-L6-v2)
   - Normalized embeddings for cosine similarity

4. Vector Storage
   - Persistent ChromaDB vector database
   - UUID-based chunk IDs
   - Metadata-aware indexing (source + chunk number)

5. Retrieval + Grounded Generation
   - Top-K semantic retrieval
   - Context injection into LLM prompt
   - Citation-backed responses
   - Hallucination mitigation via prompt constraints

---

 Features

- Semantic search over custom documents
- Citation-based answer generation
- Transparent chunk-level retrieval display
- CLI + Web Interface
- Fully local (no external API dependency)

---

 Run Locally

1. Install dependencies

```bash
pip install -r requirements.txt
```

 2. Start Ollama (Llama3)

```bash
ollama run llama3.2:3b
```

 3. Index documents

```bash
python ingest.py
```

 4. Run Web App

```bash
streamlit run streamlit_app.py
```

---

 Example Queries

- What is gradient descent?
- Explain overfitting and how to prevent it.
- Difference between logistic and linear regression.
- How much protein is required for muscle gain?

---

 Key Engineering Decisions

- Overlapping chunking to preserve semantic boundaries
- Normalized embeddings for stable similarity comparison
- Persistent vector storage for scalability
- Prompt constraints to reduce hallucinations
- Citation enforcement for explainability

---

 Future Improvements

- Hybrid retrieval (BM25 + vector search)
- Re-ranking model integration
- RAG evaluation metrics (Precision@K)
- FastAPI backend deployment
- Conversational memory layer

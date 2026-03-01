import uuid
from typing import List, Dict
import requests
from sentence_transformers import SentenceTransformer
import chromadb


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """
    Simple character-based chunking with overlap.
    """
    chunks: List[str] = []
    i = 0
    text = text or ""
    while i < len(text):
        chunk = text[i : i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        i += max(1, chunk_size - overlap)
    return chunks


class RAG:
    def __init__(
        self,
        persist_dir: str = "chroma_db",
        collection_name: str = "docs",
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        ollama_url: str = "http://localhost:11434/api/generate",
        ollama_model: str = "llama3.2:3b",
    ):
        self.embedder = SentenceTransformer(embed_model_name)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.col = self.client.get_or_create_collection(name=collection_name)
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    def add_documents(self, docs: List[Dict]) -> int:
        """
        docs: [{"source": "file_name", "text": "..."}]
        Uses UUID IDs so re-indexing doesn't collide.
        """
        ids, texts, metas = [], [], []

        for d in docs:
            source = d.get("source", "unknown")
            full_text = d.get("text", "")
            for idx, chunk in enumerate(chunk_text(full_text)):
                cid = str(uuid.uuid4())
                ids.append(cid)
                texts.append(chunk)
                metas.append({"source": source, "chunk": idx})

        if not ids:
            return 0

        vectors = self.embedder.encode(texts, normalize_embeddings=True).tolist()
        self.col.add(ids=ids, documents=texts, metadatas=metas, embeddings=vectors)
        return len(ids)

    def retrieve(self, query: str, k: int = 4) -> List[Dict]:
        qvec = self.embedder.encode([query], normalize_embeddings=True).tolist()
        res = self.col.query(query_embeddings=qvec, n_results=k)

        out: List[Dict] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        ids = res.get("ids", [[]])[0]

        for doc, meta, _id in zip(docs, metas, ids):
            out.append({"id": _id, "text": doc, "meta": meta})
        return out

    def generate(self, query: str, contexts: List[Dict]) -> str:
        context_block = "\n\n".join(
            [
                f"[Source: {c['meta']['source']} | chunk {c['meta']['chunk']}]\n{c['text']}"
                for c in contexts
            ]
        )

        prompt = f"""
You are a helpful assistant answering using ONLY the provided context.
If the context is insufficient, say: "I don't have enough information in the documents."

Question:
{query}

Context:
{context_block}

Answer:
- Give a clear answer.
- Then provide citations as bullet points like:
  - source_file :: chunk_number
"""

        payload = {"model": self.ollama_model, "prompt": prompt, "stream": False}
        r = requests.post(self.ollama_url, json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()
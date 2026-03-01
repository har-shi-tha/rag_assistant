from loaders import load_documents
from rag import RAG


if __name__ == "__main__":
    rag = RAG()
    docs = load_documents("data/docs")
    added = rag.add_documents(docs)
    print(f" Indexed {added} chunks from {len(docs)} files.")
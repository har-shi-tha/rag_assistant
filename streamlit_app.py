import shutil
from pathlib import Path
import streamlit as st
from pypdf import PdfReader
from rag import RAG

APP_DIR = Path(__file__).parent
DB_DIR = APP_DIR / "chroma_db"


def extract_text_from_upload(uploaded_file) -> str:
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        pages = []
        for p in reader.pages:
            pages.append(p.extract_text() or "")
        return "\n".join(pages).strip()

    return uploaded_file.read().decode("utf-8", errors="ignore").strip()


def reset_vector_db():
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR, ignore_errors=True)


st.set_page_config(page_title="RAG Knowledge Assistant", layout="wide")
st.title("📚 RAG Knowledge Assistant")

if "rag" not in st.session_state:
    st.session_state.rag = RAG()
if "indexed" not in st.session_state:
    st.session_state.indexed = False

with st.sidebar:
    st.header("⚙️ Controls")
    st.caption("1) Upload docs  2) Index  3) Ask questions")

    uploaded_files = st.file_uploader(
        "Upload PDFs / TXT / MD",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    colA, colB = st.columns(2)

    with colA:
        if st.button("📥 Index Documents", use_container_width=True):
            if not uploaded_files:
                st.warning("Upload at least one document first.")
            else:
                docs = []
                for f in uploaded_files:
                    text = extract_text_from_upload(f)
                    if text:
                        docs.append({"source": f.name, "text": text})

                if not docs:
                    st.error("No text could be extracted from the uploaded files.")
                else:
                    added = st.session_state.rag.add_documents(docs)
                    st.session_state.indexed = True
                    st.success(f"Indexed {added} chunks from {len(docs)} file(s).")

    with colB:
        if st.button("🧹 Reset Index", use_container_width=True):
            reset_vector_db()
            st.session_state.rag = RAG()
            st.session_state.indexed = False
            st.success("Vector database cleared. Upload + Index again.")

st.divider()

query = st.text_input("Ask a question (grounded in your uploaded docs):", placeholder="e.g., What is gradient descent?")

col1, col2 = st.columns([2, 1])
with col2:
    k = st.slider("Top-K chunks", min_value=2, max_value=8, value=4, step=1)
with col1:
    ask_btn = st.button("🔎 Ask", type="primary", use_container_width=True)

if ask_btn:
    if not st.session_state.indexed:
        st.warning("Index documents first (upload → Index Documents).")
    elif not query.strip():
        st.warning("Type a question first.")
    else:
        with st.spinner("Retrieving relevant chunks..."):
            contexts = st.session_state.rag.retrieve(query, k=k)

        with st.spinner("Generating answer using Ollama..."):
            answer = st.session_state.rag.generate(query, contexts)

        st.subheader("✅ Answer")
        st.write(answer)

        st.subheader("📌 Retrieved Context")
        for i, c in enumerate(contexts, 1):
            src = c["meta"]["source"]
            chk = c["meta"]["chunk"]
            with st.expander(f"{i}. {src} :: chunk {chk}"):
                st.write(c["text"])
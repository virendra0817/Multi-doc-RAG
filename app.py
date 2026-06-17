import streamlit as st
import tempfile
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_all_documents
from search import RAGSearch

st.set_page_config(page_title="RAG Assistant", page_icon="🧠", layout="wide")
st.title("🧠 RAG Document Assistant")

# --- Session State ---
if "rag" not in st.session_state:
    st.session_state.rag = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed" not in st.session_state:
    st.session_state.indexed = False

# --- Sidebar: File Upload ---
with st.sidebar:
    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF / TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("🔨 Build Index"):
        with st.spinner("Processing and indexing documents..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                for f in uploaded_files:
                    path = os.path.join(tmpdir, f.name)
                    with open(path, "wb") as out:
                        out.write(f.read())

                docs = load_all_documents(tmpdir)
                st.session_state.rag = RAGSearch()
                st.session_state.rag.vectorstore.build_from_documents(docs)
                st.session_state.indexed = True

        st.success(f"✅ Indexed {len(uploaded_files)} file(s)!")

    if st.session_state.indexed:
        st.info("Index is ready. Start chatting!")
    
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- Chat UI ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask something about your documents..."):
    if not st.session_state.indexed:
        st.warning("⚠️ Please upload and index documents first.")
    else:
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                results = st.session_state.rag.vectorstore.query(prompt, top_k=5)
                
                # Build context from results
                context = "\n\n".join(
                    [f"**Chunk {i+1}:** {r['metadata']['text']}" 
                     for i, r in enumerate(results) if r['metadata']]
                )
                
                response = f"### 🔍 Top Matching Chunks\n\n{context}"
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
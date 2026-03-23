import os
import tempfile
import streamlit as st
from src.chunker import chunk_pages
from src.pdf_processor import extract_pages
from src.rag_pipeline import RAGPipeline
from src.vector_store import VectorStore

st.set_page_config(page_title="Document Q&A", layout="wide")
st.title("Document Q&A")
st.caption("Upload up to 3 PDF documents, then ask questions about them.")

# Initializing Session State
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline(st.session_state.vector_store)

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# Sidebar document upload
with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files (max 3)",
        type="pdf",
        accept_multiple_files=True,
    )

    if uploaded_files:
        if len(uploaded_files) > 3:
            st.error("Please upload a maximum of 3 PDF files.")
        else:
            current_names = sorted(f.name for f in uploaded_files)

            # Only re-process when the uploaded file set changes
            if current_names != st.session_state.processed_files:
                with st.spinner("Processing documents…"):
                    st.session_state.vector_store.reset()
                    all_chunks = []

                    for uploaded_file in uploaded_files:
                        # Write to a temp file so pdfplumber can open it by path
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name

                        pages = extract_pages(tmp_path, display_name=uploaded_file.name)
                        all_chunks.extend(chunk_pages(pages))
                        os.unlink(tmp_path)

                    st.session_state.vector_store.add_chunks(all_chunks)
                    st.session_state.processed_files = current_names

                st.success(f"Indexed {len(uploaded_files)} document(s).")

            # Always show loaded documents
            st.markdown("**Loaded documents:**")
            for f in uploaded_files:
                st.markdown(f"- {f.name}")

# Main Q&A area
if st.session_state.vector_store.is_empty():
    st.info("Upload one or more PDF documents in the sidebar to get started.")
else:
    question = st.text_input(
        "Ask a question about your documents:",
        placeholder="e.g. What are the main conclusions?",
    )

    if question:
        with st.spinner("Searching for an answer…"):
            result = st.session_state.pipeline.answer(question)

        st.markdown("### Answer")
        st.write(result["answer"])

        st.markdown("### Sources")
        if result["citations"]:
            for citation in result["citations"]:
                st.markdown(f"- **{citation['source_file']}** — Page {citation['page_number']}")
        else:
            st.markdown("_No specific sources identified._")

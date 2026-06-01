from pathlib import Path

import streamlit as st
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM

from APP.chunking import chunk_documents, save_chunks_jsonl
from APP.pdf_loading import load_pdf
from APP.quality_gate import apply_quality_gate, save_chunks_jsonl as save_processed_jsonl
from APP.vector_store import build_hybrid_indices, hybrid_retrieve, expand_with_neighbors
from dotenv import load_dotenv
import os

load_dotenv() 

host = os.getenv("OLLAMA_HOST")


DEFAULT_MODEL = "llama3.1:8b"


def build_prompt(question: str, contexts: list[str]) -> str:
    context_blob = "\n\n".join(contexts)
    return f"""You are an expert document analyst and retrieval-augmented AI assistant.

Your task is to answer the user's question using ONLY the information provided in the retrieved document context.

INSTRUCTIONS:

1. Read ALL retrieved chunks carefully before answering.
2. Information may be distributed across multiple chunks.
3. Combine and synthesize information from different chunks whenever necessary.
4. If a principle, concept, case study, example, statistic, recommendation, or conclusion appears in separate chunks, connect them logically.
5. If the answer is partially available across multiple chunks, construct the most complete answer possible.
6. Prioritize factual accuracy over brevity.
7. Do NOT invent, assume, or hallucinate information that is not supported by the context.
8. If page numbers are available in the context metadata, mention them when relevant.
9. If the context contains enough evidence to reasonably infer the answer, provide the answer.
10. Only respond with "I cannot find this information in the document." when NONE of the retrieved context is relevant to the question.

ANSWERING RULES:

* For factual questions: provide a direct answer followed by supporting details.
* For explanatory questions: provide a concise explanation followed by evidence from the context.
* For comparison questions: compare all relevant information found across chunks.
* For summary questions: synthesize the key ideas from all relevant chunks.
* For analytical questions: connect related information across chunks and explain the relationship.

CONTEXT:
{context_blob}

QUESTION:
{question}

FINAL ANSWER:
"""


def process_uploaded_pdf(
    pdf_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    quality_threshold: float,
) -> tuple[object, object, list[Document], list[Document], dict]:
    pages = load_pdf(pdf_path)
    docs = [Document(page_content=p.page_content, metadata=p.metadata) for p in pages]
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    Path("chunks").mkdir(parents=True, exist_ok=True)
    save_chunks_jsonl(chunks, "chunks/chunks.jsonl")

    processed_chunks = apply_quality_gate(chunks, threshold_score=quality_threshold)
    save_processed_jsonl(processed_chunks, "chunks/chunks_processed.jsonl")
    passed_chunks = [c for c in processed_chunks if c.metadata.get("passed_gate") is True]
    retrieval_chunks = [c for c in processed_chunks if c.page_content.strip()]

    if not retrieval_chunks:
        raise RuntimeError("No retrievable chunks were produced. Inspect document extraction quality.")

    vectorstore, bm25 = build_hybrid_indices(retrieval_chunks)
    stats = {
        "pages": len(docs),
        "chunks": len(chunks),
        "passed_chunks": len(passed_chunks),
        "dropped_chunks": len(chunks) - len(passed_chunks),
        "indexed_chunks": len(retrieval_chunks),
    }
    return vectorstore, bm25, retrieval_chunks, processed_chunks, stats


def main() -> None:
    st.set_page_config(page_title="RAG PDF Chat", layout="wide")
    st.title("RAG PDF Chat")
    st.caption("Upload PDF -> chunk -> quality gate -> hybrid index -> chat. Evaluation is separate.")

    with st.sidebar:
        st.subheader("Pipeline Settings")
        model_name = st.text_input("Ollama model", value=DEFAULT_MODEL)
        chunk_size = st.slider("Chunk size", min_value=400, max_value=2000, value=1000, step=100)
        chunk_overlap = st.slider("Chunk overlap", min_value=50, max_value=400, value=150, step=25)
        quality_threshold = st.slider("Quality threshold", min_value=0.0, max_value=7.0, value=4.0, step=0.5)
        top_n = st.slider("Retrieved chunks", min_value=1, max_value=8, value=4)
        show_sources = st.checkbox("Show retrieved chunks", value=True)

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None:
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = data_dir / uploaded_file.name
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded: {uploaded_file.name}")

        if st.button("Process PDF and Build RAG Index", type="primary"):
            with st.spinner("Running PDF loading, chunking, quality gate, and indexing..."):
                try:
                    vectorstore, bm25, retrieval_chunks, all_chunks, stats = process_uploaded_pdf(
                        pdf_path=pdf_path,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        quality_threshold=quality_threshold,
                    )
                except Exception as exc:
                    st.error(f"Pipeline failed: {exc}")
                    return

            st.session_state.vectorstore = vectorstore
            st.session_state.bm25 = bm25
            st.session_state.retrieval_chunks = retrieval_chunks
            st.session_state.chunks = all_chunks
            st.session_state.pipeline_stats = stats
            st.session_state.chat_history = []
            st.success("RAG pipeline is ready for chat.")

    if "pipeline_stats" in st.session_state:
        s = st.session_state.pipeline_stats
        st.info(
            f"Pages: {s['pages']} | Chunks: {s['chunks']} | "
            f"Passed Gate: {s['passed_chunks']} | Failed Gate: {s['dropped_chunks']} | "
            f"Indexed: {s.get('indexed_chunks', s['passed_chunks'])}"
        )

    if "vectorstore" not in st.session_state:
        st.warning("Upload and process a PDF to start chatting.")
        return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

    question = st.chat_input("Ask a question about the uploaded PDF...")
    if not question:
        return

    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            try:
                results = hybrid_retrieve(
                    query=question,
                    vectorstore=st.session_state.vectorstore,
                    bm25=st.session_state.bm25,
                    chunks=st.session_state.get("retrieval_chunks", st.session_state.chunks),
                    top_n=top_n,
                )
                expanded_docs = expand_with_neighbors(
                    results=results,
                    chunks=st.session_state.chunks,
                    window=1,
                )
                contexts = []
                for doc in expanded_docs:
                    page = doc.metadata.get("page")
                    if page is not None:
                        contexts.append(f"[Page {page}] {doc.page_content}")
                    else:
                        contexts.append(doc.page_content)
                prompt = build_prompt(question, contexts)
                llm = OllamaLLM(model=model_name, temperature=0)
                answer = llm.invoke(prompt)
            except Exception as exc:
                st.error(f"RAG request failed: {exc}")
                return

        st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        if show_sources:
            with st.expander("Retrieved Chunks"):
                for idx, (doc, score) in enumerate(results, start=1):
                    page = doc.metadata.get("page", "?")
                    chunk_id = doc.metadata.get("chunk_id", "?")
                    st.markdown(
                        f"**#{idx}** | Score: `{score:.4f}` | Page: `{page}` | Chunk: `{chunk_id}`"
                    )
                    st.write(doc.page_content[:800] + ("..." if len(doc.page_content) > 800 else ""))
                    st.divider()


if __name__ == "__main__":
    main()

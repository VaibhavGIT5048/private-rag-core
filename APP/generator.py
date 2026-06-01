print("[DEBUG] Script started...") # This must print immediately

import os
import json
import pickle
import numpy as np
from pathlib import Path
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

print(" [DEBUG] Libraries loaded successfully.")

# ─────────────────────────────────────────────────────────────────────
# 1. RETRIEVAL LOGIC (Integrated to avoid import errors)
# ─────────────────────────────────────────────────────────────────────

def hybrid_retrieve_standalone(query, vectorstore, bm25, chunks, k=60, top_n=4):
    print(f"🔍 [DEBUG] Searching for: {query}")
    # Semantic
    semantic_results = vectorstore.similarity_search(query, k=10)
    # Keyword
    tokenized_query = query.lower().split()
    keyword_scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(keyword_scores)[::-1][:10]
    keyword_results = [chunks[i] for i in top_indices if keyword_scores[i] > 0]

    rrf_scores = {}
    doc_map = {}
    for rank, doc in enumerate(semantic_results, 1):
        cid = doc.metadata["chunk_id"]
        doc_map[cid] = doc
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank)
    for rank, doc in enumerate(keyword_results, 1):
        cid = doc.metadata["chunk_id"]
        doc_map[cid] = doc
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank)

    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[cid], score) for cid, score in sorted_results[:top_n]]

# ─────────────────────────────────────────────────────────────────────
# 2. PROMPT & LLM LOGIC
# ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert analyst. Answer ONLY using the provided <context>. 
Cite sources as [Page X]. If unknown, say the document does not contain the info."""

def run_rag_chat():
    print("🚩 [DEBUG] Entering run_rag_chat()...")
    
    # Check paths
    idx_path = Path("indexes/faiss_index")
    bm25_path = Path("indexes/bm25_data.pkl")
    
    if not idx_path.exists() or not bm25_path.exists():
        print(f"❌ Error: Index files not found in /indexes. Run vector_store.py first.")
        return

    # Load everything
    print("🔄 Loading indices into memory...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(str(idx_path), embeddings, allow_dangerous_deserialization=True)
    
    with open(bm25_path, "rb") as f:
        data = pickle.load(f)
        bm25, chunks = data["bm25"], data["chunks"]

    print("\n✅ RAG SYSTEM READY. Type 'exit' to quit.")
    
    llm = OllamaLLM(model="llama3", temperature=0)

    while True:
        query = input("\n👤 Question: ")
        if query.lower() in ["exit", "quit"]: break

        # Retrieve
        results = hybrid_retrieve_standalone(query, vectorstore, bm25, chunks)
        
        # Build Context
        context_text = "\n\n".join([f"--- Page {d.metadata['page']} ---\n{d.page_content}" for d, s in results])
        
        # Build Prompt
        full_prompt = f"{SYSTEM_PROMPT}\n\n<context>\n{context_text}\n</context>\n\nUser Question: {query}"
        
        print("🤖 Llama-3 is thinking...")
        answer = llm.invoke(full_prompt)
        print(f"\n📝 ANSWER:\n{answer}\n")

# ─────────────────────────────────────────────────────────────────────
# 3. GLOBAL EXECUTION (This ensures the script runs)
# ─────────────────────────────────────────────────────────────────────

print("🚩 [DEBUG] Main block reached.")
run_rag_chat()
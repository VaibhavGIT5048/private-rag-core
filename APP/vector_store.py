import os
import json
import pickle
import re
import numpy as np
from pathlib import Path
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def tokenize_for_bm25(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())

# ─────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────
def load_passed_chunks(path="chunks/chunks_processed.jsonl"):
    # Robust path checking: check current and parent dir
    potential_paths = [Path(path), Path("../") / path]
    target_path = None
    for p in potential_paths:
        if p.exists():
            target_path = p
            break
            
    if not target_path:
        print(f"❌ Error: Could not find {path}")
        return []

    processed_chunks = []
    print(f"🔄 Loading data from: {target_path}")
    with open(target_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            processed_chunks.append(Document(
                page_content=data["page_content"],
                metadata=data["metadata"]
            ))
    return processed_chunks

# ─────────────────────────────────────────────────────────────────────
# 2. BUILD INDICES
# ─────────────────────────────────────────────────────────────────────
def build_hybrid_indices(chunks):
    if not chunks:
        print("⚠️ No chunks to index. Skipping build.")
        return None, None

    # Create directory for indexes
    idx_dir = Path("indexes")
    idx_dir.mkdir(exist_ok=True)

    # A. Build FAISS (Semantic)
    print("🧠 Building FAISS Semantic Index (Dense)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(idx_dir / "faiss_index"))

    # B. Build BM25 (Keyword)
    print("📝 Building BM25 Keyword Index (Sparse)...")
    tokenized_corpus = [tokenize_for_bm25(doc.page_content) for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(idx_dir / "bm25_data.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
        
    print(f"✅ All indices built. Saved to: {idx_dir.absolute()}")
    return vectorstore, bm25

# ─────────────────────────────────────────────────────────────────────
# 3. HYBRID RETRIEVAL (RRF)
# ─────────────────────────────────────────────────────────────────────
def apply_quality_penalty(chunks, base_scores):
    penalised = []
    for chunk, score in zip(chunks, base_scores):
        gate = chunk.metadata.get("passed_gate", False)
        weight = 1.0 if gate else 0.7
        penalised.append((chunk, score * weight))
    return sorted(penalised, key=lambda x: x[1], reverse=True)


def hybrid_retrieve(query, vectorstore, bm25, chunks, k=60, top_n=3):
    candidate_k = max(top_n * 6, 20)
    faiss_weight = 0.4
    bm25_weight = 0.6

    # 1. Semantic Search
    semantic_results = vectorstore.similarity_search(query, k=min(candidate_k, len(chunks)))
    
    # 2. Keyword Search
    tokenized_query = tokenize_for_bm25(query)
    keyword_scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(keyword_scores)[::-1][:candidate_k]
    keyword_results = [chunks[i] for i in top_indices if keyword_scores[i] > 0]

    # 3. Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    doc_map = {}

    for rank, doc in enumerate(semantic_results, 1):
        cid = doc.metadata["chunk_id"]
        doc_map[cid] = doc
        rrf_scores[cid] = rrf_scores.get(cid, 0) + (faiss_weight / (k + rank))

    for rank, doc in enumerate(keyword_results, 1):
        cid = doc.metadata["chunk_id"]
        doc_map[cid] = doc
        rrf_scores[cid] = rrf_scores.get(cid, 0) + (bm25_weight / (k + rank))

    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    base_results = [(doc_map[cid], score) for cid, score in sorted_results]
    penalised = apply_quality_penalty(
        chunks=[doc for doc, _ in base_results],
        base_scores=[score for _, score in base_results],
    )
    return penalised[:top_n]


def expand_with_neighbors(
    results: list[tuple[Document, float]],
    chunks: list[Document],
    window: int = 1,
) -> list[Document]:
    id_to_doc = {doc.metadata["chunk_id"]: doc for doc in chunks}
    expanded_ids: list[int] = []

    for doc, _ in results:
        cid = doc.metadata.get("chunk_id")
        if cid is None:
            continue
        for neighbor_id in range(cid - window, cid + window + 1):
            if neighbor_id in id_to_doc:
                expanded_ids.append(neighbor_id)

    seen: set[int] = set()
    expanded_docs: list[Document] = []
    for cid in expanded_ids:
        if cid not in seen:
            expanded_docs.append(id_to_doc[cid])
            seen.add(cid)

    return expanded_docs

# ─────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 STARTING PHASE 5: VECTOR STORE & HYBRID SEARCH")
    print("="*50)
    
    # 1. Load Data
    all_passed = [c for c in load_passed_chunks() if c.page_content.strip()]
    print(f"📥 Found {len(all_passed)} chunks to index.")
    
    if all_passed:
        # 2. Build
        vs, bm = build_hybrid_indices(all_passed)

        # 3. Test Retrieval
        test_query = input("\nEnter a test query to retrieve relevant chunks: ")
        print(f"\n🔍 Testing Hybrid Search: '{test_query}'")
        
        results = hybrid_retrieve(test_query, vs, bm, all_passed)
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n[{i}] RRF Score: {score:.4f} | Page: {doc.metadata.get('page')}")
            print(f"Content: {doc.page_content[:150]}...")
    else:
        print("❌ Build stopped: No data loaded. Check chunks/chunks_processed.jsonl")

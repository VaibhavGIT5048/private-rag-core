import json
from pathlib import Path
import numpy as np
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_filtered_chunks(input_path: str, passed_only: bool = False) -> list[Document]:
    passed_chunks = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if passed_only and record["metadata"].get("passed_gate") is not True:
                continue
            passed_chunks.append(Document(
                page_content=record["page_content"],
                metadata=record["metadata"]
            ))

    if passed_only:
        print(f"📥 Loaded {len(passed_chunks)} high-quality chunks from {input_path}")
    else:
        print(f"📥 Loaded {len(passed_chunks)} chunks from {input_path}")
    return passed_chunks


def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    print(f"🔄 Loading embedding model: {model_name}...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print(f"✅ Model loaded (384 dimensions)")
    return embedding_model


def generate_embeddings(chunks: list[Document], model: HuggingFaceEmbeddings):
    print(f"\n🔄 Generating vectors for {len(chunks)} chunks...")
    texts = [chunk.page_content for chunk in chunks]
    vectors = model.embed_documents(texts)
    vectors_np = np.array(vectors)
    print(f"✅ Created embedding matrix of shape: {vectors_np.shape}")
    return vectors


if __name__ == "__main__":
    INPUT_FILE = "chunks/chunks_processed.jsonl"

    high_quality_chunks = load_filtered_chunks(INPUT_FILE)

    if not high_quality_chunks:
        print("❌ No passed chunks found. Check your quality gate thresholds.")
    else:
        emb_model = load_embedding_model()
        chunk_vectors = generate_embeddings(high_quality_chunks, emb_model)

        print("\n--- 🔍 Quick Retrieval Test ---")
        query = "What are the growth drivers for tourism in Asia?"
        query_vector = emb_model.embed_query(query)

        for i in range(min(5, len(chunk_vectors))):
            sim = np.dot(query_vector, chunk_vectors[i])
            print(f"Chunk[{i}] Similarity: {sim:.4f}")
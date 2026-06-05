import argparse
import json
import re
from pathlib import Path
import tiktoken
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION & INITIALIZATION
# ─────────────────────────────────────────────────────────────────────
load_dotenv()

tokenizer = tiktoken.get_encoding("cl100k_base")
embedder = SentenceTransformer("intfloat/e5-base-v2")

W_TOKENS, W_PUNC, W_ENTITIES, W_OVERLAP = 2, 2, 1, 2
MAX_SCORE = 7
TOKEN_MIN, TOKEN_MAX = 50, 300
URL_COUNT_MAX, URL_CHAR_RATIO_MAX = 2, 0.15
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
GOOD_END = {".", "!", "?"}
WEAK_END = {";"}


def _content_richness(text: str) -> float:
    """Fraction of characters that are alphabetic."""
    return sum(c.isalpha() for c in text) / max(len(text), 1)


def _same_section(prev: Document, curr: Document) -> bool:
    return (
        prev.metadata.get("page") == curr.metadata.get("page")
        or prev.metadata.get("section") == curr.metadata.get("section")
    )

# ─────────────────────────────────────────────────────────────────────
# THE REFINED QUALITY EVALUATOR
# ─────────────────────────────────────────────────────────────────────
def evaluate_chunk_refined(current_chunk: Document, previous_chunk: Document | None) -> dict:
    text = current_chunk.page_content.strip()
    score = 0.0
    features = {"X_T": 0, "X_P": 0, "X_E": 0, "X_C": 0}

    # --- URL FILTER ---
    url_matches = URL_PATTERN.findall(text)
    url_count = len(url_matches)
    url_char_ratio = sum(len(u) for u in url_matches) / max(len(text), 1)
    
    if url_count > URL_COUNT_MAX or url_char_ratio > URL_CHAR_RATIO_MAX:
        return {"total_score": 0.0, "features": features, "metrics": {"url_filtered": True}}

    # 1. X_T (Token Count)
    token_count = len(tokenizer.encode(text))
    if TOKEN_MIN <= token_count <= TOKEN_MAX:
        features["X_T"] = 1
        score += (1 * W_TOKENS)

    # 2. X_P (Refined Punctuation)
    clean_text = text.rstrip()
    starts_mid = bool(clean_text) and clean_text[0].islower()
    ends_good = clean_text[-1] in GOOD_END if clean_text else False
    ends_weak = clean_text[-1] in WEAK_END if clean_text else False

    if ends_good and not starts_mid:
        features["X_P"] = 1
        score += (1 * W_PUNC)
    elif ends_weak or (ends_good and starts_mid):
        features["X_P"] = 0.5
        score += (0.5 * W_PUNC)

    # 3. X_E (Content Richness)
    content_richness = _content_richness(text)
    if content_richness >= 0.45:
        features["X_E"] = 1
        score += (1 * W_ENTITIES)

    # 4. X_C (Refined Overlap)
    overlap_sim = 0.0
    if previous_chunk is None or not _same_section(previous_chunk, current_chunk):
        features["X_C"] = 1
        score += (1 * W_OVERLAP)
    else:
        embeddings = embedder.encode([previous_chunk.page_content, text], normalize_embeddings=True)
        # FIX: Explicitly cast the numpy float32 to a standard Python float
        overlap_sim = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
        
        if 0.05 <= overlap_sim <= 0.95:
            features["X_C"] = 1
            score += (1 * W_OVERLAP)

    return {
        "total_score": round(float(score), 2),
        "features_passed": features,
        "metrics": {
            "token_count": token_count,
            "content_richness": round(float(content_richness), 3),
            "overlap_similarity": round(overlap_sim, 3)
        }
    }

# ─────────────────────────────────────────────────────────────────────
# QUALITY GATE EXECUTION
# ─────────────────────────────────────────────────────────────────────
def apply_quality_gate(chunks: list[Document], threshold_score: float = 4.0):
    processed_chunks = []
    passed_count = 0
    
    print(f"\n{'═'*60}\n RUNNING REFINED QUALITY GATE (Threshold: {threshold_score}/{MAX_SCORE})\n{'═'*60}")

    previous_chunk = None
    for chunk in chunks:
        evaluation = evaluate_chunk_refined(chunk, previous_chunk)
        
        # Tag every chunk with its status
        chunk.metadata["strict_score"] = evaluation["total_score"]
        chunk.metadata["passed_gate"] = evaluation["total_score"] >= threshold_score
        chunk.metadata["gate_metrics"] = evaluation.get("metrics", {})
        
        if chunk.metadata["passed_gate"]:
            passed_count += 1
            
        processed_chunks.append(chunk)
        previous_chunk = chunk

    print(f"Total Chunks Processed : {len(chunks)}")
    print(f"Passed (Status: True)  : {passed_count} chunks")
    print(f"Dropped (Status: False): {len(chunks) - passed_count} chunks")
    return processed_chunks

def load_chunks_jsonl(path: str) -> list[Document]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            chunks.append(Document(page_content=record["page_content"], metadata=record["metadata"]))
    return chunks

def save_chunks_jsonl(chunks: list[Document], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            # All values in metadata are now standard Python types and serializable
            f.write(json.dumps({"page_content": chunk.page_content, "metadata": chunk.metadata}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="chunks/chunks.jsonl")
    parser.add_argument("--output", default="chunks/chunks_processed.jsonl")
    parser.add_argument("--threshold", type=float, default=4.0)
    args = parser.parse_args()

    all_chunks = load_chunks_jsonl(args.input)
    processed = apply_quality_gate(all_chunks, threshold_score=args.threshold)
    save_chunks_jsonl(processed, args.output)
    print(f"Saved ALL chunks to {args.output}")

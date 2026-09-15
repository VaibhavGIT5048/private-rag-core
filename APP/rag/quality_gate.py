import argparse
import json
import re
from pathlib import Path
import tiktoken
from dotenv import load_dotenv
from langchain_core.documents import Document

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION & INITIALIZATION
# ─────────────────────────────────────────────────────────────────────
load_dotenv()

tokenizer = tiktoken.get_encoding("cl100k_base")

# Lazy embedding provider — the same self-hosted default (bge-m3) the rest of
# the stack embeds with, so a chunk scored here is measured against vectors
# comparable to the ones actually indexed. Only reached when apply_quality_gate
# is called WITHOUT precomputed vectors; the /ingest path passes them in, so
# this normally never loads at all.
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from APP.providers.embedding import build_default_embedding_provider
        _embedder = build_default_embedding_provider()
    return _embedder


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


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity — no sklearn/numpy required."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ─────────────────────────────────────────────────────────────────────
# THE REFINED QUALITY EVALUATOR
# ─────────────────────────────────────────────────────────────────────
def token_window_for(chunks: list[Document]) -> tuple[int, int]:
    """Token window scaled to this document's own median chunk size.

    A fixed 50-300 window silently encodes an assumption that chunks are
    uniform, which is only true for fixed-size splitting: measured on a
    44-page PDF, recursive chunks scored 80% against semantic chunks' 64%
    purely because variable-size chunks fall outside a static window more
    often. That penalised the chunker, not the chunks. Scaling to the median
    keeps "is this chunk anomalously short or long *for this document*",
    which is the thing actually worth scoring, and works for heading-bounded
    sections whose size varies by design.

    Falls back to the original constants when there's too little to measure.
    """
    counts = sorted(len(tokenizer.encode(c.page_content.strip())) for c in chunks if c.page_content.strip())
    if len(counts) < 5:
        return TOKEN_MIN, TOKEN_MAX
    median = counts[len(counts) // 2]
    if median <= 0:
        return TOKEN_MIN, TOKEN_MAX
    # Generous band: a chunk is only penalised when well outside the document's
    # own norm, not for ordinary variation.
    return max(20, int(median * 0.25)), max(int(median * 3), 60)


def evaluate_chunk_refined(
    current_chunk: Document,
    previous_chunk: Document | None,
    current_vector: list[float] | None = None,
    previous_vector: list[float] | None = None,
    token_window: tuple[int, int] | None = None,
) -> dict:
    text = current_chunk.page_content.strip()
    score = 0.0
    features = {"X_T": 0, "X_P": 0, "X_E": 0, "X_C": 0}

    # --- URL FILTER ---
    url_matches = URL_PATTERN.findall(text)
    url_count = len(url_matches)
    url_char_ratio = sum(len(u) for u in url_matches) / max(len(text), 1)

    if url_count > URL_COUNT_MAX or url_char_ratio > URL_CHAR_RATIO_MAX:
        # Key must match the normal return path below, or callers reading
        # features_passed silently get nothing for URL-filtered chunks.
        return {"total_score": 0.0, "features_passed": features, "metrics": {"url_filtered": True}}

    # 1. X_T (Token Count) — window is per-document when supplied, so the
    # score reflects "unusual for this document" rather than "unusual for a
    # fixed-size splitter".
    lo, hi = token_window if token_window is not None else (TOKEN_MIN, TOKEN_MAX)
    token_count = len(tokenizer.encode(text))
    if lo <= token_count <= hi:
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

    # 4. X_C (Refined Overlap) — reuses precomputed vectors when the caller
    # has them (the embedding-pass consolidation: chunk embeddings are
    # computed once up front and threaded through here and into indexing,
    # instead of this function re-embedding every adjacent pair itself).
    # Falls back to embedding on demand if no vector was supplied, so
    # standalone callers (the CLI entry point below) keep working unchanged.
    overlap_sim = 0.0
    if previous_chunk is None or not _same_section(previous_chunk, current_chunk):
        features["X_C"] = 1
        score += (1 * W_OVERLAP)
    else:
        if current_vector is not None and previous_vector is not None:
            vec_current, vec_previous = current_vector, previous_vector
        else:
            embedder = _get_embedder()
            vecs = embedder.embed_documents([previous_chunk.page_content, text])
            vec_previous, vec_current = vecs[0], vecs[1]
        overlap_sim = _cosine_sim(vec_previous, vec_current)

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
def apply_quality_gate(chunks: list[Document], threshold_score: float = 4.0, vectors: list[list[float]] | None = None):
    """`vectors`, if given, must be aligned index-for-index with `chunks` —
    computed once by the caller instead of re-embedded here per adjacent pair.
    """
    processed_chunks = []
    passed_count = 0

    token_window = token_window_for(chunks)
    print(f"\n{'═'*60}\n RUNNING REFINED QUALITY GATE (Threshold: {threshold_score}/{MAX_SCORE})\n{'═'*60}")
    print(f"Token window (per-doc)  : {token_window[0]}-{token_window[1]} tokens")

    previous_chunk = None
    previous_vector = None
    for i, chunk in enumerate(chunks):
        current_vector = vectors[i] if vectors is not None else None
        evaluation = evaluate_chunk_refined(chunk, previous_chunk, current_vector, previous_vector, token_window)

        chunk.metadata["strict_score"] = evaluation["total_score"]
        chunk.metadata["passed_gate"] = evaluation["total_score"] >= threshold_score
        chunk.metadata["gate_metrics"] = evaluation.get("metrics", {})

        if chunk.metadata["passed_gate"]:
            passed_count += 1

        processed_chunks.append(chunk)
        previous_chunk = chunk
        previous_vector = current_vector

    print(f"Total Chunks Processed : {len(chunks)}")
    print(f"Passed (Status: True)  : {passed_count} chunks")
    # Not dropped — every non-empty chunk is still indexed. Failing the gate
    # only applies a 0.7 ranking multiplier in apply_quality_penalty().
    print(f"Rank-penalised (0.7x)  : {len(chunks) - passed_count} chunks")
    return processed_chunks


def save_chunks_jsonl(chunks: list[Document], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps({
                "page_content": chunk.page_content,
                "metadata": chunk.metadata
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/chunks/chunks.jsonl")
    parser.add_argument("--output", default="data/chunks/chunks_processed.jsonl")
    parser.add_argument("--threshold", type=float, default=4.0)
    args = parser.parse_args()

    all_chunks = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            all_chunks.append(Document(
                page_content=record["page_content"],
                metadata=record["metadata"]
            ))

    processed = apply_quality_gate(all_chunks, threshold_score=args.threshold)
    save_chunks_jsonl(processed, args.output)
    print(f"Saved ALL chunks to {args.output}")
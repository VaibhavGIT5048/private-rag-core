import os
import argparse
import asyncio
import json
import random
import re
import pickle
import sys
import hashlib
from pathlib import Path
import pandas as pd
import time
from openai import AsyncOpenAI

def experiment(*args, **kwargs):
    return lambda fn: fn

# Ensure project root is on sys.path when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ─────────────────────────────────────────────────────────────────────
# 1. INITIALIZATION
# ─────────────────────────────────────────────────────────────────────
print("🚀 Initializing High-Performance Evaluation Engine...")

def _default_concurrency() -> int:
    cpu = os.cpu_count() or 1
    return max(1, cpu)

# Ollama OpenAI-compatible endpoint (best speed + structured eval support)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
STUDENT_MODEL = os.getenv("OLLAMA_STUDENT_MODEL", "llama3.1:8b")
JUDGE_MODEL = os.getenv("OLLAMA_JUDGE_MODEL", "mistral")
GENERATOR_MODEL = os.getenv("OLLAMA_GENERATOR_MODEL", STUDENT_MODEL)
EVAL_CONCURRENCY = int(os.getenv("EVAL_CONCURRENCY", "10"))
GEN_CONCURRENCY = int(os.getenv("GEN_CONCURRENCY", "10"))
JUDGE_CACHE_PATH = os.getenv("JUDGE_CACHE_PATH", "evals/judge_cache.json")

if JUDGE_MODEL == STUDENT_MODEL:
    print(
        f"⚠️  WARNING: JUDGE_MODEL and STUDENT_MODEL are both '{JUDGE_MODEL}'. "
        "A model judging its own answers inflates faithfulness scores. "
        "Set OLLAMA_JUDGE_MODEL to a different model (e.g. mistral) in .env"
    )

# Student LLM (Lower temperature for Faithfulness)
llm = OllamaLLM(model=STUDENT_MODEL, temperature=0)

# Judge LLM (async-capable)
judge_client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

emb_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load Indices
vs = FAISS.load_local("indexes/faiss_index", emb_model, allow_dangerous_deserialization=True)
with open("indexes/bm25_data.pkl", "rb") as f:
    bm_data = pickle.load(f)
    bm25, chunks_ref = bm_data["bm25"], bm_data["chunks"]

# ─────────────────────────────────────────────────────────────────────
# 2. JUDGE PROMPT (FAST + ROBUST PARSING)
# ─────────────────────────────────────────────────────────────────────
REFUSAL_PATTERN = re.compile(
    r"\b(cannot find|not in the document|not provided|not available"
    r"|cannot locate|don't have information|no information"
    r"|outside the scope|not mentioned|not covered"
    r"|based on the (provided |given )?context|cannot answer)\b",
    re.IGNORECASE,
)

JUDGE_SYSTEM = "You are a strict evaluator. Return only the JSON object requested."

JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

_judge_cache: dict[str, dict] = {}
_judge_cache_lock = asyncio.Lock()

RAGAS_EVAL_PROMPT = """You are an expert evaluation assistant for Retrieval-Augmented Generation (RAG) pipelines.
Your task is to rigorously evaluate the quality of a RAG system across ALL RAGAS metrics.

You will be provided with:
- `document` (context): The retrieved context/chunk passed to the RAG system
- `question`: The user query
- `answer`: The RAG system's generated answer (if applicable)

---

## TASK: OUT-OF-SCOPE QUESTION GENERATION

Generate exactly ONE question that satisfies ALL of the following criteria:

1. **Out of Scope**: The question CANNOT be answered using information found anywhere in the provided document/context.
2. **Answerable in General**: The question has a clear, correct answer in the real world or from general knowledge - it is NOT unanswerable by nature.
3. **Topically Plausible**: The question should appear related to the document's domain or subject area, so it feels like a natural but unsupported query.
4. **Non-Trivial**: Avoid yes/no questions or overly simple factual questions. Prefer questions that require explanation, comparison, or specific details not covered in the document.

---

## RAGAS EVALUATION DIMENSIONS

After generating the out-of-scope question, evaluate the provided `question`, `document`, and `answer` across the following RAGAS metrics. For each metric, assign a score between 0.0 and 1.0 and provide a brief justification.

### 1. **Faithfulness**
- Does the generated `answer` contain ONLY claims that are directly supported by the `document`?
- Penalize hallucinations, fabrications, or unsupported assertions.
- Score: 1.0 = fully grounded, 0.0 = completely hallucinated.

### 2. **Answer Relevance**
- Does the `answer` directly and completely address the `question`?
- Penalize answers that are vague, off-topic, or only partially responsive.
- Score: 1.0 = fully relevant and complete, 0.0 = irrelevant.

### 3. **Context Precision**
- Does the retrieved `document` contain information that is specifically useful for answering the `question`?
- Penalize retrieval of generic or loosely related context that does not directly support the answer.
- Score: 1.0 = highly precise context, 0.0 = context is irrelevant to the question.

### 4. **Context Recall**
- Does the `document` contain ALL the information needed to fully answer the `question`?
- Penalize if critical pieces of information are missing from the retrieved context.
- Score: 1.0 = all necessary info is present, 0.0 = key information is missing.

### 5. **Context Entity Recall**
- Are all key named entities (people, places, dates, organizations, technical terms) required for a complete answer present in the `document`?
- Score: 1.0 = all required entities present, 0.0 = no required entities present.

### 6. **Answer Semantic Similarity**
- How semantically similar is the generated `answer` to the ideal/reference answer (if provided)?
- Consider meaning, intent, and coverage - not just surface-level wording.
- Score: 1.0 = semantically identical, 0.0 = completely different meaning.

### 7. **Answer Correctness**
- Is the factual content of the `answer` accurate and correct based on the `document` and/or general knowledge?
- Penalize factual errors, contradictions, or misleading statements.
- Score: 1.0 = fully correct, 0.0 = completely incorrect.

---

## OUTPUT FORMAT

Return a single JSON object with the following structure:

{
    "out_of_scope_question": {
        "question": "<Your generated out-of-scope question>",
        "ground_truth": "The answer to this question is not present in the provided document. However, the correct general answer is: <real-world answer>"
    },
    "ragas_evaluation": {
        "faithfulness": {
            "score": <0.0 - 1.0>,
            "justification": "<brief explanation>"
        },
        "answer_relevance": {
            "score": <0.0 - 1.0>,
            "justification": "<brief explanation>"
        },
        "context_precision": {
            "score": <0.0 - 1.0>,
            "justification": "<brief explanation>"
        },
        "context_recall": {
            "score": <0.0 - 1.0>,
            "justification": "<brief explanation>"
        },
        "context_entity_recall": {
            "score": <0.0 - 1.0>,
            "justification": "<brief explanation>"
        },
        "answer_semantic_similarity": {
            "score": <0.0 - 1.0>,
            "justification": "<brief explanation>"
        },
        "answer_correctness": {
            "score": <0.0 - 1.0>,
            "justification": "<brief explanation>"
        }
    },
    "overall_rag_quality_score": <weighted average of all scores, 0.0 - 1.0>
}

Return ONLY the JSON object. No preamble, no markdown fences, no extra commentary.
"""

def build_ragas_eval_prompt(document: str, question: str, answer: str) -> str:
        return f"""{RAGAS_EVAL_PROMPT}

document:
{document}

question:
{question}

answer:
{answer}
"""

def coerce_score(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def extract_json_object(text: str) -> dict | None:
    match = JSON_OBJECT_PATTERN.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

def _load_judge_cache(path: str) -> dict[str, dict]:
    cache_file = Path(path)
    if not cache_file.exists():
        return {}
    try:
        return json.loads(cache_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

def _save_judge_cache(path: str, cache: dict[str, dict]) -> None:
    cache_file = Path(path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

def _judge_cache_key(question: str, response: str, contexts: list[str]) -> str:
    payload = {"question": question, "answer": response, "contexts": contexts}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

async def judge_ragas_eval(question: str, response: str, contexts: list[str]) -> tuple[dict | None, str]:
    document = "\n\n".join(contexts)
    prompt = build_ragas_eval_prompt(document=document, question=question, answer=response)
    completion = await judge_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=700,
    )
    content = completion.choices[0].message.content or ""
    payload = extract_json_object(content)
    return payload, content.strip()

async def judge_ragas_eval_cached(question: str, response: str, contexts: list[str]) -> tuple[dict | None, str]:
    key = _judge_cache_key(question, response, contexts)
    cached = _judge_cache.get(key)
    if cached:
        return cached.get("payload"), cached.get("raw", "")
    payload, raw = await judge_ragas_eval(question=question, response=response, contexts=contexts)
    async with _judge_cache_lock:
        _judge_cache[key] = {"payload": payload, "raw": raw}
    return payload, raw

def verdict_from_ragas(payload: dict | None) -> tuple[str, str]:
    if not payload:
        return "Irrelevant", "No valid evaluation JSON returned."
    metrics = payload.get("ragas_evaluation", {}) or {}
    faithfulness = metrics.get("faithfulness", {}) or {}
    answer_relevance = metrics.get("answer_relevance", {}) or {}
    faith_score = coerce_score(faithfulness.get("score"))
    relevance_score = coerce_score(answer_relevance.get("score"))

    if faith_score is not None and faith_score < 0.5:
        return "Hallucinated", faithfulness.get("justification", "")
    if relevance_score is not None and relevance_score < 0.5:
        return "Irrelevant", answer_relevance.get("justification", "")
    return "Excellent", answer_relevance.get("justification", "") or faithfulness.get("justification", "")

# ─────────────────────────────────────────────────────────────────────
# 3. DATASET GENERATION FROM CHUNKS
# ─────────────────────────────────────────────────────────────────────
QA_JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
DOC_SCOPE = "the provided document"

QA_TEMPLATES = [
    """Generate ONE specific factual question answerable from this context.
Return a JSON object with keys: question, ground_truth.

Context:
{context}

JSON:""",
    """Generate ONE question asking about a specific number, percentage,
or statistic from this context. If no numbers exist, generate a factual question.
Return a JSON object with keys: question, ground_truth.

Context:
{context}

JSON:""",
    """Generate ONE question asking what something means or how something
works based on this context.
Return a JSON object with keys: question, ground_truth.

Context:
{context}

JSON:""",
    """Generate ONE question comparing two things or asking about
a relationship between concepts in this context.
Return a JSON object with keys: question, ground_truth.

Context:
{context}

JSON:""",
]

OOD_PROMPT_TEMPLATE = """Generate ONE question that is OUT OF SCOPE for {doc_scope}.
The question should be answerable in general, but not from {doc_scope}.
Return a JSON object with keys: question, ground_truth.

Use ground_truth to say the answer is not in the document.

JSON:"""

def load_chunks_jsonl(path: str) -> list[str]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            content = record.get("page_content", "")
            if content:
                chunks.append(content)
    return chunks

async def generate_qa_from_chunk(chunk_text: str, q_type: int = 0) -> dict | None:
    template = QA_TEMPLATES[q_type % len(QA_TEMPLATES)]
    prompt = template.format(context=chunk_text)
    completion = await judge_client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[
            {"role": "system", "content": "You generate concise QA pairs from context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=200,
    )
    content = completion.choices[0].message.content or ""
    match = QA_JSON_PATTERN.search(content)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not payload.get("question") or not payload.get("ground_truth"):
        return None
    return {
        "question": payload["question"].strip(),
        "ground_truth": payload["ground_truth"].strip(),
        "ood": False,
        "gold_context": chunk_text.strip(),
    }

async def generate_ood_question() -> dict | None:
    prompt = OOD_PROMPT_TEMPLATE.format(doc_scope=DOC_SCOPE)
    completion = await judge_client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[
            {"role": "system", "content": "You create out-of-scope questions."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=200,
    )
    content = completion.choices[0].message.content or ""
    match = QA_JSON_PATTERN.search(content)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not payload.get("question"):
        return None
    return {
        "question": payload["question"].strip(),
        "ground_truth": "The document does not contain this information.",
        "ood": True,
    }

def coverage_sample(chunks: list[str], n: int, seed: int | None = None) -> list[str]:
    if not chunks or n <= 0:
        return []
    n = min(n, len(chunks))
    if seed is not None:
        random.seed(seed)
    step = max(len(chunks) / n, 1)
    picks = []
    for i in range(n):
        start = int(i * step)
        end = int(min((i + 1) * step, len(chunks)))
        if end <= start:
            end = min(start + 1, len(chunks))
        segment = chunks[start:end]
        picks.append(random.choice(segment))
    return picks

async def build_dataset_from_chunks(
    chunks_path: str,
    num_questions: int,
    seed: int,
    max_chars: int,
    ood_ratio: float,
) -> list[dict]:
    chunks = load_chunks_jsonl(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")

    ood_ratio = max(0.0, min(ood_ratio, 1.0))
    ood_count = int(round(num_questions * ood_ratio))
    in_scope_count = max(num_questions - ood_count, 0)

    sample = coverage_sample(chunks, in_scope_count, seed=None)

    async def _generate(chunk: str, q_type: int) -> dict | None:
        async with gen_sem:
            return await generate_qa_from_chunk(chunk[:max_chars], q_type=q_type)

    tasks = [_generate(chunk, i % len(QA_TEMPLATES)) for i, chunk in enumerate(sample)]
    results = await asyncio.gather(*tasks)
    dataset = [r for r in results if r is not None]

    if ood_count > 0:
        ood_tasks = [generate_ood_question() for _ in range(ood_count)]
        ood_results = await asyncio.gather(*ood_tasks)
        dataset.extend([r for r in ood_results if r is not None])

    random.shuffle(dataset)
    return dataset

# ─────────────────────────────────────────────────────────────────────
# 4. METRIC HELPERS
# ─────────────────────────────────────────────────────────────────────
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()

def exact_match(pred: str, truth: str) -> int:
    return 1 if normalize_text(pred) == normalize_text(truth) else 0

def token_f1(pred: str, truth: str) -> float:
    pred_tokens = re.findall(r"\w+", pred.lower())
    truth_tokens = re.findall(r"\w+", truth.lower())
    if not pred_tokens or not truth_tokens:
        return 0.0
    pred_counts = {}
    truth_counts = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    for t in truth_tokens:
        truth_counts[t] = truth_counts.get(t, 0) + 1
    overlap = 0
    for t, c in pred_counts.items():
        overlap += min(c, truth_counts.get(t, 0))
    precision = overlap / len(pred_tokens)
    recall = overlap / len(truth_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def retrieval_metrics(gold_context: str | None, retrieved_texts: list[str]) -> dict:
    if not gold_context:
        return {
            "retrieval_hit_rank": None,
            "retrieval_mrr": None,
            "context_precision": None,
            "context_recall": None,
        }
    gold_norm = normalize_text(gold_context)
    hit_rank = None
    for idx, ctx in enumerate(retrieved_texts, start=1):
        ctx_norm = normalize_text(ctx)
        if gold_norm in ctx_norm or ctx_norm in gold_norm:
            hit_rank = idx
            break
    k = max(len(retrieved_texts), 1)
    if hit_rank is None:
        return {
            "retrieval_hit_rank": None,
            "retrieval_mrr": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }
    return {
        "retrieval_hit_rank": hit_rank,
        "retrieval_mrr": 1.0 / hit_rank,
        "context_precision": 1.0 / k,
        "context_recall": 1.0,
    }

# ─────────────────────────────────────────────────────────────────────
# 5. FIXED RAG STUDENT (Fixed Tuple Error + Logic for Relevancy)
# ─────────────────────────────────────────────────────────────────────
def get_student_response(question: str) -> tuple[str, list[str], float]:
    from APP.vector_store import hybrid_retrieve
    # top_n=6 gives more evidence to increase Faithfulness
    start = time.time()
    docs_with_scores = hybrid_retrieve(question, vs, bm25, chunks_ref, top_n=6)
    retrieval_ms = (time.time() - start) * 1000.0
    
    # FIXED: Correctly unpacking the (Document, Score) tuple
    contexts = [doc.page_content for doc, score in docs_with_scores]
    
    # PROMPT TUNING: Forcing strictness to increase Relevancy/Faithfulness
    prompt = f"""You are an expert document analyst and retrieval-augmented AI assistant.

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
    {" ".join(contexts)}

    QUESTION: {question}

    FINAL ANSWER:"""
    
    answer = llm.invoke(prompt)
    return answer, contexts, retrieval_ms

# ─────────────────────────────────────────────────────────────────────
# 6. ASYNC EXPERIMENT WITH PARALLEL LIMIT (3 Workers)
# ─────────────────────────────────────────────────────────────────────
# Semaphores ensure we never exceed configured Ollama concurrency
eval_sem = asyncio.Semaphore(EVAL_CONCURRENCY)
gen_sem = asyncio.Semaphore(GEN_CONCURRENCY)

async def run_eval_experiment(row):
    async with eval_sem:
        # Step 1: Run RAG
        answer, contexts, retrieval_ms = await asyncio.to_thread(
            get_student_response,
            row["question"],
        )
        retrieved_texts = contexts
        gold_context = row.get("gold_context")
        ood = row.get("ood", False)
        rm = retrieval_metrics(None if ood else gold_context, retrieved_texts)
        
        # Step 2: Run Judge
        ragas_payload, ragas_raw = await judge_ragas_eval_cached(
            question=row["question"],
            response=answer,
            contexts=contexts,
        )
        verdict, reason = verdict_from_ragas(ragas_payload)

        ragas_eval = (ragas_payload or {}).get("ragas_evaluation", {}) or {}
        ragas_overall = coerce_score((ragas_payload or {}).get("overall_rag_quality_score"))

        def _metric_score(key: str) -> float | None:
            return coerce_score((ragas_eval.get(key, {}) or {}).get("score"))

        def _metric_justification(key: str) -> str | None:
            return (ragas_eval.get(key, {}) or {}).get("justification")

        em = exact_match(answer, row["ground_truth"]) if not ood else None
        f1 = token_f1(answer, row["ground_truth"]) if not ood else None
        refused = bool(REFUSAL_PATTERN.search(answer)) if ood else None
        
        return {
            "question": row["question"],
            "answer": answer,
            "verdict": verdict,
            "reason": reason,
            "score_numeric": 1 if verdict == "Excellent" else 0,
            "retrieval_ms": round(retrieval_ms, 2),
            "ood": ood,
            "gold_context": gold_context,
            "ragas_overall": ragas_overall,
            "ragas_faithfulness": _metric_score("faithfulness"),
            "ragas_answer_relevance": _metric_score("answer_relevance"),
            "ragas_context_precision": _metric_score("context_precision"),
            "ragas_context_recall": _metric_score("context_recall"),
            "ragas_context_entity_recall": _metric_score("context_entity_recall"),
            "ragas_answer_semantic_similarity": _metric_score("answer_semantic_similarity"),
            "ragas_answer_correctness": _metric_score("answer_correctness"),
            "ragas_faithfulness_justification": _metric_justification("faithfulness"),
            "ragas_answer_relevance_justification": _metric_justification("answer_relevance"),
            "ragas_context_precision_justification": _metric_justification("context_precision"),
            "ragas_context_recall_justification": _metric_justification("context_recall"),
            "ragas_context_entity_recall_justification": _metric_justification("context_entity_recall"),
            "ragas_answer_semantic_similarity_justification": _metric_justification("answer_semantic_similarity"),
            "ragas_answer_correctness_justification": _metric_justification("answer_correctness"),
            "ragas_raw": ragas_raw,
            "retrieval_hit_rank": rm["retrieval_hit_rank"],
            "retrieval_mrr": rm["retrieval_mrr"],
            "context_precision": rm["context_precision"],
            "context_recall": rm["context_recall"],
            "exact_match": em,
            "f1": f1,
            "refused": refused,
        }

# ─────────────────────────────────────────────────────────────────────
# 7. MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────
async def main():
    global _judge_cache
    _judge_cache = _load_judge_cache(JUDGE_CACHE_PATH)
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="chunks/chunks.jsonl")
    parser.add_argument("--dataset", default="evals/datasets/auto_eval.jsonl")
    parser.add_argument("--num-questions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--ood-ratio", type=float, default=0.3)
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()

    if "chunks_processed" not in args.chunks:
        print(
            "⚠️  WARNING: Running eval on raw chunks. "
            "Use --chunks chunks/chunks_processed.jsonl for quality-gated results."
        )

    os.makedirs(os.path.dirname(args.dataset), exist_ok=True)

    if args.regenerate or not os.path.exists(args.dataset):
        dataset = await build_dataset_from_chunks(
            chunks_path=args.chunks,
            num_questions=args.num_questions,
            seed=args.seed,
            max_chars=args.max_chars,
            ood_ratio=args.ood_ratio,
        )
        with open(args.dataset, "w", encoding="utf-8") as f:
            for row in dataset:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        dataset = []
        with open(args.dataset, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                dataset.append(json.loads(line))

    print(f"\n📊 Starting Parallel Experiment ({EVAL_CONCURRENCY} Workers)...")
    start_time = time.time()

    # Create tasks for parallel execution
    tasks = [run_eval_experiment(row) for row in dataset]
    total = len(tasks)
    done = 0
    results = []
    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)
        done += 1
        elapsed = max(time.time() - start_time, 0.001)
        rate = done / elapsed
        eta = (total - done) / rate if rate > 0 else 0
        bar_len = 24
        filled = int(bar_len * done / total) if total else bar_len
        bar = "#" * filled + "-" * (bar_len - filled)
        sys.stdout.write(
            f"\rProgress [{bar}] {done}/{total} | {elapsed:.0f}s elapsed | ETA {eta:.0f}s"
        )
        sys.stdout.flush()
    if total:
        print()

    # Report Generation
    df = pd.DataFrame(results)
    elapsed = round(time.time() - start_time, 2)
    total_n = len(df)
    avg_retrieval_ms = df["retrieval_ms"].mean()
    p95_retrieval_ms = df["retrieval_ms"].quantile(0.95)
    ragas_avg = None
    if "ragas_overall" in df.columns and df["ragas_overall"].notna().any():
        ragas_avg = df["ragas_overall"].mean()

    df_in = df[df["ood"] == False]
    df_ood = df[df["ood"] == True]

    in_n = len(df_in)
    ood_n = len(df_ood)

    in_excellent_n = (df_in["verdict"] == "Excellent").sum()
    in_hallucinated_n = (df_in["verdict"] == "Hallucinated").sum()
    in_irrelevant_n = (df_in["verdict"] == "Irrelevant").sum()

    in_relevance_rate = (in_excellent_n / in_n) * 100 if in_n else 0.0
    in_hallucination_rate = (in_hallucinated_n / in_n) * 100 if in_n else 0.0
    in_irrelevance_rate = (in_irrelevant_n / in_n) * 100 if in_n else 0.0
    in_faithfulness_rate = 100.0 - in_hallucination_rate
    in_em = df_in["exact_match"].mean() * 100 if in_n else 0.0
    in_f1 = df_in["f1"].mean() * 100 if in_n else 0.0
    in_recall = df_in["context_recall"].mean() * 100 if in_n else 0.0
    in_mrr = df_in["retrieval_mrr"].mean() if in_n else 0.0
    in_ctx_precision = df_in["context_precision"].mean() * 100 if in_n else 0.0

    ood_refusal_rate = df_ood["refused"].mean() * 100 if ood_n else 0.0
    ood_hallucination_rate = (df_ood["verdict"] == "Hallucinated").sum() / ood_n * 100 if ood_n else 0.0

    print(f"\n{'═'*60}")
    print(
        f"🏁 EVALUATION COMPLETE | Time: {elapsed}s | "
        f"Avg Retrieval: {avg_retrieval_ms:.1f}ms | P95: {p95_retrieval_ms:.1f}ms"
    )
    if ragas_avg is not None:
        print(f"RAGAS Overall Avg: {ragas_avg:.3f}")
    print(
        f"IN-SCOPE (N={in_n}) | "
        f"Answer Relevance: {in_relevance_rate:.1f}% | "
        f"Faithfulness: {in_faithfulness_rate:.1f}% | "
        f"Hallucination: {in_hallucination_rate:.1f}% | "
        f"Irrelevance: {in_irrelevance_rate:.1f}% | "
        f"EM: {in_em:.1f}% | F1: {in_f1:.1f}% | "
        f"Recall@k: {in_recall:.1f}% | MRR: {in_mrr:.3f} | "
        f"Context Precision: {in_ctx_precision:.1f}%"
    )
    print(
        f"OOD (N={ood_n}) | "
        f"Refusal Rate: {ood_refusal_rate:.1f}% | "
        f"Hallucination: {ood_hallucination_rate:.1f}%"
    )
    print(f"{'═'*60}")
    print(df[["question", "verdict", "answer"]])

    os.makedirs("evals/experiments", exist_ok=True)
    baseline_path = "evals/experiments/baseline_scores.json"
    if not os.path.exists(baseline_path):
        baseline = {
            "ragas_overall": round(ragas_avg, 3) if ragas_avg is not None else None,
            "faithfulness": round(df["ragas_faithfulness"].mean(), 3),
            "context_precision": round(df["ragas_context_precision"].mean(), 3),
            "ood_refusal_rate": round(ood_refusal_rate, 1),
            "mrr": round(in_mrr, 3),
            "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        }
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=2)
        print("📌 Baseline scores saved to evals/experiments/baseline_scores.json")
    df.to_csv("evals/experiments/fast_eval_report.csv", index=False)
    _save_judge_cache(JUDGE_CACHE_PATH, _judge_cache)

if __name__ == "__main__":
    asyncio.run(main())
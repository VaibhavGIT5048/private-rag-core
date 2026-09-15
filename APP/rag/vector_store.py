from __future__ import annotations

import os
import pickle
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pathlib import Path
from langchain_core.documents import Document
from APP.providers.embedding import build_default_embedding_provider

# Filtered HNSW search loses recall unless ef_search is raised above the
# collection's default ef_construct (100) to compensate — every query is
# document/owner-filtered now, so this always applies, not just sometimes.
EF_SEARCH = 128

# Must match APP/api_service.py's BackendSettings.qdrant_collection default —
# if these two drift, ingest writes to one collection while /health and the
# ingest response report another.
DEFAULT_COLLECTION = "chunks_collection_v2"

# Same rationale as BackendSettings.qdrant_timeout in rag/service.py: without
# a bound, a stale connection from before a Qdrant Cloud pause hangs a request
# forever instead of failing and letting the pool reconnect.
QDRANT_TIMEOUT_SECONDS = float(os.getenv("QDRANT_TIMEOUT_SECONDS", "15"))


def document_index_dir(owner_id: str, document_id: str) -> Path:
    return Path("data/documents") / owner_id / document_id


def qdrant_point_id(document_id: str, chunk_index: int) -> str:
    """Stable, globally-unique point id: the document's own UUID is used as
    the uuid5 namespace, so the same local chunk_index in a different
    document can never collide — this is what let multi-document ingest
    stop destructively recreating the collection every time.
    """
    return str(uuid.uuid5(uuid.UUID(document_id), str(chunk_index)))

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

try:
    from qdrant_client import QdrantClient
    from qdrant_client import models as qdrant_models
except Exception:
    QdrantClient = None
    qdrant_models = None

try:
    from flashrank import Ranker, RerankRequest
except Exception:
    Ranker = None
    RerankRequest = None


class QdrantVectorStore:
    def __init__(self, client, collection_name: str, embeddings, document_id: str | None = None, owner_id: str | None = None):
        self.client = client
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.document_id = document_id
        self.owner_id = owner_id

    def _scope_filter(self):
        """Every query is scoped to one document/owner now — this is what
        replaces "one collection per document" with "one shared collection,
        filtered per query," and is why EF_SEARCH exists (filtered HNSW
        search loses recall unless ef_search compensates for it).
        """
        if qdrant_models is None or self.document_id is None:
            return None
        conditions = [qdrant_models.FieldCondition(key="document_id", match=qdrant_models.MatchValue(value=self.document_id))]
        if self.owner_id is not None:
            conditions.append(qdrant_models.FieldCondition(key="owner_id", match=qdrant_models.MatchValue(value=self.owner_id)))
        return qdrant_models.Filter(must=conditions)

    def similarity_search(self, query: str, k: int = 5):
        qvec = None
        try:
            qvec = self.embeddings.embed_documents([query])[0]
        except Exception:
            try:
                qvec = self.embeddings.embed_query(query)
            except Exception:
                raise RuntimeError("Embedding backend not available for queries.")

        query_filter = self._scope_filter()
        search_params = qdrant_models.SearchParams(hnsw_ef=EF_SEARCH) if qdrant_models is not None else None

        hits = None
        try:
            hits = self.client.query_points(
                collection_name=self.collection_name,
                query=qvec,
                query_filter=query_filter,
                search_params=search_params,
                limit=k,
                with_payload=True,
            ).points
        except AttributeError:
            pass

        if hits is None:
            try:
                hits = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=qvec,
                    query_filter=query_filter,
                    search_params=search_params,
                    limit=k,
                    with_payload=True,
                )
            except TypeError:
                hits = self.client.search(
                    self.collection_name, qvec, limit=k, with_payload=True
                )

        results = []
        for h in hits:
            payload = getattr(h, "payload", None) or (h.get("payload") if isinstance(h, dict) else None)
            if payload is None:
                payload = {}

            text = payload.get("text") if isinstance(payload, dict) else None
            metadata = payload.get("metadata") if isinstance(payload, dict) else {}
            if text is None:
                text = payload.get("payload", {}).get("text") if isinstance(payload, dict) else ""
                metadata = payload.get("payload", {}).get("metadata", {}) if isinstance(payload, dict) else {}

            doc = Document(page_content=text or "", metadata=metadata or {})
            results.append(doc)

        return results


_flashrank_ranker = None


def _get_ranker():
    """Builds the ranker once, on first use.

    The cache dir is where the ranker's weights live. In the deployed image
    FLASHRANK_CACHE_DIR points at the baked copy under /opt/models; the default
    below is the local-dev path and sits under data/, which is the Azure Files
    share in production — slow to read, and the reason the env var is set.
    """
    global _flashrank_ranker
    if _flashrank_ranker is None:
        cache_dir = os.getenv("FLASHRANK_CACHE_DIR", "data/flashrank_cache")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        _flashrank_ranker = Ranker(cache_dir=cache_dir)
    return _flashrank_ranker


def warm_reranker() -> bool:
    """Loads the ranker ahead of the first query. Returns whether it is ready.

    Separate from flashrank_rerank because that one short-circuits on empty
    candidates and so never reaches the constructor — warming through it would
    silently do nothing.
    """
    if Ranker is None or RerankRequest is None:
        return False
    _get_ranker()
    return True


def flashrank_rerank(query: str, candidates: list[Document], k: int = 5) -> list[Document]:
    if Ranker is None or RerankRequest is None or not candidates:
        return candidates[:k]

    ranker = _get_ranker()

    passages = []
    for idx, doc in enumerate(candidates):
        passages.append({
            "id": str(idx),
            "text": doc.page_content,
            "meta": doc.metadata,
        })

    request = RerankRequest(query=query, passages=passages)
    ranked = ranker.rerank(request)
    top_docs = []
    for item in ranked[:k]:
        index = int(item["id"]) if isinstance(item, dict) and "id" in item else int(getattr(item, "id", 0))
        if 0 <= index < len(candidates):
            top_docs.append(candidates[index])
    return top_docs or candidates[:k]


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def tokenize_for_bm25(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _collection_exists(client, collection_name: str) -> bool:
    try:
        return client.collection_exists(collection_name)
    except AttributeError:
        try:
            client.get_collection(collection_name)
            return True
        except Exception:
            return False


# Every retrieval filters on document_id (and owner_id), and Qdrant refuses to
# filter on a payload key that has no index — the search fails outright with
# "Index required but not found", not merely slowly. Creating an index that
# already exists is a no-op, so this runs on every ingest rather than only at
# collection creation: collections built before these filters existed have to
# pick the indexes up too.
_INDEXED_PAYLOAD_KEYS = ("document_id", "owner_id")


def _ensure_payload_indexes(client, collection_name: str) -> None:
    for key in _INDEXED_PAYLOAD_KEYS:
        try:
            if qdrant_models is not None:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=key,
                    field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
                )
            else:
                client.create_payload_index(
                    collection_name=collection_name, field_name=key, field_schema="keyword"
                )
        except Exception:
            # Already-indexed is the common case and reports as an error on
            # some server versions; a genuine failure surfaces on the next
            # filtered search rather than blocking the ingest that just ran.
            pass


def build_hybrid_indices(chunks, document_id: str, owner_id: str, vectors: list[list[float]] | None = None, embeddings=None, client=None):
    """Indexes one document's chunks into the shared Qdrant collection
    (create-if-missing + upsert — never destructive, unlike the old
    per-ingest recreate_collection) and writes that document's own BM25
    pickle under data/documents/{owner_id}/{document_id}/.

    `vectors`, if given, are used as-is instead of re-embedding here — this
    is the embedding-pass consolidation: the caller computes chunk
    embeddings once, in parallel, and feeds the same vectors into both the
    quality-gate overlap check and this indexing step.

    `client`, if given, is reused as-is instead of constructing a new one —
    RAGService passes its own pooled, timeout-configured client so every
    ingest shares the same connection pool the rest of the service uses,
    rather than opening and discarding a fresh one per call. Falls back to
    building one (e.g. for the CLI/eval-harness callers that have no
    RAGService instance to borrow from) when not given.
    """
    if not chunks:
        print("⚠️ No chunks to index. Skipping build.")
        return None, None

    print("🧠 Indexing into Qdrant (Dense)...")
    # Defaults to the same self-hosted provider the caller embedded with —
    # never the old OpenAI adapter. Getting this wrong is silent but fatal:
    # the vectorstore returned below would embed queries at a different
    # dimension than the collection was built with.
    embeddings = embeddings or build_default_embedding_provider()

    if QdrantClient is None:
        print("⚠️ qdrant-client not installed.")
        return None, None

    if client is None:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=QDRANT_TIMEOUT_SECONDS)
    collection_name = os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)

    if vectors is None:
        texts = [c.page_content for c in chunks]
        batch_size = 64
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        print(f"   Embedding {len(texts)} chunks in {len(batches)} batches (parallel)...")
        with ThreadPoolExecutor(max_workers=min(8, len(batches) or 1)) as executor:
            batch_vectors = list(executor.map(embeddings.embed_documents, batches))
        vectors = [vec for batch in batch_vectors for vec in batch]

    vector_size = len(vectors[0]) if vectors else 0

    if not _collection_exists(client, collection_name):
        try:
            if qdrant_models is not None:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qdrant_models.VectorParams(size=vector_size, distance=qdrant_models.Distance.COSINE),
                )
            else:
                client.create_collection(collection_name=collection_name, vectors_config={"size": vector_size, "distance": "Cosine"})
        except Exception:
            pass

    _ensure_payload_indexes(client, collection_name)

    points = []
    for c, v in zip(chunks, vectors):
        chunk_index = int(c.metadata.get("chunk_id", len(points)))
        pid = qdrant_point_id(document_id, chunk_index)
        payload = {"text": c.page_content, "metadata": c.metadata, "document_id": document_id, "owner_id": owner_id}
        if qdrant_models is not None:
            points.append(qdrant_models.PointStruct(id=pid, vector=v, payload=payload))
        else:
            points.append({"id": pid, "vector": v, "payload": payload})

    client.upsert(collection_name=collection_name, points=points)
    vectorstore = QdrantVectorStore(
        client=client, collection_name=collection_name, embeddings=embeddings,
        document_id=document_id, owner_id=owner_id,
    )

    print("📝 Building this document's BM25 Keyword Index (Sparse)...")
    tokenized_corpus = [tokenize_for_bm25(doc.page_content) for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    doc_dir = document_index_dir(owner_id, document_id)
    doc_dir.mkdir(parents=True, exist_ok=True)
    with open(doc_dir / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    print(f"✅ Document indexed. BM25 saved to: {doc_dir.absolute()}")
    return vectorstore, bm25


def load_document_index(document_id: str, owner_id: str, embeddings=None, client=None) -> tuple[QdrantVectorStore | None, object | None, list[Document] | None]:
    """Loads the pieces needed for /query on one document: a document-scoped
    QdrantVectorStore plus that document's own BM25 index and chunks.

    `client`, if given, is reused as-is — see build_hybrid_indices's
    docstring for why (same pooled-client rationale, same CLI/eval fallback).
    """
    doc_dir = document_index_dir(owner_id, document_id)
    bm25_path = doc_dir / "bm25.pkl"
    if not bm25_path.exists():
        return None, None, None

    with open(bm25_path, "rb") as f:
        data = pickle.load(f)
    bm25 = data["bm25"]
    chunks = data["chunks"]

    if QdrantClient is None:
        return None, bm25, chunks

    embeddings = embeddings or build_default_embedding_provider()
    if client is None:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=QDRANT_TIMEOUT_SECONDS)
    collection_name = os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
    vectorstore = QdrantVectorStore(
        client=client, collection_name=collection_name, embeddings=embeddings,
        document_id=document_id, owner_id=owner_id,
    )
    return vectorstore, bm25, chunks


def apply_quality_penalty(chunks, base_scores):
    penalised = []
    for chunk, score in zip(chunks, base_scores):
        gate = chunk.metadata.get("passed_gate", False)
        weight = 1.0 if gate else 0.7
        penalised.append((chunk, score * weight))
    return sorted(penalised, key=lambda x: x[1], reverse=True)


def hybrid_retrieve(
    query: str,
    vectorstore: QdrantVectorStore,
    bm25,
    chunks: list[Document],
    top_n: int = 5,
    rrf_k: int = 15,
) -> list[tuple[Document, float]]:
    # candidate pool: headroom for the reranker without bloating latency
    candidate_k = max(top_n * 5, 25)

    # dense leads for semantic QA; sparse supports exact-term recall
    dense_weight = 0.65
    bm25_weight = 0.35

    semantic_results = vectorstore.similarity_search(query, k=min(candidate_k, len(chunks)))

    tokenized_query = tokenize_for_bm25(query)
    keyword_scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(keyword_scores)[::-1][:candidate_k]
    keyword_results = [chunks[i] for i in top_indices if keyword_scores[i] > 0]

    rrf_scores: dict = {}
    doc_map: dict = {}

    # chunk_id can be absent if a Qdrant payload was written by an older or external
    # indexer; skip those rather than dying mid-query.
    for rank, doc in enumerate(semantic_results, 1):
        cid = doc.metadata.get("chunk_id")
        if cid is None:
            continue
        doc_map[cid] = doc
        rrf_scores[cid] = rrf_scores.get(cid, 0) + dense_weight / (rrf_k + rank)

    for rank, doc in enumerate(keyword_results, 1):
        cid = doc.metadata.get("chunk_id")
        if cid is None:
            continue
        doc_map[cid] = doc
        rrf_scores[cid] = rrf_scores.get(cid, 0) + bm25_weight / (rrf_k + rank)

    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    base_results = [(doc_map[cid], score) for cid, score in sorted_results]

    penalised = apply_quality_penalty(
        chunks=[doc for doc, _ in base_results],
        base_scores=[score for _, score in base_results],
    )

    reranked_docs = flashrank_rerank(query, [doc for doc, _ in penalised[:candidate_k]], k=top_n)
    reranked_set = {doc.metadata.get("chunk_id"): doc for doc in reranked_docs}
    if reranked_set:
        return [
            (reranked_set.get(doc.metadata.get("chunk_id"), doc), score)
            for doc, score in penalised
            if doc.metadata.get("chunk_id") in reranked_set
        ][:top_n]
    return penalised[:top_n]


def expand_with_neighbors(
    results: list[tuple[Document, float]],
    chunks: list[Document],
    window: int = 1,
) -> list[Document]:
    """Original expand — returns list[Document] without scores."""
    id_to_doc = {
        doc.metadata["chunk_id"]: doc
        for doc in chunks
        if doc.metadata.get("chunk_id") is not None
    }
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


def expand_with_neighbors_scored(
    results: list[tuple[Document, float]],
    chunks: list[Document],
    window: int = 1,
) -> list[tuple[Document, float]]:
    """Like expand_with_neighbors but returns (doc, score) tuples.
    Neighbors inherit the score of the retrieved doc that pulled them in.
    Directly-retrieved docs keep their own score. Order is preserved and
    de-duplicated (first occurrence / highest-priority anchor wins).
    """
    id_to_doc = {
        doc.metadata["chunk_id"]: doc
        for doc in chunks
        if doc.metadata.get("chunk_id") is not None
    }
    seen: set = set()
    expanded: list[tuple[Document, float]] = []

    for doc, score in results:
        cid = doc.metadata.get("chunk_id")
        if cid is None:
            continue
        for neighbor_id in range(cid - window, cid + window + 1):
            if neighbor_id in id_to_doc and neighbor_id not in seen:
                expanded.append((id_to_doc[neighbor_id], score))
                seen.add(neighbor_id)

    return expanded
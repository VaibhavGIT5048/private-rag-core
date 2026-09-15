"""Embedding providers — self-hosted bge-m3 by default, plain OpenAI for BYO-key.

Both expose embed_documents()/embed_query() returning plain float lists, the
same shape APP/rag/vector_store.py and APP/rag/quality_gate.py already consume
via the old EmbeddingAdapter — swapping providers needs no changes downstream.
"""

from __future__ import annotations

import os
import threading

DEFAULT_BGE_MODEL = os.getenv("BGE_MODEL_NAME", "BAAI/bge-m3")
DEFAULT_BYO_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# "torch" (default) or "onnx". ONNX Runtime generally beats PyTorch on CPU
# thanks to graph optimisation, and BAAI/bge-m3 publishes prebuilt ONNX
# weights so nothing has to be exported at build time.
#
# Deliberately NOT the default yet: the meaningful INT8 speedup comes from
# AVX512-VNNI kernels, which are x86-only. The dev machine here is arm64, so
# a local benchmark would understate (or misrepresent) what production gets.
# Validate on staging — x86 Container Apps — before flipping the default.
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "torch").strip().lower()

# Measured on 60 real chunks, 10 cores: batch 8 -> 36.2s, 16 -> 40.9s,
# 32 -> 48.6s. Smaller wins because bge-m3 takes long sequences and a big
# batch pads every short text up to the longest one in it.
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))


class EmbeddingProvider:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class BgeEmbeddingProvider(EmbeddingProvider):
    """Self-hosted BAAI/bge-m3 via sentence-transformers, CPU-only — no
    Azure resource, no per-token API cost. Loaded lazily on first use since
    the model weights (~2GB) shouldn't be pulled at process startup.
    """

    def __init__(self, model_name: str = DEFAULT_BGE_MODEL, backend: str = EMBEDDING_BACKEND):
        self.model_name = model_name
        self.backend = backend
        self._model = None
        # Double-checked locking, and the lock is not optional: ingest embeds
        # via a ThreadPoolExecutor, so without it every worker sees
        # `_model is None` at once and each loads its own 2.3GB copy of the
        # model. Measured cost of that race on a 44-page PDF: 3 concurrent
        # loads, ~5m00s total ingest versus ~1m30s once serialised.
        self._load_lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is not None:  # another thread won the race
                return self._model

            from sentence_transformers import SentenceTransformer

            print(f"🔄 Loading self-hosted embedding model: {self.model_name} (backend={self.backend})...")
            model = None
            if self.backend == "onnx":
                try:
                    model = SentenceTransformer(self.model_name, device="cpu", backend="onnx")
                except Exception as exc:
                    # A missing/incompatible ONNX artifact must not take ingest
                    # down — PyTorch weights always work, just slower.
                    print(f"⚠️ ONNX backend unavailable ({exc}); falling back to torch")
            if model is None:
                model = SentenceTransformer(self.model_name, device="cpu")
            print(f"✅ Self-hosted embedding model ready: {self.model_name}")
            # Publish only once fully constructed, so no other thread can see
            # a half-initialised model through the fast path above.
            self._model = model
            return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        # One call for the whole list: sentence-transformers batches internally
        # AND sorts by length first, so short texts aren't padded up to the
        # longest in an arbitrary caller-made batch. Hand-splitting the work
        # across threads defeats that and oversubscribes the CPU, since torch
        # already parallelises across every core.
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=EMBEDDING_BATCH_SIZE,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """BYO-key — constructed per-request from a header, never persisted."""

    def __init__(self, api_key: str | None, model: str = DEFAULT_BYO_EMBEDDING_MODEL):
        self.api_key = api_key
        self.model = model
        self._backend = None

    def _load(self):
        if self._backend is None:
            from langchain_openai import OpenAIEmbeddings

            self._backend = OpenAIEmbeddings(model=self.model, api_key=self.api_key)
        return self._backend

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._load().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._load().embed_query(text)


def build_default_embedding_provider() -> EmbeddingProvider:
    return BgeEmbeddingProvider()

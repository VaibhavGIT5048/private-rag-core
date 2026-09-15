import os

from langchain_openai import OpenAIEmbeddings

DEFAULT_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


class EmbeddingAdapter:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, api_key: str | None = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._backend = None

    def _load_backend(self):
        if self._backend is not None:
            return self._backend

        self._backend = OpenAIEmbeddings(model=self.model_name, api_key=self.api_key)
        return self._backend

    def embed_documents(self, texts: list[str]):
        backend = self._load_backend()
        return backend.embed_documents(texts)

    def embed_query(self, text: str):
        backend = self._load_backend()
        return backend.embed_query(text)


def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    print(f"🔄 Loading embedding adapter: {model_name}...")
    embedding_model = EmbeddingAdapter(model_name=model_name)
    print(f"✅ Embedding adapter ready (primary: {model_name})")
    return embedding_model

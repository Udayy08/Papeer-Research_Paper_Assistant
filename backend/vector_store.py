import os
import time
from typing import Any, cast

from dotenv import load_dotenv
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.embeddings.cache import batch_iterate
from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from backend.usage_tracker import get_current_tracker

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────

EMBEDDING_DIM = 768  # gemini-embedding-2


class TrackingCacheBackedEmbeddings(CacheBackedEmbeddings):
    """Subclass of CacheBackedEmbeddings that tracks query and document cache
    hits, misses, and Gemini API requests without modifying the cache structure."""

    def embed_query(self, text: str) -> list[float]:
        tracker = get_current_tracker()
        if not self.query_embedding_store:
            if tracker:
                tracker.record_gemini_query(hit=False, char_count=len(text))
            return self.underlying_embeddings.embed_query(text)

        (cached,) = self.query_embedding_store.mget([text])
        if cached is not None:
            if tracker:
                tracker.record_gemini_query(hit=True)
            return cached

        if tracker:
            tracker.record_gemini_query(hit=False, char_count=len(text))
        vector = self.underlying_embeddings.embed_query(text)
        self.query_embedding_store.mset([(text, vector)])
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float] | None] = self.document_embedding_store.mget(texts)
        all_missing_indices: list[int] = [
            i for i, vector in enumerate(vectors) if vector is None
        ]

        hits = len(texts) - len(all_missing_indices)
        misses = len(all_missing_indices)
        missing_chars = sum(len(texts[i]) for i in all_missing_indices)

        tracker = get_current_tracker()
        if tracker:
            tracker.record_gemini_doc(hits=hits, misses=misses, char_count=missing_chars)

        for missing_indices in batch_iterate(self.batch_size, all_missing_indices):
            missing_texts = [texts[i] for i in missing_indices]
            missing_vectors = self.underlying_embeddings.embed_documents(missing_texts)
            self.document_embedding_store.mset(
                list(zip(missing_texts, missing_vectors, strict=False)),
            )
            for index, updated_vector in zip(
                missing_indices, missing_vectors, strict=False
            ):
                vectors[index] = updated_vector

        return cast("list[list[float]]", vectors)

    @classmethod
    def from_bytes_store(
        cls,
        underlying_embeddings: Any,
        document_embedding_cache: Any,
        *,
        namespace: str = "",
        batch_size: int | None = None,
        query_embedding_cache: Any = False,
        key_encoder: Any = "sha1",
    ) -> "TrackingCacheBackedEmbeddings":
        res = super().from_bytes_store(
            underlying_embeddings,
            document_embedding_cache,
            namespace=namespace,
            batch_size=batch_size,
            query_embedding_cache=query_embedding_cache,
            key_encoder=key_encoder,
        )
        return cls(
            underlying_embeddings=res.underlying_embeddings,
            document_embedding_store=res.document_embedding_store,
            batch_size=res.batch_size,
            query_embedding_store=res.query_embedding_store,
        )


# ── Singletons ────────────────────────────────────────────────────────────────

base_embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    output_dimensionality=EMBEDDING_DIM
)
embedding_file_store = LocalFileStore("./embedding_cache/")
embeddings = TrackingCacheBackedEmbeddings.from_bytes_store(
    base_embeddings,
    embedding_file_store,
    namespace=base_embeddings.model,
    query_embedding_cache=True,
    key_encoder="blake2b",
)

qdrant_client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    timeout=120,
)


# ── Collection ───────────────────────────────────────────────────────────────

def get_collection_name(session_id: str) -> str:
    return f"papeer_{session_id.replace('-', '_')}"


def get_vectorstore(session_id: str) -> QdrantVectorStore:
    collection_name = get_collection_name(session_id)
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
    )


# ── Public API ───────────────────────────────────────────────────────────────

def add_paper(docs: list[Document], session_id: str) -> None:
    get_vectorstore(session_id).add_documents(docs)


def list_papers(session_id: str) -> list[str]:
    collection_name = get_collection_name(session_id)
    if not qdrant_client.collection_exists(collection_name):
        return []
    seen: set[str] = set()
    titles: list[str] = []
    offset = None
    while True:
        points, offset = qdrant_client.scroll(
            collection_name=collection_name,
            with_payload=True,
            limit=100,
            offset=offset,
        )
        for point in points:
            title = (point.payload or {}).get("metadata", {}).get("title")
            if title and title not in seen:
                seen.add(title)
                titles.append(title)
        if offset is None:
            break
    return titles


def search(query: str, session_id: str, k: int = 4) -> list[Document]:
    start_t = time.perf_counter()
    docs = get_vectorstore(session_id).similarity_search(query, k=k)
    lat_ms = (time.perf_counter() - start_t) * 1000
    tracker = get_current_tracker()
    if tracker:
        tracker.record_qdrant(searches=1, chunks=len(docs), latency_ms=lat_ms)
    return docs

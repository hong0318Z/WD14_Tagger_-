import hashlib
import os
import time

import numpy as np
from openai import OpenAI

DEFAULT_EMBEDDING_BASE_URL = "http://192.168.0.116:8000/v1"
DEFAULT_EMBEDDING_MODELS = [
    "bge-m3",
    "multilingual-e5-large",
    "nomic-embed-text",
    "text-embedding-3-small",
]
DEFAULT_EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODELS[0]

EMBED_BATCH_SIZE = 64
REQUEST_TIMEOUT = 120
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "embedding_cache")


def _client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(
        api_key=api_key or "local",
        base_url=base_url or DEFAULT_EMBEDDING_BASE_URL,
        timeout=REQUEST_TIMEOUT,
    )


def embed_texts(api_key: str, texts: list, model: str, base_url: str = None):
    """Returns (vectors: np.ndarray[N, D], usage_dict)."""
    model = model or DEFAULT_EMBEDDING_MODEL
    client = _client(api_key, base_url)
    started = time.time()
    vectors = []
    prompt_tokens = 0
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        resp = client.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in resp.data)
        if getattr(resp, "usage", None):
            prompt_tokens += getattr(resp.usage, "prompt_tokens", 0) or getattr(resp.usage, "total_tokens", 0) or 0
    elapsed = time.time() - started
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": 0, "elapsed": elapsed}
    print(f"[embedding] model={model} texts={len(texts)} elapsed={elapsed:.1f}s usage_tokens={prompt_tokens}")
    return np.array(vectors, dtype=np.float32), usage


def _db_signature(db) -> str:
    h = hashlib.md5()
    for name in sorted(db.by_name.keys()):
        entry = db.by_name[name]
        h.update(name.encode("utf-8"))
        h.update(entry["description"].encode("utf-8"))
    return h.hexdigest()


def _cache_path(model: str, signature: str) -> str:
    safe_model = model.replace("/", "_").replace(" ", "_")
    return os.path.join(CACHE_DIR, f"{safe_model}_{signature}.npz")


def get_or_build_tag_embeddings(db, api_key: str, base_url: str, model: str):
    """Returns (names: list[str], vectors: np.ndarray) for all tags in db,
    using a disk cache keyed by (model, db content signature)."""
    signature = _db_signature(db)
    path = _cache_path(model, signature)
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        return list(data["names"]), data["vectors"]

    names = sorted(db.by_name.keys())
    texts = []
    for name in names:
        entry = db.by_name[name]
        kw = " ".join(entry["keywords"])
        texts.append(f"{name.replace('_', ' ')} {kw} {entry['description']}".strip())

    vectors, _usage = embed_texts(api_key, texts, model, base_url)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(path, names=np.array(names, dtype=object), vectors=vectors)
    return names, vectors


def top_k_similar(query_vector, names, vectors, k: int = 24):
    """Cosine similarity top-k. query_vector: (D,), vectors: (N, D)."""
    q = query_vector / (np.linalg.norm(query_vector) + 1e-8)
    v_norms = np.linalg.norm(vectors, axis=1) + 1e-8
    sims = (vectors @ q) / v_norms
    top_idx = np.argsort(-sims)[:k]
    return [(names[i], float(sims[i])) for i in top_idx]

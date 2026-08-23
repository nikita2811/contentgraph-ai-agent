# app/memory/research_cache.py
import os
import json
import hashlib
from functools import lru_cache
from typing import Optional

from upstash_vector import Index
from google import genai
from dotenv import load_dotenv
from google.genai import types

load_dotenv()

# ── Init ──────────────────────────────────────────────────────────────────────

_index = Index(
    url=os.getenv("UPSTASH_VECTOR_REST_URL"),
    token=os.getenv("UPSTASH_VECTOR_REST_TOKEN"),
)

_client = None

def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        _client = genai.Client(api_key=api_key)
    return _client


SIMILARITY_THRESHOLD = 0.85


@lru_cache(maxsize=1024)
def _get_embedder(text: str) -> tuple:
    """Returns the embedding vector for `text` as a tuple (hashable, cacheable)."""
    client = _get_client()
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=768),
    )
    return tuple(result.embeddings[0].values)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_query_text(product_details: dict) -> str:
    return (
        f"{product_details.get('product_name', '')} "
        f"{product_details.get('category', '')} "
        f"{' '.join(product_details.get('key_features', []))} "
        f"{product_details.get('target_audience', '')}"
    ).strip()


def _make_doc_id(product_details: dict) -> str:
    return hashlib.md5(_make_query_text(product_details).encode()).hexdigest()


# ── Public API ────────────────────────────────────────────────────────────────

def get_similar_research(product_details: dict) -> Optional[dict]:
    query_text = _make_query_text(product_details)
    embedding = list(_get_embedder(query_text))   # ← no .encode(), just use it directly

    try:
        results = _index.query(
            vector=embedding,
            top_k=1,
            include_metadata=True,
            include_data=True,
        )
    except Exception as e:
        print(f"⚠️  [Research Cache] Query failed: {e}")
        return None

    if not results:
        return None

    top = results[0]
    score = top.score

    if score < SIMILARITY_THRESHOLD:
        print(f"🔍 [Research Cache] MISS — best similarity: {score:.3f}")
        return None

    print(
        f"🧠 [Research Cache] HIT — similarity: {score:.3f} "
        f"(matched: {top.metadata.get('product_name')})"
    )

    try:
        return json.loads(top.data)
    except (json.JSONDecodeError, TypeError):
        print("⚠️  [Research Cache] Corrupted entry, treating as miss")
        return None


def store_research(product_details: dict, serp_results: dict) -> None:
    query_text = _make_query_text(product_details)
    embedding = list(_get_embedder(query_text))   # ← no .encode(), just use it directly
    doc_id = _make_doc_id(product_details)

    try:
        _index.upsert(
            vectors=[{
                "id": doc_id,
                "vector": embedding,
                "data": json.dumps(serp_results),
                "metadata": {
                    "product_name": product_details.get("product_name", ""),
                    "category": product_details.get("category", ""),
                    "tone": product_details.get("tone", ""),
                },
            }]
        )
        print(f"💾 [Research Cache] Stored: '{product_details.get('product_name')}'")
    except Exception as e:
        print(f"⚠️  [Research Cache] Store failed (non-fatal): {e}")


def delete_research(product_details: dict) -> None:
    doc_id = _make_doc_id(product_details)
    try:
        _index.delete(ids=[doc_id])
        print(f"🗑️  [Research Cache] Deleted: '{product_details.get('product_name')}'")
    except Exception as e:
        print(f"⚠️  [Research Cache] Delete failed: {e}")


def clear_research_cache() -> None:
    try:
        _index.reset()
        print("🗑️  [Research Cache] Cleared all entries")
    except Exception as e:
        print(f"⚠️  [Research Cache] Clear failed: {e}")
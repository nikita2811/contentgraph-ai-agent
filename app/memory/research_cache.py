# app/memory/research_cache.py
import os
import json
import hashlib
from typing import Optional

from upstash_vector import Index
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ── Init ──────────────────────────────────────────────────────────────────────

_index = Index(
    url=os.getenv("UPSTASH_VECTOR_REST_URL"),
    token=os.getenv("UPSTASH_VECTOR_REST_TOKEN"),
)

_embedder = SentenceTransformer("all-MiniLM-L6-v2")

SIMILARITY_THRESHOLD = 0.85


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_query_text(product_details: dict) -> str:
    """
    Combines product fields into a single string for embedding.
    More fields = better semantic matching.
    """
    return (
        f"{product_details.get('product_name', '')} "
        f"{product_details.get('category', '')} "
        f"{' '.join(product_details.get('key_features', []))} "
        f"{product_details.get('target_audience', '')}"
    ).strip()


def _make_doc_id(product_details: dict) -> str:
    """Stable unique ID — used for upsert deduplication."""
    return hashlib.md5(_make_query_text(product_details).encode()).hexdigest()


# ── Public API ────────────────────────────────────────────────────────────────

def get_similar_research(product_details: dict) -> Optional[dict]:
    """
    Query Upstash Vector for semantically similar cached SERP results.
    Returns cached research dict if similarity >= threshold, else None.
    """
    query_text = _make_query_text(product_details)
    embedding  = _embedder.encode(query_text).tolist()

    try:
        results = _index.query(
            vector=embedding,
            top_k=1,
            include_metadata=True,
            include_data=True,       # ← returns the stored JSON string
        )
    except Exception as e:
        # network error or empty index — treat as cache miss, never crash
        print(f"⚠️  [Research Cache] Query failed: {e}")
        return None

    if not results:
        return None

    top    = results[0]
    score  = top.score   # cosine similarity, 0–1

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
        # corrupted entry — treat as miss
        print("⚠️  [Research Cache] Corrupted entry, treating as miss")
        return None


def store_research(product_details: dict, serp_results: dict) -> None:
    """
    Store SERP results in Upstash Vector.
    Only called after SERP validator has scored and filtered — never stores junk.
    """
    query_text = _make_query_text(product_details)
    embedding  = _embedder.encode(query_text).tolist()
    doc_id     = _make_doc_id(product_details)

    try:
        _index.upsert(
            vectors=[{
                "id":       doc_id,
                "vector":   embedding,
                "data":     json.dumps(serp_results),   # stored as-is, retrieved in query
                "metadata": {
                    "product_name": product_details.get("product_name", ""),
                    "category":     product_details.get("category", ""),
                    "tone":         product_details.get("tone", ""),
                },
            }]
        )
        print(f"💾 [Research Cache] Stored: '{product_details.get('product_name')}'")
    except Exception as e:
        # storage failure should never crash the pipeline
        print(f"⚠️  [Research Cache] Store failed (non-fatal): {e}")


def delete_research(product_details: dict) -> None:
    """Delete a specific entry — useful for cache invalidation."""
    doc_id = _make_doc_id(product_details)
    try:
        _index.delete(ids=[doc_id])
        print(f"🗑️  [Research Cache] Deleted: '{product_details.get('product_name')}'")
    except Exception as e:
        print(f"⚠️  [Research Cache] Delete failed: {e}")


def clear_research_cache() -> None:
    """Wipe entire index — use carefully, mainly for testing."""
    try:
        _index.reset()
        print("🗑️  [Research Cache] Cleared all entries")
    except Exception as e:
        print(f"⚠️  [Research Cache] Clear failed: {e}")
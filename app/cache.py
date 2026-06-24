# app/cache.py
import os, json, hashlib
from langchain_core.globals import set_llm_cache
from upstash_redis import Redis
from dotenv import load_dotenv
from langchain_core.caches import InMemoryCache

load_dotenv()

# ── Single shared client ───────────────────────────────────
redis = Redis(
    url=os.getenv("UPSTASH_REDIS_REST_URL"),
    token=os.getenv("UPSTASH_REDIS_REST_TOKEN"),
)

def init_llm_cache(ttl: int = 3600):
    """LLM-level cache — caches individual Gemini calls."""
    set_llm_cache(InMemoryCache())
    print("✅ InMemory cache LLM cache initialized")


# ── Generic helpers ────────────────────────────────────────
def make_key(prefix: str, data: dict | str) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, sort_keys=True)
    return f"{prefix}:{hashlib.md5(payload.encode()).hexdigest()}"

def cache_get(key: str):
    try:
        val = redis.get(key)
        return json.loads(val) if val else None
    except Exception:
        return None

def cache_set(key: str, value, ttl: int = 3600):
    try:
        redis.setex(key, ttl, json.dumps(value, default=str))
    except Exception as e:
        print(f"⚠️ Cache set failed: {e}")


# ── Node-level cache ───────────────────────────────────────
# ── Pipeline-level cache ───────────────────────────────────

# Only hash fields that define the product identity
_PIPELINE_CACHE_FIELDS = [
    "product_name", "category", "target_audience", "key_features", "tone"
]

def get_node_cache(node_name: str, prompt: str):
    key = make_key(f"node:{node_name}", prompt)
    val = cache_get(key)
    if val:
        print(f"⚡ Node cache HIT  [{node_name}]")
    else:
        print(f"❌ Node cache MISS [{node_name}]")
    return val, key
def set_node_cache(key: str, output, ttl: int = 3600):
    cache_set(key, output, ttl)



# ── Pipeline-level cache ───────────────────────────────────
def get_pipeline_cache(product_details: dict):
    stable = {k: product_details[k] for k in _PIPELINE_CACHE_FIELDS if k in product_details}
    key = make_key("pipeline", stable)
    val = cache_get(key)
    if val:
        print(f"⚡ Pipeline cache HIT  — key: {key}")
    else:
        print(f"❌ Pipeline cache MISS — key: {key}")
    return val, key

def set_pipeline_cache(key: str, final_state: dict, ttl: int = 3600):
    cacheable = {
        "research_output": final_state.get("research_output"),
        "serp_output":     final_state.get("serp_output"),
        "content_output":  final_state.get("content_output"),
        "token_usage":     final_state.get("token_usage", {}),
        "current_step":    final_state.get("current_step"),
        "error":           final_state.get("error"),
    }
    cache_set(key, cacheable, ttl)
    print(f"✅ Pipeline result cached")
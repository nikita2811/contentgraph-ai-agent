# tools.py
import json
import re
import time
from collections import Counter

from dotenv import load_dotenv
from fastapi.concurrency import run_in_threadpool
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()

tavily_tool = TavilySearch(
    max_results=3,
    search_depth="basic",
    include_answer=True,
    include_raw_content=False,
)

serp_wrapper = SerpAPIWrapper()

# ── Tunables ───────────────────────────────────────────────────────────────

SERP_ORGANIC_LIMIT_SEARCH   = 8   # results kept by serp_search
SERP_SHOPPING_LIMIT         = 5
SERP_ORGANIC_LIMIT_ANALYZE  = 6
SERP_KEYWORD_TOP_N          = 20
SERP_SNIPPET_LIMIT          = 6
SERP_CACHE_TTL_SECONDS      = 900  # 15 min — SerpAPI is metered per call

STOPWORDS = {
    "with", "your", "this", "that", "from", "best", "here", "have",
    "will", "what", "when", "where", "which", "their", "about", "into",
    "more", "than", "then", "them", "these", "those", "such", "only",
    "just", "also", "over", "under", "very", "some", "each", "most",
}

# ── Tiny in-process TTL cache — avoids paying for the same SERP query twice ─
# Swap for the existing Redis cache layer if this needs to survive restarts
# or be shared across workers.

_serp_cache: dict[str, tuple[float, dict]] = {}


async def _cached_serp_results(query: str) -> dict:
    now = time.monotonic()
    cached = _serp_cache.get(query)
    if cached and (now - cached[0]) < SERP_CACHE_TTL_SECONDS:
        return cached[1]

    results = await run_in_threadpool(serp_wrapper.results, query)
    _serp_cache[query] = (now, results)
    return results


def _dig(d: dict, *keys, default=None):
    """Safely walk a chain of nested .get() calls."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


def _price_to_float(price_str):
    """SerpAPI prices arrive as strings like '$29.99' — pull the number out."""
    if not price_str:
        return None
    m = re.search(r"[\d,]+\.?\d*", price_str)
    if not m:
        return None
    try:
        return float(m.group().replace(",", ""))
    except ValueError:
        return None


# ── Tavily — has native async, use ainvoke ────────────────────────────────

@tool
async def tavily_search_async(query: str) -> str:
    """Async web search via Tavily — returns an answer plus supporting results."""
    try:
        result = await tavily_tool.ainvoke(query)
    except Exception as e:
        return json.dumps({"error": f"Tavily search failed: {e}"})
    return result if isinstance(result, str) else json.dumps(result, default=str)


# ── SerpAPI — no native async, offload to threadpool ─────────────────────

def _serp_search_sync(query: str) -> str:
    """Pure sync SerpAPI call — called via threadpool, never directly."""
    results = serp_wrapper.results(query)
    organic = results.get("organic_results", [])
    snippets = []
    for r in organic[:SERP_ORGANIC_LIMIT_SEARCH]:
        snippets.append({
            "position": r.get("position"),
            "title":    r.get("title"),
            "snippet":  r.get("snippet"),
            "link":     r.get("link"),
        })
    return json.dumps(snippets, indent=2)


@tool
async def serp_search(query: str) -> str:
    """Search Google via SerpAPI and return structured results for SEO analysis."""
    try:
        return await run_in_threadpool(_serp_search_sync, query)
    except Exception as e:
        return json.dumps({"error": f"SerpAPI search failed: {e}"})


# ── Main product SERP analysis — both sync helpers + async orchestration ──

def _parse_serp_results(query: str, results: dict) -> str:
    """
    Pure data-processing function — no I/O, no blocking.
    Separated out so it's easy to unit test.
    """
    shopping_results = results.get("shopping_results", [])
    organic          = results.get("organic_results", [])[:SERP_ORGANIC_LIMIT_ANALYZE]

    shopping_signals = []
    for p in shopping_results[:SERP_SHOPPING_LIMIT]:
        shopping_signals.append({
            "title":      p.get("title"),
            "price":      p.get("price"),
            "rating":     p.get("rating"),
            "reviews":    p.get("reviews"),
            "source":     p.get("source"),
            "badge":      p.get("badge"),
            "snippet":    p.get("snippet"),
            "extensions": p.get("extensions", []),
        })

    organic_signals = []
    for r in organic:
        organic_signals.append({
            "title":               r.get("title"),
            "snippet":             r.get("snippet"),
            "source":              r.get("source"),
            "detected_extensions": r.get("detected_extensions", {}),
            "rating":  _dig(r, "rich_snippet", "top", "detected_extensions", "rating"),
            "reviews": _dig(r, "rich_snippet", "top", "detected_extensions", "reviews"),
            "price":   _dig(r, "rich_snippet", "top", "detected_extensions", "price"),
        })

    answer_box = results.get("answer_box", {})
    featured_snippet = {
        "title":   answer_box.get("title"),
        "snippet": answer_box.get("snippet"),
        "list":    answer_box.get("list", []),
        "type":    answer_box.get("type"),
    }

    paa     = [q.get("question") for q in results.get("related_questions", [])]
    related = [s.get("query")    for s in results.get("related_searches",  [])]

    all_titles = [p.get("title", "") for p in shopping_results[:6]] + \
                 [r.get("title", "") for r in organic[:5]]
    words = [
        w.lower()
        for t in all_titles
        for w in re.findall(r"\b\w{4,}\b", t)
        if w.lower() not in STOPWORDS
    ]
    keyword_freq = Counter(words).most_common(SERP_KEYWORD_TOP_N)

    # Prices as real numbers, sorted, so range/positioning aren't
    # order-dependent or comparing strings.
    prices = sorted(
        p for p in (_price_to_float(x.get("price")) for x in shopping_results)
        if p is not None
    )
    price_context = {
        "prices_found": prices,
        "price_range":  f"{prices[0]} – {prices[-1]}" if len(prices) > 1 else (prices[0] if prices else None),
        "positioning":  (
            "budget"    if len(prices) > 1 and prices[-1] > prices[0] * 2 else
            "premium"   if len(prices) > 1 and prices[0] > prices[-1] * 0.7 else
            "mid-range"
        ) if prices else "unknown",
    }

    all_snippets = [p.get("snippet", "") for p in shopping_results if p.get("snippet")] + \
                   [r.get("snippet", "") for r in organic          if r.get("snippet")]
    badges     = sorted(set(p.get("badge") for p in shopping_results if p.get("badge")))
    extensions = sorted(set(ext for p in shopping_results for ext in p.get("extensions", [])))

    seo_hints = {
        "primary_keyword_candidate": query,
        "long_tail_seed_queries": [
            f"best {query} 2025",
            f"{query} buying guide",
            f"top rated {query}",
        ],
    }

    return json.dumps({
        "seo_brief_hints": seo_hints,
        "description_generation_context": {
            "product_name":     query,
            "featured_snippet": featured_snippet,
            "price_context":    price_context,
        },
        "copy_patterns": {
            "top_keyword_frequencies": keyword_freq,
            "competitor_snippets":     all_snippets[:SERP_SNIPPET_LIMIT],
            "trust_badges":            badges,
            "shipping_extensions":     extensions,
        },
        "buyer_intent_signals": {
            "paa_questions":    paa,
            "related_searches": related,
        },
        "shopping_listings": shopping_signals,
        "organic_listings":  organic_signals,
    }, indent=2)


@tool
async def analyze_product_serp(query: str) -> str:
    """
    Analyse product SERP. Extracts shopping signals, competitor copy,
    buyer intent, reviews, pricing, PAA questions, and related searches.
    Returns structured data for keyword extraction and content generation.
    """
    try:
        # Cached + threadpooled — SerpAPI is sync and metered, so repeat
        # queries within the TTL window don't re-hit the API.
        raw_results = await _cached_serp_results(query)
    except Exception as e:
        return json.dumps({"error": f"SerpAPI analysis failed: {e}"})

    # parsing is pure CPU work — fast enough to run inline
    return _parse_serp_results(query, raw_results)
# tools.py — keep your original tools, just fix the output format
from langchain_core.tools import tool
from langchain_community.utilities import SerpAPIWrapper
from langchain_tavily import TavilySearch
from collections import Counter
from dotenv import load_dotenv
import json, re

load_dotenv()

tavily_tool = TavilySearch(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
)

serp_wrapper = SerpAPIWrapper()


@tool
def serp_search(query: str) -> str:
    """Search Google via SerpAPI and return structured results for SEO analysis."""
    results  = serp_wrapper.results(query)
    organic  = results.get("organic_results", [])
    snippets = []
    for r in organic[:8]:
        snippets.append({
            "position": r.get("position"),
            "title":    r.get("title"),
            "snippet":  r.get("snippet"),
            "link":     r.get("link"),
        })
    return json.dumps(snippets, indent=2)


@tool
def analyze_product_serp(query: str) -> str:
    """
    Analyse product SERP. Extracts shopping signals, competitor copy,
    buyer intent, reviews, pricing, PAA questions, and related searches.
    Returns structured data for keyword extraction and content generation.
    """
    results          = serp_wrapper.results(query)
    shopping_results = results.get("shopping_results", [])
    organic          = results.get("organic_results", [])[:6]

    # ── Shopping / PLAs ───────────────────────────────────────────────────
    shopping_signals = []
    for p in shopping_results[:5]:
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

    # ── Organic results ───────────────────────────────────────────────────
    organic_signals = []
    for r in organic:
        rich      = r.get("rich_snippet", {})
        top_attrs = rich.get("top", {})
        organic_signals.append({
            "title":               r.get("title"),
            "snippet":             r.get("snippet"),
            "source":              r.get("source"),
            "detected_extensions": r.get("detected_extensions", {}),
            "rating":  top_attrs.get("detected_extensions", {}).get("rating"),
            "reviews": top_attrs.get("detected_extensions", {}).get("reviews"),
            "price":   top_attrs.get("detected_extensions", {}).get("price"),
        })

    # ── Featured snippet ──────────────────────────────────────────────────
    answer_box = results.get("answer_box", {})
    featured_snippet = {
        "title":   answer_box.get("title"),
        "snippet": answer_box.get("snippet"),
        "list":    answer_box.get("list", []),
        "type":    answer_box.get("type"),
    }

    # ── PAA + related searches ────────────────────────────────────────────
    paa     = [q.get("question") for q in results.get("related_questions", [])]
    related = [s.get("query")    for s in results.get("related_searches",  [])]

    # ── Keyword frequency from titles ─────────────────────────────────────
    all_titles   = [p.get("title", "") for p in shopping_results[:6]] + \
                   [r.get("title", "") for r in organic[:5]]
    words        = [w.lower() for t in all_titles for w in re.findall(r'\b\w{4,}\b', t)]
    keyword_freq = Counter(words).most_common(20)

    # ── Price context ─────────────────────────────────────────────────────
    prices = [p.get("price") for p in shopping_results if p.get("price")]
    price_context = {
        "prices_found": prices,
        "price_range":  f"{min(prices)} – {max(prices)}" if len(prices) > 1 else (prices[0] if prices else None),
        "positioning":  (
            "budget"   if len(prices) > 1 and prices[0] < prices[-1] * 0.5 else
            "premium"  if len(prices) > 1 and prices[0] > prices[-1] * 0.7 else
            "mid-range"
        ) if prices else "unknown",
    }

    # ── Badges + snippets ─────────────────────────────────────────────────
    all_snippets = [p.get("snippet", "") for p in shopping_results if p.get("snippet")] + \
                   [r.get("snippet", "") for r in organic          if r.get("snippet")]
    badges       = list(set(p.get("badge") for p in shopping_results if p.get("badge")))
    extensions   = list(set(ext for p in shopping_results for ext in p.get("extensions", [])))

    # ── SEO brief hints ── NEW: added for keyword JSON extraction ─────────
    seo_hints = {
        "primary_keyword_candidate":    query,
        "long_tail_seed_queries": [
            f"best {query} 2025",
            f"{query} buying guide",
            f"top rated {query}",
        ],
    }

    return json.dumps({
        "seo_brief_hints":    seo_hints,           # ← new

        "description_generation_context": {
            "product_name":     query,
            "featured_snippet": featured_snippet,
            "price_context":    price_context,
        },
        "copy_patterns": {
            "top_keyword_frequencies": keyword_freq,
            "competitor_snippets":     all_snippets[:6],
            "trust_badges":            badges,
            "shipping_extensions":     extensions,
        },
        "buyer_intent_signals": {
            "paa_questions":    paa,      # → user_intent_questions
            "related_searches": related,  # → long_tail_clusters
        },
        "shopping_listings": shopping_signals,
        "organic_listings":  organic_signals,
    }, indent=2)
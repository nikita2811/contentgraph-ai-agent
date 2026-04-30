from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import SerpAPIWrapper
import json
from dotenv import load_dotenv
load_dotenv()

# --- Tavily research tool ---
tavily_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
)

# --- SERP analysis tool ---
serp_wrapper = SerpAPIWrapper()
 
@tool
def serp_search(query: str) -> str:
    """Search Google via SerpAPI and return structured results for analysis."""
    results = serp_wrapper.results(query)
    print(results)
    organic = results.get("organic_results", [])
    snippets = []
    for r in organic[:8]:
        snippets.append({
            "position": r.get("position"),
            "title": r.get("title"),
            "snippet": r.get("snippet"),
            "link": r.get("link"),
        })
    return json.dumps(snippets, indent=2)




@tool
def analyze_product_serp(query: str) -> str:
    """
    Analyse product SERP for AI-powered description generation.
    Extracts: shopping signals, competitor copy, buyer intent, reviews, pricing.
    """
    results = serp_wrapper.results(query)

    # ── 1. Shopping / PLAs (Product Listing Ads) ──────────────────────────
    # These are the richest product signals — title patterns, pricing, ratings
    shopping_results = results.get("shopping_results", [])
    shopping_signals = []
    for p in shopping_results[:5]:
        shopping_signals.append({
            "title":        p.get("title"),
            "price":        p.get("price"),
            "rating":       p.get("rating"),
            "reviews":      p.get("reviews"),
            "source":       p.get("source"),        # brand/retailer
            "badge":        p.get("badge"),          # "Best Seller", "Top Pick"
            "snippet":      p.get("snippet"),        # short product blurb
            "extensions":   p.get("extensions", []), # "Free shipping", "In stock"
        })

     # ── 2. Organic results — competitor product page copy ─────────────────
    organic = results.get("organic_results", [])[:6]
    organic_signals = []
    for r in organic:
        rich = r.get("rich_snippet", {})
        top_attrs = rich.get("top", {})
        organic_signals.append({
            "title":           r.get("title"),
            "snippet":         r.get("snippet"),
            "source":          r.get("source"),
            # Structured data Google extracts from product pages
            "detected_extensions": r.get("detected_extensions", {}),
            "rating":          top_attrs.get("detected_extensions", {}).get("rating"),
            "reviews":         top_attrs.get("detected_extensions", {}).get("reviews"),
            "price":           top_attrs.get("detected_extensions", {}).get("price"),
        })
         # ── 3. Featured snippet — Google's chosen best answer ─────────────────
    answer_box = results.get("answer_box", {})
    featured_snippet = {
        "title":   answer_box.get("title"),
        "snippet": answer_box.get("snippet"),
        "list":    answer_box.get("list", []),   # bullet features if list type
        "table":   answer_box.get("table", {}),  # spec table if table type
        "type":    answer_box.get("type"),        # paragraph / list / table
    }

         # ── 4. Knowledge Graph — product entity facts ─────────────────────────
    kg = results.get("knowledge_graph", {})
    knowledge_graph = {
        "name":         kg.get("title"),
        "type":         kg.get("type"),
        "description":  kg.get("description"),
        "rating":       kg.get("rating"),
        "reviews":      kg.get("reviews"),
        # Key-value attributes Google structures (dimensions, material, etc.)
        "attributes":   {
            k: v for k, v in kg.items()
            if k not in ("title", "type", "description",
                         "rating", "reviews", "header_images")
            and isinstance(v, str)
        },
    }

    # ── 5. PAA — buyer intent questions ───────────────────────────────────
    # Directly map to "FAQ" or "You may wonder" section in descriptions
    paa = [q.get("question") for q in results.get("related_questions", [])]

    # ── 6. Related searches — feature & use-case keywords ─────────────────
    related = [s.get("query") for s in results.get("related_searches", [])]

    # ── 7. Title pattern analysis — word patterns across top results ───────
    all_titles = (
        [p.get("title", "") for p in shopping_results[:6]] +
        [r.get("title", "") for r in organic[:5]]
    )
    # Extract repeated keywords across titles (these are must-have keywords)
    words = [w.lower() for t in all_titles for w in re.findall(r'\b\w{4,}\b', t)]
    from collections import Counter
    keyword_freq = Counter(words).most_common(20)

    # ── 8. Price & positioning context ────────────────────────────────────
    prices = [
        p.get("price") for p in shopping_results if p.get("price")
    ]
    price_context = {
        "prices_found":   prices,
        "price_range":    f"{min(prices)} – {max(prices)}" if len(prices) > 1 else (prices[0] if prices else None),
        "positioning_hint": (
            "budget" if len(prices) > 1 and prices[0] < prices[-1] * 0.5
            else "premium" if len(prices) > 1 and prices[0] > prices[-1] * 0.7
            else "mid-range"
        ) if prices else "unknown",
    }

    # ── 9. Review sentiment keywords (from snippets) ──────────────────────
    # Pulls adjectives/power words competitors use in their copy
    all_snippets = (
        [p.get("snippet", "") for p in shopping_results if p.get("snippet")] +
        [r.get("snippet", "") for r in organic if r.get("snippet")]
    )
    combined_snippet_text = " ".join(filter(None, all_snippets))

    # ── 10. Badges & trust signals ────────────────────────────────────────
    badges = list(set(
        p.get("badge") for p in shopping_results if p.get("badge")
    ))
    extensions = list(set(
        ext for p in shopping_results
        for ext in p.get("extensions", [])
    ))

    # ── Final structured output for the AI writer agent ───────────────────
    analysis = {

        "description_generation_context": {
            "product_name":        query,
            "knowledge_graph":     knowledge_graph,
            "featured_snippet":    featured_snippet,
            "price_context":       price_context,
        },

        "copy_patterns": {
            "top_keyword_frequencies": keyword_freq,     # must-include keywords
            "competitor_snippets":     all_snippets[:6], # style & tone reference
            "combined_snippet_text":   combined_snippet_text,  # for NLP analysis
            "trust_badges":            badges,           # "Best Seller" etc.
            "shipping_extensions":     extensions,       # "Free shipping" etc.
        },

        "buyer_intent_signals": {
            "paa_questions":        paa,        # FAQ content for description
            "related_searches":     related,    # feature/use-case angles
        },

        "shopping_listings": shopping_signals,  # full PLA data
        "organic_listings":  organic_signals,   # full organic data
    }

    return json.dumps(analysis, indent=2)

    
from typing import TypedDict, List, Optional, Annotated
import operator
from langgraph.graph import StateGraph, END
from .agent import research_agent,serp_agent,writer_agent
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.types import RetryPolicy
from langchain_core.exceptions import LangChainException
import httpx
import json
from .cache import (
    get_node_cache, set_node_cache,
    get_pipeline_cache, set_pipeline_cache
)

class AgentState(TypedDict):

    # ── Inputs ──────────────────────────────────────────
    product_name:    str
    category:        str
    target_audience: str
    key_features:    List[str]
    tone:            str

        # Outputs
    research_output: Optional[str]
    serp_output:     Optional[str]   # parsed JSON dict
    content_output:  Optional[str]   # parsed JSON dict

    # ── Pipeline control ─────────────────────────────────
    messages:       Annotated[List[BaseMessage], operator.add]
    current_step:   str
    error:          Optional[str]
   
    

def run_research(state: AgentState) -> AgentState:
    """Node 1 — Deep research via Tavily."""
    print("\n🔍 [Research Agent] Starting...")
    product_name    = state.get("product_name", "")
    category        = state.get("category", "")
    target_audience = state.get("target_audience", "")
    key_features    = state.get("key_features", [])
    tone            = state.get("tone", "professional")

    prompt = f"""
    Research this product thoroughly for generating an eCommerce description.
    Product Name    : {product_name}
    Category        : {category}
    Target Audience : {target_audience} years
    Key Features    : {", ".join(key_features)}
    Tone            : {tone}
    Find:
    - Product benefits and use cases
    - Competitor positioning
    - Common customer pain points this product solves
    - Industry keywords and terminology
    - Trending features in this category
    """.strip()
    # ✅ Add this back
    cached, cache_key = get_node_cache("research", prompt)
    if cached:
        return {
            **state,
            "research_output": cached["output"],
            "current_step":    "serp_analysis",
            "error":           None,
            "messages":        [AIMessage(content=f"[Research Agent - cached]\n{cached['output']}")],
        }

   


    try:
        result = research_agent.invoke({
            "messages": [HumanMessage(content=prompt)]
        })
        output = result["messages"][-1].content
        print(f"✅ Research complete ({len(output)} chars)")

        set_node_cache(cache_key, {"output": output}, ttl=86400)

        return {
            **state,                            # ✅ carry forward all keys
            "research_output": output,
            "current_step":    "serp_analysis",
            "error":           None,
            "messages":        [AIMessage(content=f"[Research Agent]\n{output}")],
        }
    except Exception as e:
        print(f"❌ Research Agent failed: {e}")
        return {
            **state,
            "research_output": None,
            "current_step":    "error",
            "error":           str(e),
            "messages":        [],
        }


def run_serp_analysis(state: AgentState) -> AgentState:
    """Node 2 — SERP analysis via SerpAPI."""
    print("\n📊 [SERP Agent] Starting...")
    product_name = state.get("product_name", "")
    category     = state.get("category", "")
    search_query = f"{product_name} {category}".strip()

    prompt = (
        f"Analyse the product SERP for: '{search_query}'. "
        f"Research brief: {state.get('research_output', '')[:500]}"
    )
     # ── Check node cache ───────────────────────────────────
    cached, cache_key = get_node_cache("serp", prompt)
    if cached:
        return {
            **state,
            "serp_output":  cached["output"],
            "current_step": "writing",
            "error":        None,
            "messages":     [AIMessage(content=f"[SERP Agent - cached]\n{json.dumps(cached['output'])}")],
        }

    try:
        result = serp_agent.invoke({"messages": [HumanMessage(content=prompt)]})
        output = result["messages"][-1].content

        # try:
        #     parsed = json.loads(output)
        # except json.JSONDecodeError:
        #     parsed = {"raw": output}

        # ── Store in node cache ────────────────────────────
        set_node_cache(cache_key, {"output": output}, ttl=3600)  # 1hr


        return {
            **state,                            # ✅ keeps research_output alive
            "serp_output":  output,      # ✅ dict, not string
            "current_step": "writing",
            "error":        None,
            "messages":     [AIMessage(content=f"[SERP Agent]\n{output}")],
        }
    except Exception as e:
        print(f"❌ SERP Agent failed: {e}")
        return {**state, "current_step": "error", "error": str(e), "messages": []}


def run_writer(state: AgentState) -> AgentState:
    """Node 3 — Content generation."""
    print("\n✍️  [Writer Agent] Starting...")
    prompt = f"""
    Product Name    : {state.get('product_name')}
    Category        : {state.get('category')}
    Target Audience : {state.get('target_audience')} years
    Key Features    : {", ".join(state.get('key_features', []))}
    Tone            : {state.get('tone')}
    === RESEARCH BRIEF ===
    {state.get('research_output', 'N/A')}
    === SERP BRIEF ===
    {json.dumps(state.get('serp_output', {}), indent=2)}
    """.strip()
    # ── Check node cache ───────────────────────────────────
    cached, cache_key = get_node_cache("writer", prompt)
    if cached:
        return {
            **state,
            "content_output": cached["output"],
            "current_step":   "done",
            "error":          None,
            "messages":       [AIMessage(content=f"[Writer Agent - cached]\n{json.dumps(cached['output'])}")],
        }

    try:
        result = writer_agent.invoke({
            "messages": [HumanMessage(content=prompt)]
        })
        output = result["messages"][-1].content

        # # Parse JSON string → dict
        # try:
        #     parsed_output = json.loads(output.strip().strip("```json").strip("```"))
        # except json.JSONDecodeError:
        #     parsed_output = {"raw": output}
         # ── Store in node cache ────────────────────────────
        set_node_cache(cache_key, {"output": output}, ttl=86400)  # 24hrs

        return {
            **state,                            # ✅ carry forward
            "content_output": output,
            "current_step":   "done",
            "error":          None,
            "messages":       [AIMessage(content=f"[Writer Agent]\n{output}")],
        }
    except Exception as e:
        print(f"❌ Writer Agent failed: {e}")
        return {**state, "current_step": "error", "error": str(e), "messages": []}
 


def route_after_research(state: AgentState) -> str:
    """Router — after research always go to SERP."""
    if state.get("error"):
        return END
    return "serp_analysis"
 
 
def route_after_serp(state: AgentState) -> str:
    """Router — after SERP always go to writing."""
    if state.get("error"):
        return END
    return "writer"

def build_pipeline() -> StateGraph:

    retry_policy = RetryPolicy(
     max_attempts=4,
     initial_interval=1.0,    # start with 1s wait
     backoff_factor=2.0,      # double each attempt: 1s, 2s, 4s, 8s
     jitter=True,             # ✅ adds randomness to each wait
     retry_on=(           # ✅ only retry on these
        httpx.HTTPStatusError,   # 429, 500, 503
        LangChainException,
        TimeoutError,
     )
    )
    graph = StateGraph(AgentState)
 
    # Add nodes
    graph.add_node("research", run_research,retry=retry_policy)
    graph.add_node("serp_analysis", run_serp_analysis,retry=retry_policy)
    graph.add_node("writer", run_writer,retry=retry_policy)
 
    # Entry point
    graph.set_entry_point("research")
 
    # Conditional edges
    graph.add_conditional_edges("research", route_after_research, {
        "serp_analysis": "serp_analysis",
        END: END,
    })
    graph.add_conditional_edges("serp_analysis", route_after_serp, {
        "writer": "writer",
        END: END,
    })
    graph.add_edge("writer", END)

    return graph.compile()

def run_pipeline(product_details:dict) -> dict:
    """Run the full 3-agent pipeline for a given topic."""

    # ── 1. Check pipeline cache first ─────────────────────
    cached, pipe_key = get_pipeline_cache(product_details)
    if cached:
        return cached  # ⚡ entire pipeline skipped
    pipeline = build_pipeline()

     # ── All keys must be initialised ─────────────────────
    initial_state: AgentState = {
        # Inputs
        "product_name":    product_details.get("product_name", ""),
        "category":        product_details.get("category", ""),
        "target_audience": str(product_details.get("target_audience", "")),
        "key_features":    product_details.get("key_features", []),
        "tone":            product_details.get("tone", "professional"),
         "research_output":None,
         "serp_output":None,
         "content_output":None,
        # Control
        "messages":     [],
        "current_step": "research",
        "error":        None,
    }
 
   
 
    print(f"\n🚀 Pipeline starting for topic: '{initial_state['product_name']}'")
    print("=" * 60)
 
    final_state = pipeline.invoke(initial_state)
     # ── 3. Cache final result (only on success) ────────────
    if not final_state.get("error"):
        set_pipeline_cache(pipe_key, final_state)

    print("\n" + "=" * 60)
    print("🎉 Pipeline complete!")
    return final_state


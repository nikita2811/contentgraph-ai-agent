# agentstate.py
from typing import TypedDict, List, Optional, Annotated
import operator
import json
import asyncio
import httpx

from langgraph.graph import StateGraph, END
from langgraph.types import RetryPolicy
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.exceptions import LangChainException

# ✅ import the async helpers, not the raw agents
from .agent import run_research_agent, run_serp_agent, run_writer_agent
from .cache import (
    get_node_cache, set_node_cache,
    get_pipeline_cache, set_pipeline_cache,
)


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Inputs
    product_name:     str
    category:         str
    target_audience:  str
    key_features:     List[str]
    tone:             str
    regenerate:       bool

    # Outputs
    research_output:  Optional[str]
    serp_output:      Optional[str]
    content_output:   Optional[str]

    # Pipeline control
    messages:         Annotated[List[BaseMessage], operator.add]
    current_step:     str
    error:            Optional[str]


# ── Node 1: Research ──────────────────────────────────────────────────────────

async def run_research(state: AgentState) -> AgentState:
    """Node 1 — Deep research via Tavily."""
    print("\n🔍 [Research Agent] Starting...")

    product_details = {
        "product_name":    state["product_name"],
        "category":        state["category"],
        "target_audience": state["target_audience"],
        "key_features":    state["key_features"],
        "tone":            state["tone"],
    }

    prompt = (
        f"Research this product thoroughly for generating an eCommerce description.\n"
        f"Product Name    : {state['product_name']}\n"
        f"Category        : {state['category']}\n"
        f"Target Audience : {state['target_audience']}\n"
        f"Key Features    : {', '.join(state['key_features'])}\n"
        f"Tone            : {state['tone']}\n"
        f"Find:\n"
        f"- Product benefits and use cases\n"
        f"- Competitor positioning\n"
        f"- Common customer pain points this product solves\n"
        f"- Industry keywords and terminology\n"
        f"- Trending features in this category"
    )

    # ── Cache check ───────────────────────────────────────
    if not state.get("regenerate"):
        cached, cache_key = get_node_cache("research", prompt)
        if cached:
            print("⚡ [Research Agent] Cache HIT")
            return {
                **state,
                "research_output": cached["output"],
                "current_step":    "serp_analysis",
                "error":           None,
                "messages":        [AIMessage(content=f"[Research Agent - cached]\n{cached['output']}")],
            }
    else:
        _, cache_key = get_node_cache("research", prompt)  # key only, no hit

    try:
        # ✅ async call via helper from agents.py
        output = await run_research_agent(product_details)
        print(f"✅ Research complete ({len(output)} chars)")

        set_node_cache(cache_key, {"output": output}, ttl=86400)

        return {
            **state,
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


# ── Node 2: SERP Analysis ─────────────────────────────────────────────────────

async def run_serp_analysis(state: AgentState) -> AgentState:
    """Node 2 — SERP analysis via SerpAPI + Tavily."""
    print("\n📊 [SERP Agent] Starting...")

    prompt = (
        f"Analyse the product SERP for: '{state['product_name']} {state['category']}'. "
        f"Research brief: {(state.get('research_output') or '')[:500]}"
    )

    # ── Cache check ───────────────────────────────────────
    if not state.get("regenerate"):
        cached, cache_key = get_node_cache("serp", prompt)
        if cached:
            print("⚡ [SERP Agent] Cache HIT")
            return {
                **state,
                "serp_output":  cached["output"],
                "current_step": "writing",
                "error":        None,
                "messages":     [AIMessage(content=f"[SERP Agent - cached]\n{json.dumps(cached['output'])}")],
            }
    else:
        _, cache_key = get_node_cache("serp", prompt)

    try:
        # ✅ async call — returns parsed dict via _safe_parse_json in agents.py
        product_details = {
            "product_name": state["product_name"],
            "category":     state["category"],
        }
        output = await run_serp_agent(
            research_output=state.get("research_output", ""),
            product_details=product_details,
        )

        set_node_cache(cache_key, {"output": output}, ttl=3600)

        return {
            **state,
            "serp_output":  output,
            "current_step": "writing",
            "error":        None,
            "messages":     [AIMessage(content=f"[SERP Agent]\n{json.dumps(output)}")],
        }

    except Exception as e:
        print(f"❌ SERP Agent failed: {e}")
        return {
            **state,
            "current_step": "error",
            "error":        str(e),
            "messages":     [],
        }


# ── Node 3: Writer ────────────────────────────────────────────────────────────

async def run_writer(state: AgentState) -> AgentState:
    """Node 3 — Content generation."""
    print("\n✍️  [Writer Agent] Starting...")

    prompt = (
        f"Product Name    : {state['product_name']}\n"
        f"Category        : {state['category']}\n"
        f"Target Audience : {state['target_audience']}\n"
        f"Key Features    : {', '.join(state.get('key_features', []))}\n"
        f"Tone            : {state['tone']}\n"
        f"=== RESEARCH BRIEF ===\n"
        f"{state.get('research_output', 'N/A')}\n"
        f"=== SERP BRIEF ===\n"
        f"{json.dumps(state.get('serp_output', {}), indent=2)}"
    )

    # ── Cache check ───────────────────────────────────────
    if not state.get("regenerate"):
        cached, cache_key = get_node_cache("writer", prompt)
        if cached:
            print("⚡ [Writer Agent] Cache HIT")
            return {
                **state,
                "content_output": cached["output"],
                "current_step":   "done",
                "error":          None,
                "messages":       [AIMessage(content=f"[Writer Agent - cached]\n{json.dumps(cached['output'])}")],
            }
    else:
        _, cache_key = get_node_cache("writer", prompt)

    try:
        product_details = {
            "product_name":    state["product_name"],
            "category":        state["category"],
            "target_audience": state["target_audience"],
            "key_features":    state["key_features"],
            "tone":            state["tone"],
        }
        # ✅ async call — returns parsed dict
        output = await run_writer_agent(
            serp_output=state.get("serp_output", {}),
            product_details=product_details,
        )

        set_node_cache(cache_key, {"output": output}, ttl=86400)

        return {
            **state,
            "content_output": output,
            "current_step":   "done",
            "error":          None,
            "messages":       [AIMessage(content=f"[Writer Agent]\n{json.dumps(output)}")],
        }

    except Exception as e:
        print(f"❌ Writer Agent failed: {e}")
        return {
            **state,
            "current_step": "error",
            "error":        str(e),
            "messages":     [],
        }


# ── Routers ───────────────────────────────────────────────────────────────────

def route_after_research(state: AgentState) -> str:
    return END if state.get("error") else "serp_analysis"

def route_after_serp(state: AgentState) -> str:
    return END if state.get("error") else "writer"


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    retry_policy = RetryPolicy(
        max_attempts=4,
        initial_interval=1.0,
        backoff_factor=2.0,
        jitter=True,
        retry_on=(
            httpx.HTTPStatusError,
            LangChainException,
            TimeoutError,
        ),
    )

    graph = StateGraph(AgentState)

    graph.add_node("research",     run_research,     retry=retry_policy)
    graph.add_node("serp_analysis", run_serp_analysis, retry=retry_policy)
    graph.add_node("writer",       run_writer,       retry=retry_policy)

    graph.set_entry_point("research")

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


# ── Public entry point ────────────────────────────────────────────────────────

async def run_pipeline(product_details: dict) -> dict:
    """
    Async entry point for the full 3-agent pipeline.
    Call with: await run_pipeline(product_details)
    """
    regenerate = product_details.get("regenerate", False)

    # ── Pipeline cache check ──────────────────────────────
    if not regenerate:
        cached, pipe_key = get_pipeline_cache(product_details)
        if cached:
            print("⚡ Pipeline cache HIT")
            return cached
    else:
        print("🔄 Regenerate requested — skipping cache")
        _, pipe_key = get_pipeline_cache(product_details)

    pipeline = build_pipeline()

    initial_state: AgentState = {
        "product_name":    product_details.get("product_name", ""),
        "category":        product_details.get("category", ""),
        "target_audience": str(product_details.get("target_audience", "")),
        "key_features":    product_details.get("key_features", []),
        "tone":            product_details.get("tone", "professional"),
        "regenerate":      regenerate,
        "research_output": None,
        "serp_output":     None,
        "content_output":  None,
        "messages":        [],
        "current_step":    "research",
        "error":           None,
    }

    print(f"\n🚀 Pipeline starting for: '{initial_state['product_name']}'")
    print("=" * 60)

    # ✅ ainvoke — non-blocking, works with async nodes
    final_state = await pipeline.ainvoke(initial_state)

    if not final_state.get("error"):
        set_pipeline_cache(pipe_key, final_state)

    print("=" * 60)
    print("🎉 Pipeline complete!")
    return final_state
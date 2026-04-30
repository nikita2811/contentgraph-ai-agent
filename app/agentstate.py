from typing import TypedDict, List, Optional, Annotated
import operator
from langgraph.graph import StateGraph, END
from .agent import research_agent,serp_agent,writer_agent
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

class AgentState(TypedDict):

    # ── Inputs ──────────────────────────────────────────
    product_name:    str
    category:        str
    target_audience: str
    key_features:    List[str]
    tone:            str

    # ── Intermediate results ─────────────────────────────
    query:                  Optional[str]
    raw_content:            Optional[str]
    extracted_keywords:     Optional[List[str]]
    keyword_research_data:  Optional[str]
    serp_output:            Optional[str]        # ← add this
    # ── Final output ─────────────────────────────────────
    final_content:  Optional[str]

    # ── Pipeline control ─────────────────────────────────
    messages:       Annotated[List[BaseMessage], operator.add]
    current_step:   str
    error:          Optional[str]
   
    

def run_research(state: AgentState) -> AgentState:
    """Node 1 — Deep research via Tavily."""
    print("\n [Research Agent] Starting...")
     # ── Safe extraction from state ───────────────────────
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
    try:
        result = research_agent.invoke({
            "messages": [HumanMessage(content=prompt)]
        })
        output = result["messages"][-1].content
        print(f"✅ Research complete ({len(output)} chars)")

        # ── Must return ALL state keys ────────────────────
        return {
            **state,                              # ← carry forward all existing keys
            "raw_content":  output,
            "current_step": "serp_analysis",
            "error":        None,
            "messages":     [AIMessage(content=f"[Research Agent]\n{output}")],
        }
    except Exception as e:
        print(f"❌ Research Agent failed: {e}")
        return {
            **state,
            "raw_content":  None,
            "current_step": "error",
            "error":        str(e),
            "messages":     [],
        }
   

def run_serp_analysis(state:AgentState) -> AgentState:
    """Node 2 — SERP analysis via SerpAPI."""
    print("\n📊 [SERP Agent] Starting...")
    product_name = state.get("product_name", "")
    category     = state.get("category", "")
    search_query = f"{product_name} {category}".strip()

    try:
        result = serp_agent.invoke({
            "messages": [HumanMessage(content=
                f"Analyse the product SERP for: '{search_query}'. "
                f"Research brief: {state.get('raw_content', '')[:500]}"  # pass context
            )]
        })
        output = result["messages"][-1].content
        return {
            **state,
            "serp_output":  output,
            "query":        search_query,
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
    {state.get('raw_content', 'N/A')}

    === SERP BRIEF ===
    {state.get('serp_output', 'N/A')}

    Write:
    1. SEO Title (50-60 chars)
    2. Meta Description (150-160 chars)
    3. Short Description (2-3 sentences)
    4. Long Description (with H2s, bullet features, FAQ)
    5. Tags (15-20 comma-separated)
    """.strip()

    try:
        result = writer_agent.invoke({
            "messages": [HumanMessage(content=prompt)]
        })
        output = result["messages"][-1].content

        return {
            **state,
            "final_content": output,
            "current_step":  "done",
            "error":         None,
            "messages":      [AIMessage(content=f"[Writer Agent]\n{output}")],
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
    graph = StateGraph(AgentState)
 
    # Add nodes
    graph.add_node("research", run_research)
    graph.add_node("serp_analysis", run_serp_analysis)
    graph.add_node("writer", run_writer)
 
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
    pipeline = build_pipeline()

     # ── All keys must be initialised ─────────────────────
    initial_state: AgentState = {
        # Inputs
        "product_name":    product_details.get("product_name", ""),
        "category":        product_details.get("category", ""),
        "target_audience": str(product_details.get("target_audience", "")),
        "key_features":    product_details.get("key_features", []),
        "tone":            product_details.get("tone", "professional"),
         # Intermediate — all None at start
        "query":                 None,
        "raw_content":           None,
        "extracted_keywords":    None,
        "keyword_research_data": None,
        "serp_output":           None,

        # Output — None at start
        "final_content": None,
        # Control
        "messages":     [],
        "current_step": "research",
        "error":        None,
    }
 
   
 
    print(f"\n🚀 Pipeline starting for topic: '{initial_state['product_name']}'")
    print("=" * 60)
 
    final_state = pipeline.invoke(initial_state)
 
    print("\n" + "=" * 60)
    print("🎉 Pipeline complete!")
    return final_state


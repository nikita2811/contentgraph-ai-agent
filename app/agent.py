import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from .tools import tavily_tool, serp_search, analyze_product_serp
from langchain_core.runnables import RunnableConfig
load_dotenv()


# ── LLM ───────────────────────────────────────────────────────────────────────

def _build_llm() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY")
    model   = os.getenv("GEMINI_MODEL","gemini-1.5-flash")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set")
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0,
        streaming=False,
    )

llm = _build_llm()


# ── Agent 1: Research ─────────────────────────────────────────────────────────

research_agent = create_react_agent(
    model=llm,
    tools=[tavily_tool],
    prompt=SystemMessage(content=(
        "TASK: Competitor Analysis via Tavily search.\n"
        "OBJECTIVE: Identify 5 top-rated/most-reviewed competitors based on provided profile.\n"
        "SEARCH STRATEGY: Combine product_name + target_audience + key_features for high-intent queries.\n"
        "OUTPUT SCHEMA (Strict):\n"
        "1. [Brand] | Rating: [X/5] ([N] reviews) | Price: [$] | Buy: [Link]\n"
        "   - Features: [List 3 key features]\n"
        "ANALYSIS: Concise comparison vs User Product + 1 Best Pick recommendation.\n"
        "TONE: Professional. No conversational filler."
    )),
)


# ── Agent 2: SERP ─────────────────────────────────────────────────────────────

serp_agent = create_react_agent(
    model=llm,
    tools=[serp_search, analyze_product_serp],
    prompt=SystemMessage(content="""TASK: SEO/SERP Intelligence Extraction.
INPUT: Research Data + SERP Results.
OUTPUT: Valid JSON ONLY. No markdown. No prose. No backticks.

SCHEMA:
{
  "primary_keyword": "str",
  "secondary_keywords": ["list"],
  "long_tail_clusters": ["list"],
  "user_intent_questions": ["list"],
  "semantic_entities": ["list"],
  "pla_listings": [{"title":"str","price":"str","rating":"str","reviews":"str","source":"str","badge":"str"}],
  "price_context": {"price_range":"str","positioning":"str","trust_badges":[]}
}

INSTRUCTION: Parse input to populate schema. Focus on high-volume search intent and commercial competition. If a field is missing, use null or [].
"""),
)


# ── Agent 3: Writer ───────────────────────────────────────────────────────────

writer_agent = create_react_agent(
    model=llm,
    tools=[],
    prompt=SystemMessage(content="""TASK: SEO Content Generation.
INPUT: SERP Brief JSON.
OUTPUT: Valid JSON ONLY. No markdown. No prose. No backticks.

SCHEMA:
{
  "seo_title": "str",
  "meta_description": "str",
  "tags": ["list"],
  "h1": "str",
  "h2_subheadings": ["list"],
  "intro_paragraph": "str",
  "product_roundup": [{"product_name": "str", "paragraph": "str"}],
  "faq": [{"question": "str", "answer": "str"}],
  "conclusion": "str"
}

INSTRUCTION: Generate professional, high-conversion SEO copy based on the provided SERP brief. Ensure product_roundup reflects competitive analysis accurately. Strictly follow the JSON schema.
"""),
)


# ── JSON parser ───────────────────────────────────────────────────────────────

def _safe_parse_json(raw: str, agent: str) -> dict:
    """
    Strips markdown fences if the LLM added them despite instructions,
    then parses JSON. Raises ValueError with context on failure.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"{agent} returned invalid JSON: {e}\nRaw output: {raw[:300]}"
        ) from e


# ── Async helpers (imported by agentstate.py) ─────────────────────────────────

# agent.py


  # ── Usage extractor ───────────────────────────────────────────────────────────

def _extract_usage_from_messages(messages: list) -> dict:
    prompt_tokens     = 0
    completion_tokens = 0
    total_tokens      = 0

    for msg in messages:
        usage = getattr(msg, "usage_metadata", None)
        if not usage:
            continue
        prompt_tokens     += usage.get("input_tokens",  0)
        completion_tokens += usage.get("output_tokens", 0)
        total_tokens      += usage.get("total_tokens",  0)

    return {
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens":      total_tokens,
    }


# ── Async helpers ─────────────────────────────────────────────────────────────

async def run_research_agent(product_details: dict, config: RunnableConfig | None = None) -> tuple[str, dict]:
    message = HumanMessage(content=json.dumps(product_details))
    try:
        response = await research_agent.ainvoke(
            {"messages": [message]},
            config=config,
        )
        usage = _extract_usage_from_messages(response["messages"])
        return response["messages"][-1].content, usage
    except Exception as e:
        raise RuntimeError(f"Research agent failed: {e}") from e


async def run_serp_agent(research_output: str, product_details: dict, config: RunnableConfig | None = None) -> tuple[str, dict]:
    payload = json.dumps({"product_details": product_details, "research": research_output})
    try:
        response = await serp_agent.ainvoke(
            {"messages": [HumanMessage(content=payload)]},
            config=config,
        )
        usage = _extract_usage_from_messages(response["messages"])
        return response["messages"][-1].content, usage
    except Exception as e:
        raise RuntimeError(f"SERP agent failed: {e}") from e


async def run_writer_agent(
    serp_output: dict,
    product_details: dict,
    config: RunnableConfig | None = None,
    system_prompt_override: str | None = None,
) -> tuple[str, dict]:
    payload = json.dumps({"product_details": product_details, "serp_brief": serp_output})
    try:
        response = await writer_agent.ainvoke(
            {"messages": [HumanMessage(content=payload)]},
            config=config,
        )
        usage = _extract_usage_from_messages(response["messages"])
        return response["messages"][-1].content, usage
    except Exception as e:
        raise RuntimeError(f"Writer agent failed: {e}") from e
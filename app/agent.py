import asyncio
import json
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from .tools import tavily_search_async, analyze_product_serp

load_dotenv()

LLM_TIMEOUT_SECONDS = 20


# ── LLM ───────────────────────────────────────────────────────────────────────

def _build_llm(model_override: str | None = None) -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY")
    model = model_override or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set")
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0,
        streaming=False,
    )


llm = _build_llm()
# Deterministic extraction/formatting stages don't need the heavier model —
# point this at a faster tier once you've confirmed current Gemini flash
# naming/availability.
llm_fast = _build_llm(model_override=os.getenv("GEMINI_MODEL_FAST", os.getenv("GEMINI_MODEL", "gemini-1.5-flash")))


# ── Prompts (module-level constants — built once, reused across calls) ────

RESEARCH_SYSTEM_MESSAGE = SystemMessage(content=(
    "TASK: Competitor Analysis via Tavily search.\n"
    "OBJECTIVE: Identify 5 top-rated/most-reviewed competitors based on provided profile.\n"
    "SEARCH STRATEGY: Combine product_name + target_audience + key_features for high-intent queries.\n"
    "OUTPUT SCHEMA (Strict):\n"
    "1. [Brand] | Rating: [X/5] ([N] reviews) | Price: [$] | Buy: [Link]\n"
    "   - Features: [List 3 key features]\n"
    "ANALYSIS: Concise comparison vs User Product + 1 Best Pick recommendation.\n"
    "TONE: Professional. No conversational filler."
))

SERP_SYSTEM_MESSAGE = SystemMessage(content="""TASK: SEO/SERP Intelligence Extraction.
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
""")

WRITER_SYSTEM_MESSAGE = SystemMessage(content="""TASK: SEO Content Generation.
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
""")


# ── Agent 1: Research — kept as ReAct, since query count/strategy is genuinely
#    open-ended (not a fixed tool sequence like stages 2 and 3) ────────────

research_agent = create_react_agent(
    model=llm,
    tools=[tavily_search_async],
    prompt=RESEARCH_SYSTEM_MESSAGE,
)


# ── JSON parser ───────────────────────────────────────────────────────────────

def _safe_parse_json(raw: str, agent: str) -> dict:
    """
    Strips markdown fences if the LLM added them despite instructions,
    then parses JSON. Tolerates trailing "extra data" after a valid JSON
    object (e.g. duplicated output or trailing prose from the model).
    Raises ValueError with context on failure.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0].strip()

    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(cleaned)
        return obj
    except json.JSONDecodeError as e:
        raise ValueError(
            f"{agent} returned invalid JSON: {e}\nRaw output: {raw[:300]}"
        ) from e


# ── Usage extractor ─────────────────────────────────────────────────────────

def _extract_usage_from_messages(messages: list) -> dict:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    for msg in messages:
        usage = getattr(msg, "usage_metadata", None)
        if not usage:
            continue
        prompt_tokens += usage.get("input_tokens", 0)
        completion_tokens += usage.get("output_tokens", 0)
        total_tokens += usage.get("total_tokens", 0)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


# ── Async helpers (imported by agentstate.py) ────────────────────────────────

async def run_research_agent(product_details: dict, config: RunnableConfig | None = None) -> tuple[str, dict]:
    message = HumanMessage(content=json.dumps(product_details))
    merged_config = {**(config or {}), "recursion_limit": 6}
    try:
        response = await asyncio.wait_for(
            research_agent.ainvoke({"messages": [message]}, config=merged_config),
            timeout=LLM_TIMEOUT_SECONDS * 2,  # ReAct loop — allow room for tool round-trips
        )
    except Exception as e:
        raise RuntimeError(f"Research agent failed: {e}") from e

    usage = _extract_usage_from_messages(response["messages"])
    raw_text = response["messages"][-1].content
    return raw_text, usage


async def run_serp_agent(
    research_output: str,
    product_details: dict,
    config: RunnableConfig | None = None,
) -> tuple[dict, dict]:
    """
    Direct tool call + single completion — no ReAct loop. The tool to call
    (analyze_product_serp, on the primary product name) is known in advance,
    so there's no decision for the model to make here.
    """
    serp_data = await analyze_product_serp.ainvoke(
        {"query": product_details.get("product_name", "")}
    )
    payload = json.dumps({
        "product_details": product_details,
        "research": research_output,
        "serp_data": serp_data,
    })

    try:
        response = await asyncio.wait_for(
            llm_fast.ainvoke(
                [SERP_SYSTEM_MESSAGE, HumanMessage(content=payload)],
                config=config,
            ),
            timeout=LLM_TIMEOUT_SECONDS,
        )
    except Exception as e:
        raise RuntimeError(f"SERP agent failed: {e}") from e

    usage = _extract_usage_from_messages([response])
    parsed = _safe_parse_json(response.content, "serp_agent")
    return parsed, usage


async def run_writer_agent(
    serp_output: dict,
    product_details: dict,
    config: RunnableConfig | None = None,
    system_prompt_override: str | None = None,
) -> tuple[dict, dict]:
    """
    Direct completion — no tools, so no ReAct loop needed. system_prompt_override
    (used by the rewrite node) fully replaces the schema prompt, same behavior
    as before.
    """
    payload = json.dumps({"product_details": product_details, "serp_brief": serp_output})
    system_msg = (
        SystemMessage(content=system_prompt_override)
        if system_prompt_override
        else WRITER_SYSTEM_MESSAGE
    )

    try:
        response = await asyncio.wait_for(
            llm_fast.ainvoke(
                [system_msg, HumanMessage(content=payload)],
                config=config,
            ),
            timeout=LLM_TIMEOUT_SECONDS,
        )
    except Exception as e:
        raise RuntimeError(f"Writer agent failed: {e}") from e

    usage = _extract_usage_from_messages([response])
    parsed = _safe_parse_json(response.content, "writer_agent")
    return parsed, usage
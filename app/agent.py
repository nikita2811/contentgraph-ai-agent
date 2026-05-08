# agents.py
import os, json, asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from .tools import tavily_tool, serp_search, analyze_product_serp

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)



# ── Agent 1: Research Agent (unchanged — was already fast) ────────────────────
research_agent = create_react_agent(
    model=llm,
    tools=[tavily_tool],
    prompt=(
        "TASK: Competitor Analysis via Tavily search.\n"
        "OBJECTIVE: Identify 5 top-rated/most-reviewed competitors based on provided profile.\n"
        "SEARCH STRATEGY: Combine 'product_name' + 'target_audience' + 'key_features' for high-intent queries.\n"
        "OUTPUT SCHEMA (Strict):\n"
        "1. [Brand] | Rating: [X/5] ([N] reviews) | Price: [$] | Buy: [Link]\n"
        "   - Features: [List 3 key features]\n"
        "ANALYSIS: Concise comparison vs User Product + 1 'Best Pick' recommendation.\n"
        "TONE: Professional. No conversational filler."
    ),
)

# ── Agent 2: SERP Agent (unchanged tools — just updated output format) ────────
serp_agent = create_react_agent(
    model=llm,
    tools=[serp_search, analyze_product_serp],
    prompt="""TASK: SEO/SERP Intelligence Extraction.
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

INSTRUCTION: Parse input to populate schema. Focus on high-volume search intent and commercial competition. If a field is missing, use null or []."""
)

# ── Agent 3: Content Writer (unchanged — was already fast) ────────────────────
writer_agent = create_react_agent(
    model=llm,
    tools=[],  # No tools needed as this is a generation-only node
    prompt="""TASK: SEO Content Generation.
INPUT: SERP Brief JSON.
OUTPUT: Valid JSON ONLY. No markdown. No prose. No backticks.

SCHEMA:
{
  "seo_title": "str",
  "meta_description": "str",
  "tags":["list],
  "h1": "str",
  "h2_subheadings": ["list"],
  "intro_paragraph": "str",
  "product_roundup": [{"product_name": "str", "paragraph": "str"}],
  "faq": [{"question": "str", "answer": "str"}],
  "conclusion": "str"
}

INSTRUCTION: Generate professional, high-conversion SEO copy based on the provided SERP brief. Ensure the 'product_roundup' reflects the competitive analysis accurately. Strictly follow the JSON schema."""
)






from langchain.agents import initialize_agent,AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from .tools import tavily_tool,serp_search,analyze_product_serp
import os
import asyncio
from langgraph.prebuilt import create_react_agent

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key="AIzaSyCRXpaj1oOvEqNWJhDqc0olL7xvCqZcb1w",
    temperature=0
)


# --- Agent 1: Research Agent (Tavily) ---
research_agent = create_react_agent(
    model=llm,
    tools=[tavily_tool],
    state_modifier=(
        "You are an expert research analyst. "
        "Your job is to deeply research a given topic using Tavily. "
        "Collect facts, statistics, authoritative sources, and key insights. "
        "Structure your findings clearly: summary, key facts, notable sources."
    ),
)

# --- Agent 2: SERP Analysis Agent ---
serp_agent = create_react_agent(
    model=llm,
    tools=[serp_search, analyze_product_serp],
    state_modifier=(
        "You are an SEO and SERP analysis expert. "
        "Given a topic, analyse: top-ranking pages, featured snippets, "
        "People Also Ask questions, common heading patterns, and content gaps. "
        "Return a structured SERP brief: competitor angles, target keywords, "
        "recommended headings, and word-count benchmarks."
    ),
)

# --- Agent 3: Content Writer Agent ---
writer_agent = create_react_agent(
    model=llm,
    tools=[],          # pure generation — no tools needed
    state_modifier=(
        "You are a world-class SEO content writer. "
        "You receive a research brief and a SERP analysis brief. "
        "Write a comprehensive, engaging, SEO-optimised article that: "
        "covers all key facts from research, addresses PAA questions, "
        "follows recommended heading structure, and targets identified keywords. "
        "Format: H1 title, intro, H2/H3 sections, FAQ section, conclusion. "
        "Tone: authoritative yet accessible."
    ),
)
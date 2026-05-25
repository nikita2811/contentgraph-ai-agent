# ContentGraph AI Agent

An agentic content generation pipeline that researches the web, extracts keywords, and produces high-quality marketing copy for any product — powered by LangGraph, Google Gemini, and Tavily.

---

## What It Does

Given a product name, key features, category, and tone, the pipeline:

1. **Researches** — queries Tavily for live SERP data about the product and market
2. **Filters** — scores and deduplicates search results by credibility and relevance
3. **Extracts keywords** — uses KeyBERT to pull semantically relevant terms
4. **Generates content** — passes research + keywords to Gemini via a structured LangGraph pipeline
5. **Validates output** — checks for empty, too-short, or hallucinated content before returning

---

## Architecture

```
POST /generate
      │
      ▼
┌─────────────────────────────────────────────────┐
│                  LangGraph Pipeline              │
│                                                  │
│  ┌──────────┐    ┌──────────┐    ┌───────────┐  │
│  │  SERP    │───▶│ Keyword  │───▶│  Content  │  │
│  │  Node    │    │  Node    │    │  Writer   │  │
│  │ (Tavily) │    │(KeyBERT) │    │  (Gemini) │  │
│  └──────────┘    └──────────┘    └───────────┘  │
│       │                                  │       │
│  SERP Validator                  Output Validator│
└─────────────────────────────────────────────────┘
      │
      ▼
  JSONResponse  ←  Redis Cache (Upstash, TTL 1h)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Pydantic v2 |
| Agent Orchestration | LangGraph |
| LLM | Google Gemini (via `langchain-google-genai`) |
| Web Research | Tavily Search API |
| Keyword Extraction | KeyBERT |
| Caching | Upstash Redis |
| Auth | JWT (`python-jose`) |
| Containerisation | Docker + docker-compose |
| Observability | LangSmith *(coming soon)* |
| Evals | deepeval *(coming soon)* |

---

## Project Structure

```
contentgraph-ai-agent/
├── app/
│   ├── agentstate.py        # LangGraph pipeline definition
│   ├── main.py              # FastAPI app + routes
│   ├── cache.py             # Upstash Redis LLM cache
│   ├── auth/
│   │   └── jwt_verify.py    # JWT service token verification
│   └── validators/
│       ├── serp_validator.py    # SERP scoring + deduplication
│       └── output_validator.py  # Content quality checks
├── evals/                   # deepeval test suite (coming soon)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── main.py                  # entrypoint
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker (optional)
- API keys for: Google Gemini, Tavily, Upstash Redis

### 1. Clone the repo

```bash
git clone https://github.com/nikita2811/contentgraph-ai-agent.git
cd contentgraph-ai-agent
```

### 2. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
# LLM
GOOGLE_API_KEY=your_gemini_api_key

# Search
TAVILY_API_KEY=your_tavily_api_key

# Cache
UPSTASH_REDIS_REST_URL=your_upstash_url
UPSTASH_REDIS_REST_TOKEN=your_upstash_token

# Auth
SERVICE_JWT_SECRET=your_jwt_secret

# Observability (optional)
LANGCHAIN_API_KEY=ls__your_langsmith_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=contentgraph-ai-agent
```

### 3. Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

### 4. Run with Docker

```bash
docker-compose up --build
```

---

## API Reference

### `GET /health`

```json
{ "status": "healthy", "version": "1.0.0" }
```

---

### `POST /generate`

Requires `Authorization: Bearer <service_token>` header.

**Request body:**

```json
{
  "product_name": "TrailMax Running Shoes",
  "key_features": ["waterproof", "lightweight", "grip sole"],
  "category": "ecom",
  "tone": "energetic",
  "target_audience": "outdoor runners aged 25-40",
  "regenerate": false
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `product_name` | string | ✅ | Name of the product |
| `key_features` | string[] | ✅ | Features the content must mention |
| `category` | string | ✅ | e.g. `ecom`, `saas`, `lifestyle` |
| `tone` | string | ✅ | `professional`, `energetic`, `friendly`, `luxury` |
| `target_audience` | string | | Who the content is aimed at |
| `regenerate` | bool | | Skip cache and regenerate fresh content |

**Response:**

```json
{
  "product_name": "TrailMax Running Shoes",
  "final_content": "Conquer every trail with TrailMax...",
  "serp": "Top competitors include...",
  "status": "success"
}
```

**Error responses:**

| Code | Meaning |
|---|---|
| `401` | Missing or invalid JWT token |
| `422` | Output validation failed (empty, too short, missing features) |
| `500` | Pipeline execution error |

---

## Design Decisions

**Why LangGraph over a simple chain?**
LangGraph lets each step (SERP fetch, keyword extraction, content generation) be an independent node with its own state, retry logic, and conditional routing. This makes the pipeline easier to debug, test, and extend.

**Why validate SERP results before the LLM?**
The LLM's context window is a resource to protect. Feeding 10 mediocre search results is worse than feeding 3 high-quality ones — it dilutes signal and wastes tokens. The SERP validator scores results by domain credibility and Tavily's relevance score before anything reaches Gemini.

**Why Redis caching at the LLM layer?**
Identical or near-identical product queries (e.g. "Running Shoes" asked twice) should not re-run the full pipeline. LLM responses are cached with a 1-hour TTL keyed on the input hash. The `regenerate: true` flag bypasses this when fresh content is needed.

**Why JWT auth?**
This pipeline is designed to be called service-to-service (e.g. from an e-commerce backend), not directly by end users. A short-lived JWT service token is the appropriate auth primitive for that pattern.

**CORS `allow_origins: ["*"]`**
Intentionally open for development. In production, replace with specific allowed domains or drive it from an environment variable.

---

## Roadmap

- [ ] LangSmith tracing for per-node token usage and latency
- [ ] deepeval evaluation suite (faithfulness, relevance, keyword coverage)
- [done] Async pipeline execution to unblock FastAPI workers
- [ ] Streaming response endpoint (`/generate/stream`)
- [ ] Research memory cache — store high-scoring SERP results in a vector DB to reduce redundant Tavily calls
- [ ] GitHub Actions CI with eval regression tests

---

## License

MIT

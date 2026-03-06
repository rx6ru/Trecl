# Trecl

A multi-agent job research tool. Given a company name, Trecl scrapes the web, embeds company data into a vector database, searches the company's GitHub organization for open issues and stale PRs, finds job listings, filters everything against the user's preferences, and generates tailored cold outreach emails.

Built with [LangGraph](https://github.com/langchain-ai/langgraph), [Qdrant](https://qdrant.tech/), and [Cerebras](https://cerebras.ai/).

---

## How It Works

Trecl runs 6 agents in a stateful directed acyclic graph (DAG):

```
START
  |
  v
data_ingester        Scrape 8 web sources, chunk, embed into Qdrant, synthesize company profile
  |
  |--- (parallel) ---|
  v                  v
job_decoder     github_analyst     Find open listings / Search GitHub org for issues and PRs
  |                  |
  |--- (fan-in) -----|
  v
opportunity_curator              Filter against anti-persona, classify into 3 tiers, rank
  v
  === PAUSE ===                  User selects which opportunities to pursue
  v
pain_synthesizer                 Query VectorDB for context, deduce pain points, pitch a project
  v
writer                           Generate one cold email per target + a fused combined email
  v
END
```

**Key mechanics:**

- **Fan-out / fan-in**: The job decoder and GitHub analyst run in parallel after the data ingester finishes. The opportunity curator waits for both to complete before executing.
- **Human-in-the-loop**: The graph suspends after curation and presents ranked opportunities in the terminal. The user selects targets by number, and the graph resumes.
- **RAG pipeline**: The data ingester embeds web-scraped context into Qdrant with typed metadata. Downstream agents query the VectorDB with source-type and recency filters for context-augmented reasoning.
- **ReAct sub-graph**: The GitHub analyst is a compiled sub-graph with its own isolated state, 7 tools, and a 4-layer guardrail system that prevents the LLM from hallucinating repository names or guessing label filters.

---

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM | Cerebras `gpt-oss-120b` via `langchain-cerebras` |
| Embeddings | Google Gemini `gemini-embedding-001` via `google-genai` |
| Vector DB | Qdrant Cloud |
| Web search | Tavily API |
| GitHub | PyGitHub |
| Structured output | Pydantic + LangChain `with_structured_output()` |
| State persistence | SQLite checkpointer |
| Retry logic | tenacity (exponential backoff, transient-error-only) |
| Observability | LangSmith |
| Testing | pytest (smoke, unit, integration, health) |

---

## Project Structure

```
src/
├── main.py                        Graph construction, HITL CLI, DAG edges
├── core/
│   ├── config.py                  Environment loading, round-robin key manager, feature flags
│   ├── state.py                   TreclState, GithubAnalystState, typed schemas
│   └── knowledge_store.py         Qdrant client + Gemini embeddings
├── llm/
│   └── model.py                   ChatCerebrasWithRetry proxy, MockLLM, tenacity retry
├── agents/
│   ├── data_ingester.py           Scrape -> Chunk -> Embed -> Synthesize
│   ├── job_decoder.py             Tavily job search + structured extraction
│   ├── github_analyst.py          ReAct sub-graph with 7 tools and guardrails
│   ├── opportunity_curator.py     Anti-persona filtering + tier classification
│   ├── pain_synthesizer.py        RAG-augmented pain points + project pitch
│   └── writer.py                  Per-target + fused cold email generation
└── tools/
    ├── search.py                  Tavily search wrappers
    ├── knowledge.py               LangChain @tool bridge to Qdrant
    └── github.py                  7 GitHub tools + 4-layer guardrail system
```

---

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
git clone https://github.com/rx6ru/trecl.git
cd trecl
uv sync
```

### Environment Variables

Create a `.env` file in the project root:

```env
# LLM (Cerebras) — comma-separated for round-robin rotation
CEREBRAS_API_KEY=your_key_1,your_key_2

# Web search (Tavily) — comma-separated
TAVILY_API_KEY=your_key_1,your_key_2

# GitHub API
GITHUB_ACCESS_TOKEN=your_github_pat

# Qdrant Cloud
QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key

# Google Gemini (for embeddings)
GEMINI_API_KEYS=your_gemini_api_key

# Feature flags (set to True to use mock data and skip API calls)
USE_MOCK_SEARCH=False
USE_MOCK_LLM=False
USE_MOCK_GITHUB=False
```

All API keys support comma-separated values for round-robin rotation across requests.

---

## Usage

```bash
uv run src/main.py
```

The CLI will prompt for:

1. **Company name** — the startup to research (e.g., `TraceRoot.AI`)
2. **Anti-persona** — roles or skills to exclude (e.g., `No ML research, no model training`)

The pipeline runs for approximately 2-3 minutes, then presents curated opportunities for selection.

### Example Output

```
Available Opportunities:

[1] Founding Engineer - Full-Stack & AI Debugging Platform
    Tier:      Tier 1: Active Listing
    Action:    Apply via LinkedIn, highlighting Python/Docker/LLM projects.

[2] Software Engineering Intern - Full-Stack AI Agent
    Tier:      Tier 1: Active Listing
    Action:    Submit application, attaching FastAPI and LangGraph projects.

[3] Implement abstract Agent base class (#84)
    Tier:      Tier 2: OSS Pitch
    Action:    Fork the repo, add a Python AbstractBaseAgent class, submit PR.

Enter the numbers of the opportunities to target (e.g., '1, 3'): 1,3
```

After selection, the system generates pain-point analysis, a custom project pitch, and per-target cold emails.

---

## Testing

```bash
# Run all non-health tests (no API calls needed)
uv run pytest -m "not health"

# Run the full suite including live API health checks
uv run pytest
```

The test suite is organized into 4 tiers:

- **Smoke** — imports, config loading, state schema validation
- **Unit** — agent node logic, retry behavior, GitHub tool guardrails, knowledge store operations
- **Integration** — full graph execution with MockLLM, end-to-end Qdrant ingestion and search
- **Health** — live connectivity checks for Cerebras, Gemini, GitHub, Qdrant, and Tavily APIs

---

## License

MIT

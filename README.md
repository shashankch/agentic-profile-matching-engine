# Agentic Profile Matching Engine

<p align="left">
  <a href="https://github.com/shashankch/agentic-profile-matching-engine/actions/workflows/ci.yml"><img src="https://github.com/shashankch/agentic-profile-matching-engine/actions/workflows/ci.yml/badge.svg" alt="Python CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg" alt="Python Version"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Linter: Ruff"></a>
  <a href="CONVENTIONS.md"><img src="https://img.shields.io/badge/Conventions-Architectural-purple.svg" alt="Conventions"></a>
  <a href="CONTRIBUTING.md"><img src="https://img.shields.io/badge/Contributing-Welcome-green.svg" alt="Contributing"></a>
</p>


## Overview

This project implements an interactive **Agentic Profile Matching Engine** built with [LangGraph]. It acts as an intelligent AI recruiter assistant that parses job descriptions, searches resumes, executes a multi-round screening cascade, and allows interactive constraints refinement mid-conversation.

This project is fully **standalone** and encapsulates the document processing, vector storage ([ChromaDB]), and hybrid search (semantic + BM25 Okapi) logic replicated from [Milestone 1 (llm_file_system_assistant)](https://github.com/shashankch/llm_file_system_assistant) and [Milestone 2 (rag_profile_matching)](https://github.com/shashankch/rag-profile-match) to operate independently.

Detailed design diagrams, specifications, and requirements can be found in the [docs/](docs/) directory. Engineering conventions and contributing rules are maintained in [CONVENTIONS.md](CONVENTIONS.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Core Features & Architecture

- **LangGraph Agent Workflow**: Orchestrates requirements extraction, coarse search, deep profile diagnostics, hiring recommendations, and human feedback loops.
- **Production Observability & Node Tracing**: Structured JSON logging (`JsonFormatter`, `get_logger`) and `@trace_node` decorator instrumenting all 9 graph nodes with millisecond-level execution duration tracking (`node_start`, `node_end`, `node_error`) plus opt-in **Langfuse** and **OpenTelemetry** tracing.
- **RAG Evaluation Pipeline**: Benchmark evaluation suite measuring hybrid retrieval Recall@K, Mean Reciprocal Rank (MRR), and screening report faithfulness against ground-truth evaluation scenarios (`data/eval_scenarios.json`).
- **Layout-Aware PDF Ingestion**: **PyMuPDF** (`fitz`) layout-sorted multi-column text extraction with opt-in **Unstructured.io** PDF partitioning (`USE_UNSTRUCTURED=True`) and graceful `pypdf` fallbacks.
- **[Model Context Protocol (MCP)][mcp] Dual-Mode Gateway**: Supports running direct local modules (Local Mode) or interfacing via stdio JSON-RPC 2.0 with separate MCP servers (MCP Mode) to handle file processes, directory-watching ingestions, and background thread-pool batch files parsing.
- **Protocol-Enabled Search Engine**: Features a dedicated search MCP server supporting:
  - Live web searching via Tavily API (with fallback mock profiles for fictitious sandbox resumes).
  - Semantic vector search over ChromaDB databases returning similarity scores, document chunks, and matching metrics.
  - Mock candidate notes fetching from internal HR screens.
- **Multi-Round Screening**:
  - **Round 1 (Coarse Filter)**: Quick constraints filtering and 60/40 hybrid semantic-keyword ranking across all resumes.
  - **Round 2 (Deep Analysis)**: LLM profile auditing highlighting candidates' core strengths, gaps, and improvements.
  - **Round 3 (Final Screening)**: Automatic Hire/No-Hire recommendations and tailored technical screening questions.
- **[Streamlit] Recruiter Dashboard**: Interactive user interface providing real-time sidebar constraint updates, conversational chat log feed, and structured candidate comparison matrix tabs.
- **Free API Integrations**: Built to use 100% free developer tiers for LLM orchestration (Groq API using GPT OSS/Qwen models or Google Gemini Pro) alongside local, self-hosted embeddings.

---

## Project Structure

```text
agentic_profile_matching/
├── data/
│   └── eval_scenarios.json        # Ground-truth evaluation scenarios benchmark dataset
├── src/
│   └── agentic_profile_matching/  # Packaged Module Namespace
│       ├── __init__.py            # Package initialization marker
│       ├── config.py              # Ingestion paths and model configurations
│       ├── observability.py       # JsonFormatter, get_logger, and @trace_node instrumentation
│       ├── fs_tools.py            # Replicated filesystem utility layer
│       ├── fs_client.py           # Unified client gateway (Direct vs. MCP Mode)
│       ├── filesystem_mcp_server.py # MCP Protocol server exposing FS tools
│       ├── search_mcp_server.py   # Secondary MCP server for Multi-MCP orchestration
│       ├── mcp_client.py          # Thread-safe persistent connection MCP manager
│       ├── resume_rag.py          # Chunking and ChromaDB ingestion pipeline
│       ├── job_matcher.py         # Semantic/BM25 hybrid query ranking
│       ├── generate_dataset.py    # Local mock resumes generation tool
│       ├── agent/                 # Modular LangGraph Agent package
│       │   ├── __init__.py        # Graph assembly & MemorySaver compiler
│       │   ├── state.py           # Shared AgentState definitions
│       │   ├── prompts.py         # Decoupled LLM node prompt templates
│       │   ├── nodes.py           # Vetting and report compilation node logic
│       │   └── routers.py         # Conditional edge input routing logic
│       ├── matching_agent.py      # Compatibility shim re-exporting agent package
│       ├── tools.py               # Custom AI tools (compare, extract, qgen)
│       ├── app.py                 # Interactive Streamlit GUI dashboard app
│       └── run_scenarios.py       # Automated scenarios runner script
├── tests/                         # Unit & integration tests directory
│   ├── __init__.py
│   ├── eval/                      # RAG evaluation pipeline benchmark tests
│   │   ├── __init__.py
│   │   ├── test_retrieval_recall.py     # Recall@K and MRR evaluation suite
│   │   └── test_response_faithfulness.py # Report faithfulness & groundedness tests
│   ├── test_fs_tools.py           # Unit tests for filesystem utilities
│   ├── test_ingestion_service.py  # Unit tests for IngestionService layer
│   ├── test_job_matcher.py        # Unit tests for job matching algorithm
│   ├── test_tools.py              # Unit tests for assessment tools
│   ├── test_observability.py      # Unit tests for structured logging and trace decorator
│   └── test_mcp.py                # Unit tests for MCP server/client & fallbacks
├── docs/
│   ├── architecture.md            # Detailed technical design specifications
│   ├── ARCHITECTURE_DECISIONS.md  # Formal Architecture Decision Records (ADRs 001-008)
│   ├── CONVENTIONS.md             # Engineering conventions & architectural standards
│   ├── ROADMAP.md                 # Phased implementation roadmap with status tracking
│   ├── state_machine.mermaid      # Mermaid diagram code of LangGraph state machine
│   └── state_machine.png          # Rendered visual image of the state machine
├── .github/
│   ├── workflows/ci.yml           # GitHub Actions CI workflow config
│   └── CONTRIBUTING.md            # GitHub contributing guidelines
├── pyproject.toml                 # PEP 621 compliant package setup configurations
├── Dockerfile                     # Streamlit app containerization config
├── requirements.txt               # Dependencies list
├── CONVENTIONS.md                 # Root engineering conventions document
├── CONTRIBUTING.md                # Root development setup and PR guide
├── ROADMAP.md                     # Root roadmap document with emoji status key
├── CHANGELOG.md                   # Chronological log of notable changes
└── README.md                      # Project documentation
```

---

## Engineering Highlights & Architectural Decisions

The codebase incorporates 7 formal Architecture Decision Records (**ADRs**), documented in detail in [`docs/ARCHITECTURE_DECISIONS.md`](docs/ARCHITECTURE_DECISIONS.md):

- **ADR-001: MCP Dual-Mode Gateway**: Decouples in-process Python tools (`USE_MCP=False`) from FastMCP `stdio` JSON-RPC servers (`USE_MCP=True`), enabling seamless local development and production microservice deployment.
- **ADR-002: `TypedDict` State Management**: Avoids Pydantic runtime validation overhead during LangGraph state transitions while enforcing strict Pydantic V2 models (`JobRequirementsOutput`, `DeepScreenOutput`) at LLM response boundaries.
- **ADR-003: `BaseVectorStore` Structural Protocol**: Leverages Python `typing.Protocol` for structural subtyping without inheritance coupling, enabling interchangeable ChromaDB and Qdrant vector backends.
- **ADR-004: BM25 Index Caching**: Caches tokenized BM25 sparse index matrices on `JobMatcher` with MD5 document fingerprint validation, cutting hybrid search latency from ~200ms down to ~0.1ms.
- **ADR-005: Idempotent `upsert()` Ingestion**: Uses section-scoped deterministic chunk keys (`{filename}_chunk_{idx}`) to ensure 100% idempotent document re-ingestion without collection bloat.
- **ADR-006: Celery + Redis Worker Architecture**: Non-blocking background worker queue (`tasks.py`, `celery_app.py`, `docker-compose.yml`) for asynchronous document processing and deep screening audits.
- **ADR-007: Production Observability & Tracing**: Structured JSON logging (`JsonFormatter`, `get_logger`) and `@trace_node` decorator instrumenting all 9 workflow nodes with millisecond duration tracking and opt-in **Langfuse** / **OpenTelemetry** tracing.

---

## Project Roadmap & Guidelines

Details on implementation progress, milestones, and status tracking (using clean status emojis: `✅` Completed, `⏳` In Progress, `⬜` Planned) are maintained in [ROADMAP.md](ROADMAP.md).

For development standards and contributing instructions, refer to:
- 📐 **[Engineering Conventions](CONVENTIONS.md)**: Architectural patterns, Pydantic V2 schemas, error boundaries, and testing rules.
- 🤝 **[Contributing Guide](CONTRIBUTING.md)**: Local setup, running unit tests (`pytest`), Ruff formatting, and PR submission checklist.

---

## Setup & Execution

### 1. Environment Setup

It is recommended to use a standard virtual environment or `uv` for package management:

```bash
# Create a virtual environment using Python 3.10+
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install the package in editable mode along with requirements
pip install -e .

# Or if using uv:
# uv pip install -e .
```

### 2. Configure Local Secrets

Create a `.env` file in the root of the project to add your keys and configuration parameters:

```env
GROQ_API_KEY="your-groq-api-key"
GEMINI_API_KEY="your-gemini-api-key"

# MCP Protocol Configuration:
# - Set to False (default) to run using standard direct local tools
# - Set to True to launch MCP servers and route file operations via JSON-RPC
USE_MCP=False
```

### 3. Pipeline Ingestion

Run the following commands sequentially to build the candidate resume database:

```bash
# A. Generate the mock resume dataset (31 files)
python -m agentic_profile_matching.generate_dataset

# B. Ingest and vector-index the resume chunks into ChromaDB
python -m agentic_profile_matching.resume_rag
```

### 4. Launch the Interactive App & Scenarios

Run the Streamlit application to start the conversational interface:

```bash
streamlit run src/agentic_profile_matching/app.py
```

Or run the automated scenario suite:

```bash
python -m agentic_profile_matching.run_scenarios
```

### 5. Running Unit Tests, RAG Evaluation & Quality Checks

Run the test suite to verify code modules, MCP protocol scenarios, and RAG evaluation benchmarks:

```bash
# Run standard unit tests
pytest tests/

# Run RAG Evaluation Pipeline (Recall@K, MRR & Faithfulness)
pytest tests/eval/ -m eval

# Run Ruff linter and formatter checks
ruff check src/ tests/
ruff format src/ tests/
```


### 6. [Docker] Deployment

Build and launch the complete stack (Streamlit Web Dashboard, Redis Broker, and Celery Background Worker) using Docker Compose:

```bash
# Launch all services in the background (Redis, Celery Worker, Streamlit UI)
docker compose up -d

# Or build and run a single standalone container
docker build -t agentic-profile-matching .
docker run -p 8501:8501 --env-file .env agentic-profile-matching
```

---

## Rate Limit & Token Usage Management

To prevent `429` rate limit exceptions and TPM/RPM limits exhaustion on free API tiers, the engine implements five layers of safeguards:
1. **Tiered Cascading Pipeline**: 
   - **Round 1 (Coarse Filtering)** is executed 100% locally using [Sentence Transformers] and BM25 indexing (costing 0 API requests and 0 LLM tokens). This narrows the search space from 100+ resumes down to the Top 10.
   - **Round 2 & 3 (Deep Analysis & Recommendations)** are only executed on the narrowed candidates (Top 10 and Top 5 respectively).
2. **Sequential Requests Throttling**: A delay (`config.THROTTLE_DELAY = 1.5` seconds) is enforced between sequential LLM screening calls to space out queries and stay under RPM limits.
3. **Token Input Truncation**: Resume texts are truncated to a safe maximum length of 12,000 characters (approx. 3,000 tokens) before prompt generation to prevent TPM spikes.
4. **Compact Structured Outputs**: Node prompts enforce concise JSON structures, keeping output tokens under ~200 per call.
5. **Exponential Backoff Retry**: Every LLM function is wrapped in `execute_with_retry`, which catches 429 errors and retries with doubling delays (up to 5 attempts).

---

## Interactive Explainability & State Persistence

1. **Ranking Changes Explanation**: When the user refines requirements mid-conversation (e.g. `"make Python a must-have"`), the agent compares the previous shortlist with the new one and uses the LLM to explain why candidates rose, fell, or entered/left the shortlist.
2. **LangGraph Checkpointer**: The graph is compiled with `MemorySaver` in-memory checkpointing. This allows native LangGraph session state and chat history tracking via `thread_id` parameters.
3. **Structured Candidate Profile Layout**: Shortlisted candidate matches are mapped dynamically with structured metadata properties including matching scores, experience years, education targets, and matched skills tags.
4. **Interactive Graph Response Logs**: The recruiter agent appends conversational summary messages containing top match details and ranking shift reasons back to the chat state messages array to ensure conversational history syncs across UI reruns.

---

<!-- References -->

[langgraph]: https://langchain-ai.github.io/langgraph/
[streamlit]: https://streamlit.io/
[ChromaDB]: https://www.trychroma.com/
[mcp]: https://modelcontextprotocol.io/
[Docker]: https://www.docker.com/
[Sentence Transformers]: https://sbert.net/
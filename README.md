# Agentic Profile Matching Engine

## Overview

This project implements an interactive **Agentic Profile Matching Engine** built with LangGraph. It acts as an intelligent AI recruiter assistant that parses job descriptions, searches resumes, executes a multi-round screening cascade, and allows interactive constraints refinement mid-conversation.

This project is fully **standalone** and encapsulates the document processing, vector storage (ChromaDB), and hybrid search (semantic + BM25 Okapi) logic replicated from [Milestone 1 (llm_file_system_assistant)](https://github.com/shashankch/llm_file_system_assistant) and [Milestone 2 (rag_profile_matching)](https://github.com/shashankch/rag-profile-match) to operate independently.

Detailed design diagrams, specifications, and requirements can be found in the [docs/](docs/) directory.

---

## Core Features & Architecture

- **LangGraph Agent Workflow**: Orchestrates requirements extraction, coarse search, deep profile diagnostics, hiring recommendations, and human feedback loops.
- **Model Context Protocol (MCP) Dual-Mode Gateway**: Supports running direct local modules (Local Mode) or interfacing via stdio JSON-RPC 2.0 with separate MCP servers (MCP Mode) to handle file processes, directory-watching ingestions, and background thread-pool batch files parsing.
- **Protocol-Enabled Search Engine**: Features a dedicated search MCP server supporting:
  - Keyless live web searching via DuckDuckGo (with fallback mock profiles for fictitious sandbox resumes).
  - Semantic vector search over ChromaDB databases returning similarity scores, document chunks, and matching metrics.
  - Mock candidate notes fetching from internal HR screens.
- **Multi-Round Screening**:
  - **Round 1 (Coarse Filter)**: Quick constraints filtering and 60/40 hybrid semantic-keyword ranking across all resumes.
  - **Round 2 (Deep Analysis)**: LLM profile auditing highlighting candidates' core strengths, gaps, and improvements.
  - **Round 3 (Final Screening)**: Automatic Hire/No-Hire recommendations and tailored technical screening questions.
- **Streamlit Recruiter Dashboard**: Interactive user interface providing real-time sidebar constraint updates, conversational chat log feed, and structured candidate comparison matrix tabs.
- **Free API Integrations**: Built to use 100% free developer tiers for LLM orchestration (Groq API using GPT OSS/Qwen models or Google Gemini Pro) alongside local, self-hosted embeddings.

---

## Project Structure

```text
agentic_profile_matching/
├── src/
│   └── agentic_profile_matching/  # Packaged Module Namespace
│       ├── __init__.py            # Package initialization marker
│       ├── config.py              # Ingestion paths and model configurations
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
├── tests/                         # Unit tests directory
│   ├── __init__.py
│   ├── test_fs_tools.py           # Unit tests for filesystem utilities
│   ├── test_job_matcher.py        # Unit tests for job matching algorithm
│   ├── test_tools.py              # Unit tests for assessment tools
│   └── test_mcp.py                # Unit tests for MCP server/client & fallbacks
├── docs/
│   ├── architecture.md            # Detailed technical design specifications
│   ├── state_machine.mermaid      # Mermaid diagram code of LangGraph state machine
│   └── state_machine.png          # Rendered visual image of the state machine
├── pyproject.toml                 # PEP 621 compliant package setup configurations
├── Dockerfile                     # Streamlit app containerization config
├── .github/workflows/ci.yml       # GitHub Actions CI workflow config
├── requirements.txt               # Dependencies list
├── ROADMAP.md                     # Project implementation roadmap and future backlog
└── README.md                      # Project documentation
```


---

## Project Roadmap

Details on implementation progress, milestones, and future goals (like multi-agent debate protocols, fairness/bias auditing, and native ATS integrations) are maintained in the root [ROADMAP.md](ROADMAP.md).

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

### 5. Running Unit Tests

Run the test suite to verify code modules, including the new MCP protocol test scenarios:

```bash
# Run all tests
pytest tests/

# Run the MCP server/client specific tests
python -m unittest tests/test_mcp.py
```


### 6. Docker Deployment

Build and run the Streamlit application inside a container:

```bash
# Build the Docker image
docker build -t agentic-profile-matching .

# Run the container (passes your local environment keys)
docker run -p 8501:8501 --env-file .env agentic-profile-matching
```

---

## Rate Limit & Token Usage Management

To prevent `429` rate limit exceptions and TPM/RPM limits exhaustion on free API tiers, the engine implements five layers of safeguards:
1. **Tiered Cascading Pipeline**: 
   - **Round 1 (Coarse Filtering)** is executed 100% locally using Sentence Transformers and BM25 indexing (costing 0 API requests and 0 LLM tokens). This narrows the search space from 100+ resumes down to the Top 10.
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
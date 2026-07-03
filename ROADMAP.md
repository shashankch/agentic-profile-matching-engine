# Agentic Profile Matching Engine: Project Roadmap

This document outlines the phased plan for building the Agentic Profile Matching Engine, along with key upcoming backlog features.

---

## 📍 Core Implementation Phases

1. **Phase 1: Environment & Setup** [Completed]
   - Install dependencies (`langgraph`, `streamlit`, `langchain-groq`, `langchain-google-genai`).
   - Create a local `.env` file for credentials (`GROQ_API_KEY`, `GEMINI_API_KEY`).

2. **Phase 2: Dataset Ingestion & Retrieval** [Completed]
   - Generate synthetic candidate resume files (PDF, DOCX, TXT).
   - Ingest files, generate local embeddings, and populate ChromaDB/BM25 indexes.

3. **Phase 3: LangGraph Agent & State Machine** [Completed]
   - Design the `AgentState` schema and setup conversational graph memory.
   - Implement workflow nodes (parsing, search, deep screening, report generation, and human feedback interrupts).

4. **Phase 4: Assessment & Screening Tools** [Completed]
   - Build custom agent tools for requirements extraction, multi-profile side-by-side comparison matrices, and candidate-specific interview question generation.

5. **Phase 5: Streamlit UI & End-to-End Validation** [Completed]
   - Create the frontend dashboard (`app.py`) for dual-pane chat and shortlist inspection.
   - Test conversational refinement loops against test recruitment scenarios.

6. **Phase 6: MCP Server & Client Integration** [Completed]
   - Implement FastMCP filesystem server (`filesystem_mcp_server.py`) exposing files and resource namespaces (`resumes://`).
   - Add secondary search server (`search_mcp_server.py`) coordinating live web queries and ChromaDB vector calls.
   - Create synchronous async-bridged manager client (`mcp_client.py`) and unified gateway client (`fs_client.py`).

7. **Phase 7: Modular Graph Refactoring & Codebase Resilience** [Completed]
   - Break down monolithic 833-line agent file into decoupled packaged folder `agent/` (`state.py`, `prompts.py`, `nodes.py`, `routers.py`, `__init__.py`).
   - Add try-except bounds, capped experience extraction validation, and warning alert flags in state.

8. **Phase 8: Enterprise Abstractions & Background Workers** [Next Iteration]
   - Introduce `BaseVectorStore` interfaces to support pluggable ChromaDB, Qdrant, or pgvector.
   - Migrate state structures to strict Pydantic V2 schemas and structured outputs.
   - Implement `Celery` + `Redis` worker queue tasks for non-blocking UI operations.

---

## 🔮 Future Backlog (Post-Release)

- **Multi-Agent Consensus Loop**: Incorporate independent agent personas (e.g., Technical Screener vs. HR/Sourcing Screener) to run an internal review before final recommendation.
- **Bias & Fairness Auditing**: Add an automated checker to audit Job Descriptions for inclusive language and flag potential constraint biases.


# Agentic Profile Matching Engine: Project Roadmap

This document outlines the phased plan for building the Agentic Profile Matching Engine, along with key upcoming backlog features.

---

## 📍 Core Implementation Phases

1. **Phase 1: Environment & Setup** ✅
   - Install dependencies (`[langgraph]`, `[streamlit]`, `[langchain-groq]`, `[langchain-google-genai]`).
   - Create a local `.env` file for credentials (`GROQ_API_KEY`, `GEMINI_API_KEY`).

2. **Phase 2: Dataset Ingestion & Retrieval** ✅
   - Generate synthetic candidate resume files (PDF, DOCX, TXT).
   - Ingest files, generate local embeddings, and populate [ChromaDB] / BM25 indexes.

3. **Phase 3: LangGraph Agent & State Machine** ✅
   - Design the `AgentState` schema and setup conversational graph memory.
   - Implement workflow nodes (parsing, search, deep screening, report generation, and human feedback interrupts).

4. **Phase 4: Assessment & Screening Tools** ✅
   - Build custom agent tools for requirements extraction, multi-profile side-by-side comparison matrices, and candidate-specific interview question generation.

5. **Phase 5: Streamlit UI & End-to-End Validation** ✅
   - Create the frontend dashboard (`app.py`) for dual-pane chat and shortlist inspection.
   - Test conversational refinement loops against test recruitment scenarios.

6. **Phase 6: MCP Server & Client Integration** ✅
   - Implement [FastMCP] filesystem server (`filesystem_mcp_server.py`) exposing files and resource namespaces (`resumes://`).
   - Add secondary search server (`search_mcp_server.py`) coordinating live web queries and ChromaDB vector calls.
   - Create synchronous async-bridged manager client (`mcp_client.py`) and unified gateway client (`fs_client.py`).

7. **Phase 7: Modular Graph Refactoring & Codebase Resilience** ✅
   - Break down monolithic 833-line agent file into decoupled packaged folder `agent/` (`state.py`, `prompts.py`, `nodes.py`, `routers.py`, `__init__.py`).
   - Add try-except bounds, capped experience extraction validation, and warning alert flags in state.
   - Integrate automated local pre-commit hooks and GitHub Actions CI code quality lints using [Ruff].

8. **Phase 8: Enterprise Abstractions & Background Workers** ✅
   - **8.1 — Protocol / Business Logic Separation** ✅: Introduce a dedicated `IngestionService` layer to cleanly decouple file-processing business logic from the MCP protocol tool handlers.
   - **8.2 — Pluggable Vector Store Abstraction** ✅: Define a `BaseVectorStore` protocol, implement a default `ChromaVectorStore` with idempotent upsert ingestion and deterministic chunk IDs, and add a `QdrantVectorStore` stub to validate the abstraction boundary.
   - **8.3 — Store Injection & BM25 Index Caching** ✅: Wire `BaseVectorStore` into `ResumeRAGPipeline` and `JobMatcher` via constructor injection; cache the BM25 corpus index on construction to eliminate per-query O(n) rebuilds.
   - **8.4 — Type-Safe State Migration & Security Hardening** ✅: Migrate `AgentState` to `TypedDict`; migrate LLM output schemas to Pydantic V2 `BaseModel`; remove API credentials from serializable graph state.
   - **8.5 — Pydantic V2 LLM Output Contracts** ✅: Replace fragile regex-based JSON parsing of LLM responses with Pydantic V2 `model_validate_json()` with graceful fallback.
   - **8.6 — Celery + Redis Background Workers** ✅: Implement [Celery] + [Redis] async task queue for non-blocking deep screening and background ingestion; add Docker Compose orchestration.

9. **Phase 9: Production Observability, RAG Evaluation & Richer Ingestion** ✅
   - **9.1 — Structured Logging & Node Tracing Decorator** ✅: Introduce structured JSON logging (`JsonFormatter`, `get_logger`) and a `@trace_node` decorator with per-node latency timing and trace event correlation; replace diagnostic `print()` calls throughout.
   - **9.2 — Langfuse / OpenTelemetry Tracing Integration** ✅: Integrate **[Langfuse]** and **[OpenTelemetry]** as opt-in tracing backends wired dynamically through the `@trace_node` decorator with zero changes to node logic.
   - **9.3 — RAG Evaluation Pipeline (Recall & Faithfulness)** ✅: Incorporate automated evaluation pipelines (`tests/eval/`) measuring candidate retrieval recall@K, MRR, and LLM screening faithfulness against ground-truth benchmark scenarios (`data/eval_scenarios.json`).
   - **9.4 — PyMuPDF / Unstructured.io Ingestion Upgrade** ✅: Upgraded PDF ingestion using **[PyMuPDF]** (`fitz`) for layout-sorted multi-column parsing, with opt-in **[Unstructured.io]** support plugged through `fs_tools.py` and `IngestionService`.

10. **Phase 10: Architecture Documentation & Project Closeout** ✅
    - Published formal Architecture Decision Records (ADRs 001–008) in `docs/ARCHITECTURE_DECISIONS.md`.
    - Synchronized all repository documentation (`README.md`, `docs/ROADMAP.md`, `docs/CONVENTIONS.md`, `CONTRIBUTING.md`, `docs/architecture.md`) to reflect full `v1.0.0` production baseline.

---

## 🔮 Future Backlog (Post-Release)

- **Indirect Prompt Injection Defense**: Pre-screening sanitization of adversarial instructions in resume text.
- **PII Anonymization Layer**: Reversible candidate entity tokenization for privacy compliance.
- **Semantic Embedding Cache**: Sub-10ms repeat candidate screening cache.
- **Parallel Async Batch Screening**: Rate-limited semaphore concurrency for Round 2 audits.
- **Human-in-the-Loop Workflow Gate**: LangGraph `interrupt()` approval step before recruiter outreach.
- **Multi-Agent Consensus Loop**: Independent screener personas (Technical vs. HR/Sourcing) review before final recommendation.
- **Bias & Fairness Auditing**: Automated checker auditing JDs and evaluations for inclusive language and constraint biases.

---

<!-- References -->

[langgraph]: https://langchain-ai.github.io/langgraph/
[streamlit]: https://streamlit.io/
[langchain-groq]: https://github.com/langchain-ai/langchain/tree/master/libs/partners/groq
[langchain-google-genai]: https://github.com/langchain-ai/langchain/tree/master/libs/partners/google-genai
[ChromaDB]: https://www.trychroma.com/
[FastMCP]: https://github.com/modelcontextprotocol/python-sdk
[Ruff]: https://github.com/astral-sh/ruff
[Qdrant]: https://qdrant.tech/
[pgvector]: https://github.com/pgvector/pgvector
[Celery]: https://docs.celeryq.dev/
[Redis]: https://redis.io/
[Langfuse]: https://langfuse.com/
[Arize Phoenix]: https://phoenix.arize.com/
[Ragas]: https://github.com/explodinggradients/ragas
[DeepEval]: https://github.com/confident-ai/deepeval
[Unstructured.io]: https://unstructured.io/
[PyMuPDF]: https://github.com/pymupdf/PyMuPDF
[OpenTelemetry]: https://opentelemetry.io/
[Pydantic]: https://docs.pydantic.dev/

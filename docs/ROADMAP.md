# Agentic Profile Matching Engine: Project Roadmap

This document outlines the strategic milestones for the **Agentic Profile Matching Engine**. 

> 💡 **Implementation Note**: Detailed technical task breakdowns, file modification maps, and code patterns are maintained in [`internal/implementation_plan.md`](../internal/implementation_plan.md) and individual [Architecture Decision Records (ADRs)](adr/README.md).

---

## 📍 Implementation Milestones

### Phase 1: Environment & Foundational Setup ✅ (`v0.1.0`)
- Core package environment initialization with LangGraph, Streamlit, Groq, and Google GenAI.
- Secure environment configuration via `.env` credentials.

### Phase 2: Ingestion & Vector Indexing Subsystem ✅ (`v0.2.0`)
- Multi-format synthetic candidate dataset generation (PDF, DOCX, TXT).
- Document ingestion pipeline with local SentenceTransformers embeddings and ChromaDB vector indexing.

### Phase 3: LangGraph Agent & Conversational State Machine ✅ (`v0.3.0`)
- 9-node `StateGraph` workflow with deterministic state transitions and `MemorySaver` checkpointing.
- End-to-end recruiter loop (parsing, search, deep screening, report generation, conversational adjustment).

### Phase 4: Assessment, Comparison & Screening Tools ✅ (`v0.4.0`)
- Schema-validated requirement extraction and multi-profile side-by-side comparison matrices.
- Fact-grounded candidate interview question generation tailored to identified skill gaps.

### Phase 5: Streamlit Recruiter Dashboard ✅ (`v0.5.0`)
- Interactive dual-pane recruiter UI (`app.py`) for live conversation and shortlist inspection.
- Conversational refinement workflow testing against recruitment scenarios.

### Phase 6: Model Context Protocol (MCP) Integration ✅ (`v0.6.0`)
- FastMCP filesystem (`resumes://`) and candidate search stdio protocol servers.
- Thread-safe dual-mode gateway client (`USE_MCP=True/False`) supporting local in-process and JSON-RPC execution ([ADR-001](adr/ADR-001-mcp-dual-mode-gateway-architecture.md)).

### Phase 7: Modular Graph Decoupling & Linting Hygiene ✅ (`v0.7.0`)
- Decomposed monolithic agent into a decoupled `agent/` package (`state.py`, `prompts.py`, `nodes.py`, `routers.py`).
- Automated pre-commit hooks and GitHub Actions CI quality gates with Ruff.

### Phase 8: Enterprise Abstractions & Background Workers ✅ (`v0.8.0` – `v0.8.6`)
- **Protocol Separation**: Dedicated `IngestionService` isolating business logic from MCP transport.
- **Storage Abstraction**: `BaseVectorStore` protocol with ChromaDB default and Qdrant stub ([ADR-003](adr/ADR-003-basevectorstore-structural-protocol.md), [ADR-005](adr/ADR-005-idempotent-upsert-ingestion.md)).
- **Retrieval Optimization**: Cached BM25Okapi sparse matrix with corpus fingerprinting ([ADR-004](adr/ADR-004-bm25-corpus-index-caching.md)).
- **State & Schema Safety**: TypedDict `AgentState` with Pydantic V2 LLM output contracts ([ADR-002](adr/ADR-002-using-typeddict-for-agentstate.md)).
- **Distributed Worker Queue**: Celery + Redis task queue with Docker Compose orchestration ([ADR-006](adr/ADR-006-celery-redis-task-queue.md)).

### Phase 9: Observability, Evaluation & Layout Ingestion ✅ (`v0.9.0` – `v0.9.4`)
- **Structured Observability**: Structured JSON logging and `@trace_node` latency instrumentation with opt-in Langfuse and OpenTelemetry backends ([ADR-007](adr/ADR-007-structured-json-logging-and-tracing.md)).
- **RAG Evaluation Suite**: Automated Recall@K, MRR, and LLM faithfulness benchmarks against ground-truth datasets.
- **Advanced Document Ingestion**: PyMuPDF (`fitz`) vertical column-sorted extraction with opt-in Unstructured.io support.

### Phase 10: Semantic Routing, Multi-Provider Scale & Architecture Baseline ✅ (`v1.0.0` – `v1.1.0`)
- **Multi-Factor Hybrid Scoring**: Min-max normalized 60/40 dense-sparse candidate ranking with grounded guardrails ([ADR-008](adr/ADR-008-multi-factor-hybrid-scoring-and-hierarchy.md)).
- **Tiered Intent Routing**: <2ms local semantic vector routing with LLM structured classification fallback ([ADR-009](adr/ADR-009-tiered-semantic-embedding-intent-routing.md)).
- **Multi-Provider & Indic LLMs**: Unified provider factory with Sarvam AI (`sarvam-105b`), Groq, Gemini, and OpenAI ([ADR-010](adr/ADR-010-multi-provider-sarvam-indic-llm.md)).
- **Architectural Documentation**: Published formal Architecture Decision Records catalog ([ADRs 001–015](adr/README.md)).

---

## 🔮 Planned Enterprise Releases

### Phase 11: Production Hardening, Security & State Invariance ⏳ (`v1.2.0`)
- **11.1 — Stateless Credential Isolation**: Purge API credentials from `AgentState`; inject via `RunnableConfig` (CWE-312 compliance) ([ADR-011](adr/ADR-011-stateless-credential-isolation.md)).
- **11.2 — Functional State Immutability**: Enforce copy-on-write candidate dict updates for safe LangGraph retries ([ADR-012](adr/ADR-012-functional-state-immutability.md)).
- **11.3 — Dynamic Skills Taxonomy**: Declarative YAML taxonomy with canonical alias stemming (`K8s` $\to$ `Kubernetes`) ([ADR-013](adr/ADR-013-dynamic-skills-taxonomy-alias-normalization.md)).
- **11.4 — Floating-Point Precision**: Migrate `CandidateMatch.score` typing from `int` to `float` across state schemas.

### Phase 12: High-Throughput Screening & Resource Lifecycle ⏳ (`v1.3.0`)
- **12.1 — Async Candidate Deep Screening**: Concurrency-controlled worker pool (`ThreadPoolExecutor` + `Semaphore(2)`) reducing screening wall-clock time from ~75s to ~15s ([ADR-014](adr/ADR-014-concurrency-controlled-async-screening.md)).
- **12.2 — Singleton Model Caching**: Cache `JobMatcher` and embedder instances via `@st.cache_resource` and container DI.
- **12.3 — Open/Closed Provider Registry**: Extensible `PROVIDER_REGISTRY` mapping with runtime `register_provider()` API ([ADR-015](adr/ADR-015-open-closed-llm-provider-registry.md)).
- **12.4 — Modular UI Decomposition**: Refactor `app.py` into testable `ui/components/` and `ui/session_manager.py`.

### Phase 13: Enterprise CI/CD, Supply Chain Security & Quality Gates ⏳ (`v1.4.0`)
- **13.1 — Static Type Analysis**: Strict `mypy` type checking in CI.
- **13.2 — Test Coverage Enforcement**: Automated `pytest-cov` gate enforcing $\ge 75\%$ code coverage.
- **13.3 — Security Scanning**: Automated secret detection (`gitleaks`) and CVE vulnerability auditing (`pip-audit`).
- **13.4 — Multi-Version Matrix**: Python 3.11 and 3.12 CI test matrix validation.

### Phase 14: Engine Resilience, Cache Correctness & Retrieval Precision ⏳ (`v1.5.0`)
- **14.1 — Deterministic BM25 Fingerprinting**: Global corpus hash validation for robust cache invalidation.
- **14.2 — Guardrail Boundary Normalization**: Refined threshold boundaries in recommendation nodes.
- **14.3 — Async Subprocess Lifecycle**: Clean stdio stream and event loop teardown.
- **14.4 — Developer Experience Tooling**: Comprehensive `Makefile` and CLI scenario evaluation tooling.

---

## 🚀 Future Backlog (Post-Phase 14)

- **Indirect Prompt Injection Sanitizer**: Pre-screening sanitization scanning candidate resume text for adversarial prompt overrides.
- **PII Anonymization & Redaction Layer**: Reversible entity tokenization (`[CANDIDATE_A]`) for privacy compliance.
- **Semantic Embedding Cache**: Redis-backed semantic vector query cache ($< 10\text{ms}$).
- **Multi-Agent Consensus Loop**: Independent Technical Architect and HR Sourcing screener debate before recommendation.
- **Bias & Fairness Auditing**: Automated inclusivity auditing for job descriptions and screening assessments.

---

<!-- References -->
[langgraph]: https://langchain-ai.github.io/langgraph/
[streamlit]: https://streamlit.io/
[ChromaDB]: https://www.trychroma.com/
[FastMCP]: https://github.com/modelcontextprotocol/python-sdk
[Ruff]: https://github.com/astral-sh/ruff
[Qdrant]: https://qdrant.tech/
[Celery]: https://docs.celeryq.dev/
[Redis]: https://redis.io/
[Langfuse]: https://langfuse.com/
[OpenTelemetry]: https://opentelemetry.io/
[PyMuPDF]: https://github.com/pymupdf/PyMuPDF
[Sarvam AI]: https://www.sarvam.ai/

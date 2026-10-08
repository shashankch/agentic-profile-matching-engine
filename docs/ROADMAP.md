# 🗺️ Yojaka AI: Product Roadmap

This document outlines high-level implementation milestones and strategic release targets for **Yojaka AI (Agentic Profile Matching Engine)**.

> 📚 **Detailed Specifications**: For system architecture diagrams, mathematical formulations, and formal design decisions, see [**System Architecture**](architecture.md), [**Architecture Decision Records (ADRs)**](adr/index.md), and [**Changelog**](CHANGELOG.md).

---

## 📍 Implementation Milestones (Phases 1–17)

### Foundational Pipeline & Core State Machine (Phases 1–7 • v0.1.0) ✅
- **Agentic Core**: 9-node LangGraph `StateGraph` workflow with deterministic state transitions and `MemorySaver` checkpointing.
- **Tooling & Gateway**: Dual-mode tool execution gateway supporting local in-process and FastMCP JSON-RPC `stdio` servers ([ADR-001](adr/ADR-001-mcp-dual-mode-gateway-architecture.md)).
- **Candidate Vetting Tools**: Schema-validated requirement extraction, side-by-side comparison matrix, and tailored interview question generation.

### Enterprise Abstractions & Background Workers (Phase 8 • v0.8.0) ✅
- **Service Decoupling**: Dedicated `IngestionService` isolating core business logic from transport protocols.
- **Pluggable Storage**: `BaseVectorStore` structural protocol with ChromaDB default and Qdrant stub ([ADR-003](adr/ADR-003-basevectorstore-structural-protocol.md)).
- **Async Task Queue**: Distributed Celery workers with Redis 7 message broker and Docker Compose orchestration ([ADR-006](adr/ADR-006-celery-redis-task-queue.md)).

### Observability, Hybrid Retrieval & Multi-Provider Scale (Phases 9–10 • v1.0.0) ✅
- **Enterprise Observability**: Structured JSON logging and latency tracing with Langfuse and OpenTelemetry backends ([ADR-007](adr/ADR-007-structured-json-logging-and-tracing.md)).
- **Hybrid Scoring**: Min-max normalized dense vector cosine + BM25Okapi lexical retrieval with cached sparse index ([ADR-004](adr/ADR-004-bm25-corpus-index-caching.md), [ADR-008](adr/ADR-008-multi-factor-hybrid-scoring-and-hierarchy.md)).
- **Multi-Provider LLM Gateway**: Unified provider abstraction supporting Groq (Llama 3.3 70B), Google Gemini, Sarvam AI (105B Indic), and OpenAI ([ADR-010](adr/ADR-010-multi-provider-sarvam-indic-llm.md)).

### Security, State Invariance & Async Screening (Phases 11–12 • v1.1.0) ✅
- **Stateless Credential Isolation**: Credentials passed via `RunnableConfig`, eliminating secret exposure in checkpoints (CWE-312) ([ADR-011](adr/ADR-011-stateless-credential-isolation.md)).
- **Functional State Immutability**: Copy-on-write candidate dictionary updates ensuring safe LangGraph retries ([ADR-012](adr/ADR-012-functional-state-immutability.md)).
- **Concurrent Screening**: Parallel candidate screening with `ThreadPoolExecutor` and rate-limit semaphores ([ADR-014](adr/ADR-014-concurrency-controlled-async-screening.md)).

### Quality Gates & Engine Resilience (Phases 13–14 • v1.1.5) ✅
- **Automated Quality Gates**: Strict `mypy` static typing, `pytest-cov` ≥ 75% coverage gate, and `gitleaks`/`pip-audit` security scanning.
- **Deterministic Cache Correctness**: Global BM25 corpus fingerprinting and async subprocess lifecycle management.

### In-Memory Ingestion, Modern Routing & UI (Phase 15 • v1.2.0) ✅
- **Zero-Disk In-Memory Ingestion**: Stream-based file parsing (`io.BytesIO`) without disk persistence, layered with ephemeral session stores ([ADR-016](adr/ADR-016-zero-disk-in-memory-resume-ingestion.md)).
- **Dynamic Semantic Skill Expansion**: LLM requirement expansion with word-boundary matching, replacing brittle static taxonomies ([ADR-013](adr/ADR-013-dynamic-skills-taxonomy-alias-normalization.md)).
- **LLM Intent Routing & Caching**: Structured zero-shot intent router with dynamic anchor synthesis and sub-millisecond query caching ([ADR-009](adr/ADR-009-tiered-semantic-embedding-intent-routing.md)).
- **UI Ergonomics**: Compact zero-scroll recruiter sidebar with instant repository path auto-discovery.

### Modular UI Architecture & Headless API Sidecar (Phase 16 • v1.3.0) ✅
- **Modular Presentation Layer**: Monolithic `app.py` decomposed into an executive `<75-LOC` conductor and modular `ui/` components.
- **Streamlit-Native Streaming**: Real-time typewriter output via `st.write_stream()` and dynamic intent-aware execution tracking.
- **Headless FastAPI Gateway**: Additive ASGI sidecar exposing REST and Server-Sent Events (SSE) streaming endpoints.

### Layout Parsing, Contextual Retrieval & Two-Stage Reranking (Phase 17 • v1.4.0) ✅
- **Layout-Aware Section Parsing**: PyMuPDF visual bounding block extraction preserving work experience integrity without boundary truncation.
- **Anthropic Contextual Retrieval Prepending**: Document-level metadata banners prepended to chunks before dense and BM25 embedding.
- **Two-Stage Hybrid Reranking**: Fine-grained Cross-Encoder joint reranking (`ms-marco-MiniLM-L-6-v2`) with Sigmoid calibration and Reciprocal Rank Fusion.
- **Serverless Cloud Resilience**: Zero-disk pure NumPy `InMemoryVectorStore` fallback ensuring 100% crash-free ephemeral execution.

---

## 🚀 Future Milestones (Phases 18–20)

### Phase 18: Agent Topology, Calibrated Routing & Native Commands
- **Calibrated Margin Routing**: Embedding margin scoring (Δ ≥ 0.12) to bypass LLM latency on high-confidence queries while escalating ambiguous queries to structured classifiers.
- **LangGraph Native Commands**: Refactor node transitions to native LangGraph `Command(goto=...)` primitives for dynamic graph traversal.
- **Parallel Dual-Rubric Evaluation**: Parallel evaluation rubrics (Technical Architecture Competence vs HR Sourcing Fit), avoiding conversational latency bloat while delivering balanced committee scorecards.

### Phase 19: Enterprise Multi-Tenancy & Zero-Trust PII Redaction
- **Zero-Trust PII Tokenization Vault**: Pre-screening redaction replacing candidate personal identifiers with cryptographic tokens (`[CANDIDATE_A]`), enforcing objective blind hiring meritocracy.
- **Tenant-Isolated Namespaces**: Hardware-partitioned collection namespaces across vector stores and session checkpoints.
- **Adversarial Prompt Injection Sanitizer**: Ingestion heuristics intercepting indirect prompt injection payloads hidden inside uploaded resume files (OWASP LLM01).
- **ATS Batch Staging Adapter**: Pre-signed S3/Blob storage adapter with automated lifecycle expiration for asynchronous multi-thousand resume batch uploads from enterprise ATS platforms (Workday, Greenhouse).

### Phase 20: Unit Economics, Semantic Caching & Continuous Evals
- **Pool-Aware Semantic Cache**: Sub-10ms response cache for semantically equivalent recruiter queries, guarded by pool-version hashes to prevent stale candidate rankings.
- **Bias, Fairness & Inclusivity Auditing**: Automated inclusivity scanning for job descriptions and demographic score parity auditing for global regulatory compliance.
- **Granular Cost & Token Budgeting**: Real-time per-query and per-tenant cost accounting and token quota enforcement.
- **Automated Continuous Evaluation Gate**: Automated RAG Triad benchmarks (Context Precision, Recall@K, Faithfulness) blocking pull requests on score regression.

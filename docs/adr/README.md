# Architecture Decision Records (ADRs)

This directory contains the individual Architecture Decision Records (ADRs) for the **Agentic Profile Matching Engine**.

---

## 📑 ADR Catalog & Executive Matrix

| ADR | Title | Status | Release | Category | Standalone File |
|:---|:---|:---|:---|:---|:---|
| **ADR-001** | Model Context Protocol (MCP) Dual-Mode Gateway Architecture | Accepted | `v0.6.0` | Protocol & Tooling | [ADR-001](ADR-001-mcp-dual-mode-gateway-architecture.md) |
| **ADR-002** | Using `TypedDict` for `AgentState` Over Pydantic `BaseModel` | Accepted | `v0.7.0` | State & Serialization | [ADR-002](ADR-002-using-typeddict-for-agentstate.md) |
| **ADR-003** | `BaseVectorStore` Structural Protocol Over `abc.ABC` | Accepted | `v0.8.2` | Storage Abstraction | [ADR-003](ADR-003-basevectorstore-structural-protocol.md) |
| **ADR-004** | BM25 Okapi Corpus Index Caching with Invalidation | Accepted | `v0.8.3` | Performance & Retrieval | [ADR-004](ADR-004-bm25-corpus-index-caching.md) |
| **ADR-005** | Idempotent `upsert()` Ingestion with Section-Scoped Chunk Keys | Accepted | `v0.8.2` | Data Integrity | [ADR-005](ADR-005-idempotent-upsert-ingestion.md) |
| **ADR-006** | Celery + Redis Task Queue for Asynchronous Heavy Operations | Accepted | `v0.8.6` | Distributed Workers | [ADR-006](ADR-006-celery-redis-task-queue.md) |
| **ADR-007** | Structured JSON Logging & Pluggable Tracing Pipeline | Accepted | `v0.9.1` | Observability & APM | [ADR-007](ADR-007-structured-json-logging-and-tracing.md) |
| **ADR-008** | Multi-Factor Hybrid Candidate Scoring & Grounded LLM Recommendation Hierarchy | Accepted | `v1.0.0` | AI Evaluation & Guardrails | [ADR-008](ADR-008-multi-factor-hybrid-scoring-and-hierarchy.md) |
| **ADR-009** | Tiered Semantic Embedding & LLM Structured Intent Routing | Accepted | `v1.1.0` | Semantic Routing | [ADR-009](ADR-009-tiered-semantic-embedding-intent-routing.md) |
| **ADR-010** | Multi-Provider LLM Abstraction with Sarvam AI Indic Model Integration | Accepted | `v1.1.0` | Multi-Provider & Indic Scale | [ADR-010](ADR-010-multi-provider-sarvam-indic-llm.md) |
| **ADR-011** | Stateless Credential Isolation & Checkpoint Security | Accepted | `v1.2.0` | Security (CWE-312) | [ADR-011](ADR-011-stateless-credential-isolation.md) |
| **ADR-012** | Functional State Immutability & LangGraph Node Invariance | Accepted | `v1.2.0` | State & Reliability | [ADR-012](ADR-012-functional-state-immutability.md) |
| **ADR-013** | Configuration-Driven Skills Taxonomy & Semantic Alias Normalization | Accepted | `v1.2.0` | Domain Extensibility | [ADR-013](ADR-013-dynamic-skills-taxonomy-alias-normalization.md) |
| **ADR-014** | Concurrency-Controlled Asynchronous Candidate Screening | Accepted | `v1.3.0` | Throughput & Latency | [ADR-014](ADR-014-concurrency-controlled-async-screening.md) |
| **ADR-015** | Open/Closed LLM Provider Registry Pattern | Accepted | `v1.3.0` | Architecture & OCP | [ADR-015](ADR-015-open-closed-llm-provider-registry.md) |

---

## Executive Summaries

### [ADR-001: Model Context Protocol (MCP) Dual-Mode Gateway Architecture](ADR-001-mcp-dual-mode-gateway-architecture.md)
- **Context**: Enabling both local in-process testing and standardized FastMCP JSON-RPC 2.0 tool execution.
- **Decision**: Dual-mode client manager (`fs_client.py`, `mcp_client.py`) toggled dynamically via `USE_MCP=True/False`.
- **Consequence**: Zero-overhead local testing + full microservice protocol compliance.

### [ADR-002: Using `TypedDict` for `AgentState` Over Pydantic `BaseModel`](ADR-002-using-typeddict-for-agentstate.md)
- **Context**: Minimizing LangGraph state checkpointer serialization overhead while guaranteeing type safety.
- **Decision**: `TypedDict` for graph state transitions; strict Pydantic V2 models exclusively at LLM tool boundaries.
- **Consequence**: Eliminates checkpointer overhead and avoids custom object serialization issues.

### [ADR-003: `BaseVectorStore` Structural Protocol Over `abc.ABC`](ADR-003-basevectorstore-structural-protocol.md)
- **Context**: Enabling swappable vector store backends without framework class coupling.
- **Decision**: `typing.Protocol` structural typing for `ChromaVectorStore` and `QdrantVectorStore`.
- **Consequence**: Clean dependency injection into `IngestionService` and `JobMatcher` with zero vendor lock-in.

### [ADR-004: BM25 Okapi Corpus Index Caching with Invalidation](ADR-004-bm25-corpus-index-caching.md)
- **Context**: Rebuilding the BM25 index on every search query adds $O(N)$ latency (~200ms).
- **Decision**: Cache `BM25Okapi` sparse matrix with MD5 fingerprint checks across the indexed corpus.
- **Consequence**: Reduces hybrid search query latency from ~200ms to **~0.1ms**.

### [ADR-005: Idempotent `upsert()` Ingestion with Section-Scoped Chunk Keys](ADR-005-idempotent-upsert-ingestion.md)
- **Context**: Duplicate vector entries skew term frequencies and corrupt similarity scores on re-ingestion.
- **Decision**: Deterministic chunk IDs formatted as `{filename}_chunk_{section_index}`.
- **Consequence**: 100% idempotent document ingestion with zero collection bloat.

### [ADR-006: Celery + Redis Task Queue for Asynchronous Heavy Operations](ADR-006-celery-redis-task-queue.md)
- **Context**: Multi-document PDF parsing and batch LLM screening block web UI event loops.
- **Decision**: Celery distributed workers with Redis broker and Docker Compose orchestration.
- **Consequence**: Offloads high-latency background operations from the web application UI thread.

### [ADR-007: Structured JSON Logging & Pluggable Tracing Pipeline](ADR-007-structured-json-logging-and-tracing.md)
- **Context**: Unformatted console output lacks structured timestamps and APM latency metrics.
- **Decision**: Single-line JSON log formatter + `@trace_node` decorator supporting Langfuse and OpenTelemetry.
- **Consequence**: Production-grade APM log compatibility with millisecond execution tracking.

### [ADR-008: Multi-Factor Hybrid Candidate Scoring & Grounded LLM Recommendation Hierarchy](ADR-008-multi-factor-hybrid-scoring-and-hierarchy.md)
- **Context**: Preventing score compression, magic-number overrides, and LLM qualification hallucinations.
- **Decision**: Min-max normalized vector + BM25 scoring with grounded status hierarchy and safety guardrails.
- **Consequence**: Deterministic scoring and elimination of hallucinated qualifications.

### [ADR-009: Tiered Semantic Embedding & LLM Structured Intent Routing](ADR-009-tiered-semantic-embedding-intent-routing.md)
- **Context**: Brittle substring keyword routing vs expensive LLM classification on every turn.
- **Decision**: Tier 1 SentenceTransformer vector similarity (< 2ms) + Tier 2 structured LLM intent fallback.
- **Consequence**: Zero token cost for 85%+ requests while eliminating brittle keyword lists.

### [ADR-010: Multi-Provider LLM Abstraction with Sarvam AI Indic Model Integration](ADR-010-multi-provider-sarvam-indic-llm.md)
- **Context**: Multilingual candidate screening (Indic languages) and zero vendor lock-in.
- **Decision**: Unified provider factory with Sarvam AI (`sarvam-105b`), Groq, Gemini, and OpenAI.
- **Consequence**: Native Indic language resume screening with seamless provider switching.

### [ADR-011: Stateless Credential Isolation & Checkpoint Security](ADR-011-stateless-credential-isolation.md)
- **Context**: Storing API keys in `AgentState` leaks credentials into checkpoint dumps and traces (CWE-312).
- **Decision**: Purge credentials from state; inject pre-instantiated LLM via `RunnableConfig` (`configurable["llm"]`).
- **Consequence**: Eliminates credential leakage risk across state checkpoints, logs, and telemetry.

### [ADR-012: Functional State Immutability & LangGraph Node Invariance](ADR-012-functional-state-immutability.md)
- **Context**: In-place dictionary mutations break LangGraph checkpoint idempotency and retry safety.
- **Decision**: Pure functional node transitions with copy-on-write candidate dict updates (`{**c, ...}`).
- **Consequence**: Deterministic checkpointing and race-free node re-execution.

### [ADR-013: Configuration-Driven Skills Taxonomy & Semantic Alias Normalization](ADR-013-dynamic-skills-taxonomy-alias-normalization.md)
- **Context**: Hardcoded skill lists cause silent false negatives for unlisted technologies.
- **Decision**: Declarative YAML taxonomy (`config/skills_taxonomy.yaml`) with alias stemming (`K8s` $\to$ `Kubernetes`).
- **Consequence**: Zero-code domain vocabulary customizability with alias normalization.

### [ADR-014: Concurrency-Controlled Asynchronous Candidate Screening](ADR-014-concurrency-controlled-async-screening.md)
- **Context**: Sequential LLM screening causes 60–75s UI freezes; unbounded parallel calls trigger 429 rate limits.
- **Decision**: Bounded `ThreadPoolExecutor` worker pool paired with `Semaphore(max_concurrent=2)`.
- **Consequence**: 3.5x throughput improvement (~75s $\to$ ~15–20s) with rate-limit protection.

### [ADR-015: Open/Closed LLM Provider Registry Pattern](ADR-015-open-closed-llm-provider-registry.md)
- **Context**: Procedural `if/elif` chains in model factories violate the Open/Closed Principle.
- **Decision**: Extensible `PROVIDER_REGISTRY: dict[str, Callable]` with `register_provider()` API.
- **Consequence**: Open/Closed compliance; zero-code modification when adding new model providers.

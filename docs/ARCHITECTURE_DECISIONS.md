# Architecture Decision Records (ADRs)

This document records the key architectural decisions, design choices, trade-offs, and consequences for the **Agentic Profile Matching Engine**.

---

## ADR-001: Model Context Protocol (MCP) Dual-Mode Gateway Architecture

### Status
Accepted (Implemented in `v0.6.0`)

### Context
The agent workflow requires interaction with local filesystem utilities (file parsing, directory listing) and external/internal search services. In production, tools may be hosted as decoupled, isolated microservices operating over standard protocol specifications. However, forcing all local CLI and unit testing workflows through subprocess JSON-RPC servers introduces unnecessary runtime setup overhead.

### Decision
Implement a **Dual-Mode Gateway Architecture** (`fs_client.py` and `mcp_client.py`) toggled dynamically via the `config.USE_MCP` boolean flag (`USE_MCP=True/False` in `.env` / `config.py`):
1. **Local Direct Mode (`USE_MCP=False`)**: Directly executes local Python modules (`fs_tools.py`, `job_matcher.py`) in-process for lightweight local development and rapid testing.
2. **MCP Protocol Mode (`USE_MCP=True`)**: Connects via `mcp.ClientSession` over `stdio` transport to standalone FastMCP protocol servers (`filesystem_mcp_server.py` and `search_mcp_server.py`), executing tools over standard JSON-RPC 2.0.

### Consequences
- **Positive**: Allows zero-overhead local development while guaranteeing full compliance with the Model Context Protocol specification for remote or containerized tool servers.
- **Negative**: Requires maintaining dual invocation pathways in `fs_client.py`.

---

## ADR-002: Using `TypedDict` for `AgentState` Over Pydantic `BaseModel`

### Status
Accepted (Implemented in `v0.7.0`, Refined in `v0.8.4`)

### Context
LangGraph state machines require state objects passed across graph nodes and checkpointed into state memory (`MemorySaver`). Python developers often choose Pydantic `BaseModel` for automatic validation. However, strict Pydantic serialization adds validation overhead during graph state transitions and can raise runtime state serialization exceptions when custom non-serializable objects (such as LLM client instances) are temporarily stored.

### Decision
Use a standard Python `TypedDict` for the core `AgentState` schema in `agent/state.py`. Enforce payload contracts using Pydantic V2 JSON models exclusively at LLM tool boundary outputs (`JobRequirementsOutput`, `DeepScreenOutput`).

### Consequences
- **Positive**: Eliminates state checkpointer serialization overhead; matches official LangGraph idiomatic state management patterns; provides clean, native dictionary access across all graph nodes.
- **Negative**: Type checking relies on static type checkers (`mypy`, `pyright`) rather than runtime validation during state assignment.

---

## ADR-003: `BaseVectorStore` Structural Protocol Over Abstract Base Class (`abc.ABC`)

### Status
Accepted (Implemented in `v0.8.2`)

### Context
To prevent vendor lock-in to ChromaDB and allow switching to other vector engines (e.g. Qdrant), vector storage implementations must expose a unified interface (`add_documents`, `upsert_documents`, `similarity_search`, `delete`). Standard Python inheritance (`abc.ABC`) enforces rigid subclass relationships and requires explicit inheritance declarations.

### Decision
Define `BaseVectorStore` using Python's `typing.Protocol` (structural subtyping / duck typing). `ChromaVectorStore` and `QdrantVectorStore` implement `BaseVectorStore` implicitly without inheriting from a common base class.

### Consequences
- **Positive**: Enforces compile-time and static interface compliance without coupling store classes to a framework base class; enables flexible dependency injection into `IngestionService` and `JobMatcher`.
- **Negative**: Missing interface methods are caught during static analysis rather than at class definition time.

---

## ADR-004: BM25 Okapi Corpus Index Caching with Invalidation

### Status
Accepted (Implemented in `v0.8.3`)

### Context
Hybrid candidate search combines dense semantic vector scores from ChromaDB with sparse BM25 keyword matching scores on tokenized candidate text. Rebuilding the BM25 index on every search query parses the entire document corpus ($O(N)$), adding ~200ms latency per query at 1,000 document chunks.

### Decision
Cache the initialized `BM25Okapi` instance on `JobMatcher` construction and compute an MD5 hash of the indexed document contents. Re-tokenize and rebuild the index only when corpus document count or MD5 hash fingerprint changes.

### Consequences
- **Positive**: Reduces BM25 hybrid query latency from ~200ms down to ~0.1ms for cached queries.
- **Negative**: Retains tokenized corpus arrays in process memory.

---

## ADR-005: Idempotent `upsert()` Ingestion with Section-Scoped Chunk Keys

### Status
Accepted (Implemented in `v0.8.2`)

### Context
Re-running resume directory ingestion using standard vector store `add()` calls generates new random UUIDs for identical document chunks. This duplicates entries in ChromaDB, skewing term frequency calculations and corrupting hybrid vector similarity rankings.

### Decision
Implement `upsert_documents()` in `ChromaVectorStore` using deterministic section-scoped chunk identifiers formatted as `{document_filename}_chunk_{section_index}`. Existing document chunks with matching IDs are updated in-place instead of duplicated.

### Consequences
- **Positive**: Guarantees 100% idempotent document re-ingestion; keeps vector database collection size stable across repeated runs.
- **Negative**: Requires callers to supply deterministic chunk IDs or compute hash keys before storage.

---

## ADR-006: Celery + Redis Task Queue for Asynchronous Heavy Operations

### Status
Accepted (Implemented in `v0.8.6`)

### Context
Heavy operations such as multi-document PDF parsing, batch vector embedding ingestion, and multi-candidate LLM audits can block web servers (e.g. Streamlit or HTTP API endpoints) if run synchronously on the main event loop thread.

### Decision
Incorporate **Celery** with **Redis** as a distributed task queue broker (`tasks.py` and `celery_app.py`). Expose non-blocking background task endpoints (`async_ingest_directory`, `async_deep_screen_candidate`) and provide Docker Compose containerization (`docker-compose.yml`) bundling Redis, Streamlit, and Celery worker services.

### Consequences
- **Positive**: Offloads high-latency background operations from the web application UI thread; provides horizontal scalability for worker processes.
- **Negative**: Adds Redis as an operational infrastructure dependency for asynchronous deployment.

---

## ADR-007: Structured JSON Logging & Pluggable Tracing Pipeline

### Status
Accepted (Implemented in `v0.9.1` and `v0.9.2`)

### Context
Standard unformatted `print()` statements lack execution context, timestamps, log levels, and duration metrics required for production debugging and APM monitoring in cloud environments.

### Decision
Implement structured JSON logging (`JsonFormatter`, `get_logger`) outputting log records with standard keys (`timestamp`, `level`, `logger`, `message`). Wrap all 9 LangGraph workflow nodes with a `@trace_node` decorator that tracks node execution start (`node_start`), completion (`node_end`), errors (`node_error`), and millisecond latency (`elapsed_ms`). Extend `@trace_node` with opt-in tracing for **Langfuse** (`OBSERVABILITY_BACKEND=langfuse`) and **OpenTelemetry** (`OBSERVABILITY_BACKEND=opentelemetry`).

### Consequences
- **Positive**: Enables zero-dependency structured JSON log parsing for Datadog, AWS CloudWatch, and ELK stacks; provides opt-in integration with modern LLM tracing platforms.
- **Negative**: Requires setting environment variables to enable external tracing backends.

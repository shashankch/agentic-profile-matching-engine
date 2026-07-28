# Agentic Profile Matching Engine: Engineering Conventions

This document defines the architectural guidelines, code quality standards, and design conventions for the **Agentic Profile Matching Engine**. These standards reflect production-grade software development expectations.

---

## 🏛️ 1. Architectural Principles & Layering

1. **Separation of Concerns (SoC)**
   - **Domain Logic**: RAG processing (`resume_rag.py`), hybrid matching algorithm (`job_matcher.py`), and graph state machine (`agent/`) must remain clean and decoupled from presentation layers (Streamlit UI) and RPC transport protocols (MCP servers).
   - **Service Layer**: Business operations spanning multiple domain components should be encapsulated in a dedicated service layer (e.g., `IngestionService`).

2. **Protocol-Driven Design & Abstractions**
   - High-level orchestrators must depend on interfaces (`typing.Protocol` or `abc.ABC`), not concrete implementations (e.g., `BaseVectorStore` instead of direct `ChromaDB` calls).
   - Concrete implementations (e.g., `ChromaVectorStore`, `QdrantVectorStore`) must be swappable via dependency injection without changing consumer code.

3. **Dependency Injection**
   - Classes must receive external dependencies (vector store, clients, LLM providers) via constructor arguments rather than instantiating them internally.

---

## 🛡️ 2. State Management & Type Safety

1. **Explicit Type Hinting**
   - All function parameters, return values, class attributes, and module constants must carry explicit Python 3.10+ type annotations.
   - Avoid `Any` wherever possible; use `Union`, `Optional`, `TypeVar`, or generic collections (`list[str]`, `dict[str, float]`).

2. **Structured Models with Pydantic V2**
   - Use Pydantic V2 `BaseModel` for config validation, external API payloads, and LLM output parsing.
   - LLM responses must be parsed and validated using `model_validate_json()` with fallback handling.

3. **Immutable Agent State**
   - Graph state schemas (e.g., `AgentState` in `agent/state.py`) must use `TypedDict` or Pydantic models.
   - State updates within graph nodes should return partial state dictionaries explicitly rather than mutating global/external references.

---

## ⚠️ 3. Resilience & Error Handling Boundaries

1. **No Silent Exception Swallowing**
   - Catch specific exceptions (`FileNotFoundError`, `JSONDecodeError`, `KeyError`, `httpx.HTTPError`).
   - Never use empty `except:` or `except Exception: pass` without logging or populating state warning flags.

2. **Graceful State Degradation**
   - If a non-critical tool node (e.g., DuckDuckGo web search or external candidate notes fetching) fails, log a warning, set a warning flag in `AgentState`, and allow the core pipeline to complete.

3. **Deterministic LLM Fallbacks**
   - Any LLM invocation expecting JSON must include retry or default template fallback logic in case of schema validation failures or API rate limits.

---

## ⚡ 4. Concurrency & Protocol Integration

1. **Async / Sync Bridge Safety**
   - MCP servers operate over async stdio streams. When called from synchronous environments (like Streamlit UI or legacy tools), bridges must use thread-safe event loop execution patterns (e.g., `mcp_client.py` background loop runner).

2. **Resource Lifecycle Teardown**
   - Connections, subprocesses, vector store handles, and open file handles must be managed via context managers (`with` / `async with`) or explicit `.close()` / `.stop()` teardown hooks.

---

## 🧪 5. Testing & Code Quality Standards

1. **Test Coverage Expectations**
   - **Core Logic**: Math algorithms (e.g. 60/40 hybrid BM25 + vector ranking), graph router conditionals, and candidate assessment tools must have unit tests.
   - **Protocol Tests**: Mock external LLMs and MCP transport in unit tests (`pytest`).

2. **Linter & Formatter Integrity**
   - All code must pass `ruff check .` and `ruff format .` with zero errors or warnings before merging.
   - Maintain maximum line length limit of **100 characters**.

---

## 📝 6. Commit & Versioning Standards

1. **Conventional Commits**
   - Format: `<type>(<scope>): <short summary>`
   - Allowed Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `ci`.
   - Example: `feat(ingestion): Phase 8.1 - IngestionService Layer (v0.4.0)`

2. **Semantic Versioning**
   - Follow `MAJOR.MINOR.PATCH` versioning tracked in `pyproject.toml` and documented chronologically in `CHANGELOG.md`.

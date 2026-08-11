# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.3] - 2026-08-11
### Added
- Created ground-truth RAG evaluation benchmark dataset (`data/eval_scenarios.json`) defining test scenarios, required skills, experience bounds, and expected candidate ranking recall metrics.
- Introduced RAG evaluation test suite package (`tests/eval/__init__.py`) with custom pytest `@pytest.mark.eval` marker.
- Added hybrid search Retrieval Recall@K (e.g. Recall@10) and Mean Reciprocal Rank (MRR) metrics evaluation in `tests/eval/test_retrieval_recall.py`.
- Added candidate screening report faithfulness and groundedness evaluation suite in `tests/eval/test_response_faithfulness.py`.

## [0.9.2] - 2026-08-10
### Added
- Extended `@trace_node` in `observability.py` with opt-in tracing integration for **Langfuse** (`OBSERVABILITY_BACKEND=langfuse`) and **OpenTelemetry** (`OBSERVABILITY_BACKEND=opentelemetry`).
- Added tracing configuration parameters (`OBSERVABILITY_BACKEND`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_HOST`, `OTEL_EXPORTER_OTLP_ENDPOINT`) to `config.py`.
- Created `.env.example` template documenting all local environment variables and optional observability backend settings.
- Added unit tests in `tests/test_observability.py` verifying graceful fallback when optional tracing SDKs are uninstalled or unconfigured.

## [0.9.1] - 2026-08-09
### Added
- Created structured observability module (`src/agentic_profile_matching/observability.py`) containing `JsonFormatter`, `get_logger()`, and `@trace_node()` decorator for millisecond latency instrumentation and JSON event logging (`node_start`, `node_end`, `node_error`).
- Added `@trace_node` latency instrumentation across all 9 LangGraph agent workflow nodes (`parse_input`, `extract_requirements`, `search_resumes`, `rank_candidates`, `deep_screen`, `recommendation`, `generate_report`, `adjust_requirements`, `conversational_query`).
- Replaced 20+ unformatted `print()` statements across `nodes.py`, `job_matcher.py`, and `resume_rag.py` with structured logger instances (`logger.info`, `logger.warning`, `logger.error`).
- Added unit test suite in `tests/test_observability.py` covering JSON formatting, logger creation, node trace execution, and exception handling.

## [0.8.6] - 2026-08-06
### Added
- Introduced Celery background worker module (`tasks.py`) and application instance (`celery_app.py`) for non-blocking asynchronous document ingestion (`async_ingest_directory`) and candidate deep screening (`async_deep_screen_candidate`).
- Created Docker containerization suite including `Dockerfile` (multi-stage Python 3.12 build), `docker-compose.yml` (orchestrating `redis:7-alpine`, Streamlit web app, and Celery worker services), and `.dockerignore`.
- Added Celery task unit tests in `tests/test_tasks.py`.

## [0.8.5] - 2026-08-05
### Added
- Created comprehensive agent workflow graph and node test suite (`tests/test_nodes.py`, `tests/test_agent_graph.py`) testing all 9 graph nodes and end-to-end multi-round execution loops with mocked LLM models and vector stores.

## [0.8.4] - 2026-08-04
### Added
- Introduced domain exception hierarchy (`EngineError`, `IngestionError`, `RetrievalError`, `LLMParseError`, `VectorStoreError`, `CollectionNotFoundError`) in `exceptions.py` and `stores/exceptions.py`.
- Created Pydantic V2 output models (`JobRequirementsOutput`, `DeepScreenOutput`) and `parse_json_output()` parsing helper in `tools.py`.

### Changed
- Refactored LLM output parsing in `tools.py` and `agent/nodes.py` (`extract_requirements`, `generate_interview_questions`, `deep_screen_node`, `adjust_requirements_node`) to use Pydantic V2 `model_validate_json()` with lenient fallback.

## [0.8.3] - 2026-08-03
### Security
- Hardened state safety by removing sensitive credentials (`api_key`, `api_url`, `llm_provider`, `llm_model`) from serializable `AgentState` dictionary, passing them securely via graph invocation configuration (`config["configurable"]`).

### Changed
- Migrated `JobRequirements`, `CandidateMatch`, and `AgentState` in `agent/state.py` to standard `typing.TypedDict` for type safety and reducer compatibility.
- Deduplicated control flow logic by consolidating JD-detection heuristics into `route_input` (`agent/routers.py`) as the single source of truth.
- Configured configurable resume truncation character limit (`config.RESUME_TRUNCATION_LIMIT`) replacing hardcoded limit in `deep_screen_node`.

## [0.8.2] - 2026-08-02
### Added
- Wired `BaseVectorStore` into `ResumeRAGPipeline`, `JobMatcher`, `IngestionService`, and `search_mcp_server.py` via constructor and tool dependency injection.
- Implemented BM25 Okapi corpus index caching in `JobMatcher` with MD5 fingerprint hash-invalidation to eliminate per-query O(n) index rebuilds.
- Refactored `agent/nodes.py` workflow nodes to leverage `_get_llm()` and `_get_store()` helpers to accept pre-instantiated LLM models and vector stores via graph configuration (`config["configurable"]`).

## [0.8.1] - 2026-08-01
### Added
- Created pluggable vector store package (`stores/`) featuring `BaseVectorStore` structural `typing.Protocol` interface (`upsert`, `query`, `get_all`, `count`), `ChromaVectorStore`, and `QdrantVectorStore` stub.
- Added vector store unit tests in `tests/test_stores.py`.

### Fixed
- Made Vector store ingestion idempotent by migrating from `collection.add()` to `collection.upsert()` to prevent `DuplicateIDError` on re-runs.

### Changed
- Updated `IngestionService` and `ResumeRAGPipeline` chunk ID formatting to section-scoped deterministic keys (`{filename}_{section}_{index}`) to prevent chunk duplication or orphaned vectors on re-ingest.

## [0.8.0] - 2026-07-31
### Added
- Created `IngestionService` (`services/ingestion_service.py`) to separate candidate resume ingestion business logic from MCP protocol tool handlers.
- Added `IngestionService` unit tests in `tests/test_ingestion_service.py`.

## [0.7.0] - 2026-07-30
### Added
- Refactored monolithic agent script into decoupled packaged directory `agent/` (`state.py`, `prompts.py`, `nodes.py`, `routers.py`, `__init__.py`).
- Integrated automated local pre-commit hooks and GitHub Actions CI workflow using [Ruff].

## [0.6.0] - 2026-07-29
### Added
- Implemented [FastMCP] filesystem server (`filesystem_mcp_server.py`) exposing files and resource namespaces (`resumes://`).
- Created secondary search MCP server (`search_mcp_server.py`) coordinating live web search via Tavily API and ChromaDB vector search.
- Built unified MCP client manager (`mcp_client.py`) and gateway client (`fs_client.py`).

## [0.5.0] - 2026-07-28
### Added
- Created engineering conventions guidelines (`CONVENTIONS.md`) defining architectural principles, Pydantic V2 schemas, state immutability, and resilience bounds.
- Created contributing guide (`CONTRIBUTING.md`) with environment setup, unit testing, Ruff linting, and PR checklists.
- Added topic badges and repository metadata recommendations for recruiter and technical evaluation visibility.

### Changed
- Refactored `ROADMAP.md` status indicators from text markers (`[Completed]`, `[Planned]`) to clean emojis (`✅`, `⏳`, `⬜`).
- Updated `README.md` project tree, top status badges, and documentation cross-references.

## [0.4.0] - 2026-07-26
### Added
- Created custom agent tools in `tools.py` (`extract_requirements`, `compare_candidates`, `generate_interview_questions`).

## [0.3.0] - 2026-07-25
### Added
- Built initial Streamlit web dashboard (`app.py`) for dual-pane chat and candidate shortlist inspection.

## [0.2.0] - 2026-07-24
### Added
- Created synthetic candidate resume dataset generator (`generate_dataset.py`).
- Implemented RAG chunking and ChromaDB ingestion pipeline (`resume_rag.py`).
- Created hybrid semantic-BM25 job matcher (`job_matcher.py`).

## [0.1.0] - 2026-07-22
### Added
- Initial project scaffolding, virtual environment configuration, and model client setup (`config.py`).

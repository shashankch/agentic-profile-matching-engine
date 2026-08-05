# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Created pluggable vector store package (`stores/`) featuring `BaseVectorStore` structural `typing.Protocol` interface (`upsert`, `query`, `get_all`, `count`), `ChromaVectorStore`, and `QdrantVectorStore` stub.
- Introduced domain exception hierarchy (`EngineError`, `IngestionError`, `RetrievalError`, `LLMParseError`, `VectorStoreError`, `CollectionNotFoundError`) in `exceptions.py` and `stores/exceptions.py`.
- Created Pydantic V2 output models (`JobRequirementsOutput`, `DeepScreenOutput`) and `parse_json_output()` parsing helper in `tools.py`.
- Introduced Celery background worker module (`tasks.py`) and application instance (`celery_app.py`) for non-blocking asynchronous document ingestion (`async_ingest_directory`) and candidate deep screening (`async_deep_screen_candidate`).
- Created Docker containerization suite including `Dockerfile` (multi-stage Python 3.12 build), `docker-compose.yml` (orchestrating `redis:7-alpine`, Streamlit web app, and Celery worker services), and `.dockerignore`.
- Added comprehensive unit test suites in `tests/test_stores.py`, `tests/test_exceptions.py`, `tests/test_routers.py`, and `tests/test_tasks.py`.
- Implemented BM25 Okapi corpus index caching in `JobMatcher` with MD5 fingerprint hash-invalidation to eliminate per-query O(n) index rebuilds.

### Security
- Hardened state safety by removing sensitive credentials (`api_key`, `api_url`, `llm_provider`, `llm_model`) from serializable `AgentState` dictionary, passing them securely via graph invocation configuration (`config["configurable"]`).

### Fixed
- Made Vector store ingestion idempotent by migrating from `collection.add()` to `collection.upsert()` to prevent `DuplicateIDError` on re-runs.
- Resolved `asyncio` SIGCHLD deadlock in CI tests by upgrading CI to Python 3.12 (which uses signal-free `PidfdChildWatcher`) and bypassing MCP subprocess transport in CI environments.
- Fixed `ImportError` on fresh installs by ensuring search dependencies are explicitly listed in manifest files.

### Changed
- Swapped `duckduckgo-search` for `tavily-python` in MCP Search Server to utilize a reliable, AI-native search engine with 1,000 free API credits/month.
- Updated `IngestionService` and `ResumeRAGPipeline` chunk ID formatting to section-scoped deterministic keys (`{filename}_{section}_{index}`) to prevent chunk duplication or orphaned vectors on re-ingest.
- Wired `BaseVectorStore` into `ResumeRAGPipeline`, `JobMatcher`, `IngestionService`, and `search_mcp_server.py` via constructor and tool dependency injection.
- Refactored `agent/nodes.py` workflow nodes to leverage `_get_llm()` and `_get_store()` helpers to accept pre-instantiated LLM models and vector stores via graph configuration (`config["configurable"]`).
- Migrated `JobRequirements`, `CandidateMatch`, and `AgentState` in `agent/state.py` to standard `typing.TypedDict` for type safety and reducer compatibility.
- Deduplicated control flow logic by consolidating JD-detection heuristics into `route_input` (`agent/routers.py`) as the single source of truth.
- Configured configurable resume truncation character limit (`config.RESUME_TRUNCATION_LIMIT`) replacing hardcoded limit in `deep_screen_node`.
- Refactored LLM output parsing in `tools.py` and `agent/nodes.py` (`extract_requirements`, `generate_interview_questions`, `deep_screen_node`, `adjust_requirements_node`) to use Pydantic V2 `model_validate_json()` with lenient fallback.


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
- Created `IngestionService` (`services/ingestion_service.py`) to separate candidate resume ingestion business logic from MCP protocol tool handlers.
- Added comprehensive unit tests for `IngestionService` in `tests/test_ingestion_service.py`.

### Changed
- Refactored `filesystem_mcp_server.py` watchdog thread to delegate auto-ingestion to `IngestionService.ingest_file()` for single-file incremental ingestion.
- Updated `docs/architecture.md` to detail the `IngestionService` boundary.

## [0.3.0] - 2026-07-04
### Added
- Integrated automated local pre-commit hooks and GitHub Actions CI workflow using Ruff for code linting and formatting.
- Added license file (`LICENSE`) to the repository.
- Added project roadmap document (`ROADMAP.md`).
- Added robust try-except bounds and warning alert flags to the conversational agent state machine.
- Added validation checks to cap maximum experience extraction fields.

### Changed
- Refactored monolithic 833-line agent file into a modular package structure in `agent/` (decoupled into `state.py`, `prompts.py`, `nodes.py`, `routers.py`, and `__init__.py`).
- Updated project documentation (`README.md` and `ROADMAP.md`) to utilize clean, reference-style external links for core dependencies.

## [0.2.0] - 2026-07-01
### Added
- Implemented FastMCP filesystem server (`filesystem_mcp_server.py`) exposing files and resource namespaces (`resumes://`).
- Added secondary search server (`search_mcp_server.py`) coordinating live web queries and ChromaDB vector queries.
- Added synchronous async-bridged manager client (`mcp_client.py`) and unified gateway client (`fs_client.py`) for dual-mode execution (Local vs. MCP).

## [0.1.0] - 2026-06-25
### Added
- Designed LangGraph conversational workflow orchestrating requirements extraction, search, deep profile screening, and human-in-the-loop refinement.
- Created candidate resume dataset ingestion pipeline with ChromaDB and BM25 Okapi indexes.
- Built custom assessor screening tools for job requirements extraction, side-by-side matrices, and question generation.
- Created Streamlit UI dashboard supporting real-time candidate vetting and dual-pane chat logs.

[0.5.0]: https://github.com/shashankch/agentic-profile-matching-engine/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/shashankch/agentic-profile-matching-engine/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/shashankch/agentic-profile-matching-engine/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/shashankch/agentic-profile-matching-engine/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/shashankch/agentic-profile-matching-engine/releases/tag/v0.1.0

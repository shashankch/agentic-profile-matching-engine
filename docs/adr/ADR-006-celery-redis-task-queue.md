# ADR-006: Celery + Redis Task Queue for Asynchronous Heavy Operations

## Status
Accepted (Implemented in `v0.8.6`)

## Context
Heavy operations such as multi-document PDF parsing, batch vector embedding ingestion, and multi-candidate LLM audits can block web servers (e.g. Streamlit or HTTP API endpoints) if run synchronously on the main event loop thread.

## Decision
Incorporate **Celery** with **Redis** as a distributed task queue broker (`tasks.py` and `celery_app.py`). Expose non-blocking background task endpoints (`async_ingest_directory`, `async_deep_screen_candidate`) and provide Docker Compose containerization (`docker-compose.yml`) bundling Redis, Streamlit, and Celery worker services.

## Consequences
- **Positive**: Offloads high-latency background operations from the web application UI thread; provides horizontal scalability for worker processes.
- **Negative**: Adds Redis as an operational infrastructure dependency for asynchronous deployment.

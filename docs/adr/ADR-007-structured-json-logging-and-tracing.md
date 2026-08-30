# ADR-007: Structured JSON Logging & Pluggable Tracing Pipeline

## Status
Accepted (Implemented in `v0.9.1` and `v0.9.2`)

## Context
Standard unformatted `print()` statements lack execution context, timestamps, log levels, and duration metrics required for production debugging and APM monitoring in cloud environments.

## Decision
Implement structured JSON logging (`JsonFormatter`, `get_logger`) outputting log records with standard keys (`timestamp`, `level`, `logger`, `message`). Wrap all 9 LangGraph workflow nodes with a `@trace_node` decorator that tracks node execution start (`node_start`), completion (`node_end`), errors (`node_error`), and millisecond latency (`elapsed_ms`). Extend `@trace_node` with opt-in tracing for **Langfuse** (`OBSERVABILITY_BACKEND=langfuse`) and **OpenTelemetry** (`OBSERVABILITY_BACKEND=opentelemetry`).

## Consequences
- **Positive**: Enables zero-dependency structured JSON log parsing for Datadog, AWS CloudWatch, and ELK stacks; provides opt-in integration with modern LLM tracing platforms.
- **Negative**: Requires setting environment variables to enable external tracing backends.

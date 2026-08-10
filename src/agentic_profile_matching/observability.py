import functools
import json
import logging
import os
import time
from typing import Any, Callable, Dict, Optional

# Base logger setup
LOG_FORMAT = "json"  # Default log format


class JsonFormatter(logging.Formatter):
    """
    Structured JSON log formatter for structured observability.
    Renders log records as single-line JSON objects with standard fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include custom extra fields if provided
        if hasattr(record, "event"):
            log_data["event"] = getattr(record, "event")
        if hasattr(record, "node"):
            log_data["node"] = getattr(record, "node")
        if hasattr(record, "elapsed_ms"):
            log_data["elapsed_ms"] = getattr(record, "elapsed_ms")

        # Capture additional extra payload
        if hasattr(record, "extra") and isinstance(getattr(record, "extra"), dict):
            log_data.update(getattr(record, "extra"))

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def get_logger(name: str = "agentic_profile_matching") -> logging.Logger:
    """
    Returns a configured structured Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger("agentic_profile_matching")


def trace_node(node_name: str) -> Callable:
    """
    Decorator for tracking LangGraph node execution latency, inputs, and outputs.
    Logs structured node_start and node_end events with elapsed_ms duration.
    Supports opt-in tracing integration via Langfuse or OpenTelemetry backends.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(state: Any, config: Optional[Any] = None, *args: Any, **kwargs: Any) -> Any:
            node_logger = get_logger(f"agentic_profile_matching.node.{node_name}")
            start_time = time.perf_counter()

            node_logger.info(
                f"Starting node execution: {node_name}",
                extra={"event": "node_start", "node": node_name},
            )

            # Check configured tracing backend
            backend = os.getenv("OBSERVABILITY_BACKEND", "none").lower()

            def _run_func() -> Any:
                if config is not None:
                    return func(state, config, *args, **kwargs)
                return func(state, *args, **kwargs)

            try:
                if backend == "opentelemetry":
                    try:
                        from opentelemetry import trace

                        tracer = trace.get_tracer("agentic_profile_matching")
                        with tracer.start_as_current_span(node_name) as span:
                            if span.is_recording():
                                span.set_attribute("node.name", node_name)
                            result = _run_func()
                            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
                            if span.is_recording():
                                span.set_attribute("elapsed_ms", elapsed_ms)
                    except ImportError:
                        node_logger.warning(
                            "OBSERVABILITY_BACKEND set to 'opentelemetry', but opentelemetry SDK is not installed. Falling back to default JSON logging."
                        )
                        result = _run_func()
                elif backend == "langfuse":
                    try:
                        from langfuse.decorators import observe

                        @observe(name=node_name)
                        def _observed_execution():
                            return _run_func()

                        result = _observed_execution()
                    except ImportError:
                        node_logger.warning(
                            "OBSERVABILITY_BACKEND set to 'langfuse', but langfuse SDK is not installed. Falling back to default JSON logging."
                        )
                        result = _run_func()
                else:
                    result = _run_func()

                elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
                node_logger.info(
                    f"Completed node execution: {node_name} in {elapsed_ms}ms",
                    extra={"event": "node_end", "node": node_name, "elapsed_ms": elapsed_ms},
                )
                return result
            except Exception as e:
                elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
                node_logger.error(
                    f"Failed node execution: {node_name} - Error: {e}",
                    extra={"event": "node_error", "node": node_name, "elapsed_ms": elapsed_ms},
                    exc_info=True,
                )
                raise e

        return wrapper

    return decorator

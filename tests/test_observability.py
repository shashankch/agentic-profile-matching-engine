import json
import logging
from agentic_profile_matching.observability import JsonFormatter, get_logger, trace_node


def test_json_formatter_structure():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Test log message",
        args=(),
        exc_info=None,
    )
    record.event = "unit_test"
    record.node = "test_node"
    record.elapsed_ms = 42.5

    output = formatter.format(record)
    parsed = json.loads(output)

    assert parsed["logger"] == "test_logger"
    assert parsed["level"] == "INFO"
    assert parsed["message"] == "Test log message"
    assert parsed["event"] == "unit_test"
    assert parsed["node"] == "test_node"
    assert parsed["elapsed_ms"] == 42.5
    assert "timestamp" in parsed


def test_get_logger_singleton_handler():
    logger_instance = get_logger("test_get_logger")
    assert isinstance(logger_instance, logging.Logger)
    assert len(logger_instance.handlers) == 1
    assert isinstance(logger_instance.handlers[0].formatter, JsonFormatter)

    # Calling get_logger again should return same instance without adding duplicate handlers
    logger_instance_2 = get_logger("test_get_logger")
    assert len(logger_instance_2.handlers) == 1


def test_trace_node_decorator_success(caplog):
    caplog.set_level(logging.INFO)

    @trace_node("sample_node")
    def sample_node_func(state, config=None):
        return {"processed": True, "count": state.get("count", 0) + 1}

    state_input = {"count": 5}
    result = sample_node_func(state_input, config={"configurable": {}})

    assert result["processed"] is True
    assert result["count"] == 6

    # Verify log messages captured
    messages = [r.getMessage() for r in caplog.records]
    assert any("Starting node execution: sample_node" in m for m in messages)
    assert any("Completed node execution: sample_node" in m for m in messages)


def test_trace_node_decorator_exception(caplog):
    caplog.set_level(logging.ERROR)

    @trace_node("failing_node")
    def failing_node_func(state):
        raise ValueError("Simulated node failure")

    try:
        failing_node_func({"test": "data"})
    except ValueError as e:
        assert str(e) == "Simulated node failure"

    messages = [r.getMessage() for r in caplog.records]
    assert any("Failed node execution: failing_node" in m for m in messages)

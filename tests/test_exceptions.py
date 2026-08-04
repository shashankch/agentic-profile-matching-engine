import pytest
from agentic_profile_matching.exceptions import (
    EngineError,
    IngestionError,
    RetrievalError,
    LLMParseError,
)
from agentic_profile_matching.stores.exceptions import VectorStoreError, CollectionNotFoundError
from agentic_profile_matching.tools import (
    JobRequirementsOutput,
    DeepScreenOutput,
    parse_json_output,
)


def test_domain_exception_hierarchy():
    assert issubclass(IngestionError, EngineError)
    assert issubclass(RetrievalError, EngineError)
    assert issubclass(LLMParseError, EngineError)
    assert issubclass(VectorStoreError, EngineError)
    assert issubclass(CollectionNotFoundError, VectorStoreError)


def test_parse_json_output_clean():
    raw_json = '{"title": "Senior Engineer", "min_experience_years": 5}'
    parsed = parse_json_output(raw_json, model_cls=JobRequirementsOutput)
    assert parsed["title"] == "Senior Engineer"
    assert parsed["min_experience_years"] == 5
    assert parsed["must_have_skills"] == []


def test_parse_json_output_markdown_wrapped():
    raw = """```json
    {
        "title": "Staff AI Engineer",
        "must_have_skills": ["Python", "PyTorch"]
    }
    ```"""
    parsed = parse_json_output(raw, model_cls=JobRequirementsOutput)
    assert parsed["title"] == "Staff AI Engineer"
    assert parsed["must_have_skills"] == ["Python", "PyTorch"]


def test_parse_json_output_deep_screen_contract():
    raw = '{"strengths": ["Deep learning", "Distributed systems"], "screening_status": "Strong Hire"}'
    parsed = parse_json_output(raw, model_cls=DeepScreenOutput)
    assert parsed["strengths"] == ["Deep learning", "Distributed systems"]
    assert parsed["screening_status"] == "Strong Hire"
    assert parsed["gaps"] == []


def test_parse_json_output_malformed_raises_llm_parse_error():
    raw = "Not a valid json response at all!"
    with pytest.raises(LLMParseError):
        parse_json_output(raw)

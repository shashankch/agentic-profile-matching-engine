from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage
from agentic_profile_matching.agent.routers import (
    route_input,
    _classify_via_semantic_similarity,
    _classify_via_llm,
)
from agentic_profile_matching.agent.state import AgentState


def test_route_input_empty_messages():
    state: AgentState = {"messages": [], "requirements": {}}
    assert route_input(state) == "extract_requirements"


def test_route_input_raw_job_description():
    jd_text = """
    Software Engineer - Backend
    Requirements:
    - 5+ years Python experience
    - Microservices and Docker/Kubernetes
    - SQL database design
    """
    state: AgentState = {
        "messages": [HumanMessage(content=jd_text)],
        "requirements": {},
    }
    assert route_input(state) == "extract_requirements"


def test_route_input_semantic_search():
    state: AgentState = {
        "messages": [HumanMessage(content="Search for candidates with Java and Python Experience")],
        "requirements": {"title": "Old Search"},
        "shortlist": [],
    }
    assert route_input(state) == "extract_requirements"


def test_route_input_semantic_adjust():
    state: AgentState = {
        "messages": [
            HumanMessage(content="Make Python a mandatory must-have skill and increase experience to 7 years")
        ],
        "requirements": {"title": "Python Developer", "min_experience_years": 3},
        "shortlist": [{"candidate_id": "1", "name": "Alice Smith", "score": 90}],
    }
    assert route_input(state) == "adjust_requirements"


def test_route_input_semantic_conversational():
    state: AgentState = {
        "messages": [HumanMessage(content="Why is candidate Alice Smith ranked higher than Bob?")],
        "requirements": {"title": "Python Developer"},
        "shortlist": [
            {"candidate_id": "1", "name": "Alice Smith", "score": 90},
            {"candidate_id": "2", "name": "Bob Jones", "score": 75},
        ],
    }
    assert route_input(state) == "conversational_query"


def test_classify_via_semantic_similarity_direct():
    intent = _classify_via_semantic_similarity("Looking for a full stack developer with React and Node.js")
    assert intent == "extract_requirements"

    intent_adjust = _classify_via_semantic_similarity("Exclude candidates without Docker experience")
    assert intent_adjust == "adjust_requirements"


@patch("agentic_profile_matching.tools.invoke_structured")
@patch("agentic_profile_matching.config.get_llm_model")
def test_classify_via_llm(mock_get_llm, mock_invoke_structured):
    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm
    mock_invoke_structured.return_value = {
        "intent": "conversational_query",
        "reasoning": "User is asking for comparison between candidates.",
    }

    state: AgentState = {
        "messages": [HumanMessage(content="Break down candidate strengths and differences")],
        "requirements": {"title": "Lead Dev"},
        "shortlist": [{"candidate_id": "1", "name": "Alice", "score": 90}],
        "api_key": "dummy-key",
    }

    decision = _classify_via_llm(state)
    assert decision == "conversational_query"


def test_route_input_general_tech_query_with_empty_requirements():
    state: AgentState = {
        "messages": [HumanMessage(content="can you tell me about graph enginnering in 2026")],
        "requirements": {},
        "shortlist": [],
    }
    assert route_input(state) == "conversational_query"


def test_route_input_web_search_query():
    state: AgentState = {
        "messages": [HumanMessage(content="search google for python 3.14 features")],
        "requirements": {},
        "shortlist": [],
    }
    assert route_input(state) == "conversational_query"


def test_route_input_explicit_candidate_sourcing():
    state: AgentState = {
        "messages": [HumanMessage(content="Search resumes for Python cloud architects with 5+ years experience")],
        "requirements": {},
        "shortlist": [],
    }
    assert route_input(state) == "extract_requirements"


@patch("agentic_profile_matching.tools.invoke_structured")
def test_generate_dynamic_intent_anchors(mock_invoke_structured):
    mock_invoke_structured.return_value = {
        "extract_requirements": ["Find candidates with Go", "Source Java architects"],
        "adjust_requirements": ["Add Rust to skills", "Require 5+ years"],
        "conversational_query": ["What is LangGraph?", "Compare top profiles"],
    }
    from agentic_profile_matching.agent.routers import generate_dynamic_intent_anchors

    try:
        mock_llm = MagicMock()
        anchors = generate_dynamic_intent_anchors(mock_llm)
        assert "extract_requirements" in anchors
        assert len(anchors["extract_requirements"]) == 2
        assert "conversational_query" in anchors
    finally:
        import agentic_profile_matching.agent.routers as r_mod

        r_mod._DYNAMIC_INTENT_ANCHORS = None
        r_mod._ANCHOR_EMBEDDINGS = None


def test_calculate_intent_margin_high_confidence():
    from agentic_profile_matching.agent.routers import calculate_intent_margin

    query = "Search resumes for candidates with Java and Python experience."
    res = calculate_intent_margin(query)
    assert res.top1_intent == "extract_requirements"
    assert res.top1_score >= 0.55
    assert res.margin >= 0.12
    assert res.is_confident is True
    assert res.all_scores


@patch("agentic_profile_matching.agent.routers._get_embedder")
def test_calculate_intent_margin_ambiguous_escalation(mock_get_embedder):
    import numpy as np
    from agentic_profile_matching.agent.routers import calculate_intent_margin

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = [np.array([1.0, 0.0])]

    mock_anchors = {
        "extract_requirements": np.array([[0.50, 0.0]]),
        "adjust_requirements": np.array([[0.48, 0.0]]),  # Delta = 0.02 < 0.12 -> Ambiguous!
        "conversational_query": np.array([[0.10, 0.0]]),
    }
    mock_get_embedder.return_value = (mock_embedder, mock_anchors)

    res = calculate_intent_margin("Ambiguous query text")
    assert res.top1_score == 0.50
    assert abs(res.margin - 0.02) < 1e-4
    assert res.is_confident is False
    assert res.needs_escalation is True


@patch("agentic_profile_matching.agent.routers._classify_via_llm")
@patch("agentic_profile_matching.agent.routers.calculate_intent_margin")
def test_route_input_escalates_on_ambiguity(mock_calc_margin, mock_classify_llm):
    from agentic_profile_matching.agent.routers import RouteMarginResult

    mock_calc_margin.return_value = RouteMarginResult(
        top1_intent="adjust_requirements",
        top1_score=0.48,
        top2_intent="extract_requirements",
        top2_score=0.45,
        margin=0.03,
        is_confident=False,
        needs_escalation=True,
        all_scores={},
    )
    mock_classify_llm.return_value = "conversational_query"

    state: AgentState = {
        "messages": [HumanMessage(content="Uncertain query prompt")],
        "requirements": {"title": "Engineer"},
        "shortlist": [],
    }

    routed = route_input(state)
    assert routed == "conversational_query"
    mock_classify_llm.assert_called_once_with(state)

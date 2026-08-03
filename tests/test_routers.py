from langchain_core.messages import HumanMessage
from agentic_profile_matching.agent.routers import route_input
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


def test_route_input_conversational_query_with_shortlist():
    state: AgentState = {
        "messages": [HumanMessage(content="Why is candidate Alice Smith ranked higher than Bob?")],
        "requirements": {"title": "Python Developer"},
        "shortlist": [
            {"candidate_id": "1", "name": "Alice Smith", "score": 90},
            {"candidate_id": "2", "name": "Bob Jones", "score": 75},
        ],
    }
    assert route_input(state) == "conversational_query"


def test_route_input_adjust_requirements():
    state: AgentState = {
        "messages": [HumanMessage(content="Focus more on candidates with Docker and Kubernetes experience")],
        "requirements": {"title": "Python Developer", "min_experience_years": 3},
        "shortlist": [{"candidate_id": "1", "name": "Alice Smith", "score": 90}],
    }
    assert route_input(state) == "adjust_requirements"

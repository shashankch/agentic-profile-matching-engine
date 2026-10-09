from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage
from agentic_profile_matching.agent.subgraphs import (
    build_jd_analyzer_subgraph,
    build_talent_retrieval_subgraph,
    build_deep_screening_subgraph,
    build_synthesis_subgraph,
    JDAnalyzerState,
    TalentRetrievalState,
    DeepScreeningState,
    SynthesisState,
)


def test_jd_analyzer_subgraph_execution():
    """Verify isolated execution of JDAnalyzerSubgraph with mocked LLM."""
    subgraph = build_jd_analyzer_subgraph()
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = (
        '{"title": "Lead Backend Engineer", "must_have_skills": ["Go", "Kubernetes", "gRPC"], '
        '"nice_to_have_skills": [], "min_experience_years": 7, "education_level": "BS", "other_constraints": []}'
    )

    state: JDAnalyzerState = {
        "messages": [
            HumanMessage(
                content="Looking for a Lead Backend Engineer with 7+ years of Go, Kubernetes, and gRPC experience."
            )
        ],
        "requirements": {},
        "current_round": 1,
        "errors": [],
    }

    config = {"configurable": {"llm": mock_llm}}
    result = subgraph.invoke(state, config=config)
    assert "requirements" in result
    assert result["requirements"].get("title") == "Lead Backend Engineer"
    assert result["requirements"].get("must_have_skills") == ["Go", "Kubernetes", "gRPC"]
    assert result.get("errors") == []


def test_jd_analyzer_subgraph_fallback_on_unconfigured_llm():
    """Verify isolated execution of JDAnalyzerSubgraph gracefully falls back when LLM fails."""
    subgraph = build_jd_analyzer_subgraph()
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("Simulated missing API key or client error")

    state: JDAnalyzerState = {
        "messages": [
            HumanMessage(
                content="Looking for a Lead Backend Engineer with 7+ years of Go, Kubernetes, and gRPC experience."
            )
        ],
        "requirements": {},
        "current_round": 1,
        "errors": [],
    }

    config = {"configurable": {"llm": mock_llm}}
    result = subgraph.invoke(state, config=config)
    assert "requirements" in result
    assert result["requirements"].get("title") == "Software Engineer"


def test_talent_retrieval_subgraph_execution():
    """Verify isolated execution of TalentRetrievalSubgraph with mock store."""
    subgraph = build_talent_retrieval_subgraph()

    mock_store = MagicMock()
    mock_store.get_all.return_value = {
        "documents": ["Candidate A has 6 years Python and Kubernetes experience."],
        "metadatas": [
            {
                "candidate_name": "Dev A",
                "skills": "Python, Kubernetes",
                "experience_years": 6,
                "education": "BS CS",
                "resume_path": "/fake/path.txt",
                "filename": "dev_a.txt",
            }
        ],
        "ids": ["dev_a_chunk_0"],
    }
    mock_store.query.return_value = {
        "documents": [["Candidate A has 6 years Python and Kubernetes experience."]],
        "metadatas": [
            [
                {
                    "candidate_name": "Dev A",
                    "skills": "Python, Kubernetes",
                    "experience_years": 6,
                    "education": "BS CS",
                    "resume_path": "/fake/path.txt",
                    "filename": "dev_a.txt",
                }
            ]
        ],
        "ids": [["dev_a_chunk_0"]],
        "distances": [[0.15]],
    }

    state: TalentRetrievalState = {
        "requirements": {
            "title": "Backend Engineer",
            "must_have_skills": ["Python", "Kubernetes"],
            "nice_to_have_skills": [],
            "skill_expansions": {},
            "min_experience_years": 5,
            "education_level": "BS CS",
            "other_constraints": [],
        },
        "shortlist": [],
        "coarse_screen_limit": 5,
        "current_round": 1,
        "errors": [],
    }

    config = {"configurable": {"store": mock_store}}
    result = subgraph.invoke(state, config=config)
    assert "shortlist" in result
    assert len(result["shortlist"]) > 0
    assert result["shortlist"][0]["name"] == "Dev A"


def test_deep_screening_subgraph_fallback():
    """Verify isolated execution of DeepScreeningSubgraph with unconfigured LLM."""
    subgraph = build_deep_screening_subgraph()
    state: DeepScreeningState = {
        "requirements": {
            "title": "Cloud Architect",
            "must_have_skills": ["AWS", "Terraform"],
            "nice_to_have_skills": [],
            "skill_expansions": {},
            "min_experience_years": 5,
            "education_level": "BS",
            "other_constraints": [],
        },
        "shortlist": [
            {
                "candidate_id": "cand_1",
                "name": "Jane Cloud",
                "score": 92.0,
                "raw_text": "Jane Cloud. 8 years AWS cloud architecture and Terraform.",
                "matched_skills": ["AWS", "Terraform"],
                "missing_skills": [],
                "experience_years": 8,
                "education": "BS Computer Science",
                "relevance_excerpts": [],
            }
        ],
        "deep_screen_limit": 1,
        "current_round": 1,
        "errors": [],
    }

    # In isolation, should perform screening cleanly
    result = subgraph.invoke(state)
    assert "shortlist" in result
    assert len(result["shortlist"]) == 1
    screened = result["shortlist"][0]
    assert screened.get("screening_status") in [
        "Strong Hire",
        "Borderline Hire",
        "Rejected / No-Hire",
        "Screened",
    ]
    assert "technical_score" in screened
    assert "domain_fit_score" in screened


def test_synthesis_subgraph_execution():
    """Verify isolated execution of SynthesisSubgraph with mocked LLM."""
    subgraph = build_synthesis_subgraph()
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = (
        "1. How do you design scalable React component trees?\n"
        "2. Describe your experience with Node.js event-loop concurrency."
    )

    state: SynthesisState = {
        "requirements": {
            "title": "Full Stack Dev",
            "must_have_skills": ["React", "Node.js"],
            "nice_to_have_skills": [],
            "skill_expansions": {},
            "min_experience_years": 3,
            "education_level": "Bachelor",
            "other_constraints": [],
        },
        "shortlist": [
            {
                "candidate_id": "cand_fs",
                "name": "Alex Stack",
                "score": 88.0,
                "matched_skills": ["React", "Node.js"],
                "missing_skills": [],
                "experience_years": 4,
                "education": "BS",
                "relevance_excerpts": [],
                "screening_status": "Strong Hire",
                "screening_reasoning": "Solid full stack developer",
                "strengths": ["React expert"],
                "gaps": [],
                "improvement_suggestions": "None",
            }
        ],
        "previous_shortlist": [],
        "recommendation_limit": 1,
        "current_round": 1,
        "final_report": "",
        "ranking_explanation": "",
        "errors": [],
    }

    config = {"configurable": {"llm": mock_llm}}
    result = subgraph.invoke(state, config=config)
    assert "final_report" in result
    assert "Alex Stack" in result["final_report"]
    assert "Tailored Screening Questions" in result["final_report"] or "Questions" in result["final_report"]

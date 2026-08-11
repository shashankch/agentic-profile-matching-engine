import pytest
from unittest.mock import MagicMock
from agentic_profile_matching.agent.nodes import deep_screen_node


@pytest.mark.eval
def test_deep_screen_faithfulness_and_groundedness():
    """
    Evaluates that deep_screen_node produces structured candidate match diagnostics
    grounded strictly in provided candidate profile data.
    """
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content="""{
            "candidates": [
                {
                    "candidate_name": "Marcus Vance",
                    "overall_score": 88,
                    "strengths": ["Strong Python experience", "Kubernetes cluster administration"],
                    "gaps": ["No direct AWS cloud experience"],
                    "improvements": ["Acquire AWS Cloud Practitioner certification"],
                    "recommendation": "Hire"
                }
            ]
        }"""
    )

    state = {
        "job_requirements": {
            "title": "Senior Python Engineer",
            "must_have_skills": ["Python", "Kubernetes"],
            "nice_to_have_skills": ["AWS"],
            "min_experience_years": 5,
            "education_level": "Not Specified",
            "other_constraints": [],
        },
        "shortlist": [
            {
                "candidate_id": "resumes/resume_01.txt",
                "resume_id": "resume_01",
                "candidate_name": "Marcus Vance",
                "matched_skills": ["Python", "Kubernetes"],
                "experience_years": 6,
                "degree": "B.S. Computer Science",
                "raw_text": "Marcus Vance is a Senior Python Developer with 6 years experience managing Kubernetes clusters.",
            }
        ],
        "messages": [],
    }

    config = {"configurable": {"llm": mock_llm}}
    result = deep_screen_node(state, config=config)

    assert "shortlist" in result
    shortlist = result["shortlist"]
    assert len(shortlist) == 1

    candidate = shortlist[0]
    assert candidate["candidate_name"] == "Marcus Vance"
    assert candidate["screening_status"] == "Screened"

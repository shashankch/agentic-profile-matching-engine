import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage
from agentic_profile_matching.agent import matching_agent_workflow


def test_agent_graph_assembly_and_invocation():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_resume = Path(tmpdir) / "resume_alice.txt"
        tmp_resume.write_text("Alice Smith\nEXPERIENCE\n5 years of Python development experience.\nEDUCATION\nBS CS\n")

        def _mock_llm_invoke(prompt_messages):
            prompt_text = str(prompt_messages)
            mock_resp = MagicMock()
            if "must_have_skills" in prompt_text or "Job Description" in prompt_text:
                mock_resp.content = json.dumps(
                    {
                        "title": "Backend Engineer",
                        "must_have_skills": ["Python"],
                        "nice_to_have_skills": [],
                        "min_experience_years": 3,
                        "education_level": "Bachelor",
                        "other_constraints": [],
                    }
                )
            else:
                mock_resp.content = json.dumps(
                    {
                        "strengths": ["Strong Python experience"],
                        "gaps": [],
                        "improvement_suggestions": "None",
                        "screening_status": "Strong Hire",
                        "screening_reasoning": "Excellent technical fit",
                        "questions": ["What Python frameworks do you use?"],
                    }
                )
            return mock_resp

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = _mock_llm_invoke

        mock_store = MagicMock()
        mock_store.get_all.return_value = {
            "documents": [tmp_resume.read_text()],
            "metadatas": [
                {
                    "filename": tmp_resume.name,
                    "candidate_name": "Alice Smith",
                    "skills": "Python, SQL",
                    "experience_years": 5,
                    "education": "BS Computer Science",
                    "resume_path": str(tmp_resume),
                }
            ],
            "ids": [f"{tmp_resume.name}_general_0"],
        }
        mock_store.query.return_value = {
            "documents": [[tmp_resume.read_text()]],
            "metadatas": [
                [
                    {
                        "filename": tmp_resume.name,
                        "candidate_name": "Alice Smith",
                        "skills": "Python, SQL",
                        "experience_years": 5,
                        "education": "BS Computer Science",
                        "resume_path": str(tmp_resume),
                    }
                ]
            ],
            "ids": [[f"{tmp_resume.name}_general_0"]],
            "distances": [[0.1]],
        }

        state_input = {
            "messages": [
                HumanMessage(
                    content="Job Description:\nSenior Backend Engineer\nMust have 5+ years experience in Python and SQL."
                )
            ],
            "requirements": {},
            "shortlist": [],
            "coarse_screen_limit": 1,
            "deep_screen_limit": 1,
            "recommendation_limit": 1,
            "errors": [],
        }

        config_dict = {
            "configurable": {
                "thread_id": "test-thread-graph",
                "llm": mock_llm,
                "store": mock_store,
            }
        }

        result = matching_agent_workflow.invoke(state_input, config=config_dict)

        assert "requirements" in result
        assert "shortlist" in result
        assert "final_report" in result
        assert len(result["shortlist"]) > 0
        assert result["shortlist"][0]["name"] == "Alice Smith"

import pytest
from unittest.mock import MagicMock, patch
from agentic_profile_matching.tools import (
    execute_with_retry,
    extract_requirements,
    compare_candidates,
    generate_interview_questions,
)


def test_execute_with_retry_success():
    func = MagicMock(return_value="success")
    res = execute_with_retry(func)
    assert res == "success"
    assert func.call_count == 1


def test_execute_with_retry_rate_limit():
    call_count = 0

    def func_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("Rate limit hit (429)")
        return "success"

    func = MagicMock(side_effect=func_side_effect)

    with patch("time.sleep") as mock_sleep:
        res = execute_with_retry(func)
        assert res == "success"
        assert call_count == 2
        # verify sleep was called to wait/throttle
        assert mock_sleep.call_count >= 1  # rate limit backoff sleep


def test_execute_with_retry_non_retryable_error():
    func = MagicMock(side_effect=ValueError("Some other error"))

    with pytest.raises(ValueError):
        execute_with_retry(func)


def test_extract_requirements_success():
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = """
    ```json
    {
        "title": "Senior Python Developer",
        "must_have_skills": ["Python", "Docker"],
        "nice_to_have_skills": ["Kubernetes"],
        "min_experience_years": 5,
        "education_level": "Bachelor",
        "other_constraints": []
    }
    ```
    """
    mock_llm.invoke.return_value = mock_response

    jd = "Need 5 years Python experience."
    res = extract_requirements(jd, mock_llm)

    assert res["title"] == "Senior Python Developer"
    assert "Python" in res["must_have_skills"]
    assert res["min_experience_years"] == 5


def test_extract_requirements_fallback():
    mock_llm = MagicMock(side_effect=Exception("API limit or failure"))
    jd = "Need 5 years Python experience."

    res = extract_requirements(jd, mock_llm)
    # Check fallback structure is returned
    assert res["title"] == "Software Engineer"
    assert isinstance(res["must_have_skills"], list)


def test_compare_candidates():
    shortlist = [
        {
            "candidate_id": "c1.pdf",
            "name": "Alice Smith",
            "score": 90,
            "experience_years": 6,
            "education": "Master",
            "matched_skills": ["Python", "Docker"],
            "missing_skills": [],
            "screening_status": "Shortlisted",
            "strengths": ["Clear communication"],
            "gaps": [],
        },
        {
            "candidate_id": "c2.pdf",
            "name": "Bob Jones",
            "score": 75,
            "experience_years": 4,
            "education": "Bachelor",
            "matched_skills": ["Python"],
            "missing_skills": ["Docker"],
            "screening_status": "Screened",
            "strengths": ["Quick learner"],
            "gaps": ["No docker experience"],
        },
    ]

    # Test comparing candidate list matching Alice and Bob
    table = compare_candidates(["c1.pdf", "c2.pdf"], shortlist)

    assert "Alice Smith" in table
    assert "Bob Jones" in table
    assert "90/100" in table
    assert "75/100" in table
    assert "Experience" in table


def test_generate_interview_questions_success():
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '["What is your experience with Docker?", "Explain python decorators."]'
    mock_llm.invoke.return_value = mock_response

    res = generate_interview_questions(
        candidate_name="Alice",
        skills=["Python"],
        gaps=["Docker"],
        requirements={"title": "Python Dev"},
        llm=mock_llm,
    )

    assert len(res) == 2
    assert "Docker" in res[0]


def test_generate_interview_questions_fallback():
    mock_llm = MagicMock(side_effect=Exception("Error"))
    res = generate_interview_questions(
        candidate_name="Alice",
        skills=["Python"],
        gaps=["Docker"],
        requirements={"title": "Python Dev"},
        llm=mock_llm,
    )

    assert len(res) == 3  # fallback questions
    assert any("Docker" in q or "Docker" in res[0] for q in res) or any("software engineering" in q for q in res)

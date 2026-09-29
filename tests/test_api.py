"""
Integration tests for Yojaka AI Headless FastAPI Gateway.
Tests health check, requirements extraction, candidate matching, and SSE streaming.
"""

from unittest.mock import patch
from fastapi.testclient import TestClient
import pytest

from agentic_profile_matching.api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_health_endpoints(client):
    """Verifies that /health and /api/v1/health return healthy status and version."""
    r1 = client.get("/health")
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["status"] == "healthy"
    assert d1["version"] == "1.3.0"
    assert d1["service"] == "yojaka-ai-gateway"

    r2 = client.get("/api/v1/health")
    assert r2.status_code == 200
    assert r2.json()["version"] == "1.3.0"


def test_jobs_extract_endpoint(client):
    """Verifies /api/v1/jobs/extract returns structured requirements."""
    mock_reqs = {
        "title": "Senior Python Engineer",
        "must_have_skills": ["Python", "FastAPI"],
        "nice_to_have_skills": ["Docker"],
        "min_experience_years": 5,
        "education_level": "BS",
    }
    with patch("agentic_profile_matching.api.routes.extract_requirements", return_value=mock_reqs):
        payload = {"job_description": "We are seeking a Senior Python Engineer with 5+ years of FastAPI experience."}
        response = client.post("/api/v1/jobs/extract", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["requirements"]["title"] == "Senior Python Engineer"
        assert "Python" in data["requirements"]["must_have_skills"]


def test_candidates_match_endpoint(client):
    """Verifies /api/v1/candidates/match parses JD and returns candidate profiles."""
    mock_candidates = [
        {
            "candidate_id": "data/resumes/resume_alice.pdf",
            "name": "Alice Smith",
            "score": 92,
            "matched_skills": ["Python", "FastAPI"],
            "missing_skills": [],
            "experience_years": 6,
            "education": "BS Computer Science",
            "screening_status": "Strong Hire",
            "strengths": ["Strong architectural background"],
            "gaps": [],
            "improvement_suggestions": "",
            "interview_questions": ["Explain ASGI vs WSGI"],
        }
    ]
    with patch(
        "agentic_profile_matching.api.routes.extract_requirements",
        return_value={"title": "Python Engineer", "must_have_skills": ["Python"]},
    ):
        with patch("agentic_profile_matching.job_matcher.JobMatcher.match", return_value=mock_candidates):
            payload = {
                "job_description": "Looking for a seasoned Python Engineer.",
                "top_k": 5,
                "must_have_skills": ["Python"],
            }
            response = client.post("/api/v1/candidates/match", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["total_matched"] == 1
            assert data["candidates"][0]["name"] == "Alice Smith"
            assert data["candidates"][0]["score"] == 92


def test_workflow_stream_endpoint(client):
    """Verifies /api/v1/workflow/stream returns a Server-Sent Events stream with events."""
    mock_events = [
        {"extract_requirements": {"requirements": {"title": "Backend Architect"}}},
        {"rank_candidates": {"shortlist": [{"name": "Bob", "score": 88}]}},
    ]
    with patch(
        "agentic_profile_matching.matching_agent.matching_agent_workflow.stream",
        return_value=iter(mock_events),
    ):
        payload = {"query": "Find senior backend architects with Kubernetes"}
        response = client.post("/api/v1/workflow/stream", json=payload)
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        body = response.text
        assert "event: session_start" in body
        assert "event: node_update" in body
        assert "event: done" in body

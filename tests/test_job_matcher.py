import pytest
from unittest.mock import MagicMock, patch
from agentic_profile_matching.job_matcher import JobMatcher

@patch("agentic_profile_matching.job_matcher.SentenceTransformer")
@patch("agentic_profile_matching.job_matcher.chromadb.PersistentClient")
def test_job_matcher_match(mock_chroma_client, mock_sentence_transformer):
    # Mock Chroma setup
    mock_collection = MagicMock()
    mock_chroma_client.return_value.get_collection.return_value = mock_collection
    
    # Mock SentenceTransformer setup
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = MagicMock()
    # Mock the return value of encode().tolist()
    mock_embedder.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
    mock_sentence_transformer.return_value = mock_embedder
    
    # Define mock DB documents, metadatas, ids
    mock_collection.get.return_value = {
        "documents": [
            "Experienced Python developer working with Docker and microservices.",
            "Java engineer with experience in Spring Boot."
        ],
        "metadatas": [
            {
                "candidate_name": "Alice Smith",
                "resume_path": "alice.pdf",
                "experience_years": 5,
                "skills": "Python, Docker, Microservices",
                "education": "Master",
                "section": "WORK EXPERIENCE"
            },
            {
                "candidate_name": "Bob Jones",
                "resume_path": "bob.pdf",
                "experience_years": 2,
                "skills": "Java, Spring Boot",
                "education": "Bachelor",
                "section": "WORK EXPERIENCE"
            }
        ],
        "ids": ["id1", "id2"]
    }
    
    # Mock query result (Alice is closer to query, lower distance)
    mock_collection.query.return_value = {
        "ids": [["id1", "id2"]],
        "distances": [[0.2, 0.8]]
    }
    
    # Instantiate JobMatcher
    matcher = JobMatcher()
    
    # Run match with min_exp=3 and Python filter
    res = matcher.match(
        job_description="Python developer with 3+ years experience.",
        k=10,
        min_exp=3,
        must_have_skills=["Python"],
        apply_filters=True
    )
    
    assert len(res["top_matches"]) == 1 # Bob has 2 years exp and no Python, so filtered out
    match = res["top_matches"][0]
    assert match["candidate_name"] == "Alice Smith"
    assert match["match_score"] > 50
    assert "Python" in match["matched_skills"]

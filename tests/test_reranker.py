"""
Unit tests for Two-Stage Hybrid Retrieval & Cross-Encoder Reranking (Deliverable 17.3).
Tests RRF calculation, pair scoring, score blending, and offline fallback modes.
"""

from unittest.mock import MagicMock, patch
from agentic_profile_matching.services.reranker import (
    CrossEncoderReranker,
    compute_rrf_scores,
)
from agentic_profile_matching.job_matcher import JobMatcher


def test_compute_rrf_scores():
    """Verifies that RRF assigns highest score to candidates ranked high in both dense and sparse lists."""
    dense_ranks = ["cand_A", "cand_B", "cand_C"]
    sparse_ranks = ["cand_B", "cand_A", "cand_D"]

    rrf = compute_rrf_scores(dense_ranks, sparse_ranks, k_rrf=60)

    # Both cand_A and cand_B appear in top 2 in both lists
    assert rrf["cand_A"] > rrf["cand_C"]
    assert rrf["cand_B"] > rrf["cand_C"]
    assert rrf["cand_D"] > 0


def test_cross_encoder_rerank_candidates():
    """Tests reranking candidates using mocked CrossEncoder predictions."""
    reranker = CrossEncoderReranker(enabled=True)

    # Mock internal model predict
    mock_model = MagicMock()
    # Candidate 2 receives higher cross-attention logit (e.g. 2.5 vs 0.5)
    mock_model.predict.return_value = [0.5, 2.5]
    reranker._model = mock_model

    candidates = [
        {
            "name": "Candidate One",
            "score": 80.0,
            "max_score": 80.0,
            "match_score": 80.0,
            "experience_years": 4,
            "skills": ["Python"],
            "chunks": [{"content": "Basic Python programming."}],
        },
        {
            "name": "Candidate Two",
            "score": 70.0,
            "max_score": 70.0,
            "match_score": 70.0,
            "experience_years": 6,
            "skills": ["Python", "FastAPI", "Docker"],
            "chunks": [{"content": "Extensive microservice architecture in Python and FastAPI."}],
        },
    ]

    reranked = reranker.rerank_candidates(
        query="Senior Python and FastAPI developer",
        candidates=candidates,
        top_k=2,
        blend_weight=0.6,
    )

    assert len(reranked) == 2
    # Candidate Two should be boosted to rank 1 due to high cross-encoder logit (2.5)
    assert reranked[0]["name"] == "Candidate Two"
    assert reranked[0]["reranked"] is True
    assert "cross_encoder_score" in reranked[0]
    assert reranked[0]["match_score"] > reranked[1]["match_score"]


def test_cross_encoder_disabled_fallback():
    """Verifies that if reranker is disabled, candidate order and scores are preserved."""
    reranker = CrossEncoderReranker(enabled=False)
    candidates = [
        {"name": "Alice", "score": 90.0, "match_score": 90.0},
        {"name": "Bob", "score": 80.0, "match_score": 80.0},
    ]

    res = reranker.rerank_candidates(query="Developer", candidates=candidates, top_k=2)
    assert res[0]["name"] == "Alice"
    assert res[1]["name"] == "Bob"


@patch("agentic_profile_matching.job_matcher.SentenceTransformer")
@patch("agentic_profile_matching.job_matcher.ChromaVectorStore")
def test_job_matcher_with_two_stage_reranker(mock_chroma, mock_transformer):
    """Verifies that JobMatcher successfully calls CrossEncoder reranking when use_reranker=True."""
    mock_store = MagicMock()
    mock_chroma.return_value = mock_store

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.1, 0.2]
    mock_transformer.return_value = mock_embedder

    mock_store.get_all.return_value = {
        "documents": ["Python dev with AWS", "Java dev with Spring"],
        "metadatas": [
            {
                "candidate_name": "Dev Alice",
                "resume_path": "alice.pdf",
                "experience_years": 5,
                "skills": "Python, AWS",
            },
            {
                "candidate_name": "Dev Bob",
                "resume_path": "bob.pdf",
                "experience_years": 4,
                "skills": "Java, Spring",
            },
        ],
        "ids": ["id1", "id2"],
    }
    mock_store.query.return_value = {
        "ids": [["id1", "id2"]],
        "distances": [[0.2, 0.5]],
    }

    mock_reranker = MagicMock()
    # Mock reranker returning Alice as rank 1
    mock_reranker.rerank_candidates.side_effect = lambda query, candidates, top_k: candidates[:top_k]

    matcher = JobMatcher(store=mock_store, reranker=mock_reranker, use_reranker=True)
    res = matcher.match(job_description="Python AWS Engineer", k=2)

    assert len(res["top_matches"]) == 2
    mock_reranker.rerank_candidates.assert_called_once()

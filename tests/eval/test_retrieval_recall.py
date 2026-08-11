import json
from pathlib import Path
import pytest

from agentic_profile_matching.job_matcher import JobMatcher
from agentic_profile_matching.stores import ChromaVectorStore


@pytest.fixture
def eval_scenarios():
    data_path = Path(__file__).parent.parent.parent / "data" / "eval_scenarios.json"
    if not data_path.exists():
        pytest.skip(f"Evaluation scenarios dataset not found at {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.eval
def test_retrieval_recall_and_mrr(eval_scenarios):
    """
    Evaluates hybrid retrieval recall@k and Mean Reciprocal Rank (MRR)
    against ground-truth evaluation scenarios.
    """
    store = ChromaVectorStore(collection_name="resumes")
    if store.count() == 0:
        pytest.skip("ChromaDB vector store is empty. Ingest resumes before running eval suite.")

    matcher = JobMatcher(store=store)
    recalls = []
    reciprocal_ranks = []

    for scenario in eval_scenarios:
        jd_text = scenario["job_description"]
        must_haves = scenario.get("must_have_skills", [])
        min_exp = scenario.get("min_experience_years", 0)
        expected_candidates = scenario.get("expected_top_candidates", [])

        # Run hybrid retrieval matching
        results = matcher.match(
            job_description=jd_text,
            k=10,
            min_exp=min_exp,
            must_have_skills=must_haves,
        )
        matched_candidates = [c["candidate_name"] for c in results.get("top_matches", [])]

        if not expected_candidates:
            continue

        # Calculate Recall@10
        hits = sum(
            1
            for expected in expected_candidates
            if any(expected.lower() in candidate.lower() for candidate in matched_candidates)
        )
        recall = hits / len(expected_candidates)
        recalls.append(recall)

        # Calculate Reciprocal Rank (MRR)
        rank = 0
        for idx, candidate in enumerate(matched_candidates, start=1):
            if any(expected.lower() in candidate.lower() for expected in expected_candidates):
                rank = idx
                break

        rr = 1.0 / rank if rank > 0 else 0.0
        reciprocal_ranks.append(rr)

        assert recall >= scenario.get("expected_min_recall_at_5", 0.0), (
            f"Scenario '{scenario['scenario_id']}' recall {recall:.2f} below target threshold {scenario.get('expected_min_recall_at_5')}."
        )

    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    mean_rr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    print(f"\n[RAG Eval Metrics] Mean Recall@10: {mean_recall:.2f} | MRR: {mean_rr:.2f}")
    assert mean_recall > 0.0

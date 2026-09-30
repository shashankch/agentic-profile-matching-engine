"""
Two-Stage Hybrid Retrieval & Cross-Encoder Reranker for Yojaka AI (Deliverable 17.3).
Combines Stage 1 high-recall Reciprocal Rank Fusion (RRF) and hybrid scoring
with Stage 2 high-precision Cross-Encoder joint cross-attention reranking.
Significantly enhances NDCG@5 and eliminates keyword-stuffing false positives.
"""

import math
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("reranker")

_RERANKER_CACHE: Dict[str, Any] = {}


def compute_rrf_scores(
    dense_ranked_ids: List[str],
    sparse_ranked_ids: List[str],
    k_rrf: int = 60,
) -> Dict[str, float]:
    """
    Computes Reciprocal Rank Fusion (RRF) scores across dense vector and sparse BM25 rankings.
    Formula: RRF(d) = sum(1 / (k_rrf + rank_m(d)))
    """
    rrf_map: Dict[str, float] = {}

    for rank, doc_id in enumerate(dense_ranked_ids):
        rrf_map[doc_id] = rrf_map.get(doc_id, 0.0) + (1.0 / (k_rrf + rank + 1))

    for rank, doc_id in enumerate(sparse_ranked_ids):
        rrf_map[doc_id] = rrf_map.get(doc_id, 0.0) + (1.0 / (k_rrf + rank + 1))

    return rrf_map


class CrossEncoderReranker:
    """
    Precision Cross-Encoder reranker using transformer joint cross-attention.
    Evaluates (query, document) pairs simultaneously to capture deep semantic relevance.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: Optional[str] = None,
        enabled: bool = True,
    ):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.enabled = enabled
        self._model = None

    @property
    def model(self):
        """Lazily load and cache the CrossEncoder model singleton."""
        if not self.enabled:
            return None

        if self._model is not None:
            return self._model

        if self.model_name in _RERANKER_CACHE:
            self._model = _RERANKER_CACHE[self.model_name]
            return self._model

        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            logger.info(f"Loading CrossEncoder model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            _RERANKER_CACHE[self.model_name] = self._model
            return self._model
        except Exception as e:
            logger.warning(f"Failed to load CrossEncoder '{self.model_name}' ({e}). Falling back to Stage 1 scoring.")
            self.enabled = False
            return None

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Applies stable sigmoid to convert raw logits to probabilities [0.0, 1.0]."""
        if x > 15:
            return 1.0
        if x < -15:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    def predict_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Runs cross-attention prediction over pairs of (query, candidate_document).
        Returns normalized scores in range [0.0, 1.0].
        """
        if not pairs or not self.enabled:
            return [0.5] * len(pairs)

        model = self.model
        if model is None:
            return [0.5] * len(pairs)

        try:
            inputs = [[q, doc] for q, doc in pairs]
            raw_logits = model.predict(inputs, batch_size=32)

            scores = []
            for val in raw_logits:
                # Some models output raw logits, others already calibrated probabilities
                logit = float(val)
                scores.append(self._sigmoid(logit))
            return scores
        except Exception as e:
            logger.warning(f"CrossEncoder prediction failed: {e}. Returning uniform defaults.")
            return [0.5] * len(pairs)

    def rerank_candidates(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        blend_weight: float = 0.50,
    ) -> List[Dict[str, Any]]:
        """
        Reranks Stage 1 candidate matches using Stage 2 Cross-Encoder joint attention.

        Args:
            query: Job description or recruiter query string.
            candidates: List of candidate profile dictionaries produced by Stage 1.
            top_k: Number of top candidates to return.
            blend_weight: Proportion of final score determined by CrossEncoder (0.0 to 1.0).

        Returns:
            Reranked list of candidate dictionaries with calibrated scores.
        """
        if not candidates:
            return []

        # If reranker disabled or failed to load, preserve Stage 1 order
        if not self.enabled or self.model is None:
            return candidates[:top_k]

        pairs = []
        for cand in candidates:
            # Build representative candidate text combining title, skills, and top chunks
            name = cand.get("name") or cand.get("candidate_name") or "Candidate"
            exp = cand.get("experience_years", 0)
            skills = cand.get("skills", [])
            skills_str = ", ".join(skills) if isinstance(skills, list) else str(skills)

            chunk_texts = [ch.get("content", "") for ch in cand.get("chunks", [])[:3] if ch.get("content")]
            exp_text = "\n".join(chunk_texts)

            cand_summary = (
                f"Candidate: {name} | Experience: {exp} years\nSkills: {skills_str}\nExperience Details:\n{exp_text}"
            )
            pairs.append((query, cand_summary))

        raw_scores = self.predict_pairs(pairs)

        reranked = []
        for cand, ce_score in zip(candidates, raw_scores):
            cand_copy = dict(cand)
            stage1_score = float(cand_copy.get("score", cand_copy.get("max_score", 50.0)))
            ce_score_100 = ce_score * 100.0

            # Blend Stage 1 multi-factor score (BM25 + vector + hard filters) with Stage 2 CrossEncoder
            final_blended = ((1.0 - blend_weight) * stage1_score) + (blend_weight * ce_score_100)
            final_score = round(max(0.0, min(100.0, final_blended)), 2)

            cand_copy["stage1_score"] = stage1_score
            cand_copy["cross_encoder_score"] = round(ce_score_100, 2)
            cand_copy["score"] = final_score
            cand_copy["max_score"] = final_score
            cand_copy["match_score"] = final_score
            cand_copy["reranked"] = True
            reranked.append(cand_copy)

        # Sort descending by blended match_score
        reranked.sort(key=lambda x: x["match_score"], reverse=True)
        return reranked[:top_k]

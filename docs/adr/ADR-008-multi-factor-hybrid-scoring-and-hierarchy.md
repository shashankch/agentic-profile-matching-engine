# ADR-008: Multi-Factor Hybrid Candidate Scoring & Grounded LLM Recommendation Hierarchy

## Status
Accepted (Implemented in `v1.0.0`)

## Context
Relying solely on raw vector similarity distance or rigid hardcoded score cutoffs (`if score >= 65`) can compress candidate scores and produce fragile recommendation overrides. Furthermore, passing incomplete candidate summary metrics to LLM prompts risks LLM hallucinations regarding candidate qualifications (e.g. incorrectly asserting a candidate lacks experience when their experience exceeds requirements).

## Decision
1. **Dynamic Multi-Factor Scoring**: Calculate candidate match scores ($0 - 100$) in `job_matcher.py` using min-max normalized vector similarity, stop-word filtered BM25 keyword matching, must-have skill match ratios, and experience satisfaction ratios.
2. **Grounded LLM Recommendation Hierarchy**: Allow the LLM in `deep_screen_node` to assign qualitative hiring statuses (`"Strong Hire"`, `"Borderline Hire"`, `"Rejected / No-Hire"`) based on full resume text audits. In `recommendation_node`, preserve the LLM's deep screening status while enforcing mandatory safety guardrails (e.g., overriding to `Rejected` if mandatory skills or minimum experience are missing).
3. **Fact-Grounded Prompting**: Pass explicit candidate metadata (`Experience Years`, `Matched Skills`, `Missing Skills`) into all report explanation prompts and enforce strict system instructions prohibiting ungrounded claims.

## Consequences
- **Positive**: Guarantees deterministic, transparent scoring across arbitrary queries; prevents score compression; eliminates arbitrary magic-number overrides while maintaining hard safety guardrails.
- **Negative**: Requires passing candidate skill and experience metrics across all graph state node transitions.

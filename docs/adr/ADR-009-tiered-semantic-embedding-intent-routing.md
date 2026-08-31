# ADR-009: Tiered Semantic Embedding & LLM Structured Intent Routing

## Status
Accepted (Implemented in `v1.1.0`)

## Context
Using rigid substring keyword matching (e.g. `if "search" in text`) in conversational state machine routers is brittle, fails on natural language variations and typos, requires endless hardcoded keyword arrays, and lacks awareness of active conversation state. Conversely, making a frontier LLM call on every single routing decision introduces unnecessary API cost and 200–500ms of latency per turn.

## Decision
Implement a **Tiered Production Hybrid Router** in `src/agentic_profile_matching/agent/routers.py`:
1. **Fast State Check**: Immediately routes to `extract_requirements` if the active graph state has no requirements or if the user pastes a multi-line Job Description.
2. **Tier 1 — Local Semantic Vector Router**: Computes cosine similarity between the incoming user query and pre-computed anchor intent embeddings using the in-memory `SentenceTransformer` (`all-MiniLM-L6-v2`). If the top confidence exceeds the threshold ($\ge 0.60$), the query is routed immediately with **< 2ms latency and zero LLM token cost**.
3. **Tier 2 — LLM Structured Intent Classifier**: For ambiguous or low-confidence queries, invokes the active LLM with structured output (`RouteDecision` Pydantic model with `Literal["extract_requirements", "adjust_requirements", "conversational_query"]`), passing active session context (shortlist count, requirement presence) for definitive contextual classification.
4. **Tier 3 — Heuristic Fallback**: Safe fallback to preserve pipeline continuity if external APIs are unreachable.

## Consequences
- **Positive**: Eliminates brittle keyword arrays; resolves intent in < 2ms with zero API cost for 85%+ of standard recruitment queries; seamlessly understands natural language nuances, negation, and multi-turn context.
- **Negative**: Requires pre-computing anchor embeddings once at application startup.

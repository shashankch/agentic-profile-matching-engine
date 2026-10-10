# 🔄 LangGraph Agent State Machine Specification

This document details the state machine topology, node transitions, checkpointing boundaries, and state schema for the **Yojaka AI** agent workflow.

> 🌐 **Interactive Documentation**: Available on the [**Yojaka AI Documentation Site**](https://shashankch.github.io/yojaka-ai-profile-matching-engine/state_machine/).

---

## 🗺️ State Machine Topology

The agent workflow is structured as a 9-node `StateGraph` compiled with in-memory `MemorySaver` checkpointing keyed by `thread_id`:

![LangGraph Agentic State Machine](assets/diagrams/state_machine.png)

---

## 📋 State Graph Nodes & Transitions

### 1. Ingress & Calibrated Margin Routing (ADR-009, ADR-018)
- `__start__` → `parse_input`: Ingress message normalization, timestamp initialization, and clean session field setup.
- `parse_input` → Native LangGraph 1.x `RoutingCommand(goto=..., update={...})` with compile-time destination routing:
    - **Structural Fast Path (0ms)**: Multi-line pasted JDs or empty initial state dispatch directly to `extract_requirements`.
    - **In-Memory Query Cache (0ms)**: Instant replay for identical user messages within the current session.
    - **Calibrated Margin Gating (<2ms, $0 token cost)**: Evaluates $\Delta = \text{Top1} - \text{Top2}$. If $\Delta \ge 0.12$ and $\text{Top1} \ge 0.55$, dispatches directly to destination intent without LLM latency.
    - **Structured Ambiguity Escalation**: When $\Delta < 0.12$, escalates to a lightweight structured LLM (`IntentResolution`) with candidate intent choices to guarantee 0% misrouting.
    - Destination Targets:
        - `extract_requirements`: Dispatched when a new job description is pasted or requirements are absent.
        - `adjust_requirements`: Dispatched on slider adjustments, qualification edits, or recruiter refinement feedback.
        - `conversational_query`: Dispatched on candidate Q&A, comparison inquiries, or live external web research.

### 2. Retrieval, Screening & Synthesis Cascade
- `extract_requirements` / `adjust_requirements` → `search_resumes`: Queries `CompositeVectorStore` and cached in-memory `BM25Okapi` index using `JobMatcher` with two-stage cross-encoder reranking.
- `search_resumes` → `rank_candidates`: Applies multi-factor hybrid scoring (dense cosine 50% + BM25 35% + qualification satisfaction 15%) and slices the coarse Top 10 shortlist.
- `rank_candidates` → `deep_screen`: Executes parallel dual-rubric structured evaluations (Technical Architecture 60% + Domain Fit 40%) bounded by `Semaphore(max_concurrent=2)` over the top 5 candidates.
- `deep_screen` → `recommendation`: Applies deterministic safety guardrails (missing must-have skill / experience deficit overrides) and generates 3–5 tailored technical interview questions.
- `recommendation` → `generate_report`: Compiles side-by-side comparison matrix and candidate evaluation summaries.
- `generate_report` → `__end__`: Persists final report in checkpoint state and returns to recruiter UI.
- `conversational_query` → `__end__`: Returns conversational answer or web search synthesis without modifying active candidate shortlist.

---

## 🗃️ State Schema (`AgentState`)

```python
from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage

class JobRequirements(TypedDict, total=False):
    title: str
    must_have_skills: List[str]
    nice_to_have_skills: List[str]
    min_experience_years: int
    education_level: str
    other_constraints: List[str]
    skill_expansions: dict[str, List[str]]

class CandidateMatch(TypedDict, total=False):
    candidate_id: str
    name: str
    score: int
    matched_skills: List[str]
    missing_skills: List[str]
    experience_years: int
    education: str
    relevance_excerpts: List[str]
    strengths: List[str]
    gaps: List[str]
    improvement_suggestions: str
    screening_status: str
    screening_reasoning: str
    interview_questions: List[str]
    # Phase 18: Parallel Dual-Rubric Structured Evaluation fields
    technical_score: Optional[float]
    domain_fit_score: Optional[float]
    technical_strengths: Optional[List[str]]
    technical_gaps: Optional[List[str]]
    sourcing_strengths: Optional[List[str]]
    sourcing_gaps: Optional[List[str]]
    architecture_notes: Optional[str]
    trajectory_notes: Optional[str]

class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    requirements: JobRequirements
    shortlist: List[CandidateMatch]
    previous_shortlist: List[CandidateMatch]
    ranking_explanation: str
    coarse_screen_limit: Optional[int]
    deep_screen_limit: Optional[int]
    recommendation_limit: Optional[int]
    current_round: int
    final_report: str
    feedback_pending: bool
    user_feedback: str
    errors: List[str]
    # Phase 18: Calibrated Margin Routing Telemetry
    routing_decision: Optional[Dict[str, Any]]
```

---

## 🛡️ Invariant Guarantees (ADR-011, ADR-012, ADR-018)

1. **Stateless Credential Isolation (CWE-312)**: `AgentState` strictly isolates domain metadata. API keys and secrets are never serialized into checkpoints.
2. **Functional Copy-on-Write Immutability**: All node transitions construct fresh dictionary instances (`{**c, ...}`) preventing mutation side-effects and ensuring checkpoint replay safety.
3. **Deterministic Score Invariance**: Dual-rubric evaluation calculates composite scores mathematically ($\text{Score} = 0.60 \times \text{Tech} + 0.40 \times \text{Domain}$) without hallucinated score drifts across repeated evaluations.

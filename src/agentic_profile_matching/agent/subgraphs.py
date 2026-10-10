"""
Modular Typed Subgraphs for Yojaka AI (Phase 18).

Decomposes the monolithic AgentState pipeline into isolated, typed subgraphs:
1. JDAnalyzerSubgraph: JD parsing, skill expansion & requirements extraction.
2. TalentRetrievalSubgraph: Stage 1 coarse hybrid RAG search & rank slicing.
3. DeepScreeningSubgraph: Stage 2 parallel dual-rubric candidate deep audits.
4. SynthesisSubgraph: Stage 3 recommendation hierarchy, question generation & report compilation.
"""

from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END

from agentic_profile_matching.agent.state import JobRequirements, CandidateMatch
from agentic_profile_matching.agent.nodes import (
    extract_requirements_node,
    search_resumes_node,
    rank_candidates_node,
    deep_screen_node,
    recommendation_node,
    generate_report_node,
)


# ---------------------------------------------------------------------------
# 1. JD Analyzer Subgraph
# ---------------------------------------------------------------------------
class JDAnalyzerState(TypedDict, total=False):
    messages: List[BaseMessage]
    requirements: JobRequirements
    current_round: int
    errors: List[str]


def build_jd_analyzer_subgraph():
    """Builds and compiles the JD Analysis and Requirement Extraction Subgraph."""
    builder = StateGraph(JDAnalyzerState)
    builder.add_node("extract_requirements", extract_requirements_node)
    builder.add_edge(START, "extract_requirements")
    builder.add_edge("extract_requirements", END)
    return builder.compile()


# ---------------------------------------------------------------------------
# 2. Talent Retrieval Subgraph
# ---------------------------------------------------------------------------
class TalentRetrievalState(TypedDict, total=False):
    requirements: JobRequirements
    shortlist: List[CandidateMatch]
    coarse_screen_limit: Optional[int]
    current_round: int
    errors: List[str]


def build_talent_retrieval_subgraph():
    """Builds and compiles the Hybrid Talent Retrieval & Ranking Subgraph."""
    builder = StateGraph(TalentRetrievalState)
    builder.add_node("search_resumes", search_resumes_node)
    builder.add_node("rank_candidates", rank_candidates_node)
    builder.add_edge(START, "search_resumes")
    builder.add_edge("search_resumes", "rank_candidates")
    builder.add_edge("rank_candidates", END)
    return builder.compile()


# ---------------------------------------------------------------------------
# 3. Deep Screening Subgraph
# ---------------------------------------------------------------------------
class DeepScreeningState(TypedDict, total=False):
    requirements: JobRequirements
    shortlist: List[CandidateMatch]
    deep_screen_limit: Optional[int]
    current_round: int
    errors: List[str]


def build_deep_screening_subgraph():
    """Builds and compiles the Parallel Dual-Rubric Deep Screening Subgraph."""
    builder = StateGraph(DeepScreeningState)
    builder.add_node("deep_screen", deep_screen_node)
    builder.add_edge(START, "deep_screen")
    builder.add_edge("deep_screen", END)
    return builder.compile()


# ---------------------------------------------------------------------------
# 4. Synthesis Subgraph
# ---------------------------------------------------------------------------
class SynthesisState(TypedDict, total=False):
    requirements: JobRequirements
    shortlist: List[CandidateMatch]
    previous_shortlist: List[CandidateMatch]
    recommendation_limit: Optional[int]
    current_round: int
    final_report: str
    ranking_explanation: str
    errors: List[str]


def build_synthesis_subgraph():
    """Builds and compiles the Final Recommendation & Reporting Subgraph."""
    builder = StateGraph(SynthesisState)
    builder.add_node("recommendation", recommendation_node)
    builder.add_node("generate_report", generate_report_node)
    builder.add_edge(START, "recommendation")
    builder.add_edge("recommendation", "generate_report")
    builder.add_edge("generate_report", END)
    return builder.compile()


__all__ = [
    "JDAnalyzerState",
    "build_jd_analyzer_subgraph",
    "TalentRetrievalState",
    "build_talent_retrieval_subgraph",
    "DeepScreeningState",
    "build_deep_screening_subgraph",
    "SynthesisState",
    "build_synthesis_subgraph",
]

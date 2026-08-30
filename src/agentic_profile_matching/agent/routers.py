import numpy as np
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from agentic_profile_matching.agent.state import AgentState
from agentic_profile_matching.observability import get_logger

logger = get_logger("agentic_profile_matching.agent.routers")


class RouteDecision(BaseModel):
    intent: Literal["extract_requirements", "adjust_requirements", "conversational_query"] = Field(
        description="The target workflow branch for this user input."
    )
    reasoning: str = Field(default="", description="Short 1-sentence reasoning for the routing classification.")


# Anchor phrases for Tier 1 Local Semantic Routing
SEMANTIC_INTENT_ANCHORS: Dict[str, List[str]] = {
    "extract_requirements": [
        "Search for candidates with Java and Python experience",
        "Find senior software engineers with 5+ years experience",
        "Here is the job description for a backend engineer",
        "Looking for a full stack developer with React and Node.js",
        "Requirements: 3 years experience in machine learning and PyTorch",
        "Need a DevOps engineer who knows Kubernetes and Terraform",
        "Hiring for data science lead",
        "Find developers with Golang and AWS experience",
        "Search resumes for cloud architects",
        "Source candidates matching these requirements",
    ],
    "adjust_requirements": [
        "Make Python a mandatory must-have skill",
        "Increase required experience to 7 years",
        "Exclude candidates without Docker experience",
        "Focus more on AWS and Kubernetes knowledge",
        "Remove Java from the must-have requirements",
        "Change the job title to Senior Backend Engineer",
        "Add GraphQL as a nice-to-have skill",
        "Filter only candidates with a Master's degree",
        "Tighten the experience requirement to 10 years",
        "Relax the skills constraint to include Golang",
    ],
    "conversational_query": [
        "Why is candidate Alice ranked higher than Bob?",
        "Compare the top three candidates side by side",
        "Who is the strongest candidate for system design?",
        "Explain the key differences between candidate 1 and candidate 2",
        "Tell me about Shashank's background and project experience",
        "Search the web for candidate's GitHub portfolio",
        "Search Google for candidate public notes",
        "What are the main gaps in the top ranked profile?",
        "Break down candidate strengths and weakness analysis",
        "Give me a detailed comparison of candidate 1 versus candidate 2",
    ],
}

# Cached embedder and pre-computed anchor embeddings
_EMBEDDER = None
_ANCHOR_EMBEDDINGS: Optional[Dict[str, np.ndarray]] = None


def _get_embedder():
    global _EMBEDDER, _ANCHOR_EMBEDDINGS
    if _EMBEDDER is None:
        try:
            from sentence_transformers import SentenceTransformer
            from agentic_profile_matching import config as app_config

            _EMBEDDER = SentenceTransformer(app_config.EMBEDDING_MODEL)
            # Pre-compute anchor embeddings for fast dot-product inference
            _ANCHOR_EMBEDDINGS = {}
            for intent, phrases in SEMANTIC_INTENT_ANCHORS.items():
                embeddings = _EMBEDDER.encode(phrases, normalize_embeddings=True)
                _ANCHOR_EMBEDDINGS[intent] = np.array(embeddings)
        except Exception as e:
            logger.warning(f"Could not initialize SentenceTransformer for semantic routing: {e}")
            _EMBEDDER = None
            _ANCHOR_EMBEDDINGS = None
    return _EMBEDDER, _ANCHOR_EMBEDDINGS


def _classify_via_semantic_similarity(query: str, confidence_threshold: float = 0.60) -> Optional[str]:
    """
    Tier 1: Computes cosine similarity against intent anchor clusters.
    Returns the predicted intent if max similarity exceeds confidence_threshold, else None.
    """
    embedder, anchor_dict = _get_embedder()
    if embedder is None or not anchor_dict:
        return None

    try:
        query_emb = embedder.encode([query], normalize_embeddings=True)[0]
        best_intent = None
        best_score = -1.0

        for intent, anchor_embs in anchor_dict.items():
            sims = np.dot(anchor_embs, query_emb)
            max_sim = float(np.max(sims))
            if max_sim > best_score:
                best_score = max_sim
                best_intent = intent

        logger.debug(f"Semantic routing evaluated top intent: {best_intent} (confidence: {best_score:.3f})")
        if best_score >= confidence_threshold:
            return best_intent
    except Exception as e:
        logger.warning(f"Error during semantic vector classification: {e}")

    return None


ROUTER_SYSTEM_PROMPT = """You are an expert intent classifier for an AI Recruiter workflow engine.
Classify the user's latest message into one of three routing branches:

1. 'extract_requirements':
   - Used when the user provides a new Job Description, or asks to search/find/source candidates from scratch.
2. 'adjust_requirements':
   - Used when the user modifies active constraints (adding must-have skills, changing experience years, relaxing filters) for the existing search session.
3. 'conversational_query':
   - Used when the user asks a question about active shortlisted candidates (comparing candidates, asking why one ranked higher, asking about gaps/strengths), or asks for external web/notes lookup.

Current Session Context:
- Has Active Requirements: {has_requirements}
- Active Shortlist Count: {shortlist_count}"""


def _classify_via_llm(state: AgentState) -> Optional[str]:
    """
    Tier 2: LLM Structured Intent Classifier for complex or ambiguous multi-step conversations.
    """
    try:
        from agentic_profile_matching import config as app_config
        from agentic_profile_matching.tools import invoke_structured

        llm_provider = state.get("llm_provider") or app_config.DEFAULT_PROVIDER
        llm_model = state.get("llm_model") or app_config.DEFAULT_MODEL
        api_key = state.get("api_key") or ""
        api_url = state.get("api_url")

        if not api_key:
            return None

        llm = app_config.get_llm_model(
            provider=llm_provider,
            model_name=llm_model,
            api_key=api_key,
            api_url=api_url,
        )

        has_requirements = bool(state.get("requirements"))
        shortlist_count = len(state.get("shortlist", []))
        messages = state.get("messages", [])
        last_msg = messages[-1].content if messages else ""

        system_content = ROUTER_SYSTEM_PROMPT.format(
            has_requirements=has_requirements,
            shortlist_count=shortlist_count,
        )

        prompt_messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=f"User Message to route: {last_msg}"),
        ]

        result = invoke_structured(llm, prompt_messages, RouteDecision)
        intent = result.get("intent")
        if intent in ["extract_requirements", "adjust_requirements", "conversational_query"]:
            logger.info(f"Tier 2 LLM routing decision: {intent} (Reason: {result.get('reasoning', 'N/A')})")
            return intent
    except Exception as e:
        logger.warning(f"Tier 2 LLM routing fallback error: {e}")

    return None


def route_input(state: AgentState) -> str:
    """
    Tiered Production Hybrid Router:
    1. Fast state check (empty messages or empty requirements -> extract_requirements).
    2. Multi-line raw JD detection.
    3. Tier 1: Local Semantic Embedding Router (< 2ms, zero token cost).
    4. Tier 2: LLM Structured Intent Classifier (with_structured_output fallback).
    5. Heuristic fallback.
    """
    messages = state.get("messages", [])
    if not messages:
        return "extract_requirements"

    # If no requirements exist yet in state, we must extract them first
    if not state.get("requirements"):
        return "extract_requirements"

    last_msg = messages[-1].content.strip()
    lines = [line.strip() for line in last_msg.split("\n") if line.strip()]

    # Fast check: Multi-line paste or explicit JD keywords -> new requirements extraction
    if len(lines) > 3 or any(
        w in last_msg.lower() for w in ["job description", "requirements:", "duties:", "responsibilities:"]
    ):
        return "extract_requirements"

    # Tier 1: Local Semantic Similarity Classification (< 2ms, 0 API cost)
    semantic_intent = _classify_via_semantic_similarity(last_msg, confidence_threshold=0.60)
    if semantic_intent:
        # Context guardrail: only allow conversational_query if shortlist exists or web query
        if semantic_intent == "conversational_query" and len(state.get("shortlist", [])) == 0:
            if not any(w in last_msg.lower() for w in ["web", "internet", "google", "search online", "notes"]):
                return "extract_requirements"
        logger.info(f"Tier 1 Semantic Router resolved intent: {semantic_intent}")
        return semantic_intent

    # Tier 2: LLM Structured Intent Classification (State-aware fallback)
    llm_intent = _classify_via_llm(state)
    if llm_intent:
        return llm_intent

    # Tier 3: Heuristic Fallback
    lower_msg = last_msg.lower()
    if len(state.get("shortlist", [])) > 0 and any(
        kw in lower_msg
        for kw in [
            "why",
            "compare",
            "higher",
            "better",
            "explain",
            "vs",
            "versus",
            "who",
            "tell me about",
        ]
    ):
        return "conversational_query"

    return "adjust_requirements"

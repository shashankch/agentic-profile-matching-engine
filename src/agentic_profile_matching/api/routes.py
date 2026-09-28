"""
FastAPI Route Handlers for Yojaka AI Headless Gateway.
Exposes endpoints for job parsing, candidate matching, and Server-Sent Event (SSE) streaming.
"""

from collections.abc import AsyncGenerator
import json
import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage

from agentic_profile_matching import config
from agentic_profile_matching.api.schemas import (
    HealthResponse,
    JobExtractRequest,
    JobExtractResponse,
    CandidateMatchRequest,
    CandidateMatchResponse,
    CandidateProfileDTO,
    WorkflowStreamRequest,
)
from agentic_profile_matching.job_matcher import JobMatcher
from agentic_profile_matching.matching_agent import matching_agent_workflow
from agentic_profile_matching.stores import ChromaVectorStore
from agentic_profile_matching.tools import extract_requirements

router = APIRouter()
START_TIME = time.time()


@router.get("/health", response_model=HealthResponse, tags=["System"])
@router.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def get_health() -> HealthResponse:
    """Health check endpoint returning system status and uptime."""
    return HealthResponse(
        status="healthy",
        version="1.3.0",
        service="yojaka-ai-gateway",
        uptime_seconds=round(time.time() - START_TIME, 2),
    )


@router.post("/api/v1/jobs/extract", response_model=JobExtractResponse, tags=["Jobs"])
async def extract_job_requirements_endpoint(request: JobExtractRequest) -> JobExtractResponse:
    """
    Extracts structured requirements (must-have skills, experience, title)
    from a raw recruiter job description using LLM parsing.
    """
    try:
        reqs = extract_requirements(request.job_description)
        return JobExtractResponse(success=True, requirements=reqs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract requirements: {str(e)}") from e


@router.post("/api/v1/candidates/match", response_model=CandidateMatchResponse, tags=["Matching"])
async def match_candidates_endpoint(request: CandidateMatchRequest) -> CandidateMatchResponse:
    """
    Matches and ranks candidates against the provided job description using
    hybrid BM25 + dense vector scoring and skill constraints.
    """
    try:
        # 1. Parse or build requirements
        reqs = extract_requirements(request.job_description)
        if request.must_have_skills:
            reqs["must_have_skills"] = request.must_have_skills
        if request.nice_to_have_skills:
            reqs["nice_to_have_skills"] = request.nice_to_have_skills
        if request.min_experience_years > 0:
            reqs["min_experience_years"] = request.min_experience_years

        # 2. Match candidates using active vector store
        store = ChromaVectorStore()
        matcher = JobMatcher(store=store)
        results = matcher.match(
            job_description=request.job_description,
            requirements=reqs,
            top_k=request.top_k,
        )

        candidates_dto = []
        for r in results:
            candidates_dto.append(
                CandidateProfileDTO(
                    candidate_id=r.get("candidate_id", ""),
                    name=r.get("name", "Unknown"),
                    score=int(r.get("score", 0)),
                    matched_skills=r.get("matched_skills", []),
                    missing_skills=r.get("missing_skills", []),
                    experience_years=int(r.get("experience_years", 0)),
                    education=r.get("education", "Not Specified"),
                    screening_status=r.get("screening_status", "Shortlisted"),
                    strengths=r.get("strengths", []),
                    gaps=r.get("gaps", []),
                    improvement_suggestions=r.get("improvement_suggestions", ""),
                    interview_questions=r.get("interview_questions", []),
                )
            )

        explanation = (
            f"Retrieved and ranked {len(candidates_dto)} candidate profiles via hybrid BM25 and dense vector scoring."
        )

        return CandidateMatchResponse(
            success=True,
            total_matched=len(candidates_dto),
            candidates=candidates_dto,
            ranking_explanation=explanation,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Candidate matching failed: {str(e)}") from e


@router.post("/api/v1/workflow/stream", tags=["Streaming"])
async def stream_workflow_post(request: WorkflowStreamRequest) -> StreamingResponse:
    """
    Executes the agentic workflow and streams progress events and response tokens
    via Server-Sent Events (SSE: text/event-stream).
    """
    return _build_sse_response(request.query, request.thread_id, request.requirements)


@router.get("/api/v1/workflow/stream/{task_id}", tags=["Streaming"])
async def stream_workflow_get(task_id: str, query: str = "Search top candidates") -> StreamingResponse:
    """
    GET endpoint for Server-Sent Events (SSE) streaming matching task_id sessions.
    """
    return _build_sse_response(query, task_id, None)


def _build_sse_response(query: str, thread_id: str, requirements: dict | None) -> StreamingResponse:
    """Constructs a Server-Sent Events StreamingResponse."""

    async def sse_event_generator() -> AsyncGenerator[str, None]:
        # Emit session start event
        yield (f"event: session_start\ndata: {json.dumps({'thread_id': thread_id, 'timestamp': time.time()})}\n\n")

        state_input = {
            "messages": [HumanMessage(content=query)],
            "requirements": requirements
            or {
                "title": "Software Engineer",
                "must_have_skills": [],
                "nice_to_have_skills": [],
                "min_experience_years": 0,
                "education_level": "Not Specified",
                "other_constraints": [],
            },
            "shortlist": [],
            "coarse_screen_limit": config.DEFAULT_COARSE_LIMIT,
            "deep_screen_limit": config.DEFAULT_DEEP_LIMIT,
            "recommendation_limit": config.DEFAULT_RECOMMENDATION_LIMIT,
            "current_round": 1,
            "final_report": "",
            "feedback_pending": False,
            "user_feedback": "",
            "errors": [],
        }

        workflow_config = {
            "configurable": {
                "thread_id": thread_id,
                "store": ChromaVectorStore(),
            }
        }

        try:
            for event in matching_agent_workflow.stream(state_input, config=workflow_config, stream_mode="updates"):
                for node_name, node_update in event.items():
                    if isinstance(node_update, dict):
                        # Extract lightweight summary
                        summary = {
                            "node": node_name,
                            "timestamp": time.time(),
                        }
                        if node_name == "extract_requirements":
                            summary["title"] = node_update.get("requirements", {}).get("title")
                        elif node_name == "rank_candidates":
                            summary["shortlist_count"] = len(node_update.get("shortlist", []))
                        elif node_name == "conversational_query":
                            # Emit conversational tokens
                            for m in node_update.get("messages", []):
                                if isinstance(m, AIMessage) or (hasattr(m, "type") and m.type == "ai"):
                                    summary["response"] = m.content

                        yield f"event: node_update\ndata: {json.dumps(summary)}\n\n"

            yield f"event: done\ndata: {json.dumps({'status': 'completed', 'thread_id': thread_id})}\n\n"

        except Exception as ex:
            error_data = {"status": "error", "error": str(ex), "thread_id": thread_id}
            yield f"event: error\ndata: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

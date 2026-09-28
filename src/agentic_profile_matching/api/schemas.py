"""
Pydantic V2 Schemas for Yojaka AI Headless FastAPI Gateway.
Defines strict API request and response contracts for jobs, matching, and streaming.
"""

from typing import Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="Service health state")
    version: str = Field(default="1.3.0", description="Yojaka AI platform version")
    service: str = Field(default="yojaka-ai-gateway", description="Service identifier")
    uptime_seconds: float = Field(default=0.0, description="Uptime in seconds")


class JobExtractRequest(BaseModel):
    job_description: str = Field(..., min_length=10, description="Raw job description text to parse")


class JobExtractResponse(BaseModel):
    success: bool = Field(default=True, description="Indicates if extraction succeeded")
    requirements: dict[str, Any] = Field(..., description="Extracted structured job requirements")


class CandidateProfileDTO(BaseModel):
    candidate_id: str
    name: str
    score: int
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    experience_years: int = 0
    education: str = "Not Specified"
    screening_status: str = "Shortlisted"
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    improvement_suggestions: str = ""
    interview_questions: list[str] = Field(default_factory=list)


class CandidateMatchRequest(BaseModel):
    job_description: str = Field(..., min_length=10, description="Job description or search query")
    top_k: int = Field(default=10, ge=1, le=50, description="Maximum candidate matches to retrieve")
    min_experience_years: int = Field(default=0, ge=0, description="Minimum required experience in years")
    must_have_skills: list[str] = Field(default_factory=list, description="Explicit must-have skill requirements")
    nice_to_have_skills: list[str] = Field(default_factory=list, description="Preferred skills")


class CandidateMatchResponse(BaseModel):
    success: bool = Field(default=True)
    total_matched: int = Field(default=0)
    candidates: list[CandidateProfileDTO] = Field(default_factory=list)
    ranking_explanation: str = Field(default="")


class WorkflowStreamRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Recruiter conversational command or job description")
    thread_id: str = Field(default="headless-session", description="Session thread identifier")
    requirements: dict[str, Any] | None = Field(default=None, description="Optional pre-existing requirements dict")

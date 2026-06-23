import re
import time
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage


def execute_with_retry(func, *args, **kwargs):
    """
    Executes an LLM function call with exponential backoff on 429 (rate limit) errors.
    """
    max_retries = 5
    delay = 2.0
    for attempt in range(max_retries):
        try:
            # Proactively introduce a small sleep to throttle queries
            time.sleep(0.5)
            return func(*args, **kwargs)
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "rate_limit" in err_msg.lower() or "too many requests" in err_msg.lower():
                print(f"Rate limit hit (429). Retrying in {delay:.1f} seconds (Attempt {attempt+1}/{max_retries})...")
                time.sleep(delay)
                delay *= 2.0
            else:
                raise e
    return func(*args, **kwargs)


def extract_requirements(jd: str, llm) -> Dict[str, Any]:
    """
    Extracts structured job requirements from an unstructured JD string.
    """
    system_prompt = """You are a professional recruiting assistant. Analyze the provided Job Description (JD) and extract the job requirements.
You MUST return a valid JSON object ONLY. Do not include markdown code blocks, explanation text, or anything else. Just the raw JSON.

The JSON structure must match this exact format:
{
    "title": "Job Title (string)",
    "must_have_skills": ["List of critical required technical skills/tools (array of strings)"],
    "nice_to_have_skills": ["List of preferred technical skills/tools (array of strings)"],
    "min_experience_years": 5 (integer, minimum years of experience required. If not specified, default to 0),
    "education_level": "Education level required e.g., Bachelor, Master, PhD, B.Tech, MS (string, or 'Not Specified')",
    "other_constraints": ["Any other critical requirements like visa, locations, certifications (array of strings)"]
}"""
    
    def _call():
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Extract requirements for this Job Description:\n\n{jd}")
        ]
        response = llm.invoke(messages)
        content = response.content.strip()
        
        # Clean JSON markdown blocks
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n", "", content)
            content = re.sub(r"\n```$", "", content)
            
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx:end_idx+1]
            
        return json.loads(content)

    try:
        return execute_with_retry(_call)
    except Exception as e:
        print(f"Error in extract_requirements tool: {e}")
        # Safe fallback structure
        return {
            "title": "Software Engineer",
            "must_have_skills": [],
            "nice_to_have_skills": [],
            "min_experience_years": 0,
            "education_level": "Not Specified",
            "other_constraints": []
        }


def compare_candidates(candidate_ids: List[str], shortlist: List[Dict]) -> str:
    """
    Generates a side-by-side Markdown grid comparing selected candidates.
    """
    # Filter candidates from the shortlist by candidate_id (filepath) or name
    candidates = []
    for c_id in candidate_ids:
        # Match by filename stem, candidate_id, or full name
        matched = None
        for c in shortlist:
            if c['candidate_id'] == c_id or c['name'] == c_id or Path(c['candidate_id']).stem == c_id:
                matched = c
                break
        if matched and matched not in candidates:
            candidates.append(matched)
            
    # Fallback to compare the top candidates if none were found matching
    if not candidates and len(shortlist) > 0:
        candidates = shortlist[:3]

    if not candidates:
        return "No matching candidates found in the shortlist to compare."
    
    # Build Markdown table
    header_row = "| Match Category | " + " | ".join(f"**{c['name']}**" for c in candidates) + " |"
    separator_row = "|---| " + " | ".join("---" for _ in candidates) + " |"
    
    rows = [
        ("Match Score", lambda c: f"**{c.get('score', 0)}/100**"),
        ("Experience", lambda c: f"{c.get('experience_years', 0)} Years"),
        ("Education", lambda c: f"{c.get('education', 'Not Specified')}"),
        ("Matched Skills", lambda c: ", ".join(c.get('matched_skills', [])) if c.get('matched_skills') else "None"),
        ("Missing Core Skills", lambda c: ", ".join(c.get('missing_skills', [])) if c.get('missing_skills') else "None"),
        ("Decision Status", lambda c: (
            f":green[**Strong Hire**]" if "strong" in c.get('screening_status', '').lower()
            else f":orange[**Borderline Hire**]" if "borderline" in c.get('screening_status', '').lower()
            else f":red[**Rejected / No-Hire**]" if "reject" in c.get('screening_status', '').lower() or "no-hire" in c.get('screening_status', '').lower()
            else f"`{c.get('screening_status', 'Screened')}`"
        )),
        ("Core Strengths", lambda c: "; ".join(c.get('strengths', [])) if c.get('strengths') else "Not Specified"),
        ("Identified Gaps", lambda c: "; ".join(c.get('gaps', [])) if c.get('gaps') else "None")
    ]
    
    table_lines = [header_row, separator_row]
    for label, extractor_fn in rows:
        line = f"| **{label}** | " + " | ".join(extractor_fn(c) for c in candidates) + " |"
        table_lines.append(line)
        
    return "\n".join(table_lines)


def generate_interview_questions(
    candidate_name: str,
    skills: List[str],
    gaps: List[str],
    requirements: Dict[str, Any],
    llm
) -> List[str]:
    """
    Generates 3-5 technical questions tailored to probe the candidate's gaps.
    """
    system_prompt = """You are a technical interviewer. Design 3 to 5 customized screening/interview questions for a candidate.
Focus on probing the candidate's skills gaps, validating their experience, and assessing how they fit the job requirements.
You MUST return a JSON list of strings ONLY. Do not include markdown code blocks, explanation text, or anything else. Just the raw JSON array.

Example:
[
    "Question 1...",
    "Question 2..."
]"""
    
    prompt_content = f"""Candidate: {candidate_name}
Job Title: {requirements.get('title', 'Software Engineer')}
Must-Have Skills: {requirements.get('must_have_skills', [])}
Candidate Skills: {skills}
Identified Gaps: {gaps}

Generate 3-5 targeted interview questions."""

    def _call():
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt_content)
        ]
        response = llm.invoke(messages)
        content = response.content.strip()
        
        # Clean JSON markdown blocks
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n", "", content)
            content = re.sub(r"\n```$", "", content)
            
        start_idx = content.find('[')
        end_idx = content.rfind(']')
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx:end_idx+1]
            
        return json.loads(content)

    try:
        return execute_with_retry(_call)
    except Exception as e:
        print(f"Error in generate_interview_questions tool: {e}")
        # Safe fallback questions
        fallback = [
            f"Can you tell me about your experience working with {', '.join(gaps) if gaps else 'software engineering architectures'}?",
            "Can you walk me through a complex technical project you engineered and the design tradeoffs you made?",
            "What strategies do you use when adapting to new tech stacks or programming paradigms on the job?"
        ]
        return fallback

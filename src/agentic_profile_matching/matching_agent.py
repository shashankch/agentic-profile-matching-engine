import re
import time
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from agentic_profile_matching import config
from agentic_profile_matching.fs_tools import read_file
from agentic_profile_matching.job_matcher import JobMatcher
from agentic_profile_matching.tools import extract_requirements, compare_candidates, generate_interview_questions, execute_with_retry


# State design schemas matching architecture.md
class JobRequirements(Dict):
    title: str
    must_have_skills: List[str]
    nice_to_have_skills: List[str]
    min_experience_years: int
    education_level: str
    other_constraints: List[str]


class CandidateMatch(Dict):
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


class AgentState(Dict):
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
    llm_provider: str
    llm_model: str
    api_key: str
    api_url: Optional[str]


# ----------------------------------------------------
# Nodes Logic Specification
# ----------------------------------------------------

def parse_input_node(state: AgentState) -> Dict[str, Any]:
    """
    Inspects the latest user message to classify it as a raw Job Description or refinement query.
    """
    messages = state.get("messages", [])
    
    # Capture the previous shortlist before updating
    prev_shortlist = state.get("shortlist", [])
    
    if not messages:
        return {"current_round": 1, "previous_shortlist": prev_shortlist}

    last_msg = messages[-1].content
    # Simple, highly reliable heuristic: multi-line strings or presence of typical JD words
    lines = [line.strip() for line in last_msg.split("\n") if line.strip()]
    is_jd = len(lines) > 3 or any(w in last_msg.lower() for w in ["job description", "requirements:", "duties:", "responsibilities:"])
    
    if is_jd:
        print("Input classified as raw Job Description.")
        return {"current_round": 1, "previous_shortlist": prev_shortlist}
    else:
        print("Input classified as refinement query.")
        return {"current_round": 1, "previous_shortlist": prev_shortlist}


def extract_requirements_node(state: AgentState) -> Dict[str, Any]:
    """
    LLM extracts structured job requirements from raw input message content.
    """
    messages = state.get("messages", [])
    if not messages:
        return {}

    last_msg = messages[-1].content
    
    # Dynamic LLM builder
    llm = config.get_llm_model(
        provider=state.get("llm_provider", config.DEFAULT_PROVIDER),
        model_name=state.get("llm_model", config.DEFAULT_MODEL),
        api_key=state.get("api_key", ""),
        api_url=state.get("api_url")
    )
    
    print("Extracting job requirements from input...")
    requirements = extract_requirements(last_msg, llm)
    return {"requirements": requirements, "current_round": 1}


def search_resumes_node(state: AgentState) -> Dict[str, Any]:
    """
    Retrieves candidate resumes matching constraints using local hybrid search.
    """
    requirements = state.get("requirements", {})
    title = requirements.get("title", "Software Engineer")
    must_have = requirements.get("must_have_skills", [])
    min_exp = requirements.get("min_experience_years", 0)
    
    coarse_limit = state.get("coarse_screen_limit") or config.DEFAULT_COARSE_LIMIT
    retrieval_k = max(int(coarse_limit * 1.5), 15)
    
    # Query with requirements to find top candidates
    matcher = JobMatcher()
    
    query_text = f"Job Title: {title}. Must-Have Skills: {', '.join(must_have)}. Experience: {min_exp} years."
    print(f"Retrieving candidate resumes for requirements: {requirements}")
    
    # Try strict filtering first
    results = matcher.match(
        job_description=query_text,
        k=retrieval_k,
        min_exp=min_exp,
        must_have_skills=must_have,
        apply_filters=True
    )
    
    # Fallback to no strict filtering if zero matches exist
    if not results.get("top_matches"):
        print("Zero candidates satisfied strict criteria. Relaxing filters...")
        results = matcher.match(
            job_description=query_text,
            k=retrieval_k,
            min_exp=min_exp,
            must_have_skills=must_have,
            apply_filters=False
        )
        
    # Store top raw candidates
    raw_matches = results.get("top_matches", [])
    return {"shortlist": raw_matches}


def rank_candidates_node(state: AgentState) -> Dict[str, Any]:
    """
    Round 1: Performs coarse scoring, matches core skills, and filters shortlist to Top 10.
    """
    requirements = state.get("requirements", {})
    must_have_skills = [s.lower().strip() for s in requirements.get("must_have_skills", [])]
    nice_to_have_skills = [s.lower().strip() for s in requirements.get("nice_to_have_skills", [])]
    
    raw_matches = state.get("shortlist", [])
    ranked_shortlist = []
    
    for c in raw_matches:
        candidate_skills = [s.lower().strip() for s in c.get("matched_skills", []) + c.get("skills", [])]
        
        # Calculate matched must-have and nice-to-have skills
        matched_must = [s for s in requirements.get("must_have_skills", []) if s.lower().strip() in candidate_skills]
        missing_must = [s for s in requirements.get("must_have_skills", []) if s.lower().strip() not in candidate_skills]
        
        # Structure CandidateMatch profile
        candidate_profile = {
            "candidate_id": c.get("resume_path"),
            "name": c.get("candidate_name", "Unknown"),
            "score": c.get("match_score", 0),
            "matched_skills": matched_must,
            "missing_skills": missing_must,
            "experience_years": c.get("experience_years", 0),
            "education": c.get("education", "Not Specified"),
            "relevance_excerpts": c.get("relevant_excerpts", []),
            "strengths": [],
            "gaps": [],
            "improvement_suggestions": "",
            "screening_status": "Shortlisted",
            "screening_reasoning": "Coarse filtering match",
            "interview_questions": []
        }
        ranked_shortlist.append(candidate_profile)
        
    coarse_limit = state.get("coarse_screen_limit") or config.DEFAULT_COARSE_LIMIT
    # Sort and slice to limit downstream token consumption
    ranked_shortlist.sort(key=lambda x: x["score"], reverse=True)
    return {"shortlist": ranked_shortlist[:int(coarse_limit)], "current_round": 1}


def deep_screen_node(state: AgentState) -> Dict[str, Any]:
    """
    Round 2: profile deep text audit.
    Evaluates Top 5 candidate resumes sequentially, mapping strengths, gaps, and suggestions.
    """
    shortlist = state.get("shortlist", [])
    requirements = state.get("requirements", {})
    
    llm = config.get_llm_model(
        provider=state.get("llm_provider", config.DEFAULT_PROVIDER),
        model_name=state.get("llm_model", config.DEFAULT_MODEL),
        api_key=state.get("api_key", ""),
        api_url=state.get("api_url")
    )
    
    deep_limit = state.get("deep_screen_limit") or config.DEFAULT_DEEP_LIMIT
    # Screen candidates dynamically based on deep_limit
    candidates_to_screen = shortlist[:int(deep_limit)]
    print(f"Executing Round 2 (Deep Screening) on top {len(candidates_to_screen)} candidates...")
    
    system_prompt = """You are a senior technical recruiter. Analyze the candidate's resume text against the active job requirements.
Identify their core strengths, key skills gaps or missing experience items, write actionable improvement suggestions, and assign a status ("Screened" or "Borderline").
You MUST return a valid JSON object ONLY. Do not include markdown code blocks, explanation text, or anything else. Just the raw JSON.

JSON structure must be:
{
    "strengths": ["list of 2-3 technical/project strengths relative to the JD"],
    "gaps": ["list of 1-2 key missing technologies, concepts, or experience constraints"],
    "improvement_suggestions": "Actionable feedback for the candidate on bridging their gaps (string)",
    "screening_status": "Screened" or "Borderline",
    "screening_reasoning": "A concise summary of their alignment to the requirements"
}"""

    for idx, c in enumerate(candidates_to_screen):
        # Read full resume text from file
        res = read_file(c["candidate_id"])
        if not res.get("success"):
            print(f"Skipping deep screening for {c['name']} (read error).")
            continue
            
        resume_text = res["content"]
        # Truncate content to avoid model token limits (approx 12,000 characters)
        if len(resume_text) > 12000:
            resume_text = resume_text[:12000] + "... [truncated]"
            
        # Throttling delay between LLM calls to respect API limits
        if idx > 0:
            time.sleep(config.THROTTLE_DELAY)
            
        prompt_content = f"""Candidate: {c['name']}
Job Title: {requirements.get('title', 'Software Engineer')}
Job Requirements: {requirements}
Candidate Resume Text:
{resume_text}"""

        def _call_deep_screen():
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt_content)
            ]
            response = llm.invoke(messages)
            content = response.content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n", "", content)
                content = re.sub(r"\n```$", "", content)
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            if start_idx != -1 and end_idx != -1:
                content = content[start_idx:end_idx+1]
            return json.loads(content)

        try:
            result = execute_with_retry(_call_deep_screen)
            c["strengths"] = result.get("strengths", [])
            c["gaps"] = result.get("gaps", [])
            c["improvement_suggestions"] = result.get("improvement_suggestions", "")
            c["screening_status"] = result.get("screening_status", "Screened")
            c["screening_reasoning"] = result.get("screening_reasoning", "")
        except Exception as e:
            print(f"Failed to screen {c['name']}: {e}")
            c["screening_status"] = "Screened"
            c["screening_reasoning"] = "Fallback screening due to LLM error"
            
    return {"shortlist": shortlist, "current_round": 2}


def recommendation_node(state: AgentState) -> Dict[str, Any]:
    """
    Round 3: Final Hiring decisions & customized technical screening questions generator.
    """
    shortlist = state.get("shortlist", [])
    requirements = state.get("requirements", {})
    
    llm = config.get_llm_model(
        provider=state.get("llm_provider", config.DEFAULT_PROVIDER),
        model_name=state.get("llm_model", config.DEFAULT_MODEL),
        api_key=state.get("api_key", ""),
        api_url=state.get("api_url")
    )
    
    rec_limit = state.get("recommendation_limit") or config.DEFAULT_RECOMMENDATION_LIMIT
    # Generate questions and recommendations dynamically based on rec_limit
    candidates_to_decide = shortlist[:int(rec_limit)]
    print(f"Executing Round 3 (Hire Decision & QGen) for top {len(candidates_to_decide)} candidates...")
    
    for idx, c in enumerate(candidates_to_decide):
        if idx > 0:
            time.sleep(config.THROTTLE_DELAY)
            
        # Generate interview questions targeting gaps
        questions = generate_interview_questions(
            candidate_name=c["name"],
            skills=c["matched_skills"] + c.get("skills", []),
            gaps=c["gaps"] if c["gaps"] else c["missing_skills"],
            requirements=requirements,
            llm=llm
        )
        c["interview_questions"] = questions
        
        # Heuristics + LLM recommendations validation
        if c["score"] >= 72 and len(c["missing_skills"]) == 0:
            c["screening_status"] = "Strong Hire"
        elif c["score"] >= 60 and len(c["missing_skills"]) <= 1:
            c["screening_status"] = "Borderline Hire"
        else:
            c["screening_status"] = "Rejected / No-Hire"
            
    return {"shortlist": shortlist, "current_round": 3}


def generate_report_node(state: AgentState) -> Dict[str, Any]:
    """
    Compiles candidate records, analysis reports, and interview matrices into a Markdown report.
    """
    shortlist = state.get("shortlist", [])
    previous_shortlist = state.get("previous_shortlist", [])
    requirements = state.get("requirements", {})
    messages = state.get("messages", [])
    
    ranking_explanation = ""
    
    # Check if there is a previous shortlist and it is different from the current one
    has_changes = False
    if previous_shortlist and shortlist:
        prev_names = [c["name"] for c in previous_shortlist[:5]]
        curr_names = [c["name"] for c in shortlist[:5]]
        if prev_names != curr_names:
            has_changes = True
            
    if has_changes and messages:
        feedback_instructions = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) or (hasattr(msg, "type") and msg.type == "human"):
                feedback_instructions = msg.content
                break
                
        if feedback_instructions:
            llm = config.get_llm_model(
                provider=state.get("llm_provider", config.DEFAULT_PROVIDER),
                model_name=state.get("llm_model", config.DEFAULT_MODEL),
                api_key=state.get("api_key", ""),
                api_url=state.get("api_url")
            )
            
            prev_summary = "\n".join(f"  {idx+1}. {c['name']} (Score: {c['score']}/100, Status: {c.get('screening_status', 'Shortlisted')}, Matched Skills: {c.get('matched_skills', [])})" for idx, c in enumerate(previous_shortlist[:5]))
            curr_summary = "\n".join(f"  {idx+1}. {c['name']} (Score: {c['score']}/100, Status: {c.get('screening_status', 'Shortlisted')}, Matched Skills: {c.get('matched_skills', [])})" for idx, c in enumerate(shortlist[:5]))
            
            system_prompt = """You are a professional recruiting coordinator. Explain why the candidate rankings changed after the user instructions.
Compare the previous shortlist with the current new shortlist. Highlight key movements (e.g. who went up/down, who is new) and explain the specific reasons based on the user's updated requirements (e.g. adding a new must-have skill).
Keep your explanation concise, professional, and directly actionable for the recruiter (maximum 2 short paragraphs). Do not include markdown code blocks, just plain markdown text."""

            user_prompt = f"""User Instruction: "{feedback_instructions}"

Previous Shortlist:
{prev_summary}

New Shortlist:
{curr_summary}

Explain the changes in rankings:"""

            def _call_explain():
                prompt_msgs = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                response = llm.invoke(prompt_msgs)
                return response.content.strip()
                
            try:
                ranking_explanation = execute_with_retry(_call_explain)
                print(f"Ranking changes explanation generated:\n{ranking_explanation}")
            except Exception as e:
                print(f"Failed to generate ranking explanation: {e}")
                ranking_explanation = "Candidate rankings updated based on the new constraints."
                
    rec_limit = state.get("recommendation_limit") or config.DEFAULT_RECOMMENDATION_LIMIT
    deep_limit = state.get("deep_screen_limit") or config.DEFAULT_DEEP_LIMIT
    
    # Generate side-by-side comparison table
    candidate_ids = [c["candidate_id"] for c in shortlist[:int(rec_limit)]]
    compare_matrix = compare_candidates(candidate_ids, shortlist)
    
    report_lines = [
        "# Candidate Match & Screening Report",
        "",
        f"### Active Job Profile: {requirements.get('title', 'Software Engineer')}",
        f"- **Min Experience**: {requirements.get('min_experience_years', 0)} Years",
        f"- **Must-Have Skills**: {', '.join(requirements.get('must_have_skills', [])) if requirements.get('must_have_skills') else 'None'}",
        f"- **Nice-To-Have Skills**: {', '.join(requirements.get('nice_to_have_skills', [])) if requirements.get('nice_to_have_skills') else 'None'}",
        f"- **Education Target**: {requirements.get('education_level', 'Not Specified')}",
        "",
    ]
    
    if ranking_explanation:
        report_lines.extend([
            "## 🔄 Ranking Changes Explanation",
            "",
            ranking_explanation,
            ""
        ])
        
    report_lines.extend([
        "## Candidate Comparison Matrix",
        "",
        compare_matrix,
        "",
        "## Screening Details",
        ""
    ])
    
    for idx, c in enumerate(shortlist[:int(deep_limit)]):  # Deep screening up to dynamic deep_limit candidates
        status = c.get("screening_status", "").lower()
        if "reject" in status or "no-hire" in status:
            status_color = "🔴"
        elif "borderline" in status:
            status_color = "🟡"
        else:
            status_color = "🟢"
        
        report_lines.extend([
            f"### {idx+1}. {c['name']} (Score: {c['score']}/100) - {status_color} {c.get('screening_status', 'Shortlisted')}",
            f"- **Experience**: {c['experience_years']} Years",
            f"- **Education**: {c['education']}",
            f"- **Matched Skills**: {', '.join(c['matched_skills']) if c['matched_skills'] else 'None'}",
            f"- **Missing Skills**: {', '.join(c['missing_skills']) if c['missing_skills'] else 'None'}",
            "",
            "#### Diagnostic Breakdown:",
            f"- **Strengths**: {', '.join(c.get('strengths', [])) if c.get('strengths') else 'Not evaluated yet'}",
            f"- **Gaps**: {', '.join(c.get('gaps', [])) if c.get('gaps') else 'None'}",
            f"- **Suggestions**: *{c.get('improvement_suggestions', 'None')}*",
            f"- **Reasoning**: {c.get('screening_reasoning', 'Coarse filtering match')}",
            ""
        ])
        
        if c.get("interview_questions"):
            report_lines.extend([
                "#### Tailored Screening Questions:",
                "\n".join(f"  - {q}" for q in c["interview_questions"]),
                ""
            ])
            
    report = "\n".join(report_lines)
    
    # Generate conversational summary chat message for the chat workspace logs
    summary_lines = [
        "I have completed the screening cascade for the candidate resumes.",
        "",
        "**Top Shortlisted Candidates**:"
    ]
    for idx, c in enumerate(shortlist[:3]):
        status = c.get("screening_status", "").lower()
        if "reject" in status or "no-hire" in status:
            status_color = "🔴"
        elif "borderline" in status:
            status_color = "🟡"
        else:
            status_color = "🟢"
        summary_lines.append(f"- **#{idx+1} {c['name']}** (Score: {c['score']}/100) — {status_color} `{c.get('screening_status', 'Shortlisted')}`")
        
    if ranking_explanation:
        summary_lines.extend([
            "",
            "**Ranking Changes Explanation**:",
            ranking_explanation
        ])
        
    summary_lines.extend([
        "",
        "*(Note: You can view full head-to-head metrics in the **Shortlist & Comparison** tab, and read deep audits in the **Deep Screening Reports** tab.)*"
    ])
    
    ai_msg = AIMessage(content="\n".join(summary_lines))
    
    return {
        "final_report": report, 
        "ranking_explanation": ranking_explanation,
        "messages": messages + [ai_msg]
    }


def adjust_requirements_node(state: AgentState) -> Dict[str, Any]:
    """
    Conversational feedback loop: refines requirements constraints using user instructions.
    """
    messages = state.get("messages", [])
    if not messages:
        return {}

    last_msg = messages[-1].content
    current_reqs = state.get("requirements", {})
    
    llm = config.get_llm_model(
        provider=state.get("llm_provider", config.DEFAULT_PROVIDER),
        model_name=state.get("llm_model", config.DEFAULT_MODEL),
        api_key=state.get("api_key", ""),
        api_url=state.get("api_url")
    )
    
    print(f"Refining job requirements based on feedback: '{last_msg}'")
    
    system_prompt = f"""You are a recruiting coordinator. Update the active Job Requirements JSON based on the user's conversational instructions.
Merge the updates into the current requirements structure. Ensure to ADD or REMOVE skills as instructed.
You MUST return a valid JSON object ONLY. Do not include markdown code blocks, explanation text, or anything else. Just the raw JSON.

Current Active Requirements JSON:
{json.dumps(current_reqs, indent=2)}

Target JSON format:
{{
    "title": "Job Title (string)",
    "must_have_skills": ["List of must-have skills"],
    "nice_to_have_skills": ["List of nice-to-have skills"],
    "min_experience_years": 5 (integer),
    "education_level": "e.g., Bachelor, Master, PhD, B.Tech",
    "other_constraints": ["List of constraints"]
}}"""

    def _call_adjust():
        messages_prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Update the requirements using this instruction: '{last_msg}'")
        ]
        response = llm.invoke(messages_prompt)
        content = response.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n", "", content)
            content = re.sub(r"\n```$", "", content)
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx:end_idx+1]
        return json.loads(content)

    try:
        updated_reqs = execute_with_retry(_call_adjust)
        print(f"Updated requirements: {updated_reqs}")
        return {"requirements": updated_reqs, "current_round": 1}
    except Exception as e:
        print(f"Failed to adjust requirements: {e}")
        return {}


def conversational_query_node(state: AgentState) -> Dict[str, Any]:
    """
    Directly answers conversational questions from the recruiter (e.g. comparison queries, rankings explanation)
    using the active candidate shortlist data.
    """
    messages = state.get("messages", [])
    if not messages:
        return {}

    last_msg = messages[-1].content
    shortlist = state.get("shortlist", [])
    requirements = state.get("requirements", {})
    
    llm = config.get_llm_model(
        provider=state.get("llm_provider", config.DEFAULT_PROVIDER),
        model_name=state.get("llm_model", config.DEFAULT_MODEL),
        api_key=state.get("api_key", ""),
        api_url=state.get("api_url")
    )
    
    print(f"Executing Conversational Query: '{last_msg}'")
    
    system_prompt = """You are an experienced recruiter assistant. Answer the user's question regarding the active candidate shortlist or requirements.
Use the provided candidate shortlist details and requirements to construct a professional, clear, and well-reasoned answer.
If the user asks to compare candidates, construct a clear comparison breakdown or markdown comparison table.
If they ask why one candidate ranked higher than another, contrast their relative match scores, Matched Skills, Missing Skills, and Experience Years.

Job Requirements:
{reqs_json}

Active Shortlist:
{shortlist_json}"""

    def _call_query():
        messages_prompt = [
            SystemMessage(content=system_prompt.format(
                reqs_json=json.dumps(requirements, indent=2),
                shortlist_json=json.dumps(shortlist, indent=2)
            )),
            HumanMessage(content=f"Answer this query: '{last_msg}'")
        ]
        return llm.invoke(messages_prompt)

    try:
        response = execute_with_retry(_call_query)
        ai_msg = AIMessage(content=response.content)
        return {"messages": messages + [ai_msg]}
    except Exception as e:
        print(f"Failed to execute conversational query: {e}")
        err_msg = AIMessage(content=f"I encountered an error trying to analyze that: {str(e)}")
        return {"messages": messages + [err_msg]}


# ----------------------------------------------------
# Routing Logic
# ----------------------------------------------------

def route_input(state: AgentState) -> str:
    """
    Decides whether to parse a new JD, update requirements, or answer conversational questions.
    """
    messages = state.get("messages", [])
    if not messages:
        return "extract_requirements"
        
    last_msg = messages[-1].content.lower()
    lines = [line.strip() for line in last_msg.split("\n") if line.strip()]
    is_jd = len(lines) > 3 or any(w in last_msg for w in ["job description", "requirements:", "duties:", "responsibilities:"])
    
    if is_jd or not state.get("requirements"):
        return "extract_requirements"
        
    # Heuristics for comparison/explanation queries
    conversational_keywords = ["why", "compare", "higher", "better", "explain", "vs", "versus", "who", "show", "tell me about"]
    if any(kw in last_msg for kw in conversational_keywords) and len(state.get("shortlist", [])) > 0:
        return "conversational_query"
        
    return "adjust_requirements"


# ----------------------------------------------------
# LangGraph Workflow Construction
# ----------------------------------------------------

builder = StateGraph(AgentState)

# Add Nodes
builder.add_node("parse_input", parse_input_node)
builder.add_node("extract_requirements", extract_requirements_node)
builder.add_node("adjust_requirements", adjust_requirements_node)
builder.add_node("conversational_query", conversational_query_node)
builder.add_node("search_resumes", search_resumes_node)
builder.add_node("rank_candidates", rank_candidates_node)
builder.add_node("deep_screen", deep_screen_node)
builder.add_node("recommendation", recommendation_node)
builder.add_node("generate_report", generate_report_node)

# Add Edges
builder.add_edge(START, "parse_input")

# Conditional Router from parse_input
builder.add_conditional_edges(
    "parse_input",
    route_input,
    {
        "extract_requirements": "extract_requirements",
        "adjust_requirements": "adjust_requirements",
        "conversational_query": "conversational_query"
    }
)

builder.add_edge("extract_requirements", "search_resumes")
builder.add_edge("adjust_requirements", "search_resumes")
builder.add_edge("search_resumes", "rank_candidates")
builder.add_edge("rank_candidates", "deep_screen")
builder.add_edge("deep_screen", "recommendation")
builder.add_edge("recommendation", "generate_report")
builder.add_edge("generate_report", END)
builder.add_edge("conversational_query", END)

# Compile matching workflow using in-memory checkpointer
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
matching_agent_workflow = builder.compile(checkpointer=memory)


if __name__ == "__main__":
    import os
    print("Compiling LangGraph Workflow and generating state machine diagram...")
    
    # Create docs/ folder in root if it doesn't exist
    docs_dir = Path(__file__).resolve().parent.parent.parent / "docs"
    docs_dir.mkdir(exist_ok=True)
    
    try:
        # Save mermaid representation
        graph = matching_agent_workflow.get_graph()
        mermaid_code = graph.draw_mermaid()
        
        mermaid_path = docs_dir / "state_machine.mermaid"
        with open(mermaid_path, "w") as f:
            f.write(mermaid_code)
        print(f"Mermaid state machine diagram saved to {mermaid_path}")
        
        # Draw and save PNG
        png_data = graph.draw_mermaid_png()
        png_path = docs_dir / "state_machine.png"
        with open(png_path, "wb") as f:
            f.write(png_data)
        print(f"PNG state machine diagram saved to {png_path}")
    except Exception as e:
        print(f"Note: Could not generate visual PNG diagram (probably missing graphviz/pygraphviz). Error: {e}")
        print("Mermaid representation is still saved, which can be rendered in markdown.")

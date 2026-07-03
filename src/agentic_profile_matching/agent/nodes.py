import re
import time
import json
from typing import Dict, List, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from agentic_profile_matching import config
from agentic_profile_matching.fs_client import read_file
from agentic_profile_matching.job_matcher import JobMatcher
from agentic_profile_matching.tools import (
    extract_requirements,
    compare_candidates,
    generate_interview_questions,
    execute_with_retry
)
from agentic_profile_matching.agent.state import AgentState
from agentic_profile_matching.agent.prompts import (
    DEEP_SCREEN_SYSTEM_PROMPT,
    RANKING_EXPLANATION_SYSTEM_PROMPT,
    RANKING_EXPLANATION_USER_PROMPT,
    ADJUST_REQUIREMENTS_SYSTEM_PROMPT,
    CONVERSATIONAL_QUERY_SYSTEM_PROMPT
)


def parse_input_node(state: AgentState) -> Dict[str, Any]:
    """
    Inspects the latest user message to classify it as a raw Job Description or refinement query.
    """
    messages = state.get("messages", [])
    errors = state.get("errors")
    if errors is None:
        errors = []
    
    # Capture the previous shortlist before updating
    prev_shortlist = state.get("shortlist", [])
    
    if not messages:
        return {"current_round": 1, "previous_shortlist": prev_shortlist, "errors": errors}

    try:
        last_msg = messages[-1].content
        # Simple, highly reliable heuristic: multi-line strings or presence of typical JD words
        lines = [line.strip() for line in last_msg.split("\n") if line.strip()]
        is_jd = len(lines) > 3 or any(w in last_msg.lower() for w in ["job description", "requirements:", "duties:", "responsibilities:"])
        
        if is_jd:
            print("Input classified as raw Job Description.")
            return {"current_round": 1, "previous_shortlist": prev_shortlist, "errors": errors}
        else:
            print("Input classified as refinement query.")
            return {"current_round": 1, "previous_shortlist": prev_shortlist, "errors": errors}
    except Exception as e:
        print(f"Error parsing input: {e}")
        return {"current_round": 1, "previous_shortlist": prev_shortlist, "errors": errors + [f"Input parsing warning: {str(e)}"]}


def extract_requirements_node(state: AgentState) -> Dict[str, Any]:
    """
    LLM extracts structured job requirements from raw input message content.
    """
    messages = state.get("messages", [])
    errors = state.get("errors")
    if errors is None:
        errors = []
        
    fallback_reqs = {
        "title": "Software Engineer",
        "must_have_skills": [],
        "nice_to_have_skills": [],
        "min_experience_years": 0,
        "education_level": "Not Specified",
        "other_constraints": []
    }
    
    if not messages:
        return {"requirements": fallback_reqs, "errors": errors}

    try:
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
        if not isinstance(requirements, dict):
            requirements = fallback_reqs
            errors.append("Invalid structure returned by extract_requirements, using fallback.")
            
        return {"requirements": requirements, "current_round": 1, "errors": errors}
    except Exception as e:
        print(f"Error in extract_requirements_node: {e}")
        errors.append(f"Requirements extraction failed: {str(e)}. Using fallback requirements.")
        return {"requirements": fallback_reqs, "current_round": 1, "errors": errors}


def search_resumes_node(state: AgentState) -> Dict[str, Any]:
    """
    Retrieves candidate resumes matching constraints using local hybrid search.
    """
    errors = state.get("errors")
    if errors is None:
        errors = []
        
    requirements = state.get("requirements", {}) or {}
    title = requirements.get("title", "Software Engineer")
    must_have = requirements.get("must_have_skills", [])
    min_exp = requirements.get("min_experience_years", 0)
    
    coarse_limit = state.get("coarse_screen_limit") or config.DEFAULT_COARSE_LIMIT
    retrieval_k = max(int(coarse_limit * 1.5), 15)
    
    try:
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
        if not results or not results.get("top_matches"):
            print("Zero candidates satisfied strict criteria. Relaxing filters...")
            results = matcher.match(
                job_description=query_text,
                k=retrieval_k,
                min_exp=min_exp,
                must_have_skills=must_have,
                apply_filters=False
            )
            
        raw_matches = results.get("top_matches", []) if results else []
        return {"shortlist": raw_matches, "errors": errors}
    except Exception as e:
        print(f"Error in search_resumes_node: {e}")
        errors.append(f"Resume search failed: {str(e)}. RAG retrieval skipped.")
        return {"shortlist": [], "errors": errors}


def rank_candidates_node(state: AgentState) -> Dict[str, Any]:
    """
    Round 1: Performs coarse scoring, matches core skills, and filters shortlist to Top 10.
    """
    errors = state.get("errors")
    if errors is None:
        errors = []
        
    try:
        requirements = state.get("requirements", {}) or {}
        must_have_skills = [s.lower().strip() for s in requirements.get("must_have_skills", []) if s]
        nice_to_have_skills = [s.lower().strip() for s in requirements.get("nice_to_have_skills", []) if s]
        
        raw_matches = state.get("shortlist", []) or []
        ranked_shortlist = []
        
        for c in raw_matches:
            candidate_skills = [s.lower().strip() for s in c.get("matched_skills", []) + c.get("skills", []) if s]
            
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
        ranked_shortlist.sort(key=lambda x: x.get("score", 0), reverse=True)
        return {"shortlist": ranked_shortlist[:int(coarse_limit)], "current_round": 1, "errors": errors}
    except Exception as e:
        print(f"Error in rank_candidates_node: {e}")
        errors.append(f"Ranking failed: {str(e)}")
        return {"shortlist": [], "current_round": 1, "errors": errors}


def deep_screen_node(state: AgentState) -> Dict[str, Any]:
    """
    Round 2: profile deep text audit.
    Evaluates Top 5 candidate resumes sequentially, mapping strengths, gaps, and suggestions.
    """
    shortlist = state.get("shortlist", [])
    requirements = state.get("requirements", {})
    errors = state.get("errors")
    if errors is None:
        errors = []
    
    try:
        llm = config.get_llm_model(
            provider=state.get("llm_provider", config.DEFAULT_PROVIDER),
            model_name=state.get("llm_model", config.DEFAULT_MODEL),
            api_key=state.get("api_key", ""),
            api_url=state.get("api_url")
        )
    except Exception as e:
        print(f"Error building LLM model in deep_screen_node: {e}")
        errors.append(f"Failed to build LLM for deep screening: {str(e)}")
        llm = None
    
    deep_limit = state.get("deep_screen_limit") or config.DEFAULT_DEEP_LIMIT
    # Screen candidates dynamically based on deep_limit
    candidates_to_screen = shortlist[:int(deep_limit)]
    print(f"Executing Round 2 (Deep Screening) on top {len(candidates_to_screen)} candidates...")
    
    for idx, c in enumerate(candidates_to_screen):
        # Read full resume text from file
        res = read_file(c["candidate_id"])
        if not res or not res.get("success"):
            err_msg = f"Incomplete parsing: could not read resume for {c.get('name')} ({res.get('error') if res else 'Empty response'})"
            print(err_msg)
            errors.append(err_msg)
            c["strengths"] = ["Strong skill overlap based on RAG indexing"]
            c["gaps"] = ["Could not audit text (file unreadable / unparsed)"]
            c["improvement_suggestions"] = "Review resume file formatting before interviewing candidate."
            c["screening_status"] = "Screened"
            c["screening_reasoning"] = f"Fallback screening (unreadable file: {res.get('error') if res else 'Unknown error'})"
            continue
            
        resume_text = res["content"]
        # Truncate content to avoid model token limits (approx 12,000 characters)
        if len(resume_text) > 12000:
            resume_text = resume_text[:12000] + "... [truncated]"
            
        # Throttling delay between LLM calls to respect API limits
        if idx > 0:
            time.sleep(config.THROTTLE_DELAY)
            
        if llm is None:
            c["strengths"] = ["Semantic match based on vector DB indexing"]
            c["gaps"] = ["Skipped deep screening audit due to missing LLM configuration"]
            c["improvement_suggestions"] = "Configure LLM provider with a valid API key."
            c["screening_status"] = "Screened"
            c["screening_reasoning"] = "Fallback screening due to unconfigured LLM"
            continue

        prompt_content = f"""Candidate: {c['name']}
Job Title: {requirements.get('title', 'Software Engineer')}
Job Requirements: {requirements}
Candidate Resume Text:
{resume_text}"""

        def _call_deep_screen():
            messages = [
                SystemMessage(content=DEEP_SCREEN_SYSTEM_PROMPT),
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
            err_msg = f"Failed to screen {c['name']}: {e}"
            print(err_msg)
            errors.append(err_msg)
            c["strengths"] = ["Semantic match based on vector DB indexing"]
            c["gaps"] = ["Skipped deep screening audit due to LLM error"]
            c["improvement_suggestions"] = "Schedule interview to evaluate candidates skills directly."
            c["screening_status"] = "Screened"
            c["screening_reasoning"] = f"Fallback screening due to LLM or parse error: {str(e)}"
            
    return {"shortlist": shortlist, "current_round": 2, "errors": errors}


def recommendation_node(state: AgentState) -> Dict[str, Any]:
    """
    Round 3: Final Hiring decisions & customized technical screening questions generator.
    """
    shortlist = state.get("shortlist", [])
    requirements = state.get("requirements", {})
    errors = state.get("errors")
    if errors is None:
        errors = []
        
    try:
        llm = config.get_llm_model(
            provider=state.get("llm_provider", config.DEFAULT_PROVIDER),
            model_name=state.get("llm_model", config.DEFAULT_MODEL),
            api_key=state.get("api_key", ""),
            api_url=state.get("api_url")
        )
    except Exception as e:
        print(f"Error building LLM model in recommendation_node: {e}")
        errors.append(f"Failed to build LLM for interview question generation: {str(e)}")
        llm = None
    
    rec_limit = state.get("recommendation_limit") or config.DEFAULT_RECOMMENDATION_LIMIT
    # Generate questions and recommendations dynamically based on rec_limit
    candidates_to_decide = shortlist[:int(rec_limit)]
    print(f"Executing Round 3 (Hire Decision & QGen) for top {len(candidates_to_decide)} candidates...")
    
    for idx, c in enumerate(candidates_to_decide):
        if idx > 0:
            time.sleep(config.THROTTLE_DELAY)
            
        # Generate interview questions targeting gaps
        try:
            if llm is not None:
                questions = generate_interview_questions(
                    candidate_name=c.get("name", "Unknown"),
                    skills=c.get("matched_skills", []) + c.get("skills", []),
                    gaps=c.get("gaps") if c.get("gaps") else c.get("missing_skills", []),
                    requirements=requirements,
                    llm=llm
                )
                c["interview_questions"] = questions
            else:
                raise ValueError("LLM not configured.")
        except Exception as e:
            err_msg = f"Failed to generate interview questions for {c.get('name')}: {str(e)}"
            print(err_msg)
            errors.append(err_msg)
            c["interview_questions"] = [
                "Can you walk me through your engineering experience?",
                "What is your approach to learning new technologies?",
                "How do you handle microservices architecture issues?"
            ]
        
        # Heuristics + LLM recommendations validation
        try:
            score = c.get("score", 0)
            missing = c.get("missing_skills", [])
            if score >= 72 and len(missing) == 0:
                c["screening_status"] = "Strong Hire"
            elif score >= 60 and len(missing) <= 1:
                c["screening_status"] = "Borderline Hire"
            else:
                c["screening_status"] = "Rejected / No-Hire"
        except Exception as e:
            c["screening_status"] = "Screened"
            
    return {"shortlist": shortlist, "current_round": 3, "errors": errors}


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
            
            user_prompt = RANKING_EXPLANATION_USER_PROMPT.format(
                feedback_instructions=feedback_instructions,
                prev_summary=prev_summary,
                curr_summary=curr_summary
            )
            
            def _call_explain():
                prompt_msgs = [
                    SystemMessage(content=RANKING_EXPLANATION_SYSTEM_PROMPT),
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
            
            
    if state.get("errors"):
        report_lines.extend([
            "---",
            "### ⚠️ System warnings / execution errors during matching run:",
            "",
            "\n".join(f"- {err}" for err in state["errors"]),
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
        
    if state.get("errors"):
        summary_lines.extend([
            "",
            "⚠️ **Warnings encountered** (see deep audits or report logs for details)"
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
    
    system_prompt = ADJUST_REQUIREMENTS_SYSTEM_PROMPT.format(
        current_reqs_json=json.dumps(current_reqs, indent=2)
    )

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

    def _call_query():
        messages_prompt = [
            SystemMessage(content=CONVERSATIONAL_QUERY_SYSTEM_PROMPT.format(
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

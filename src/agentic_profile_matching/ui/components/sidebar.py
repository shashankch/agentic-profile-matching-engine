"""
Sidebar Component for Yojaka AI UI.
Manages LLM provider configuration, stateless credentials, live talent pool inventory badge,
concurrency/limit sliders, and dynamic requirements constraints editing.
"""

import os
from pathlib import Path
import streamlit as st

from agentic_profile_matching import config
from agentic_profile_matching.ui.runner import run_agent_workflow_with_status


def render_sidebar() -> dict:
    """
    Renders the configuration and filters sidebar.
    Returns the workflow_config dictionary containing provider credentials and vector store handles.
    """
    st.sidebar.title("Configuration & Filters")

    # 1. LLM Provider Setup
    st.sidebar.markdown("### 1. LLM Provider Setup")
    llm_provider = st.sidebar.selectbox("LLM Provider", list(config.SUPPORTED_PROVIDERS.keys()), index=0)
    supported_models = config.SUPPORTED_PROVIDERS[llm_provider]
    llm_model = st.sidebar.selectbox("Model Name", supported_models, index=0)

    # Pre-populate keys from environment secrets
    default_key = ""
    if llm_provider == "Groq":
        default_key = os.getenv("GROQ_API_KEY", "")
    elif llm_provider == "Gemini":
        default_key = os.getenv("GEMINI_API_KEY", "")
    elif llm_provider == "Sarvam AI":
        default_key = os.getenv("SARVAM_API_KEY", "")
    elif llm_provider == "OpenAI":
        default_key = os.getenv("OPENAI_API_KEY", "")

    api_key = st.sidebar.text_input("API Key", value=default_key, type="password")

    api_url = None
    if llm_provider == "Custom (OpenAI-compatible)":
        api_url = st.sidebar.text_input("API Base URL (Endpoint)", value="https://api.openai.com/v1")

    tavily_key = st.sidebar.text_input(
        "Tavily Search API Key (Optional)",
        value=os.getenv("TAVILY_API_KEY", ""),
        type="password",
        help="Enables real-time web search for tech trends & market intelligence in chat. If omitted, built-in search fallback is used.",
    )
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key

    # 2. Active Talent Pool Summary Badge
    try:
        total_indexed_chunks = st.session_state["session_vector_store"].count()
    except Exception:
        total_indexed_chunks = 0

    try:
        resumes_path = Path(config.RESUMES_DIR)
        if not resumes_path.exists():
            fallback_path = Path(config.BASE_DIR).parent / "data" / "resumes"
            if fallback_path.exists():
                resumes_path = fallback_path
        all_files = list(resumes_path.glob("*.*"))
        disk_count = len([f for f in all_files if f.suffix.lower() in [".pdf", ".docx", ".txt"]])
        valid_count = disk_count + len(st.session_state.get("uploaded_candidates", []))
    except Exception:
        valid_count = 34 + len(st.session_state.get("uploaded_candidates", []))

    st.sidebar.markdown(
        f"""
        <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.22); border-radius: 8px; padding: 7px 12px; margin: 0.6rem 0 0.5rem 0; font-size: 0.85rem; color: #cbd5e1; display: flex; align-items: center; justify-content: space-between;">
            <span>📂 <strong>{valid_count} Profiles</strong> Active</span>
            <span style="font-family: monospace; background: rgba(99, 102, 241, 0.2); padding: 2px 6px; border-radius: 4px; font-size: 0.78rem;">{total_indexed_chunks} chunks</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Throttling & Limits Control
    with st.sidebar.expander("⚙️ 3. Throttling & Limits Control ▾", expanded=False):
        coarse_limit = st.slider(
            "Round 1: Coarse Limit",
            min_value=5,
            max_value=20,
            value=int(st.session_state["coarse_limit"]),
        )
        deep_limit = st.slider(
            "Round 2: Deep Screen Limit",
            min_value=3,
            max_value=15,
            value=int(st.session_state["deep_limit"]),
        )
        recommendation_limit = st.slider(
            "Round 3: Recommendation Limit",
            min_value=2,
            max_value=10,
            value=int(st.session_state["recommendation_limit"]),
        )

        st.session_state["coarse_limit"] = coarse_limit
        st.session_state["deep_limit"] = deep_limit
        st.session_state["recommendation_limit"] = recommendation_limit

    # 4. Active Requirements Constraints
    with st.sidebar.expander("📋 4. Active Requirements Constraints ▾", expanded=False):
        reqs = st.session_state["requirements"]
        title_input = st.text_input("Extracted Job Title", value=reqs.get("title", "Software Engineer"))
        min_exp_slider = st.slider(
            "Min Experience Years",
            min_value=0,
            max_value=20,
            value=int(reqs.get("min_experience_years", 0)),
        )
        must_have_input = st.text_area(
            "Must-Have Skills (comma separated)",
            value=", ".join(reqs.get("must_have_skills", [])),
        )
        nice_have_input = st.text_area(
            "Nice-To-Have Skills (comma separated)",
            value=", ".join(reqs.get("nice_to_have_skills", [])),
        )
        education_level = st.text_input("Education Level Target", value=reqs.get("education_level", "Not Specified"))

        if st.button("Sync Constraints & Re-Rank"):
            updated_reqs = {
                "title": title_input,
                "must_have_skills": [s.strip() for s in must_have_input.split(",") if s.strip()],
                "nice_to_have_skills": [s.strip() for s in nice_have_input.split(",") if s.strip()],
                "min_experience_years": min_exp_slider,
                "education_level": education_level,
                "other_constraints": reqs.get("other_constraints", []),
                "skill_expansions": reqs.get("skill_expansions", {}),
            }
            st.session_state["requirements"] = updated_reqs

            state_input = {
                "messages": st.session_state["messages"],
                "requirements": updated_reqs,
                "shortlist": [],
                "coarse_screen_limit": st.session_state["coarse_limit"],
                "deep_screen_limit": st.session_state["deep_limit"],
                "recommendation_limit": st.session_state["recommendation_limit"],
                "current_round": 1,
                "final_report": "",
                "feedback_pending": False,
                "user_feedback": "Recruiter updated requirements manually via sidebar.",
                "errors": [],
            }

            workflow_config = {
                "configurable": {
                    "thread_id": "streamlit-session-thread",
                    "api_key": api_key,
                    "api_url": api_url,
                    "tavily_api_key": tavily_key,
                    "llm_provider": llm_provider,
                    "llm_model": llm_model,
                    "store": st.session_state["session_vector_store"],
                }
            }

            result = run_agent_workflow_with_status(state_input, workflow_config)
            st.session_state["shortlist"] = result.get("shortlist", [])
            st.session_state["final_report"] = result.get("final_report", "")
            st.session_state["ranking_explanation"] = result.get("ranking_explanation", "")
            st.session_state["errors"] = result.get("errors", [])
            st.success("Shortlist re-ranked successfully!")

    return {
        "configurable": {
            "thread_id": "streamlit-session-thread",
            "api_key": api_key,
            "api_url": api_url,
            "tavily_api_key": tavily_key,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "store": st.session_state["session_vector_store"],
        }
    }

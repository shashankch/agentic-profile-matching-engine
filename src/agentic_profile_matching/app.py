import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Load configurations
load_dotenv()
from agentic_profile_matching import config
from agentic_profile_matching.matching_agent import matching_agent_workflow
from agentic_profile_matching.tools import compare_candidates


# Setup page config
st.set_page_config(
    page_title="AI Recruiter - Agentic Profile Matching",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Global Typography */
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Main App Header with Gradient */
    .app-header {
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.75rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
        letter-spacing: -0.025em;
    }
    
    .app-subtitle {
        font-size: 1.1rem;
        color: #9ca3af;
        margin-bottom: 1.75rem;
    }

    /* Candidate Card Layout */
    .candidate-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.15rem;
        margin-bottom: 1rem;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .candidate-card:hover {
        transform: translateY(-3px);
        border-color: rgba(167, 139, 250, 0.4);
        box-shadow: 0 10px 18px -6px rgba(167, 139, 250, 0.25);
        background: rgba(30, 41, 59, 0.7);
    }

    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }

    .card-name {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff;
    }

    .score-badge {
        font-size: 1.05rem;
        font-weight: 700;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 3px 9px;
        border-radius: 6px;
        box-shadow: 0 2px 4px rgba(16, 185, 129, 0.15);
    }

    /* Status Pills */
    .status-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .status-strong {
        background-color: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }

    .status-borderline {
        background-color: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.25);
    }

    .status-rejected {
        background-color: rgba(239, 68, 68, 0.12);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.25);
    }

    /* Skills Badges */
    .skill-tag {
        display: inline-block;
        background: rgba(255, 255, 255, 0.05);
        color: #e5e7eb;
        padding: 2px 7px;
        border-radius: 5px;
        font-size: 0.72rem;
        margin-right: 4px;
        margin-bottom: 4px;
        border: 1px solid rgba(255, 255, 255, 0.02);
    }

    /* Customizing Streamlit Tabs & Buttons */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: transparent;
        padding: 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 6px 6px 0 0;
        background-color: rgba(255, 255, 255, 0.01);
        color: #9ca3af;
        border: 1px solid transparent;
        padding: 0 14px;
        transition: all 0.2s;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(167, 139, 250, 0.08) !important;
        color: #c084fc !important;
        border-color: rgba(167, 139, 250, 0.2) rgba(167, 139, 250, 0.2) transparent rgba(167, 139, 250, 0.2) !important;
        font-weight: 600;
    }

    /* Styled buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 8px 20px !important;
        border-radius: 8px !important;
        box-shadow: 0 3px 5px rgba(79, 70, 229, 0.2) !important;
        transition: all 0.2s !important;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-1.5px) !important;
        box-shadow: 0 6px 10px rgba(79, 70, 229, 0.3) !important;
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
    }

    .stButton>button:active {
        transform: translateY(0) !important;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State Variables
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "requirements" not in st.session_state:
    st.session_state["requirements"] = {
        "title": "Software Engineer",
        "must_have_skills": [],
        "nice_to_have_skills": [],
        "min_experience_years": 0,
        "education_level": "Not Specified",
        "other_constraints": []
    }
if "shortlist" not in st.session_state:
    st.session_state["shortlist"] = []
if "final_report" not in st.session_state:
    st.session_state["final_report"] = ""
if "ranking_explanation" not in st.session_state:
    st.session_state["ranking_explanation"] = ""
if "coarse_limit" not in st.session_state:
    st.session_state["coarse_limit"] = config.DEFAULT_COARSE_LIMIT
if "deep_limit" not in st.session_state:
    st.session_state["deep_limit"] = config.DEFAULT_DEEP_LIMIT
if "recommendation_limit" not in st.session_state:
    st.session_state["recommendation_limit"] = config.DEFAULT_RECOMMENDATION_LIMIT
if "errors" not in st.session_state:
    st.session_state["errors"] = []


# ----------------------------------------------------
# Sidebar: LLM Configuration & Requirements Details
# ----------------------------------------------------

st.sidebar.title("Configuration & Filters")

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
elif llm_provider == "OpenAI":
    default_key = os.getenv("OPENAI_API_KEY", "")

api_key = st.sidebar.text_input("API Key", value=default_key, type="password")

api_url = None
if llm_provider == "Custom (OpenAI-compatible)":
    api_url = st.sidebar.text_input("API Base URL (Endpoint)", value="https://api.openai.com/v1")

st.sidebar.markdown("---")
st.sidebar.markdown("### 2. Throttling & Limits Control")
with st.sidebar.expander("⚙️ Limit Configurations (RPM/TPM Optimization)"):
    coarse_limit = st.slider("Round 1: Coarse Limit", min_value=5, max_value=20, value=int(st.session_state["coarse_limit"]))
    deep_limit = st.slider("Round 2: Deep Screen Limit", min_value=3, max_value=15, value=int(st.session_state["deep_limit"]))
    recommendation_limit = st.slider("Round 3: Recommendation Limit", min_value=2, max_value=10, value=int(st.session_state["recommendation_limit"]))
    
    st.session_state["coarse_limit"] = coarse_limit
    st.session_state["deep_limit"] = deep_limit
    st.session_state["recommendation_limit"] = recommendation_limit

st.sidebar.markdown("---")
st.sidebar.markdown("### 3. Active Requirements Constraints")

# Show current requirement constraints in UI controls to support direct modification
reqs = st.session_state["requirements"]

title_input = st.sidebar.text_input("Extracted Job Title", value=reqs.get("title", "Software Engineer"))
min_exp_slider = st.sidebar.slider("Min Experience Years", min_value=0, max_value=20, value=int(reqs.get("min_experience_years", 0)))
must_have_input = st.sidebar.text_area("Must-Have Skills (comma separated)", value=", ".join(reqs.get("must_have_skills", [])))
nice_have_input = st.sidebar.text_area("Nice-To-Have Skills (comma separated)", value=", ".join(reqs.get("nice_to_have_skills", [])))
education_level = st.sidebar.text_input("Education Level Target", value=reqs.get("education_level", "Not Specified"))

# If user modifies sidebar requirements directly, click here to sync and re-trigger match
if st.sidebar.button("Sync Constraints & Re-Rank"):
    updated_reqs = {
        "title": title_input,
        "must_have_skills": [s.strip() for s in must_have_input.split(",") if s.strip()],
        "nice_to_have_skills": [s.strip() for s in nice_have_input.split(",") if s.strip()],
        "min_experience_years": min_exp_slider,
        "education_level": education_level,
        "other_constraints": reqs.get("other_constraints", [])
    }
    st.session_state["requirements"] = updated_reqs
    
    # Run matching graph synchronously with updated constraints
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
        "user_feedback": "",
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "api_key": api_key,
        "api_url": api_url,
        "errors": []
    }
    
    with st.spinner("Re-ranking candidates based on updated constraints..."):
        result = matching_agent_workflow.invoke(state_input, config={"configurable": {"thread_id": "streamlit-session-thread"}})
        st.session_state["shortlist"] = result.get("shortlist", [])
        st.session_state["final_report"] = result.get("final_report", "")
        st.session_state["ranking_explanation"] = result.get("ranking_explanation", "")
        st.session_state["errors"] = result.get("errors", [])
        st.success("Shortlist re-ranked successfully!")


# ----------------------------------------------------
# Main Layout Workspace
# ----------------------------------------------------

st.markdown('<h1 class="app-header">💼 Agentic Profile Matching Engine</h1>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Your interactive AI assistant to search, rank, screen, and compare candidate resumes.</p>', unsafe_allow_html=True)

# Render active warnings/errors from the agent run
if st.session_state["errors"]:
    for err in st.session_state["errors"]:
        st.warning(f"⚠️ {err}")


tab1, tab2, tab3 = st.tabs(["💬 Chat Workspace", "📊 Shortlist & Comparison", "🔎 Deep Screening Reports"])

# TAB 1: Chat Workspace
with tab1:
    st.markdown("### Chat with Recruiter Assistant")
    st.caption("Paste a Job Description (JD) to extract requirements, or type conversational search and refinement commands.")

    # Render conversational chat log
    for msg in st.session_state["messages"]:
        if isinstance(msg, HumanMessage) or (hasattr(msg, "type") and msg.type == "human"):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage) or (hasattr(msg, "type") and msg.type == "ai"):
            with st.chat_message("assistant"):
                st.markdown(msg.content)

    # Chat input area
    user_query = st.chat_input("Enter message (e.g. 'Search resumes for React developers with 3+ years experience')")
    
    if user_query:
        # Display user input in UI immediately
        with st.chat_message("user"):
            st.markdown(user_query)
            
        # Append to message list
        st.session_state["messages"].append(HumanMessage(content=user_query))
        
        # Prepare graph state inputs
        state_input = {
            "messages": st.session_state["messages"],
            "requirements": st.session_state["requirements"],
            "shortlist": st.session_state["shortlist"],
            "coarse_screen_limit": st.session_state["coarse_limit"],
            "deep_screen_limit": st.session_state["deep_limit"],
            "recommendation_limit": st.session_state["recommendation_limit"],
            "current_round": 1,
            "final_report": st.session_state["final_report"],
            "feedback_pending": False,
            "user_feedback": "",
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "api_key": api_key,
            "api_url": api_url,
            "errors": []
        }
        
        # Execute workflow
        with st.spinner("Recruiter Agent is working..."):
            try:
                result = matching_agent_workflow.invoke(state_input, config={"configurable": {"thread_id": "streamlit-session-thread"}})
                
                # Copy updated state outputs to session state
                st.session_state["messages"] = result.get("messages", [])
                st.session_state["requirements"] = result.get("requirements", {})
                st.session_state["shortlist"] = result.get("shortlist", [])
                st.session_state["final_report"] = result.get("final_report", "")
                st.session_state["ranking_explanation"] = result.get("ranking_explanation", "")
                st.session_state["errors"] = result.get("errors", [])
                
                # Render assistant output
                with st.chat_message("assistant"):
                    if st.session_state["shortlist"]:
                        st.markdown(f"Analyzed candidates and successfully updated active requirements.")
                        if st.session_state["ranking_explanation"]:
                            st.info(st.session_state["ranking_explanation"])
                        st.markdown(f"**Top Candidates Shortlisted**: {', '.join(c['name'] for c in st.session_state['shortlist'][:3])}")
                    else:
                        st.markdown("Ingested input. Please verify the active requirements are updated.")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error executing agentic loop: {e}")
                st.session_state["messages"].append(AIMessage(content=f"Sorry, I encountered an error: {str(e)}"))

# TAB 2: Shortlist & Comparison Matrix
with tab2:
    st.markdown("### Candidate Shortlist Matrix")
    shortlist = st.session_state["shortlist"]
    
    if not shortlist:
        st.info("No candidates shortlisted yet. Paste a JD or write a search command in the Chat tab.")
    else:
        # Display side-by-side comparison table
        st.markdown("#### Head-to-Head Comparison Matrix")
        rec_limit = st.session_state["recommendation_limit"]
        candidate_ids = [c["candidate_id"] for c in shortlist[:int(rec_limit)]]
        compare_md = compare_candidates(candidate_ids, shortlist)
        st.markdown(compare_md)
        
        st.markdown("---")
        st.markdown("#### Ranked Candidate Shortlist")
        
        for idx, c in enumerate(shortlist):
            status = c.get('screening_status', 'Shortlisted')
            
            # Strict classification to avoid operator precedence / substring bugs (like 'no-hire' matching 'hire')
            if "reject" in status.lower() or "no-hire" in status.lower():
                status_class = "status-rejected"
            elif "borderline" in status.lower():
                status_class = "status-borderline"
            else:
                status_class = "status-strong"
            
            # Format skills
            skills_html = "".join(f'<span class="skill-tag">{s}</span>' for s in c.get('matched_skills', []))
            if not skills_html:
                skills_html = '<span class="skill-tag" style="opacity: 0.5;">None matched</span>'
                
            # Get path relative to the project root directory
            try:
                project_root = Path(config.BASE_DIR).parent
                rel_path = Path(c['candidate_id']).relative_to(project_root)
            except Exception:
                rel_path = Path(c['candidate_id']).name
                
            card_html = f"""
            <div class="candidate-card">
                <div class="card-header">
                    <div>
                        <span class="card-name">{c['name']}</span>
                        <span style="color: #9ca3af; margin-left: 8px; font-size: 0.9rem;">(Rank #{idx+1})</span>
                    </div>
                    <span class="score-badge">{c['score']}/100</span>
                </div>
                <div style="margin-bottom: 0.75rem;">
                    <span class="status-pill {status_class}">{status}</span>
                </div>
                <div style="display: flex; gap: 2rem; font-size: 0.9rem; margin-bottom: 0.75rem; color: #d1d5db;">
                    <div>💼 <b>Experience</b>: {c['experience_years']} Years</div>
                    <div>🎓 <b>Education</b>: {c['education']}</div>
                </div>
                <div style="margin-bottom: 0.5rem;">
                    <span style="font-size: 0.8rem; color: #9ca3af; font-weight: 600; display: block; margin-bottom: 0.25rem;">MATCHED SKILLS:</span>
                    {skills_html}
                </div>
                <div style="font-size: 0.78rem; color: #6b7280; word-break: break-all;">
                    📄 <i>Path: {rel_path}</i>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

# TAB 3: Deep Screening & Interview Qs
with tab3:
    st.markdown("### Deep Profile Audits")
    shortlist = st.session_state["shortlist"]
    
    if not shortlist:
        st.info("No candidate screening data available yet.")
    else:
        deep_limit = st.session_state["deep_limit"]
        for idx, c in enumerate(shortlist[:int(deep_limit)]):
            # Expander for each shortlisted candidate
            expander_title = f"{idx+1}. {c['name']} (Match Score: {c['score']}/100) — {c.get('screening_status', 'Shortlisted')}"
            with st.expander(expander_title):
                st.markdown(f"**Screening Reasoning**: {c.get('screening_reasoning', 'No deep reasoning generated.')}")
                
                cols = st.columns(2)
                with cols[0]:
                    st.markdown("**Core Strengths**:")
                    if c.get("strengths"):
                        st.markdown("\n".join(f"- {s}" for s in c["strengths"]))
                    else:
                        st.caption("No strengths evaluated yet.")
                with cols[1]:
                    st.markdown("**Identified Gaps**:")
                    if c.get("gaps"):
                        st.markdown("\n".join(f"- {g}" for g in c["gaps"]))
                    else:
                        st.caption("No gaps evaluated yet.")
                        
                st.markdown(f"**Improvement Suggestions**: *{c.get('improvement_suggestions', 'None')}*")
                
                # Show interview questions
                if c.get("interview_questions"):
                    st.markdown("#### Custom Interview Questions:")
                    for q in c["interview_questions"]:
                        st.markdown(f"- {q}")

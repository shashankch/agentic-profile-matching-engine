"""
Comparison Matrix & Ranked Shortlist Component for Yojaka AI UI.
Renders head-to-head candidate benchmarking tables and glassmorphic cards
with anti-XSS HTML sanitization and status classification chips.
"""

import html
from pathlib import Path
import streamlit as st

from agentic_profile_matching import config
from agentic_profile_matching.tools import compare_candidates


def render_matrix_tab() -> None:
    """Renders the Head-to-Head Comparison Matrix and ranked candidate profile cards."""
    st.markdown("### Candidate Shortlist Matrix")
    shortlist = st.session_state["shortlist"]

    if not shortlist:
        st.info("No candidates shortlisted yet. Paste a JD or write a search command in the Chat tab.")
        return

    # 1. Head-to-head comparison markdown table
    st.markdown("#### Head-to-Head Comparison Matrix")
    rec_limit = st.session_state["recommendation_limit"]
    candidate_ids = [c["candidate_id"] for c in shortlist[: int(rec_limit)]]
    compare_md = compare_candidates(candidate_ids, shortlist)
    st.markdown(compare_md)

    st.markdown("---")
    st.markdown("#### Ranked Candidate Shortlist")

    # 2. Render individual candidate cards
    for idx, c in enumerate(shortlist):
        status = c.get("screening_status", "Shortlisted")

        # Strict classification to avoid operator precedence / substring bugs
        if "reject" in status.lower() or "no-hire" in status.lower():
            status_class = "status-rejected"
        elif "borderline" in status.lower():
            status_class = "status-borderline"
        else:
            status_class = "status-strong"

        # Escape candidate values for security (anti-XSS)
        safe_name = html.escape(str(c.get("name", "Unknown")), quote=True)
        safe_status = html.escape(str(status), quote=True)
        safe_exp = html.escape(str(c.get("experience_years", 0)), quote=True)
        safe_edu = html.escape(str(c.get("education", "Not Specified")), quote=True)
        safe_score = html.escape(str(c.get("score", 0)), quote=True)

        # Format skills
        matched = c.get("matched_skills", [])
        if matched:
            skills_html = "".join(f'<span class="skill-tag">{html.escape(str(s), quote=True)}</span>' for s in matched)
        else:
            skills_html = '<span class="skill-tag" style="opacity: 0.5;">None matched</span>'

        # Get path relative to the project root directory
        try:
            project_root = Path(config.BASE_DIR).parent
            rel_path = str(Path(c["candidate_id"]).relative_to(project_root))
        except Exception:
            rel_path = Path(c.get("candidate_id", "")).name
        safe_rel_path = html.escape(rel_path, quote=True)

        card_html = f"""
        <div class="candidate-card">
            <div class="card-header">
                <div>
                    <span class="card-name">{safe_name}</span>
                    <span class="card-rank">(Rank #{idx + 1})</span>
                </div>
                <span class="score-badge">{safe_score}/100</span>
            </div>
            <div style="margin-bottom: 0.75rem;">
                <span class="status-pill {status_class}">{safe_status}</span>
            </div>
            <div class="card-meta">
                <div>💼 <b>Experience</b>: {safe_exp} Years</div>
                <div>🎓 <b>Education</b>: {safe_edu}</div>
            </div>
            <div style="margin-bottom: 0.5rem;">
                <span class="skills-label">MATCHED SKILLS:</span>
                {skills_html}
            </div>
            <div class="card-path">
                📄 <i>Path: {safe_rel_path}</i>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

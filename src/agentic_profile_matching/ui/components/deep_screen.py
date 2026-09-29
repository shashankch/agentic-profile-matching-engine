"""
Deep Screening Reports Component for Yojaka AI UI.
Renders granular candidate evaluations: LLM audit reasoning, core strengths,
verified skill gaps, improvement recommendations, and tailored interview questions.
"""

import streamlit as st


def render_deep_screen_tab() -> None:
    """Renders candidate deep audits and generated technical interview guides."""
    st.markdown("### Deep Profile Audits")
    shortlist = st.session_state["shortlist"]

    if not shortlist:
        st.info("No candidate screening data available yet. Run a search in the Chat tab.")
        return

    deep_limit = st.session_state["deep_limit"]
    for idx, c in enumerate(shortlist[: int(deep_limit)]):
        expander_title = (
            f"{idx + 1}. {c['name']} (Match Score: {c['score']}/100) — {c.get('screening_status', 'Shortlisted')}"
        )
        with st.expander(expander_title, expanded=(idx == 0)):
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

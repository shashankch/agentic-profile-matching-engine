"""
Recruiter Chat Workspace Component for Yojaka AI UI.
Supports multi-turn interactive requirements refinement, conversational search,
and real-time token-by-token streaming via st.write_stream().
"""

from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st

from agentic_profile_matching.ui.runner import run_agent_workflow_with_status, stream_text_chunks


def render_chat_tab(workflow_config: dict) -> None:
    """
    Renders the recruiter conversational workspace tab.
    Allows pasting JDs, conversational search, and live response streaming.
    """
    st.markdown("### Chat with Recruiter Assistant")
    st.caption(
        "Paste a Job Description (JD) to extract requirements, or type conversational search and refinement commands."
    )

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

        # Prepare graph state inputs (stateless credential isolation)
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
            "errors": [],
        }

        try:
            result = run_agent_workflow_with_status(state_input, workflow_config)

            # Copy updated state outputs to session state
            st.session_state["messages"] = result.get("messages", [])
            st.session_state["requirements"] = result.get("requirements", {})
            st.session_state["shortlist"] = result.get("shortlist", [])
            st.session_state["final_report"] = result.get("final_report", "")
            st.session_state["ranking_explanation"] = result.get("ranking_explanation", "")
            st.session_state["errors"] = result.get("errors", [])

            # Render assistant output with token-by-token streaming
            with st.chat_message("assistant"):
                # Determine conversational response vs candidate shortlist response
                latest_ai_msg = ""
                for m in reversed(st.session_state["messages"]):
                    if isinstance(m, AIMessage) or (hasattr(m, "type") and m.type == "ai"):
                        latest_ai_msg = m.content
                        break

                if st.session_state["shortlist"]:
                    summary_text = (
                        "Analyzed candidate profiles and successfully updated active requirements.\n\n"
                        f"**Top Candidates Shortlisted**: {', '.join(c['name'] for c in st.session_state['shortlist'][:3])}"
                    )
                    st.write_stream(stream_text_chunks(summary_text))
                    if st.session_state["ranking_explanation"]:
                        st.info(st.session_state["ranking_explanation"])
                elif latest_ai_msg:
                    # Stream the full conversational assistant response
                    st.write_stream(stream_text_chunks(latest_ai_msg))
                else:
                    st.write_stream(
                        stream_text_chunks("Ingested input. Please verify the active requirements are updated.")
                    )

            st.rerun()

        except Exception as e:
            st.error(f"Error executing agentic loop: {e}")
            st.session_state["messages"].append(AIMessage(content=f"Sorry, I encountered an error: {str(e)}"))

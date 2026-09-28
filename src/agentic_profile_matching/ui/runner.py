"""
Workflow Execution Runner for Yojaka AI UI.
Provides progressive status indicators, dynamic intent-aware headers,
intra-node tool execution visibility, and word-by-word streaming generation.
"""

from collections.abc import Generator
import time
from typing import Any
import streamlit as st

from agentic_profile_matching.matching_agent import matching_agent_workflow


def run_agent_workflow_with_status(state_input: dict, config_dict: dict) -> dict:
    """
    Executes the LangGraph profile matching workflow with progressive live checkpoints,
    dynamic intent-aware headers, and intra-node tool execution status updates.
    """
    # Initial status container
    status_header = "📋 Recruiter Agent is analyzing candidates..."
    with st.status(status_header, expanded=True) as status_box:
        result = dict(state_input)
        emitted_any = False
        try:
            for event in matching_agent_workflow.stream(state_input, config=config_dict, stream_mode="updates"):
                emitted_any = True
                for node_name, node_update in event.items():
                    if isinstance(node_update, dict):
                        result.update(node_update)

                        if node_name == "parse_input":
                            status_box.write("🔍 Parsing recruiter input and verifying conversation history...")

                        elif node_name == "extract_requirements":
                            title_ext = node_update.get("requirements", {}).get("title", "Software Engineer")
                            status_box.write(f"📋 Extracted job requirements: **{title_ext}**")

                        elif node_name == "adjust_requirements":
                            status_box.write("🔄 Adjusted skill constraints and experience thresholds...")

                        elif node_name == "search_resumes":
                            count = len(node_update.get("shortlist", []))
                            status_box.write(
                                f"📂 Retrieved **{count}** candidate profiles via hybrid BM25 + dense search..."
                            )

                        elif node_name == "rank_candidates":
                            count = len(node_update.get("shortlist", []))
                            status_box.write(f"📊 Ranked Top **{count}** candidate profiles...")

                        elif node_name == "deep_screen":
                            status_box.write("🔬 Completed parallel deep audits (strengths, gaps, reasoning)...")

                        elif node_name == "recommendation":
                            status_box.write("⚖️ Formulated hire decisions & custom interview questions...")

                        elif node_name == "generate_report":
                            status_box.write("📝 Compiled candidate comparison matrix and executive report...")

                        elif node_name == "conversational_query":
                            # Dynamic update for conversational / web query
                            status_box.update(
                                label="🌐 Researching external query & trends...",
                                state="running",
                                expanded=True,
                            )
                            # Display tool execution badge if web search was engaged
                            tavily_key = config_dict.get("configurable", {}).get("tavily_api_key")
                            provider_tag = "Tavily AI Search" if tavily_key else "Built-in Search Engine"
                            status_box.write(f"🌐 Executed real-time web intelligence query via **{provider_tag}**...")
                            status_box.write("💬 Formulated conversational response with citations...")

            status_box.update(label="Screening workflow complete!", state="complete", expanded=False)
            return result

        except Exception as ex:
            if not emitted_any:
                status_box.write(f"⚠️ Direct execution fallback ({ex})...")
                result = matching_agent_workflow.invoke(state_input, config=config_dict)
                status_box.update(label="Screening workflow complete!", state="complete", expanded=False)
                return result
            else:
                status_box.write(f"⚠️ Workflow stopped with partial error: {ex}")
                status_box.update(label="Screening completed with errors", state="error", expanded=False)
                return result


def stream_text_chunks(text: str, chunk_size: int = 4, delay: float = 0.015) -> Generator[str, Any, None]:
    """
    Simulates real-time token/word streaming for Streamlit's st.write_stream().
    Delivers a responsive, typewriter UX for synthesized assistant responses.
    """
    words = text.split(" ")
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        if i + chunk_size < len(words):
            chunk += " "
        yield chunk
        if delay > 0:
            time.sleep(delay)

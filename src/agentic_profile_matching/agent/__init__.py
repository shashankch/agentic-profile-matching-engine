from pathlib import Path
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agentic_profile_matching.agent.state import AgentState
from agentic_profile_matching.agent.routers import route_input
from agentic_profile_matching.agent.nodes import (
    parse_input_node,
    extract_requirements_node,
    search_resumes_node,
    rank_candidates_node,
    deep_screen_node,
    recommendation_node,
    generate_report_node,
    adjust_requirements_node,
    conversational_query_node,
)

# Build LangGraph Workflow
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
        "conversational_query": "conversational_query",
    },
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
memory = MemorySaver()
matching_agent_workflow = builder.compile(checkpointer=memory)


# Code for CLI diagram compilation execution helper
def generate_diagrams():
    print("Compiling LangGraph Workflow and generating state machine diagram...")
    docs_dir = Path(__file__).resolve().parent.parent.parent.parent / "docs"
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
        print(f"Note: Could not generate visual PNG diagram. Error: {e}")
        print(
            "Mermaid representation is still saved, which can be rendered in markdown."
        )

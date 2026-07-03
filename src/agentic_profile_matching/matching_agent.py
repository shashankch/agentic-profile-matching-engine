# Compatibility shim re-exporting from agent/ folder to prevent import statement breaks.

from agentic_profile_matching.agent.nodes import (
    parse_input_node as parse_input_node,
    extract_requirements_node as extract_requirements_node,
    search_resumes_node as search_resumes_node,
    rank_candidates_node as rank_candidates_node,
    deep_screen_node as deep_screen_node,
    recommendation_node as recommendation_node,
    generate_report_node as generate_report_node,
    adjust_requirements_node as adjust_requirements_node,
    conversational_query_node as conversational_query_node,
)
from agentic_profile_matching.agent.state import AgentState as AgentState
from agentic_profile_matching.agent import (
    matching_agent_workflow as matching_agent_workflow,
    generate_diagrams as generate_diagrams,
)

if __name__ == "__main__":
    generate_diagrams()

# Compatibility shim re-exporting from agent/ folder to prevent import statement breaks.

from agentic_profile_matching.agent.state import JobRequirements, CandidateMatch, AgentState
from agentic_profile_matching.agent.nodes import (
    parse_input_node,
    extract_requirements_node,
    search_resumes_node,
    rank_candidates_node,
    deep_screen_node,
    recommendation_node,
    generate_report_node,
    adjust_requirements_node,
    conversational_query_node
)
from agentic_profile_matching.agent import matching_agent_workflow, generate_diagrams

if __name__ == "__main__":
    generate_diagrams()

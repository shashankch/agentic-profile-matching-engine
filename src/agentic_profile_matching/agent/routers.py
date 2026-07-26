from agentic_profile_matching.agent.state import AgentState


def route_input(state: AgentState) -> str:
    """
    Decides whether to parse a new JD, update requirements, or answer conversational questions.
    """
    messages = state.get("messages", [])
    if not messages:
        return "extract_requirements"

    last_msg = messages[-1].content.lower()
    lines = [line.strip() for line in last_msg.split("\n") if line.strip()]
    is_jd = len(lines) > 3 or any(
        w in last_msg
        for w in ["job description", "requirements:", "duties:", "responsibilities:"]
    )

    if is_jd or not state.get("requirements"):
        return "extract_requirements"

    # Heuristics for comparison/explanation queries or search tasks
    conversational_keywords = [
        "why",
        "compare",
        "higher",
        "better",
        "explain",
        "vs",
        "versus",
        "who",
        "show",
        "tell me about",
        "search",
        "web",
        "internet",
        "news",
        "notes",
        "google",
    ]
    if any(kw in last_msg for kw in conversational_keywords):
        is_comparison = any(
            kw in last_msg
            for kw in [
                "why",
                "compare",
                "higher",
                "better",
                "explain",
                "vs",
                "versus",
                "who",
                "show",
            ]
        )
        if is_comparison:
            if len(state.get("shortlist", [])) > 0:
                return "conversational_query"
        else:
            return "conversational_query"

    return "adjust_requirements"

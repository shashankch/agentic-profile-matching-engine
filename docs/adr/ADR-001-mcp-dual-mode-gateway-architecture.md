# ADR-001: Model Context Protocol (MCP) Dual-Mode Gateway Architecture

## Status
Accepted (Implemented in `v0.6.0`)

## Context
The agent workflow requires interaction with local filesystem utilities (file parsing, directory listing) and external/internal search services. In production, tools may be hosted as decoupled, isolated microservices operating over standard protocol specifications. However, forcing all local CLI and unit testing workflows through subprocess JSON-RPC servers introduces unnecessary runtime setup overhead.

## Decision
Implement a **Dual-Mode Gateway Architecture** (`fs_client.py` and `mcp_client.py`) toggled dynamically via the `config.USE_MCP` boolean flag (`USE_MCP=True/False` in `.env` / `config.py`):
1. **Local Direct Mode (`USE_MCP=False`)**: Directly executes local Python modules (`fs_tools.py`, `job_matcher.py`) in-process for lightweight local development and rapid testing.
2. **MCP Protocol Mode (`USE_MCP=True`)**: Connects via `mcp.ClientSession` over `stdio` transport to standalone FastMCP protocol servers (`filesystem_mcp_server.py` and `search_mcp_server.py`), executing tools over standard JSON-RPC 2.0.

## Consequences
- **Positive**: Allows zero-overhead local development while guaranteeing full compliance with the Model Context Protocol specification for remote or containerized tool servers.
- **Negative**: Requires maintaining dual invocation pathways in `fs_client.py`.

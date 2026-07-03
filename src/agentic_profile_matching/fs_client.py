import logging
from typing import Dict, List, Optional

from agentic_profile_matching import config
from agentic_profile_matching import fs_tools
from agentic_profile_matching.mcp_client import mcp_client

logger = logging.getLogger("fs_client")


def read_file(filepath: str) -> Dict:
    """
    Read file content. Uses MCP if enabled, otherwise falls back to direct local execution.
    """
    if getattr(config, "USE_MCP", False):
        logger.info(f"[fs_client] Routing read_file to MCP filesystem server for path: {filepath}")
        return mcp_client.call_tool("filesystem", "read_file", {"filepath": filepath})
    
    logger.info(f"[fs_client] Routing read_file directly (local mode) for path: {filepath}")
    return fs_tools.read_file(filepath)


def list_files(directory: str, extension: Optional[str] = None) -> List[Dict]:
    """
    List files in directory. Uses MCP if enabled, otherwise falls back to direct local execution.
    """
    if getattr(config, "USE_MCP", False):
        logger.info(f"[fs_client] Routing list_files to MCP filesystem server for dir: {directory}")
        return mcp_client.call_tool("filesystem", "list_files", {"directory": directory, "extension": extension})
    
    logger.info(f"[fs_client] Routing list_files directly (local mode) for dir: {directory}")
    return fs_tools.list_files(directory, extension)


def write_file(filepath: str, content: str) -> Dict:
    """
    Write file content. Uses MCP if enabled, otherwise falls back to direct local execution.
    """
    if getattr(config, "USE_MCP", False):
        logger.info(f"[fs_client] Routing write_file to MCP filesystem server for path: {filepath}")
        return mcp_client.call_tool("filesystem", "write_file", {"filepath": filepath, "content": content})
    
    logger.info(f"[fs_client] Routing write_file directly (local mode) for path: {filepath}")
    return fs_tools.write_file(filepath, content)


def search_in_file(
    filepath: str,
    keyword: str,
    context_size: int = 150,
    limit: int = 10,
    offset: int = 0
) -> Dict:
    """
    Search text inside file. Uses MCP if enabled, otherwise falls back to direct local execution.
    """
    if getattr(config, "USE_MCP", False):
        logger.info(f"[fs_client] Routing search_in_file to MCP filesystem server for: {filepath}")
        return mcp_client.call_tool(
            "filesystem", 
            "search_in_file", 
            {
                "filepath": filepath,
                "keyword": keyword,
                "context_size": context_size,
                "limit": limit,
                "offset": offset
            }
        )
    
    logger.info(f"[fs_client] Routing search_in_file directly (local mode) for: {filepath}")
    return fs_tools.search_in_file(filepath, keyword, context_size, limit, offset)


# ----------------------------------------------------
# New Capabilities (Exposed through Gateway)
# ----------------------------------------------------

def watch_directory(directory: str) -> Dict:
    """
    Watch directory for new resumes.
    If MCP mode is active, it calls the watch tool on the MCP server.
    If in local mode, it starts a local background thread doing the same polling.
    """
    if getattr(config, "USE_MCP", False):
        logger.info(f"[fs_client] Routing watch_directory to MCP filesystem server: {directory}")
        return mcp_client.call_tool("filesystem", "watch_directory", {"directory": directory})
    
    logger.info(f"[fs_client] Starting local watch_directory background monitor (local mode) for: {directory}")
    # Reuse the watch logic by importing from the server module dynamically, or spawning a thread locally
    from agentic_profile_matching.filesystem_mcp_server import watch_directory as server_watch
    return server_watch(directory)


def batch_process(filepaths: List[str]) -> List[Dict]:
    """
    Process multiple files concurrently.
    If MCP mode is active, it delegates to the MCP server.
    If in local mode, it runs concurrently locally.
    """
    if getattr(config, "USE_MCP", False):
        logger.info(f"[fs_client] Routing batch_process to MCP filesystem server for {len(filepaths)} files")
        res = mcp_client.call_tool("filesystem", "batch_process", {"filepaths": filepaths})
        if isinstance(res, dict):
            return [res]
        return res
    
    logger.info(f"[fs_client] Running local batch_process concurrently (local mode) for {len(filepaths)} files")
    from agentic_profile_matching.filesystem_mcp_server import batch_process as server_batch
    return server_batch(filepaths)

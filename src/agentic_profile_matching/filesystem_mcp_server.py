import sys
import logging
import time
import threading
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

from mcp.server.fastmcp import FastMCP

# Add parent directory to sys.path to allow running as script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic_profile_matching import config
from agentic_profile_matching.fs_tools import read_file as direct_read_file
from agentic_profile_matching.fs_tools import list_files as direct_list_files
from agentic_profile_matching.fs_tools import write_file as direct_write_file
from agentic_profile_matching.fs_tools import search_in_file as direct_search_in_file
from agentic_profile_matching.services.ingestion_service import IngestionService

# Configure logging to stderr for MCP compliance
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger("filesystem_mcp_server")

# Initialize FastMCP Server
mcp = FastMCP(
    "FileSystemServer",
    dependencies=["pypdf", "python-docx", "mcp"]
)

# Active directory watcher threads tracking
_active_watchers = {}
_watcher_lock = threading.Lock()

# ----------------------------------------------------
# Milestone 1 Tools Ported to MCP
# ----------------------------------------------------

@mcp.tool()
def read_file(filepath: str) -> Dict:
    """
    Read content and metadata from a PDF, TXT, or DOCX file.
    
    Args:
        filepath: Absolute path to the file.
    """
    logger.info(f"MCP Tool call: read_file for {filepath}")
    return direct_read_file(filepath)


@mcp.tool()
def list_files(directory: str, extension: Optional[str] = None) -> List[Dict]:
    """
    List all files in a directory, optionally filtered by file extension.
    
    Args:
        directory: Absolute path to the directory.
        extension: File extension to filter by (e.g. '.pdf', '.txt', '.docx').
    """
    logger.info(f"MCP Tool call: list_files in {directory} (filter: {extension})")
    return direct_list_files(directory, extension)


@mcp.tool()
def write_file(filepath: str, content: str) -> Dict:
    """
    Write content to a file, creating parent directories if they do not exist.
    
    Args:
        filepath: Absolute path to target file.
        content: Text content to write.
    """
    logger.info(f"MCP Tool call: write_file to {filepath}")
    return direct_write_file(filepath, content)


@mcp.tool()
def search_in_file(
    filepath: str,
    keyword: str,
    context_size: int = 150,
    limit: int = 10,
    offset: int = 0
) -> Dict:
    """
    Perform a case-insensitive keyword search in a file with context excerpts and pagination.
    
    Args:
        filepath: Absolute path to the file.
        keyword: Search keyword/phrase.
        context_size: Surrounding characters to return as context around matches.
        limit: Max number of matches to return.
        offset: Pagination offset.
    """
    logger.info(f"MCP Tool call: search_in_file in {filepath} for '{keyword}'")
    return direct_search_in_file(filepath, keyword, context_size, limit, offset)


# ----------------------------------------------------
# New MCP-Specific Capabilities
# ----------------------------------------------------

@mcp.tool()
def batch_process(filepaths: List[str]) -> List[Dict]:
    """
    Concurrently read and parse multiple files (PDF/TXT/DOCX) using a thread pool.
    
    Args:
        filepaths: List of absolute file paths to process.
    """
    logger.info(f"MCP Tool call: batch_process on {len(filepaths)} files")
    
    def process_one(path: str) -> Dict:
        try:
            return direct_read_file(path)
        except Exception as e:
            return {"success": False, "filepath": path, "error": str(e)}

    # Production-ready concurrency using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(len(filepaths), 8)) as executor:
        results = list(executor.map(process_one, filepaths))
        
    return results


def _directory_watcher_worker(directory: str, poll_interval: float = 2.0):
    """Background thread polling directory for new resumes and auto-ingesting them."""
    logger.info(f"Directory watcher started for {directory}")
    path = Path(directory)
    
    # Initialize cache of existing files and modification times
    seen_files = {}
    try:
        if path.exists():
            for f in path.rglob("*"):
                if f.is_file() and f.suffix.lower() in {".txt", ".pdf", ".docx"}:
                    seen_files[str(f.resolve())] = f.stat().st_mtime
    except Exception as e:
        logger.error(f"Error initializing watcher cache: {e}")
        
    ingestion_service = None
    
    while True:
        with _watcher_lock:
            # Check if this watcher has been cancelled/stopped
            if directory not in _active_watchers or not _active_watchers[directory]["active"]:
                logger.info(f"Directory watcher stopped for {directory}")
                break
                
        try:
            if not path.exists():
                time.sleep(poll_interval)
                continue
                
            for f in path.rglob("*"):
                if not f.is_file():
                    continue
                if f.suffix.lower() not in {".txt", ".pdf", ".docx"}:
                    continue
                    
                f_abs = str(f.resolve())
                mtime = f.stat().st_mtime
                
                # If file is new or modified
                if f_abs not in seen_files or mtime > seen_files[f_abs]:
                    is_new = f_abs not in seen_files
                    seen_files[f_abs] = mtime
                    
                    event_type = "New" if is_new else "Modified"
                    logger.info(f"[Watchdog] {event_type} resume detected: {f.name}")
                    
                    # Auto-ingest via IngestionService boundary
                    try:
                        if ingestion_service is None:
                            ingestion_service = IngestionService()
                        logger.info(f"[Watchdog] Triggering auto-ingestion for {f.name}")
                        result = ingestion_service.ingest_file(f_abs)
                        logger.info(f"[Watchdog] Auto-ingestion completed for {f.name}: success={result.get('success')}")
                    except Exception as ex:
                        logger.error(f"[Watchdog] RAG auto-ingestion failed for {f.name}: {ex}")
                        
        except Exception as e:
            logger.error(f"Error in directory watcher polling loop: {e}")
            
        time.sleep(poll_interval)


@mcp.tool()
def watch_directory(directory: str) -> Dict:
    """
    Monitor a directory for new or modified resumes (PDF/TXT/DOCX) and trigger RAG auto-ingestion.
    
    Args:
        directory: Absolute path of the directory to monitor.
    """
    logger.info(f"MCP Tool call: watch_directory for {directory}")
    path = Path(directory)
    if not path.exists() or not path.is_dir():
        return {"success": False, "error": f"Path is not an existing directory: {directory}"}
        
    abs_dir = str(path.resolve())
    
    with _watcher_lock:
        if abs_dir in _active_watchers and _active_watchers[abs_dir]["active"]:
            return {"success": True, "message": f"Already actively watching directory: {abs_dir}"}
            
        # Start background watcher thread
        _active_watchers[abs_dir] = {"active": True}
        thread = threading.Thread(
            target=_directory_watcher_worker,
            args=(abs_dir,),
            daemon=True
        )
        thread.start()
        
    return {"success": True, "message": f"Successfully started watching directory: {abs_dir}"}


# ----------------------------------------------------
# Resource Discovery Endpoints (resumes:// namespace)
# ----------------------------------------------------

@mcp.resource("resumes://{filename}")
def get_resume_resource(filename: str) -> str:
    """
    Retrieve the text content of a candidate resume.
    
    Args:
        filename: Name of the resume file (e.g. resume_john_doe.pdf).
    """
    logger.info(f"MCP Resource fetch: resumes://{filename}")
    resumes_dir = Path(config.RESUMES_DIR)
    filepath = resumes_dir / filename
    
    if not filepath.exists():
        logger.error(f"Resource not found: {filepath}")
        raise FileNotFoundError(f"Resume file not found: {filename}")
        
    result = direct_read_file(str(filepath))
    if not result.get("success"):
        logger.error(f"Failed to read resource {filename}: {result.get('error')}")
        raise IOError(f"Failed to read resume: {result.get('error')}")
        
    return result["content"]


if __name__ == "__main__":
    logger.info("Starting FastMCP FileSystemServer...")
    mcp.run()

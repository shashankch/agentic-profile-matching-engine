import os
import sys
import logging
from pathlib import Path
from typing import Dict

from mcp.server.fastmcp import FastMCP

import warnings

# Suppress duckduckgo_search library renaming runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*duckduckgo_search.*")

# Add parent directory to sys.path to allow running as script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure logging to stderr for MCP compliance
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("search_mcp_server")

# Initialize FastMCP Server for Search
mcp = FastMCP("SearchServer")


@mcp.tool()
def search_web(query: str) -> Dict:
    """
    Search the web for candidate portfolios, Github repositories, or professional profiles.
    Returns live DuckDuckGo search results, with structured mock fallbacks for training portfolios.

    Args:
        query: Query string.
    """
    logger.info(f"MCP Search Tool call: search_web for '{query}'")
    query_lower = query.lower()

    # Standard mock search responses for training datasets
    mock_results = []

    if "marcus" in query_lower:
        mock_results.append(
            {
                "title": "Marcus Aurelius - Backend Portfolio",
                "url": "https://github.com/marcus-aurelius-dev",
                "snippet": "Lead Backend Engineer. Project repositories: distributed locking systems in Go, custom TCP proxies, and high-volume data ingest pipelines.",
            }
        )
    if "john" in query_lower:
        mock_results.append(
            {
                "title": "John Doe - Github Portfolio",
                "url": "https://github.com/johndoe-dev",
                "snippet": "Full stack engineer. Repositories include microservices architecture, react templates, and fastapi backend tools. Python/TypeScript.",
            }
        )
    if "jane" in query_lower:
        mock_results.append(
            {
                "title": "Jane Smith - Professional Blog",
                "url": "https://janesmith.io",
                "snippet": "Senior Frontend developer portfolio and technical blog. Articles on React performance tuning, styling systems, and NextJS architectures.",
            }
        )
    if "shashank" in query_lower:
        mock_results.append(
            {
                "title": "Shashank Chandel - Technical Portfolio",
                "url": "https://github.com/shashankch",
                "snippet": "Shashank has 5+ years of experience as a backend engineer specializing in distributed systems. Key skills include Java, Spring Boot, Kafka, SQL, Python, GenAI, Cloud technologies, Docker, and Kubernetes. Proficient in building scalable backend solutions. AI Focused backend engineer.",
            }
        )

    # Attempt actual live search using DuckDuckGo if no mock results found
    live_results = []
    if not mock_results and not os.getenv("SKIP_LIVE_SEARCH"):
        try:
            from duckduckgo_search import DDGS

            with DDGS(timeout=5) as ddgs:
                # Query top 5 live results keylessly
                ddgs_results = list(ddgs.text(query, max_results=5))
                for r in ddgs_results:
                    live_results.append(
                        {
                            "title": r.get("title", ""),
                            "url": r.get("href", ""),
                            "snippet": r.get("body", ""),
                        }
                    )
        except Exception as e:
            logger.warning(f"Live DuckDuckGo search failed or rate-limited: {e}")

    # Combine results, prioritizing candidate mock results for standard sandbox files
    all_results = mock_results + live_results

    # If no results matched, provide a fallback search entry
    if not all_results:
        all_results.append(
            {
                "title": f"Mock Web search results for: {query}",
                "url": "https://search-mock.com/results",
                "snippet": f"Found reference links for {query} on LinkedIn and GitHub. No specific portfolios resolved.",
            }
        )

    return {"query": query, "success": True, "results": all_results}


@mcp.tool()
def search_chroma_db(query: str, limit: int = 5) -> Dict:
    """
    Search the local candidate resume ChromaDB vector store.

    Args:
        query: Semantic query text (e.g., 'React developer', 'AWS microservices').
        limit: Max number of chunk results to return.
    """
    logger.info(f"MCP Search Tool call: search_chroma_db for '{query}'")
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        from agentic_profile_matching import config

        # Check persistent client and collection first
        client = chromadb.PersistentClient(path=config.VECTOR_DB_PATH)
        collection_name = "resumes"
        try:
            collection = client.get_collection(collection_name)
        except Exception:
            return {
                "success": False,
                "error": f"Collection '{collection_name}' not found. Ensure resumes are ingested.",
            }

        # Load embedding model only after confirming collection exists
        embedder = SentenceTransformer(config.EMBEDDING_MODEL)

        # Run semantic search
        query_emb = embedder.encode(query).tolist()
        results = collection.query(query_embeddings=[query_emb], n_results=limit)

        formatted_results = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]
            ids = results["ids"][0]

            for idx in range(len(docs)):
                # Calculate similarity score
                sim = 1.0 - (distances[idx] / 2.0)
                sim_score = max(0, min(100, int(sim * 100)))

                formatted_results.append(
                    {
                        "id": ids[idx],
                        "candidate_name": metas[idx].get("candidate_name", "Unknown"),
                        "section": metas[idx].get("section", "GENERAL"),
                        "score": sim_score,
                        "excerpt": docs[idx][:300] + ("..." if len(docs[idx]) > 300 else ""),
                        "resume_path": metas[idx].get("resume_path", ""),
                    }
                )

        return {"query": query, "success": True, "results": formatted_results}
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def fetch_candidate_notes(candidate_name: str) -> Dict:
    """
    Retrieve mock HR coordinator screening notes for a specific candidate name.

    Args:
        candidate_name: The first or full name of the candidate.
    """
    logger.info(f"MCP Search Tool call: fetch_candidate_notes for '{candidate_name}'")
    name_lower = candidate_name.lower()

    notes = "No HR notes found for this candidate. Profile is new to the applicant tracking system."

    if "john" in name_lower:
        notes = "John Doe: Screened by recruiter on 2026-06-15. Excellent backend understanding, strong communication, but salary expectations are on the higher end."
    elif "jane" in name_lower:
        notes = "Jane Smith: Passed initial phone screening. Strong UX/Frontend designs. Indicated interest in hybrid working model."
    elif "shashank" in name_lower:
        notes = "Shashank Chandel: Technical screen completed on 2026-07-01. Outstanding design skills, designed complete agentic framework architectures. Outstanding fit."

    return {"candidate_name": candidate_name, "success": True, "notes": notes}


if __name__ == "__main__":
    logger.info("Starting FastMCP SearchServer...")
    mcp.run()

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file at startup
load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", str(BASE_DIR / "chroma_db"))
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
RESUMES_DIR = os.getenv("RESUMES_DIR", str(DATA_DIR / "resumes"))
JOB_DESCRIPTIONS_DIR = os.getenv("JOB_DESCRIPTIONS_DIR", str(DATA_DIR / "job_descriptions"))

# Embedding Config
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "10"))

# Supported LLM Models & Providers
SUPPORTED_PROVIDERS = {
    "Groq": [
        "openai/gpt-oss-120b",
        "qwen/qwen3.6-27b",
        "llama-3.3-70b-versatile",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
    ],
    "Gemini": ["gemini-1.5-pro", "gemini-1.5-flash"],
    "OpenAI": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    "Custom (OpenAI-compatible)": ["custom-model"],
}

DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "Groq")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai/gpt-oss-120b")
THROTTLE_DELAY = float(os.getenv("THROTTLE_DELAY", "1.5"))  # Delay in seconds to prevent hitting LLM rate limits

# Default Candidate Processing Limits
DEFAULT_COARSE_LIMIT = int(os.getenv("DEFAULT_COARSE_LIMIT", "10"))
DEFAULT_DEEP_LIMIT = int(os.getenv("DEFAULT_DEEP_LIMIT", "10"))
DEFAULT_RECOMMENDATION_LIMIT = int(os.getenv("DEFAULT_RECOMMENDATION_LIMIT", "5"))
RESUME_TRUNCATION_LIMIT = int(os.getenv("RESUME_TRUNCATION_LIMIT", "12000"))

# MCP Protocol Configuration
USE_MCP = os.getenv("USE_MCP", "False").lower() in ("true", "1", "yes")
FILESYSTEM_SERVER_PATH = os.getenv(
    "FILESYSTEM_SERVER_PATH",
    str(BASE_DIR / "agentic_profile_matching" / "filesystem_mcp_server.py"),
)
SEARCH_SERVER_PATH = os.getenv(
    "SEARCH_SERVER_PATH",
    str(BASE_DIR / "agentic_profile_matching" / "search_mcp_server.py"),
)
MCP_TIMEOUT = float(os.getenv("MCP_TIMEOUT", "30.0"))

# Celery & Redis Async Task Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)


def get_llm_model(provider: str, model_name: str, api_key: str, api_url: Optional[str] = None):
    if provider == "Groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model_name, api_key=api_key)
    elif provider == "Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
    elif provider == "OpenAI":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, api_key=api_key)
    elif provider == "Custom (OpenAI-compatible)":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, api_key=api_key, base_url=api_url)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

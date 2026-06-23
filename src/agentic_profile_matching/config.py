import os
from pathlib import Path
from typing import Optional, List, Dict

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DB_PATH = str(BASE_DIR / "chroma_db")
DATA_DIR = BASE_DIR / "data"
RESUMES_DIR = str(DATA_DIR / "resumes")
JOB_DESCRIPTIONS_DIR = str(DATA_DIR / "job_descriptions")

# Embedding Config
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 10

# Supported LLM Models & Providers
SUPPORTED_PROVIDERS = {
    "Groq": [
        "llama-3.3-70b-versatile",
        "llama3-70b-8192",
        "mixtral-8x7b-32768"
    ],
    "Gemini": [
        "gemini-1.5-pro",
        "gemini-1.5-flash"
    ],
    "OpenAI": [
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-3.5-turbo"
    ],
    "Custom (OpenAI-compatible)": [
        "custom-model"
    ]
}

DEFAULT_PROVIDER = "Groq"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
THROTTLE_DELAY = 1.5  # Delay in seconds to prevent hitting LLM rate limits

# Default Candidate Processing Limits
DEFAULT_COARSE_LIMIT = 10
DEFAULT_DEEP_LIMIT = 10
DEFAULT_RECOMMENDATION_LIMIT = 5


def get_llm_model(
    provider: str,
    model_name: str,
    api_key: str,
    api_url: Optional[str] = None
):
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

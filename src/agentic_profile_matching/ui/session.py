"""
Session State and Resource Management for Yojaka AI UI.
Manages lazy-loaded vector stores, ephemeral session-scoped memory collections,
and centralized session state initialization.
"""

from pathlib import Path
import uuid
import streamlit as st

from agentic_profile_matching import config
from agentic_profile_matching.stores import ChromaVectorStore, CompositeVectorStore


@st.cache_resource(show_spinner=False)
def ensure_vector_store_initialized() -> ChromaVectorStore:
    """
    Lazy-loads and auto-bootstraps the persistent baseline candidate vector database.
    Decoupled from heavy model imports when existing vectors exist for 0.03s instant cold starts.
    """
    store = ChromaVectorStore()
    if store.count() == 0:
        from agentic_profile_matching.services.ingestion_service import IngestionService

        resumes_dir = Path(config.RESUMES_DIR)
        if not resumes_dir.exists():
            resumes_dir = Path(config.BASE_DIR).parent / "data" / "resumes"
        if not resumes_dir.exists():
            resumes_dir = Path(config.BASE_DIR) / "data" / "resumes"
        if resumes_dir.exists():
            service = IngestionService(store=store)
            service.ingest_directory(str(resumes_dir))
    return store


def init_session_state() -> None:
    """Initializes all required Streamlit session state keys with robust defaults."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "requirements" not in st.session_state:
        st.session_state["requirements"] = {
            "title": "Software Engineer",
            "must_have_skills": [],
            "nice_to_have_skills": [],
            "min_experience_years": 0,
            "education_level": "Not Specified",
            "other_constraints": [],
        }
    if "shortlist" not in st.session_state:
        st.session_state["shortlist"] = []
    if "final_report" not in st.session_state:
        st.session_state["final_report"] = ""
    if "ranking_explanation" not in st.session_state:
        st.session_state["ranking_explanation"] = ""
    if "coarse_limit" not in st.session_state:
        st.session_state["coarse_limit"] = config.DEFAULT_COARSE_LIMIT
    if "deep_limit" not in st.session_state:
        st.session_state["deep_limit"] = config.DEFAULT_DEEP_LIMIT
    if "recommendation_limit" not in st.session_state:
        st.session_state["recommendation_limit"] = config.DEFAULT_RECOMMENDATION_LIMIT
    if "errors" not in st.session_state:
        st.session_state["errors"] = []
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = uuid.uuid4().hex[:8]

    # Initialize base and session vector stores
    base_store = ensure_vector_store_initialized()
    if "ephemeral_store" not in st.session_state:
        try:
            st.session_state["ephemeral_store"] = ChromaVectorStore(
                collection_name=f"uploads_{st.session_state['session_id']}",
                ephemeral=True,
            )
        except Exception:
            from agentic_profile_matching.stores.in_memory_store import InMemoryVectorStore

            st.session_state["ephemeral_store"] = InMemoryVectorStore(
                collection_name=f"uploads_{st.session_state['session_id']}"
            )
    if "session_vector_store" not in st.session_state:
        st.session_state["session_vector_store"] = CompositeVectorStore(
            base_store=base_store,
            ephemeral_store=st.session_state["ephemeral_store"],
        )
    if "uploaded_candidates" not in st.session_state:
        st.session_state["uploaded_candidates"] = []
    if "processed_uploads" not in st.session_state:
        st.session_state["processed_uploads"] = set()


def reset_session_state() -> None:
    """Clears candidate shortlist, reports, and messages while preserving configuration."""
    st.session_state["messages"] = []
    st.session_state["shortlist"] = []
    st.session_state["final_report"] = ""
    st.session_state["ranking_explanation"] = ""
    st.session_state["errors"] = []

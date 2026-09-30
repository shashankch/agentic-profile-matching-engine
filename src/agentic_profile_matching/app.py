# Streamlit Cloud / Linux SQLite compatibility shim (requires sqlite3 >= 3.35.0 for ChromaDB)
try:
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

from dotenv import load_dotenv
import streamlit as st

from agentic_profile_matching.ui.styles import apply_custom_styles
from agentic_profile_matching.ui.session import init_session_state
from agentic_profile_matching.ui.components import (
    render_sidebar,
    render_chat_tab,
    render_talent_pool_tab,
    render_matrix_tab,
    render_deep_screen_tab,
)

# Load environment configuration
load_dotenv()

# 1. Setup page config
st.set_page_config(
    page_title="Yojaka AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inject responsive tokens and glassmorphic styles
apply_custom_styles()

# 3. Bootstrap session state and vector store handles
init_session_state()

# 4. Render configuration sidebar and extract workflow parameters
workflow_config = render_sidebar()

# 5. Render Main Layout Workspace Header
st.markdown(
    '<h1 class="app-header">💼 Yojaka AI (Agentic Profile Matching Engine)</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="app-subtitle">'
    "Autonomous multi-agent intelligence to extract, hybrid-search, screen, and benchmark talent profiles."
    "</p>",
    unsafe_allow_html=True,
)

# 6. Render active warnings/errors from agent runs
if st.session_state["errors"]:
    st.session_state["errors"] = [e for e in st.session_state["errors"] if "File not found" not in e]
    for err in st.session_state["errors"]:
        st.warning(f"⚠️ {err}")

# 7. Render 4 Core Workspace Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "💬 Chat Workspace",
        "📤 Resume Ingestion & Talent Pool",
        "📊 Shortlist & Comparison",
        "🔬 Deep Screening Reports",
    ]
)

with tab1:
    render_chat_tab(workflow_config)

with tab2:
    render_talent_pool_tab()

with tab3:
    render_matrix_tab()

with tab4:
    render_deep_screen_tab()

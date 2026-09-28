"""
Unit tests for Yojaka AI UI modular components.
Tests styles, session state bootstrap, streaming text generators, and tab component rendering.
"""

from unittest.mock import MagicMock, patch

from agentic_profile_matching.ui.styles import CUSTOM_CSS, THEME_DETECTOR_JS, apply_custom_styles
from agentic_profile_matching.ui.session import init_session_state, reset_session_state
from agentic_profile_matching.ui.runner import stream_text_chunks
from agentic_profile_matching.ui.components.matrix import render_matrix_tab
from agentic_profile_matching.ui.components.deep_screen import render_deep_screen_tab


def test_styles_definitions():
    """Verifies that style templates contain expected glassmorphic and theme tokens."""
    assert "candidate-card" in CUSTOM_CSS
    assert "--c-card-bg" in CUSTOM_CSS
    assert "Outfit" in CUSTOM_CSS
    assert "data-theme" in THEME_DETECTOR_JS


@patch("streamlit.markdown")
@patch("streamlit.html")
def test_apply_custom_styles(mock_html, mock_markdown):
    """Verifies apply_custom_styles executes markdown and html injection."""
    apply_custom_styles()
    mock_markdown.assert_called_once_with(CUSTOM_CSS, unsafe_allow_html=True)
    mock_html.assert_called_once_with(THEME_DETECTOR_JS)


def test_session_state_initialization():
    """Verifies init_session_state populates all necessary default state keys."""
    mock_session = {}
    with patch("streamlit.session_state", mock_session):
        with patch("agentic_profile_matching.ui.session.ensure_vector_store_initialized") as mock_ensure:
            mock_store = MagicMock()
            mock_ensure.return_value = mock_store

            init_session_state()

            assert "messages" in mock_session
            assert "requirements" in mock_session
            assert "shortlist" in mock_session
            assert "final_report" in mock_session
            assert "coarse_limit" in mock_session
            assert "deep_limit" in mock_session
            assert "recommendation_limit" in mock_session
            assert "ephemeral_store" in mock_session
            assert "session_vector_store" in mock_session
            assert "session_id" in mock_session


def test_reset_session_state():
    """Verifies reset_session_state clears conversation and reports while preserving limits."""
    mock_session = {
        "messages": ["test message"],
        "shortlist": [{"name": "Test Candidate"}],
        "final_report": "Some report",
        "ranking_explanation": "Some explanation",
        "errors": ["error 1"],
        "coarse_limit": 10,
    }
    with patch("streamlit.session_state", mock_session):
        reset_session_state()
        assert mock_session["messages"] == []
        assert mock_session["shortlist"] == []
        assert mock_session["final_report"] == ""
        assert mock_session["ranking_explanation"] == ""
        assert mock_session["errors"] == []
        assert mock_session["coarse_limit"] == 10


def test_stream_text_chunks():
    """Verifies stream_text_chunks generates words in typewriter chunks."""
    sample_text = "Senior Python Developer with 5 years experience"
    chunks = list(stream_text_chunks(sample_text, chunk_size=2, delay=0.0))
    reconstructed = "".join(chunks).strip()
    assert reconstructed == sample_text
    assert len(chunks) > 1


@patch("streamlit.info")
def test_render_matrix_tab_empty(mock_info):
    """Verifies render_matrix_tab displays info banner when shortlist is empty."""
    mock_session = {"shortlist": []}
    with patch("streamlit.session_state", mock_session):
        render_matrix_tab()
        mock_info.assert_called_once()


@patch("streamlit.info")
def test_render_deep_screen_tab_empty(mock_info):
    """Verifies render_deep_screen_tab displays info banner when shortlist is empty."""
    mock_session = {"shortlist": []}
    with patch("streamlit.session_state", mock_session):
        render_deep_screen_tab()
        mock_info.assert_called_once()


@patch("streamlit.markdown")
def test_render_matrix_tab_with_candidates(mock_markdown):
    """Verifies render_matrix_tab properly renders candidate cards and escapes HTML."""
    mock_session = {
        "shortlist": [
            {
                "candidate_id": "data/resumes/resume_alice.pdf",
                "name": "Alice Smith <script>alert(1)</script>",
                "score": 95,
                "matched_skills": ["Python", "FastAPI"],
                "experience_years": 6,
                "education": "BS Computer Science",
                "screening_status": "Strong Hire",
            }
        ],
        "recommendation_limit": 3,
    }
    with patch("streamlit.session_state", mock_session):
        with patch("agentic_profile_matching.ui.components.matrix.compare_candidates", return_value="| Matrix |"):
            render_matrix_tab()
            # Verify compare markdown and candidate card were rendered
            assert mock_markdown.called
            calls = [str(c) for c in mock_markdown.call_args_list]
            # Ensure script tags are escaped
            assert any("&lt;script&gt;" in c for c in calls)


@patch("streamlit.sidebar.selectbox")
@patch("streamlit.sidebar.text_input")
@patch("streamlit.sidebar.slider")
@patch("streamlit.sidebar.text_area")
@patch("streamlit.sidebar.button")
def test_render_sidebar(mock_btn, mock_textarea, mock_slider, mock_textinput, mock_selectbox):
    """Verifies render_sidebar constructs workflow_config and manages limits."""
    from agentic_profile_matching.ui.components.sidebar import render_sidebar

    mock_selectbox.side_effect = ["Groq", "llama-3.3-70b-versatile"]
    mock_textinput.side_effect = ["dummy-key", ""]
    mock_slider.side_effect = [10, 5, 3, 3]
    mock_textarea.side_effect = ["Python", "Docker"]
    mock_btn.return_value = False

    mock_session = {
        "coarse_limit": 10,
        "deep_limit": 5,
        "recommendation_limit": 3,
        "requirements": {
            "title": "Engineer",
            "min_experience_years": 0,
            "must_have_skills": [],
            "nice_to_have_skills": [],
        },
        "session_vector_store": MagicMock(),
        "uploaded_candidates": [],
    }
    with patch("streamlit.session_state", mock_session):
        cfg = render_sidebar()
        assert "configurable" in cfg
        assert cfg["configurable"]["llm_provider"] == "Groq"


@patch("streamlit.chat_input", return_value=None)
def test_render_chat_tab_display(mock_chat_input):
    """Verifies render_chat_tab renders message bubbles without crashing."""
    from langchain_core.messages import HumanMessage, AIMessage
    from agentic_profile_matching.ui.components.chat import render_chat_tab

    mock_session = {
        "messages": [HumanMessage(content="Hello"), AIMessage(content="Hi there!")],
        "shortlist": [],
    }
    with patch("streamlit.session_state", mock_session):
        with patch("streamlit.chat_message") as mock_chat_msg:
            render_chat_tab({"configurable": {}})
            assert mock_chat_msg.called


@patch("streamlit.file_uploader", return_value=None)
def test_render_talent_pool_tab(mock_uploader):
    """Verifies render_talent_pool_tab displays inventory metrics."""
    from agentic_profile_matching.ui.components.talent_pool import render_talent_pool_tab

    mock_store = MagicMock()
    mock_store.count.return_value = 34
    mock_session = {
        "session_vector_store": mock_store,
        "uploaded_candidates": [
            {
                "candidate_name": "Test",
                "filename": "test.pdf",
                "format": "PDF",
                "cohort": "Upload",
                "status": "Ready",
            }
        ],
    }
    with patch("streamlit.session_state", mock_session):
        render_talent_pool_tab()
        assert mock_store.count.called


def test_runner_workflow_status():
    """Verifies run_agent_workflow_with_status executes streaming events and updates UI status box."""
    from agentic_profile_matching.ui.runner import run_agent_workflow_with_status

    mock_events = [
        {"parse_input": {}},
        {"extract_requirements": {"requirements": {"title": "Architect"}}},
        {"adjust_requirements": {}},
        {"search_resumes": {"shortlist": [{"name": "Cand1"}]}},
        {"rank_candidates": {"shortlist": [{"name": "Cand1"}]}},
        {"deep_screen": {}},
        {"recommendation": {}},
        {"generate_report": {}},
        {"conversational_query": {}},
    ]
    with patch(
        "agentic_profile_matching.matching_agent.matching_agent_workflow.stream",
        return_value=iter(mock_events),
    ):
        with patch("streamlit.status") as mock_status:
            mock_box = MagicMock()
            mock_status.return_value.__enter__.return_value = mock_box
            state_input = {"messages": [], "shortlist": []}
            result = run_agent_workflow_with_status(state_input, {"configurable": {}})
            assert mock_box.write.called
            assert mock_box.update.called
            assert "requirements" in result

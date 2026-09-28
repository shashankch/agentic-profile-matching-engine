"""
Presentation Layer Styles and Design Tokens for Yojaka AI.
Implements modern 2026 aesthetics: adaptive glassmorphism, responsive contrast,
custom Outfit typography, and dynamic theme detection.
"""

import streamlit as st


CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Global Typography */
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Main App Header with Gradient */
    .app-header {
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.75rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
        letter-spacing: -0.025em;
    }
    
    .app-subtitle {
        font-size: 1.1rem;
        color: #9ca3af;
        margin-bottom: 1.75rem;
    }

    /* Adaptive Theme Variables for Candidate Cards (Default: Clean Crisp Light Mode) */
    :root, [data-theme="light"], .stApp {
        --c-card-bg: #ffffff;
        --c-card-border: #e2e8f0;
        --c-card-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.08), 0 2px 6px -1px rgba(15, 23, 42, 0.04);
        --c-card-hover-bg: #f8fafc;
        --c-card-hover-border: #6366f1;
        --c-card-name: #0f172a;
        --c-card-rank: #64748b;
        --c-card-meta: #334155;
        --c-card-label: #475569;
        --c-skill-bg: #f1f5f9;
        --c-skill-text: #1e293b;
        --c-skill-border: #cbd5e1;
        --c-card-path: #64748b;
    }

    /* Dark Mode: Applied strictly when data-theme="dark" */
    [data-theme="dark"], [data-theme="dark"] .stApp {
        --c-card-bg: rgba(30, 41, 59, 0.65);
        --c-card-border: rgba(255, 255, 255, 0.1);
        --c-card-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.35);
        --c-card-hover-bg: rgba(30, 41, 59, 0.9);
        --c-card-hover-border: rgba(167, 139, 250, 0.5);
        --c-card-name: #ffffff;
        --c-card-rank: #94a3b8;
        --c-card-meta: #cbd5e1;
        --c-card-label: #cbd5e1;
        --c-skill-bg: rgba(15, 23, 42, 0.7);
        --c-skill-text: #f1f5f9;
        --c-skill-border: rgba(255, 255, 255, 0.15);
        --c-card-path: #94a3b8;
    }

    /* Candidate Card Base */
    .candidate-card {
        background-color: var(--c-card-bg);
        border: 1px solid var(--c-card-border);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: var(--c-card-shadow);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .candidate-card:hover {
        transform: translateY(-2px);
        border-color: var(--c-card-hover-border);
        background-color: var(--c-card-hover-bg);
        box-shadow: 0 12px 24px -4px rgba(15, 23, 42, 0.12), 0 4px 8px -2px rgba(15, 23, 42, 0.06);
    }

    /* Card Elements */
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
    }

    .card-name {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--c-card-name);
        letter-spacing: -0.01em;
    }

    .card-rank {
        font-size: 0.85rem;
        color: var(--c-card-rank);
        margin-left: 8px;
        font-weight: 500;
    }

    .card-meta {
        font-size: 0.9rem;
        color: var(--c-card-meta);
        margin-bottom: 0.85rem;
        display: flex;
        gap: 1.5rem;
    }

    .card-path {
        font-size: 0.75rem;
        color: var(--c-card-path);
        word-break: break-all;
        margin-top: 0.75rem;
    }

    .skills-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: var(--c-card-label);
        letter-spacing: 0.05em;
        margin-right: 6px;
    }

    /* Score Badges */
    .score-badge {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.875rem;
        box-shadow: 0 2px 4px rgba(16, 185, 129, 0.2);
    }

    /* Status Pills */
    .status-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    .status-strong {
        background-color: rgba(16, 185, 129, 0.15);
        color: #059669;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }

    .status-borderline {
        background-color: rgba(245, 158, 11, 0.15);
        color: #d97706;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }

    .status-rejected {
        background-color: rgba(239, 68, 68, 0.15);
        color: #dc2626;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }

    @media (prefers-color-scheme: dark) {
        .status-strong { color: #34d399; }
        .status-borderline { color: #fbbf24; }
        .status-rejected { color: #f87171; }
    }
    [data-theme="dark"] .status-strong { color: #34d399; }
    [data-theme="dark"] .status-borderline { color: #fbbf24; }
    [data-theme="dark"] .status-rejected { color: #f87171; }

    /* Skills Badges */
    .skill-tag {
        display: inline-block;
        background: var(--c-skill-bg);
        color: var(--c-skill-text);
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 5px;
        margin-bottom: 5px;
        border: 1px solid var(--c-skill-border);
    }

    /* Customizing Streamlit Tabs & Buttons */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: transparent;
        padding: 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 6px 6px 0 0;
        background-color: rgba(255, 255, 255, 0.01);
        color: #9ca3af;
        border: 1px solid transparent;
        padding: 0 14px;
        transition: all 0.2s;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(167, 139, 250, 0.08) !important;
        color: #c084fc !important;
        border-color: rgba(167, 139, 250, 0.2) rgba(167, 139, 250, 0.2) transparent rgba(167, 139, 250, 0.2) !important;
        font-weight: 600;
    }

    /* Styled buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 8px 20px !important;
        border-radius: 8px !important;
        box-shadow: 0 3px 5px rgba(79, 70, 229, 0.2) !important;
        transition: all 0.2s !important;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-1.5px) !important;
        box-shadow: 0 6px 10px rgba(79, 70, 229, 0.3) !important;
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
    }

    .stButton>button:active {
        transform: translateY(0) !important;
    }
</style>
"""

THEME_DETECTOR_JS = """
<script>
(function() {
    function detectTheme() {
        try {
            const app = document.querySelector('.stApp');
            if (!app) return;
            const bg = window.getComputedStyle(app).backgroundColor;
            const isDark = bg.includes('14, 17, 23') || bg.includes('14,17,23') || bg === 'rgb(14, 17, 23)';
            const newTheme = isDark ? 'dark' : 'light';
            if (document.documentElement.getAttribute('data-theme') !== newTheme) {
                document.documentElement.setAttribute('data-theme', newTheme);
            }
        } catch (e) {}
    }
    const observer = new MutationObserver(detectTheme);
    observer.observe(document.body, { attributes: true, childList: true, subtree: true });
    detectTheme();
    setInterval(detectTheme, 400);
})();
</script>
"""


def apply_custom_styles() -> None:
    """Injects responsive glassmorphic styles and theme detection script into the active Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.html(THEME_DETECTOR_JS)

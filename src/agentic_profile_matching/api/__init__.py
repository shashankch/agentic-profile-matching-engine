"""
Yojaka AI Headless FastAPI Sidecar Package.
Exposes REST and Server-Sent Event (SSE) endpoints for programmatic headless integrations.
"""

from agentic_profile_matching.api.app import create_app, app

__all__ = ["create_app", "app"]

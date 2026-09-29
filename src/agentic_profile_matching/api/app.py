"""
FastAPI Application Factory for Yojaka AI Headless Gateway.
Configures CORS, OpenAPI docs, and mounts REST / SSE route handlers.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agentic_profile_matching.api.routes import router


def create_app() -> FastAPI:
    """Creates and configures the headless FastAPI application."""
    application = FastAPI(
        title="Yojaka AI Gateway API",
        description=(
            "Headless REST & Server-Sent Events (SSE) API for autonomous candidate profile matching, "
            "job description requirements extraction, and agentic screening workflows."
        ),
        version="1.3.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Enable CORS for modern web clients and decoupled frontends
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routes
    application.include_router(router)

    return application


# Global ASGI instance for uvicorn
app = create_app()

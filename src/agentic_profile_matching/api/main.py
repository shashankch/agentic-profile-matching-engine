"""
CLI Entrypoint for running Yojaka AI Headless FastAPI Gateway.
Usage:
    python -m agentic_profile_matching.api.main [--host 0.0.0.0] [--port 8000]
"""

import argparse
import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Yojaka AI Headless FastAPI Gateway")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    uvicorn.run(
        "agentic_profile_matching.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()

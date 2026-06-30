"""FastAPI HTTP service for bluenamer.

Lives behind the optional ``[web]`` extra. The package imports cleanly
without ``fastapi``/``uvicorn`` installed; this subpackage is only
imported when the user explicitly opts in.
"""

from .app import create_app

__all__ = ["create_app"]

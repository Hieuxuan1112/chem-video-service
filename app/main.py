"""ASGI entry point: uvicorn app.main:app (runs the embedded worker unless disabled)."""

from app.api.app import create_app
from app.container import build_container
from app.observability.logging import configure_logging

configure_logging()
app = create_app(build_container())

"""FastAPI application factory."""

import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.errors import install_error_handlers
from app.api.routes import router
from app.container import Container

log = logging.getLogger(__name__)


def create_app(container: Container) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        stop = threading.Event()
        thread: threading.Thread | None = None
        worker = container.worker
        if worker is not None and container.settings.run_embedded_worker:
            thread = threading.Thread(
                target=worker.run_forever, args=(stop,), name="embedded-worker", daemon=True
            )
            thread.start()
            log.info("embedded worker started", extra={"data": {"worker_id": worker.worker_id}})
        try:
            yield
        finally:
            stop.set()
            if thread is not None:
                # A running job is not interrupted; if it outlives the grace period the daemon
                # thread dies with the process and the job is reclaimed after its lease expires.
                thread.join(timeout=container.settings.worker_shutdown_grace_s)
                if thread.is_alive():
                    log.warning(
                        "embedded worker still busy at shutdown; leaving job to lease expiry"
                    )

    app = FastAPI(title="AI Chemistry Video Request Service", version="1.0.0", lifespan=lifespan)
    app.state.container = container
    install_error_handlers(app)
    app.include_router(router)
    return app

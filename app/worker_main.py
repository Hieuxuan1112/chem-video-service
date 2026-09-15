"""Standalone worker process: python -m app.worker_main (any number may share the database)."""

import signal
import threading

from app.container import build_container
from app.observability.logging import configure_logging


def main() -> int:
    configure_logging()
    container = build_container()
    assert container.worker is not None
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    container.worker.run_forever(stop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

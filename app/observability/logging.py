"""JSON-lines logging on stdout (spec §9.4)."""

import json
import logging
import sys
from typing import Any

from app.observability.context import attempt_var, job_id_var, stage_var, worker_id_var


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": record.created,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "job_id": job_id_var.get(),
            "attempt": attempt_var.get(),
            "stage": stage_var.get(),
            "worker_id": worker_id_var.get(),
            "data": getattr(record, "data", None),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

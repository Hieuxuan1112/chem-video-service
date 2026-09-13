import io
import json
import logging

import pytest

from app.observability.context import (
    attempt_var,
    job_context,
    job_id_var,
    stage_context,
    stage_var,
    worker_id_var,
)
from app.observability.logging import JsonFormatter, configure_logging


def _capture() -> tuple[logging.Logger, io.StringIO]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("test.observability")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger, stream


def _lines(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_json_line_contains_context() -> None:
    logger, stream = _capture()
    with (
        job_context(job_id="vid_abc", attempt=2, worker_id="host:1:deadbeef"),
        stage_context("voicing"),
    ):
        logger.info("clip synthesized", extra={"data": {"scene": 3}})
    (line,) = _lines(stream)
    assert line["msg"] == "clip synthesized"
    assert line["level"] == "INFO"
    assert line["logger"] == "test.observability"
    assert (line["job_id"], line["attempt"], line["stage"], line["worker_id"]) == (
        "vid_abc",
        2,
        "voicing",
        "host:1:deadbeef",
    )
    assert line["data"] == {"scene": 3}
    assert isinstance(line["ts"], float)


def test_context_reset_after_exit() -> None:
    logger, stream = _capture()
    with job_context(job_id="vid_1", attempt=1, worker_id="w"):
        with stage_context("rendering"):
            assert stage_var.get() == "rendering"
        assert stage_var.get() is None
        assert job_id_var.get() == "vid_1"
    assert (job_id_var.get(), attempt_var.get(), stage_var.get(), worker_id_var.get()) == (
        None,
        None,
        None,
        None,
    )
    logger.info("outside")
    (line,) = _lines(stream)
    assert (line["job_id"], line["attempt"], line["stage"], line["worker_id"]) == (
        None,
        None,
        None,
        None,
    )


def test_context_reset_on_exception() -> None:
    with pytest.raises(RuntimeError), job_context(job_id="vid_2", attempt=1, worker_id="w"):
        raise RuntimeError("boom")
    assert job_id_var.get() is None


def test_exception_info_serialized() -> None:
    logger, stream = _capture()
    try:
        raise ValueError("bad value")
    except ValueError:
        logger.exception("failed")
    (line,) = _lines(stream)
    assert "ValueError: bad value" in line["exc"]


def test_configure_logging_installs_json_handler() -> None:
    root = logging.getLogger()
    saved = root.handlers[:], root.level
    try:
        configure_logging("DEBUG")
        assert root.level == logging.DEBUG
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JsonFormatter)
    finally:
        root.handlers, level = saved[0], saved[1]
        root.setLevel(level)

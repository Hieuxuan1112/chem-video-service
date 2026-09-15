"""Subprocesses in their own process group, killed as a whole tree on timeout or cancel."""

import contextlib
import os
import subprocess
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

_POLL_S = 0.2
_STDERR_TAIL_CHARS = 2000


@dataclass(frozen=True)
class ProcResult:
    returncode: int
    stdout: bytes
    stderr_tail: str
    timed_out: bool


def kill_tree(pid: int) -> None:
    """Kill every descendant, then the process itself. Already-exited processes are ignored."""
    try:
        parent = psutil.Process(pid)
        procs = [*parent.children(recursive=True), parent]
    except psutil.NoSuchProcess:
        return
    for proc in procs:
        with contextlib.suppress(psutil.NoSuchProcess):
            proc.kill()
    psutil.wait_procs(procs, timeout=5)


def run_subprocess(
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout_s: float,
    input_bytes: bytes | None = None,
    cancel: threading.Event | None = None,
) -> ProcResult:
    group: dict[str, Any] = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}  # type: ignore[attr-defined]
        if os.name == "nt"
        else {"start_new_session": True}
    )
    proc = subprocess.Popen(
        list(args),
        cwd=cwd,
        stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **group,
    )
    deadline = time.monotonic() + timeout_s
    timed_out = False
    while True:
        try:
            stdout, stderr = proc.communicate(input_bytes, timeout=_POLL_S)
            break
        except subprocess.TimeoutExpired:
            expired = time.monotonic() >= deadline
            if expired or (cancel is not None and cancel.is_set()):
                timed_out = expired
                kill_tree(proc.pid)
                stdout, stderr = proc.communicate()
                break
    tail = stderr.decode("utf-8", "replace")[-_STDERR_TAIL_CHARS:]
    return ProcResult(proc.returncode, stdout, tail, timed_out)

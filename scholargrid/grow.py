"""Growing the collection in the background.

``start`` launches ``python -m scholargrid.pipeline --status-file ...`` as a
detached process, so harvesting tens of thousands of papers never blocks the
app. The child writes its progress to ``data/grow_status.json``; the app reads
it with ``read_status`` and picks up the new release when the job publishes.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from typing import Dict, Optional

from .config import REPO_ROOT, Config
from .utils import ensure_dir, get_logger, load_json, save_json

log = get_logger("grow")


def status_path(cfg: Config) -> str:
    return os.path.join(os.path.dirname(cfg.live_dir), "grow_status.json")


def log_path(cfg: Config) -> str:
    return os.path.join(os.path.dirname(cfg.live_dir), "grow.log")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        # A finished child stays a zombie (and passes kill(0)) until it is reaped.
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def write_status(path: str, **fields) -> Dict:
    current = load_json(path) if os.path.exists(path) else {}
    current.update(fields, updated_at=_now())
    ensure_dir(os.path.dirname(path) or ".")
    tmp = path + ".tmp"
    save_json(current, tmp)
    os.replace(tmp, path)
    return current


def read_status(cfg: Config) -> Dict:
    """The last job's status; a running job whose process died is reported as failed."""
    path = status_path(cfg)
    if not os.path.exists(path):
        return {"state": "idle"}
    try:
        status = load_json(path)
    except Exception:
        return {"state": "idle"}
    if status.get("state") == "running" and not _alive(int(status.get("pid") or 0)):
        status = write_status(path, state="failed", message="The background job stopped unexpectedly.",
                              finished_at=_now())
    return status


def is_running(cfg: Config) -> bool:
    return read_status(cfg).get("state") == "running"


def start(cfg: Config, extra_args: Optional[list] = None) -> Dict:
    """Launch the pipeline in a detached process (no-op if one is already running)."""
    status = read_status(cfg)
    if status.get("state") == "running":
        return status
    path = status_path(cfg)
    args = [sys.executable, "-m", "scholargrid.pipeline", "--status-file", path]
    if cfg.path:
        args += ["--config", cfg.path]
    args += list(extra_args or [])
    ensure_dir(os.path.dirname(path) or ".")
    kwargs: Dict = {"cwd": REPO_ROOT, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL,
                    "close_fds": True}
    if sys.platform == "win32":
        kwargs["creationflags"] = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                                   | subprocess.CREATE_NO_WINDOW)
    else:
        kwargs["start_new_session"] = True
    with open(log_path(cfg), "a", encoding="utf-8") as logf:
        proc = subprocess.Popen(args, stdout=logf, **kwargs)
    log.info("Started background pipeline (pid %d); progress in %s", proc.pid, path)
    return write_status(path, state="running", pid=proc.pid, started_at=_now(), progress=0.0,
                        message="Starting…", target=cfg.max_papers, release=None, finished_at=None,
                        error=None)


def log_tail(cfg: Config, lines: int = 15) -> str:
    path = log_path(cfg)
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as fh:
        return "".join(fh.readlines()[-lines:])

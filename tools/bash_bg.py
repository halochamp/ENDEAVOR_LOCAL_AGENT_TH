# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""bash_bg.py — background bash jobs: start long-running commands, poll/list/kill.

Separate tool from `bash` (run-now, blocks until done or timeout) — background jobs
are register-then-poll, same reasoning as awake vs tool_loop's separation (mixing a
run-now and a register-then-return contract in one docstring blurs the model's tool
choice). Runs under the SAME sandbox profile as `bash` (workspace + /tmp writable only).
"""
from __future__ import annotations
import fcntl
import json
import os
import signal
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from langchain_core.tools import tool
from tools.bash import _build_sandbox_profile

_MAX_JOBS = 5
_KILL_GRACE_SECONDS = 3
_TAIL_CHARS = 2_000

# Popen handles for jobs started by THIS process — gives an exact exit code via
# proc.poll(). Lost across an agent restart (registry survives, this dict doesn't);
# status/kill fall back to a pid-liveness check in that case (see _refresh_status).
_PROCS: dict[str, subprocess.Popen] = {}

# Sandbox profile temp files, keyed by job_id — kept alive until the job is confirmed
# exited (see _cleanup_profile). sandbox-exec reads/compiles this file from inside the
# freshly-exec'd child, which happens AFTER Popen() returns to the parent; deleting it
# right after Popen() races the child's own read and intermittently fails the job with
# "sandbox-exec: ...: No such file or directory" (caught by a real functional test, not
# by unit tests — the race doesn't reproduce every time).
_PROFILE_PATHS: dict[str, str] = {}


def _cleanup_profile(job_id: str) -> None:
    path = _PROFILE_PATHS.pop(job_id, None)
    if path:
        try:
            os.unlink(path)
        except OSError:
            pass


def _registry_path() -> Path:
    from config import WORKSPACE
    return Path(WORKSPACE) / "bash_jobs.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class _Registry:
    """Flock-guarded JSON registry."""

    def __init__(self):
        self.path = _registry_path()
        self._lock_path = self.path.with_suffix(".lock")

    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self._lock_path, "w")
        fcntl.flock(handle, fcntl.LOCK_EX)
        return handle

    def load(self) -> list[dict]:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError:
            return []
        try:
            return json.loads(raw).get("jobs", [])
        except Exception:
            return []

    def mutate(self, fn):
        handle = self._locked()
        try:
            jobs = self.load()
            result = fn(jobs)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"jobs": jobs}, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
            return result
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _ps_lstart(pid: int) -> str | None:
    """Process start-time signature (`ps -o lstart=`) — the only cheap way on macOS to tell
    'the process we started' from 'the OS recycled that pid to something unrelated' once the
    in-memory Popen handle is gone (agent restart). None if the pid has no process."""
    try:
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)],
                              capture_output=True, text=True, timeout=5)
        line = out.stdout.strip()
        return line or None
    except Exception:
        return None


def _refresh_status(job: dict) -> dict:
    if job["status"] != "running":
        return job
    proc = _PROCS.get(job["id"])
    if proc is not None:
        rc = proc.poll()
        if rc is not None:
            job["status"] = "exited"
            job["exit_code"] = rc
            job["ended_at"] = _now_iso()
            _cleanup_profile(job["id"])
        return job
    # No in-memory handle (agent restarted since this job started) — pid liveness alone
    # can't tell "still our job" from "pid recycled to an unrelated process"; require the
    # start-time signature captured at spawn to still match before trusting "running".
    sig = _ps_lstart(job["pid"])
    if sig is None or sig != job.get("start_sig"):
        job["status"] = "exited"
        job["exit_code"] = None
        job["ended_at"] = _now_iso()
        _cleanup_profile(job["id"])
    return job


def _tail(path: str, n_chars: int = _TAIL_CHARS) -> str:
    try:
        data = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(log not readable yet)"
    return data[-n_chars:]


@tool
def bash_bg(action: str, command: str = "", job_id: str = "") -> str:
    """Background bash jobs — start a long-running command without blocking, then poll/list/kill it.

    Use for anything that legitimately runs past `bash`'s timeout and where you don't
    need the result the instant it finishes: dev servers (npm start, uvicorn), big
    downloads, long builds/installs. For anything that finishes in seconds, use `bash`
    directly — it returns the result immediately, no polling needed.

    actions:
      start  — command required. Spawns under the same sandbox as `bash` (workspace/
               and /tmp writable only), stdout+stderr redirected to a log file in
               workspace/. Returns job_id + log path right away; the command keeps
               running after this call returns. Max 5 concurrent jobs.
      status — job_id required. Returns running/exited (+ exit code when known) and
               the last ~2,000 chars of the log.
      list   — no args. Lists every tracked job: id, status, command, started_at.
      kill   — job_id required. SIGTERM, then SIGKILL after 3s if still alive.

    ❌ bash_bg(action="start", command="pytest test_foo.py") — finishes in 5s, just use bash.
    ✅ bash_bg(action="start", command="npm run dev") → poll with status later, or leave it
       running and check back when the user asks.
    """
    action = (action or "").strip().lower()
    if action not in {"start", "status", "list", "kill"}:
        return "[error] action must be one of: start, status, list, kill"

    from config import WORKSPACE
    reg = _Registry()

    if action == "list":
        jobs = reg.mutate(lambda js: [_refresh_status(j) for j in js])
        if not jobs:
            return "(no background jobs)"
        lines = [f"{j['id']} [{j['status']}] {j['command'][:60]} (started {j['started_at']})" for j in jobs]
        return "\n".join(lines)

    if action == "start":
        if not command:
            return "[error] command is required for action=start"
        jobs = reg.mutate(lambda js: [_refresh_status(j) for j in js])
        running = [j for j in jobs if j["status"] == "running"]
        if len(running) >= _MAX_JOBS:
            return (f"[error] {_MAX_JOBS} background jobs already running — "
                    f"kill one first (bash_bg(action='list') to see them)")

        job_id = uuid.uuid4().hex[:8]
        log_path = str(Path(WORKSPACE) / f"_bash_bg_{job_id}.log")
        profile = _build_sandbox_profile(WORKSPACE)
        profile_fd, profile_path = tempfile.mkstemp(suffix=".sb")
        try:
            with os.fdopen(profile_fd, "w") as f:
                f.write(profile)
            with open(log_path, "w", encoding="utf-8") as logf:
                proc = subprocess.Popen(
                    ["sandbox-exec", "-f", profile_path, "bash", "-c", command],
                    stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    cwd=WORKSPACE,
                )
        except Exception as e:
            try:
                os.unlink(profile_path)
            except OSError:
                pass
            return f"[error] failed to start: {e}"

        # Profile stays on disk until the job is confirmed exited — see _cleanup_profile.
        _PROCS[job_id] = proc
        _PROFILE_PATHS[job_id] = profile_path
        job = {
            "id": job_id, "command": command, "pid": proc.pid,
            "log": log_path, "status": "running", "exit_code": None,
            "started_at": _now_iso(), "ended_at": None,
            "start_sig": _ps_lstart(proc.pid),  # see _refresh_status pid-recycle guard
        }
        reg.mutate(lambda js: js.append(job) or js)
        return (f"[bash_bg] started job {job_id} (pid {proc.pid}) — log: {log_path}\n"
                f"Poll with bash_bg(action='status', job_id='{job_id}')")

    # status / kill both need to find the job first
    if not job_id:
        return f"[error] job_id is required for action={action}"
    jobs = reg.mutate(lambda js: [_refresh_status(j) for j in js])
    job = next((j for j in jobs if j["id"] == job_id), None)
    if job is None:
        return f"[error] unknown job_id: {job_id} — check bash_bg(action='list')"

    if action == "status":
        tail = _tail(job["log"])
        exit_note = f" exit_code={job['exit_code']}" if job["status"] == "exited" else ""
        return f"job {job['id']} [{job['status']}]{exit_note}\ncommand: {job['command']}\n--- log tail ---\n{tail}"

    # kill — only ever signal a pid we can verify is still the process we started.
    # The `_refresh_status()` call above (shared with `status`) already enforces this:
    # a job with no trusted `_PROCS` handle only survives as "running" if its start-time
    # signature still matches what was recorded at spawn — otherwise it's already been
    # downgraded to "exited" by the time we get here, and the early return above catches
    # it. So by this point job["status"] == "running" IS the trust proof; no separate
    # check is needed. This matters because the tool can fire from unsupervised AWAKE
    # turns — there's no human in the loop to catch a wrong-process kill.
    if job["status"] != "running":
        return f"job {job_id} already {job['status']}"

    try:
        os.kill(job["pid"], signal.SIGTERM)
        time.sleep(_KILL_GRACE_SECONDS)
        if _is_alive(job["pid"]):
            os.kill(job["pid"], signal.SIGKILL)
    except OSError:
        pass

    def _mark_killed(js):
        for j in js:
            if j["id"] == job_id:
                j["status"] = "killed"
                j["ended_at"] = _now_iso()
        return js

    reg.mutate(_mark_killed)
    _cleanup_profile(job_id)
    return f"job {job_id} killed"

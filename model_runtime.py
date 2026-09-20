"""Standalone/local model-server lifecycle for ENDEAVOR_LOCAL_AGENT_TH.

Agent TH owns its normal local mlx_vlm server without depending on an external
monitor. A special ``shared_max`` mode is intentionally read-only: it may attach
to Agent MAX VLM's already-running test server (default :8085), but it never
starts, stops, resets, adopts, or changes that server's model.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Iterator
from urllib import error as _urlerror
from urllib import request as _urlrequest

import config

_PROJECT_DIR = Path(__file__).resolve().parent
_LAUNCHER = _PROJECT_DIR / "scripts" / "run_mlx_server_owned.py"
_LOG_PATH = _PROJECT_DIR / "logs" / "model_server.log"
_LOCK_PATH = Path(tempfile.gettempdir()) / "endeavor_agent_th_model_switch.lock"
_CONTROL_SETTINGS_PATH = _PROJECT_DIR / "workspace" / "model_server_settings.json"
_DEFAULT_SWITCH_TIMEOUT = 240.0
_STOP_TIMEOUT = 20.0
_CONTROL_DEFAULTS = {"watchdog_enabled": True, "desired_state": "running"}


def _port(port: int | None = None) -> int:
    value = config.get_server_port() if port is None else int(port)
    if not 1024 <= int(value) <= 65535:
        raise RuntimeError(f"invalid model-server port: {value}")
    return int(value)


def _health_url(port: int | None = None) -> str:
    return f"http://127.0.0.1:{_port(port)}/health"


def probe_model_health(timeout: float = 2.0, *, port: int | None = None) -> dict:
    req = _urlrequest.Request(
        _health_url(port),
        headers={"Authorization": f"Bearer {config.API_KEY}"},
        method="GET",
    )
    try:
        with _urlrequest.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except (_urlerror.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"model server health unavailable: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("model server returned invalid /health JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("model server returned invalid /health payload")
    return payload


def _listener_pids(port: int | None = None) -> list[int]:
    target_port = _port(port)
    proc = subprocess.run(
        ["/usr/sbin/lsof", f"-tiTCP:{target_port}", "-sTCP:LISTEN"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if proc.returncode not in (0, 1):
        raise RuntimeError(proc.stderr.strip() or "unable to inspect model-server listener")
    pids: list[int] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pids.append(int(line))
        except ValueError as exc:
            raise RuntimeError(f"invalid listener pid: {line!r}") from exc
    return sorted(set(pids))


def _process_command(pid: int) -> str:
    proc = subprocess.run(
        ["/bin/ps", "-p", str(pid), "-o", "command="],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"unable to verify listener pid {pid}")
    return proc.stdout.strip()


def _is_owned_model_server(command: str, *, port: int | None = None) -> bool:
    text = str(command or "")
    port_token = str(_port(port))
    owns_entrypoint = str(_LAUNCHER) in text
    has_port = f"--port {port_token}" in text or f"--port={port_token}" in text
    return bool(owns_entrypoint and has_port)


def _is_max_shared_server(command: str, *, port: int | None = None) -> bool:
    """Recognize the explicit Agent MAX VLM test-server process signature."""
    text = str(command or "")
    port_token = str(_port(port))
    has_launcher = "run_vlm_server_patched.py" in text
    has_project = "ENDEAVOR_LOCAL_AGENT_MAX_VLM" in text
    has_port = f"--port {port_token}" in text or f"--port={port_token}" in text
    return bool(has_launcher and has_project and has_port)


def _wait_until_port_free(timeout: float = _STOP_TIMEOUT, *, port: int | None = None) -> None:
    target_port = _port(port)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _listener_pids(target_port):
            return
        time.sleep(0.2)
    raise TimeoutError(f"model server :{target_port} did not stop within {timeout:g}s")


def stop_model_server(timeout: float = _STOP_TIMEOUT, *, port: int | None = None) -> None:
    """Stop only a verified Agent TH-owned launcher."""
    target_port = _port(port)
    pids = _listener_pids(target_port)
    if not pids:
        return
    if len(pids) != 1:
        raise RuntimeError(f"refusing to stop ambiguous listeners on :{target_port}: {pids}")
    pid = pids[0]
    command = _process_command(pid)
    if not _is_owned_model_server(command, port=target_port):
        raise RuntimeError(
            f"refusing to stop unrecognized listener pid {pid} on :{target_port}"
        )
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    _wait_until_port_free(timeout, port=target_port)


def _server_env(model: str) -> dict[str, str]:
    """Environment for the TH-owned Standalone launcher only.

    APC settings come from the model registry rather than repo-ID branches, so
    a future model inherits the same owner lifecycle by declaring its native
    APC contract in model_registry.json. Shared MAX never receives these values.
    """
    from model_registry import get_native_apc_contract

    contract = get_native_apc_contract(model)
    env = dict(os.environ)
    env["APC_ENABLED"] = "1" if contract.enabled else "0"
    env["APC_EXACT_CACHE_ENTRIES"] = str(contract.exact_cache_entries)
    env["APC_EXACT_PREFIX_GUARD_TOKENS"] = env.get("APC_EXACT_PREFIX_GUARD_TOKENS", "64")
    return env


def start_model_server(model: str, *, port: int | None = None) -> subprocess.Popen:
    model = str(model or "").strip()
    if model not in config.MODEL_CHOICES:
        raise ValueError(f"unsupported model: {model!r}")
    if not _LAUNCHER.is_file():
        raise RuntimeError(f"model launcher missing: {_LAUNCHER}")
    target_port = _port(port)
    if _listener_pids(target_port):
        raise RuntimeError(f"cannot start model server: :{target_port} is already occupied")

    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(_LOG_PATH, "ab", buffering=0)
    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                str(_LAUNCHER),
                "--model", model,
                "--host", "127.0.0.1",
                "--port", str(target_port),
                "--enable-thinking",
            ],
            cwd=str(_PROJECT_DIR),
            env=_server_env(model),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_fh.close()
    return proc


def _log_tail(max_bytes: int = 3000) -> str:
    try:
        with open(_LOG_PATH, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes), os.SEEK_SET)
            return fh.read().decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def wait_for_model_ready(
    model: str,
    *,
    process: subprocess.Popen | None = None,
    timeout: float = _DEFAULT_SWITCH_TIMEOUT,
    port: int | None = None,
) -> dict:
    deadline = time.monotonic() + timeout
    last: dict | str = {}
    while time.monotonic() < deadline:
        if process is not None:
            code = process.poll()
            if code is not None:
                tail = _log_tail()
                detail = f"; log tail: {tail}" if tail else ""
                raise RuntimeError(f"model server exited during startup (rc={code}){detail}")
        try:
            last = probe_model_health(timeout=2.0, port=port)
            if last.get("status") == "healthy" and str(last.get("loaded_model") or "") == str(model):
                return last
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(0.5)
    raise TimeoutError(f"model {model!r} did not become ready; last={last}")


@contextlib.contextmanager
def _switch_lock() -> Iterator[None]:
    _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_LOCK_PATH, "a+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _ensure_model_server_locked(model: str, timeout: float, *, port: int | None = None) -> dict:
    target_port = _port(port)
    pids = _listener_pids(target_port)
    if pids:
        if len(pids) != 1:
            raise RuntimeError(f"refusing ambiguous listeners on :{target_port}: {pids}")
        command = _process_command(pids[0])
        if not _is_owned_model_server(command, port=target_port):
            raise RuntimeError(
                f"refusing to take over non-owner listener pid {pids[0]} on :{target_port}"
            )
        try:
            health = probe_model_health(port=target_port)
        except Exception:
            health = {}
        if health.get("status") == "healthy" and str(health.get("loaded_model") or "") == model:
            return health
        stop_model_server(port=target_port)
    proc = start_model_server(model, port=target_port)
    return wait_for_model_ready(model, process=proc, timeout=timeout, port=target_port)


def _restart_model_server_locked(
    model: str,
    timeout: float,
    *,
    port: int | None = None,
    previous_port: int | None = None,
) -> dict:
    target_port = _port(port)
    old_port = target_port if previous_port is None else _port(previous_port)
    if target_port != old_port and _listener_pids(target_port):
        raise RuntimeError(f"cannot move model server: :{target_port} is already occupied")
    stop_model_server(port=old_port)
    proc = start_model_server(model, port=target_port)
    return wait_for_model_ready(model, process=proc, timeout=timeout, port=target_port)


def get_model_server_control_settings() -> dict:
    try:
        with open(_CONTROL_SETTINGS_PATH, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return dict(_CONTROL_DEFAULTS)
    if not isinstance(payload, dict):
        return dict(_CONTROL_DEFAULTS)
    watchdog = bool(payload.get("watchdog_enabled", _CONTROL_DEFAULTS["watchdog_enabled"]))
    desired = str(payload.get("desired_state") or _CONTROL_DEFAULTS["desired_state"])
    if desired not in {"running", "stopped"}:
        desired = _CONTROL_DEFAULTS["desired_state"]
    return {"watchdog_enabled": watchdog, "desired_state": desired}


def _write_control_unlocked(*, watchdog_enabled: bool, desired_state: str) -> dict:
    desired = str(desired_state or "").strip().lower()
    if desired not in {"running", "stopped"}:
        raise ValueError(f"invalid model-server desired state: {desired!r}")
    payload = {"watchdog_enabled": bool(watchdog_enabled), "desired_state": desired}
    _CONTROL_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".model_server_settings.", suffix=".tmp", dir=str(_CONTROL_SETTINGS_PATH.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, sort_keys=True)
        os.replace(tmp_path, _CONTROL_SETTINGS_PATH)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return payload


def set_model_server_watchdog(enabled: bool) -> dict:
    if config.get_server_mode() == "shared_max":
        raise RuntimeError("external model server is read-only; watchdog is disabled")
    with _switch_lock():
        current = get_model_server_control_settings()
        return _write_control_unlocked(
            watchdog_enabled=bool(enabled), desired_state=current["desired_state"]
        )


def shared_max_server_status(*, port: int | None = None) -> dict:
    """Inspect a candidate MAX test server without mutating either project."""
    target_port = _port(port)
    result = {
        "owner": "agent_max_vlm",
        "mode": "shared_max",
        "managed_by_th": False,
        "port": target_port,
        "selected_model": "",
        "watchdog_enabled": False,
        "desired_state": "shared",
        "running": False,
        "owned": False,
        "healthy": False,
        "state": "shared_offline",
        "loaded_model": "",
        "error": "",
    }
    try:
        pids = _listener_pids(target_port)
    except Exception as exc:
        result.update(state="error", error=f"listener inspection failed: {exc}")
        return result
    if not pids:
        result["error"] = f"MAX test server is not listening on :{target_port}"
        return result
    result["running"] = True
    if len(pids) != 1:
        result.update(state="ambiguous", error=f"multiple listeners: {pids}")
        return result
    try:
        command = _process_command(pids[0])
    except Exception as exc:
        result.update(state="error", error=str(exc))
        return result
    result["owned"] = _is_max_shared_server(command, port=target_port)
    if not result["owned"]:
        result.update(state="foreign", error=f":{target_port} is not a compatible external model server")
        return result
    try:
        health = probe_model_health(port=target_port)
    except Exception as exc:
        result.update(state="unhealthy", error=str(exc))
        return result
    loaded = str(health.get("loaded_model") or "")
    result["loaded_model"] = loaded
    result["selected_model"] = loaded
    result["healthy"] = health.get("status") == "healthy" and loaded in config.MODEL_CHOICES
    result["state"] = "shared_ready" if result["healthy"] else "unhealthy"
    if not result["healthy"]:
        result["error"] = f"external server model {loaded!r} is unsupported"
    return result


def _base_status() -> dict:
    control = get_model_server_control_settings()
    shared = config.get_server_mode() == "shared_max"
    return {
        "owner": "agent_max_vlm" if shared else "agent_th",
        "mode": config.get_server_mode(),
        "managed_by_th": not shared,
        "port": _port(),
        "selected_model": config.get_model(),
        "watchdog_enabled": False if shared else bool(control["watchdog_enabled"]),
        "desired_state": "shared" if shared else str(control["desired_state"]),
        "running": False,
        "owned": False,
        "healthy": False,
        "state": "stopped",
        "loaded_model": "",
        "error": "",
    }


def model_server_status() -> dict:
    if config.get_server_mode() == "shared_max":
        result = shared_max_server_status(port=config.get_server_port())
        if result.get("healthy"):
            config.adopt_shared_model(str(result.get("loaded_model") or ""))
        return result

    result = _base_status()
    target_port = _port()
    try:
        pids = _listener_pids(target_port)
    except Exception as exc:
        result.update(state="error", error=f"listener inspection failed: {exc}")
        return result
    if not pids:
        return result
    result["running"] = True
    if len(pids) != 1:
        result.update(state="ambiguous", error=f"multiple listeners: {pids}")
        return result
    try:
        command = _process_command(pids[0])
    except Exception as exc:
        result.update(state="error", error=str(exc))
        return result

    result["owned"] = _is_owned_model_server(command, port=target_port)
    if not result["owned"]:
        result.update(
            state="foreign",
            error=f":{target_port} is owned by another process",
        )
        return result

    try:
        health = probe_model_health(port=target_port)
    except Exception as exc:
        result.update(state="unhealthy", error=str(exc))
        return result
    loaded = str(health.get("loaded_model") or "")
    result["loaded_model"] = loaded
    selected = config.get_model()
    result["healthy"] = health.get("status") == "healthy" and loaded == selected
    result["state"] = "ready" if result["healthy"] else "unhealthy"
    if not result["healthy"]:
        result["error"] = f"loaded model {loaded!r} does not match owner selection {selected!r}"
    return result


def auto_attach_max_test_server_if_present() -> dict | None:
    """Switch to read-only shared mode only for a verified MAX listener.

    This preserves the historical developer convenience of launching Agent TH
    while MAX VLM already owns the default test port, but unlike the old
    generic adoption path it never trusts an arbitrary external listener.
    """
    if config.get_server_mode() != "standalone":
        return None
    own = model_server_status()
    if own.get("state") != "foreign":
        return None
    shared = shared_max_server_status(port=config.get_server_port())
    if not shared.get("healthy"):
        return None
    loaded = str(shared.get("loaded_model") or "")
    config.set_runtime_settings(
        model=loaded,
        thinking_budget=config.get_thinking_budget(),
        server_mode="shared_max",
        server_port=config.get_server_port(),
    )
    config.adopt_shared_model(loaded)
    return shared


def reconcile_runtime_server(timeout: float = _DEFAULT_SWITCH_TIMEOUT) -> dict:
    """Reconcile current mode without ever mutating a shared MAX server."""
    if config.get_server_mode() == "shared_max":
        status = model_server_status()
        if not status.get("healthy"):
            raise RuntimeError(status.get("error") or "external model server is unavailable")
        return status
    with _switch_lock():
        control = get_model_server_control_settings()
        if control["desired_state"] == "stopped":
            stop_model_server()
            return model_server_status()
        _ensure_model_server_locked(config.get_model(), timeout)
        return model_server_status()


def start_owner_model_server(timeout: float = _DEFAULT_SWITCH_TIMEOUT) -> dict:
    if config.get_server_mode() == "shared_max":
        raise RuntimeError("external model server is read-only; lifecycle controls are unavailable")
    with _switch_lock():
        control = get_model_server_control_settings()
        _write_control_unlocked(
            watchdog_enabled=control["watchdog_enabled"], desired_state="running"
        )
        _ensure_model_server_locked(config.get_model(), timeout)
        return model_server_status()


def stop_owner_model_server() -> dict:
    if config.get_server_mode() == "shared_max":
        raise RuntimeError("external model server is read-only; stop is unavailable")
    with _switch_lock():
        control = get_model_server_control_settings()
        _write_control_unlocked(
            watchdog_enabled=control["watchdog_enabled"], desired_state="stopped"
        )
        stop_model_server()
        return model_server_status()


def restart_owner_model_server(
    timeout: float = _DEFAULT_SWITCH_TIMEOUT,
    *,
    previous_port: int | None = None,
) -> dict:
    if config.get_server_mode() == "shared_max":
        raise RuntimeError("external model server is read-only; reset is unavailable")
    with _switch_lock():
        control = get_model_server_control_settings()
        _write_control_unlocked(
            watchdog_enabled=control["watchdog_enabled"], desired_state="running"
        )
        _restart_model_server_locked(
            config.get_model(), timeout, port=config.get_server_port(), previous_port=previous_port,
        )
        return model_server_status()


def _control_main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1 or args[0] not in {"ensure", "start", "restart", "stop", "status"}:
        print("usage: model_runtime.py {ensure|start|restart|stop|status}", file=sys.stderr)
        return 2
    action = args[0]
    try:
        if action == "ensure":
            payload = reconcile_runtime_server()
        elif action == "start":
            payload = start_owner_model_server()
        elif action == "restart":
            payload = restart_owner_model_server()
        elif action == "stop":
            payload = stop_owner_model_server()
        else:
            payload = model_server_status()
    except Exception as exc:
        print(f"owner control failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_control_main())

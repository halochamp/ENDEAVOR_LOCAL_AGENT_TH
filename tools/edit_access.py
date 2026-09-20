"""Shared persistent Approved Edit Folders and temporary Active Workspace state.

Persistent approvals are explicit user choices. Focus is a separate,
temporary grant and is never copied into the persistent approval list.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable


MAX_APPROVED_EDIT_FOLDERS = 10
_PROJECT_DIR = Path(__file__).resolve().parents[1]
APPROVED_EDIT_FOLDERS_PATH = str(_PROJECT_DIR / "approved_edit_folders.json")
_STATE_LOCK_PATH = f"{APPROVED_EDIT_FOLDERS_PATH}.lock"
_STATE_GUARD = threading.RLock()
_SESSION_FOCUS_FOLDER = ""


def _normalize_folder(value: str, *, must_exist: bool) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("folder path is required")
    if any(ord(char) < 32 for char in raw):
        raise ValueError("folder path contains control characters")
    if not os.path.isabs(raw):
        raise ValueError("folder path must be absolute")
    resolved = os.path.realpath(raw)
    if must_exist and not os.path.isdir(resolved):
        raise ValueError("folder path must be an existing directory")
    return resolved


def _normalize_folders(values: Iterable[str], *, must_exist: bool) -> list[str]:
    normalized: list[str] = []
    for value in values:
        resolved = _normalize_folder(value, must_exist=must_exist)
        if resolved not in normalized:
            normalized.append(resolved)
    if len(normalized) > MAX_APPROVED_EDIT_FOLDERS:
        raise ValueError(f"at most {MAX_APPROVED_EDIT_FOLDERS} folders may be approved")
    return normalized


@contextmanager
def _state_lock(exclusive: bool):
    os.makedirs(os.path.dirname(APPROVED_EDIT_FOLDERS_PATH), exist_ok=True)
    with open(_STATE_LOCK_PATH, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _empty_state() -> dict[str, Any]:
    return {"folders": []}


def _load_state_unlocked() -> dict[str, Any]:
    if not os.path.isfile(APPROVED_EDIT_FOLDERS_PATH):
        return _empty_state()
    try:
        with open(APPROVED_EDIT_FOLDERS_PATH, "r", encoding="utf-8") as state_file:
            payload = json.load(state_file)
    except (OSError, ValueError, TypeError):
        return _empty_state()

    if isinstance(payload, list):
        folders_payload = payload
    elif isinstance(payload, dict):
        folders_payload = payload.get("folders", [])
    else:
        return _empty_state()

    if not isinstance(folders_payload, list):
        folders_payload = []
    try:
        folders = _normalize_folders(folders_payload, must_exist=False)
    except (TypeError, ValueError):
        folders = []

    return {"folders": folders}


def load_edit_access_state() -> dict[str, Any]:
    with _STATE_GUARD:
        with _state_lock(exclusive=False):
            state = _load_state_unlocked()
    with _STATE_GUARD:
        focus = _SESSION_FOCUS_FOLDER
    return {"folders": list(state["folders"]), "focus_folder": focus}


def load_approved_edit_folders() -> list[str]:
    return list(load_edit_access_state()["folders"])


def get_focus_folder() -> str:
    with _STATE_GUARD:
        return str(_SESSION_FOCUS_FOLDER or "")


def _write_state_unlocked(state: dict[str, Any]) -> None:
    state_path = Path(APPROVED_EDIT_FOLDERS_PATH)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{state_path.name}.",
        suffix=".tmp",
        dir=str(state_path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as state_file:
            json.dump(
                {"folders": list(state["folders"])},
                state_file,
                ensure_ascii=False,
                indent=2,
            )
            state_file.write("\n")
            state_file.flush()
            os.fsync(state_file.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, APPROVED_EDIT_FOLDERS_PATH)
    finally:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass


def save_approved_edit_folders(folders: Iterable[str]) -> dict[str, Any]:
    normalized = _normalize_folders(folders, must_exist=True)
    with _STATE_GUARD:
        with _state_lock(exclusive=True):
            current = _load_state_unlocked()
            _write_state_unlocked({"folders": normalized})
    return load_edit_access_state()


def add_approved_edit_folders(folders: Iterable[str]) -> dict[str, Any]:
    additions = _normalize_folders(folders, must_exist=True)
    with _STATE_GUARD:
        with _state_lock(exclusive=True):
            current = _load_state_unlocked()
            merged = list(current["folders"])
            for folder in additions:
                if folder not in merged:
                    merged.append(folder)
            if len(merged) > MAX_APPROVED_EDIT_FOLDERS:
                raise ValueError(f"at most {MAX_APPROVED_EDIT_FOLDERS} folders may be approved")
            _write_state_unlocked({"folders": merged})
    return load_edit_access_state()


def remove_approved_edit_folder(folder: str) -> dict[str, Any]:
    normalized = _normalize_folder(folder, must_exist=False)
    with _STATE_GUARD:
        with _state_lock(exclusive=True):
            current = _load_state_unlocked()
            remaining = [item for item in current["folders"] if item != normalized]
            _write_state_unlocked({"folders": remaining})
    return load_edit_access_state()


def save_focus_folder(folder: str | None) -> dict[str, Any]:
    global _SESSION_FOCUS_FOLDER
    normalized = ""
    if folder:
        normalized = _normalize_folder(folder, must_exist=True)
    with _STATE_GUARD:
        _SESSION_FOCUS_FOLDER = normalized
    return load_edit_access_state()


def path_is_approved_for_edit(path: str) -> bool:
    resolved = os.path.realpath(os.path.abspath(str(path)))
    state = load_edit_access_state()
    candidates = list(state["folders"])
    if state["focus_folder"]:
        candidates.append(state["focus_folder"])
    for folder in candidates:
        try:
            if os.path.commonpath([resolved, folder]) == folder:
                return True
        except ValueError:
            continue
    return False

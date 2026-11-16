# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""User-chosen settings that are not model settings (persisted the moment they change).

Keys:
``internet`` = "online" | "offline".  It is the ONE source of truth for whether the
internet tools (web_search, browse_url, browser_use, recall_web) are bound to the model.
``ocr_engine`` = "apple" | "openthai".  The OCR engine every OCR caller uses (read_image, read_pdf, read_file,
pdf_to_text, computer_use, ...); tools/_ocr.py reads it on each call.  Unset or unknown = "apple".

Why a setting and not a probe: the runtime used to ping 1.1.1.1:53 once at start (1 s, one try)
and decide for the whole session.  One network blip at that instant removed ``web_search`` for
good while the UI's own live probe still showed Internet as up, so an agent asked for the weather
quietly fell back to ``bash`` + ``curl``.  Now the owner picks Internet/Offline in Settings; the
choice is saved here and read by the Agent server and the CLI alike.

File: user_settings.json in the project folder (machine-local, gitignored); override the
path with V2_USER_SETTINGS_PATH (tests).  Missing/unreadable file or an unknown value = online,
the safe default for the tools: only an explicit "offline" hides them.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading

_SCHEMA_VERSION = 1
_LOCK = threading.Lock()
INTERNET_VALUES = ("online", "offline")
OCR_ENGINES = ("apple", "openthai")


def settings_path() -> str:
    return os.getenv("V2_USER_SETTINGS_PATH") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "user_settings.json"
    )


def _read() -> dict:
    try:
        with open(settings_path(), encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def internet_online() -> bool:
    """True unless the owner explicitly chose Offline."""
    return _read().get("internet") != "offline"


def _write(updates: dict) -> None:
    """Merge ``updates`` into the saved file atomically, keeping any other saved keys.  Raises OSError on failure."""
    path = settings_path()
    with _LOCK:
        data = _read()
        data["schema_version"] = _SCHEMA_VERSION
        data.update(updates)
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".user_settings-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=1)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


def set_internet(online: bool) -> None:
    """Persist the choice atomically, keeping any other saved keys.  Raises OSError on failure."""
    if not isinstance(online, bool):
        raise ValueError("online must be a bool")
    _write({"internet": "online" if online else "offline"})


def ocr_engine() -> str:
    """The OCR engine the owner chose; "apple" when unset, unreadable or not a known engine."""
    value = _read().get("ocr_engine")
    return value if value in OCR_ENGINES else "apple"


def set_ocr_engine(engine: str) -> None:
    """Persist the OCR engine atomically, keeping any other saved keys.  ValueError for an unknown engine, OSError on failure."""
    if engine not in OCR_ENGINES:
        raise ValueError(f"ocr engine must be one of {OCR_ENGINES}")
    _write({"ocr_engine": engine})

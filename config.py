# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""config.py — ENDEAVOR_AGENT_V2 configuration

default: Qwen/Qwen3-14B-MLX-4bit @ :8085 via mlx_vlm.server
optional high-quality model: unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit
สลับด้วย shared runtime config หรือ env var — ไม่ต้องแก้ code
"""
from __future__ import annotations
import json
import os
import tempfile
import threading
from pathlib import Path

# Auto-load .env from project root (silent if not found; shell exports take priority)
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# ── Model + backend ───────────────────────────────────────────────────────
_DEFAULT_URL = "http://localhost:8085/v1"
DEFAULT_MODEL = "Qwen/Qwen3-14B-MLX-4bit"
HIGH_QUALITY_MODEL = "unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit"
LOW_RAM_WARNING_BYTES = 24 * 1024 * 1024 * 1024

MLX_BASE_URL = os.getenv("MLX_BASE_URL", _DEFAULT_URL)
# Keep custom-backend behavior explicit: V2_MODEL becomes authoritative only
# when MLX_BASE_URL also points away from the default local endpoint. Normal
# 14B/35B switching on :8085 is owned by the shared runtime settings instead.
_model_env = os.getenv("V2_MODEL")
MODEL = _model_env if (_model_env and MLX_BASE_URL != _DEFAULT_URL) else DEFAULT_MODEL
if MLX_BASE_URL != _DEFAULT_URL and not _model_env:
    import sys
    print(
        f"[config] WARNING: MLX_BASE_URL overridden to {MLX_BASE_URL} but V2_MODEL is not set — "
        f"requesting default model '{DEFAULT_MODEL}' from this non-default server.",
        file=sys.stderr,
    )
API_KEY      = os.getenv("MLX_API_KEY",  "x")  # mlx_vlm.server ใช้ --api-key ได้ แต่ ChatOpenAI ต้องมี non-empty

# ── Runtime model + generation ────────────────────────────────────────────
# The desktop UI may share an already-running :8085 with Agent MAX VLM during
# development. Runtime selection therefore belongs to the TH agent client;
# the Electron host only restarts :8085 when it started that server itself.
MODEL_CHOICES = (
    DEFAULT_MODEL,
    HIGH_QUALITY_MODEL,
)
MODEL_LABELS = {
    DEFAULT_MODEL: "Qwen3 14B · text",
    HIGH_QUALITY_MODEL: "Qwen3.6 35B",
}

TEMPERATURE     = float(os.getenv("V2_TEMPERATURE",     "0.1"))
MAX_TOKENS      = int(os.getenv("V2_MAX_TOKENS",        "4096"))  # 8192→4096: caps thinking+response at ~230s (was 449s); thinking tokens count toward this limit
# mlx_vlm.server accepts thinking_budget as a top-level request field.
THINKING_BUDGET = int(os.getenv("V2_THINKING_BUDGET",   "1536"))
THINKING_BUDGET_LEVELS = (
    ("Low", 256),
    ("Medium", 512),
    ("High", 1024),
    ("xhigh", 1536),
    ("Max", 2048),
)
_THINKING_BUDGET_VALUES = frozenset(value for _label, value in THINKING_BUDGET_LEVELS)
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_runtime_settings_override = os.getenv("V2_RUNTIME_SETTINGS_PATH", "").strip()
_RUNTIME_SETTINGS_PATH = (
    _runtime_settings_override
    if os.path.isabs(_runtime_settings_override)
    else os.path.join(_PROJECT_DIR, _runtime_settings_override)
    if _runtime_settings_override
    else os.path.join(_PROJECT_DIR, "workspace", "runtime_settings.json")
)
_runtime_settings_lock = threading.Lock()
_SHARED_MLX_MODEL = os.getenv("TH_SHARED_MLX_MODEL", "").strip()
_ENV_MODEL_LOCKED = bool(_model_env and MLX_BASE_URL != _DEFAULT_URL)
_LOCKED_MODEL = _SHARED_MLX_MODEL or (MODEL if _ENV_MODEL_LOCKED else "")
_current_model = _LOCKED_MODEL or MODEL
_current_thinking_budget = THINKING_BUDGET


def physical_memory_bytes() -> int:
    """Best-effort physical RAM size for user-facing large-model warnings."""
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total = pages * page_size
        if total > 0:
            return total
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    return 0


def high_quality_model_warning_required(model: str, *, ram_bytes: int | None = None) -> bool:
    """Warn (never block) before selecting 35B on machines below 24 GB RAM."""
    total = physical_memory_bytes() if ram_bytes is None else int(ram_bytes)
    return str(model or "").strip() == HIGH_QUALITY_MODEL and 0 < total < LOW_RAM_WARNING_BYTES


def _runtime_payload(model: str, thinking_budget: int) -> dict:
    return {
        "owner": "agent_th",
        "model": model,
        "thinking_budget": thinking_budget,
    }


def _validate_runtime_values(model: str, thinking_budget: int) -> None:
    if _LOCKED_MODEL:
        if model != _LOCKED_MODEL:
            reason = "shared server" if _SHARED_MLX_MODEL else "environment override"
            raise ValueError(f"model is locked by {reason}: {_LOCKED_MODEL}")
    elif model not in MODEL_CHOICES:
        raise ValueError(f"unsupported model: {model}")
    if thinking_budget not in _THINKING_BUDGET_VALUES:
        raise ValueError(f"unsupported thinking budget: {thinking_budget}")


def _read_runtime_settings_file() -> dict | None:
    try:
        with open(_RUNTIME_SETTINGS_PATH, encoding="utf-8") as f:
            payload = json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("owner") != "agent_th":
        return None
    model = payload.get("model")
    try:
        budget = int(payload.get("thinking_budget"))
    except (TypeError, ValueError):
        return None
    valid_model = model in MODEL_CHOICES or bool(_LOCKED_MODEL and model == _LOCKED_MODEL)
    if not valid_model or budget not in _THINKING_BUDGET_VALUES:
        return None
    return _runtime_payload(model, budget)


def get_persisted_runtime_settings() -> dict | None:
    """Return the validated shared UI config without changing live runtime state."""
    return _read_runtime_settings_file()


def refresh_runtime_settings_from_file() -> bool:
    global _current_model, _current_thinking_budget
    payload = _read_runtime_settings_file()
    if payload is None:
        return False
    model = _LOCKED_MODEL or payload["model"]
    changed = (
        model != _current_model
        or payload["thinking_budget"] != _current_thinking_budget
    )
    _current_model = model
    _current_thinking_budget = payload["thinking_budget"]
    return changed


def get_model() -> str:
    return _current_model


def get_model_label(model: str | None = None) -> str:
    value = model or get_model()
    return MODEL_LABELS.get(value, value.split("/")[-1])


def get_thinking_budget() -> int:
    return _current_thinking_budget


def get_thinking_budget_label(thinking_budget: int | None = None) -> str:
    value = get_thinking_budget() if thinking_budget is None else int(thinking_budget)
    for label, candidate in THINKING_BUDGET_LEVELS:
        if candidate == value:
            return label
    return str(value)


def get_runtime_settings() -> dict:
    if _LOCKED_MODEL:
        prefix = "Shared" if _SHARED_MLX_MODEL else "Env"
        model_options = [{
            "value": _LOCKED_MODEL,
            "label": f"{prefix} · {MODEL_LABELS.get(_LOCKED_MODEL, _LOCKED_MODEL)}",
        }]
    else:
        model_options = [
            {"value": model, "label": MODEL_LABELS[model]}
            for model in MODEL_CHOICES
        ]
    return {
        "model": get_model(),
        "thinking_budget": get_thinking_budget(),
        "shared_server": bool(_SHARED_MLX_MODEL),
        "model_locked": bool(_LOCKED_MODEL),
        "model_options": model_options,
        "thinking_options": [
            {"label": label, "value": value}
            for label, value in THINKING_BUDGET_LEVELS
        ],
    }


def set_runtime_settings(*, model: str, thinking_budget: int) -> dict:
    """Persist TH client selection without taking ownership of the MLX server."""
    global _current_model, _current_thinking_budget
    model = str(model or "").strip()
    thinking_budget = int(thinking_budget)
    _validate_runtime_values(model, thinking_budget)
    with _runtime_settings_lock:
        old_model = _current_model
        old_budget = _current_thinking_budget
        payload = _runtime_payload(model, thinking_budget)
        state_dir = os.path.dirname(_RUNTIME_SETTINGS_PATH)
        os.makedirs(state_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".runtime_settings.", suffix=".tmp", dir=state_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, sort_keys=True)
            os.replace(tmp_path, _RUNTIME_SETTINGS_PATH)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        _current_model = model
        _current_thinking_budget = thinking_budget
    return {
        "model_changed": model != old_model,
        "thinking_budget_changed": thinking_budget != old_budget,
        **get_runtime_settings(),
    }


refresh_runtime_settings_from_file()
# Penalises repeated tokens over a 20-token window — breaks thinking loops at root cause.
# 1.1: raised from 1.05 — disrupts repetitive thinking loops faster without corrupting tool JSON.
# 0.0 = disabled (server default). Lower to 1.02 if JSON breaks.
REPETITION_PENALTY = float(os.getenv("V2_REPETITION_PENALTY", "1.1"))

# Single-node design: 1 loop ครอบ create_plan + N steps × ~3 tool calls + synthesis
# 4-step research ≈ 1 + 12 + 1 ≈ 14, ใส่ headroom เป็น 50 (advisor Q5)
RECURSION_LIMIT = int(os.getenv("V2_RECURSION_LIMIT", "60"))

# Context window management — trim messages ก่อนส่งให้ agent
# 35B-A3B context = 262K tokens ≈ ~1M chars; เก็บ 200K chars (~50K tokens) เผื่อ tool results
# ปรับได้: export V2_CONTEXT_MAX_CHARS=300000 ถ้าต้องการ session ยาวขึ้น
CONTEXT_MAX_CHARS = int(os.getenv("V2_CONTEXT_MAX_CHARS", "200000"))

# ── Web cache + summarization ─────────────────────────────────────────────
# Process-level in-memory cache for web tool outputs (web_search / browse_url /
# browser_use). Raw bodies are stored once; messages only carry compact
# summaries. Agent can call recall_web(url) to fetch full content back.
WEB_CACHE_MAX_ENTRIES   = int(os.getenv("V2_WEB_CACHE_MAX_ENTRIES",   "50"))
WEB_CACHE_MAX_BYTES     = int(os.getenv("V2_WEB_CACHE_MAX_BYTES",     "3000000"))
WEB_CACHE_PER_ENTRY_MAX = int(os.getenv("V2_WEB_CACHE_PER_ENTRY_MAX", "50000"))
SUMMARY_MAX_CHARS       = int(os.getenv("V2_SUMMARY_MAX_CHARS",       "800"))
# Skip LLM summarization if raw ≤ this — saves ~15-40s per call.
# Most web_search trafilatura results are ≤1500 chars → skip LLM entirely
SUMMARY_SKIP_LLM_BELOW  = int(os.getenv("V2_SUMMARY_SKIP_LLM_BELOW",  "1500"))
RECALL_WEB_MAX_CHARS    = int(os.getenv("V2_RECALL_WEB_MAX_CHARS",    "20000"))

# ── Web tool limits ───────────────────────────────────────────────────────
BROWSE_URL_MAX_CHARS     = int(os.getenv("V2_BROWSE_URL_MAX_CHARS",     "20000"))
BROWSER_USE_MAX_CHARS    = int(os.getenv("V2_BROWSER_USE_MAX_CHARS",    "8000"))
WEB_SEARCH_MAX_RESULTS   = int(os.getenv("V2_WEB_SEARCH_MAX_RESULTS",   "10"))
WEB_SEARCH_FETCH_TOP     = int(os.getenv("V2_WEB_SEARCH_FETCH_TOP",     "4"))
WEB_SEARCH_MAX_CHARS_URL = int(os.getenv("V2_WEB_SEARCH_MAX_CHARS_URL", "1500"))
WEB_SEARCH_FETCH_TIMEOUT = int(os.getenv("V2_WEB_SEARCH_FETCH_TIMEOUT", "6"))
BATCH_BROWSE_MAX_WORKERS = int(os.getenv("V2_BATCH_BROWSE_MAX_WORKERS", "2"))
BATCH_BROWSE_MAX_URLS    = int(os.getenv("V2_BATCH_BROWSE_MAX_URLS",    "8"))
SUMMARY_MAX_TOKENS       = int(os.getenv("V2_SUMMARY_MAX_TOKENS",       "1024"))
SUMMARY_BATCH_MAX_TOKENS = int(os.getenv("V2_SUMMARY_BATCH_MAX_TOKENS", "3072"))
FETCH_SITEMAP_MAX_URLS   = int(os.getenv("V2_FETCH_SITEMAP_MAX_URLS",   "200"))
FETCH_SITEMAP_SHOWN      = int(os.getenv("V2_FETCH_SITEMAP_SHOWN",      "30"))  # URLs shown to the model per call
SCRAPE_TABLE_MAX_CHARS   = int(os.getenv("V2_SCRAPE_TABLE_MAX_CHARS",   "10000"))

# ── File + loop tool limits ───────────────────────────────────────────────
READ_FILE_MAX_CHARS                = int(os.getenv("V2_READ_FILE_MAX_CHARS",                "10000"))
READ_FILE_MAX_BYTES                = int(os.getenv("V2_READ_FILE_MAX_BYTES",                str(50 * 1024 * 1024)))
READ_FILE_AUDIO_VIDEO_MAX_BYTES        = int(os.getenv("V2_READ_FILE_AUDIO_VIDEO_MAX_BYTES",        str(500 * 1024 * 1024)))
READ_FILE_AUDIO_VIDEO_MAX_DURATION_SEC = int(os.getenv("V2_READ_FILE_AUDIO_VIDEO_MAX_DURATION_SEC", str(90 * 60)))
TOOL_LOOP_DDG_MAX_RESULTS          = int(os.getenv("V2_TOOL_LOOP_DDG_MAX_RESULTS",          "8"))
TOOL_LOOP_READ_MAX_CHARS           = int(os.getenv("V2_TOOL_LOOP_READ_MAX_CHARS",           "6000"))
TOOL_LOOP_READ_SUMMARIZE_THRESHOLD = int(os.getenv("V2_TOOL_LOOP_READ_SUMMARIZE_THRESHOLD", "3000"))
TOOL_LOOP_BASH_MAX_CHARS           = int(os.getenv("V2_TOOL_LOOP_BASH_MAX_CHARS",           "2000"))
TOOL_LOOP_BASH_TIMEOUT             = int(os.getenv("V2_TOOL_LOOP_BASH_TIMEOUT",             "30"))

# ── Agent server ──────────────────────────────────────────────────────────
SERVER_PORT   = int(os.getenv("AGENT_SERVER_PORT",   "8765"))
AUTH_DISABLED = os.getenv("AGENT_AUTH_DISABLED") == "1"

# ── Workspace ─────────────────────────────────────────────────────────────
WORKSPACE = os.getenv("V2_WORKSPACE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace"))
os.makedirs(WORKSPACE, exist_ok=True)

# ── MCP client ─────────────────────────────────────────────────────────────
# Developer-provisioned servers are optional and empty by default in this public
# release. Users can register Streamable HTTP or guarded local stdio servers at
# runtime through mcp_add_server; those entries live under workspace/tool_mcp/.
MCP_SERVERS: dict[str, dict] = {}
MCP_MAX_CHARS = int(os.getenv("V2_MCP_MAX_CHARS", "4000"))
MCP_TIMEOUT   = int(os.getenv("V2_MCP_TIMEOUT",   "60"))

# ── Activity logging ───────────────────────────────────────────────────────
LOG_DIR         = os.getenv("V2_LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
LOG_MAX_ENTRIES = int(os.getenv("V2_LOG_MAX_ENTRIES", "5000"))
os.makedirs(LOG_DIR, exist_ok=True)

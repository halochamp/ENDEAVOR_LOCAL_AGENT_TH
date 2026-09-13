# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""config.py — ENDEAVOR_AGENT_V2 configuration

default: Qwen/Qwen3-14B-MLX-4bit @ :8085 via mlx_vlm.server
compact VLM option: mlx-community/Qwen3.5-9B-4bit
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
_DEFAULT_SERVER_PORT = 8085
_SHARED_MAX_TEST_PORT = 8085
_DEFAULT_URL = f"http://localhost:{_DEFAULT_SERVER_PORT}/v1"
DEFAULT_MODEL = "Qwen/Qwen3-14B-MLX-4bit"
COMPACT_VLM_MODEL = "mlx-community/Qwen3.5-9B-4bit"
HIGH_QUALITY_MODEL = "unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit"
LOW_RAM_WARNING_BYTES = 24 * 1024 * 1024 * 1024
SERVER_MODES = ("standalone", "shared_max")
SERVER_MODE_LABELS = {
    "standalone": "Standalone · Agent TH owns server",
    "shared_max": "Shared MAX test server · read-only",
}

# Explicit MLX_BASE_URL + V2_MODEL remains a development/custom-backend override.
# Without that override, Agent TH owns the local endpoint and may move its
# standalone port at runtime. The special shared_max mode is read-only and
# points at the configured MAX test-server port (default :8085).
_MLX_BASE_URL_OVERRIDE = os.getenv("MLX_BASE_URL", "").strip()
MLX_BASE_URL = _MLX_BASE_URL_OVERRIDE or _DEFAULT_URL
_model_env = os.getenv("V2_MODEL")
MODEL = _model_env if (_model_env and _MLX_BASE_URL_OVERRIDE) else DEFAULT_MODEL
if _MLX_BASE_URL_OVERRIDE and not _model_env:
    import sys
    print(
        f"[config] WARNING: MLX_BASE_URL overridden to {MLX_BASE_URL} but V2_MODEL is not set — "
        f"requesting default model '{DEFAULT_MODEL}' from this non-default server.",
        file=sys.stderr,
    )
API_KEY      = os.getenv("MLX_API_KEY",  "x")  # mlx_vlm.server ใช้ --api-key ได้ แต่ ChatOpenAI ต้องมี non-empty

# ── Runtime model + generation ────────────────────────────────────────────
# Agent TH owns its standalone model-server lifecycle. A special read-only
# shared-test mode may attach to Agent MAX VLM's current test server (default
# :8085) without taking process/model ownership from MAX VLM.
MODEL_CHOICES = (
    DEFAULT_MODEL,
    COMPACT_VLM_MODEL,
    HIGH_QUALITY_MODEL,
)
MODEL_LABELS = {
    DEFAULT_MODEL: "Qwen3 14B · text",
    COMPACT_VLM_MODEL: "Qwen3.5 9B · VLM",
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
_ENV_MODEL_LOCKED = bool(_model_env and _MLX_BASE_URL_OVERRIDE)
_LOCKED_MODEL = MODEL if _ENV_MODEL_LOCKED else ""
_current_model = _LOCKED_MODEL or MODEL
_current_standalone_model = _current_model
_current_thinking_budget = THINKING_BUDGET
_current_server_mode = "standalone"
_current_server_port = _DEFAULT_SERVER_PORT


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


def _validate_server_port(server_port: int) -> int:
    port = int(server_port)
    if not 1024 <= port <= 65535:
        raise ValueError(f"unsupported model server port: {port}")
    return port


def _runtime_payload(
    model: str,
    thinking_budget: int,
    server_mode: str,
    server_port: int,
    standalone_model: str,
) -> dict:
    return {
        "owner": "agent_th",
        "model": model,
        "standalone_model": standalone_model,
        "thinking_budget": int(thinking_budget),
        "server_mode": str(server_mode),
        "server_port": int(server_port),
    }


def _validate_runtime_values(model: str, thinking_budget: int, server_mode: str, server_port: int) -> None:
    if _LOCKED_MODEL and model != _LOCKED_MODEL:
        raise ValueError(f"model is locked by environment override: {_LOCKED_MODEL}")
    if not _LOCKED_MODEL and model not in MODEL_CHOICES:
        raise ValueError(f"unsupported model: {model}")
    if int(thinking_budget) not in _THINKING_BUDGET_VALUES:
        raise ValueError(f"unsupported thinking budget: {thinking_budget}")
    if str(server_mode) not in SERVER_MODES:
        raise ValueError(f"unsupported server mode: {server_mode}")
    _validate_server_port(server_port)


def _read_runtime_settings_file() -> dict | None:
    try:
        with open(_RUNTIME_SETTINGS_PATH, encoding="utf-8") as f:
            payload = json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("owner") != "agent_th":
        return None
    model = payload.get("model")
    standalone_model = payload.get("standalone_model", model)
    try:
        budget = int(payload.get("thinking_budget"))
        mode = str(payload.get("server_mode") or "standalone")
        port = _validate_server_port(payload.get("server_port", _DEFAULT_SERVER_PORT))
        _validate_runtime_values(model, budget, mode, port)
        if standalone_model not in MODEL_CHOICES:
            raise ValueError("invalid standalone model")
    except (TypeError, ValueError):
        return None
    return _runtime_payload(model, budget, mode, port, standalone_model)


def get_persisted_runtime_settings() -> dict | None:
    """Return validated persisted owner settings without changing live state."""
    return _read_runtime_settings_file()


def refresh_runtime_settings_from_file() -> bool:
    global _current_model, _current_standalone_model, _current_thinking_budget, _current_server_mode, _current_server_port, MLX_BASE_URL
    if _MLX_BASE_URL_OVERRIDE:
        return False
    payload = _read_runtime_settings_file()
    if payload is None:
        return False
    model = _LOCKED_MODEL or payload["model"]
    changed = (
        model != _current_model
        or payload["thinking_budget"] != _current_thinking_budget
        or payload["standalone_model"] != _current_standalone_model
        or payload["server_mode"] != _current_server_mode
        or payload["server_port"] != _current_server_port
    )
    _current_model = model
    _current_standalone_model = payload["standalone_model"]
    _current_thinking_budget = payload["thinking_budget"]
    _current_server_mode = payload["server_mode"]
    _current_server_port = payload["server_port"]
    MLX_BASE_URL = get_mlx_base_url()
    return changed


def get_model() -> str:
    return _current_model


def adopt_shared_model(model: str) -> bool:
    """Use the model observed on the read-only shared MAX server for this process.

    This is intentionally transient: switching back to standalone persists an
    explicit owner selection through ``set_runtime_settings``.
    """
    global _current_model
    model = str(model or "").strip()
    if model not in MODEL_CHOICES:
        raise ValueError(f"unsupported shared MAX model: {model}")
    changed = model != _current_model
    _current_model = model
    return changed


def get_standalone_model() -> str:
    return _current_standalone_model


def get_model_label(model: str | None = None) -> str:
    value = model or get_model()
    return MODEL_LABELS.get(value, value.split("/")[-1])


def get_server_mode() -> str:
    return str(_current_server_mode)


def get_server_port() -> int:
    return int(_current_server_port)


def get_mlx_base_url() -> str:
    if _MLX_BASE_URL_OVERRIDE:
        return _MLX_BASE_URL_OVERRIDE
    return f"http://localhost:{get_server_port()}/v1"


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
        model_options = [{
            "value": _LOCKED_MODEL,
            "label": f"Env · {MODEL_LABELS.get(_LOCKED_MODEL, _LOCKED_MODEL)}",
        }]
    elif get_server_mode() == "shared_max":
        model_options = [{
            "value": get_model(),
            "label": f"Shared MAX · {MODEL_LABELS.get(get_model(), get_model())}",
        }]
    else:
        model_options = [
            {"value": model, "label": MODEL_LABELS[model]}
            for model in MODEL_CHOICES
        ]
    return {
        "model": get_model(),
        "standalone_model": get_standalone_model(),
        "thinking_budget": get_thinking_budget(),
        "server_mode": get_server_mode(),
        "server_port": get_server_port(),
        "shared_server": get_server_mode() == "shared_max",
        "model_locked": bool(_LOCKED_MODEL) or get_server_mode() == "shared_max",
        "runtime_locked": bool(_MLX_BASE_URL_OVERRIDE),
        "model_options": model_options,
        "server_mode_options": [
            {"value": mode, "label": SERVER_MODE_LABELS[mode]}
            for mode in SERVER_MODES
        ],
        "thinking_options": [
            {"label": label, "value": value}
            for label, value in THINKING_BUDGET_LEVELS
        ],
    }


def set_runtime_settings(
    *,
    model: str,
    thinking_budget: int,
    server_mode: str | None = None,
    server_port: int | None = None,
) -> dict:
    """Persist Agent TH model/budget/server-mode/port atomically."""
    global _current_model, _current_standalone_model, _current_thinking_budget, _current_server_mode, _current_server_port, MLX_BASE_URL
    if _MLX_BASE_URL_OVERRIDE:
        raise ValueError("runtime UI selection is disabled when MLX_BASE_URL is overridden")
    model = str(model or "").strip()
    thinking_budget = int(thinking_budget)
    server_mode = get_server_mode() if server_mode is None else str(server_mode or "").strip()
    server_port = get_server_port() if server_port is None else _validate_server_port(server_port)
    _validate_runtime_values(model, thinking_budget, server_mode, server_port)
    with _runtime_settings_lock:
        old_model = _current_model
        old_budget = _current_thinking_budget
        old_mode = _current_server_mode
        old_port = _current_server_port
        standalone_model = model if server_mode == "standalone" else _current_standalone_model
        payload = _runtime_payload(
            model, thinking_budget, server_mode, server_port, standalone_model,
        )
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
        _current_standalone_model = standalone_model
        _current_thinking_budget = thinking_budget
        _current_server_mode = server_mode
        _current_server_port = server_port
        MLX_BASE_URL = get_mlx_base_url()
    return {
        "model_changed": model != old_model,
        "thinking_budget_changed": thinking_budget != old_budget,
        "server_mode_changed": server_mode != old_mode,
        "server_port_changed": server_port != old_port,
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
READ_FILE_BATCH_MAX_FILES          = int(os.getenv("V2_READ_FILE_BATCH_MAX_FILES",          "4"))
READ_FILE_BATCH_MAX_CHARS          = int(os.getenv("V2_READ_FILE_BATCH_MAX_CHARS",          "20000"))
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

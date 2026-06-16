# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""config.py — ENDEAVOR_AGENT_V2 configuration

default: unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit @ :8080 (MoE, bench 6/6, 7.7× faster than 27B)
dev:     export V2_MODEL="mlx-community/Qwen3-1.7B-4bit" MLX_BASE_URL="http://localhost:8888/v1"
สลับด้วย env var — ไม่ต้องแก้ code
"""
from __future__ import annotations
import os
from pathlib import Path

# Auto-load .env from project root (silent if not found; shell exports take priority)
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# ── Model + backend ───────────────────────────────────────────────────────
_PROD_URL   = "http://localhost:8080/v1"
_PROD_MODEL = "unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit"

MLX_BASE_URL = os.getenv("MLX_BASE_URL", _PROD_URL)
# MODEL ใช้ production model เสมอ ยกเว้นตอน dev ที่เปลี่ยน MLX_BASE_URL ด้วย
# (ป้องกัน mlx_lm.server โหลด model ผิดเมื่อ V2_MODEL ถูก override โดยไม่เปลี่ยน URL)
_model_env = os.getenv("V2_MODEL")
MODEL = _model_env if (_model_env and MLX_BASE_URL != _PROD_URL) else _PROD_MODEL
if MLX_BASE_URL != _PROD_URL and not _model_env:
    import sys
    print(
        f"[config] WARNING: MLX_BASE_URL overridden to {MLX_BASE_URL} but V2_MODEL is not set — "
        f"requesting production model '{_PROD_MODEL}' from this non-default server.",
        file=sys.stderr,
    )
API_KEY      = os.getenv("MLX_API_KEY",  "x")  # mlx_lm.server ไม่เช็ค แต่ ChatOpenAI ต้องมี non-empty

# ── Generation ────────────────────────────────────────────────────────────
TEMPERATURE     = float(os.getenv("V2_TEMPERATURE",     "0.1"))
MAX_TOKENS      = int(os.getenv("V2_MAX_TOKENS",        "4096"))  # 8192→4096: caps thinking+response at ~230s (was 449s); thinking tokens count toward this limit
# NOTE: thinking_budget is a no-op — mlx_lm 0.31.3 does not implement it.
THINKING_BUDGET = int(os.getenv("V2_THINKING_BUDGET",   "2048"))
# Penalises repeated tokens over a 20-token window — breaks thinking loops at root cause.
# 1.1: raised from 1.05 — disrupts repetitive thinking loops faster without corrupting tool JSON.
# 0.0 = disabled (mlx_lm default). Lower to 1.02 if JSON breaks.
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
SCRAPE_TABLE_MAX_CHARS   = int(os.getenv("V2_SCRAPE_TABLE_MAX_CHARS",   "10000"))

# ── File + loop tool limits ───────────────────────────────────────────────
READ_FILE_MAX_CHARS                = int(os.getenv("V2_READ_FILE_MAX_CHARS",                "10000"))
READ_FILE_MAX_BYTES                = int(os.getenv("V2_READ_FILE_MAX_BYTES",                str(5 * 1024 * 1024)))
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

# ── Activity logging ───────────────────────────────────────────────────────
LOG_DIR         = os.getenv("V2_LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
LOG_MAX_ENTRIES = int(os.getenv("V2_LOG_MAX_ENTRIES", "5000"))
os.makedirs(LOG_DIR, exist_ok=True)

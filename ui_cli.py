"""ui.py — Terminal UI for ENDEAVOR_AGENT_V2

ใช้ rich สำหรับ markdown rendering + ANSI colors + Spinner
"""
from __future__ import annotations
import sys
import threading
import time
import itertools
import re

from prompt_toolkit.application import Application as _Application
from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.buffer import Buffer as _Buffer
from prompt_toolkit.formatted_text import ANSI as _ANSI
from prompt_toolkit.key_binding import KeyBindings as _KeyBindings, merge_key_bindings as _merge_kb
from prompt_toolkit.key_binding.defaults import load_key_bindings as _load_kb
from prompt_toolkit.layout import Layout as _Layout, HSplit as _HSplit, Window as _Window
from prompt_toolkit.layout.controls import BufferControl as _BufferControl, FormattedTextControl as _FTControl
from prompt_toolkit.layout.processors import AppendAutoSuggestion as _AppendAutoSuggestion

_skill_completions: list[str] = []


class _SkillAutoSuggest(AutoSuggest):
    """Ghost-text suggestion for /skill commands — shows remainder as dimmed inline text."""

    def get_suggestion(self, buffer, document):
        text = document.text_before_cursor
        if text.startswith("/") and len(text) > 1:
            for c in _skill_completions:
                if c.startswith(text) and len(c) > len(text):
                    return Suggestion(c[len(text):])
        return None


def setup_skill_completer(skill_names: list[str]) -> None:
    # Each prompt_user() call builds a fresh Buffer with _SkillAutoSuggest(),
    # which reads this module-level list — so updating it here is enough.
    global _skill_completions
    _skill_completions = [f"/{n}" for n in skill_names]

from rich.console import Console
from rich.markdown import Markdown
from rich.theme import Theme

_theme = Theme({
    "markdown.h1":         "bold white",
    "markdown.h2":         "bold white",
    "markdown.h3":         "bold white",
    "markdown.code":       "bold cyan",
    "markdown.code_block": "dim",
})
_console = Console(theme=_theme, highlight=False)

# ── ANSI colors ─────────────────────────────────────────────────────────────
R      = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

C_USER   = "\033[38;5;75m"    # blue  — You
C_AGENT  = "\033[38;5;86m"    # green — Endeavor
C_TOOL   = "\033[38;5;208m"   # orange — tool calls
C_META   = "\033[38;5;244m"   # gray  — metadata
C_ARROW  = "\033[38;5;240m"   # dark gray
C_WARN   = "\033[38;5;214m"   # amber
C_DIM    = "\033[38;5;237m"   # very dark — divider
C_HEADER = "\033[38;5;252m"   # near-white
C_ORANGE = "\033[38;5;208m"   # prompt marker
C_REF    = "\033[38;5;69m"    # blue-ish — references

WIDTH = 62

# ── Tool labels ──────────────────────────────────────────────────────────────
_TOOL_LABELS: dict[str, str] = {
    "web_search":        "Web Search",
    "browse_url":        "Browse",
    "browser_use":       "Browser",
    "recall_web":        "Recall",
    "bash":              "Bash",
    "python_exec":       "Python",
    "plot":              "Plot",
    "read_file":         "Read",
    "write_file":        "Write",
    "edit":              "Edit",
    "grep":              "Grep",
    "workspace_ls":      "Workspace",
    "create_plan":       "Planning",
    "remember":          "Remember",
    "scratch_write":     "Note",
    "scratch_read":      "Read Notes",
    "scratch_clear":     "Clear Notes",
}

_SPINNER_LABELS: dict[str, str] = {
    "web_search":        "web_search",
    "browse_url":        "browse_url",
    "browser_use":       "browser_use",
    "recall_web":        "recall_web",
    "bash":              "bash",
    "python_exec":       "python_exec",
    "plot":              "plot",
    "read_file":         "read_file",
    "write_file":        "write_file",
    "edit":              "edit",
    "grep":              "grep",
    "create_plan":       "create_plan",
    "workspace_ls":      "workspace_ls",
    "remember":          "remember",
    "scratch_write":     "scratch_write",
    "scratch_read":      "scratch_read",
    "scratch_clear":     "scratch_clear",
    "tool_loop":         "tool_loop",
}

# Default phase labels
PHASE_THINKING  = "thinking…"
PHASE_EXECUTING = "executing…"
PHASE_SYNTH     = "synthesizing…"


# All stdout access (spinner frames + main-thread prints) is serialized through
# this lock so the animating spinner thread and the main thread never interleave
# cursor-control writes. RLock so live_print's fn may itself print safely.
_IO_LOCK = threading.RLock()


# ── Spinner ───────────────────────────────────────────────────────────────────
class Spinner:
    """Two-line spinner: main label (phase) + sub-status (tool internals).

    Layout (terminal):
        ⠹ <main label>
              ⎿  <sub-status>

    Both lines are refreshed via cursor-up + clear, so output stays in place.

    ONE spinner thread is created per turn (see main._Turn): tool events only
    mutate `label`/`sub`, never start/stop the thread. This avoids per-tool
    thread churn — under concurrent tool dispatch (langgraph ToolNode runs
    multiple tool calls in parallel threads) a start/stop-per-tool design
    orphaned spinner threads whose `_stop` was never set (B2 hang).
    """
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, label: str = "Thinking…"):
        self.label    = label
        self.sub      = ""
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._spin, daemon=True, name="spinner")
        self._lines   = 0  # how many lines we've drawn (0 = nothing yet)

    def _clear(self) -> None:
        # Erase exactly _lines lines from bottom up. Caller holds _IO_LOCK.
        if self._lines == 0:
            sys.stdout.write("\r\033[2K")
        else:
            for i in range(self._lines):
                sys.stdout.write("\r\033[2K")
                if i < self._lines - 1:
                    sys.stdout.write("\033[F")
        sys.stdout.flush()

    def _draw(self, frame: str) -> None:
        # Caller holds _IO_LOCK.
        self._clear()
        n = 0
        sys.stdout.write(f"   {C_META}{frame} {self.label}{R}")
        n += 1
        if self.sub:
            sys.stdout.write(f"\n      {C_ARROW}{DIM}⎿  {self.sub[:60]}{R}")
            n += 1
        # Box frame below status — top border / empty (user's spot) / bottom border
        sys.stdout.write(f"\n{_BORDER}\n\n{_BORDER}\n")
        n += 3
        sys.stdout.write(_render_ctx_bar())
        n += 1
        self._lines = n
        sys.stdout.flush()

    def _spin(self) -> None:
        for frame in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            with _IO_LOCK:
                if self._stop.is_set():  # re-check under lock — stop() may have fired
                    break
                self._draw(frame)
            time.sleep(0.08)
        with _IO_LOCK:
            self._clear()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=2.0)  # don't block forever if a write is mid-flight

    def update(self, label: str) -> None:
        """Change main label (phase)."""
        self.label = label

    def update_sub(self, msg: str) -> None:
        """Change sub-status (tool internals)."""
        self.sub = msg or ""

    def live_print(self, fn) -> None:
        """Run a main-thread print without racing the spinner: clear the spinner
        lines, run `fn` (which prints permanent output), then let the next frame
        redraw the spinner below it. Serialized via _IO_LOCK."""
        with _IO_LOCK:
            self._clear()
            self._lines = 0
            fn()


# ── Header ────────────────────────────────────────────────────────────────────
def print_header(model: str, tool_count: int, online: bool = True) -> None:
    short = model.split("/")[-1]
    net = f"\033[38;5;82m● online{R}" if online else f"\033[38;5;196m● offline{R}"
    # inner content width = WIDTH - 2  (one ║ each side)
    inner = WIDTH - 2
    bar = f" {C_DIM}{'═' * WIDTH}{R}"
    # padding per line = inner - visible chars (excluding ANSI)
    pad_title = inner - 22                        # "  ENDEAVOR AGENT CLI TH" = 22
    pad_model = max(0, inner - 9 - len(short))    # "  Model  " = 9
    pad_power = max(0, inner - 44)                # "  Powered by HaloChamp  champoomwat@gmail.com" = 44
    print()
    print(bar)
    print(f" {C_DIM}║{R}  {BOLD}\033[38;5;220mENDEAVOR AGENT CLI TH{R}"
          f"{C_DIM}{'':>{pad_title}}║{R}")
    print(f" {C_DIM}║{R}  {C_META}Model  {R}{BOLD}{C_AGENT}{short}{R}"
          f"{C_DIM}{'':>{pad_model}}║{R}")
    print(f" {C_DIM}║{R}  {C_META}Powered by {R}{C_HEADER}HaloChamp{R}"
          f"  {C_META}champoomwat@gmail.com{R}"
          f"{C_DIM}{'':>{pad_power}}║{R}")
    print(bar)
    print(f"   {C_META}{tool_count} tools{R}  {net}")
    print()


def print_startup_hint() -> None:
    print(f"   {C_DIM}Type {R}{C_ORANGE}menu{R}{C_DIM} to see options{R}")
    print()


# ── Divider ────────────────────────────────────────────────────────────────────
def print_divider() -> None:
    print(f"\n{C_DIM}{'╌' * (WIDTH // 2)}{R}\n")


# ── User prompt ────────────────────────────────────────────────────────────────
def print_user_prompt(text: str, mode: str = "") -> None:
    mode_tag = f" {C_DIM}[{mode}]{R}" if mode else ""
    print(f" {C_USER}{BOLD}You{R}{mode_tag}  {text}")
    print()


# ── Tool step (⎿ style) ────────────────────────────────────────────────────────
def print_tool_step(name: str, detail: str = "") -> None:
    label = _TOOL_LABELS.get(name, name)
    line = f"   {C_TOOL}⎿ {label}{R}"
    if detail:
        line += f"  {C_META}{DIM}{detail[:60]}{R}"
    print(f"\n{line}")


# ── Plan block ────────────────────────────────────────────────────────────────
def print_plan(steps: list[str]) -> None:
    print(f"\n   {C_META}แผน · {len(steps)} ขั้นตอน{R}")
    for i, s in enumerate(steps, 1):
        print(f"   {C_ARROW}{DIM} {i}. {s[:65]}{R}")
    print(f"   {C_META}(กำลังดำเนินการ…){R}")


# ── Synthesis indicator ────────────────────────────────────────────────────────
def print_synthesizing() -> None:
    sys.stdout.write(f"\n   {C_TOOL}⎿ Synthesizing…{R}\n")
    sys.stdout.flush()


# ── Agent response (rich markdown) ────────────────────────────────────────────
def print_agent_response(text: str) -> None:
    print()
    _console.print(Markdown(text), end="\n")


# ── Web references ────────────────────────────────────────────────────────────
def print_web_refs(refs: list[tuple[str, str]]) -> None:
    """refs = [(title, url), ...]"""
    if not refs:
        return
    print(f"\n   {C_META}📎 แหล่งอ้างอิง{R}")
    for i, (title, url) in enumerate(refs, 1):
        short_title = title[:55] if title else url
        print(f"   {C_ARROW}{DIM} {i}. {short_title}{R}")
        print(f"      {C_REF}{url}{R}")


def extract_refs_from_search_result(text: str) -> list[tuple[str, str]]:
    """ดึง (title, url) จาก web_search ToolMessage — format: [web:<url>] summary"""
    refs: list[tuple[str, str]] = []
    for line in (text or "").splitlines():
        # Current format: [web:<url>] summary
        m = re.match(r"^\[web:(https?://[^\]]+)\]\s*(.*)", line)
        if m:
            url = m.group(1)
            summary = m.group(2).strip()
            title = summary[:55] if summary else url
            refs.append((title, url))
    return refs


C_GREEN = "\033[38;5;82m"
C_RED   = "\033[38;5;196m"

# ── Context status bar ────────────────────────────────────────────────────────
from config import CONTEXT_MAX_CHARS as _CONTEXT_MAX_CHARS
_ctx_info: dict = {"chars": 0, "max_chars": _CONTEXT_MAX_CHARS}


def update_ctx_info(chars: int, max_chars: int) -> None:
    _ctx_info["chars"] = chars
    _ctx_info["max_chars"] = max_chars


def _render_ctx_bar() -> str:
    chars = _ctx_info["chars"]
    max_c = _ctx_info["max_chars"]
    pct   = chars / max_c * 100 if max_c > 0 else 0

    if pct >= 90:
        c_pct = C_RED
    elif pct >= 70:
        c_pct = C_WARN
    else:
        c_pct = C_META

    # Progress bar: 18 blocks
    filled = round(pct / 100 * 18)
    bar = "▓" * filled + "░" * (18 - filled)

    def _fmt(v: int) -> str:
        return f"{v / 1000:.1f}k" if v >= 1000 else str(v)

    return (
        f" {C_DIM}⎿{R}  "
        f"{c_pct}{pct:.0f}%{R}  "
        f"{C_DIM}{bar}{R}  "
        f"{C_META}{_fmt(chars)} / {_fmt(max_c)} chars{R}"
    )


def print_compact_notice(n_msgs: int) -> None:
    print(f"   {C_META}⎿ context compacted — {n_msgs} messages → summary{R}")


# ── Mode menu ────────────────────────────────────────────────────────────────────
def print_mode_menu() -> None:
    print()
    print(f" {C_DIM}{'─' * (WIDTH - 2)}{R}")
    print(f"   {C_ORANGE}[1]{R} Build RAG Index")
    print(f"   {C_ORANGE}[2]{R} Skills")
    print(f"   {C_ORANGE}[3]{R} Special Commands")
    print(f"   {C_ORANGE}[q]{R} Quit")
    print(f" {C_DIM}{'─' * (WIDTH - 2)}{R}")
    print()


def print_special_commands(builtin_cmds: list | None = None) -> None:
    cmds = [(f"/{c['name']}", c["desc"]) for c in (builtin_cmds or [])]
    cmds += [
        ("menu",                "เปิด menu นี้"),
        ("exit / quit / ออก",  "ออกจากโปรแกรม"),
    ]
    aliases = [
        ("โหลดความจำ",  "/history"),
        ("load history", "/history"),
        ("โหลด history", "/history"),
        ("จำเก่า",      "/history"),
    ]
    print()
    print(f" {C_DIM}{'─' * (WIDTH - 2)}{R}")
    print(f"   {BOLD}{C_HEADER}Special Commands{R}")
    print(f" {C_DIM}{'─' * (WIDTH - 2)}{R}")
    for cmd, desc in cmds:
        pad = max(1, 24 - len(cmd))
        print(f"   {C_ORANGE}{cmd}{R}{' ' * pad}{C_META}{desc}{R}")
    print()
    print(f"   {C_META}Aliases สำหรับ /history:{R}")
    for alias, target in aliases:
        print(f"   {C_DIM}{alias!r:20}{R}{C_META}→ {target}{R}")
    print(f" {C_DIM}{'─' * (WIDTH - 2)}{R}")
    print()


def print_skill_help(data: dict) -> None:
    """แสดง skill guide โหลดจาก skills/skill.json"""
    usage   = data.get("usage", [])
    skills  = data.get("skills", [])
    print()
    print(f" {C_DIM}{'─' * (WIDTH - 2)}{R}")
    print(f"   {BOLD}{C_HEADER}Skills{R}  {C_META}— extend agent with /commands{R}")
    print(f" {C_DIM}{'─' * (WIDTH - 2)}{R}")
    if usage:
        print(f"   {C_META}Usage:{R}")
        for key, desc in usage:
            pad = max(1, 10 - len(key))
            print(f"   {C_ORANGE}{key}{R}{' ' * pad}{C_META}{desc}{R}")
    print()
    if skills:
        print(f"   {C_META}Available:{R}")
        for s in skills:
            name = s.get("name", "")
            desc = s.get("description", "")[:55]
            pad  = max(1, 10 - len(name))
            print(f"   {C_ORANGE}/{name}{R}{' ' * pad}{C_HEADER}{desc}{R}")
    else:
        print(f"   {C_META}(ยังไม่มี skill){R}")
    print(f" {C_DIM}{'─' * (WIDTH - 2)}{R}")
    print()




# ── Input prompt ────────────────────────────────────────────────────────────────
_BORDER = f" {C_DIM}{'─' * (WIDTH - 2)}{R}"

def prompt_user(mode: str = "") -> str:
    """Draw the full input box, with the ctx bar pinned directly below it and
    real-time ghost-text skill completion — all visible while typing.

    Layout (a single non-fullscreen prompt_toolkit Application, 4 windows):
        ─────────────────────  ← top border
         > /bu[ild]            ← input + dim ghost text (AppendAutoSuggestion)
        ─────────────────────  ← bottom border
         ⎿  0%  ░░  0/200k     ← ctx bar (FormattedTextControl, redraws live)

    This replaces PromptSession.prompt(bottom_toolbar=...) — bottom_toolbar
    anchors to the terminal's physical bottom (black reverse-video bar + a big
    blank gap when the screen isn't full). A custom HSplit renders the 4 lines
    adjacent at the cursor instead, so the ctx bar sits right under the box and
    stays visible the whole time the user is composing.
    """
    mode_tag = f"{C_DIM}[{mode}]{R} " if mode else ""
    prefix = _ANSI(f" {mode_tag}{C_ORANGE}{BOLD}>{R}  ")

    buf = _Buffer(auto_suggest=_SkillAutoSuggest(), multiline=False)

    kb = _KeyBindings()

    @kb.add("enter")
    def _accept(event):
        event.app.exit(result=buf.text)

    @kb.add("tab")
    def _complete(event):
        if buf.suggestion:
            buf.insert_text(buf.suggestion.text)

    @kb.add("c-c")
    @kb.add("c-d")
    def _abort(event):
        event.app.exit(result=None)

    input_win = _Window(
        _BufferControl(buffer=buf, input_processors=[_AppendAutoSuggestion()]),
        height=1,
        get_line_prefix=lambda *_: prefix,
    )
    box = _HSplit([
        _Window(_FTControl(lambda: _ANSI(_BORDER)), height=1),
        input_win,
        _Window(_FTControl(lambda: _ANSI(_BORDER)), height=1),
        _Window(_FTControl(lambda: _ANSI(_render_ctx_bar())), height=1),
    ])
    app = _Application(
        layout=_Layout(box, focused_element=input_win),
        key_bindings=_merge_kb([_load_kb(), kb]),
        full_screen=False,
        # Erase the whole box on submit so only the agent's answer is left in the
        # scrollback — the next turn draws a fresh box with updated ctx. (No stale
        # question/border/ctx accumulating above each answer.)
        erase_when_done=True,
    )

    result = app.run()
    if result is None:
        return "exit"
    return result.strip()

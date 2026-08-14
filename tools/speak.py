# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""speak.py — read text aloud through this Mac's speaker via macOS `say`.

Standalone tool: audio output doesn't share any of `bash`'s sandbox/timeout
concerns, so it gets its own small, single-purpose docstring instead of being
folded into another tool.

Cleans markdown + caps length at a natural pause boundary (sentence > clause
> word) before speaking — a much lighter cousin of ENDEAVOR_VOX's chunker.py
(that pipeline chunks whole books for a neural TTS engine with audio
splicing; this only ever needs ONE cut point to keep a spoken answer short,
so no pythainlp dependency — agent answers already carry normal sentence
punctuation, unlike VOX's raw book text).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading

try:
    from langchain_core.tools import tool
except ImportError:  # lets deterministic tests run without the agent environment
    def tool(fn):
        fn.func = fn
        return fn

SPEAK_MAX_CHARS = int(os.getenv("V2_SPEAK_MAX_CHARS", "800"))
SPEAK_VOICE_TH = os.getenv("V2_SPEAK_VOICE_TH", "Kanya")
_THAI_RANGE = range(0x0E00, 0x0E7F + 1)

_MD_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_MD_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_MD_LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_MD_HEADER_RE = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_MD_EMPHASIS_RE = re.compile(r"(\*\*\*|\*\*|\*|__|_)")
_MD_BULLET_RE = re.compile(r"^[ \t]*[-*+]\s+", re.MULTILINE)
_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")

# Thai "ฯ" (abbreviation/etc. mark) included alongside the usual terminators —
# common enough in Thai prose to be worth a pause the same as a period.
_SENTENCE_END_RE = re.compile(r"[.!?…ฯ][\"')\]]?(?=\s|$)")
_CLAUSE_PAUSE_RE = re.compile(r"[,;:—](?=\s|$)")

# Serializes actual playback across overlapping speak() calls (e.g. the model
# calling it twice in one turn) so two `say` processes never talk over each
# other — held only inside the background thread below, never by speak()
# itself, so the tool call always returns immediately regardless of whether
# an earlier utterance is still playing.
_PLAYBACK_LOCK = threading.Lock()


def _clean_for_speech(text: str) -> str:
    t = _MD_CODE_BLOCK_RE.sub(" (code) ", text)
    t = _MD_INLINE_CODE_RE.sub(r"\1", t)
    t = _MD_LINK_RE.sub(r"\1", t)
    t = _MD_HEADER_RE.sub("", t)
    t = _MD_BULLET_RE.sub("", t)
    t = _MD_EMPHASIS_RE.sub("", t)
    t = _WHITESPACE_RE.sub(" ", t)
    t = _BLANKLINES_RE.sub("\n\n", t)
    return t.strip()


def _detect_thai(text: str) -> bool:
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return False
    thai = sum(1 for c in alpha if ord(c) in _THAI_RANGE)
    return thai / len(alpha) > 0.25


def _find_cut(text: str, target: int) -> int:
    """Best cut offset <= len(text) at or before `target`: prefer the LAST
    sentence-end, else the last clause-pause, else the last whitespace, else
    `target` itself (never past it, never mid-word when any boundary exists
    in-window)."""
    if len(text) <= target:
        return len(text)
    window = text[: target + 1]
    for pattern in (_SENTENCE_END_RE, _CLAUSE_PAUSE_RE):
        ends = [m.end() for m in pattern.finditer(window)]
        if ends:
            return ends[-1]
    space = window.rfind(" ")
    return space if space > 0 else target


def _prepare(text: str) -> tuple[str, bool]:
    """Returns (speakable_text, was_truncated)."""
    cleaned = _clean_for_speech(text)
    if len(cleaned) <= SPEAK_MAX_CHARS:
        return cleaned, False
    cut = _find_cut(cleaned, SPEAK_MAX_CHARS)
    return cleaned[:cut].rstrip(), True


def _run_say(cmd: list[str]) -> None:
    """Runs `say` to completion on a background thread — reaps the process
    (avoids zombie accumulation in a long-running host process) and holds
    _PLAYBACK_LOCK for the duration so a second speak() call queues behind
    this one instead of overlapping it. Never raises; speak() has already
    returned by the time this runs."""
    with _PLAYBACK_LOCK:
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            proc.wait()
        except Exception:
            pass


@tool
def speak(text: str) -> str:
    """Read text aloud through THIS MAC'S OWN SPEAKER using its built-in
    text-to-speech (macOS `say`) — even when you were asked via Telegram or
    another remote host, the audio plays locally on the machine running this
    agent process, not back into the chat.

    ONLY call this when the user explicitly asks to hear something spoken —
    "อ่านให้ฟังด้วย", "พูดให้ฟัง", "read it out loud", "read this to me" —
    never on your own initiative. Compose your final answer FIRST, then call
    speak with that SAME text, then still give the normal text answer too —
    speaking is an ADDITIONAL delivery channel, not a replacement for the
    text answer.

      ❌ user asks "อ่านให้ฟังด้วย" → you only write the text answer (no sound plays)
      ✅ compose the answer → speak(text="<the exact same answer>") → give that answer as your normal reply too

    Markdown (bold/headers/bullets/links/code) is stripped automatically
    since raw markdown reads badly aloud — pass your answer as-is, no need
    to pre-clean it yourself. Capped at ~800 characters (V2_SPEAK_MAX_CHARS),
    cut at the nearest sentence/clause boundary, never mid-word — mention in
    your text answer if you expect a long answer to be read only in part.
    Thai text automatically uses a Thai voice; runs in the background, so
    this call returns immediately and does not wait for playback to finish.

    text : what to speak — plain text or markdown, cleaned automatically
    Returns "[ok] speak: <n> chars queued" (add "(truncated)" if capped), or
    "[error] ..." if text is missing or `say` is unavailable (non-macOS).
    """
    if not text or not text.strip():
        return "[error] text is required"
    if shutil.which("say") is None:
        return "[error] speak unavailable — 'say' not found (macOS only)"
    try:
        prepared, truncated = _prepare(text)
        if not prepared:
            return "[error] nothing left to speak after removing markdown/whitespace"
        cmd = ["say"]
        if _detect_thai(prepared):
            cmd += ["-v", SPEAK_VOICE_TH]
        cmd.append(prepared)
        threading.Thread(target=_run_say, args=(cmd,), daemon=True, name="speak-say").start()
        note = " (truncated)" if truncated else ""
        return f"[ok] speak: {len(prepared)} chars queued{note}"
    except Exception as e:
        return f"[error] speak failed: {e}"

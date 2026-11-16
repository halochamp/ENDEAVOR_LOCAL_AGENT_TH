# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""Sentence/clause splitting and WAV splicing for the Text to Speech job (pure Python + pythainlp + numpy, no model).

``_split_for_tts`` cuts a paragraph into pieces a voice model can read (Thai sentence ends, clause breaks, isolated Latin words)
and says what kind of break follows each piece; ``_concat_wavs`` joins the rendered pieces with the right silence and a short
fade so the splices do not click.  This is this project's own copy: it imports nothing from any other project, so a fix made
in another copy of it does not flow here.
"""
from __future__ import annotations

import io
import re
import wave
from pathlib import Path

import numpy as np

_THAI_RE = re.compile(r"[฀-๿]")

_LATIN_RE = re.compile(r"[A-Za-z]")

# ═══ break finder: where a long text may be cut ═══════════════════════════════
_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n")

_SENTENCE_END_RE = re.compile(r"[.!?…][\"')\]]?(?=\s|$)")

_CLAUSE_PAUSE_RE = re.compile(r"[,;:—](?=\s|$|[\u0E00-\u0E7F])")

_THAI_RANGE = range(0x0E00, 0x0E7F + 1)

_HARD_CAP = 1000

CHUNK_WINDOW_MIN = 0  # default window; _split_for_tts always passes window_min, so it is never consulted

def detect_lang(text: str) -> str:
    """Thai-codepoint ratio of alphabetic chars > 0.25 -> "th", else "en"."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return "en"
    thai = sum(1 for c in alpha if ord(c) in _THAI_RANGE)
    return "th" if thai / len(alpha) > 0.25 else "en"

def _find_break(
    text: str, target: int, window_min: int | None = None, with_tier: bool = False,
) -> int | tuple[int, str]:
    """2026-07-12: prefer a human-like pause over a raw char-count cut.
    Builds ranked tiers of candidate cut offsets (strongest pause signal
    first) over text[:limit], then hands off to _pick_break() to choose.

    `window_min` overrides the default [min(CHUNK_WINDOW_MIN, target), target]
    search window — needed by callers whose target is far below
    CHUNK_WINDOW_MIN (e.g. tts._split_for_tts's sub-chunking), where the
    default would otherwise collapse to a single point instead of a real
    range. A caller that wants the older behaviour simply does not pass it, so its
    behavior is unchanged.

    `with_tier=True` returns (cut, boundary_kind): paragraph, sentence,
    clause, word, or forced. The cut offset itself is identical either way
    (same _pick_break decision); naming the kind lets TTS give F5 a matching
    breath pause without turning every clause into a separate short request."""
    limit = min(len(text), _HARD_CAP)
    window = text[:limit]
    lang = detect_lang(window)
    tiers = _en_pause_tiers(window) if lang == "en" else _thai_pause_tiers(window)
    tier_names = ("paragraph", "sentence", "clause", "word")
    effective_window_min = window_min if window_min is not None else min(CHUNK_WINDOW_MIN, target)
    cut, tier_idx = _pick_break(tiers, target, effective_window_min, limit)
    boundary_kind = tier_names[tier_idx] if tier_idx < len(tier_names) else "forced"
    return (cut, boundary_kind) if with_tier else cut

def _pick_break(tiers: list[list[int]], target: int, window_min: int, limit: int) -> tuple[int, int]:
    """tiers = ranked candidate-offset lists, strongest signal first. Prefer
    the strongest tier with a candidate inside [window_min, target]; if none
    of the tiers has one, fall back tier-by-tier to the best candidate at or
    before target (still capped at target); only if literally no tier has
    any candidate at or before target (degenerate: e.g. one giant token with
    no space/punctuation anywhere) does the cut move past target, and even
    then only as far as the earliest candidate beyond it, else `limit`.
    Returns (cut, index of the tier that supplied it; len(tiers) if none)."""
    for tier_idx, ends in enumerate(tiers):
        in_window = [e for e in ends if window_min <= e <= target]
        if in_window:
            return in_window[-1], tier_idx
    for tier_idx, ends in enumerate(tiers):
        at_or_before = [e for e in ends if e <= target]
        if at_or_before:
            return at_or_before[-1], tier_idx
    for tier_idx, ends in enumerate(tiers):
        if ends:
            return ends[0], tier_idx
    return limit, len(tiers)

def _en_pause_tiers(window: str) -> list[list[int]]:
    paragraph_ends = _paragraph_ends(window)
    sentence_ends = [m.end() for m in _SENTENCE_END_RE.finditer(window)]
    clause_ends = [m.end() for m in _CLAUSE_PAUSE_RE.finditer(window)]
    space_ends = [m.end() for m in re.finditer(r" ", window)]
    return [paragraph_ends, sentence_ends, clause_ends, space_ends]

def _thai_pause_tiers(window: str) -> list[list[int]]:
    # Thai has no mandatory whitespace after punctuation, but translated prose
    # often carries explicit commas/colons. Rank paragraph, terminal/CRF
    # sentence, and clause marks before newmm word boundaries so the model gets
    # the original punctuation plus a matching splice pause.
    sentence_ends = sorted(set(
        [m.end() for m in _SENTENCE_END_RE.finditer(window)] + _thai_sentence_ends(window)
    ))
    clause_ends = [m.end() for m in _CLAUSE_PAUSE_RE.finditer(window)]
    return [_paragraph_ends(window), sentence_ends, clause_ends, _thai_word_boundaries(window)]

def _paragraph_ends(window: str) -> list[int]:
    return [m.end() for m in _PARAGRAPH_BREAK_RE.finditer(window)]

def _thai_sentence_ends(window: str) -> list[int]:
    try:
        from pythainlp.tokenize import sent_tokenize
        sents = sent_tokenize(window, engine="crfcut")
    except Exception:
        return []
    ends, pos = [], 0
    for s in sents[:-1]:  # last piece's end == len(window), not a real internal boundary
        pos += len(s)
        ends.append(pos)
    return ends

def _thai_word_boundaries(window: str) -> list[int]:
    from pythainlp.tokenize import word_tokenize
    tokens = word_tokenize(window, engine="newmm", keep_whitespace=True)
    boundaries, pos = [], 0
    for tok in tokens:
        pos += len(tok)
        boundaries.append(pos)
    return boundaries

# ═══ piece splitter ═══════════════════════════════════════════════════════════
_MIN_TAIL_CHARS = 40  # a trailing remainder shorter than this gets synthesized

_LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.#'’/-]*[.,!?;:]*")

_MIN_ISOLATED_PIECE_CHARS = 5  # 2026-07-13: an isolated Latin token shorter

def _merge_short_latin_pieces(pieces: list[tuple[str, bool]]) -> list[str]:
    """Fold a too-short Latin token into an adjacent piece for context."""
    merged: list[str] = []
    for text_piece, is_latin in pieces:
        short = is_latin and len(text_piece) < _MIN_ISOLATED_PIECE_CHARS
        if short and merged:
            merged[-1] = f"{merged[-1]} {text_piece}"
        else:
            merged.append(text_piece)
    if len(merged) > 1 and pieces[0][1] and len(pieces[0][0]) < _MIN_ISOLATED_PIECE_CHARS:
        merged[0:2] = [f"{merged[0]} {merged[1]}"]
    return merged

def _split_latin_words(text: str) -> list[str]:
    """Split Latin words into isolated pieces while retaining surrounding text."""
    pieces: list[tuple[str, bool]] = []  # (text, is_latin_token)
    cursor = 0
    for match in _LATIN_WORD_RE.finditer(text):
        gap = text[cursor:match.start()].strip()
        token = match.group()
        if gap:
            if any(char.isalnum() for char in gap):
                pieces.append((gap, False))
            else:
                token = f"{gap}{token}"
        pieces.append((token, True))
        cursor = match.end()
    tail = text[cursor:].strip()
    if tail:
        if any(char.isalnum() for char in tail) or not pieces:
            pieces.append((tail, False))
        else:
            prev_text, prev_is_latin = pieces[-1]
            pieces[-1] = (f"{prev_text}{tail}", prev_is_latin)
    if not pieces:
        return [text.strip()]
    return _merge_short_latin_pieces(pieces)

def _split_for_tts(
    text: str, target: int, split_latin_words: bool = False, window_min: int | None = None,
    with_tiers: bool = False,
) -> list[str] | list[tuple[str, str]]:
    """Break text into ~target-char pieces using the same word-boundary-aware
    logic as the break finder above, then merge undersized trailing context.

    `window_min` searches for a pause point within [window_min, target]
    instead of only "the latest pause at or before target" (see
    _find_break) — passed through from the voice's
    sub_chunk_window_min so pieces land on a real sentence/clause boundary
    within the range instead of an arbitrarily early one.

    `with_tiers=True` returns (piece, boundary_after) tuples instead —
    boundary_after names the splice that FOLLOWS the piece: paragraph,
    sentence, clause, word, forced, latin, or end. Piece texts are identical
    either way; punctuation stays inside the F5 request that precedes it."""
    pieces: list[tuple[str, str]] = []
    remaining = text.strip()
    while len(remaining) > target:
        cut, boundary = _find_break(remaining, target, window_min=window_min, with_tier=True)
        pieces.append((remaining[:cut].strip(), boundary))
        remaining = remaining[cut:].strip()
    if remaining:
        merged = f"{pieces[-1][0]} {remaining}" if pieces else remaining
        if pieces and len(remaining) < _MIN_TAIL_CHARS and len(merged) <= target:
            pieces[-1] = (merged, "end")
        else:
            pieces.append((remaining, "end"))
    elif pieces:
        pieces[-1] = (pieces[-1][0], "end")
    if split_latin_words:
        expanded: list[tuple[str, str]] = []
        for piece_text, boundary in pieces:
            parts = _split_latin_words(piece_text)
            expanded.extend((part, "latin") for part in parts[:-1])
            expanded.append((parts[-1], boundary))
        pieces = expanded
    if with_tiers:
        return pieces
    return [piece_text for piece_text, _ in pieces]

# ═══ piece sizing and pauses ══════════════════════════════════════════════════
_PIECE_TARGET = 120   # piece sizing accepted for F5 by listening tests (target / window_min)

_PIECE_WINDOW_MIN = 70

_OMNI_PIECE_TARGET = 400   # OmniVoice takes whole paragraphs and puts its own pauses inside them

_OMNI_PIECE_WINDOW_MIN = 200

_SHORT_PIECE_CHARS = 25   # a piece this short has no context for F5's prosody (measured: "ได้เลย" failed 3/3 takes)

_MERGED_PIECE_MAX = 150   # merge ceiling; 240-char pieces were rejected by listening, the CFM clamp is ~450

# splice silence per boundary kind (ms): the values accepted for F5-TTS-TH-V2, sentence/paragraph raised to the 420 ms the owner chose by A/B (2026-11-07)
PAUSE_TABLE_MS = {"sentence": 420, "paragraph": 420, "clause": 180, "word": 40, "latin": 50}

def language_mix(paragraphs: list[str]) -> tuple[int, int]:
    text = "\n".join(paragraphs)
    return len(_THAI_RE.findall(text)), len(_LATIN_RE.findall(text))

def is_english(text: str) -> bool:
    """More Latin letters than Thai characters: an English paragraph/piece (digits and symbols do not count)."""
    thai, latin = language_mix([text])
    return latin > thai

def pause_for(boundary: str) -> int:
    return PAUSE_TABLE_MS.get(boundary, PAUSE_TABLE_MS["word"])

def merge_short_pieces(rows: list[list[str]]) -> list[list[str]]:
    """Fold pieces below _SHORT_PIECE_CHARS into a neighbour of the same paragraph; the merged piece keeps the boundary of its last constituent."""
    out: list[list[str]] = []
    for text, boundary in rows:
        if out and len(text.replace(" ", "")) < _SHORT_PIECE_CHARS and len(out[-1][0]) + 1 + len(text) <= _MERGED_PIECE_MAX:
            out[-1] = [f"{out[-1][0]} {text}", boundary]
        else:
            out.append([text, boundary])
    if len(out) > 1 and len(out[0][0].replace(" ", "")) < _SHORT_PIECE_CHARS and len(out[0][0]) + 1 + len(out[1][0]) <= _MERGED_PIECE_MAX:
        out[0:2] = [[f"{out[0][0]} {out[1][0]}", out[1][1]]]
    return out


def wav_params(wav_bytes: bytes) -> wave._wave_params:
    """Params of a PCM16 WAV (the only thing the workers write); raises on anything else."""
    with wave.open(io.BytesIO(wav_bytes)) as w:
        if w.getsampwidth() != 2 or w.getnframes() <= 0:
            raise RuntimeError("worker returned a WAV that is empty or not 16-bit PCM")
        return w.getparams()

def peak_amplitude(path: str | Path) -> int:
    with wave.open(str(path), "rb") as w:
        frames = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return int(np.max(np.abs(frames.astype(np.int32)))) if frames.size else 0

_SPLICE_FADE_MS = 20.0  # measured: a piece's tail commonly sits at

_EDGE_SILENCE_ABS = 300  # int16 abs amplitude regarded as silence when

# trimming piece edges (~1% FS; measured genuine speech peaks >=10000, the
# model's residual edge noise sits well below this)
_EDGE_KEEP_MS = 40.0  # natural edge retained on each side after a trim so

def _trim_edge_silence(arr: np.ndarray, framerate: int) -> np.ndarray:
    """Cut leading/trailing silence down to at most _EDGE_KEEP_MS per side."""
    loud = np.flatnonzero(np.abs(arr) > _EDGE_SILENCE_ABS)
    if loud.size == 0:
        return arr
    keep = int(framerate * _EDGE_KEEP_MS / 1000)
    return arr[max(0, loud[0] - keep):min(len(arr), loud[-1] + 1 + keep)]

def _concat_wavs(
    wav_bytes_list: list[bytes], out_path: Path, pause_ms: list[int] | None = None,
) -> None:
    """Concatenate same-format WAV byte strings (all from the same TTS model
    call, so params match) by copying raw frames — avoids adding pydub as a
    dependency of the `mlx` env just for this. Fades each internal splice
    point (see _SPLICE_FADE_MS) to hide the model's non-decayed piece tails.

    `pause_ms` (len == len(wav_bytes_list)-1) switches on designed pausing:
    each piece's edge silence is first trimmed to _EDGE_KEEP_MS, then
    pause_ms[k] of true silence is inserted at splice k — so the audible gap
    at every join is a chosen, boundary-type-dependent constant instead of
    the model's random per-call lead-in. None (default) preserves the
    historical trim-free butt-join exactly."""
    params = wav_params(wav_bytes_list[0])
    n_fade = int(params.framerate * _SPLICE_FADE_MS / 1000)
    arrays = []
    for wb in wav_bytes_list:
        with wave.open(io.BytesIO(wb)) as w:
            candidate = w.getparams()
            if (
                candidate.nchannels, candidate.sampwidth, candidate.framerate,
                candidate.comptype, candidate.compname,
            ) != (
                params.nchannels, params.sampwidth, params.framerate,
                params.comptype, params.compname,
            ):
                raise RuntimeError("companion returned WAV pieces with inconsistent formats")
            arrays.append(np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).copy())
    if pause_ms is not None:
        if len(pause_ms) != len(arrays) - 1:
            raise ValueError("pause_ms must have exactly one entry per splice")
        arrays = [_trim_edge_silence(arr, params.framerate) for arr in arrays]
    for i, arr in enumerate(arrays):
        if len(arr) < n_fade * 2:
            continue
        if i < len(arrays) - 1:
            arr[-n_fade:] = (arr[-n_fade:] * np.linspace(1.0, 0.0, n_fade)).astype(np.int16)
        if i > 0:
            arr[:n_fade] = (arr[:n_fade] * np.linspace(0.0, 1.0, n_fade)).astype(np.int16)
    if pause_ms is not None:
        spliced: list[np.ndarray] = []
        for i, arr in enumerate(arrays):
            spliced.append(arr)
            if i < len(arrays) - 1 and pause_ms[i] > 0:
                spliced.append(np.zeros(int(params.framerate * pause_ms[i] / 1000), dtype=np.int16))
        arrays = spliced
    with wave.open(str(out_path), "wb") as out:
        out.setparams(params)
        out.writeframes(np.concatenate(arrays).tobytes())

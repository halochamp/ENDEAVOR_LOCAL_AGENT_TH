# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""read_image.py — OCR assist + direct vision for ENDEAVOR_LOCAL_AGENT_TH

The image itself is the primary evidence and is queued for the main mlx-vlm model.
The first source-only call queues the whole original image and performs no OCR,
classification, table, or QR sensor work. Explicit follow-up detail modes add
OCR/table/QR assistance, structural crops, or native-resolution find/zoom crops
only after the original pixels were exposed in this outer turn.

The tool is a sensor: it never accepts or rewrites a semantic user question.
The main conversation owns meaning and decides whether a follow-up sensor is needed.
"""
from __future__ import annotations
import base64
import functools
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import urllib.request
import uuid
from pathlib import Path

from langchain_core.tools import tool

from tools._ocr import read_layout as _ocr_layout
from tools._vision_capability import UNKNOWN, TEXT_ONLY, get_capability
from tools._progress import progress, phase
from tools._safety import resolve_read_path

log = logging.getLogger(__name__)

# VLM input is aspect-preserved and downscaled so its longest side ≤ this. The old
# webcam path used a fixed 200×150 squash (the old fixed-size webcam path) — fine for "what's
# in the scene", far too small to read document text, table cells, or chart axes.
# Direct vision is the first-pass evidence; explicit OCR/detail sensors run only
# when the agent asks for them after the original pixels are exposed. Feed the
# production VLM a much larger, undistorted image. Downscale-only: never upscale a
# small image (no detail to invent). Size settled empirically on the production VLM.
VLM_MAX_SIDE = 1024
# Kept as the public upload/UI compatibility surface: agent_server uses this
# set to route image attachments to read_image before the tool runs.
_KNOWN_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic", ".heif", ".bmp", ".tif", ".tiff")
_DETAIL_MODES = {"overview", "text", "chart", "slide"}
# Above this size the FULL combined result is always written to a workspace .md, so
# the detail is recoverable beyond this turn. Rationale: at context compaction
# (graph.py _strip_tool_messages) ToolMessages are DROPPED wholesale — the inline
# read is gone regardless of size — so an on-disk artifact is the only durable copy.
# Tiny reads (a few OCR words) stay inline-only: nothing worth persisting, re-readable.
_SAVE_FLOOR = 800
# Inline budget returned to state on the read turn. Full result lands inline (and is
# also saved) when within this; over it, a sampled view is returned inline with the
# .md path so the agent can read_file the rest. Sized so a single-screen dashboard
# (~3.6k chars of OCR+table+VLM) arrives whole on the first turn. Not matched to
# read_file's 9.5k: image OCR is chart-axis scatter (low signal-per-char), not prose.
_COMBINED_INLINE_MAX = 4500
# Prose-in-columns guard: a column whose shortest-column median cell length
# exceeds this is verbose enough to be prose, not table data. Only consulted for
# grids with no numeric content (numeric content alone proves a real data table).
_PROSE_COL_CHARS = 14

# Module-level image state is deliberately transient: neither queue is checkpointed nor
# persisted in LangGraph history.  New images enter _PENDING_IMAGES; dynamic_prompt()
# promotes them into _ACTIVE_TURN_IMAGES, which is injected on every remaining ReAct
# invocation of this turn.  This keeps pixels available for final synthesis after an
# intermediate tool call without leaking an image into a later user turn.
_PENDING_IMAGES: list[str] = []
_ACTIVE_TURN_IMAGES: list[str] = []
# Retained normalized originals let the model-call fallback OCR the exact pixels
# that were published, including URL/screen sources whose ordinary temp files are
# cleaned when the tool returns.  The copied files are process-local and deleted
# at the outer-turn boundary; user files are never added to these lists.
_PENDING_IMAGE_SOURCES: list[tuple[str, str]] = []
_ACTIVE_IMAGE_SOURCES: list[tuple[str, str]] = []
# source="screen" is a moving target, but every progressive follow-up must inspect
# the SAME pixels as the original overview the agent actually saw. Keep one native
# screenshot file pinned for this outer turn; begin/end clean it deterministically.
_SCREEN_TURN_SNAPSHOT: list[str | None] = [None]
# mlx-vlm's OpenAI surface accepts arbitrarily many image blocks, but the model
# server's documented practical ceiling is ten. Keep every direct-vision source
# atomic: a five-view market-slide pack is either wholly available to the model or
# deferred to a later outer turn -- never silently cut halfway through its evidence.
_IMAGE_TURN_MAX = 10

# ToolNode may execute synchronous calls from one AIMessage in parallel, while
# every read_image call shares the transient queues, counters, and progressive
# guards above. Serialize the complete transaction so check/encode/queue/seen
# state cannot interleave; distinct sources remain valid, only their sensor work
# is ordered. This is intentionally project-local and does not modify LangGraph.
_READ_IMAGE_CALL_LOCK = threading.RLock()


def _serialize_read_image(fn):
    @functools.wraps(fn)
    def _locked_read_image(*args, **kwargs):
        with _READ_IMAGE_CALL_LOCK:
            return fn(*args, **kwargs)

    return _locked_read_image


def _image_turn_slots() -> int:
    """Return remaining direct-vision attachment slots in this outer turn."""
    return max(0, _IMAGE_TURN_MAX - len(_PENDING_IMAGES) - len(_ACTIVE_TURN_IMAGES))


def _queue_image_bundle(images: list[str]) -> bool:
    """Atomically queue one source's direct-vision views within the turn budget."""
    if len(images) > _image_turn_slots():
        return False
    _PENDING_IMAGES.extend(images)
    return True


def _retain_original_source(source: str, local_path: str) -> bool:
    """Keep a private temp copy of a queued original for a possible OCR retry."""
    retained_path = ""
    try:
        suffix = Path(local_path).suffix or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            retained_path = handle.name
        shutil.copyfile(local_path, retained_path)
        _PENDING_IMAGE_SOURCES.append((source, retained_path))
        return True
    except Exception as exc:
        if retained_path:
            try:
                Path(retained_path).unlink(missing_ok=True)
            except Exception:
                pass
        log.warning("read_image source retention failed: %s", exc)
        return False


def _clear_retained_sources() -> None:
    paths = {path for _, path in _PENDING_IMAGE_SOURCES + _ACTIVE_IMAGE_SOURCES}
    _PENDING_IMAGE_SOURCES.clear()
    _ACTIVE_IMAGE_SOURCES.clear()
    for path in paths:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass


def _vision_budget_notice(needed: int) -> str:
    """Explain a deferred view without inviting an unsafe same-turn retry."""
    used = _IMAGE_TURN_MAX - _image_turn_slots()
    return (
        "[DIRECT VISION DEFERRED — this turn already carries "
        f"{used}/{_IMAGE_TURN_MAX} image views; this source needs {needed} intact view(s). "
        "Do not infer unseen pixels from OCR. Answer only from already visible evidence, "
        "or ask the user to send this image in a new turn.]"
    )

# Per-read-cycle zoom dedup (V2-VLM-LOOP): every (source, region, zoom) served in
# the current read cycle. A full (non-zoom) read clears it — see read_image(). Guards
# against the degenerate ReAct loop where the model re-issued a byte-identical zoom
# call forever because the crop never showed the detail it was after.
_ZOOM_SEEN: set[tuple[str, str, float]] = set()
# Hard per-read-cycle zoom-attempt cap (V2-VLM-LOOP). Sits BEFORE the region/dedup
# checks so it bounds EVERY zoom path — including a repeated `region="full"`+zoom,
# which returns its [error] before _ZOOM_SEEN is ever consulted (the exact call that
# looped 25×). 1-element list so read_image() can mutate it without a `global`.
_ZOOM_ATTEMPTS: list[int] = [0]
_ZOOM_ATTEMPT_MAX = 6

# Per-turn image-labeling counter (multi-image turns): increments once per queued
# full-read image set (tiles from one source count as ONE image — see _tile_frame).
# Reset only at a true outer-turn boundary.  dynamic_prompt runs once per ReAct
# step, so resetting there made every later image appear to be image #1.
# The 1st image
# of a turn gets no label (keeps single-image turns byte-stable for prompt-cache
# reuse — S117); only the 2nd+ image is prefixed, so the model can tell which OCR
# text belongs to which image.
_IMAGE_TURN_COUNT: list[int] = [0]

# Plain full-read loop guard (V2-VLM04): the zoom/find guard block below is
# structurally unreachable for a plain full-read (region="full", zoom=1.0 defaults →
# do_zoom=do_find=False), so a full-read had ZERO loop protection — a model doubting
# a correct-but-surprising deterministic OCR result could re-issue byte-identical
# full reads forever (observed live: 10 read_image calls / ~5min real Telegram turn,
# zero convergence, no guard ever triggered — confirmed by a direct LLM-free call:
# 10x identical full-read never once returned a dedup/cap marker). Keyed by
# (src, detail) — src not local_path, mirrors the (A)/find sig note below, since
# EXIF/HEIC normalization and URL downloads repoint local_path at a fresh temp file
# every call. Cleared ONLY by reset_read_guards() from graph.py's react_node at turn
# entry — NEVER from pop_pending_images(): dynamic_prompt drains that before EVERY
# ReAct step, so clearing there wiped the guard between two consecutive read_image
# calls and the dedup/cap could never survive to catch the very loop it was built
# for. The detail key intentionally permits progressive overview→text/chart/slide
# escalation while rejecting an identical repeat.
_READ_SEEN: dict[tuple[str, str], str] = {}
_READ_ATTEMPTS: dict[str, int] = {}
_READ_ATTEMPT_MAX = 6
# A detail pack is legal only after the original whole-image overview was actually
# exposed to the main model in this outer turn. Keep queued sources separate: a
# ToolNode can run another read_image call before react.py gets a chance to promote
# the pending pixels at the model boundary.
_OVERVIEW_PENDING: set[str] = set()
_OVERVIEW_SEEN: set[str] = set()


def invalidate_published_images() -> None:
    """Drop rejected image publication and reset affected read state.

    The model-call fallback invokes this only after copying the retained source
    list for OCR.  Since a rejected overview was never actually seen by the
    model, it must not leave ``_OVERVIEW_SEEN`` or an overview dedup result that
    could authorize/return stale pixels later in the same turn.
    """
    with _READ_IMAGE_CALL_LOCK:
        _PENDING_IMAGES.clear()
        _ACTIVE_TURN_IMAGES.clear()
        _clear_retained_sources()
        _OVERVIEW_PENDING.clear()
        _OVERVIEW_SEEN.clear()
        _READ_SEEN.clear()
        _READ_ATTEMPTS.clear()


def reset_read_guards() -> None:
    """Clear the per-turn full-read loop guards (V2-VLM04).

    Call ONLY from a true once-per-turn site (graph.py react_node entry, alongside
    the stale-image drain). Do NOT call from pop_pending_images() — that runs via
    dynamic_prompt before every ReAct step, mid-turn, which is exactly when the
    guard must stay alive.
    """
    with _READ_IMAGE_CALL_LOCK:
        _READ_SEEN.clear()
        _READ_ATTEMPTS.clear()
        # _OVERVIEW_PENDING is publication state, not a guard. begin_image_turn()
        # owns its boundary cleanup so a mid-turn guard reset cannot make a
        # queued-but-unpublished source appear eligible for sensors.
        _OVERVIEW_SEEN.clear()


def _clear_screen_turn_snapshot() -> None:
    """Delete the pinned native screenshot, if any, and forget screen read guards."""
    path = _SCREEN_TURN_SNAPSHOT[0]
    _SCREEN_TURN_SNAPSHOT[0] = None
    if path:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass
    # A later outer turn must establish a new screen original before any sensors.
    _OVERVIEW_SEEN.discard("screen")
    _READ_ATTEMPTS.pop("screen", None)
    for key in [k for k in _READ_SEEN if k[0].lower() == "screen"]:
        _READ_SEEN.pop(key, None)
    for sig in [s for s in _ZOOM_SEEN if s[0].lower() == "screen"]:
        _ZOOM_SEEN.discard(sig)


def _pin_screen_turn_snapshot(path: str) -> None:
    """Own `path` as this outer turn's immutable source="screen" snapshot."""
    old = _SCREEN_TURN_SNAPSHOT[0]
    if old and old != path:
        try:
            Path(old).unlink(missing_ok=True)
        except Exception:
            pass
    _SCREEN_TURN_SNAPSHOT[0] = path


def begin_image_turn() -> int:
    """Start a fresh outer turn and discard transient pixels/native screen snapshot.

    Returns the number discarded for boundary logging.  Call only from
    graph.react_node after warm-turn exclusion; calling this per ReAct step would
    reintroduce both the one-shot-image and multi-image-labeling bugs.
    """
    with _READ_IMAGE_CALL_LOCK:
        stale = len(_PENDING_IMAGES) + len(_ACTIVE_TURN_IMAGES)
        _PENDING_IMAGES.clear()
        _ACTIVE_TURN_IMAGES.clear()
        _clear_retained_sources()
        _OVERVIEW_PENDING.clear()
        _clear_screen_turn_snapshot()
        _IMAGE_TURN_COUNT[0] = 0
        return stale


def active_turn_images() -> list[str]:
    """Publish newly queued images and return this turn's complete pixel context.

    This is the publication point used by react.py immediately before a model
    invocation. Only here may an original source become eligible for a
    progressive sensor follow-up.
    """
    with _READ_IMAGE_CALL_LOCK:
        if _PENDING_IMAGES:
            _ACTIVE_TURN_IMAGES.extend(_PENDING_IMAGES)
            _PENDING_IMAGES.clear()
            _ACTIVE_IMAGE_SOURCES.extend(_PENDING_IMAGE_SOURCES)
            _PENDING_IMAGE_SOURCES.clear()
            _OVERVIEW_SEEN.update(_OVERVIEW_PENDING)
            _OVERVIEW_PENDING.clear()
        return list(_ACTIVE_TURN_IMAGES)


def active_turn_image_sources() -> list[tuple[str, str]]:
    """Return ordered original sources retained for a same-call OCR recovery."""
    with _READ_IMAGE_CALL_LOCK:
        return list(_ACTIVE_IMAGE_SOURCES)


def end_image_turn() -> None:
    """Release transient pixels and any pinned native screen snapshot."""
    with _READ_IMAGE_CALL_LOCK:
        _PENDING_IMAGES.clear()
        _ACTIVE_TURN_IMAGES.clear()
        _clear_retained_sources()
        _OVERVIEW_PENDING.clear()
        _clear_screen_turn_snapshot()


def pop_pending_images() -> list[str]:
    """Return and clear all pending image data-URLs.

    Compatibility drain for tests and exceptional cleanup. Normal dynamic_prompt
    uses active_turn_images() so images survive all ReAct steps. This must not
    touch per-turn guards or the image counter.
    """
    with _READ_IMAGE_CALL_LOCK:
        imgs = list(_PENDING_IMAGES)
        _PENDING_IMAGES.clear()
        _clear_retained_sources()
        _OVERVIEW_PENDING.clear()
        return imgs


def _hard_truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n - 3].rstrip() + "..."


def _deloop(text: str) -> str:
    """Drop repeated lines/sentences (VLM loop guard) without losing unique content.

    Splits on newlines first — Thai output has no sentence-final periods, so a
    period-only split made the whole answer one unit and the guard a no-op on the
    user's primary language. Within each line we also split on period-delimited
    sentences for English. A unit is skipped only when it exactly repeats one already
    emitted; we never `break`, so legitimate tail content that happens to follow a
    repeated sentence (e.g. a phrase recurring across two chart panels) is preserved.
    """
    import re
    seen: set[str] = set()
    kept_lines: list[str] = []
    for line in text.split("\n"):
        sentences = re.split(r'(?<=\.)\s+', line)
        if len(sentences) > 1:
            kept = []
            for s in sentences:
                key = s.strip().lower()
                if key and key in seen:
                    continue
                if key:
                    seen.add(key)
                kept.append(s)
            kept_lines.append(" ".join(kept).strip())
        else:
            key = line.strip().lower()
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            kept_lines.append(line)
    return "\n".join(kept_lines).strip()


def _column_intervals(boxes: list[dict], min_gap: float = 0.025) -> list[tuple[float, float]]:
    """Find column x-intervals by projecting cell spans [x, x+w] onto the x-axis
    and splitting where vertical whitespace exceeds `min_gap` (normalized).

    Uses the full span, not just the left edge, so it is robust to left-, right-,
    and centre-aligned columns alike — right-aligned numbers of differing width
    (12.34 vs 1,000,000 vs 5) share a right edge and overlap, so they cluster into
    one column instead of scattering by their left edges. Very wide cells (w>0.4,
    a title/sentence spanning the table) are excluded from interval construction so
    they cannot bridge two real columns; they are still assigned afterwards."""
    spans = sorted(
        ((b["x"], b["x"] + b["w"]) for b in boxes if b["w"] <= 0.4),
        key=lambda s: s[0],
    )
    if not spans:
        return []
    cols: list[list[float]] = [list(spans[0])]
    for lo, hi in spans[1:]:
        if lo - cols[-1][1] <= min_gap:        # overlaps / small gap → same column
            cols[-1][1] = max(cols[-1][1], hi)
        else:
            cols.append([lo, hi])
    return [(lo, hi) for lo, hi in cols]


def _assign_column(b: dict, cols: list[tuple[float, float]]) -> int:
    """Index of the column whose x-interval the cell overlaps most; if it overlaps
    none (e.g. a wide cell excluded from interval construction), the nearest column
    by centre distance."""
    bx0, bx1 = b["x"], b["x"] + b["w"]
    best_i, best_ov = 0, -1.0
    for i, (lo, hi) in enumerate(cols):
        ov = min(bx1, hi) - max(bx0, lo)
        if ov > best_ov:
            best_ov, best_i = ov, i
    if best_ov > 0:
        return best_i
    bc = bx0 + b["w"] / 2
    return min(range(len(cols)), key=lambda i: abs((cols[i][0] + cols[i][1]) / 2 - bc))


def _reconstruct_table(boxes: list[dict]) -> str | None:
    """Reconstruct a markdown table from OCR cell boxes, or None if the layout
    is not a confident grid (≥2 rows × ≥2 columns, with most rows multi-cell).
    Vision boundingBox origin is BOTTOM-LEFT, so rows sort by y-center DESCENDING
    (top of the image first)."""
    if len(boxes) < 4:
        return None

    items = sorted(boxes, key=lambda b: -(b["y"] + b["h"] / 2))
    heights = sorted(b["h"] for b in boxes)
    med_h = heights[len(heights) // 2]
    row_thresh = max(med_h * 0.6, 0.005)

    # group into rows: a cell joins the current row if its y-center is within
    # row_thresh of the row's mean y-center
    rows: list[list[dict]] = [[items[0]]]
    for b in items[1:]:
        yc = b["y"] + b["h"] / 2
        row_yc = sum(c["y"] + c["h"] / 2 for c in rows[-1]) / len(rows[-1])
        if row_yc - yc <= row_thresh:
            rows[-1].append(b)
        else:
            rows.append([b])

    cols = _column_intervals(boxes)
    if len(cols) < 2 or len(rows) < 2:
        return None

    grid: list[list[str]] = []
    for row in rows:
        cells = [""] * len(cols)
        for b in sorted(row, key=lambda b: b["x"]):
            ci = _assign_column(b, cols)
            cells[ci] = (cells[ci] + " " + b["text"]).strip() if cells[ci] else b["text"]
        grid.append(cells)

    # confidence: most rows must actually span ≥2 columns, else it's prose that
    # happens to have some horizontal alignment — fall back to flat OCR.
    multi = sum(1 for r in grid if sum(1 for c in r if c) >= 2)
    if multi < max(2, len(grid) * 0.5):
        return None

    # Density gate: a real table is a dense grid — most cells filled. A chart or
    # multi-panel figure scatters axis labels / annotations that accidentally
    # align into ≥2 columns but leave most cells empty (a 4-panel bar chart OCRs
    # to a 23×4 grid only ~45% filled, vs ~100% for a real table). Require ≥60%
    # filled — allows genuine blanks + a few OCR drops, rejects chart scatter.
    # Catches a different failure mode than the prose-guard below (which targets
    # DENSE verbose 2-column prose), so both are needed.
    filled = sum(1 for r in grid for c in r if c)
    if filled < len(grid) * len(cols) * 0.6:
        return None

    # Prose-in-columns guard: two side-by-side prose blocks also satisfy the grid
    # shape above (≥2 cols, most rows multi-cell) yet are not a data table —
    # reconstructing them scrambles the text. A real data table has terse cells:
    # at least one column of numbers or short labels; prose columns are uniformly
    # verbose. Reject only when there is NO numeric content AND the *shortest*
    # column is still verbose, so a real (esp. numeric) table is never rejected — a
    # false-negative here regresses to the original flat-dump bug, hence numeric
    # content overrides the length check. char-length, not word-count: word-count
    # is degenerate for Thai (the user's primary language) — written without
    # inter-word spaces, every Thai cell is one "word".
    flat = [c for r in grid for c in r if c]
    numeric = sum(1 for c in flat if any(ch.isdigit() for ch in c))
    if flat and numeric / len(flat) < 0.15:
        col_med = []
        for ci in range(len(cols)):
            lens = sorted(len(grid[r][ci]) for r in range(len(grid)) if grid[r][ci])
            if lens:
                col_med.append(lens[len(lens) // 2])
        if col_med and min(col_med) > _PROSE_COL_CHARS:
            return None

    def _row_md(cells: list[str]) -> str:
        return "| " + " | ".join(c.replace("|", r"\|") for c in cells) + " |"

    lines = [_row_md(grid[0]), "| " + " | ".join("---" for _ in cols) + " |"]
    lines += [_row_md(r) for r in grid[1:]]
    return "\n".join(lines)


def _combine(sections: list[str], source: str) -> str:
    """Join sensor output (table/OCR hints) into one bounded state string.

    Always saves the FULL combined result to a runtime-work .md once it exceeds
    _SAVE_FLOOR (workspace by default), then returns either the full text inline or uniform sampled coverage
    plus an actionable read_file hint. Sampling is intentionally query-agnostic:
    read_image is a sensor and never receives or rewrites the user's semantic prompt.
    The original conversation remains the agent's source of task intent."""
    full = "\n\n".join(s for s in sections if s and s.strip())
    if len(full) <= _SAVE_FLOOR:
        return full  # trivial read — nothing worth persisting, re-readable as-is

    import re
    from config import WORKSPACE
    from tools.read_file import _sample_coverage

    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", Path(source).stem)[:40] or "image"
    filename = f"image_{safe}_{uuid.uuid4().hex[:8]}.md"
    saved = ""
    try:
        out_path = Path(WORKSPACE) / filename
        out_path.write_text(full, encoding="utf-8")
        saved = str(out_path)
    except Exception:
        saved = ""

    # Full result fits inline: return it whole. Note the saved path for durability,
    # but do NOT tell the agent to read the file for "more" — there is no more.
    if not saved:
        if len(full) <= _COMBINED_INLINE_MAX:
            return full
        return _sample_coverage(full, source,
                                max_chars=_COMBINED_INLINE_MAX - 40)

    saved_note = f"บันทึกผลเต็มที่: {saved}\n\n"
    if len(saved_note) + len(full) <= _COMBINED_INLINE_MAX:
        return saved_note + full

    # Over budget: inline is only a sample. Tell the agent explicitly to read_file the
    # saved .md for full detail — a local model won't infer it from a bare path.
    prefix = (f"บันทึกผลเต็มที่: {saved}\n"
              f"(ส่วนด้านล่างเป็นเพียงตัวอย่าง — อ่านไฟล์นี้ด้วย read_file เพื่อรายละเอียดเต็ม)\n\n")
    sampled = _sample_coverage(full, source,
                               max_chars=_COMBINED_INLINE_MAX - len(prefix) - 40)
    return prefix + sampled




# Named zoom anchors (normalized x,y center, 0..1) the agent can request when a
# full-frame read (downscaled to VLM_MAX_SIDE) was too small to read a detail.
# "full" is intentionally absent — it means "no crop", handled separately.
_REGION_ANCHORS: dict[str, tuple[float, float]] = {
    "center": (0.5, 0.5), "top": (0.5, 0.0), "bottom": (0.5, 1.0),
    "left": (0.0, 0.5), "right": (1.0, 0.5),
    "top-left": (0.0, 0.0), "top-right": (1.0, 0.0),
    "bottom-left": (0.0, 1.0), "bottom-right": (1.0, 1.0),
}
# Cap on the crop magnification: 1/zoom of width/height is cropped, so 4.0 already
# crops to a 25%×25% window — enough for any small detail without shrinking the
# crop to near-nothing on typical source resolutions.
_ZOOM_MAX = 4.0


def _grid_cell(x: float, y: float) -> str:
    """Bucket a normalized (x,y) point — image-space, top-down (y=0 top, matching
    _REGION_ANCHORS' convention) — into one of the 9 named region-anchor cells by
    equal thirds."""
    col = "left" if x < 1 / 3 else ("right" if x > 2 / 3 else "")
    row = "top" if y < 1 / 3 else ("bottom" if y > 2 / 3 else "")
    if row and col:
        return f"{row}-{col}"
    return row or col or "center"


def _zoom_region_hint(boxes: list) -> str:
    """One-line hint pointing the agent at the region with the densest cluster of
    SMALL text boxes (height < 0.018 normalized, ≥5 boxes) — the full-frame encode
    downscales small print past readability, so this nudges toward a targeted zoom
    instead of the agent guessing a region blind. At most one hint, "" if none fires.
    Box y is Vision-style (origin BOTTOM-LEFT, y increases upward) — flipped to
    top-down image-space here before bucketing, to match _REGION_ANCHORS/zoom's
    convention (a box near Vision-y=0.9 is near the image TOP, not the bottom)."""
    from collections import Counter
    counts: Counter = Counter()
    for b in boxes:
        if b.get("h", 1.0) >= 0.018:
            continue
        cx = b["x"] + b["w"] / 2
        cy_image = 1.0 - (b["y"] + b["h"] / 2)   # Vision bottom-up → top-down
        counts[_grid_cell(cx, cy_image)] += 1
    if not counts:
        return ""
    region, n = counts.most_common(1)[0]
    if n < 5:
        return ""
    return (f'(ตัวอักษรเล็กหนาแน่นที่ {region} — อ่านละเอียดได้ด้วย '
            f'region="{region}", zoom=2)')


def _validate_region(region: str) -> str | None:
    """None if `region` is usable, else an "[error] ..." string listing valid names."""
    r = (region or "full").strip().lower()
    if r != "full" and r not in _REGION_ANCHORS:
        return (f"[error] invalid region '{region}' — must be 'full' or one of: "
                f"{', '.join(_REGION_ANCHORS)}")
    return None


def _clamp_zoom(zoom: float) -> float:
    try:
        z = float(zoom)
    except (TypeError, ValueError):
        z = 1.0
    return max(1.0, min(z, _ZOOM_MAX))


def _encode_frame(frame, max_side: int) -> str:
    """Downscale (aspect-preserved, INTER_AREA) so the longest side ≤ max_side and
    encode as base64 PNG. Never upscales — a frame smaller than max_side has no
    detail to invent and is encoded at its native size instead."""
    import cv2
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        frame = cv2.resize(frame, (round(w * scale), round(h * scale)),
                           interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", frame)
    if not ok:
        raise ValueError("PNG encode failed")
    return base64.b64encode(buf.tobytes()).decode()


def _encode_frame_data_url(frame, max_side: int, *, lossless: bool = False) -> str:
    """Encode a VLM image efficiently without degrading documents/screenshots.

    PNG remains the default because it preserves thin text and UI edges.  For a
    photographic frame only use JPEG when it is materially smaller (at least 25%)
    than PNG; this reduces base64/JSON transport and server-side decode work while
    avoiding a quality trade-off on line art. ``lossless=True`` is used for known
    chart/detail crops: white chart slides can make JPEG look deceptively cheap even
    though its ringing softens thin colored series and tiny axis glyphs.
    """
    import cv2
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        frame = cv2.resize(frame, (round(w * scale), round(h * scale)),
                           interpolation=cv2.INTER_AREA)
    png_ok, png = cv2.imencode(".png", frame)
    if not png_ok:
        raise ValueError("PNG encode failed")
    jpg_ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    chosen, mime = (
        (jpg, "image/jpeg")
        if not lossless and jpg_ok and len(jpg) * 4 <= len(png) * 3
        else (png, "image/png")
    )
    return f"data:{mime};base64,{base64.b64encode(chosen.tobytes()).decode()}"


def _encode_vlm(image_path: str, max_side: int = VLM_MAX_SIDE) -> str:
    """Read image_path at native resolution and encode via `_encode_frame`. Aspect
    ratio is preserved (unlike the old 200×150 squash) so table cells and chart
    axes stay readable to the VLM."""
    import cv2
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"cannot read image: {image_path}")
    return _encode_frame(frame, max_side)


def _encode_vlm_data_url(image_path: str, max_side: int = VLM_MAX_SIDE) -> str:
    """Read an image and return its adaptive VLM data URL."""
    import cv2
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"cannot read image: {image_path}")
    return _encode_frame_data_url(frame, max_side)


def _tile_frame(frame) -> list:
    """Split a frame with an extreme aspect ratio (e.g. a long chat screenshot) into
    ordered top→bottom / left→right tiles along the long axis, so each tile keeps
    enough real pixels after the VLM_MAX_SIDE downscale to stay readable — a single
    whole-frame encode of a 4000px-tall image leaves ~5px text, effectively blind
    (OCR at full res still works; this is only about what direct vision sees).
    Ratio ≤ 2.5 → returns [frame] unchanged (byte-identical single-image path for
    normal images). Ratio > 2.5 → ceil(ratio/2) tiles, capped at 3, ~8% overlap
    between adjacent tiles so text spanning a boundary isn't cleanly cut in half."""
    h, w = frame.shape[:2]
    long_side, short_side = max(h, w), min(h, w)
    ratio = long_side / short_side if short_side else 1.0
    if ratio <= 2.5:
        return [frame]
    import math
    n = max(2, min(3, math.ceil(ratio / 2)))
    vertical = h >= w
    axis_len = h if vertical else w
    overlap = 0.08
    tile_len = axis_len / (1 + (n - 1) * (1 - overlap))
    step = tile_len * (1 - overlap)
    tiles = []
    for i in range(n):
        start = round(i * step)
        end = round(start + tile_len)
        if i == n - 1:
            end = axis_len
            start = max(0, end - round(tile_len))
        start = max(0, min(start, axis_len - 1))
        end = max(start + 1, min(end, axis_len))
        tiles.append(frame[start:end, :] if vertical else frame[:, start:end])
    return tiles


def _validate_detail(detail: str) -> str | None:
    """None when `detail` is supported, otherwise a model-actionable error."""
    mode = (detail or "overview").strip().lower()
    if mode not in _DETAIL_MODES:
        return (
            f"[error] invalid detail '{detail}' — must be one of: "
            f"{', '.join(sorted(_DETAIL_MODES))}"
        )
    return None


def _visual_chart_structure(frame) -> bool:
    """Detect a high-confidence plotted chart from pixels, not OCR.

    This intentionally requires axis-like long horizontal/vertical strokes plus
    either another structural line or at least two persistent saturated hues.  It
    catches unreadable scans where OCR is empty without treating ordinary landscape
    prose or photographs as charts.
    """
    if frame is None:
        return False
    try:
        import cv2
        import numpy as np

        h, w = frame.shape[:2]
        if not h or w < h * 1.15:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        short = min(h, w)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=max(32, short // 9),
            minLineLength=max(48, int(short * 0.18)),
            maxLineGap=max(8, int(short * 0.025)),
        )
        horizontal = vertical = 0
        for x1, y1, x2, y2 in (lines.reshape(-1, 4) if lines is not None else []):
            dx, dy = abs(int(x2) - int(x1)), abs(int(y2) - int(y1))
            if dx >= max(48, int(short * 0.18)) and dy <= max(3, int(dx * 0.08)):
                horizontal += 1
            if dy >= max(48, int(short * 0.18)) and dx <= max(3, int(dy * 0.08)):
                vertical += 1
        if not (horizontal and vertical):
            return False

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        colorful = (hsv[:, :, 1] >= 90) & (hsv[:, :, 2] >= 70)
        hue_bins = np.bincount((hsv[:, :, 0][colorful] // 18).astype(int), minlength=10)
        persistent_hues = int((hue_bins >= max(40, (h * w) // 5000)).sum())
        return horizontal + vertical >= 3 or persistent_hues >= 2
    except Exception:
        return False


def _visual_grid_structure(frame) -> bool:
    """Detect a dense table/heatmap-like layout from pixels when OCR is absent."""
    if frame is None:
        return False
    try:
        import cv2
        import numpy as np

        h, w = frame.shape[:2]
        if not h or w < h * 1.15:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        short = min(h, w)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=max(28, short // 11),
            minLineLength=max(42, int(short * 0.14)),
            maxLineGap=max(8, int(short * 0.02)),
        )
        horizontal = vertical = 0
        for x1, y1, x2, y2 in (lines.reshape(-1, 4) if lines is not None else []):
            dx, dy = abs(int(x2) - int(x1)), abs(int(y2) - int(y1))
            if dx >= max(42, int(short * 0.14)) and dy <= max(3, int(dx * 0.08)):
                horizontal += 1
            if dy >= max(42, int(short * 0.14)) and dx <= max(3, int(dy * 0.08)):
                vertical += 1
        return horizontal >= 3 and vertical >= 3 and horizontal + vertical >= 8
    except Exception:
        return False


def _looks_like_chart(boxes: list, frame) -> bool:
    """Pure pixel/OCR chart diagnostic; never auto-routes a read_image request."""
    import re

    if frame is None:
        return False
    h, w = frame.shape[:2]
    if w < h * 1.15:
        return False
    if _visual_chart_structure(frame):
        return True
    texts = [str(b.get("text", "")).strip() for b in boxes]
    joined = " ".join(texts).lower()
    chart_terms = (
        "margin", "revenue", "profit", "price index", "year-over-year",
        "moving average", "industry", "industries", "policy period",
        "source:", "trend", "legend", "axis", "change",
        "กำไร", "รายได้", "ดัชนี", "แนวโน้ม", "แหล่งที่มา",
    )
    term_hits = sum(term in joined for term in chart_terms)
    numeric = sum(bool(re.search(r"\d", t)) for t in texts)
    percents = sum("%" in t for t in texts)
    years = sum(bool(re.search(r"(?:19|20)?\d{2}", t)) for t in texts)
    return term_hits >= 2 and numeric >= 10 and percents >= 3 and years >= 3


def _looks_like_market_slide(boxes: list, frame) -> bool:
    """Pure pixel/OCR dense-slide diagnostic; never auto-routes tool behavior."""
    import re

    if frame is None:
        return False
    h, w = frame.shape[:2]
    if w < h * 1.15:
        return False
    if _visual_grid_structure(frame):
        return True
    texts = [str(b.get("text", "")).strip() for b in boxes]
    joined = " ".join(texts).lower()
    numeric = sum(bool(re.search(r"\d", t)) for t in texts)
    percents = sum("%" in t for t in texts)
    source_footer = "source:" in joined or "แหล่งที่มา" in joined
    return len(texts) >= 20 and numeric >= 8 and (source_footer or percents >= 5)


def _looks_like_range_chart(boxes: list) -> bool:
    """Recognize low/current/high or valuation-range layouts.

    These charts encode one observation across the full page width, so ordinary
    left/right panel crops destroy row alignment even though both crops are sharp.
    """
    import re

    texts = " ".join(str(b.get("text", "")) for b in boxes).lower()
    anchors = (
        bool(re.search(r"\blow(?:\s+level)?\b|ระดับต่ำ", texts)),
        bool(re.search(r"\bcurrent\b|ปัจจุบัน", texts)),
        bool(re.search(r"\bhigh(?:\s+level)?\b|ระดับสูง", texts)),
    )
    return all(anchors) and bool(
        re.search(r"z-?score|valuation\s+range|range|ช่วงมูลค่า", texts)
    )


def _ocr_label_assist(boxes: list, limit: int = 60) -> str:
    """Return spelling labels only—never numeric chart evidence.

    Text-heavy OCR can dominate a small VLM's pixel reasoning. For charts/slides,
    retain only non-numeric labels so OCR helps name a series or asset while all
    values, alignment, geometry, and meaning still come from vision.
    """
    import re

    labels = []
    seen = set()
    for box in boxes:
        text = re.sub(r"\s+", " ", str(box.get("text", ""))).strip()
        key = text.casefold()
        if (
            not text
            or key in seen
            or re.search(r"\d|[%$€£¥]", text)
            or not re.search(r"[A-Za-zก-๙]", text)
        ):
            continue
        seen.add(key)
        labels.append(text)
        if len(labels) >= limit:
            break
    return "\n".join(labels)


def _chart_detail_frames(frame) -> list[tuple[str, object]]:
    """Return native-resolution crops for dense landscape charts.

    The overview preserves global relationships.  Two overlapping panel crops
    preserve axes, line endpoints, annotations, and legends; two footer crops
    preserve tiny source/footnote text without squeezing a full-width strip back
    down to 1024px.  Coordinates are intentionally layout-generic rather than
    tied to one publisher's slide template.
    """
    h, w = frame.shape[:2]

    def crop(x0: float, y0: float, x1: float, y1: float):
        xa, xb = round(x0 * w), round(x1 * w)
        ya, yb = round(y0 * h), round(y1 * h)
        return frame[max(0, ya):min(h, yb), max(0, xa):min(w, xb)]

    if w >= h:
        return [
            ("left panel/detail", crop(0.03, 0.10, 0.55, 0.91)),
            ("right panel/detail", crop(0.47, 0.10, 0.99, 0.91)),
            ("bottom-left legend/source", crop(0.03, 0.82, 0.56, 0.995)),
            ("bottom-right legend/source", crop(0.44, 0.82, 0.99, 0.995)),
        ]
    return [
        ("top panel/detail", crop(0.03, 0.05, 0.97, 0.54)),
        ("bottom panel/detail", crop(0.03, 0.43, 0.97, 0.91)),
        ("bottom-left legend/source", crop(0.03, 0.82, 0.56, 0.995)),
        ("bottom-right legend/source", crop(0.44, 0.82, 0.99, 0.995)),
    ]


def _range_chart_detail_frames(frame) -> list[tuple[str, object]]:
    """Full-width row bands for low/current/high and valuation-range charts."""
    h, w = frame.shape[:2]

    def crop(y0: float, y1: float):
        ya, yb = round(y0 * h), round(y1 * h)
        xa, xb = round(0.06 * w), round(0.98 * w)
        return frame[max(0, ya):min(h, yb), xa:xb]

    return [
        ("top rows: labels + low/current/high aligned", crop(0.12, 0.45)),
        ("middle rows: labels + low/current/high aligned", crop(0.35, 0.68)),
        ("bottom rows: labels + low/current/high aligned", crop(0.57, 0.88)),
        ("legend/source/definitions footer", crop(0.78, 0.995)),
    ]


def _market_grid_detail_frames(frame) -> list[tuple[str, object]]:
    """Column-locked crops for very dense ranking/return quilts.

    Three overlapping vertical bands keep a cell's label/value aligned beneath its
    own header.  The generic half-slide crop still leaves 6-8 year columns visible
    at once and the VLM can silently borrow a bottom value from the next column
    (observed on the real Guide to the Markets asset-class-return quilt).
    """
    h, w = frame.shape[:2]

    def crop(x0: float, y0: float, x1: float, y1: float):
        xa, xb = round(x0 * w), round(x1 * w)
        ya, yb = round(y0 * h), round(y1 * h)
        return frame[max(0, ya):min(h, yb), max(0, xa):min(w, xb)]

    return [
        ("left table columns (header-to-bottom locked)", crop(0.02, 0.08, 0.40, 0.88)),
        ("center table columns (header-to-bottom locked)", crop(0.31, 0.08, 0.70, 0.88)),
        ("right/latest table columns (header-to-bottom locked)", crop(0.61, 0.08, 0.995, 0.88)),
        ("source/definitions footer", crop(0.03, 0.82, 0.995, 0.995)),
    ]


def _clean_grid_label(text: str) -> str:
    """Conservative cleanup for fragments Apple Vision commonly splits in quilts."""
    import re

    text = re.sub(r"\s+", " ", text).strip()
    substitutions = (
        (r"\bD\s*M\b", "DM"),
        (r"\bEquitie\s+s\b", "Equities"),
        (r"\bDive\s+rsifie\s+d\b", "Diversified"),
        (r"\be\s*[xX]\s*-\s*JP\b", "ex-JP"),
        (r"\bAsia\s+n\b", "Asian"),
        (r"\bEM\s+e\s*x-", "EM ex-"),
        (r"\bEM\s+ex-\s+Asia\b", "EM ex-Asia"),
        (r"\bAPAC\s+e\s*x", "APAC ex"),
        (r"\bU\.\s*S\.\s*IG\b", "U.S. IG"),
        # Vision sometimes recognizes tiny "U.S." glyphs as Thai บ.ร. on these
        # slides; restrict the repair to the complete investment-grade label.
        (r"\bบ\.ร\.\s*IG\b", "U.S. IG"),
    )
    for pattern, replacement in substitutions:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _reconstruct_ranked_columns(boxes: list) -> str | None:
    """Reconstruct dense return/ranking quilts as header-locked label/value rows.

    OCR reading order is row-major, so a flat dump interleaves 10-15 year columns.
    This function uses bounding-box x centers to assign every fragment to the
    nearest detected column header, then groups name fragments until the printed
    percentage beneath them.  It intentionally activates only when at least five
    recognizable time/summary headers and three rows per selected column exist.
    """
    import re

    header_re = re.compile(
        r"^(?:20\d{2}|2Q\s*['’]?\s*\d{2}|YTD\s*['’]?\s*\d{2}|"
        r"Ann\.?\s*(?:Ret|Vol)\.?)$",
        re.IGNORECASE,
    )
    headers = []
    for b in boxes:
        text = re.sub(r"\s+", " ", str(b.get("text", "")).strip())
        if header_re.match(text):
            headers.append({
                "text": text,
                "cx": b["x"] + b["w"] / 2,
                "cy": b["y"] + b["h"] / 2,
            })
    headers.sort(key=lambda h: h["cx"])
    if len(headers) < 5:
        return None

    # Sensor-only output is query-agnostic. Keep the five rightmost/latest summary
    # columns inline; the agent's original conversation tells it which ones matter.
    selected = headers[-5:]

    percent_re = re.compile(r"[-+]?\s*\d+(?:\.\s*\d+)?\s*%")
    sections: list[str] = []
    all_header_x = [h["cx"] for h in headers]
    for header in selected:
        idx = headers.index(header)
        left = (
            (all_header_x[idx - 1] + header["cx"]) / 2
            if idx > 0 else max(0.0, header["cx"] - 0.035)
        )
        right = (
            (header["cx"] + all_header_x[idx + 1]) / 2
            if idx + 1 < len(headers) else min(1.0, header["cx"] + 0.035)
        )
        fragments = []
        for b in boxes:
            cx = b["x"] + b["w"] / 2
            cy = b["y"] + b["h"] / 2
            if left <= cx < right and 0.13 < cy < header["cy"] - 0.025:
                fragments.append((cy, b["x"], str(b.get("text", "")).strip()))
        fragments.sort(key=lambda item: (-item[0], item[1]))

        rows: list[tuple[str, str]] = []
        label_parts: list[str] = []
        for _cy, _x, text in fragments:
            match = percent_re.search(text.replace("−", "-"))
            if match:
                label = _clean_grid_label(" ".join(label_parts))
                value = re.sub(r"\s+", "", match.group(0))
                if label:
                    rows.append((label, value))
                label_parts = []
            elif text:
                label_parts.append(text)
        if len(rows) < 3:
            continue
        lines = [f"### {header['text']}"]
        lines.extend(
            f"{rank}. {label} | {value}"
            for rank, (label, value) in enumerate(rows, start=1)
        )
        sections.append("\n".join(lines))

    if not sections:
        return None
    return (
        "[OCR ASSIST — coordinate-grouped grid transcription; verify every "
        "label/value against the attached pixels]\n"
        + "\n\n".join(sections)
    )


def _axis_constraints(boxes: list) -> str | None:
    """Extract candidate y-axis ranges from aligned percentage tick labels.

    This is an OCR reading aid, not visual authority. The VLM must verify the
    candidate ticks against the attached pixels before using them.
    """
    import re

    ticks = []
    # Vision OCR often inserts a space after a sign ("- 1%") and may append a
    # chart-stroke/footnote glyph ("15% -", "-10% †"). Accept those bounded
    # decorations without matching arbitrary percentage prose.
    tick_re = re.compile(
        r"^\s*([-+]?)\s*(\d+(?:\.\d+)?)\s*%\s*(?:[-†•7])?\s*$"
    )
    for b in boxes:
        match = tick_re.match(str(b.get("text", "")).replace("−", "-"))
        if not match:
            continue
        ticks.append((
            b["x"] + b["w"] / 2,
            b["y"] + b["h"] / 2,
            float(f"{match.group(1)}{match.group(2)}"),
        ))
    if len(ticks) < 4:
        return None

    clusters: list[list[tuple[float, float, float]]] = []
    for tick in sorted(ticks, key=lambda t: t[0]):
        if not clusters or abs(tick[0] - sum(x for x, _, _ in clusters[-1]) / len(clusters[-1])) > 0.02:
            clusters.append([tick])
        else:
            clusters[-1].append(tick)

    axes = []
    for cluster in clusters:
        values = sorted({value for _, _, value in cluster})
        if len(values) < 4:
            continue
        diffs = [b - a for a, b in zip(values, values[1:])]
        step = sorted(diffs)[len(diffs) // 2]
        # Real axis ticks are evenly spaced numerically; a ranking-table column is
        # also vertically aligned percentages, but its return gaps are irregular.
        if step <= 0 or any(
            abs(diff - step) > max(0.25, abs(step) * 0.15)
            for diff in diffs
        ):
            continue
        x = sum(item[0] for item in cluster) / len(cluster)
        floor, ceiling = min(values), max(values)
        guard = (
            " OCR SUGGESTS NONNEGATIVE: verify the zero-floor tick in pixels."
            if floor >= 0 else ""
        )
        axes.append(
            f"- y-axis near normalized x={x:.3f}: {floor:g}% to {ceiling:g}% "
            f"(ticks: {', '.join(f'{v:g}%' for v in values)}).{guard}"
        )
    if not axes:
        return None
    return (
        "[OCR AXIS HINTS — candidate tick transcription; verify signs/range "
        "against the attached pixels before use]\n"
        + "\n".join(axes)
    )


def _detail_pack_note(kind: str, labels: list[str]) -> str:
    """Model-facing map/checklist for progressive detail crops.

    The original whole-image overview was already attached by the required first
    read and remains active in this turn; this note maps only the newly added crops.
    """
    numbered = [f"{i}={label}" for i, label in enumerate(labels, start=1)]
    view_count = len(labels)
    heading = {
        "chart": "CHART DETAIL PACK",
        "grid": "MARKET GRID DETAIL PACK",
        "range": "RANGE CHART DETAIL PACK",
    }.get(kind, "MARKET SLIDE DETAIL PACK")
    common = (
        f"[{heading} — {view_count} progressive detail views attached]\n"
        f"New-crop order: {', '.join(numbered)}. The original overview from the first read remains in context.\n"
        "VISION-FIRST: compare these crops with that original whole image. "
        "Use OCR only to help spell/transcribe small labels; it can misread signs, "
        "digits, order, and cell alignment, so it must never override pixels. "
        "First establish title, data date, panel/table "
        "structure, units, horizons, legends and definitions; finish with the "
        "source/date/footnotes. Treat an image source line as provenance printed "
        "in the slide, not as a web citation. Before finalizing, verify every "
        "top/bottom/latest claim against the evidence block/view it came from. "
        "Words like always/every period/only require checking every relevant "
        "column and naming exceptions. Do not invent macro drivers (rates, flows, "
        "policy effects) that the slide does not state; omit them or explicitly "
        "label them as hypotheses outside the image."
    )
    if kind == "range":
        return common + (
            "\nFor range charts: read the visual legend BEFORE interpreting rows. "
            "The horizontal axis may be a standardized score while printed labels "
            "at bar endpoints/markers use the real unit; never substitute axis "
            "coordinates for those printed values. Follow the legend's encoding: "
            "bar endpoints are Low/High and the distinct marker is Current. Read "
            "each row independently from its full-width row-band crop—do not copy "
            "one generic range across assets and do not shift values between rows. "
            "Do not name a marker's color or shape unless it is clearly verified "
            "from the visual legend; never assume a conventional blue circle. "
            "For many rows, report only genuinely visible outliers or groupings—"
            "do not label every row mid-range without inspecting each marker. "
            "If a printed number remains too small, describe the current marker's "
            "relative position within that row and explicitly omit the unreadable "
            "number instead of inventing it. OCR labels help spelling only."
        )
    if kind == "chart":
        return common + (
            "\nFor charts: map every series to its correct axis (including dual "
            "axes); locate the true far-right/latest endpoint before describing the "
            "latest value; inspect direction and turning points. Verify every sign "
            "against the plotted zero line and axis minimum. For highlighted "
            "periods, use the actual shaded rectangle edges—not an annotation "
            "arrow/text span—as the date boundaries."
            "\nValues inferred from line/bar positions are approximate unless an "
            "a clearly visible printed label supports them. Separate observed chart "
            "evidence from interpretation; never claim a policy caused a move "
            "unless the chart explicitly states that causal link. If tracing an "
            "endpoint to its legend+axis is ambiguous, report the direction and "
            "uncertainty instead of inventing a numeric range. A value outside an "
            "OCR AXIS HINTS are candidates only; verify every sign and endpoint "
            "against pixels."
        )
    return common + (
        "\nFor tables/heatmaps/ranking quilts: read column headers and time horizons "
        "before cells; preserve top-to-bottom rank within each column; identify "
        "latest/YTD/annualized columns explicitly; do not infer magnitude from color "
        "unless a legend defines that encoding. Lock each requested value to the "
        "header directly above it and verify the top and bottom cells independently "
        "before comparing columns. Treat OCR grouping as a reading hint and verify "
        "each printed cell directly in pixels; "
        "comparisons or interpretations must remain separate from observations."
    )


def _crop_to_tempfile(image_path: str, anchor: tuple[float, float], zoom: float) -> str:
    """Crop a `1/zoom`-sized window around `anchor` (normalized x,y) from the
    ORIGINAL image on disk at native resolution, write it to a temp PNG, and return
    the path (caller unlinks). A native-res crop spends the pipeline on fewer real
    source pixels than a full-frame read, so BOTH the VLM encode and the OCR pass on
    it read a small detail far sharper — genuine zoom (real pixels), not upscaling
    (invented ones). Returning a file (not base64) lets the zoom path also re-run
    OCR on the crop (`_ocr_layout` takes a path)."""
    import cv2
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"cannot read image: {image_path}")
    h, w = frame.shape[:2]
    frac = 1.0 / zoom
    cw, ch = max(1, round(w * frac)), max(1, round(h * frac))
    cx, cy = anchor
    x0 = min(max(0, round(cx * w - cw / 2)), max(0, w - cw))
    y0 = min(max(0, round(cy * h - ch / 2)), max(0, h - ch))
    crop = frame[y0:y0 + ch, x0:x0 + cw]
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    if not cv2.imwrite(tmp.name, crop):
        Path(tmp.name).unlink(missing_ok=True)
        raise ValueError("crop PNG write failed")
    return tmp.name


def _exif_normalize(path: str) -> str:
    """If `path` carries a non-identity EXIF orientation tag, write an upright copy
    to a temp PNG and return its path; otherwise return `path` unchanged (no temp
    file — zero overhead for the common case). cv2.imread ignores EXIF orientation
    entirely, so a photo shot in portrait but tagged rotated is processed sideways
    by both OCR and the VLM encode unless corrected here, once, before either reads
    the file. Never raises — falls back to the original path on any failure."""
    try:
        from PIL import Image, ImageOps
        with Image.open(path) as im:
            orientation = im.getexif().get(0x0112, 1)
            if orientation in (1, 0, None):
                return path
            fixed = ImageOps.exif_transpose(im)
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            fixed.save(tmp.name, format="PNG")
            return tmp.name
    except Exception:
        return path


_HEIC_EXTS = (".heic", ".heif")


def _convert_heic(path: str) -> str:
    """Convert a HEIC/HEIF file to PNG via macOS-native `sips` (no new pip deps —
    cv2.imread returns None for HEIC, the iPhone camera default). Returns the temp
    PNG path; raises on failure so the caller can fall back to the original path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    result = subprocess.run(
        ["sips", "-s", "format", "png", path, "--out", tmp.name],
        capture_output=True, timeout=20,
    )
    if result.returncode != 0 or not Path(tmp.name).exists() or Path(tmp.name).stat().st_size == 0:
        Path(tmp.name).unlink(missing_ok=True)
        raise Exception(f"sips conversion failed: {result.stderr.decode(errors='ignore')[:200]}")
    return tmp.name


def _maybe_convert_heic(local_path: str) -> str:
    """Dispatch HEIC/HEIF → PNG conversion by extension; if the extension doesn't
    say so, only probe further when cv2 genuinely can't decode the file (avoids a
    wasted sips subprocess on every normal jpg/png read) and sips itself recognizes
    the format — a truly corrupt/unsupported file falls through unchanged. Returns
    `local_path` unchanged on any failure (never raises)."""
    ext = Path(local_path).suffix.lower()
    if ext not in _HEIC_EXTS:
        try:
            import cv2 as _cv2
            if _cv2.imread(local_path) is not None:
                return local_path
        except Exception:
            return local_path
        try:
            probe = subprocess.run(["sips", "-g", "format", local_path],
                                    capture_output=True, timeout=10)
            if probe.returncode != 0:
                return local_path
        except Exception:
            return local_path
    try:
        return _convert_heic(local_path)
    except Exception:
        return local_path


def _decode_qr(image_path: str) -> list:
    """Decode any QR codes in the image via cv2's built-in detector (no extra deps —
    Thai payment slips carry QR codes the VLM cannot read visually; this recovers
    the payload as text). Swallows all failures — a QR decode error must never
    affect the OCR result. Returns a deduplicated, order-preserving, non-empty
    payload list, [] if none found or on any failure."""
    try:
        import cv2
        frame = cv2.imread(image_path)
        if frame is None:
            return []
        detector = cv2.QRCodeDetector()
        payloads: list[str] = []
        try:
            ok, decoded, _, _ = detector.detectAndDecodeMulti(frame)
            if ok:
                payloads = list(decoded)
        except Exception:
            # detectAndDecodeMulti unavailable on this cv2 build → single-code fallback
            data, _, _ = detector.detectAndDecode(frame)
            if data:
                payloads = [data]
        seen = set()
        out = []
        for p in payloads:
            if p and p not in seen:
                seen.add(p)
                out.append(p)
        return out
    except Exception:
        return []


def _enhance_for_ocr(image_path: str) -> str:
    """Grayscale + CLAHE contrast boost (+ 2x upscale if small) to a temp PNG, for a
    one-time OCR retry when the first pass finds zero text — low-contrast/dark scans
    and tiny print are the common Vision-OCR-miss cases this recovers. Raises on
    failure (caller's retry wiring falls through to the existing no-text path)."""
    import cv2
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"cannot read image: {image_path}")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    h, w = enhanced.shape[:2]
    if max(h, w) < 1200:
        enhanced = cv2.resize(enhanced, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    cv2.imwrite(tmp.name, enhanced)
    return tmp.name


_TEXT_ONLY_OCR_MAX_CHARS = 12000


def _full_ocr_boxes(image_path: str) -> list[dict]:
    """Return full-image OCR boxes, with the existing one-time enhancement retry."""
    boxes = _ocr_layout(image_path)
    if boxes:
        return boxes
    enhanced_tmp: str | None = None
    try:
        enhanced_tmp = _enhance_for_ocr(image_path)
        return _ocr_layout(enhanced_tmp)
    except Exception:
        return []
    finally:
        if enhanced_tmp:
            try:
                os.unlink(enhanced_tmp)
            except Exception:
                pass


def _bound_full_ocr(text: str) -> str:
    """Bound extreme OCR deterministically without silently summarizing it."""
    if len(text) <= _TEXT_ONLY_OCR_MAX_CHARS:
        return text
    marker = (
        f"[OCR truncated for context safety: original={len(text)} chars; "
        "deterministic head/tail retained — no summary was generated]"
    )
    available = max(_TEXT_ONLY_OCR_MAX_CHARS - len(marker) - 2, 2)
    head_len = available // 2
    tail_len = available - head_len
    return f"{text[:head_len].rstrip()}\n{marker}\n{text[-tail_len:].lstrip()}"


def full_ocr_for_text_only(image_path: str, source: str = "image") -> str:
    """Produce bounded full-image OCR for a backend that cannot accept pixels.

    The raw lines remain in Vision reading order.  A deterministic table-layout
    candidate and QR payloads are additional evidence when the existing helpers
    can recover them; nothing here asks an LLM to summarize or describe pixels.
    """
    display_source = str(source)[:160]
    try:
        boxes = _full_ocr_boxes(image_path)
    except Exception:
        boxes = []
    raw_text = "\n".join(str(box.get("text", "")) for box in boxes if box.get("text"))
    if not raw_text.strip():
        return (
            f"[TEXT-ONLY IMAGE FALLBACK — source: {display_source}]\n"
            "[no readable text found — this text-only model cannot understand image "
            "pixels, and OCR found no readable text]"
        )

    sections = [f"[OCR — full reading order]\n{raw_text}"]
    try:
        table_md = _reconstruct_table(boxes)
    except Exception:
        table_md = None
    if table_md:
        sections.append(f"[OCR TABLE LAYOUT CANDIDATE]\n{table_md}")
    try:
        qr_payloads = _decode_qr(image_path)
    except Exception:
        qr_payloads = []
    if qr_payloads:
        sections.append("[QR]\n" + "\n".join(qr_payloads))
    return (
        f"[TEXT-ONLY IMAGE FALLBACK — source: {display_source}]\n"
        + _bound_full_ocr("\n\n".join(sections))
    )


def _locate_find_box(boxes: list, find: str) -> dict | None:
    """Find the OCR box whose text contains `find` (case-insensitive substring).
    Multiple matches → the one with the SMALLEST box height (the small hard-to-read
    detail is usually the one worth zooming to, not a large heading that happens to
    repeat the text). None if no box matches."""
    needle = find.lower()
    matches = [b for b in boxes if needle in b.get("text", "").lower()]
    if not matches:
        return None
    return min(matches, key=lambda b: b["h"])


def _find_crop_bounds(box: dict, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Pixel crop bounds (x0,y0,x1,y1, top-down/left-right image space) centered on
    an OCR box, expanded to at least 25% of the image's width/height so surrounding
    context stays visible (not just the box's own tiny rectangle), clamped to image
    bounds. Box coords are Vision-style (origin BOTTOM-LEFT, y increases upward) —
    flipped here to top-down before indexing the pixel array (mirrors the sort
    convention in _reconstruct_table, -(y+h/2) = top-first)."""
    top = 1.0 - (box["y"] + box["h"])
    bottom = 1.0 - box["y"]
    left = box["x"]
    right = box["x"] + box["w"]
    cx, cy = (left + right) / 2, (top + bottom) / 2
    win_w = max(right - left, 0.25)
    win_h = max(bottom - top, 0.25)
    x0n, x1n = cx - win_w / 2, cx + win_w / 2
    y0n, y1n = cy - win_h / 2, cy + win_h / 2
    if x0n < 0: x1n -= x0n; x0n = 0.0
    if x1n > 1: x0n -= (x1n - 1); x1n = 1.0
    if y0n < 0: y1n -= y0n; y0n = 0.0
    if y1n > 1: y0n -= (y1n - 1); y1n = 1.0
    x0n, x1n = max(0.0, x0n), min(1.0, x1n)
    y0n, y1n = max(0.0, y0n), min(1.0, y1n)
    x0, x1 = max(0, round(x0n * img_w)), min(img_w, round(x1n * img_w))
    y0, y1 = max(0, round(y0n * img_h)), min(img_h, round(y1n * img_h))
    if x1 <= x0: x1 = min(img_w, x0 + 1)
    if y1 <= y0: y1 = min(img_h, y0 + 1)
    return x0, y0, x1, y1


def _crop_to_find_tempfile(image_path: str, box: dict) -> str:
    """Crop a native-res window around `box` (see _find_crop_bounds) from the
    ORIGINAL image on disk and write it to a temp PNG — mirrors _crop_to_tempfile's
    contract (caller unlinks)."""
    import cv2
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"cannot read image: {image_path}")
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = _find_crop_bounds(box, w, h)
    crop = frame[y0:y1, x0:x1]
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    if not cv2.imwrite(tmp.name, crop):
        Path(tmp.name).unlink(missing_ok=True)
        raise ValueError("crop PNG write failed")
    return tmp.name


def _download_url(url: str) -> str:
    """Download URL to a temp file, return local path. Raises on failure."""
    progress(f"downloading: {url[:70]}")
    suffix = ".jpg" if any(url.lower().endswith(e) for e in (".jpg", ".jpeg")) else ".png"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            Path(tmp.name).write_bytes(resp.read())
    except Exception:
        Path(tmp.name).unlink(missing_ok=True)
        raise
    return tmp.name




@tool
@_serialize_read_image
def read_image(source: str, region: str = "full", zoom: float = 1.0,
                find: str = "", detail: str = "overview") -> str:
    """Read and understand an image from a file path, URL, or the current screen.

    First call with default detail="overview" queues only the original image for
    direct agent vision; it deliberately does not run OCR or other content sensors.
    After the agent sees those pixels, follow-up modes can request Apple Vision OCR,
    table/QR assist, structural crops, or targeted zoom/find crops as needed.

    source   : file path in workspace / absolute path | https:// URL | "screen" (screenshot)
    region   : "full" (default) or a zoom anchor — center/top/bottom/left/right/
               top-left/top-right/bottom-left/bottom-right. Unknown name →
               "[error] invalid region '<name>' — must be...": retry with a listed
               name, do not guess.
    zoom     : magnification 1.0 (default) to 4.0, silently clamped if out of range.
               Detail too small/blurry → call read_image AGAIN on the SAME source
               with region=<anchor>+zoom (not "full") for a sharper native-res crop
               (re-runs OCR + queues for direct vision). Refuses to zoom the SAME
               region twice.
    find     : PREFERRED over region/zoom when you know the exact text/number to
               read — e.g. find="ยอดรวม" locates that OCR box and auto-crops a
               sharp native-res window around it (region/zoom ignored if find is
               set). No match → "[error] find: ไม่พบ '<text>' ...".
               Example: read_image("bill.jpg") first; total unreadable →
               read_image("bill.jpg", find="ยอดรวม").
    detail   : "overview" (default) is the REQUIRED first read and queues exactly
               one whole original image with NO OCR/classification/table/QR work.
               Only AFTER that image is attached in this outer turn may the agent
               explicitly call the SAME source with "text" for full-res OCR +
               table/QR assist, "chart" for chart/range crops, or "slide" for
               table/heatmap/presentation crops. The tool never auto-detects a
               detail mode from user wording and never receives a semantic question.

    Returns a direct-vision status on detail="overview". detail="text" returns OCR
    ASSIST text (large results spill to a workspace .md, sampled inline). A zoom/find call returns
    "[zoomed OCR — region=<name> x<zoom>]" or "[zoomed OCR — find=\"<text>\"]" plus
    "\n<crop text>" (and the crop via direct vision same-turn), or a no-text status
    if the crop has no OCR text. [error] prefix on failure, invalid region, "full"+
    zoom, or no find match. If the configured model is known text-only, no image is
    published: the full original-image OCR is returned instead, with an explicit
    limitation when OCR finds no readable text.
    """
    if not source or not source.strip():
        return "[error] source is required"

    detail = (detail or "overview").strip().lower()
    detail_err = _validate_detail(detail)
    if detail_err:
        return detail_err
    region = (region or "full").strip().lower()
    find = (find or "").strip()
    do_find = bool(find)
    if not do_find:
        region_err = _validate_region(region)
        if region_err:
            return region_err
    zoom = _clamp_zoom(zoom)
    do_zoom = zoom > 1.0 or region != "full"
    ignored_region_zoom = do_find and do_zoom  # find wins — noted in the return
    capability = get_capability()
    src = source.strip()
    needs_overview_first = (
        capability != TEXT_ONLY
        and src not in _OVERVIEW_SEEN
        and (do_find or do_zoom or detail != "overview")
    )
    deferred_sensor = ""
    if capability == TEXT_ONLY:
        phase(f"🖼 read_image · OCR/text-only · {source[:50]}")
    elif needs_overview_first:
        if do_find:
            deferred_sensor = f'find="{find}"'
        elif do_zoom:
            deferred_sensor = f'region="{region}", zoom={zoom:.1f}'
        else:
            deferred_sensor = f'detail="{detail}"'
        phase(
            f"🖼 read_image · original first (requested {deferred_sensor[:32]}) · "
            f"{source[:50]}"
        )
        if src in _OVERVIEW_PENDING:
            # The source has already been queued by another tool call in this
            # outer turn, but react.py has not published it yet. Do not run a
            # sensor or enqueue a second original: same-AIMessage ToolNode
            # calls can reach this branch before the next model boundary.
            return (
                f"[requested sensor deferred: {deferred_sensor} — the original image "
                "is queued but has not reached the model yet; no sensor ran. "
                "Wait for the next model step, then retry this SAME source."
            )
    elif do_find:
        phase(f"🖼 read_image · find '{find[:30]}' · {source[:50]}")
    elif do_zoom:
        phase(f"🖼 read_image · zoom {region} x{zoom:.1f} · {source[:50]}")
    elif detail == "text":
        phase(f"🖼 read_image · OCR/text · {source[:50]}")
    elif detail == "chart":
        phase(f"🖼 read_image · chart detail · {source[:50]}")
    elif detail == "slide":
        phase(f"🖼 read_image · slide/table detail · {source[:50]}")
    else:
        phase(f"🖼 read_image · original image · {source[:50]}")
    downloaded_tmp: str | None = None
    extra_tmps: list[str] = []   # normalize-step temp files (EXIF/HEIC) — cleaned in finally

    try:
        # ── resolve source ────────────────────────────────────────────────
        if src.lower() == "screen":
            pinned_screen = _SCREEN_TURN_SNAPSHOT[0]
            if pinned_screen and Path(pinned_screen).exists():
                # Progressive OCR/zoom/chart must inspect the exact native screenshot
                # whose encoded overview is already in the model's turn context.
                local_path = pinned_screen
            else:
                tmp_screen = f"/tmp/endeavor_screen_{uuid.uuid4().hex[:8]}.png"
                progress("กำลัง screenshot…")
                result = subprocess.run(
                    ["screencapture", "-x", "-m", tmp_screen],
                    capture_output=True, timeout=5,
                )
                if result.returncode != 0 or not Path(tmp_screen).exists():
                    return (
                        "[error] screencapture failed — กรุณาเปิดสิทธิ์ Screen Recording:\n"
                        "System Settings → Privacy & Security → Screen Recording → เปิดให้ Terminal/Claude Code"
                    )
                local_path = tmp_screen
                downloaded_tmp = tmp_screen

        elif src.startswith("http://") or src.startswith("https://"):
            local_path = _download_url(src)
            downloaded_tmp = local_path

        else:
            local_path = resolve_read_path(src)
            if not Path(local_path).exists():
                return f"[error] file not found: {src}"

        # ── HEIC/HEIF → PNG (sips, macOS-native, no new deps) ───────────────
        # cv2.imread returns None for HEIC — the iPhone camera default — so a
        # dropped-in iPhone photo would otherwise fail with an unhelpful error.
        # Runs before EXIF normalize so the tag-check below sees the converted file.
        converted_path = _maybe_convert_heic(local_path)
        if converted_path != local_path:
            extra_tmps.append(converted_path)
            local_path = converted_path

        # ── EXIF orientation normalize ──────────────────────────────────────
        # Devices tag rotated photos rather than storing pixels upright; cv2.imread
        # ignores the tag entirely, so an iPhone portrait shot is silently processed
        # sideways by both OCR and the VLM encode unless corrected here, once, before
        # either path consumes the file. Covers both branches below — zoom crops this
        # same (now-upright) local_path.
        fixed_path = _exif_normalize(local_path)
        if fixed_path != local_path:
            extra_tmps.append(fixed_path)
            local_path = fixed_path

        # Fail-safe progressive contract: if the model asks for OCR/crops before it
        # has seen the source, silently downgrade THIS transaction to overview-only.
        # The requested sensor is reported back as deferred so the next ReAct step
        # can retry it after inspecting the newly attached original pixels.
        if needs_overview_first:
            detail = "overview"
            region = "full"
            zoom = 1.0
            find = ""
            do_find = False
            do_zoom = False
            ignored_region_zoom = False

        if capability != TEXT_ONLY and (do_find or do_zoom):
            # Hard cap FIRST (V2-VLM-LOOP): bounds every zoom/find path this read
            # cycle, incl. a repeated region="full"+zoom whose [error] returns below
            # before the (A) dedup set is ever touched — that was the exact 25× loop.
            # Makes "the tool cannot loop" deterministically true, well under
            # RECURSION_LIMIT. find shares this cap (it's the same crop machinery).
            _ZOOM_ATTEMPTS[0] += 1
            if _ZOOM_ATTEMPTS[0] > _ZOOM_ATTEMPT_MAX:
                return ("[หยุดซูม] ซูมหลายครั้งแล้วบนภาพนี้ — ตอบจากรายละเอียดที่อ่านได้ "
                        "หรือบอกตรงๆ ว่าส่วนนั้นอ่านไม่ออก ห้ามเรียก read_image ซูมอีก")

            if do_find:
                # find wins over region/zoom (ignored, noted in the return below).
                # Needs OCR boxes of the ORIGINAL (now-normalized) image — cheap,
                # LLM-free, run fresh here rather than reusing a stale prior pass.
                # sig keys on `src` (the caller-visible source string), NOT local_path:
                # EXIF/HEIC normalization and URL downloads point local_path at a
                # fresh temp file every call, so a local_path sig never matches and
                # the [ซูมซ้ำ] steering silently dies for exactly those sources.
                sig = (src, f"find:{find.lower()}", 0.0)
                if sig in _ZOOM_SEEN:
                    return (f"[ซูมซ้ำ] เคยหา '{find}' บนภาพนี้แล้ว — crop + OCR อยู่ในบทสนทนาแล้ว "
                            f"ถ้ายังอ่านไม่ชัด ลองคำค้นอื่นหรือใช้ region+zoom แทน ห้ามหาคำเดิมซ้ำ")
                progress(f"finding: {find[:40]}…")
                match = _locate_find_box(_ocr_layout(local_path), find)
                if match is None:
                    return (f"[error] find: ไม่พบ '{find}' ใน OCR — ใช้ region+zoom แทน "
                            f"หรือเช็คตัวสะกด")
                _ZOOM_SEEN.add(sig)
                try:
                    crop_path = _crop_to_find_tempfile(local_path, match)
                except Exception as e:
                    return f"[error] find failed: {e}"
                try:
                    crop_image = _encode_vlm_data_url(crop_path)
                    crop_queued = _queue_image_bundle([crop_image])
                    crop_ocr = _hard_truncate(
                        "\n".join(b["text"] for b in _ocr_layout(crop_path)).strip(), 1500)
                except Exception as e:
                    return f"[error] find failed: {e}"
                finally:
                    Path(crop_path).unlink(missing_ok=True)
                note = (" (region/zoom ที่ส่งมาถูกข้าม — find มีสิทธิ์เหนือกว่า)"
                        if ignored_region_zoom else "")
                if crop_ocr:
                    return (f"[zoomed OCR — find=\"{find}\"]\n{crop_ocr}{note}\n"
                            + ("(รูป crop คมชัดแนบให้เห็นด้วย — ใช้ OCR + ภาพประกอบกัน)"
                               if crop_queued else _vision_budget_notice(1)))
                return (f"[zoomed view — find=\"{find}\" — no OCR text in crop, "
                        + (f"image attached for direct vision]{note}"
                           if crop_queued else f"direct vision deferred]{note}\n{_vision_budget_notice(1)}"))

            # B (V2-VLM-LOOP): region="full" + zoom is ambiguous — the old silent
            # center-crop read the same dead-center window every time, so a target
            # that wasn't centered was never shown and the model re-zoomed forever.
            # Force a named anchor instead.
            if region == "full":
                return ("[error] region='full' ใช้กับ zoom ไม่ได้ — ระบุบริเวณที่จะซูม: "
                        "center/top/bottom/left/right/top-left/top-right/bottom-left/"
                        "bottom-right (เช่น region='top-right', zoom=2.0)")
            # A (V2-VLM-LOOP): identical (source,region,zoom) already served this
            # read cycle → don't re-crop/re-queue; steer the model out of the loop.
            sig = (src, region, round(zoom, 1))  # src not local_path — see find sig note
            if sig in _ZOOM_SEEN:
                return (f"[ซูมซ้ำ] เคยซูม region={region} x{zoom:.1f} บนภาพนี้แล้ว — crop + OCR "
                        f"อยู่ในบทสนทนาแล้ว ถ้ายังอ่านไม่ชัด ลองบริเวณอื่น (มุม/บน/ล่าง) "
                        f"หรือตอบเท่าที่อ่านได้/บอกว่าอ่านไม่ออก ห้ามซูม region เดิมซ้ำ")
            _ZOOM_SEEN.add(sig)
            # Follow-up zoom on an already-read image: crop the ORIGINAL at native res.
            progress(f"zooming {region} x{zoom:.1f}…")
            anchor = _REGION_ANCHORS[region]  # validated & != "full" → key exists
            try:
                crop_path = _crop_to_tempfile(local_path, anchor, zoom)
            except Exception as e:
                return f"[error] zoom failed: {e}"
            try:
                crop_image = _encode_vlm_data_url(crop_path)
                crop_queued = _queue_image_bundle([crop_image])
                # D (V2-VLM-LOOP): OCR the sharp native-res crop too. The model
                # usually zooms to READ small text; OCR on the crop beats the
                # downscaled full-frame pass. Returning the text satisfies a
                # text-reading intent instead of handing back another wordless
                # image (the old loop trigger — "อ่านตัวอักษร" got no text back).
                # Bound it: the zoom path skips the full-read spill-to-.md machinery,
                # and a dense table crop could otherwise blow the MainState budget.
                crop_ocr = _hard_truncate(
                    "\n".join(b["text"] for b in _ocr_layout(crop_path)).strip(), 1500)
            except Exception as e:
                return f"[error] zoom failed: {e}"
            finally:
                Path(crop_path).unlink(missing_ok=True)
            if crop_ocr:
                return (f"[zoomed OCR — region={region} x{zoom:.1f}]\n{crop_ocr}\n"
                        + ("(รูป crop คมชัดแนบให้เห็นด้วย — ใช้ OCR + ภาพประกอบกัน)"
                           if crop_queued else _vision_budget_notice(1)))
            return (f"[zoomed view — region={region} x{zoom:.1f} — no OCR text in crop, "
                    + ("image attached for direct vision]"
                       if crop_queued else f"direct vision deferred]\n{_vision_budget_notice(1)}"))

        # Full (non-zoom) read = a fresh read cycle → clear the per-cycle zoom-dedup
        # set + attempt counter so later zooms on THIS image aren't blocked by a
        # prior turn's (A) / cap.
        _ZOOM_SEEN.clear()
        _ZOOM_ATTEMPTS[0] = 0

        # Exact-repeat dedup is keyed only by source + sensor mode. There is no
        # semantic question in the tool contract. Progressive overview→text/chart/
        # slide escalation is distinct, while an identical mode repeat is rejected.
        # source="screen" is also deduped within this outer turn because its native
        # snapshot is now pinned; the next outer turn starts with a fresh snapshot.
        is_screen = src.lower() == "screen"
        read_sig = (src, detail)
        if read_sig in _READ_SEEN:
            return (f"[อ่านซ้ำ] เคยอ่านภาพนี้ด้วย detail=\"{detail}\" แล้ว ผลลัพธ์เดิมคือ:\n"
                    f"{_READ_SEEN[read_sig]}\n"
                    "ห้ามเรียก mode เดิมซ้ำ — ตอบจากภาพที่อยู่ใน context หรือใช้ find=/region+zoom "
                    "ถ้าต้องการรายละเอียดเฉพาะจุด")

        # Hard per-source cap on TOTAL full-reads this turn, a final boundary
        # against runaway mode switching (mirrors the zoom cap).
        _READ_ATTEMPTS[src] = _READ_ATTEMPTS.get(src, 0) + 1
        if _READ_ATTEMPTS[src] > _READ_ATTEMPT_MAX:
            return (f"[ครบจำนวนอ่านภาพสูงสุด] อ่านภาพนี้ไปแล้ว {_READ_ATTEMPTS[src] - 1} ครั้งในการสนทนานี้ — "
                    f"ห้ามเรียก read_image กับภาพนี้อีก ตอบจากข้อมูลที่มีอยู่ในบทสนทนา หรือบอกตรงๆ "
                    f"ว่าข้อมูลที่ต้องการไม่มีในภาพ")

        # Once this endpoint/model is known to reject image input, never queue
        # another image or pretend that an overview was seen.  Return complete
        # deterministic OCR of the original source for every later read.
        if capability == TEXT_ONLY:
            result = full_ocr_for_text_only(local_path, src)
            if do_find or do_zoom:
                result += (
                    "\n[targeted image publication unavailable on this text-only model; "
                    "full original-image OCR returned instead]"
                )
            if src.lower() == "screen" and downloaded_tmp:
                # Text-only turns still pin the native screen source so a later
                # same-turn OCR request reads the exact pixels that produced the
                # first result, rather than a moving desktop snapshot.
                _pin_screen_turn_snapshot(downloaded_tmp)
                downloaded_tmp = None
            _READ_SEEN[read_sig] = result
            return result

        # ── Direct vision / structural image preparation ───────────────────
        # Stage 1 (overview) is intentionally pixels-only. Stage 2 chart/slide
        # needs the source frame for native-resolution crops. detail="text" does
        # not re-encode/re-attach pixels because the original overview is already
        # active for the rest of this outer turn.
        frame = None
        overview_urls: list[str] = []
        encode_error = ""
        if detail in {"overview", "chart", "slide"}:
            progress("encoding image…")
            try:
                import cv2 as _cv2
                frame = _cv2.imread(local_path)
                if frame is None:
                    raise ValueError(f"cannot read image: {local_path}")
                if detail == "overview":
                    overview_urls = [_encode_frame_data_url(frame, VLM_MAX_SIDE)]
            except Exception as e:
                log.warning("read_image direct-vision encode failed: %s", e)
                encode_error = str(e)

        # FIRST PASS CONTRACT: queue the whole original and return immediately.
        # No Apple Vision OCR, enhancement retry, table reconstruction, chart/grid
        # detection, axis hints, QR decode, or semantic preprocessing runs here.
        if detail == "overview":
            vision_queued = bool(overview_urls) and _queue_image_bundle(overview_urls)
            if vision_queued:
                if capability == UNKNOWN and not _retain_original_source(src, local_path):
                    log.warning("read_image fallback OCR may be unavailable for %s", src)
                _IMAGE_TURN_COUNT[0] += 1
                _OVERVIEW_PENDING.add(src)
                img_label = (
                    f"[ภาพที่ {_IMAGE_TURN_COUNT[0]}: {os.path.basename(source)}]\n"
                    if _IMAGE_TURN_COUNT[0] >= 2 else ""
                )
                result = (
                    f"{img_label}[original image queued for agent direct vision — "
                    "inspect pixels first; no OCR/table/QR sensor run]"
                )
                if deferred_sensor:
                    result += (
                        f"\n[requested sensor deferred: {deferred_sensor} — original was attached first. "
                        "Inspect it now; if that sensor is still needed, call read_image on the SAME "
                        f"source again with {deferred_sensor}.]"
                    )
            elif overview_urls:
                result = f"[original image direct vision deferred]\n{_vision_budget_notice(1)}"
            else:
                error_detail = f": {encode_error}" if encode_error else ""
                result = f"[error] read_image: original image encode failed{error_detail}"
            if vision_queued and is_screen and downloaded_tmp:
                _pin_screen_turn_snapshot(downloaded_tmp)
                downloaded_tmp = None  # ownership moved to outer-turn lifecycle
            _READ_SEEN[read_sig] = result
            return result

        # ── On-demand OCR (full-res, never raises) ─────────────────────────
        progress("Apple Vision OCR…")
        boxes = _ocr_layout(local_path)          # one Vision pass: text + bbox
        ocr_enhanced = False
        if not boxes:
            # Fail→detect→retry→correct: zero text often means a low-contrast/dark
            # scan or tiny print Vision missed on the raw pixels — retry once on a
            # CLAHE-boosted (+ upscaled if small) copy before giving up.
            progress("ไม่พบข้อความ ลองปรับภาพแล้ว OCR ซ้ำ…")
            enhanced_tmp: str | None = None
            try:
                enhanced_tmp = _enhance_for_ocr(local_path)
                retry_boxes = _ocr_layout(enhanced_tmp)
                if retry_boxes:
                    boxes = retry_boxes
                    ocr_enhanced = True
            except Exception:
                pass
            finally:
                if enhanced_tmp:
                    try:
                        os.unlink(enhanced_tmp)
                    except Exception:
                        pass
        ocr_lines = [b["text"] for b in boxes]
        ocr_text  = "\n".join(ocr_lines)

        # Full table/ranked reconstruction belongs only to the explicit text sensor.
        # chart/slide modes use OCR only for labels and deterministic crop selection.
        table_md = _reconstruct_table(boxes) if detail == "text" and ocr_text else None
        ranked_grid = (
            _reconstruct_ranked_columns(boxes)
            if detail == "text" and ocr_text and len(boxes) >= 80
            else None
        )

        # Progressive stage 2 is explicit only. The agent has already seen the
        # original overview and now asks for a sensor-resolution mode; read_image
        # never decides to escalate from user wording or auto classification.
        # Range/grid subtype selection below is deterministic geometry/OCR layout
        # used only AFTER that explicit request, to choose the most useful crops.
        is_range_chart = _looks_like_range_chart(boxes)
        detail_pack_kind = (
            "range" if is_range_chart and detail in {"chart", "slide"}
            else ("chart" if detail == "chart"
                  else ("grid" if detail == "slide" and len(boxes) >= 80
                        else ("slide" if detail == "slide" else "")))
        )
        detail_urls: list[str] = []
        detail_labels: list[str] = []
        if detail_pack_kind and frame is not None and not encode_error:
            try:
                detail_frames = (
                    _range_chart_detail_frames(frame)
                    if detail_pack_kind == "range"
                    else (_market_grid_detail_frames(frame)
                          if detail_pack_kind == "grid"
                          else _chart_detail_frames(frame))
                )
                for _view_name, _view in detail_frames:
                    detail_urls.append(_encode_frame_data_url(
                        _view, VLM_MAX_SIDE, lossless=True
                    ))
                    detail_labels.append(_view_name)
            except Exception as e:
                log.warning("read_image chart detail pack failed: %s", e)
                detail_urls.clear()
                detail_labels.clear()

        # Stage 2 queues ONLY explicitly requested structural crops; the original
        # overview remains active in dynamic_prompt. detail="text" adds no image.
        vision_bundle = detail_urls
        vision_queued = bool(vision_bundle) and _queue_image_bundle(vision_bundle)
        vision_deferred = ""
        if vision_queued:
            img_label = f"[รายละเอียดเพิ่มจากภาพเดิม: {os.path.basename(source)}]\n"
            chart_labels = detail_labels
        else:
            img_label = ""
            chart_labels = []
            if vision_bundle:
                vision_deferred = _vision_budget_notice(len(vision_bundle))

        vision_first_labels = _ocr_label_assist(boxes) if detail_pack_kind else ""
        if detail_pack_kind:
            progress("ส่งภาพจริงเป็นหลัก — OCR ช่วยเฉพาะชื่อ/คำ ไม่มีตัวเลข")
            primary = (
                "[OCR ASSIST — spelling labels only; no numeric evidence; "
                "read all values/relationships from pixels]\n"
                + (vision_first_labels or "(no reliable text labels)")
            )
        elif ranked_grid:
            progress("พบตารางหลายคอลัมน์ — OCR ช่วยจัดกลุ่ม; ให้ตรวจจากภาพจริง")
            # Keep the coordinate-locked block separate from the raw OCR artifact:
            # `_combine` may spill/sample a 400-line quilt and query-focused sampling
            # can omit the very requested columns. The compact locked block is
            # always prefixed inline below; raw OCR remains recoverable on disk.
            primary = f"[OCR — raw]\n{ocr_text}"
        elif table_md:
            progress(f"OCR พบโครงตาราง {len(ocr_lines)} เซลล์ — ให้ตรวจจากภาพจริง")
            primary = (
                "[OCR ASSIST — reconstructed table candidate; pixels are primary]\n"
                f"{table_md}"
            )
        elif ocr_text:
            progress(f"พบข้อความ {len(ocr_lines)} บรรทัด")
            tag = (
                "[OCR ASSIST — enhanced transcription; pixels are primary]"
                if ocr_enhanced
                else "[OCR ASSIST — transcription hint; pixels are primary]"
            )
            primary = f"{tag}\n{ocr_text}"
        else:
            primary = ""

        axis_constraints = (
            _axis_constraints(boxes)
            if detail == "text" and ocr_text
            else None
        )
        if axis_constraints:
            primary = (
                f"{axis_constraints}\n\n{primary}"
                if primary else axis_constraints
            )

        # detail="text" may suggest a targeted zoom when dense text is still tiny.
        if detail == "text" and ocr_text:
            hint = _zoom_region_hint(boxes)
            if hint:
                primary = f"{primary}\n{hint}"

        # QR/barcode decode is another on-demand text sensor, not overview work.
        # Thai payment slips carry QR codes the VLM cannot decode visually.
        qr_payloads = _decode_qr(local_path) if detail == "text" else []
        if qr_payloads:
            qr_section = "[QR]\n" + "\n".join(qr_payloads)
            primary = f"{primary}\n\n{qr_section}" if primary else qr_section

        if not primary:
            if detail == "text":
                result = "[OCR ASSIST — no text found; original image remains in direct-vision context]"
            elif vision_queued:
                result = f"{img_label}[no reliable OCR labels — detail crops attached for direct vision]"
            elif vision_deferred:
                result = f"{img_label}[no reliable OCR labels — direct vision deferred]\n{vision_deferred}"
            else:
                error_detail = f": {encode_error}" if encode_error else ""
                result = f"{img_label}[error] read_image: detail crop encode failed{error_detail}"
        else:
            combined = _combine([primary], source)
            if ranked_grid and not detail_pack_kind:
                result = f"{img_label}{ranked_grid}\n\n{combined}"
            else:
                result = f"{img_label}{combined}"
            if encode_error:
                result += f"\n[warning] direct-vision image unavailable; OCR only ({encode_error})"
            elif vision_deferred:
                result += f"\n{vision_deferred}"
        if chart_labels:
            result += f"\n\n{_detail_pack_note(detail_pack_kind, chart_labels)}"
        # Cache for the exact-repeat dedup above — an [error]/no-text result is
        # cached too, so a repeated source+detail mode is refused instead of
        # silently re-running the same deterministic sensor work forever. Screen is
        # safe to cache inside the outer turn because all sensors share one snapshot.
        _READ_SEEN[read_sig] = result
        return result

    except PermissionError as e:
        return f"[error] read_image: {e}"
    except Exception as e:
        return f"[error] read_image: {e}"
    finally:
        if downloaded_tmp:
            try:
                os.unlink(downloaded_tmp)
            except Exception:
                pass
        for _t in extra_tmps:
            try:
                os.unlink(_t)
            except Exception:
                pass

# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""read_image.py — OCR-only image reading for ENDEAVOR_LOCAL_AGENT_TH

Apple Vision OCR (full-res, Thai+English) extracts text + per-cell boxes, and
reconstructs a markdown table when the layout is a confident grid. QR/barcode
payloads are decoded alongside plain text. No VLM/vision-server dependency —
only the LLM needs to be running, so this tool cannot describe charts, photos,
or non-text image content — only what OCR/QR can read.

On zero-text, retries once on a contrast-boosted copy (low-contrast scans,
tiny print) before giving up. HEIC/EXIF are normalized transparently so a
photo shot in portrait but tagged rotated is never read sideways.
"""
from __future__ import annotations
import os
import re
import subprocess
import tempfile
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from langchain_core.tools import tool

from tools._ocr import read_layout as _ocr_layout
from tools._progress import progress, phase
from tools._safety import resolve_read_path

# Above this size the full combined result is always written to a workspace .md
# so the detail is recoverable beyond this turn — context compaction drops
# ToolMessages wholesale, so an on-disk artifact is the only durable copy for a
# dense read. Tiny reads stay inline-only: nothing worth persisting, re-readable.
_SAVE_FLOOR = 800
# Inline budget returned to state on the read turn. Within this, the full
# result lands inline (and is also saved); over it, a sampled view is returned
# with the .md path so the agent can read_file the rest.
_COMBINED_INLINE_MAX = 4500
# Prose-in-columns guard: a column whose shortest-column median cell length
# exceeds this is verbose enough to be prose, not table data. Only consulted
# for grids with no numeric content.
_PROSE_COL_CHARS = 14

_KNOWN_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic", ".heif", ".bmp", ".tif", ".tiff")
_HEIC_EXTS = (".heic", ".heif")


def _hard_truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n - 3].rstrip() + "..."


def _download_url(url: str) -> str:
    """Download URL to a temp file, return local path. Raises on failure.
    Preserves the real extension so extension-sensitive decode paths (cv2,
    sips/HEIC conversion, Apple Vision) see the actual format instead of a
    mislabeled .png — query string is stripped first so "img.webp?size=large"
    matches."""
    progress(f"downloading: {url[:70]}")
    url_path = urllib.parse.urlparse(url).path.lower()
    suffix = next((e for e in _KNOWN_IMG_EXTS if url_path.endswith(e)), ".png")
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


def _exif_normalize(path: str) -> str:
    """If `path` carries a non-identity EXIF orientation tag, write an upright
    copy to a temp PNG and return its path; otherwise return `path` unchanged
    (no temp file — zero overhead for the common case). cv2.imread ignores
    EXIF orientation entirely, so a photo shot in portrait but tagged rotated
    is processed sideways by OCR unless corrected here, once, before it reads
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


def _convert_heic(path: str) -> str:
    """Convert a HEIC/HEIF file to PNG via macOS-native `sips` (no new pip deps
    — cv2.imread returns None for HEIC, the iPhone camera default). Returns
    the temp PNG path; raises on failure so the caller can fall back to the
    original path."""
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
    """Dispatch HEIC/HEIF → PNG conversion by extension; if the extension
    doesn't say so, only probe further when cv2 genuinely can't decode the
    file (avoids a wasted sips subprocess on every normal jpg/png read) and
    sips itself recognizes the format — a truly corrupt/unsupported file
    falls through unchanged. Returns `local_path` unchanged on any failure
    (never raises)."""
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


def _prepare_for_ocr(local_path: str) -> tuple[str, list[str]]:
    """Chain HEIC→PNG then EXIF-upright normalization, in that order (the
    EXIF check must see the already-converted file). Returns (path OCR should
    read, list of temp files the caller must clean up — may be empty). No-op
    passthrough for plain non-HEIC, non-rotated images — the common case pays
    nothing extra."""
    extra_tmps: list[str] = []
    converted = _maybe_convert_heic(local_path)
    if converted != local_path:
        extra_tmps.append(converted)
        local_path = converted
    fixed = _exif_normalize(local_path)
    if fixed != local_path:
        extra_tmps.append(fixed)
        local_path = fixed
    return local_path, extra_tmps


def _decode_qr(image_path: str) -> list:
    """Decode any QR codes in the image via cv2's built-in detector (no extra
    deps — Thai payment slips carry QR codes that OCR reads as garbled text;
    this recovers the payload deterministically as text). Swallows all
    failures — a QR decode error must never affect the OCR result. Returns a
    deduplicated, order-preserving, non-empty payload list, [] if none found
    or on any failure."""
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
    """Grayscale + CLAHE contrast boost (+ 2x upscale if small) to a temp PNG,
    for a one-time OCR retry when the first pass finds zero text —
    low-contrast/dark scans and tiny print are the common Vision-OCR-miss
    cases this recovers. Raises on failure (caller's retry wiring falls
    through to the existing no-text path)."""
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
    # align into ≥2 columns but leave most cells empty. Require ≥60% filled —
    # allows genuine blanks + a few OCR drops, rejects chart scatter.
    filled = sum(1 for r in grid for c in r if c)
    if filled < len(grid) * len(cols) * 0.6:
        return None

    # Prose-in-columns guard: two side-by-side prose blocks also satisfy the grid
    # shape above (≥2 cols, most rows multi-cell) yet are not a data table —
    # reconstructing them scrambles the text. A real data table has terse cells:
    # at least one column of numbers or short labels; prose columns are uniformly
    # verbose. Reject only when there is NO numeric content AND the *shortest*
    # column is still verbose, so a real (esp. numeric) table is never rejected —
    # numeric content overrides the length check. char-length, not word-count:
    # word-count is degenerate for Thai — written without inter-word spaces,
    # every Thai cell is one "word".
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
    """Join the result sections (table / OCR / QR) into one string for state.

    Always saves the FULL combined result to a workspace .md once it exceeds
    _SAVE_FLOOR, then returns to state either the full text inline (when within
    _COMBINED_INLINE_MAX) or a sampled view + an actionable read_file hint. The
    save is unconditional (not gated on the inline cap) because context
    compaction drops ToolMessages wholesale — the .md is the only copy that
    survives, so "อ่านต่อ" needs it on disk regardless of how much fit inline
    this turn."""
    full = "\n\n".join(s for s in sections if s and s.strip())
    if len(full) <= _SAVE_FLOOR:
        return full  # trivial read — nothing worth persisting, re-readable as-is

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

    if not saved:
        if len(full) <= _COMBINED_INLINE_MAX:
            return full
        return _sample_coverage(full, source, max_chars=_COMBINED_INLINE_MAX - 40)

    saved_note = f"บันทึกผลเต็มที่: {saved}\n\n"
    if len(saved_note) + len(full) <= _COMBINED_INLINE_MAX:
        return saved_note + full

    prefix = (f"บันทึกผลเต็มที่: {saved}\n"
              f"(ส่วนด้านล่างเป็นเพียงตัวอย่าง — อ่านไฟล์นี้ด้วย read_file เพื่อรายละเอียดเต็ม)\n\n")
    sampled = _sample_coverage(full, source, max_chars=_COMBINED_INLINE_MAX - len(prefix) - 40)
    return prefix + sampled


@tool
def read_image(source: str) -> str:
    """Extract text from an image via Apple Vision OCR (full-res, Thai+English)
    — reconstructs a markdown table from cell layout when the image is a
    confident grid, and decodes QR/barcode payloads alongside plain text.

    source : file path in workspace / absolute path | https:// URL | "screen" (screenshot)

    Returns "[TABLE]"/"[OCR]" (+ "[QR]" if a code is found) when text is
    detected, or "[OCR] no text detected" when the image has no readable text
    (photos, diagrams, illustrations — this tool does not describe non-text
    image content, only what OCR/QR can read). Large results spill to a
    workspace .md with a sampled inline preview — follow with read_file for
    full detail. Returns [error] prefix on failure.
    """
    if not source or not source.strip():
        return "[error] source is required"

    phase(f"🖼 อ่านภาพ: {source[:50]}")
    downloaded_tmp: str | None = None
    extra_tmps: list[str] = []

    try:
        src = source.strip()

        # ── resolve source ────────────────────────────────────────────────
        if src.lower() == "screen":
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

        # HEIC→PNG then EXIF-upright — cv2.imread ignores EXIF orientation
        # entirely and returns None for HEIC (iPhone default), so `engine_path`
        # must always point at the normalized file. No-op passthrough for a
        # plain (non-HEIC, non-rotated) image.
        engine_path, extra_tmps = _prepare_for_ocr(local_path)

        # ── OCR (full-res, never raises) ──────────────────────────────────
        progress("Apple Vision OCR…")
        boxes = _ocr_layout(engine_path)
        ocr_enhanced = False
        if not boxes:
            # Fail→detect→retry→correct: zero text often means a low-contrast/
            # dark scan or tiny print Vision missed on the raw pixels — retry
            # once on a CLAHE-boosted (+ upscaled if small) copy before giving up.
            progress("ไม่พบข้อความ ลองปรับภาพแล้ว OCR ซ้ำ…")
            enhanced_tmp: str | None = None
            try:
                enhanced_tmp = _enhance_for_ocr(engine_path)
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

        # QR/barcode decode — Thai payment slips carry QR codes OCR reads as
        # garbled text; recover the payload as text alongside OCR. Never lets
        # a decode failure affect the OCR result (swallowed inside _decode_qr).
        qr_payloads = _decode_qr(engine_path)
        qr_section = "[QR]\n" + "\n".join(qr_payloads) if qr_payloads else ""

        ocr_lines = [b["text"] for b in boxes]
        ocr_text = "\n".join(ocr_lines)

        # Reconstruct a table from cell positions — a confident grid (≥2×2, most
        # rows multi-cell, dense) gives the agent structured rows/columns instead
        # of a flat dump. None for non-grid layouts (prose, charts, photos).
        table_md = _reconstruct_table(boxes) if ocr_text else None

        if table_md:
            progress(f"พบตาราง {len(ocr_lines)} เซลล์")
            primary = f"[TABLE — reconstructed from image OCR]\n{table_md}"
        elif ocr_text:
            progress(f"พบข้อความ {len(ocr_lines)} บรรทัด")
            tag = "[OCR — enhanced]" if ocr_enhanced else "[OCR]"
            primary = f"{tag}\n{ocr_text}"
        else:
            primary = ""

        if qr_section:
            primary = f"{primary}\n\n{qr_section}" if primary else qr_section

        if not primary:
            return "[OCR] no text detected"

        return _combine([primary], source)

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
        for _tmp in extra_tmps:
            try:
                os.unlink(_tmp)
            except Exception:
                pass

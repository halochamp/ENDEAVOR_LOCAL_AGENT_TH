"""OneOCR-shaped Thai OCR: a DB text-line detector and a CTC line recognizer on ONNX Runtime.

Why this shape: the 0.8B VLM writes a page one token at a time (~1,100 tokens,
each a full read of its weights), which costs ~28 s per page on 8 x86 cores.
A detector runs once per page and a CTC recognizer reads every character of
a line in one forward pass, so cost follows image area, not text length, and
a runaway loop is impossible by construction.

Pipeline (see the plan in NT-ONPREM-OCR-0.8B.md / the line-OCR plan):
  image -> [orientation] -> detector (probability map -> DB boxes, unclipped)
        -> reading order (rows by vertical overlap, left to right)
        -> crops from the full-resolution image -> recognizer (H=48, width
           follows the line, batched by width) -> greedy CTC decode with a
           confidence per line -> thai_norm.canonical -> joined text.

Model directory layout:
  det.onnx, det.yml     detector + its PaddleX inference.yml (thresholds)
  rec.onnx, rec.yml     recognizer + inference.yml (character_dict), or
  rec.onnx, rec_dict.txt  our trained recognizer + its charset, one per line
  config.json           optional overrides, e.g. {"det": {"limit": 1600}}
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyclipper

from .thai_norm import canonical

DEFAULTS: dict[str, dict[str, Any]] = {
    "det": {"limit": 1280, "thresh": 0.3, "box_thresh": 0.6, "unclip": 1.5, "min_size": 3,
            "max_candidates": 3000, "v_pad": 0.0,
            # split_gap: a detected line is cut where it has a blank stretch at
            # least this many line heights wide — two columns' lines or two
            # table cells the detector merged across a narrow gutter. Word
            # spacing, even justified Thai, stays well under it. 0 = off.
            "split_gap": 1.2},
    # drop_below / min_chars: lines under this confidence or length are dropped —
    # stamps, seals, signatures and handwritten marks the source text never has.
    # Tune per model on the court dev pages (train/lineocr/sweep_dev.py): for
    # rec-tiny-v0 the optimum was 0.92 / 3, halving whole-document CER.
    "rec": {"height": 48, "max_w": 3200, "batch": 16, "drop_below": 0.0, "min_chars": 1,
            # junk_below / junk_invalid: also drop a line under this confidence
            # whose Thai spelling is broken beyond this share (see invalid_share).
            "junk_below": 0.0, "junk_invalid": 1.0,
            # color: what a crop looks like before recognition. The recognizer
            # was trained on grey lines only (scans and greyscale synthetics), so
            # coloured text (slides, banners, coloured forms) reads as garbage
            # unless the crop is turned grey first. "gray" = luminance;
            # "gray_inv" also flips light-on-dark crops; "stretch" also spreads
            # low-contrast crops over the full range; "bgr" = unchanged.
            # A/B 2026-10-01 vs "bgr": stretch takes SEA-DocBench 0.521 -> 0.506
            # (25 empty pages -> 0), Wikisource 0.218 -> 0.212, scene text
            # 0.685 -> 0.240; court (0.0364) and docbench unchanged.
            # "flatten" subtracts each column's own background (from the
            # crop's top and bottom margins): any polarity, and gradients
            # (one decision per crop flips half a dark-to-light line wrong).
            # A/B on bench/robustness.py, 20 pages per condition: gradient
            # 0.0499 -> 0.0033 (worst page 0.97 -> 0.04), diagram, table and
            # colour better; court and docbench unchanged.
            "color": "flatten"},
    # ori: page orientation from the text itself, no extra model. A page whose
    # lines run vertically is turned 90 degrees, whichever way a few long lines
    # read with more confidence; a page that reads badly (median line
    # confidence under flip_below) is tried upside down on `sample` lines and
    # turned 180 if that reads better by `flip_margin`.
    "ori": {"enabled": True, "sample": 8, "flip_below": 0.8, "flip_margin": 0.15,
            # deskew_above: a page whose lines lean more than this (degrees) is
            # straightened and detected again; cutting each line out at an
            # angle loses thin marks (แต่ -> ต่ at 15-30 degrees).
            "deskew_above": 2.5},
    "join": {"thai_no_space": False},
    # OpenVINO bf16 uses AMX on Sapphire Rapids+ and falls back to f32 on CPUs
    # without it. Measured on a Xeon 8468 core (16 lines x 48 x 1024):
    # PP-OCRv6 tiny rec 541 ms on ORT fp32 vs 162 ms on OpenVINO bf16.
    "runtime": {"backend": "openvino", "precision": "bf16"},
}
DET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
DET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _yaml_postprocess(path: Path) -> dict:
    """PostProcess block of a PaddleX inference.yml, without a YAML dependency
    in the service image: only flat `key: value` pairs and the dict list."""
    out: dict[str, Any] = {}
    chars: list[str] = []
    in_post = in_dict = False
    for raw in path.read_text(encoding="utf-8").split("\n"):
        if raw.startswith("PostProcess:"):
            in_post = True
            continue
        if in_post and raw and not raw.startswith(" "):
            break
        if not in_post:
            continue
        line = raw.strip()
        if line.startswith("character_dict:"):
            in_dict = True
            continue
        if in_dict and line.startswith("- "):
            item = raw.split("- ", 1)[1]
            if item.startswith(("'", '"')) and item.endswith(item[0]) and len(item) >= 2:
                item = item[1:-1].replace("''", "'")
            chars.append(item)
            continue
        in_dict = False
        if ":" in line:
            k, v = (s.strip() for s in line.split(":", 1))
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    if chars:
        out["character_dict"] = chars
    return out


class _Model:
    """One network on OpenVINO (default) or ONNX Runtime; run(x) -> first output."""

    def __init__(self, path: Path, threads: int, backend: str, precision: str):
        self.backend = backend
        if backend == "openvino":
            import openvino as ov

            core = ov.Core()
            # No CPU pinning: OpenVINO pins the thread that runs a request, and
            # with one inference thread that is always core 0 — every service
            # worker thread (and every other process on the box) then queues on
            # one core. The OS scheduler spreads them; taskset still works.
            # The runtime cache keeps compiled kernels per input shape, and every page
            # size, crop width and batch size is a new shape: at the default 5,000
            # entries a worker grew from 0.75 to 1.3 GB over 300 images and kept going
            # (the line-ocr sidecar was OOM-killed). 200 entries holds it near 0.7 GB
            # for about 5% more time per image.
            self.compiled = core.compile_model(str(path), "CPU", {
                "INFERENCE_NUM_THREADS": threads, "NUM_STREAMS": 1,
                "PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": precision,
                "ENABLE_CPU_PINNING": os.environ.get("OV_CPU_PINNING", "0") == "1",
                "CPU_RUNTIME_CACHE_CAPACITY": int(os.environ.get("OV_RUNTIME_CACHE", "200"))})
            # An infer request serves one call at a time; the compiled model is
            # shared, so each calling thread gets its own request.
            self.local = threading.local()
            self.input = self.compiled.inputs[0].get_any_name()
            return
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.log_severity_level = 3  # paddle2onnx exports trip "unused initializer" warnings by the dozen
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Idle threads must not spin: on a shared box (and for core-seconds per
        # page) busy-waiting shows up as work that was never done.
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name

    def run(self, x: np.ndarray) -> np.ndarray:
        if self.backend == "openvino":
            req = getattr(self.local, "req", None)
            if req is None:
                req = self.local.req = self.compiled.create_infer_request()
            return req.infer({self.input: x})[0]
        return self.sess.run(None, {self.input: x})[0]


def order_points(pts: np.ndarray) -> np.ndarray:
    """4 points -> top-left, top-right, bottom-right, bottom-left."""
    pts = pts[np.argsort(pts[:, 0])]
    left, right = pts[:2][np.argsort(pts[:2, 1])], pts[2:][np.argsort(pts[2:, 1])]
    return np.array([left[0], right[0], right[1], left[1]], np.float32)


def _mini_box(contour: np.ndarray) -> tuple[np.ndarray, float]:
    rect = cv2.minAreaRect(contour)
    return order_points(cv2.boxPoints(rect)), min(rect[1])


def _box_score(prob: np.ndarray, box: np.ndarray) -> float:
    h, w = prob.shape
    xs, ys = box[:, 0], box[:, 1]
    x0, x1 = int(np.clip(np.floor(xs.min()), 0, w - 1)), int(np.clip(np.ceil(xs.max()), 0, w - 1))
    y0, y1 = int(np.clip(np.floor(ys.min()), 0, h - 1)), int(np.clip(np.ceil(ys.max()), 0, h - 1))
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    cv2.fillPoly(mask, [(box - [x0, y0]).astype(np.int32)], 1)
    return cv2.mean(prob[y0:y1 + 1, x0:x1 + 1], mask)[0]


def _unclip(box: np.ndarray, ratio: float) -> np.ndarray | None:
    poly = box.astype(np.float32)
    area, length = cv2.contourArea(poly), cv2.arcLength(poly, True)
    if length == 0:
        return None
    off = pyclipper.PyclipperOffset()
    off.AddPath([tuple(p) for p in poly.tolist()], pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    out = off.Execute(area * ratio / length)
    return np.array(out[0], np.float32) if out else None


def crop_line(img: np.ndarray, box: np.ndarray, turn: int = 1) -> np.ndarray:
    """Perspective-rectify one quad into a line image. A vertical run of text
    is turned by np.rot90(k=turn); turn=0 leaves it as cut."""
    w = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
    h = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
    w, h = max(w, 1), max(h, 1)
    m = cv2.getPerspectiveTransform(box.astype(np.float32),
                                    np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32))
    out = cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if turn and h / w >= 1.5:  # a vertical run of text: read it rotated
        out = np.ascontiguousarray(np.rot90(out, turn))
    return out


def line_height(boxes: list[np.ndarray]) -> float:
    """Typical text-line height: the median over line-shaped boxes (at least
    twice as wide as tall). Bold text on a dark slide can shed more fragment
    boxes (detached marks, 8-21 px) than it has lines (80-100 px), and a plain
    median over all boxes then measures the fragments."""
    hs = [float(max(np.linalg.norm(b[0] - b[3]), 1.0)) for b in boxes]
    lines = [h for b, h in zip(boxes, hs) if np.linalg.norm(b[0] - b[1]) >= 2 * h]
    return float(np.median(lines if len(lines) >= 3 else hs)) if hs else 1.0


def _flatten(g: np.ndarray) -> np.ndarray:
    """Dark text on white from any line crop: each column's background is the
    median of the crop's top and bottom margin rows, smoothed along the line,
    and text is whatever differs from it, light or dark."""
    h, w = g.shape
    k = max(2, h // 6)
    bg = np.median(np.concatenate([g[:k], g[-k:]], 0), axis=0).astype(np.uint8)
    win = max(3, min(h | 1, 255))
    bg = cv2.medianBlur(np.ascontiguousarray(np.repeat(bg[None, :], 3, 0)), win)[1].astype(np.float32)
    d = np.abs(g.astype(np.float32) - bg[None, :])
    hist = np.cumsum(np.bincount(d.astype(np.uint8).ravel(), minlength=256))
    hi = max(float(np.searchsorted(hist, 0.98 * hist[-1])), 24.0)
    return (255 - np.clip(d * (255.0 / hi), 0, 255)).astype(np.uint8)


def _fragments(boxes: list[np.ndarray], rec: dict) -> set[int]:
    """Boxes that are pieces of a neighbouring line (a stacked mark or a tail
    detected on its own): at most 3 characters, under 35 % of the page's line
    height, and touching a full-height line above or below. Read alone they
    come out as junk ("9/", "9]")."""
    if len(boxes) < 3:
        return set()
    hs = [float(np.linalg.norm(b[0] - b[3])) for b in boxes]
    hmed = line_height(boxes)
    out = set()
    for i, b in enumerate(boxes):
        if hs[i] >= 0.35 * hmed or len(rec.get(i, ("", 0))[0].replace(" ", "")) > 3:
            continue
        x0, x1, y0, y1 = b[:, 0].min(), b[:, 0].max(), b[:, 1].min(), b[:, 1].max()
        for j, c in enumerate(boxes):
            if j == i or hs[j] < 0.6 * hmed:
                continue
            if c[:, 0].min() <= x1 and x0 <= c[:, 0].max() and \
                    min(abs(y0 - c[:, 1].max()), abs(c[:, 1].min() - y1)) <= 0.5 * hmed:
                out.add(i)
                break
    return out


def _vertical(box: np.ndarray, ratio: float) -> int:
    """+1 for a line at least `ratio` times taller than wide, -1 for one that
    much wider than tall, 0 otherwise (box corners as _mini_box orders them)."""
    w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[0] - box[3])
    return 1 if h >= ratio * max(w, 1.0) else -1 if w >= ratio * max(h, 1.0) else 0


def split_gaps(img: np.ndarray, boxes: list[np.ndarray], ratio: float) -> list[np.ndarray]:
    """Cut each line box at blank stretches >= ratio x its height, and trim
    such a stretch off either end (see split_one)."""
    return [p for b in boxes for p in split_one(img, b, ratio)]


def split_one(img: np.ndarray, b: np.ndarray, ratio: float) -> list[np.ndarray]:
    """One line box -> its pieces between blank stretches >= ratio x its height,
    with such a stretch trimmed off either end (the detector can run a line's
    box across an empty gutter up to the next column's first letter)."""
    if not ratio:
        return [b]
    c = crop_line(img, b, turn=0)
    h, w = c.shape[:2]
    if w < 4 * h or h < 4:
        return [b]
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) if c.ndim == 3 else c
    ink = (_flatten(g) < 128).any(axis=0)
    segs, start, blank = [], None, 0
    for x, on in enumerate(ink):
        if on:
            if start is None:
                start = x
            elif blank >= ratio * h:
                segs.append((start, x - blank))
                start = x
            blank = 0
        elif start is not None:
            blank += 1
    if start is not None:
        segs.append((start, w - blank))
    if not segs or (len(segs) == 1 and segs[0][0] < ratio * h and w - segs[0][1] < ratio * h):
        return [b]
    top, bot = b[1] - b[0], b[2] - b[3]
    pad = 0.15 * h
    out = []
    for x0, x1 in segs:
        t0, t1 = max(0.0, (x0 - pad) / w), min(1.0, (x1 + pad) / w)
        out.append(np.array([b[0] + t0 * top, b[0] + t1 * top, b[3] + t1 * bot, b[3] + t0 * bot], np.float32))
    return out


def _words(chars: list | None, box: np.ndarray, text: str, conf: float) -> list[dict]:
    """Words of one line with their boxes, cut from the character positions the
    recognizer reported (space-separated tokens; Thai runs without spaces stay
    one token). Without positions, the whole line is one word.

    CTC emits a character somewhere inside its glyph, so a token's frames stop
    short of its ink on both sides. Each edge is pushed out by half a glyph
    advance (the line's median pitch), but kept clear of the middle of the space
    by a fifth of an advance: the boxes leave a visible gap where the space is,
    as OneOCR's do, and a PDF text layer built from them gets its spaces back
    from that gap. The line's first word starts one advance before its first
    frame, the last ends half an advance after its last."""
    whole = [{"text": text, "confidence": round(conf, 4), "box": np.asarray(box, np.float64)}]
    if not chars:
        return whole
    tokens, cur, gaps = [], [], []  # gaps[i]: centre of the space after token i
    for ch in chars:
        if ch[0].isspace() or ch[0] == "":
            if cur:
                tokens.append(cur)
                gaps.append((ch[1] + ch[2]) / 2)
                cur = []
            elif gaps:
                gaps[-1] = (gaps[-1] + (ch[1] + ch[2]) / 2) / 2  # a run of spaces: its middle
        else:
            cur.append(ch)
    if cur:
        tokens.append(cur)
    adv = _advance(tokens)
    b = np.asarray(box, np.float64)
    out = []
    for n, tok in enumerate(tokens):
        t = fix_scripts(canonical("".join(c[0] for c in tok)))
        if not t:
            continue
        first, last = tok[0][1], tok[-1][2]
        if n > 0:
            x0 = min(first, max(gaps[n - 1] + 0.2 * adv, first - 0.5 * adv))
        else:
            x0 = max(0.0, first - adv)
        if n < len(gaps) and n < len(tokens) - 1:
            x1 = max(last, min(gaps[n] - 0.2 * adv, last + 0.5 * adv))
        else:
            x1 = min(1.0, last + 0.5 * adv)
        quad = np.array([b[0] + x0 * (b[1] - b[0]), b[0] + x1 * (b[1] - b[0]),
                         b[3] + x1 * (b[2] - b[3]), b[3] + x0 * (b[2] - b[3])])
        out.append({"text": t, "confidence": round(float(np.mean([c[3] for c in tok])), 4), "box": quad})
    return out or whole


def _advance(tokens: list) -> float:
    """Median distance between consecutive spacing characters inside words, as a
    fraction of the line width: the glyph pitch. Marks above and below sit on
    their base character and are skipped."""
    steps = []
    for tok in tokens:
        xs = [c[1] for c in tok if c[0] and not unicodedata.combining(c[0]) and unicodedata.category(c[0][0]) != "Mn"]
        steps += [b - a for a, b in zip(xs, xs[1:]) if b > a]
    if steps:
        return float(np.median(steps))
    n = sum(len(t) for t in tokens)
    return (tokens[-1][-1][2] - tokens[0][0][1]) / max(n, 1) if tokens else 0.0


def _rot90_matrix(k: int, shape: tuple[int, int]) -> np.ndarray:
    """3x3 map from an image of `shape` to np.rot90(image, k) coordinates."""
    H, W = shape
    if k == 1:   # counter-clockwise
        return np.array([[0, 1, 0], [-1, 0, W - 1], [0, 0, 1]], np.float64)
    if k == -1:  # clockwise
        return np.array([[0, -1, H - 1], [1, 0, 0], [0, 0, 1]], np.float64)
    return np.array([[-1, 0, W - 1], [0, -1, H - 1], [0, 0, 1]], np.float64)  # 180


def _deskew(img: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate so lines leaning by `angle` degrees run level; canvas grown to
    fit, new area filled with the page's own border colour."""
    H, W = img.shape[:2]
    m = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    c, s_ = abs(m[0, 0]), abs(m[0, 1])
    nW, nH = int(round(H * s_ + W * c)), int(round(H * c + W * s_))
    m[0, 2] += nW / 2 - W / 2
    m[1, 2] += nH / 2 - H / 2
    edge = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]], 0)
    fill = tuple(int(v) for v in np.median(edge, axis=0)) if img.ndim == 3 else int(np.median(edge))
    out = cv2.warpAffine(img, m, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    return out, np.vstack([m, [0, 0, 1]])


def _unrotate(box: np.ndarray, rotation: int, shape: tuple[int, int]) -> np.ndarray:
    """Box corners on the page turned clockwise by `rotation` -> original image."""
    k = {90: -1, 270: 1, 180: 2}.get(rotation)
    return box if k is None else _apply(np.linalg.inv(_rot90_matrix(k, shape)), box)


def _apply(m: np.ndarray, box: np.ndarray) -> np.ndarray:
    pts = np.hstack([np.asarray(box, np.float64), np.ones((len(box), 1))]) @ m.T
    return pts[:, :2]


def page_skew(boxes: list[np.ndarray]) -> float:
    """Median text-line angle in degrees, from boxes at least 3x wider than tall."""
    angles = []
    for b in boxes:
        dx, dy = b[1][0] - b[0][0], b[1][1] - b[0][1]
        h = np.linalg.norm(b[0] - b[3])
        if np.hypot(dx, dy) >= 3 * max(h, 1.0):
            angles.append(math.degrees(math.atan2(dy, dx)))
    return float(np.median(angles)) if angles else 0.0


def reading_order(boxes: list[np.ndarray], angle: float | None = None, columns: bool = True,
                  texts: list[str] | None = None) -> list[list[int]]:
    """Rows top to bottom, each left to right, in DESKEWED coordinates, block by
    block: text columns are read one after the other (see _blocks).

    On a page scanned 1 degree off, a table row's cells sit 20 px apart in y
    across an A4 width — enough for plain y-grouping to put a row's numbers on
    the next row. Box corners are rotated by -skew first, then boxes grouped
    into rows by vertical overlap with the row's running centre. `texts` (the
    recognised line texts, by box) lets the column test reject number columns."""
    if not boxes:
        return []
    a = math.radians(-(page_skew(boxes) if angle is None else angle))
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], np.float32)
    pts = [b.astype(np.float32) @ rot.T for b in boxes]
    ext = [(float(p[:, 0].min()), float(p[:, 0].max()), float(p[:, 1].min()), float(p[:, 1].max())) for p in pts]
    hmed = line_height(boxes)
    idx = list(range(len(boxes)))
    blocks = _blocks(idx, ext, hmed, texts) if columns else [idx]
    cy = [(e[2] + e[3]) / 2 for e in ext]
    rx = [(e[0] + e[1]) / 2 for e in ext]
    hs = [float(max(np.linalg.norm(b[0] - b[3]), 1.0)) for b in boxes]
    tol = 0.5 * hmed
    rows: list[list[int]] = []
    for blk in blocks:
        # Rows from full-height lines only; small boxes (detached marks, a
        # stray symbol) join the nearest row afterwards without moving it.
        # Otherwise a chain of mark fragments between two lines, each within
        # tolerance of the next, merges the two lines into one row.
        full = [i for i in blk if hs[i] >= 0.6 * hmed] or list(blk)
        small = [i for i in blk if i not in set(full)]
        brows: list[list[int]] = []
        for i in sorted(full, key=lambda k: cy[k]):
            if brows and abs(cy[i] - np.mean([cy[j] for j in brows[-1]])) <= tol:
                brows[-1].append(i)
            else:
                brows.append([i])
        centre = [float(np.mean([cy[j] for j in r])) for r in brows]
        for i in small:
            d = [abs(cy[i] - c) for c in centre]
            if d and min(d) <= tol + 0.5 * hs[i]:
                brows[int(np.argmin(d))].append(i)
            else:
                brows.append([i])
                centre.append(cy[i])
        order = sorted(range(len(brows)), key=lambda r: centre[r])
        rows += [sorted(brows[r], key=lambda k: rx[k]) for r in order]
    return rows


def _gaps(iv: list[tuple[float, float]], min_gap: float) -> list[tuple[float, float]]:
    """Empty bands between intervals, at least min_gap wide."""
    iv = sorted(iv)
    out, reach = [], iv[0][1]
    for lo, hi in iv[1:]:
        if lo - reach >= min_gap:
            out.append((reach, lo))
        reach = max(reach, hi)
    return out


def _blocks(idx: list[int], ext: list[tuple], hmed: float, texts: list[str] | None = None,
            depth: int = 0) -> list[list[int]]:
    """Recursive XY-cut into reading-order blocks.

    A vertical cut (a gutter no line crosses) is taken first, but only between
    two columns of PROSE: the column on each side of the gutter needs at least
    12 lines, lines long for their height (median width >= 7 line heights), at
    least half its lines filling the column width, as wrapped text does, and
    (when texts are known) at most 30 % digits. Tables and grids fail: on the
    docbench budget and committee pages that a looser test split, the columns
    had 4-12 lines, ragged item labels (9-21 % full-width lines, against
    54-95 % for prose columns) or digit amounts, so they stay in row order
    with each amount next to its label. A list numbered across the row
    (๑๔๗. left, ๑๔๘. right) is not split either. Otherwise the widest
    horizontal gap of at least 0.8 line heights splits top from bottom (a
    full-width title above the columns)."""
    if len(idx) < 6 or depth > 12:
        return [idx]
    cuts = []
    for g0, g1 in _gaps([(ext[i][0], ext[i][1]) for i in idx], 0.8 * hmed):
        left = [i for i in idx if ext[i][1] <= g0]
        right = [i for i in idx if ext[i][0] >= g1]
        lcol, rcol = _next_to(left, ext, hmed, True), _next_to(right, ext, hmed, False)
        if (_prose(lcol, ext, hmed, texts) and _prose(rcol, ext, hmed, texts)
                and not _numbered_across(lcol, rcol, ext, hmed, texts)):
            cuts.append((g1 - g0, left, right))
    if cuts:
        _, left, right = max(cuts, key=lambda c: c[0])
        return _blocks(left, ext, hmed, texts, depth + 1) + _blocks(right, ext, hmed, texts, depth + 1)
    hg = _gaps([(ext[i][2], ext[i][3]) for i in idx], 0.8 * hmed)
    if not hg:
        return [idx]
    g0, g1 = max(hg, key=lambda g: g[1] - g[0])
    top = [i for i in idx if ext[i][3] <= g0]
    bottom = [i for i in idx if ext[i][2] >= g1]
    return _blocks(top, ext, hmed, texts, depth + 1) + _blocks(bottom, ext, hmed, texts, depth + 1)


def _next_to(side: list[int], ext: list[tuple], hmed: float, gutter_on_right: bool) -> list[int]:
    """The column of `side` that touches the gutter (a side can hold several)."""
    if not side:
        return side
    g = _gaps([(ext[i][0], ext[i][1]) for i in side], 0.8 * hmed)
    if not g:
        return side
    if gutter_on_right:
        return [i for i in side if ext[i][0] >= g[-1][1]]
    return [i for i in side if ext[i][1] <= g[0][0]]


_DIGITS = set("0123456789๐๑๒๓๔๕๖๗๘๙")
# Script fix-ups for look-alikes a line recognizer cannot tell apart without
# context. Neither pattern occurs in real Thai text: a lone Latin W glued to
# Thai letters (Prompt/Kanit Bold พ is a W with a loop: "W.ศ.", "wฤษภาคม"), and
# a Latin O or an Arabic 0 inside a number written in Thai digits ("๗o.๘๒").
_LONE_W = re.compile(r"(?<![A-Za-z])[Ww](?=\.?[\u0E01-\u0E2E\u0E40-\u0E44])")
_ZERO_IN_THAI = re.compile(r"(?<=[๐-๙])[Oo0]|[Oo0](?=[๐-๙])")


def fix_scripts(text: str) -> str:
    text = _LONE_W.sub("พ", text)
    return _ZERO_IN_THAI.sub("๐", text)
_ENUM = re.compile(r"^\s*\(?([0-9๐-๙]{1,4})\s*[.)]")


def _enum(text: str) -> int | None:
    m = _ENUM.match(text)
    return int(m.group(1).translate(str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789"))) if m else None


def _numbered_across(lcol: list[int], rcol: list[int], ext: list[tuple], hmed: float,
                     texts: list[str] | None) -> bool:
    """Most facing lines numbered n (left) and n+1 (right): a list that reads across."""
    if texts is None:
        return False
    pairs = 0
    for i in lcol:
        a = _enum(texts[i])
        if a is None:
            continue
        cy = (ext[i][2] + ext[i][3]) / 2
        if any(_enum(texts[j]) == a + 1 and abs((ext[j][2] + ext[j][3]) / 2 - cy) <= 0.5 * hmed for j in rcol):
            pairs += 1
    return pairs >= 0.5 * min(len(lcol), len(rcol))


def _prose(col: list[int], ext: list[tuple], hmed: float, texts: list[str] | None = None) -> bool:
    if len(col) < 12:
        return False
    widths = [ext[i][1] - ext[i][0] for i in col]
    span = max(ext[i][1] for i in col) - min(ext[i][0] for i in col)
    if float(np.median(widths)) < 7 * hmed or float(np.mean([w >= 0.75 * span for w in widths])) < 0.5:
        return False
    if texts is not None:
        chars = [c for i in col for c in texts[i] if not c.isspace()]
        if chars and sum(c in _DIGITS for c in chars) > 0.3 * len(chars):
            return False
    return True


def _is_thai(ch: str) -> bool:
    return "\u0e00" <= ch <= "\u0e7f"


_MARKS = set("\u0E31\u0E34\u0E35\u0E36\u0E37\u0E38\u0E39\u0E3A\u0E47\u0E48\u0E49\u0E4A\u0E4B\u0E4C\u0E4D\u0E4E")
_LEADING = set("\u0E40\u0E41\u0E42\u0E43\u0E44")


def _cons(ch: str) -> bool:
    return "\u0E01" <= ch <= "\u0E2E"


def invalid_share(text: str) -> float:
    """Share of Thai characters that break Thai spelling: a mark with no consonant
    under it, or a leading vowel (เ แ โ ใ ไ) not followed by a consonant.
    Stamps, seals and signatures read as strings like "ูี้ซ์ ริ ร้ร้ ไ้" score
    high; real text in an unfamiliar font scores near zero."""
    thai = bad = 0
    for i, ch in enumerate(text):
        if not _is_thai(ch):
            continue
        thai += 1
        prev = text[i - 1] if i else " "
        if ch in _MARKS and not (_cons(prev) or (prev in _MARKS and i >= 2 and _cons(text[i - 2]))):
            bad += 1
        elif ch in _LEADING and not (i + 1 < len(text) and _cons(text[i + 1])):
            bad += 1
    return bad / thai if thai else 0.0


class LineOCR:
    def __init__(self, model_dir: str | os.PathLike, threads: int = 1, **overrides: dict):
        d = Path(model_dir)
        self.cfg = json.loads(json.dumps(DEFAULTS))
        det_pp = _yaml_postprocess(d / "det.yml") if (d / "det.yml").exists() else {}
        for src, dst in (("thresh", "thresh"), ("box_thresh", "box_thresh"), ("unclip_ratio", "unclip")):
            if isinstance(det_pp.get(src), float):
                self.cfg["det"][dst] = det_pp[src]
        user = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
        for section in (user, overrides):
            for k, v in section.items():
                self.cfg.setdefault(k, {}).update(v)
        # CTC classes: 0 = blank, then the dictionary, then the space PaddleOCR
        # appends. Our trained models ship rec_dict.txt; stock ones rec.yml.
        if (d / "rec_dict.txt").exists():
            chars = [c for c in (d / "rec_dict.txt").read_text(encoding="utf-8").split("\n") if c]
        else:
            chars = _yaml_postprocess(d / "rec.yml")["character_dict"]
        self.chars = [""] + chars + [" "]
        # OpenCV's pool defaults to every core: each crop/resize would fan out
        # over the whole box, which oversubscribes it as soon as pages run in
        # parallel. Match the engine's own thread budget.
        cv2.setNumThreads(threads)
        rt = self.cfg["runtime"]
        backend = rt["backend"]
        if backend == "openvino":
            try:
                import openvino  # noqa: F401
            except ImportError:
                backend = "onnxruntime"
        self.backend = backend
        self.det = _Model(d / "det.onnx", threads, backend, rt["precision"])
        self.rec = _Model(d / "rec.onnx", threads, backend, rt["precision"])

    # -- detection -----------------------------------------------------------
    def detect(self, img: np.ndarray) -> list[np.ndarray]:
        c = self.cfg["det"]
        H0, W0 = img.shape[:2]
        # A white border so text touching the edge still gets a closed region,
        # then the long side scaled to the detector's working size in BOTH
        # directions (as PaddleOCR's resize_long does): a 200x50 crop of one
        # word is otherwise too small for the DB head to find anything.
        pad = max(8, int(0.05 * min(H0, W0)))
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        h, w = img.shape[:2]
        s = min(c["limit"] / max(h, w), 4.0) if c["limit"] else 1.0
        rh, rw = max(32, int(round(h * s / 32)) * 32), max(32, int(round(w * s / 32)) * 32)
        x = cv2.resize(img, (rw, rh)).astype(np.float32) / 255.0
        x = ((x - DET_MEAN) / DET_STD).transpose(2, 0, 1)[None]
        prob = self._run(self.det, x)[0, 0]
        bitmap = (prob > c["thresh"]).astype(np.uint8)
        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        sx, sy = w / prob.shape[1], h / prob.shape[0]
        boxes = []
        for cnt in contours[: int(c["max_candidates"])]:
            box, side = _mini_box(cnt)
            if side < c["min_size"] or _box_score(prob, box) < c["box_thresh"]:
                continue
            big = _unclip(box, c["unclip"])
            if big is None:
                continue
            box, side = _mini_box(big.reshape(-1, 1, 2))
            if side < c["min_size"] + 2:
                continue
            box[:, 0] = np.clip(box[:, 0] * sx - pad, 0, W0 - 1)
            box[:, 1] = np.clip(box[:, 1] * sy - pad, 0, H0 - 1)
            if c["v_pad"]:  # Thai marks above/below the core band: grow height only
                hh = max(np.linalg.norm(box[0] - box[3]), 1.0) * c["v_pad"] / 2
                box[[0, 1], 1] = np.clip(box[[0, 1], 1] - hh, 0, H0 - 1)
                box[[2, 3], 1] = np.clip(box[[2, 3], 1] + hh, 0, H0 - 1)
            if box[:, 0].max() - box[:, 0].min() >= 2 and box[:, 1].max() - box[:, 1].min() >= 2:
                boxes.append(box)
        return boxes

    # -- recognition ---------------------------------------------------------
    def prep_crop(self, crop: np.ndarray) -> np.ndarray:
        mode = self.cfg["rec"]["color"]
        if mode == "bgr" or crop.ndim != 3 or not crop.size:
            return crop
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if mode == "gray":
            return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        if mode == "flatten":
            return cv2.cvtColor(_flatten(g), cv2.COLOR_GRAY2BGR)
        # Levels from a 256-bin histogram: linear time (np.percentile sorts
        # every crop and cost ~8% of page time).
        c = np.cumsum(np.bincount(g.ravel(), minlength=256))
        lo, med, hi = (int(np.searchsorted(c, q * c[-1])) for q in (0.02, 0.5, 0.98))
        lut = np.arange(256, dtype=np.float32)
        if mode == "stretch" and hi - lo >= 16:
            lut = np.clip((lut - lo) * (255.0 / (hi - lo)), 0, 255)
        if lut[med] < 128:  # the background dominates a line crop: dark background = light text
            lut = 255 - lut
        return cv2.cvtColor(cv2.LUT(g, lut.astype(np.uint8)), cv2.COLOR_GRAY2BGR)

    def recognize(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        return [(t, c) for t, c, _ in self.recognize_detail(crops)]

    def recognize_detail(self, crops: list[np.ndarray]) -> list[tuple[str, float, list]]:
        """(text, confidence, chars) per crop; chars are (char, x0, x1, prob)
        with x0..x1 the share of the crop's width the character was read from
        (the CTC frames of its run). Word boxes are cut from these."""
        c = self.cfg["rec"]
        if c["color"] == "flatten":  # cheaper at the recognizer's height
            crops = [self.prep_crop(cv2.resize(cr, (max(1, int(math.ceil(int(c["height"]) * cr.shape[1] / max(cr.shape[0], 1)))),
                                                    int(c["height"])))) for cr in crops]
        else:
            crops = [self.prep_crop(cr) for cr in crops]
        H, max_w, bs = int(c["height"]), int(c["max_w"]), int(c["batch"])
        ratios = [cr.shape[1] / max(cr.shape[0], 1) for cr in crops]
        out: list[tuple[str, float, list]] = [("", 0.0, [])] * len(crops)
        order = sorted(range(len(crops)), key=lambda i: ratios[i])
        for b in range(0, len(order), bs):
            idx = order[b:b + bs]
            W = min(max_w, max(16, int(math.ceil(H * max(ratios[i] for i in idx)))))
            batch = np.zeros((len(idx), 3, H, W), np.float32)  # 0 = mid-grey after normalising
            widths = []
            for n, i in enumerate(idx):
                rw = min(W, max(1, int(math.ceil(H * ratios[i]))))
                widths.append(rw)
                im = cv2.resize(crops[i], (rw, H)).astype(np.float32) / 255.0
                batch[n, :, :, :rw] = ((im - 0.5) / 0.5).transpose(2, 0, 1)
            probs = self._run(self.rec, batch)
            for n, i in enumerate(idx):
                out[i] = self._ctc_detail(probs[n], W / probs[n].shape[0], widths[n])
        return out

    @staticmethod
    def _run(model: _Model, x: np.ndarray) -> np.ndarray:
        return model.run(x)

    def _ctc_detail(self, p: np.ndarray, stride: float, width: int) -> tuple[str, float, list]:
        ids, conf = p.argmax(1), p.max(1)
        chars, probs, t, T = [], [], 0, len(ids)
        while t < T:
            end = t + 1
            while end < T and ids[end] == ids[t]:
                end += 1
            if ids[t] != 0:
                ch = self.chars[ids[t]] if ids[t] < len(self.chars) else ""
                chars.append((ch, min(1.0, t * stride / width), min(1.0, end * stride / width), float(conf[t])))
                probs.append(float(conf[t]))
            t = end
        return "".join(c[0] for c in chars), (float(np.mean(probs)) if probs else 0.0), chars

    def _ctc(self, p: np.ndarray) -> tuple[str, float]:
        ids, conf = p.argmax(1), p.max(1)
        keep = np.ones_like(ids, bool)
        keep[1:] = ids[1:] != ids[:-1]
        keep &= ids != 0
        chars = [self.chars[i] if i < len(self.chars) else "" for i in ids[keep]]
        return "".join(chars), float(conf[keep].mean()) if keep.any() else 0.0

    # -- page ----------------------------------------------------------------
    def read(self, image: Any) -> dict:
        img = self._load(image)
        shape = img.shape[:2]
        t0 = time.perf_counter()
        boxes = self.detect(img)
        h, w = shape
        if not boxes and h <= 200 and w >= 1.5 * h:
            # Nothing found in a small, line-shaped image: callers send single
            # line/field crops too — read the whole image as one line.
            boxes = [np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)]
        oc, rotation, skew = self.cfg["ori"], 0, 0.0
        tf = np.eye(3)  # original image -> working image
        if oc["enabled"]:
            k = self._sideways(img, boxes)
            if k:  # np.rot90 k: +1 turns the page counter-clockwise, -1 clockwise
                tf = _rot90_matrix(k, img.shape[:2]) @ tf
                img = np.ascontiguousarray(np.rot90(img, k))
                boxes, rotation = self.detect(img), 90 if k == -1 else 270
            skew = page_skew(boxes)
            if abs(skew) >= oc["deskew_above"]:
                img, m = _deskew(img, skew)
                tf = m @ tf
                boxes = self.detect(img)
        t1 = time.perf_counter()
        page = self._page(img, boxes)
        if oc["enabled"] and not rotation and self._upside_down(img, page["boxes"], page):
            tf = _rot90_matrix(2, img.shape[:2]) @ tf
            img = np.ascontiguousarray(img[::-1, ::-1])
            boxes, rotation = self.detect(img), 180
            page = self._page(img, boxes)
        boxes = page["boxes"]
        back = np.linalg.inv(tf)
        t2 = time.perf_counter()
        rc = self.cfg["rec"]
        drop, min_chars, junk_below, junk_invalid = rc["drop_below"], rc["min_chars"], rc["junk_below"], rc["junk_invalid"]
        lines, row_texts = [], []
        frag = _fragments(boxes, page["rec"])
        # What the filter threw away, and why: a hybrid caller decides from this
        # whether the page is one this model reads badly (handwriting, scene
        # text, stylised display type) rather than one with a stamp or a
        # signature on it. Fragments of marks are not counted.
        dropped = []
        for ri, r in enumerate(page["rows"]):
            parts = []
            for i in r:
                text, conf = page["rec"][i]
                text = fix_scripts(canonical(text))
                reason = None
                if i in frag:
                    continue
                if not text:
                    reason = "empty"
                elif conf < drop:
                    reason = "low_confidence"
                elif len(text.replace(" ", "")) < min_chars:
                    reason = "too_short"
                elif conf < junk_below and invalid_share(text) > junk_invalid:
                    reason = "invalid_thai"
                box = _apply(back, boxes[i])
                entry = {"text": text, "confidence": round(conf, 4), "row": ri,
                         "box": [[round(float(x), 1), round(float(y), 1)] for x, y in box]}
                if reason:
                    dropped.append({**entry, "reason": reason})
                    continue
                parts.append(text)
                entry["words"] = [{**w, "box": [[round(float(x), 1), round(float(y), 1)] for x, y in _apply(back, w["box"])]}
                                  for w in _words(page["chars"].get(i), boxes[i], text, conf)]
                lines.append(entry)
            if parts:
                row_texts.append(" ".join(parts))
        straightened = skew if abs(skew) >= oc["deskew_above"] else 0.0
        return {"text": self._join(row_texts), "lines": lines, "dropped": dropped,
                "text_angle": round(page["angle"] + straightened, 2),
                "rotation": rotation,
                "timing": {"det_ms": round((t1 - t0) * 1000, 1), "rec_ms": round((t2 - t1) * 1000, 1),
                           "n_lines": len(page["rec"])}}

    def _page(self, img: np.ndarray, boxes: list[np.ndarray]) -> dict:
        """Recognition, then reading order (the column test looks at the text).

        Each detected line is first cut at wide blank stretches, or trimmed of
        a blank end (split_one). The pieces replace the line only if every
        piece reads well enough to survive the line filter AND together they
        read at least 90 % of the characters the whole line does; otherwise
        the line is read whole. Lines merged across a column gutter split into
        long, confident pieces; a widely spaced header ("ตอนที่ ๕๓   ง") would
        lose its short pieces (๕๓ at 0.79, ง as "oZ"), and on a dark-to-light
        gradient the ink test can take real text for a blank end."""
        groups = [split_one(img, b, self.cfg["det"]["split_gap"]) for b in boxes]
        changed = [len(g) > 1 or g[0] is not b for b, g in zip(boxes, groups)]
        pieces = [p for g in groups for p in g]
        whole = [b for b, ch in zip(boxes, changed) if ch]
        res = self._read_boxes(img, pieces + whole)
        final, out, k, e = [], [], 0, len(pieces)
        for b, g, ch in zip(boxes, groups, changed):
            r = res[k:k + len(g)]
            k += len(g)
            if ch:
                alt = res[e]
                e += 1
                n_pieces = sum(len(x[0].replace(" ", "")) for x in r)
                if (min(x[1] for x in r) < self.cfg["rec"]["drop_below"]
                        or n_pieces < 0.9 * len(alt[0].replace(" ", ""))):
                    final.append(b)
                    out.append(alt)
                    continue
            final += g
            out += r
        angle = page_skew(final)
        rows = reading_order(final, angle, texts=[x[0] for x in out])
        return {"rows": rows, "angle": angle, "rec": {i: (x[0], x[1]) for i, x in enumerate(out)},
                "chars": {i: x[2] for i, x in enumerate(out)}, "boxes": final}

    def _read_boxes(self, img: np.ndarray, boxes: list[np.ndarray]) -> list[tuple[str, float, list | None]]:
        """Recognise every box. A box taller than wide is either a vertical run
        of text (a turned table header, a margin note) or just one tall
        character (a consonant under stacked marks), so it is read as cut and
        turned both ways, and the most confident reading kept."""
        raw = [crop_line(img, b, turn=0) for b in boxes]
        tall = [n for n, c in enumerate(raw) if c.shape[0] >= 1.5 * c.shape[1]]
        res = self.recognize_detail(raw + [np.ascontiguousarray(np.rot90(raw[n], k)) for k in (1, -1) for n in tall])
        out = res[:len(raw)]
        for j, n in enumerate(tall):
            for alt in (res[len(raw) + j], res[len(raw) + len(tall) + j]):
                if alt[1] > out[n][1]:
                    out[n] = (alt[0], alt[1], None)  # read turned: no word positions along the box
        return out

    def _sideways(self, img: np.ndarray, boxes: list[np.ndarray]) -> int:
        """np.rot90 k that turns a sideways page upright, 0 if it is not
        sideways: most text lines run vertically. The direction is whichever
        way the longest vertical lines read with more confidence."""
        votes = [_vertical(b, 2.0) for b in boxes]
        vert = [b for b, v in zip(boxes, votes) if v == 1]
        if len(vert) < 3 or len(vert) <= votes.count(-1):
            return 0
        vert.sort(key=lambda b: -np.linalg.norm(b[0] - b[3]))
        raw = [crop_line(img, b, turn=0) for b in vert[: self.cfg["ori"]["sample"]]]
        res = self.recognize([np.ascontiguousarray(np.rot90(c, k)) for k in (1, -1) for c in raw])
        ccw = sum(c for _, c in res[:len(raw)])
        cw = sum(c for _, c in res[len(raw):])
        return 1 if ccw > cw else -1

    def _upside_down(self, img: np.ndarray, boxes: list[np.ndarray], page: dict) -> bool:
        """Only for a page that reads badly: do its longest lines read better
        turned 180 degrees?"""
        oc = self.cfg["ori"]
        confs = [c for t, c in page["rec"].values() if len(t) >= 2]
        if len(confs) < 3 or float(np.median(confs)) >= oc["flip_below"]:
            return False
        idx = sorted((i for i in page["rec"] if _vertical(boxes[i], 2.0) == -1),
                     key=lambda i: -np.linalg.norm(boxes[i][0] - boxes[i][1]))[: oc["sample"]]
        if len(idx) < 3:
            return False
        up = float(np.mean([page["rec"][i][1] for i in idx]))
        flipped = self.recognize([np.ascontiguousarray(crop_line(img, boxes[i], turn=0)[::-1, ::-1]) for i in idx])
        return float(np.mean([c for _, c in flipped])) > up + oc["flip_margin"]

    def _join(self, rows: list[str]) -> str:
        if not self.cfg["join"]["thai_no_space"]:
            return "\n".join(rows)
        # A Thai line wrap has no space at the break (word processors wrap
        # Thai between words without one), so Thai-Thai joins get nothing.
        out = rows[0] if rows else ""
        for r in rows[1:]:
            out += ("" if out and _is_thai(out[-1]) and _is_thai(r[0]) else "\n") + r
        return out

    @staticmethod
    def _load(image: Any) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if isinstance(image, (str, os.PathLike)):
            img = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"cannot read image {image}")
            return img
        if isinstance(image, (bytes, bytearray)):
            img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cannot decode image bytes")
            return img
        return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)  # PIL


# -- service engines -----------------------------------------------------------
# Same duck-typed API as engine.OcrEngine (ready / transcribe(png) / close), so
# main.process() drives any of them unchanged.

_WORKER: "LineOCR | None" = None


def _worker_init(model_dir: str, threads: int) -> None:
    global _WORKER
    _WORKER = LineOCR(model_dir, threads=threads)


def _worker_read(png: bytes) -> dict:
    return _WORKER.read(png)


class LineEngine:
    """LineOCR behind the service's async engine API.

    Pages run in a pool of worker PROCESSES, one LineOCR each. Threads cannot
    scale: detection post-processing, reading order and the line filter are
    Python and hold the GIL, so 8 threads on 8 cores kept only ~2 busy
    (350 pages/min against ~1,300 for 8 processes)."""

    def __init__(self, model_dir: str, workers: int = 2, threads: int = 1):
        import asyncio
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        self.url, self.model = f"line:{model_dir}", Path(model_dir).name
        # spawn, not fork: the server process already runs threads (event loop,
        # executors) and forking those is unsafe.
        self.pool = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"),
                                        initializer=_worker_init, initargs=(str(model_dir), threads))
        self.sem = asyncio.Semaphore(workers)

    async def ready(self) -> bool:
        return True

    async def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)

    async def transcribe(self, png: bytes) -> dict:
        import asyncio

        async with self.sem:
            t0 = time.perf_counter()
            r = await asyncio.get_running_loop().run_in_executor(self.pool, _worker_read, png)
        lines = r["lines"]
        chars = sum(len(l["text"]) for l in lines)
        conf = sum(l["confidence"] * len(l["text"]) for l in lines) / chars if chars else 0.0
        return {"text": r["text"], "truncated": False, "looped": False, "engine": "line",
                "confidence": round(conf, 4), "lines": lines,
                "ms": round((time.perf_counter() - t0) * 1000)}


def downscale_png(png: bytes, max_pixels: int) -> bytes:
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    if w * h <= max_pixels:
        return png
    s = (max_pixels / (w * h)) ** 0.5
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].tobytes()


class HybridEngine:
    """Line engine first; a page whose confidence falls below min_conf is re-read
    by the VLM — capped at max_share of pages, since one VLM page costs as much
    CPU/GPU as dozens of line pages. Over the cap the line text is kept and the
    page is flagged low_confidence rather than silently degrading throughput."""

    def __init__(self, line: LineEngine, vlm, min_conf: float = 0.85, max_share: float = 0.05,
                 vlm_max_pixels: int = 896 * 32 * 32):
        self.line, self.vlm = line, vlm
        self.min_conf, self.max_share, self.vlm_max_pixels = min_conf, max_share, vlm_max_pixels
        self.url, self.model = f"hybrid({line.url} | {vlm.url})", f"{line.model}+{vlm.model}"
        self.pages = self.fallbacks = 0

    async def ready(self) -> bool:
        return await self.line.ready() and await self.vlm.ready()

    async def close(self) -> None:
        await self.line.close()
        await self.vlm.close()

    async def transcribe(self, png: bytes) -> dict:
        r = await self.line.transcribe(png)
        self.pages += 1
        if r["confidence"] >= self.min_conf or not r["text"] and not r["lines"]:
            return r
        if self.fallbacks + 1 > self.max_share * max(self.pages, 20):
            return {**r, "low_confidence": True}
        self.fallbacks += 1
        v = await self.vlm.transcribe(downscale_png(png, self.vlm_max_pixels))
        return {**v, "engine": "vlm-fallback", "line_confidence": r["confidence"]}

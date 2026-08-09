"""plot.py — render matplotlib plot from agent-supplied code, save to PNG, open in viewer

agent เขียน plotting code โดยใช้ `plt` (matplotlib.pyplot) และ `pd` (pandas)
ที่ pre-imported ให้แล้ว tool จัดการ:
  - Agg backend (headless ปลอดภัย)
  - cwd = workspace
  - savefig ไปที่ output_name ใน workspace
  - auto-open ผ่าน `open <png>` (macOS)

agent ไม่ต้อง import matplotlib เอง ไม่ต้องเรียก savefig — มี last line เป็น figure expression
หรือเรียก plt.* ตามปกติแล้ว tool savefig ให้
"""
from __future__ import annotations
import os
import sys
import subprocess
import uuid
from pathlib import Path
from langchain_core.tools import tool

_TIMEOUT = 60

_PREAMBLE = """\
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import pandas as pd
import numpy as np
import io as _io
import re as _re
import tempfile as _tempfile
import os as _os2
from matplotlib.colors import to_rgba as _to_rgba
from PIL import Image as _PILImage

# Thai font — ใช้ Thonburi สำหรับ layout/spacing ที่ถูกต้อง
# (CoreText จะ render ทับทีหลัง ดังนั้น font นี้จะถูกซ่อน)
_THAI_FONTS = ["Thonburi", "Noto Sans Thai", "Garuda", "Norasi"]
_avail = {f.name for f in _fm.fontManager.ttflist}
_thai_font = next((f for f in _THAI_FONTS if f in _avail), None)
if _thai_font:
    matplotlib.rcParams["font.family"] = _thai_font
matplotlib.rcParams["axes.unicode_minus"] = False

_THAI_RE = _re.compile(r"[\\u0e00-\\u0e7f]")
_CT_FONT = _thai_font or "Thonburi"


def _ct_render(text, fontsize_pt, color_rgba, scale, oversample=3):
    \"\"\"render Thai text via CoreText (macOS) → PIL RGBA Image
    oversample: render ที่ N× แล้ว resize ลง → combining marks ชัดแม้ font เล็ก\"\"\"
    try:
        import Quartz, CoreText
        from AppKit import (NSFont, NSAttributedString, NSColor,
                            NSFontAttributeName, NSForegroundColorAttributeName)

        px = max(int(fontsize_pt * scale * oversample), 12)
        font = NSFont.fontWithName_size_(_CT_FONT, px) or NSFont.systemFontOfSize_(px)
        r, g, b, a = color_rgba
        ns_col = NSColor.colorWithRed_green_blue_alpha_(r, g, b, a)
        attr_str = NSAttributedString.alloc().initWithString_attributes_(
            text, {NSFontAttributeName: font, NSForegroundColorAttributeName: ns_col}
        )
        line = CoreText.CTLineCreateWithAttributedString(attr_str)
        bounds = CoreText.CTLineGetBoundsWithOptions(line, 0)

        pad = 4
        tw = max(int(bounds.size.width) + pad * 2, 1)
        th = max(int(bounds.size.height) + pad * 2, 1)

        ctx = Quartz.CGBitmapContextCreate(
            None, tw, th, 8, tw * 4,
            Quartz.CGColorSpaceCreateDeviceRGB(),
            Quartz.kCGBitmapByteOrder32Host | Quartz.kCGImageAlphaPremultipliedFirst,
        )
        Quartz.CGContextClearRect(ctx, Quartz.CGRectMake(0, 0, tw, th))
        Quartz.CGContextSetTextMatrix(ctx, Quartz.CGAffineTransformIdentity)
        Quartz.CGContextTranslateCTM(ctx, pad - bounds.origin.x, pad - bounds.origin.y)
        CoreText.CTLineDraw(line, ctx)

        img_ref = Quartz.CGBitmapContextCreateImage(ctx)
        tmp = _tempfile.mktemp(suffix=".png")
        url = Quartz.CFURLCreateWithFileSystemPath(None, tmp, Quartz.kCFURLPOSIXPathStyle, False)
        dest = Quartz.CGImageDestinationCreateWithURL(url, "public.png", 1, None)
        Quartz.CGImageDestinationAddImage(dest, img_ref, None)
        Quartz.CGImageDestinationFinalize(dest)
        pil_big = _PILImage.open(tmp).convert("RGBA")
        try: _os2.unlink(tmp)
        except: pass
        # resize ลงจาก oversample size → target size (LANCZOS anti-alias)
        tw_out = max(pil_big.width // oversample, 1)
        th_out = max(pil_big.height // oversample, 1)
        pil = pil_big.resize((tw_out, th_out), _PILImage.LANCZOS)
        return pil
    except Exception:
        return None


def _all_thai_texts(fig):
    items = list(fig.texts)
    for ax in fig.axes:
        items += [ax.title, ax.xaxis.label, ax.yaxis.label]
        items += list(ax.get_xticklabels()) + list(ax.get_yticklabels())
    return [t for t in items if t and t.get_visible() and _THAI_RE.search(t.get_text() or "")]


def _thai_save(fig, out_path, dpi=90):
    \"\"\"save figure with Thai text rendered via CoreText — no floating vowels\"\"\"
    thai = _all_thai_texts(fig)
    if not thai:
        fig.savefig(out_path, dpi=dpi)
        return

    try: fig.tight_layout()
    except: pass
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    CDPI = fig.dpi
    scale = dpi / CDPI
    FH = fig.get_size_inches()[1] * CDPI  # figure height in canvas pixels

    records = []
    for t in thai:
        try:
            bb = t.get_window_extent(renderer=renderer)
            try: color = _to_rgba(t.get_color())
            except: color = (0, 0, 0, 1)
            records.append({
                "text": t.get_text(), "bb": bb,
                "fs": t.get_fontsize(), "color": color,
                "rot": t.get_rotation(),
            })
            t.set_color("none")
        except: pass

    fig.canvas.draw()
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img = _PILImage.open(buf).convert("RGBA")
    W, H = img.size

    for r in records:
        txt_img = _ct_render(r["text"], r["fs"], r["color"], scale)
        if txt_img is None:
            continue

        rot = r["rot"]
        if abs(rot - 90) < 10:
            txt_img = txt_img.rotate(90, expand=True)
        elif abs(rot - 270) < 10 or abs(rot + 90) < 10:
            txt_img = txt_img.rotate(-90, expand=True)

        tw, th = txt_img.size
        bb = r["bb"]

        # x: center on bb horizontal center
        cx = int((bb.x0 + bb.x1) / 2 * scale)
        x = max(cx - tw // 2, 0)

        # y: align CoreText image BOTTOM to bb.y0 (bottom of text in matplotlib)
        # matplotlib y: 0=bottom → PIL y: 0=top
        # PIL bottom of bb = (FH - bb.y0) * scale
        pil_bottom = int((FH - bb.y0) * scale)
        y = max(pil_bottom - th, 0)

        img.paste(txt_img, (x, y), txt_img)

    img.convert("RGB").save(out_path, dpi=(dpi, dpi))
"""

_EPILOGUE = """
_thai_save(plt.gcf(), {out!r}, dpi=150)
plt.close("all")
import os as _os
print("saved:", {out!r}, _os.path.getsize({out!r}), "bytes")
"""


@tool
def plot(code: str, output_name: str = "plot.png") -> str:
    """Create a chart with matplotlib. Provide Python code that builds the figure using `plt` (matplotlib.pyplot), `pd` (pandas), `np` (numpy) — all are pre-imported. Do NOT call savefig yourself; the tool saves and opens it.
    Example code: df = pd.read_csv('data.csv'); df.groupby('cat')['val'].sum().plot.bar(); plt.title('Sales by Cat')
    Returns: saved file path and size, or error message."""
    from config import WORKSPACE
    if not code or not code.strip():
        return "[error] code is required"
    if not output_name.endswith(".png"):
        output_name += ".png"
    out_path = str(Path(WORKSPACE) / output_name)
    full_code = _PREAMBLE + "\n" + code + _EPILOGUE.format(out=out_path)
    Path(WORKSPACE).mkdir(parents=True, exist_ok=True)
    script = Path(WORKSPACE) / f"._plot_{uuid.uuid4().hex[:8]}.py"
    try:
        script.write_text(full_code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True, text=True,
                timeout=_TIMEOUT, cwd=WORKSPACE,
            )
        finally:
            try:
                script.unlink()
            except Exception:
                pass
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            tail = "\n".join(err.splitlines()[-6:])
            return f"[error] plot failed:\n{tail}"
        out = (proc.stdout or "").strip()
        # auto-open (best-effort; fail silently if no GUI)
        if os.path.exists(out_path):
            try:
                subprocess.Popen(["open", out_path],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
        return f"{out}\n[opened in viewer]" if out else f"saved: {out_path}"
    except subprocess.TimeoutExpired:
        return f"[error] plot timed out after {_TIMEOUT}s"
    except Exception as e:
        return f"[error] plot failed: {e}"

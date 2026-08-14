# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""python_exec.py — run Python code in the same env as endeavor_agent.py

ทำไมไม่ใช้ bash + python3 -c:
- shell escaping ของ pandas/matplotlib code ฝันร้าย (nested quotes, df.query, regex)
- python3 ระบบไม่มี pandas/matplotlib — ต้องใช้ sys.executable (env mlx)
- pandas group/agg อาจใช้เวลา → timeout สูงกว่า bash

scope: data analysis + matplotlib plots. agent เขียน code เอง (35B ทำได้)
"""
from __future__ import annotations
import os
import re
import sys
import subprocess
import tempfile
import uuid
from pathlib import Path
from langchain_core.tools import tool
from tools._progress import progress as _progress
from tools.bash import _build_sandbox_profile
from tools._truncate import truncate_with_save

_TIMEOUT_DEFAULT = 120  # data analysis ใช้เวลานานกว่า bash
_MAX_CHARS_DEFAULT = 10_000

# skills/ คือจุดเดียวที่ python_exec เขียน legitimate นอก workspace (/build skill เขียน skills/<name>.md)
_SKILLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "skills")


def _classify_error(stderr: str) -> str | None:
    """Match the last traceback line against known exception types and return a short
    (<=200 char) recovery hint, or None when nothing matches — a wrong hint is worse than
    no hint, so each branch is gated on evidence specific to that exception, not just the
    exception name (e.g. KeyError only gets a pandas hint when pandas actually raised it)."""
    if not stderr or not stderr.strip():
        return None
    last_line = stderr.strip().splitlines()[-1]

    m = re.search(r"ModuleNotFoundError: No module named '([^']+)'", last_line)
    if m:
        return (f"'{m.group(1)}' is not installed; available libs: pandas/numpy/matplotlib/"
                f"scipy/sklearn/statsmodels. Do not pip install — retry with an available "
                f"library instead.")
    if "FileNotFoundError" in last_line:
        return "cwd is workspace/ — call bash (for example: rg --files or find) to confirm the exact filename, then retry."
    if "UnicodeDecodeError" in last_line:
        return "retry the read with encoding='utf-8-sig' or encoding='latin-1'."
    if "KeyError" in last_line and "pandas" in stderr:
        return "likely a wrong/misspelled column or index label — print(df.columns.tolist()) first, then retry."
    if "SyntaxError" in last_line:
        return "resend the complete corrected code from scratch — check quotes/indentation, don't patch a snippet."
    if "ParserError" in last_line and "pandas" in stderr:
        return "malformed CSV (ragged rows / wrong delimiter) — retry with on_bad_lines='skip' or check sep=."
    if "MemoryError" in last_line:
        return "data too large for memory — retry with usecols=[...] to load fewer columns, or chunksize= to stream it."
    if "AttributeError" in last_line and ("DataFrame" in stderr or "Series" in stderr):
        return "likely called a method that doesn't exist on this object — print(type(x)) and check df.dtypes/df.columns first."
    return None


def _python_exec_impl(code: str, timeout: int = _TIMEOUT_DEFAULT, max_chars: int = _MAX_CHARS_DEFAULT) -> str:
    from config import WORKSPACE
    if not code or not code.strip():
        return "[error] code is required"
    script = None
    profile_path = None
    try:
        Path(WORKSPACE).mkdir(parents=True, exist_ok=True)
        script = Path(WORKSPACE) / f"._exec_{uuid.uuid4().hex[:8]}.py"
        # OS-level sandbox เหมือน bash — เขียนได้เฉพาะ workspace + /tmp + skills/ (audit P1)
        profile = _build_sandbox_profile(WORKSPACE, extra_write_paths=(_SKILLS_DIR,))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sb", delete=False) as pf:
            pf.write(profile)
            profile_path = pf.name
        _progress("running code…")
        script.write_text(code, encoding="utf-8")
        try:
            try:
                proc = subprocess.run(
                    # -u: unbuffered stdout — without it, Python block-buffers stdout (~8KB)
                    # when writing to a pipe (not a tty), so a script killed mid-run on
                    # timeout has its already-printed output still sitting in the child's
                    # buffer, never flushed to us. Partial-output-on-timeout below is a
                    # no-op without this flag.
                    ["sandbox-exec", "-f", profile_path, sys.executable, "-u", str(script)],
                    capture_output=True, text=True,
                    timeout=timeout, cwd=WORKSPACE, stdin=subprocess.DEVNULL,
                )
            except subprocess.TimeoutExpired as e:
                # e.stdout/e.stderr come back as bytes here even with text=True (subprocess
                # quirk on TimeoutExpired specifically — see tools/bash.py's identical guard)
                # — decode defensively.
                def _decode(x) -> str:
                    if x is None:
                        return ""
                    return x.decode("utf-8", errors="replace") if isinstance(x, bytes) else x
                partial = _decode(e.stdout)
                if e.stderr:
                    partial += f"\n[stderr]\n{_decode(e.stderr)}"
                partial = partial.strip()
                note = f"[error] timed out after {timeout}s"
                if partial:
                    note += " — partial output before timeout:"
                    partial = truncate_with_save(partial, max_chars, WORKSPACE, "python_exec",
                                                  file_prefix="_exec_output",
                                                  marker_first=True, keep_tail=True)
                    return f"{note}\n{partial}"
                return note
        finally:
            try:
                script.unlink()
            except Exception:
                pass
            try:
                os.unlink(profile_path)
            except Exception:
                pass
        out = proc.stdout or ""
        if proc.stderr:
            out += f"\n[stderr]\n{proc.stderr}"
            hint = _classify_error(proc.stderr)
            if hint:
                out += f"\n[hint] {hint}"
        if proc.returncode != 0 and not out.strip():
            return (f"[error] exited {proc.returncode}, no output — add print() statements "
                     f"before the crash point and re-run to see where it failed")
        out = out.strip() or "(no output)"
        # keep_tail ONLY on a failed run: a crash traceback + [hint] sit at the END of
        # stdout+stderr, so a head-only cut would hide exactly the part that explains the
        # failure. A clean successful run stays head-only — the docstring's truncation
        # section (and its BAD/GOOD example) documents head-only behavior specifically for
        # that case, and this must not silently diverge from what it teaches.
        out = truncate_with_save(
            out, max_chars, WORKSPACE, "python_exec",
            file_prefix="_exec_output",
            keep_tail=bool(proc.returncode != 0 or proc.stderr),
            extra_note=(
                "Computations already run before this point (sums, means, groupby, describe) "
                "used the COMPLETE data regardless of this cap — only what's printed below the "
                "cap is affected."
            ),
        )
        return out
    except FileNotFoundError:
        return "[error] sandbox-exec not found — macOS only"
    except Exception as e:
        return f"[error] python_exec failed: {e}"


@tool
def python_exec(code: str, timeout: int = _TIMEOUT_DEFAULT, max_chars: int = _MAX_CHARS_DEFAULT) -> str:
    """Run Python code (cwd = workspace) using the SAME interpreter that runs the agent.

    CAPABILITIES:
      - Data analysis: pandas — read CSV/Excel/JSON, filter/groupby/aggregate, .describe()/.corr()
      - Statistics: scipy.stats — hypothesis tests (ttest_ind, pearsonr, chi2_contingency, ANOVA)
      - Regression / time-series: statsmodels.api — sm.OLS(y, sm.add_constant(x)).fit().summary(), ARIMA
      - Machine learning: sklearn — regression, clustering, PCA, train_test_split
      - File I/O inside workspace/ — read a file saved by an earlier tool call, write a derived
        CSV/txt for the user or for a later step
      - Quick matplotlib plots as a side effect of an analysis script (matplotlib.use('Agg')
        before importing pyplot, then plt.savefig(...) — no display in this sandboxed subprocess).
        For an actual chart request, use the dedicated `plot` tool instead — it pre-imports
        plt/pd/np, auto-saves+opens the PNG, auto-caps bar/pie density, and renders Thai text
        safely. Reach for python_exec's own plotting only when the figure is incidental to a
        bigger computation, not the goal itself.

    HOW:
      - Write plain Python; only stdout (print(...)) reaches you back — a value you compute but
        never print is invisible to you and shows up as "(no output)".
      - workspace/ is the cwd: pd.read_csv('data.csv') / df.to_csv('out.csv') use relative paths
        directly, no need to build the workspace path yourself.
      - Errors: a traceback prints under an "[stderr]" header in the result; if the script
        crashed with zero stdout you instead get "[error] exited <code>, no output". Common
        errors (ModuleNotFoundError, FileNotFoundError, KeyError, UnicodeDecodeError,
        SyntaxError) also get a one-line "[hint]" — read it before you retry.

        A hint can list more than one option — if the FIRST one you try also fails, try the
        NEXT one before giving up. Example (decode error, don't stop after one failed retry):
          call 1: pd.read_csv('survey.csv')
                  -> UnicodeDecodeError + [hint] retry with encoding='utf-8-sig' or
                     encoding='latin-1'.
          call 2: pd.read_csv('survey.csv', encoding='utf-8-sig')
                  -> still UnicodeDecodeError (utf-8-sig only strips a BOM, it doesn't fix
                     non-UTF-8 bytes) — this does NOT mean the file is unreadable, it means
                     try the hint's OTHER option next.
          call 3: pd.read_csv('survey.csv', encoding='latin-1')
                  -> succeeds. Only after exhausting every option the hint lists should you
                     tell the user the file couldn't be read.

    LIBRARIES available: pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels.

    NOT for: shell commands (use bash), file I/O without Python (use read_file/write_file),
    standalone chart requests (use plot — see CAPABILITIES above).

    Output cap: printed output over max_chars (default 10,000) is truncated to the last
    complete line; the FULL untruncated output is saved to a workspace file and its path is
    included in a "[python_exec] truncated: ..." marker. No marker in the result means the
    output you see is complete — pass a larger max_chars if you deliberately need more.

    IMPORTANT — what truncation does and does NOT mean: it only affects what got PRINTED.
    Any aggregate you already computed (df.sum(), df.groupby(...).mean(), .describe(), etc.)
    ran on the complete in-memory data regardless of the cap — those numbers are NOT samples.
    Tell the user the data was truncated only when your answer relies on specific rows/values
    that may be beyond the visible cutoff (e.g. you printed an unaggregated row-by-row dump and
    are about to describe rows you can't actually see in the result). Example:
      BAD:  printed 10,000 rows of transactions, output got truncated at row ~900, then told
            the user "the largest transaction is row 905" — that row was never visible, it's a
            guess dressed up as fact.
      GOOD: same truncation happens, but the code already called df.nlargest(5, 'amount') and
            printed just that — the printed rows ARE the full answer, computed before any cap,
            so report it directly with no caveat needed.
    """
    return _python_exec_impl(code, timeout, max_chars)

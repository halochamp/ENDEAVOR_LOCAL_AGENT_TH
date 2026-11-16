# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""Make extracted/translated plain text safe to embed in a Markdown file (code does the structure, not the model).

Used by PDF to MD (pages the model did not write) and by the document translator.  See text_to_markdown for what is
neutralised and close_open_fence for the "keep real code blocks but never let one swallow the next page" variant.
"""
from __future__ import annotations

import re

_MD_LINE_SYNTAX_RE = re.compile(r"^(\s*)([#>]|[-+*](?=\s)|[-=_*]{2,}\s*$)")
_MD_ORDERED_ITEM_RE = re.compile(r"^(\s*[0-9]+)([.)])(?=\s)")
_MD_FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})")
_MD_HTML_START_RE = re.compile(r"<(?=[A-Za-z/!?])")
_MD_BACKSLASH_RE = re.compile(r"\\([!-/:-@\[-`{-~])")           # backslash + punctuation: Markdown would read it as an escape
_MD_LEADING_SPACE_RE = re.compile(r"^[ \t]+")


def _preserve_indent(match: re.Match) -> str:
    """Indent of 4+ columns after a blank line would render as a code block: keep the look with no-break
    spaces, which CommonMark does not count as indentation.  1-3 columns are harmless and left alone."""
    columns = len(match.group(0).expandtabs(4))
    return "\u00a0" * columns if columns >= 4 else match.group(0)


def text_to_markdown(text: str, *, hard_breaks: bool = True) -> str:
    """Make extracted plain text show as written once the .md is rendered.

    Only for text the model did not write as Markdown.  With hard_breaks (PDF to MD, where the original lines matter) every
    line that continues into a non-blank line gets a hard break (two trailing spaces; Markdown would otherwise
    join consecutive lines into one paragraph).  Everything a renderer
    would read as syntax instead of text is neutralised: line-start # > - + * bullets, 1. lists and --- / === rules;
    ``` / ~~~ fences (an unclosed one swallows the rest of the document, page numbers included); text that looks
    like an HTML tag; 4+ column indents; and a backslash before punctuation (it would be eaten as an escape).
    Thai numerals are not Markdown list markers, so "๑." needs no escape."""
    lines = text.split("\n")
    out: list[str] = []
    for index, line in enumerate(lines):
        line = _MD_BACKSLASH_RE.sub(lambda m: "\\\\\\" + m.group(1), line)    # \X -> \\\X: literal backslash, literal X
        line = _MD_FENCE_RE.sub(lambda m: m.group(1) + "".join("\\" + ch for ch in m.group(2)), line)
        line = _MD_HTML_START_RE.sub(lambda m: "\\<", line)
        line = _MD_LEADING_SPACE_RE.sub(_preserve_indent, line)
        line = _MD_ORDERED_ITEM_RE.sub(lambda m: f"{m.group(1)}\\{m.group(2)}", line)
        line = _MD_LINE_SYNTAX_RE.sub(lambda m: f"{m.group(1)}\\{m.group(2)}", line)
        if hard_breaks and line.strip() and index + 1 < len(lines) and lines[index + 1].strip():
            line += "  "
        out.append(line)
    return "\n".join(out)


_FENCE_LINE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def close_open_fence(text: str) -> str:
    """Append the closing line of a code fence the text opened and never closed.

    For text that is allowed to contain real Markdown (a translated .md document): balanced code blocks are left
    exactly as they are, but an unclosed one would swallow everything after it -- the next page's heading and
    page break included.  CommonMark rules: a closer uses the same character, at least as many of them as the
    opener, and nothing else on the line; a backtick opener's info string may not contain a backtick."""
    open_marker: str | None = None
    for line in text.split("\n"):
        match = _FENCE_LINE_RE.match(line)
        if not match:
            continue
        marker, rest = match.group(1), match.group(2)
        if open_marker is None:
            if marker[0] == "`" and "`" in rest:
                continue                        # `` `code` `` style inline span, not a fence
            open_marker = marker
        elif marker[0] == open_marker[0] and len(marker) >= len(open_marker) and not rest.strip():
            open_marker = None
    if open_marker is None:
        return text
    return text.rstrip("\n") + "\n" + open_marker + "\n"

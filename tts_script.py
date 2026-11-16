# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""Text -> speakable script for the Electron "Text to Speech" panel.

Pure functions, no GPU and no network: Markdown / plain text goes in, a list of
spoken paragraphs comes out, plus counters for what was left out.  Everything the
voice would otherwise read aloud as noise (code fences, URLs, image links, table
rules, emoji, ...) is removed here so the synthesis stage only sees prose.

Thai-specific policy (measured on a Thai neural reader):
  * ไปยาลน้อย (ฯ) never reaches the voice: "ฯลฯ" -> "และอื่น ๆ", other ฯ dropped;
  * the ำ-after-tone-mark order repair shared with PDF to Text.  The OCR *spacing*
    repair is deliberately NOT used: it deletes the space after a tone mark, but in
    user-written Thai those spaces are the clause breaks that drive pausing;
  * Arabic digits are left as digits (A/B listening: raw digits were preferred), but
    "%" becomes "เปอร์เซ็นต์": the reader skipped the symbol in 6 of 6 measured takes.

"""
from __future__ import annotations

import html
import re
import unicodedata


from pdf_to_text import repair_thai_orthographic_order

_PAIYANNOI_ETC_RE = re.compile(r"ฯลฯ")
_PAIYANNOI_RE = re.compile(r"ฯ")
_FRONT_MATTER_RE = re.compile(r"\A﻿?---[ \t]*\n.*?\n(?:---|\.\.\.)[ \t]*(?:\n|\Z)", re.S)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
_HR_RE = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")
_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d{1,3}[.)])\s+(?:\[[ xX]\]\s+)?")
_QUOTE_RE = re.compile(r"^\s{0,3}(?:>\s?)+")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$")
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_REF_LINK_RE = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
_FOOTNOTE_RE = re.compile(r"\[\^[^\]]+\]")
_AUTOLINK_RE = re.compile(r"<(?:https?://|mailto:)[^>]+>")
_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>()\[\]]+", re.I)
_CODE_SPAN_RE = re.compile(r"`+([^`]+)`+")
_BOLD_RE = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_ITALIC_STAR_RE = re.compile(r"(?<![*\w])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![*\w])")
_ITALIC_UNDER_RE = re.compile(r"(?<![_\w])_(?=\S)([^_\n]+?)(?<=\S)_(?![_\w])")
_STRIKE_RE = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~")
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF☀-➿⬀-⯿︀-️​-‏⁠­]"
)
_PERCENT_RE = re.compile(r"(\d(?:[\d,]*\d)?(?:\.\d+)?)\s*%")
_SPACED_DASH_RE = re.compile(r"\s+[—–―]+\s+|\s*[—―]{1,2}\s*")
_MULTISPACE_RE = re.compile(r"[ \t 　]+")
_TERMINAL_RE = re.compile(r"[.!?…:;)\]”’\"'»。！？]\s*$")
_THAI_RE = re.compile(r"[฀-๿]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def _strip_paiyannoi(text: str) -> str:
    return _PAIYANNOI_RE.sub("", _PAIYANNOI_ETC_RE.sub("และอื่น ๆ", text))


def _is_latin_dominant(text: str) -> bool:
    latin = len(_LATIN_RE.findall(text))
    thai = len(_THAI_RE.findall(text))
    return latin > thai


def clean_inline(text: str, counters: dict[str, int] | None = None) -> str:
    """Markdown inline syntax and noise -> plain sentence text."""
    c = counters if counters is not None else {}

    def drop(pattern: re.Pattern, key: str, repl: str = "") -> None:
        nonlocal text
        text, n = pattern.subn(repl, text)
        if n:
            c[key] = c.get(key, 0) + n

    text = html.unescape(text)
    drop(_IMAGE_RE, "images")
    drop(_LINK_RE, "links", r"\1")
    drop(_REF_LINK_RE, "links", r"\1")
    drop(_FOOTNOTE_RE, "footnotes")
    drop(_AUTOLINK_RE, "urls")
    drop(_URL_RE, "urls")
    drop(_HTML_TAG_RE, "html_tags")
    drop(_CODE_SPAN_RE, "code_spans", r"\1")
    drop(_BOLD_RE, "emphasis", r"\2")
    drop(_ITALIC_STAR_RE, "emphasis", r"\1")
    drop(_ITALIC_UNDER_RE, "emphasis", r"\1")
    drop(_STRIKE_RE, "emphasis", r"\1")
    drop(_EMOJI_RE, "symbols")
    text = _SPACED_DASH_RE.sub(", ", text)
    text = _PERCENT_RE.sub(r"\1 เปอร์เซ็นต์", text)
    text = unicodedata.normalize("NFC", text)
    text = _strip_paiyannoi(text)
    text = repair_thai_orthographic_order(text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    # A comma left dangling by an removed fragment ("... , ,") reads as a stutter.
    text = re.sub(r"\s*,(?:\s*,)+", ",", text)
    text = re.sub(r"^[,;:\s]+|[\s,;:]+$", "", text) if text else text
    return text


def _table_cells(line: str) -> list[str]:
    inner = line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    return [cell.strip() for cell in inner.split("|")]


def clean_document(raw: str, *, markdown: bool = True) -> tuple[list[str], dict[str, int]]:
    """Return (spoken paragraphs, counters).

    `markdown=False` keeps the structural rules (a blank line ends a paragraph)
    but does not interpret `#`, `*`, tables or fences, since a .txt file can
    legitimately start a line with those characters.
    """
    counters: dict[str, int] = {}
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    if markdown:
        text = _FRONT_MATTER_RE.sub("", text, count=1)
        text, n = _HTML_COMMENT_RE.subn("", text)
        if n:
            counters["comments"] = n
    lines = text.split("\n")

    paragraphs: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        joined = " ".join(part.strip() for part in buffer if part.strip())
        buffer.clear()
        cleaned = clean_inline(joined, counters) if markdown else _plain_inline(joined)
        if cleaned and any(ch.isalnum() for ch in cleaned):
            paragraphs.append(cleaned)

    def emit(block: str) -> None:
        flush()
        cleaned = clean_inline(block, counters) if markdown else _plain_inline(block)
        if cleaned and any(ch.isalnum() for ch in cleaned):
            paragraphs.append(cleaned)

    fence: str | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if markdown:
            if fence is not None:
                if stripped.startswith(fence):
                    fence = None
                continue
            if stripped.startswith(("```", "~~~")):
                fence = stripped[:3]
                counters["code_blocks"] = counters.get("code_blocks", 0) + 1
                flush()
                continue
        if not stripped:
            flush()
            continue
        if markdown:
            if _HR_RE.match(line):
                flush()
                continue
            heading = _HEADING_RE.match(line)
            if heading:
                emit(heading.group(1))
                continue
            nxt = lines[index + 1] if index + 1 < len(lines) else ""
            if "|" in line and (_TABLE_SEP_RE.match(nxt) or line.lstrip().startswith("|")):
                if _TABLE_SEP_RE.match(line):
                    continue
                cells = [cell for cell in _table_cells(line) if cell]
                emit(", ".join(cells))
                counters["table_rows"] = counters.get("table_rows", 0) + 1
                continue
            line = _QUOTE_RE.sub("", line)
            if _LIST_RE.match(line):
                emit(_LIST_RE.sub("", line, count=1))
                continue
        # Plain prose line: join a hard-wrapped Latin sentence, otherwise each
        # line is its own paragraph (Thai text is not hard-wrapped in practice).
        if buffer and _is_latin_dominant(buffer[-1]) and not _TERMINAL_RE.search(buffer[-1]):
            buffer.append(line)
        else:
            flush()
            buffer.append(line)
    flush()
    return paragraphs, counters


def _plain_inline(text: str) -> str:
    text = _URL_RE.sub("", html.unescape(text))
    text = _EMOJI_RE.sub("", text)
    text = unicodedata.normalize("NFC", text)
    text = _strip_paiyannoi(text)
    text = repair_thai_orthographic_order(text)
    return _MULTISPACE_RE.sub(" ", text).strip()



# ── fixed Thai readings for common English terms ────────────────────────────
# The local LLM transliterates leftovers, but it is not reliable on the common
# ones (measured: ChatGPT -> "แชลต์จีพีที", an extra ล that F5 then mispronounced).
# A fixed table is applied first; the LLM only sees what is still Latin.  Users can
# extend/override it with workspace/text_to_speech/glossary.json ({"Term": "คำอ่าน"}).
DEFAULT_GLOSSARY: dict[str, str] = {
    "ChatGPT": "แชตจีพีที", "GPT": "จีพีที", "OpenAI": "โอเพนเอไอ", "AI": "เอไอ", "API": "เอพีไอ",
    "PDF": "พีดีเอฟ", "URL": "ยูอาร์แอล", "USB": "ยูเอสบี", "Excel": "เอ็กเซล", "PowerPoint": "พาวเวอร์พอยต์",
    "Google": "กูเกิล", "Gmail": "จีเมล", "YouTube": "ยูทูบ", "Facebook": "เฟซบุ๊ก", "Instagram": "อินสตาแกรม",
    "TikTok": "ติ๊กต็อก", "Telegram": "เทเลแกรม", "iPhone": "ไอโฟน", "iPad": "ไอแพด", "Android": "แอนดรอยด์",
    "Windows": "วินโดวส์", "Python": "ไพธอน", "Wi-Fi": "ไวไฟ", "WiFi": "ไวไฟ", "SQL": "เอสคิวแอล",
    "Claude": "คล็อด", "Gemini": "เจมินี", "ETF": "อีทีเอฟ", "GDP": "จีดีพี", "CPI": "ซีพีไอ", "IPO": "ไอพีโอ",
    "ATM": "เอทีเอ็ม", "KPI": "เคพีไอ", "CEO": "ซีอีโอ", "HR": "เอชอาร์", "IT": "ไอที", "SMS": "เอสเอ็มเอส",
    "QR": "คิวอาร์", "TV": "ทีวี", "PIN": "พิน", "OTP": "โอทีพี", "NFT": "เอ็นเอฟที", "Bitcoin": "บิตคอยน์",
    # ALL-CAPS names people say as words, which the letter-by-letter rule would spell out
    "NASA": "นาซา", "ASEAN": "อาเซียน", "FIFA": "ฟีฟ่า", "OPEC": "โอเปก", "COVID": "โควิด", "LINE": "ไลน์",
    "UNESCO": "ยูเนสโก", "REIT": "รีท", "NATO": "นาโต", "WHO": "ดับเบิลยูเอชโอ", "UFO": "ยูเอฟโอ",
    "kg": "กิโลกรัม", "km": "กิโลเมตร", "cm": "เซนติเมตร", "mm": "มิลลิเมตร", "GB": "กิกะไบต์",
    "MB": "เมกะไบต์", "TB": "เทราไบต์", "USD": "ดอลลาร์สหรัฐ", "THB": "บาท",
    # tech and platform brands
    "Apple": "แอปเปิล", "Microsoft": "ไมโครซอฟท์", "Amazon": "อเมซอน", "Meta": "เมตา", "Netflix": "เน็ตฟลิกซ์",
    "Spotify": "สปอติฟาย", "Uber": "อูเบอร์", "Grab": "แกร็บ", "Shopee": "ช้อปปี้", "Lazada": "ลาซาด้า",
    "Tesla": "เทสลา", "Nvidia": "เอ็นวิเดีย", "Intel": "อินเทล", "Samsung": "ซัมซุง", "Huawei": "หัวเว่ย",
    "Xiaomi": "เสียวหมี่", "Sony": "โซนี่", "Nintendo": "นินเทนโด", "PlayStation": "เพลย์สเตชัน", "Xbox": "เอกซ์บอกซ์",
    "Zoom": "ซูม", "Slack": "สแล็ก", "Notion": "โนชั่น", "Figma": "ฟิกม่า", "Canva": "แคนวา", "Docker": "ด็อกเกอร์",
    "GitHub": "กิตฮับ", "Linux": "ลินุกซ์", "Ubuntu": "อูบุนตู", "Chrome": "โครม", "Safari": "ซาฟารี",
    "Firefox": "ไฟร์ฟอกซ์", "WhatsApp": "วอตส์แอป", "WeChat": "วีแชต", "Discord": "ดิสคอร์ด", "Twitter": "ทวิตเตอร์",
    "Pinterest": "พินเทอเรสต์", "LinkedIn": "ลิงก์อิน", "Reddit": "เรดดิต", "Twitch": "ทวิช", "Binance": "ไบแนนซ์",
    "Coinbase": "คอยน์เบส", "Ethereum": "อีเธอเรียม", "Blockchain": "บล็อกเชน", "Robinhood": "โรบินฮู้ด",
    "Nasdaq": "แนสแด็ก", "Photoshop": "โฟโต้ช็อป", "Illustrator": "อิลลัสเตรเตอร์", "Outlook": "เอาต์ลุก",
    "OneDrive": "วันไดรฟ์", "Dropbox": "ดรอปบอกซ์", "WordPress": "เวิร์ดเพรส", "Wikipedia": "วิกิพีเดีย",
    "Siri": "ซิริ", "Alexa": "อเล็กซา", "Bluetooth": "บลูทูธ", "Copilot": "โคไพลอต", "Midjourney": "มิดเจอร์นีย์",
    "DeepSeek": "ดีปซีก", "Anthropic": "แอนโทรปิก", "Yahoo": "ยาฮู", "Alibaba": "อาลีบาบา", "Tencent": "เทนเซนต์",
    "Bloomberg": "บลูมเบิร์ก", "Reuters": "รอยเตอร์ส", "Forbes": "ฟอร์บส์", "Bing": "บิง",
    "Tableau": "แทบโล", "Kubernetes": "คูเบอร์เนทีส", "Shazam": "ชาแซม", "Kindle": "คินเดิล", "Roblox": "โรบล็อกซ์",
    "Fitbit": "ฟิตบิต", "Lazada": "ลาซาด้า", "Python": "ไพธอน",
    # everyday loanwords written in Latin
    "online": "ออนไลน์", "offline": "ออฟไลน์", "email": "อีเมล", "internet": "อินเทอร์เน็ต", "website": "เว็บไซต์",
    "app": "แอป", "podcast": "พอดแคสต์", "startup": "สตาร์ทอัพ", "marketing": "มาร์เก็ตติ้ง", "content": "คอนเทนต์",
    "blog": "บล็อก", "link": "ลิงก์", "click": "คลิก", "dividend": "ดิวิเดนด์", "index": "อินเด็กซ์",
    # investing: people, firms, instruments
    "Buffett": "บัฟเฟตต์", "Munger": "มังเกอร์", "Berkshire": "เบิร์กเชียร์", "Hathaway": "แฮทะเวย์",
    "Bogle": "โบเกิล", "Vanguard": "แวนการ์ด", "Graham": "เกรแฮม", "Dalio": "ดาลิโอ", "Lynch": "ลินช์",
    "Fed": "เฟด", "Powell": "พาวเวลล์", "Trump": "ทรัมป์", "Musk": "มัสก์", "Elon": "อีลอน", "Jobs": "จ็อบส์",
    "Gates": "เกตส์", "Bezos": "เบโซส", "Zuckerberg": "ซักเคอร์เบิร์ก", "Warren": "วอร์เรน", "Charlie": "ชาร์ลี",
    "SET": "เซ็ต", "SET50": "เซ็ตห้าสิบ", "SET100": "เซ็ตหนึ่งร้อย", "S&P": "เอสแอนด์พี", "Dow": "ดาว",
}


def apply_glossary(text: str, extra: dict[str, str] | None = None) -> tuple[str, int]:
    """Replace whole-token Latin terms with their fixed Thai reading.  Longest
    terms first so "ChatGPT" wins over "GPT"; case-insensitive for words, exact
    for 1-3 letter acronyms so "it" in prose is never read as "ไอที"."""
    table = dict(DEFAULT_GLOSSARY)
    for term, reading in (extra or {}).items():
        if isinstance(term, str) and isinstance(reading, str) and term.strip() and reading.strip():
            table[term.strip()] = reading.strip()
    count = 0
    for term in sorted(table, key=len, reverse=True):
        flags = 0 if (len(term) <= 3 and term.isupper()) else re.IGNORECASE
        pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])", flags)
        text, n = pattern.subn(table[term], text)
        count += n
    return text, count


# ── acronyms and unknown English terms ──────────────────────────────────────
# Order of resolution: fixed glossary -> acronym letter rule -> local 2B model.
# The rule beats the model on acronyms (the 2B wrote "SCB" as "ซีซีบี"/"ส.ค.บ.").

LETTER_NAMES = {
    "A": "เอ", "B": "บี", "C": "ซี", "D": "ดี", "E": "อี", "F": "เอฟ", "G": "จี", "H": "เอช", "I": "ไอ",
    "J": "เจ", "K": "เค", "L": "แอล", "M": "เอ็ม", "N": "เอ็น", "O": "โอ", "P": "พี", "Q": "คิว", "R": "อาร์",
    "S": "เอส", "T": "ที", "U": "ยู", "V": "วี", "W": "ดับเบิลยู", "X": "เอ็กซ์", "Y": "วาย", "Z": "แซด",
}
_ACRONYM_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z]{2,6})(?![A-Za-z])(\d*)")
_LETTER_DIGIT_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z])(\d{1,3})(?![A-Za-z0-9])")
_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'’][A-Za-z0-9]+)*")


def spell_acronyms(text: str) -> tuple[str, int]:
    """ALL-CAPS runs of 2-6 letters (SCB, RMF, SET50) and letter+digit tags (v3)
    are read letter by letter, the way Thai speakers read them."""
    count = 0

    def acronym(match: re.Match) -> str:
        nonlocal count
        count += 1
        spoken = "".join(LETTER_NAMES[ch] for ch in match.group(1))
        return f"{spoken} {match.group(2)}" if match.group(2) else spoken

    def tag(match: re.Match) -> str:
        nonlocal count
        count += 1
        return f"{LETTER_NAMES[match.group(1).upper()]} {match.group(2)}"

    text = _ACRONYM_RE.sub(acronym, text)
    return _LETTER_DIGIT_RE.sub(tag, text), count


_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_MAX_NUMBER_DIGITS = 15  # num_to_thaiword is exact far beyond this; a longer run is an id or a card number


def expand_numbers(text: str) -> tuple[str, int]:
    """Arabic numbers -> Thai number words, for voices that misread digits (OmniVoice read "1,420" as
    nonsense and scored CER 10.1% against 3.6% once the numbers were words; 2026-11-15 A/B).  A leading-zero
    run (phone, id) or a very long run is read digit by digit.  Without pythainlp the text is left as it is."""
    try:
        from pythainlp.util import num_to_thaiword
    except ImportError:
        return text, 0
    count = 0

    def digits(run: str) -> str:
        return "".join(num_to_thaiword(int(ch)) for ch in run)

    def convert(match: re.Match) -> str:
        nonlocal count
        count += 1
        whole, _, fraction = match.group().replace(",", "").partition(".")
        if (len(whole) > 1 and whole.startswith("0")) or len(whole) > _MAX_NUMBER_DIGITS:
            spoken = digits(whole)
        else:
            spoken = num_to_thaiword(int(whole))
        if fraction:
            spoken = f"{spoken}จุด{digits(fraction)}"
        # a number glued to Latin letters ("SET50", "3D") must not fuse with the English word it sits next to
        text_before = match.string[:match.start()]
        text_after = match.string[match.end():]
        lead = " " if text_before[-1:].isascii() and text_before[-1:].isalpha() else ""
        trail = " " if text_after[:1].isascii() and text_after[:1].isalpha() else ""
        return f"{lead}{spoken}{trail}"

    return _NUMBER_RE.sub(convert, text), count


def find_unknown_terms(paragraphs: list[str]) -> list[str]:
    """Unique Latin words still present, in order of first appearance."""
    seen: dict[str, None] = {}
    for paragraph in paragraphs:
        for match in _TERM_RE.finditer(paragraph):
            token = match.group()
            if len(token) >= 2 and token not in seen:
                seen[token] = None
    return list(seen)


def replace_terms(text: str, readings: dict[str, str]) -> str:
    """Whole-token, case-exact replacement; longest terms first."""
    if not readings:
        return text
    alternation = "|".join(re.escape(term) for term in sorted(readings, key=len, reverse=True))
    pattern = re.compile(r"(?<![A-Za-z0-9])(?:" + alternation + r")(?![A-Za-z0-9])")
    return pattern.sub(lambda m: readings[m.group()], text)


_THAI_READING_RE = re.compile(r"^[\u0E01-\u0E4E ]{1,40}$")


def clean_reading(term: str, reading: object) -> str | None:
    """Accept a model-proposed Thai reading only if it is Thai letters and marks."""
    if not isinstance(reading, str):
        return None
    candidate = unicodedata.normalize("NFC", _strip_paiyannoi(reading)).strip()
    candidate = _MULTISPACE_RE.sub(" ", candidate)
    if not _THAI_READING_RE.match(candidate) or not re.search(r"[\u0E01-\u0E2E]", candidate):
        return None
    if len(candidate) > 5 * len(term) + 8:
        return None
    return candidate

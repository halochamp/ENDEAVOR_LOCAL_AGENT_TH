# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""Text to Speech for the Agent TH Electron panel: a .txt/.md file -> a loudness-mastered .m4a read aloud locally.

The voice is OmniVoice (k2-fsa) running in its own conda env and its own worker process (scripts/tts_omnivoice_worker.py,
JSON lines on stdin/stdout, no TCP port); the model leaves RAM when the job ends.  OmniVoice's weights are CC-BY-NC:
personal and non-commercial use only.  Nothing is downloaded: the owner installs the env and the weights.

Pipeline (everything except the voice model is plain Python and unit-tested without any model):

1. read   the file (utf-8 / cp874) and clean it for reading aloud (tts_script.clean_document: Markdown, links, code)
2. prep   Thai text gets Thai number words (OmniVoice misreads digits); English text keeps its words and digits and is
          flagged ``language="English"`` for the model
3. plan   tts_split cuts each paragraph into pieces (OmniVoice takes whole paragraphs up to ~400 characters and puts its
          own pauses inside them, so only longer paragraphs are cut, at a sentence/clause break)
4. voice  every piece is cloned from ONE reference clip (voices/<id>/ref_voice.wav + ref_voice.txt, 2-8 s, mono 16-bit),
          encoded once, so the voice cannot drift between pieces; a silent piece is retried once
5. master pieces are joined with designed pauses (tts_split._concat_wavs), loudness-normalised (-16 LUFS) and written as .m4a

Pause / resume / cancel are cooperative: the job checks between pieces, a pause keeps the worker and the finished pieces.
"""
from __future__ import annotations

from collections.abc import Callable
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path

import config
import tts_script
import tts_split

Progress = Callable[[str, int, int, dict], None]
Flag = Callable[[], bool]

_THAI_RE = re.compile(r"[฀-๿]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_VOICE_ID_RE = re.compile(r"^[a-z0-9_-]{1,32}$")
_VOICE_REF_SECONDS = (2.0, 8.0)       # the reference clip length OmniVoice clones well from
_NFE, _CFG = 32, 2.0                  # OmniVoice's own defaults (num_step, guidance_scale)
_PIECE_TARGET, _PIECE_WINDOW_MIN = 400, 200
_SILENCE_PEAK = 2000                  # int16 peak below this = the model produced near-silence: retry once
_SAMPLE_RATE = 24000
_PAUSE_POLL_SECONDS = 0.25
SPEED_RANGE = (0.8, 1.2)


class TtsCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class Options:
    voice: str = ""
    speed: float = 1.0

    def normalised(self) -> "Options":
        speed = min(SPEED_RANGE[1], max(SPEED_RANGE[0], float(self.speed or 1.0)))
        voice = self.voice if _VOICE_ID_RE.match(str(self.voice or "")) else ""
        return Options(voice=voice, speed=round(speed, 2))


# ── engine and voices ────────────────────────────────────────────────────────

def worker_path() -> Path:
    return Path(__file__).resolve().parent / "scripts" / "tts_omnivoice_worker.py"


def engine_status() -> dict:
    """{"ready": bool, "reason": str} — what is missing, in words the panel can show.  Loads nothing."""
    if not Path(config.TTS_OMNIVOICE_PYTHON).is_file():
        return {"ready": False, "reason": f"The 'omnivoice' conda env was not found ({config.TTS_OMNIVOICE_PYTHON}); see the README, or set V2_TTS_OMNIVOICE_PYTHON"}
    if not (Path(config.TTS_OMNIVOICE_DIR) / "model.safetensors").is_file():
        return {"ready": False, "reason": f"The OmniVoice model files were not found in {config.TTS_OMNIVOICE_DIR} (V2_TTS_OMNIVOICE_DIR)"}
    if not worker_path().is_file():
        return {"ready": False, "reason": "scripts/tts_omnivoice_worker.py is missing"}
    return {"ready": True, "reason": ""}


def voices_root() -> Path:
    return Path(config.TTS_VOICES_DIR)


def validate_reference(wav_path: str | Path, text: str) -> str | None:
    """None when the clip can be a reference, else a one-line reason: mono 16/32-bit WAV, 2-8 s, with its exact transcript."""
    if not str(text or "").strip():
        return "A transcript that matches the audio in the clip is required"
    try:
        with wave.open(str(wav_path), "rb") as clip:
            seconds = clip.getnframes() / float(clip.getframerate())
            if clip.getnchannels() != 1 or clip.getsampwidth() not in (2, 4):
                return "The clip must be a mono 16-bit WAV"
    except (wave.Error, OSError, EOFError):
        return "Could not read the WAV file"
    if not _VOICE_REF_SECONDS[0] <= seconds <= _VOICE_REF_SECONDS[1]:
        return f"The clip must be {_VOICE_REF_SECONDS[0]:.0f}-{_VOICE_REF_SECONDS[1]:.0f} seconds long (got {seconds:.1f})"
    return None


def resolve_voice(voice_id: str) -> dict | None:
    """{"ref_wav", "ref_text"} for a valid voice folder, else None."""
    if not _VOICE_ID_RE.match(voice_id or ""):
        return None
    folder = voices_root() / voice_id
    wav_path, txt_path = folder / "ref_voice.wav", folder / "ref_voice.txt"
    try:
        text = txt_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not wav_path.is_file() or validate_reference(wav_path, text):
        return None
    return {"ref_wav": str(wav_path), "ref_text": text}


def list_voices() -> list[dict]:
    """Every valid voice folder (folder name = id), with the label from voice.json when it has one."""
    rows: list[dict] = []
    try:
        folders = sorted(d for d in voices_root().iterdir() if d.is_dir() and _VOICE_ID_RE.match(d.name))
    except OSError:
        return rows
    for folder in folders:
        if resolve_voice(folder.name) is None:
            continue
        try:
            meta = json.loads((folder / "voice.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        rows.append({"id": folder.name, "label": str(meta.get("label") or folder.name), "note": str(meta.get("license") or "")})
    return rows


# ── pure helpers ─────────────────────────────────────────────────────────────

def read_source_text(path: Path) -> str:
    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "cp874"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def is_english(text: str) -> bool:
    """More Latin letters than Thai characters (digits and symbols do not count)."""
    return len(_LATIN_RE.findall(text)) > len(_THAI_RE.findall(text))


def plan_pieces(paragraphs: list[str]) -> list[dict]:
    """Paragraphs -> [{"text", "boundary", "english"}] in reading order; the last piece of a paragraph ends a paragraph."""
    pieces: list[dict] = []
    for paragraph in paragraphs:
        rows = tts_split._split_for_tts(paragraph, _PIECE_TARGET, False, window_min=_PIECE_WINDOW_MIN, with_tiers=True)
        rows = tts_split.merge_short_pieces([[text, boundary] for text, boundary in rows if text.strip()])
        if rows:
            rows[-1][1] = "paragraph"
        pieces.extend({"text": text, "boundary": boundary, "english": is_english(text)} for text, boundary in rows)
    return pieces


def prepare_paragraphs(paragraphs: list[str]) -> tuple[list[str], int]:
    """Thai paragraphs get Thai number words; English ones stay exactly as written.  Returns (paragraphs, numbers expanded)."""
    spoken, count = [], 0
    for text in paragraphs:
        if is_english(text):
            spoken.append(text)
        else:
            expanded, n = tts_script.expand_numbers(text)
            spoken.append(expanded)
            count += n
    return spoken, count


def wav_duration(path: str | Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def _ffmpeg() -> str:
    return shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


def master_to_m4a(wav_in: Path, m4a_out: Path, *, target_lufs: float = -16.0, true_peak: float = -1.5) -> dict:
    """Two-pass EBU R128 loudness normalisation, then AAC.  Falls back to a plain encode when the measurement is unusable."""
    ffmpeg = _ffmpeg()
    base = f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=11"
    first = subprocess.run(
        [ffmpeg, "-hide_banner", "-nostats", "-i", str(wav_in), "-af", base + ":print_format=json", "-f", "null", "-"],
        capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL,
    )
    measured = None
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", first.stderr, re.S)
    if match:
        try:
            measured = json.loads(match.group())
            float(measured["input_i"])
        except (ValueError, KeyError):
            measured = None
    info = {"normalised": False}
    if measured and float(measured["input_i"]) > -70:
        flt = (f"{base}:measured_I={measured['input_i']}:measured_LRA={measured['input_lra']}:"
               f"measured_TP={measured['input_tp']}:measured_thresh={measured['input_thresh']}:"
               f"offset={measured['target_offset']}:linear=true")
        info = {"normalised": True, "input_lufs": float(measured["input_i"]), "target_lufs": target_lufs}
    else:
        flt = "anull"
    tmp = m4a_out.with_name(m4a_out.stem + ".part.m4a")
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav_in), "-af", flt,
         "-ar", "24000", "-ac", "1", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(tmp)],
        check=True, capture_output=True, timeout=900, stdin=subprocess.DEVNULL,
    )
    os.replace(tmp, m4a_out)
    return info


def _unique_path(directory: Path, stem: str, suffix: str) -> Path:
    candidate = directory / f"{stem}{suffix}"
    n = 2
    while candidate.exists():
        candidate = directory / f"{stem}-{n}{suffix}"
        n += 1
    return candidate


def _safe_stem(name: str) -> str:
    stem = re.sub(r"[^\w.\-ก-๙ ]+", "_", Path(name).stem).strip(" ._") or "audio"
    return stem[:80]


def cleanup_cache(now: float | None = None) -> None:
    """Remove stale per-job folders left by a crashed job."""
    cutoff = (now or time.time()) - config.TTS_CACHE_MAX_AGE_SECONDS
    try:
        entries = list(Path(config.TTS_PRIVATE_DIR).iterdir())
    except OSError:
        return
    for entry in entries:
        try:
            if entry.is_dir() and entry.name.startswith("job-") and entry.stat().st_mtime <= cutoff:
                shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            pass


def list_outputs(limit: int = 50) -> list[dict]:
    try:
        files = [p for p in Path(config.TTS_OUTPUT_DIR).glob("*.m4a") if p.is_file()]
    except OSError:
        return []
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"name": p.name, "path": str(p), "bytes": p.stat().st_size} for p in files[:limit]]


# ── the voice worker ─────────────────────────────────────────────────────────

class _Worker:
    """One JSON-lines subprocess; every request gets exactly one reply."""

    def __init__(self, argv: list[str], log_path: Path):
        self.argv = argv
        self.log_path = log_path
        self._proc: subprocess.Popen | None = None
        self._lines: "queue.Queue[str | None]" = queue.Queue()
        self._log = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "ab")
        self._proc = subprocess.Popen(
            self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
            text=True, encoding="utf-8", bufsize=1,
        )
        self._lines = queue.Queue()
        threading.Thread(target=self._pump, args=(self._proc, self._lines), name="tts-reader", daemon=True).start()

    @staticmethod
    def _pump(proc: subprocess.Popen, lines: "queue.Queue[str | None]") -> None:
        assert proc.stdout
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    def _tail(self) -> str:
        try:
            data = self.log_path.read_bytes()[-600:].decode("utf-8", "replace")
        except OSError:
            data = ""
        return data.strip().splitlines()[-1] if data.strip() else ""

    def call(self, payload: dict, timeout: float, should_cancel: Flag | None = None) -> dict:
        if not self._proc or self._proc.poll() is not None:
            raise RuntimeError(f"The voice worker is not running: {self._tail()}")
        assert self._proc.stdin
        self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        deadline = time.monotonic() + timeout
        while True:
            if should_cancel and should_cancel():
                raise TtsCancelled("Cancelled by the user")
            try:
                line = self._lines.get(timeout=0.5)
            except queue.Empty:
                if time.monotonic() > deadline:
                    raise RuntimeError(f"The voice worker timed out after {int(timeout)}s")
                continue
            if line is None:
                raise RuntimeError(f"The voice worker exited: {self._tail()}")
            try:
                reply = json.loads(line)
            except ValueError:
                continue  # stray library output on the protocol channel
            if not reply.get("ok"):
                raise RuntimeError(f"voice worker: {reply.get('error', 'failed')}")
            return reply

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                if proc.poll() is None and proc.stdin:
                    proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                    proc.stdin.flush()
                    proc.wait(timeout=10)
            except Exception:
                pass
            if proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(timeout=10)
                except Exception:
                    pass
        if self._log is not None:
            self._log.close()
            self._log = None


# ── the job ──────────────────────────────────────────────────────────────────

def synthesize_document(
    source: str | Path,
    *,
    source_name: str | None = None,
    options: Options | None = None,
    progress: Progress | None = None,
    should_pause: Flag | None = None,
    should_cancel: Flag | None = None,
) -> dict:
    """Read one .txt/.md file aloud and return {"output_path", "duration_sec", "pieces", "warnings", "stats"}."""
    opts = (options or Options()).normalised()
    started = time.monotonic()
    name = Path(source_name or Path(source).name).name

    def emit(stage: str, done: int = 0, total: int = 0, **stats) -> None:
        if progress:
            progress(stage, done, total, stats)

    def check_cancel() -> None:
        if should_cancel and should_cancel():
            raise TtsCancelled("Cancelled by the user")

    engine = engine_status()
    if not engine["ready"]:
        raise ValueError(engine["reason"])
    voice = resolve_voice(opts.voice) if opts.voice else None
    if voice is None:
        raise ValueError(f"Choose a voice: put ref_voice.wav (mono, 2-8 s) and ref_voice.txt (its exact words) in a folder under {voices_root()}")

    emit("read")
    raw = read_source_text(Path(source))
    markdown = Path(name).suffix.casefold() in {".md", ".markdown"}
    paragraphs, counts = tts_script.clean_document(raw, markdown=markdown)
    if not paragraphs:
        raise ValueError("No speakable text was found in this file")
    warnings: list[str] = []
    if counts.get("code_blocks"):
        warnings.append(f"Skipped {counts['code_blocks']} code block(s)")
    if counts.get("urls"):
        warnings.append(f"Removed {counts['urls']} link(s)")
    spoken, numbers = prepare_paragraphs(paragraphs)
    emit("plan", 0, len(spoken))
    pieces = plan_pieces(spoken)
    if not pieces:
        raise ValueError("No speakable text was found in this file")

    cleanup_cache()
    job_dir = Path(config.TTS_PRIVATE_DIR) / f"job-{uuid.uuid4().hex[:12]}"
    job_dir.mkdir(parents=True, exist_ok=True)
    worker = _Worker([config.TTS_OMNIVOICE_PYTHON, str(worker_path()), config.TTS_OMNIVOICE_DIR], job_dir / "worker.log")
    wavs: list[str] = []
    retried = 0
    try:            # the job folder goes away however the job ends: done, error or cancelled
        try:
            check_cancel()
            emit("load", 0, len(pieces))
            worker.start()
            worker.call({"cmd": "load", **voice}, timeout=300, should_cancel=should_cancel)
            synth_times: list[float] = []
            for index, piece in enumerate(pieces, 1):
                paused = False
                while should_pause and should_pause():
                    check_cancel()
                    if not paused:
                        paused = True
                        emit("paused", index - 1, len(pieces))
                    time.sleep(_PAUSE_POLL_SECONDS)
                check_cancel()
                eta = int(len(pieces) - index + 1) * (sum(synth_times) / len(synth_times)) if synth_times else 0
                emit("synth", index - 1, len(pieces), eta_sec=int(eta), retried=retried)
                check_cancel()   # the panel may have cancelled while it drew that progress: do not send the next piece to the model
                out = job_dir / f"piece-{index:04d}.wav"
                request = {"cmd": "synth", "text": piece["text"], "out": str(out), "nfe": _NFE, "cfg": _CFG, "speed": opts.speed}
                if piece["english"]:
                    request["language"] = "English"
                tick = time.monotonic()
                reply = worker.call(request, timeout=300, should_cancel=should_cancel)
                if int(reply.get("peak") or 0) < _SILENCE_PEAK:
                    retried += 1
                    reply = worker.call(request, timeout=300, should_cancel=should_cancel)
                    if int(reply.get("peak") or 0) < _SILENCE_PEAK:
                        warnings.append(f"Piece {index} came out near-silent twice and was kept as it is")
                synth_times.append(time.monotonic() - tick)
                wavs.append(str(out))
            emit("synth", len(pieces), len(pieces), eta_sec=0, retried=retried)
        finally:
            worker.close()   # the model leaves RAM before mastering starts

        check_cancel()
        emit("master", len(pieces), len(pieces))
        pauses = [tts_split.pause_for(piece["boundary"]) for piece in pieces[:-1]]
        joined = job_dir / "joined.wav"
        if len(wavs) == 1:
            shutil.copyfile(wavs[0], joined)
        else:
            tts_split._concat_wavs([Path(path).read_bytes() for path in wavs], joined, pause_ms=pauses)
        output_dir = Path(config.TTS_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        output = _unique_path(output_dir, _safe_stem(name), ".m4a")
        mastered = master_to_m4a(joined, output)
        duration = wav_duration(joined)
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
    return {
        "output_path": str(output), "duration_sec": round(duration, 2), "pieces": len(pieces), "warnings": warnings,
        "stats": {"voice": opts.voice, "speed": opts.speed, "numbers": numbers, "retried": retried,
                  "english_pieces": sum(1 for p in pieces if p["english"]), "mastering": mastered,
                  "elapsed_sec": round(time.monotonic() - started, 1)},
    }

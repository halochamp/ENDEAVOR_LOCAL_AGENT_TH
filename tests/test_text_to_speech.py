"""Deterministic tests for Agent TH's Text to Speech: everything except the voice model (a fake worker stands in for it)."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import wave
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
import text_to_speech as tts

FAKE_WORKER = Path(__file__).resolve().parent / "fake_tts_worker.py"


def write_wav(path: Path, seconds: float = 4.0, channels: int = 1) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        frames = b"".join(struct.pack("<h", int(8000 * math.sin(i / 12))) * channels for i in range(int(24000 * seconds)))
        handle.writeframes(frames)


class TtsBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="agent-th-tts-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        model = self.root / "model"
        model.mkdir()
        (model / "model.safetensors").write_bytes(b"x")
        self.log = self.root / "worker.log.jsonl"
        for patcher in (
            mock.patch.object(config, "TTS_PRIVATE_DIR", str(self.root / "private")),
            mock.patch.object(config, "TTS_OUTPUT_DIR", str(self.root / "out")),
            mock.patch.object(config, "TTS_VOICES_DIR", str(self.root / "out" / "voices")),
            mock.patch.object(config, "TTS_OMNIVOICE_DIR", str(model)),
            mock.patch.object(config, "TTS_OMNIVOICE_PYTHON", sys.executable),
            mock.patch.object(tts, "worker_path", lambda: FAKE_WORKER),
            mock.patch.dict(os.environ, {"FAKE_TTS_LOG": str(self.log)}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        (self.root / "private").mkdir()
        self.install_voice("anna")

    def install_voice(self, voice_id: str, *, text: str = "ประโยคอ้างอิง", seconds: float = 4.0, label: str | None = None) -> Path:
        folder = self.root / "out" / "voices" / voice_id
        folder.mkdir(parents=True, exist_ok=True)
        write_wav(folder / "ref_voice.wav", seconds)
        (folder / "ref_voice.txt").write_text(text, encoding="utf-8")
        if label:
            (folder / "voice.json").write_text(json.dumps({"label": label}), encoding="utf-8")
        return folder

    def source(self, text: str, name: str = "story.txt") -> Path:
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def requests(self) -> list[dict]:
        return [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()] if self.log.exists() else []


class PureHelperTests(unittest.TestCase):
    def test_options_are_clamped_and_voice_ids_validated(self) -> None:
        self.assertEqual(tts.Options(voice="anna", speed=9).normalised(), tts.Options(voice="anna", speed=1.2))
        self.assertEqual(tts.Options(voice="../etc", speed=0.1).normalised(), tts.Options(voice="", speed=0.8))

    def test_english_detection_counts_letters_not_digits(self) -> None:
        self.assertTrue(tts.is_english("Hello world 123"))
        self.assertFalse(tts.is_english("สวัสดีครับ 123 hello"))

    def test_thai_numbers_become_words_and_english_stays_untouched(self) -> None:
        spoken, count = tts.prepare_paragraphs(["ราคา 1,420 บาท", "It costs 1,420 baht"])
        self.assertNotIn("1,420", spoken[0])
        self.assertGreaterEqual(count, 1)
        self.assertEqual(spoken[1], "It costs 1,420 baht")

    def test_a_short_paragraph_is_one_piece_and_a_long_one_is_cut_at_breaks(self) -> None:
        short = tts.plan_pieces(["ย่อหน้าสั้น ๆ ที่พอดีกับหนึ่งชิ้น"])
        self.assertEqual((len(short), short[0]["boundary"]), (1, "paragraph"))
        long_text = " ".join(["This is a sentence that keeps going for a while, with a clause."] * 12)
        pieces = tts.plan_pieces([long_text])
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(len(p["text"]) <= 450 for p in pieces))
        self.assertTrue(all(p["english"] for p in pieces))
        self.assertEqual(pieces[-1]["boundary"], "paragraph")

    def test_reading_a_cp874_file(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as handle:
            handle.write("สวัสดี".encode("cp874"))
        self.addCleanup(os.unlink, handle.name)
        self.assertEqual(tts.read_source_text(Path(handle.name)), "สวัสดี")


class VoiceTests(TtsBase):
    def test_a_valid_folder_is_a_voice_and_the_label_comes_from_voice_json(self) -> None:
        self.install_voice("bob", label="Bob (calm)")
        rows = {row["id"]: row for row in tts.list_voices()}
        self.assertEqual(set(rows), {"anna", "bob"})
        self.assertEqual(rows["bob"]["label"], "Bob (calm)")
        self.assertEqual(rows["anna"]["label"], "anna")

    def test_bad_references_are_not_voices_and_say_why(self) -> None:
        self.install_voice("long", seconds=12.0)
        self.install_voice("notext", text="")
        (self.root / "out" / "voices" / "stereo").mkdir()
        write_wav(self.root / "out" / "voices" / "stereo" / "ref_voice.wav", 3.0, channels=2)
        (self.root / "out" / "voices" / "stereo" / "ref_voice.txt").write_text("x", encoding="utf-8")
        self.assertEqual({row["id"] for row in tts.list_voices()}, {"anna"})
        self.assertIn("2-8 seconds", tts.validate_reference(self.root / "out" / "voices" / "long" / "ref_voice.wav", "x"))
        self.assertIn("transcript", tts.validate_reference(self.root / "out" / "voices" / "anna" / "ref_voice.wav", " "))
        self.assertIn("mono", tts.validate_reference(self.root / "out" / "voices" / "stereo" / "ref_voice.wav", "x"))
        self.assertIsNone(tts.resolve_voice("../anna"))
        self.assertIsNone(tts.resolve_voice("nobody"))

    def test_engine_status_names_what_is_missing(self) -> None:
        self.assertTrue(tts.engine_status()["ready"])
        with mock.patch.object(config, "TTS_OMNIVOICE_PYTHON", str(self.root / "nope")):
            self.assertIn("conda env", tts.engine_status()["reason"])
        with mock.patch.object(config, "TTS_OMNIVOICE_DIR", str(self.root / "nowhere")):
            self.assertIn("model files", tts.engine_status()["reason"])


@unittest.skipUnless(shutil.which("ffmpeg") or Path("/opt/homebrew/bin/ffmpeg").is_file(), "ffmpeg is needed for the mastering step")
class JobTests(TtsBase):
    def run_job(self, text: str, **kwargs):
        return tts.synthesize_document(self.source(text), source_name="story.txt", options=tts.Options(voice="anna"), **kwargs)

    def test_a_whole_job_writes_a_mastered_m4a_and_cleans_up(self) -> None:
        stages: list[str] = []
        result = self.run_job("ย่อหน้าแรกของเรื่อง\n\nย่อหน้าที่สองของเรื่อง", progress=lambda s, d, t, st: stages.append(s))
        output = Path(result["output_path"])
        self.assertEqual((output.suffix, output.parent), (".m4a", self.root / "out"))
        self.assertGreater(output.stat().st_size, 1000)
        self.assertEqual(result["pieces"], 2)
        # two 0.4 s pieces and the 420 ms paragraph pause between them
        self.assertAlmostEqual(result["duration_sec"], 0.4 + 0.42 + 0.4, delta=0.25)
        self.assertTrue(set(["read", "plan", "load", "synth", "master"]) <= set(stages))
        self.assertEqual([p for p in (self.root / "private").iterdir() if p.name.startswith("job-")], [])
        sent = self.requests()
        self.assertEqual([r["cmd"] for r in sent], ["load", "synth", "synth", "quit"])
        self.assertEqual(sent[0]["ref_text"], "ประโยคอ้างอิง")
        self.assertTrue(sent[0]["ref_wav"].endswith("anna/ref_voice.wav"))

    def test_english_pieces_are_flagged_thai_ones_are_not_and_digits_follow_the_language(self) -> None:
        self.run_job("ราคา 1,420 บาท\n\nIt costs 1,420 baht today.")
        synths = [r for r in self.requests() if r["cmd"] == "synth"]
        self.assertNotIn("language", synths[0])
        self.assertNotIn("1,420", synths[0]["text"])
        self.assertEqual(synths[1]["language"], "English")
        self.assertIn("1,420", synths[1]["text"])
        self.assertEqual((synths[0]["nfe"], synths[0]["cfg"], synths[0]["speed"]), (32, 2.0, 1.0))

    def test_markdown_is_cleaned_and_code_and_links_are_reported(self) -> None:
        text = "# หัวเรื่อง\n\nอ่านต่อที่ [ลิงก์](https://example.com) นะ\n\n```\nprint(1)\n```\n"
        result = tts.synthesize_document(self.source(text, "doc.md"), source_name="doc.md", options=tts.Options(voice="anna"))
        spoken = " ".join(r["text"] for r in self.requests() if r["cmd"] == "synth")
        self.assertNotIn("print", spoken)
        self.assertNotIn("https", spoken)
        self.assertNotIn("#", spoken)
        self.assertTrue(any("code block" in w for w in result["warnings"]), result["warnings"])

    def test_a_silent_piece_is_retried_once(self) -> None:
        with mock.patch.dict(os.environ, {"FAKE_TTS_SILENT_FIRST": "1"}):
            result = self.run_job("ประโยคเดียวที่ต้องอ่าน")
        self.assertEqual(result["stats"]["retried"], 1)
        self.assertEqual(len([r for r in self.requests() if r["cmd"] == "synth"]), 2)
        self.assertEqual(result["warnings"], [])

    def test_a_job_without_a_voice_or_with_a_bad_one_refuses_before_starting_a_worker(self) -> None:
        for voice in ("", "nobody"):
            with self.assertRaises(ValueError) as caught:
                tts.synthesize_document(self.source("ข้อความ"), source_name="a.txt", options=tts.Options(voice=voice))
            self.assertIn("Choose a voice", str(caught.exception))
        self.assertEqual(self.requests(), [])

    def test_a_file_with_nothing_to_read_is_refused_before_any_worker_starts(self) -> None:
        with self.assertRaises(ValueError) as caught:      # a Markdown file that is only a code block
            tts.synthesize_document(self.source("```\ncode only\n```\n", "code.md"), source_name="code.md", options=tts.Options(voice="anna"))
        self.assertIn("No speakable text", str(caught.exception))
        self.assertEqual(self.requests(), [])

    def test_cancel_stops_between_pieces_and_leaves_no_files_or_process(self) -> None:
        cancel = threading.Event()

        def progress(stage, done, total, stats):
            if stage == "synth" and done == 1:
                cancel.set()
        with self.assertRaises(tts.TtsCancelled):
            self.run_job("หนึ่ง\n\nสอง\n\nสาม", progress=progress, should_cancel=cancel.is_set)
        self.assertEqual(len([r for r in self.requests() if r["cmd"] == "synth"]), 1)
        self.assertEqual([p for p in (self.root / "private").iterdir() if p.name.startswith("job-")], [])
        self.assertEqual(list((self.root / "out").glob("*.m4a")), [])

    def test_pause_holds_the_job_before_the_next_piece_and_resume_continues_it(self) -> None:
        pause, seen = threading.Event(), []

        def progress(stage, done, total, stats):
            seen.append((stage, done))
            if stage == "synth" and done == 1:
                pause.set()
        outcome: dict = {}
        thread = threading.Thread(target=lambda: outcome.update(result=self.run_job("หนึ่ง\n\nสอง\n\nสาม", progress=progress, should_pause=pause.is_set)))
        thread.start()
        deadline = time.time() + 20
        while ("paused", 2) not in seen and time.time() < deadline:
            time.sleep(0.05)
        # pause was asked while piece 2 was starting: piece 2 finishes, the job then holds before piece 3
        self.assertIn(("paused", 2), seen)
        time.sleep(0.6)
        self.assertEqual(len([r for r in self.requests() if r["cmd"] == "synth"]), 2)   # held: piece 3 was not sent
        pause.clear()
        thread.join(60)
        self.assertFalse(thread.is_alive())
        self.assertEqual(outcome["result"]["pieces"], 3)
        self.assertEqual(len([r for r in self.requests() if r["cmd"] == "synth"]), 3)

    def test_cancel_while_paused_ends_the_job(self) -> None:
        pause, cancel = threading.Event(), threading.Event()
        pause.set()
        errors: list[BaseException] = []

        def go():
            try:
                self.run_job("หนึ่ง\n\nสอง", should_pause=pause.is_set, should_cancel=cancel.is_set)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        thread = threading.Thread(target=go)
        thread.start()
        time.sleep(0.5)
        cancel.set()
        thread.join(30)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(errors[0], tts.TtsCancelled)

    def test_listing_outputs_newest_first(self) -> None:
        self.run_job("เรื่องแรก")
        time.sleep(0.05)
        tts.synthesize_document(self.source("เรื่องสอง", "second.txt"), source_name="second.txt", options=tts.Options(voice="anna"))
        names = [row["name"] for row in tts.list_outputs()]
        self.assertEqual(names[:2], ["second.m4a", "story.m4a"])


if __name__ == "__main__":
    unittest.main()

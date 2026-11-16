"""Agent server: the /tts/* routes against the actual FastAPI app, with the fake voice worker (LLM-free, model-free)."""
from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import time
import threading
import unittest
import wave
from unittest import mock

os.environ.setdefault("AGENT_SERVER_TOKEN", "test-tts-server-token")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient

import agent_server as srv
import config
import text_to_speech as tts

FAKE_WORKER = Path(__file__).resolve().parent / "fake_tts_worker.py"


def write_wav(path: Path, seconds: float = 4.0) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(i / 12))) for i in range(int(24000 * seconds))))


@unittest.skipUnless(shutil.which("ffmpeg") or Path("/opt/homebrew/bin/ffmpeg").is_file(), "ffmpeg is needed for the mastering step")
class TtsRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="agent-th-tts-server-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        model = self.root / "model"
        model.mkdir()
        (model / "model.safetensors").write_bytes(b"x")
        voice = self.root / "out" / "voices" / "anna"
        voice.mkdir(parents=True)
        write_wav(voice / "ref_voice.wav")
        (voice / "ref_voice.txt").write_text("ประโยคอ้างอิง", encoding="utf-8")
        (self.root / "private").mkdir()
        self.log = self.root / "worker.jsonl"
        for patcher in (
            mock.patch.object(config, "TTS_PRIVATE_DIR", str(self.root / "private")),
            mock.patch.object(config, "TTS_OUTPUT_DIR", str(self.root / "out")),
            mock.patch.object(config, "TTS_VOICES_DIR", str(self.root / "out" / "voices")),
            mock.patch.object(config, "TTS_OMNIVOICE_DIR", str(model)),
            mock.patch.object(config, "TTS_OMNIVOICE_PYTHON", sys.executable),
            mock.patch.object(tts, "worker_path", lambda: FAKE_WORKER),
            mock.patch.object(srv, "_TTS_UPLOAD_DIR", self.root / "private" / "uploads"),
            mock.patch.dict(os.environ, {"FAKE_TTS_LOG": str(self.log)}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        srv._tts_state = None
        srv._tts_task = None
        srv._tts_control["pause"].clear()
        srv._tts_control["cancel"].clear()
        # TestClient without "with" gives every request its own event loop, which would drop the job task when the request
        # ends (the startup lifespan - model server reconcile, watchdogs - must not run here either): run the real job
        # coroutine on a loop of its own thread instead, so the control routes can act on it while it works.
        real_job = srv._run_tts_job

        def spawn(*args):
            threading.Thread(target=lambda: asyncio.run(real_job(*args)), daemon=True).start()
            return asyncio.sleep(0)
        patcher = mock.patch.object(srv, "_run_tts_job", spawn)
        patcher.start()
        self.addCleanup(patcher.stop)
        srv.api.dependency_overrides[srv._require_token] = lambda: None
        self.addCleanup(srv.api.dependency_overrides.pop, srv._require_token, None)
        self.client = TestClient(srv.api)

    def start(self, text: str = "ย่อหน้าแรก\n\nย่อหน้าสอง", name: str = "story.txt", voice: str = "anna", speed: str = "1.0"):
        return self.client.post("/tts/start", content=text.encode("utf-8"),
                                headers={"x-file-name": name, "x-voice": voice, "x-speed": speed, "content-type": "text/plain"})

    def wait_status(self, wanted: set[str], seconds: float = 30.0) -> dict:
        deadline = time.time() + seconds
        job: dict = {}
        while time.time() < deadline:
            job = self.client.get("/tts/status").json()["job"]
            if job["status"] in wanted:
                return job
            time.sleep(0.05)
        self.fail(f"status never reached {wanted}; last = {job}")

    def test_voices_route_reports_the_engine_the_voices_and_where_to_put_more(self) -> None:
        body = self.client.get("/tts/voices").json()
        self.assertTrue(body["ok"] and body["engine"]["ready"])
        self.assertEqual([v["id"] for v in body["voices"]], ["anna"])
        self.assertTrue(body["voices_dir"].endswith("voices"))

    def test_a_job_runs_to_done_and_the_output_is_listed(self) -> None:
        response = self.start()
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["already_running"])
        job = self.wait_status({"done", "error"})
        self.assertEqual(job["status"], "done", job)
        self.assertTrue(Path(job["output_path"]).is_file())
        self.assertEqual((job["done"], job["total"]), (2, 2))
        self.assertEqual(job["stats"]["voice"], "anna")
        outputs = self.client.get("/tts/voices").json()["outputs"]
        self.assertEqual(outputs[0]["name"], "story.m4a")
        self.assertEqual(list((self.root / "private" / "uploads").glob("th-tts-*")), [])   # the staged upload is removed

    def test_bad_requests_are_refused_with_a_reason(self) -> None:
        self.assertEqual(self.start(name="notes.pdf").status_code, 400)
        for kwargs, fragment in (({"voice": "nobody"}, "เลือกเสียง"), ({"voice": ""}, "เลือกเสียง"), ({"speed": "fast"}, "ความเร็ว")):
            response = self.start(**kwargs)
            self.assertEqual(response.status_code, 400, kwargs)
            self.assertIn(fragment, response.json()["error"])
        empty = self.start(text="")
        self.assertEqual((empty.status_code, empty.json()["error"]), (400, "ไฟล์ว่าง"))

    def test_an_engine_that_is_not_installed_is_refused_before_anything_is_staged(self) -> None:
        with mock.patch.object(config, "TTS_OMNIVOICE_PYTHON", str(self.root / "nope")):
            response = self.start()
        self.assertEqual(response.status_code, 400)
        self.assertIn("conda env", response.json()["error"])
        self.assertFalse((self.root / "private" / "uploads").exists())

    def test_pause_resume_and_cancel_through_the_routes(self) -> None:
        # slow the voice so there is time to act: the fake sleeps per piece
        slow = self.root / "slow_worker.py"
        slow.write_text(FAKE_WORKER.read_text(encoding="utf-8").replace('reply({"ok": True, "duration": seconds', 'import time; time.sleep(0.4); reply({"ok": True, "duration": seconds'), encoding="utf-8")
        with mock.patch.object(tts, "worker_path", lambda: slow):
            self.start(text="หนึ่ง\n\nสอง\n\nสาม\n\nสี่")
            self.wait_status({"running"})
            self.assertEqual(self.client.post("/tts/pause").status_code, 200)
            job = self.wait_status({"paused"})
            held = job["done"]
            time.sleep(0.8)
            self.assertEqual(self.client.get("/tts/status").json()["job"]["done"], held)   # nothing moved while paused
            self.assertEqual(self.client.post("/tts/resume").status_code, 200)
            self.wait_status({"running", "done"})
            self.assertEqual(self.client.post("/tts/cancel").status_code, 200)
            job = self.wait_status({"cancelled", "done"})
        self.assertEqual(job["status"], "cancelled", job)
        self.assertEqual(self.client.post("/tts/pause").status_code, 409)       # nothing active any more
        self.assertEqual(list((self.root / "private").glob("job-*")), [])
        self.assertEqual(list((self.root / "out").glob("*.m4a")), [])

    def test_a_second_start_while_one_runs_returns_the_running_job(self) -> None:
        slow = self.root / "slow_worker.py"
        slow.write_text(FAKE_WORKER.read_text(encoding="utf-8").replace('reply({"ok": True, "duration": seconds', 'import time; time.sleep(0.6); reply({"ok": True, "duration": seconds'), encoding="utf-8")
        with mock.patch.object(tts, "worker_path", lambda: slow):
            first = self.start(text="หนึ่ง\n\nสอง")
            second = self.start(text="อีกไฟล์")
            self.assertTrue(second.json()["already_running"])
            self.assertEqual(second.json()["job"]["job_id"], first.json()["job"]["job_id"])
            self.wait_status({"done", "error"})

    def test_an_unknown_job_id_is_a_404_and_a_failing_job_reports_the_reason(self) -> None:
        self.assertEqual(self.client.get("/tts/status", params={"job_id": "nope"}).status_code, 404)
        self.start(text="```\ncode only\n```\n", name="code.md")
        job = self.wait_status({"error", "done"})
        self.assertEqual(job["status"], "error")
        self.assertIn("No speakable text", job["error"])


if __name__ == "__main__":
    unittest.main()

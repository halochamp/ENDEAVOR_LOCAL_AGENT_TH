"""Stands in for scripts/tts_omnivoice_worker.py in the tests: same JSON-lines protocol, a tone instead of a voice.
FAKE_TTS_LOG = a file that receives every request as one JSON line; FAKE_TTS_SILENT_FIRST = 1 -> the first synth of each text is silent."""
import json
import math
import os
import sys
import wave

log_path = os.environ.get("FAKE_TTS_LOG")
silent_first = os.environ.get("FAKE_TTS_SILENT_FIRST") == "1"
seen: set[str] = set()


def reply(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    request = json.loads(line)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")
    cmd = request.get("cmd")
    if cmd == "quit":
        reply({"ok": True})
        break
    if cmd == "load":
        reply({"ok": True, "device": "fake"})
    elif cmd == "synth":
        text = request["text"]
        silent = silent_first and text not in seen
        seen.add(text)
        amplitude = 0 if silent else 9000
        seconds = 0.4
        with wave.open(request["out"], "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(24000)
            out.writeframes(b"".join(int(amplitude * math.sin(i / 12)).to_bytes(2, "little", signed=True) for i in range(int(24000 * seconds))))
        reply({"ok": True, "duration": seconds, "peak": amplitude})
    else:
        reply({"ok": False, "error": f"unknown_cmd:{cmd}"})

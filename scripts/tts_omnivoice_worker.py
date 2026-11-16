# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""OmniVoice (k2-fsa) voice worker for Agent TH's Text to Speech job (JSON lines on stdin/stdout).

Run ONLY in the ``omnivoice`` conda env, with the local model directory as argv[1]:

    omnivoice/bin/python scripts/tts_omnivoice_worker.py /path/to/OmniVoice/model

Own process, no TCP port, the model leaves RAM when the job ends, stdout is the protocol channel only.
Splitting the text and splicing the pieces happen in the agent process (tts_split.py); this worker only loads the model
and renders one piece per request.
The weights are CC-BY-NC: personal and non-commercial use only.

Commands (one JSON object per line, one JSON reply per line):
  {"cmd": "load", "ref_wav": path, "ref_text": str}                     (reference clip is required)
  {"cmd": "synth", "text": str, "out": path, "nfe": int, "cfg": float, "speed": float, "language": "English"?}
  {"cmd": "quit"}
``nfe`` is OmniVoice's num_step and ``cfg`` its guidance_scale, so the job sends the preset's keys unchanged.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

_SAMPLE_RATE = 24000


def _reply(channel, payload: dict) -> None:
    channel.write(json.dumps(payload, ensure_ascii=False) + "\n")
    channel.flush()


def _release_gpu_cache() -> None:
    """Low-RAM: give the MPS allocator's buffers back between pieces (memory hygiene between pieces)."""
    try:
        import gc

        import torch

        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001 - memory hygiene must never fail a synthesis
        pass


def main() -> int:
    if len(sys.argv) != 2 or not os.path.isfile(os.path.join(sys.argv[1], "model.safetensors")):
        sys.stderr.write("usage: tts_omnivoice_worker.py <OmniVoice model dir containing model.safetensors>\n")
        return 2
    channel = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    sys.stdout = sys.stderr
    model_dir = os.path.abspath(sys.argv[1])
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the weights are local; never reach for the network

    state: dict = {"model": None, "prompt": None}

    def load(req: dict) -> dict:
        ref_wav = str(req.get("ref_wav") or "")
        ref_text = str(req.get("ref_text") or "").strip()
        if not ref_text or not os.path.isfile(ref_wav):
            return {"ok": False, "error": "reference_clip_missing"}
        import torch
        from omnivoice import OmniVoice

        if state["model"] is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            dtype = torch.float16 if device == "mps" else torch.float32
            state["model"] = OmniVoice.from_pretrained(model_dir, device_map=device, dtype=dtype)
            state["device"] = device
        # encoded once: every piece of the job is cloned from the same prompt, so the voice cannot drift
        state["prompt"] = state["model"].create_voice_clone_prompt(ref_audio=ref_wav, ref_text=ref_text)
        return {"ok": True, "device": state["device"]}

    def synth(req: dict) -> dict:
        import numpy as np
        import soundfile as sf

        if state["model"] is None or state["prompt"] is None:
            return {"ok": False, "error": "omnivoice_not_loaded"}
        speed = float(req.get("speed", 1.0))
        audio = state["model"].generate(
            text=str(req["text"]), voice_clone_prompt=state["prompt"],
            num_step=int(req["nfe"]), guidance_scale=float(req.get("cfg", 2.0)),
            speed=None if abs(speed - 1.0) < 1e-6 else speed,  # None = the model's own duration estimate
            language=req.get("language") or None,  # "English" for an English piece; None = language-agnostic (Thai/mixed)
        )
        pcm = np.asarray(audio[0], dtype=np.float32).reshape(-1)
        out = str(req["out"])
        tmp = out + ".part"
        sf.write(tmp, pcm, _SAMPLE_RATE, format="WAV", subtype="PCM_16")
        os.replace(tmp, out)
        peak = int(float(np.max(np.abs(pcm))) * 32767) if pcm.size else 0
        _release_gpu_cache()
        return {"ok": True, "duration": round(len(pcm) / _SAMPLE_RATE, 3), "peak": peak}

    handlers = {"load": load, "synth": synth}
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "quit":
                _reply(channel, {"ok": True})
                return 0
            handler = handlers.get(cmd)
            result = handler(req) if handler else {"ok": False, "error": f"unknown_cmd:{cmd}"}
        except Exception as exc:  # noqa: BLE001 - protocol worker must always answer
            traceback.print_exc()
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:400]}
        _reply(channel, result)
    return 0  # stdin closed: the parent is gone, release the model


if __name__ == "__main__":
    raise SystemExit(main())

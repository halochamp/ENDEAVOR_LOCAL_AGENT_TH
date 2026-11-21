#!/usr/bin/env python3
"""The Model page of the Agent TH desktop app: which of the agent's tested models are downloaded, and downloading the others.

    python tools/models.py list [--check]       one JSON line: models (grouped by target repo) with their slots (native AR only);
                                                --check also asks Hugging Face for unknown sizes (page open / Refresh)
    python tools/models.py download <repo_id>   starts a background download of the pinned commit
    python tools/models.py cancel <repo_id>

Only the slots in the agent's model_slots.json are offered (the tested ones; Agent TH has no DFlash / MTP accelerator). "Downloaded" is
decided by the agent's own model_slots.preflight_slot (pinned revision present, weights complete, refs/main not moved), so
the page and the agent can never disagree. Downloads use the pinned commit (`snapshot_download(revision=<commit>)`), which
never moves refs/main, run one at a time, outlive the app, and resume where they stopped.
Run with the agent's `mlx` env python (huggingface_hub). State: ~/.agent_th_ui/downloads/.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

AGENT_DIR = Path(os.environ.get("AGENT_TH_DIR") or Path(__file__).resolve().parent.parent.parent)   # tools/ -> UI folder -> repo root
sys.path.insert(0, str(AGENT_DIR))
import model_slots  # noqa: E402  (the agent's registry and preflight)

STATE = Path(os.environ.get("AGENT_TH_UI_STATE") or Path.home() / ".agent_th_ui") / "downloads"
GB = 1e9
# Peak phys_footprint (GiB) of the model server for a native-AR slot, measured on an M5 48 GB: each slot loaded fresh, one
# 4.6k-token prompt + 300 tokens out. Long chats need more. A model not measured yet shows "not measured", never a guess
# (a figure measured with a speed-up (DFlash) would not apply to Agent TH's native AR).
RAM_GB = {
    "mlx-community/Qwen3.8-27B-4bit": 31.8,               # dense 27B: prefill peaks above the 35B MoE
    "Qwen/Qwen3-14B-MLX-4bit": 10.2,
    "mlx-community/Qwen3.5-9B-4bit": 8.2,
    "mlx-community/Qwen3.5-2B-OptiQ-4bit": 4.3,
}
RAM_NOTE = "วัดที่ prompt 4.6k token · บทสนทนายาวใช้มากกว่านี้"
RAM_NOTES: dict[str, str] = {}
VISION = {"Qwen3 14B"}  # labels of text-only families (the rest read images)


def safe(repo: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", repo)


def cache_root() -> Path:
    return model_slots._default_hf_cache_root()


def repo_bytes(repo: str, revision: str) -> int:
    """Bytes on disk of one pinned snapshot (blobs it points at, plus .incomplete parts while downloading)."""
    repo_dir = model_slots._repo_cache_dir(cache_root(), repo)
    snap = repo_dir / "snapshots" / revision
    total = 0
    seen = set()
    if snap.is_dir():
        for f in snap.rglob("*"):
            try:
                real = f.resolve()
                if real.is_file() and real not in seen:
                    seen.add(real)
                    total += real.stat().st_size
            except OSError:
                pass
    blobs = repo_dir / "blobs"
    if blobs.is_dir():
        total += sum(p.stat().st_size for p in blobs.glob("*.incomplete") if p.is_file())
    return total


_NETWORK = {"allowed": False, "down": False}   # set by `list --check` (the page opening, or the user's Refresh)


def expected_bytes(repo: str, revision: str) -> int | None:
    """Total size of the pinned snapshot. A known size is remembered for good; an unknown one is asked from Hugging Face
    only by `list --check` (ask when the page opens or Refresh is pressed, never on a timer), with one
    plain 5 s request; after a failure the other repos are not asked in that same run. huggingface_hub's own client was
    retrying for over 5 minutes per call when offline."""
    STATE.mkdir(parents=True, exist_ok=True)
    cache = STATE / "sizes.json"
    try:
        sizes = json.loads(cache.read_text())
    except (OSError, ValueError):
        sizes = {}
    key = f"{repo}@{revision}"
    if key in sizes:
        return sizes[key]
    if not _NETWORK["allowed"] or _NETWORK["down"]:
        return None
    import urllib.request
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    try:
        with urllib.request.urlopen(f"{endpoint}/api/models/{repo}/revision/{revision}?blobs=true", timeout=5) as r:
            info = json.loads(r.read())
        total = sum(int(s.get("size") or 0) for s in info.get("siblings") or [])
    except Exception:  # offline, rate limited, gated: the page shows the size after a later Refresh
        _NETWORK["down"] = True
        return None
    sizes[key] = total
    cache.write_text(json.dumps(sizes))
    return total


def pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def read_state(repo: str) -> dict:
    try:
        d = json.loads((STATE / f"{safe(repo)}.json").read_text())
    except (OSError, ValueError):
        return {"state": "none"}
    if d.get("state") in ("queued", "downloading") and not pid_alive(d.get("pid")) and time.time() - d.get("started", 0) > 15:
        d["state"] = "stopped"
    return d


def write_state(repo: str, **fields) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    path = STATE / f"{safe(repo)}.json"
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        d = {}
    d.update(fields)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(d))
    tmp.replace(path)


def family(label: str) -> str:
    return label.split(" · ")[0].strip()


def models() -> list[dict]:
    reg = model_slots.current_registry()
    groups: dict[str, dict] = {}
    for slot in reg.slots:
        g = groups.setdefault(slot.target.repo_id, {
            "id": slot.target.repo_id, "name": family(slot.label), "revision": slot.target.revision,
            "vision": family(slot.label) not in VISION, "ram_gb": RAM_GB.get(slot.target.repo_id),
            "ram_note": RAM_NOTES.get(slot.target.repo_id, RAM_NOTE) if slot.target.repo_id in RAM_GB else "",
            "repos": {slot.target.repo_id: slot.target.revision}, "variants": [],
        })
        if slot.draft:
            g["repos"][slot.draft.repo_id] = slot.draft.revision
        try:
            model_slots.preflight_slot(slot)
            ready, reason = True, ""
        except Exception as e:  # ModelSlotError and anything unreadable: not usable yet
            ready, reason = False, str(e)[:200]
        g["variants"].append({"slot_id": slot.slot_id, "label": slot.label, "acceleration": slot.acceleration_mode,
                              "selectable": slot.selectable, "ready": ready, "reason": reason})
    out = []
    for g in groups.values():
        done = sum(repo_bytes(r, rev) for r, rev in g["repos"].items())
        sizes = [expected_bytes(r, rev) if not all(v["ready"] for v in g["variants"]) else None for r, rev in g["repos"].items()]
        total = done if all(v["ready"] for v in g["variants"]) else (sum(sizes) if all(s is not None for s in sizes) else None)
        dl = read_state(g["id"])
        target_ready = any(v["ready"] for v in g["variants"])
        all_ready = all(v["ready"] for v in g["variants"])
        state = "installed" if all_ready else dl["state"] if dl["state"] in ("queued", "downloading") else ("partial" if target_ready else "missing")
        g.update(state=state, size_gb=round((total or done) / GB, 1) if (total or done) else None, done_gb=round(done / GB, 1),
                 percent=round(100 * done / total, 1) if total else (100.0 if all_ready else None),
                 download={k: dl.get(k) for k in ("state", "error", "started", "finished")})
        g["repos"] = [{"repo_id": r, "revision": rev} for r, rev in g["repos"].items()]
        out.append(g)
    return out


def machine() -> dict:
    ram = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip() or 0)
    st = os.statvfs(str(Path.home()))
    return {"ram_gb": round(ram / 2**30), "disk_free_gb": round(st.f_bavail * st.f_frsize / GB, 1),
            "chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()}


def start_download(repo: str) -> dict:
    group = next((m for m in models() if m["id"] == repo), None)
    if group is None:
        return {"ok": False, "error": f"not one of the tested models: {repo}"}
    if group["state"] == "installed":
        return {"ok": True, "state": "installed"}
    if group["state"] in ("queued", "downloading"):
        return {"ok": True, "state": group["state"], "note": "already downloading"}
    STATE.mkdir(parents=True, exist_ok=True)
    log = STATE / f"{safe(repo)}.log"
    write_state(repo, state="queued", pid=None, started=time.time(), finished=None, error=None)
    with open(log, "ab") as out:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_worker", repo], stdin=subprocess.DEVNULL,
                                stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    write_state(repo, pid=proc.pid)
    return {"ok": True, "state": "queued"}


def cancel(repo: str) -> dict:
    dl = read_state(repo)
    if dl["state"] not in ("queued", "downloading"):
        return {"ok": True, "state": dl["state"], "note": "nothing to cancel"}
    try:
        os.killpg(int(dl["pid"]), signal.SIGTERM)
    except (OSError, TypeError, ValueError):
        pass
    write_state(repo, state="cancelled", finished=time.time())
    return {"ok": True, "state": "cancelled", "note": "files so far are kept: download again to resume"}


def worker(repo: str) -> int:
    write_state(repo, state="queued", pid=os.getpid())
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    with open(STATE / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # one at a time: the network is the limit
        write_state(repo, state="downloading", pid=os.getpid())
        group = next(m for m in models() if m["id"] == repo)
        from huggingface_hub import snapshot_download
        try:
            for r in group["repos"]:
                print(time.strftime("[%F %T] ") + f"download {r['repo_id']}@{r['revision']}", flush=True)
                snapshot_download(r["repo_id"], revision=r["revision"])  # a commit pin: refs/main is never moved
        except Exception as e:
            write_state(repo, state="error", finished=time.time(), error=str(e)[:300])
            return 1
        after = next(m for m in models() if m["id"] == repo)
        if after["state"] == "installed":
            write_state(repo, state="done", finished=time.time(), error=None)
        else:
            bad = [v["reason"] for v in after["variants"] if not v["ready"]]
            write_state(repo, state="error", finished=time.time(), error=(bad[0] if bad else "not complete"))
    return 0


def main() -> int:
    args = sys.argv[1:] or ["list"]
    if args[0] == "_worker":
        return worker(args[1])
    if args[0] == "list":
        _NETWORK["allowed"] = "--check" in args[1:]
        out = {"ok": True, "machine": machine(), "models": models()}
    elif args[0] in ("download", "cancel") and len(args) == 2:
        out = start_download(args[1]) if args[0] == "download" else cancel(args[1])
    else:
        out = {"ok": False, "error": "usage: models.py list | download <repo_id> | cancel <repo_id>"}
    print(json.dumps(out, ensure_ascii=False))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())

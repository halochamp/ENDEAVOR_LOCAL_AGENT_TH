"""CPU-only regressions for the direct-vision publication/lifecycle boundaries.

Run from the target repository with the project's mlx interpreter:
    /opt/homebrew/anaconda3/envs/mlx/bin/python tests/test_vision_lifecycle.py

The probes use synthetic pixels and mocked sensors/model calls. They never start
or contact an MLX server and never use a private project path.
"""
from __future__ import annotations

import importlib
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_image_regression() -> None:
    ri = importlib.import_module("tools.read_image")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fixture:
        image_path = Path(fixture.name)
    Image.new("RGB", (48, 32), (20, 40, 60)).save(image_path)
    source = str(image_path)
    ocr_calls: list[str] = []
    qr_calls: list[str] = []

    def fake_ocr(path: str) -> list[dict]:
        ocr_calls.append(path)
        return [{"text": "fixture label", "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.1}]

    def invoke(**kwargs) -> str:
        payload = {"source": source}
        payload.update(kwargs)
        return ri.read_image.invoke(payload)

    patched = {
        "resolve_read_path": lambda _src: source,
        "_maybe_convert_heic": lambda path: path,
        "_exif_normalize": lambda path: path,
        "_encode_frame_data_url": lambda *_args, **_kwargs: "data:image/png;base64,ORIGINAL",
        "_ocr_layout": fake_ocr,
        "_decode_qr": lambda path: qr_calls.append(path) or [],
        "_enhance_for_ocr": lambda path: path,
        "_reconstruct_table": lambda _boxes: None,
        "_reconstruct_ranked_columns": lambda _boxes: None,
        "_looks_like_range_chart": lambda _boxes: False,
        "_zoom_region_hint": lambda _boxes: "",
        "phase": lambda *_args, **_kwargs: None,
        "progress": lambda *_args, **_kwargs: None,
    }
    try:
        with patch.multiple(ri, **patched):
            for _ in range(20):
                ri.begin_image_turn()
                ri.reset_read_guards()
                ocr_calls.clear()
                qr_calls.clear()

                first = invoke()
                immediate = invoke(detail="text")
                assert "deferred" in immediate.lower(), immediate
                assert "no sensor ran" in immediate.lower(), immediate
                assert not ocr_calls and not qr_calls, (ocr_calls, qr_calls)
                assert source in ri._OVERVIEW_PENDING
                assert source not in ri._OVERVIEW_SEEN

                assert ri.active_turn_images() == [
                    "data:image/png;base64,ORIGINAL"
                ]
                assert source in ri._OVERVIEW_SEEN
                assert source not in ri._OVERVIEW_PENDING
                after = invoke(detail="text")
                assert ocr_calls and qr_calls, (ocr_calls, qr_calls)
                assert "OCR ASSIST" in after, after

                ri.begin_image_turn()
                ri.reset_read_guards()
                ocr_calls.clear()
                qr_calls.clear()
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [
                        pool.submit(invoke),
                        pool.submit(invoke, detail="text"),
                    ]
                    same_batch = [future.result() for future in futures]
                assert not ocr_calls and not qr_calls, (ocr_calls, qr_calls, same_batch)
                assert source in ri._OVERVIEW_PENDING
                assert source not in ri._OVERVIEW_SEEN
                ri.active_turn_images()
                invoke(detail="text")
                assert ocr_calls and qr_calls, (ocr_calls, qr_calls)

                # The first result remains an overview-only transaction; this
                # guards against a test passing merely because detail ran first.
                assert any("original image queued" in result for result in same_batch)
                assert any("deferred" in result.lower() for result in same_batch)
                _ = first
        print("VISION_READ_PUBLICATION_OK trials=20")
    finally:
        ri.begin_image_turn()
        ri.reset_read_guards()
        image_path.unlink(missing_ok=True)


def _computer_state_snapshot(ri, computer):
    return {
        "read_pending": list(ri._PENDING_IMAGES),
        "read_active": list(ri._ACTIVE_TURN_IMAGES),
        "read_pending_sources": set(ri._OVERVIEW_PENDING),
        "read_seen": set(ri._OVERVIEW_SEEN),
        "read_guards": (dict(ri._READ_SEEN), dict(ri._READ_ATTEMPTS)),
        "computer_pending": list(computer._COMPUTER_PENDING_IMAGES),
        "computer_active": list(computer._COMPUTER_ACTIVE_IMAGES),
        "computer_guards": (
            computer._ACTION_ATTEMPTS[0],
            computer._LAST_SIGNATURE[0],
            computer._ACTION_MAX_OVERRIDE[0],
            computer._DESTRUCTIVE_GUARD[0],
        ),
    }


def _warm_lifecycle_regression() -> None:
    graph = importlib.import_module("graph")
    react = importlib.import_module("react")
    ri = importlib.import_module("tools.read_image")
    computer = importlib.import_module("tools.computer_use")
    original_react = graph._REACT

    class BarrierReact:
        def __init__(self, entered: threading.Event, release: threading.Event):
            self.entered = entered
            self.release = release

        def invoke(self, payload, config=None):
            # This exercises the actual dynamic-prompt publication helper. The
            # warm context must prevent it from draining a user's pending image.
            react.prepare_turn_vision_messages(list(payload["messages"]))
            self.entered.set()
            assert self.release.wait(5), "warm barrier timed out"
            return {"messages": list(payload["messages"]) + [AIMessage(content="warm")]}

    try:
        for warm_id in (graph._STARTUP_WARM_THREAD_ID, graph._REWARM_THREAD_ID):
            ri.begin_image_turn()
            ri.reset_read_guards()
            computer.reset_computer_guards()
            ri._PENDING_IMAGES[:] = ["data:image/png;base64,READ_PENDING"]
            ri._ACTIVE_TURN_IMAGES[:] = ["data:image/png;base64,READ_ACTIVE"]
            ri._OVERVIEW_PENDING.add("fixture.png")
            ri._OVERVIEW_SEEN.add("exposed.png")
            ri._READ_SEEN[("fixture.png", "overview")] = "cached"
            ri._READ_ATTEMPTS["fixture.png"] = 2
            computer._COMPUTER_PENDING_IMAGES[:] = [
                "data:image/png;base64,COMPUTER_PENDING"
            ]
            computer._COMPUTER_ACTIVE_IMAGES[:] = [
                "data:image/png;base64,COMPUTER_ACTIVE"
            ]
            computer._ACTION_ATTEMPTS[0] = 3
            computer._LAST_SIGNATURE[0] = "user-action"
            computer._ACTION_MAX_OVERRIDE[0] = 2
            computer._DESTRUCTIVE_GUARD[0] = True
            before = _computer_state_snapshot(ri, computer)

            entered = threading.Event()
            release = threading.Event()
            graph._REACT = BarrierReact(entered, release)
            state = {"messages": [HumanMessage(content="cache seed")]}
            config = {"configurable": {"thread_id": warm_id}}
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(graph.react_node, state, config)
                assert entered.wait(5), f"{warm_id}: warm invoke did not start"
                assert _computer_state_snapshot(ri, computer) == before
                release.set()
                result = future.result(timeout=5)
            assert _computer_state_snapshot(ri, computer) == before
            assert result["messages"][-1].content == "warm"
            print(f"WARM_LIFECYCLE_OK {warm_id}")
    finally:
        graph._REACT = original_react
        ri.begin_image_turn()
        ri.reset_read_guards()
        computer.end_computer_turn()


if __name__ == "__main__":
    _read_image_regression()
    _warm_lifecycle_regression()
    print("VISION_LIFECYCLE_OK")

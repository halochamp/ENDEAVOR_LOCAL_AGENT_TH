# ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
# License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
# Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

"""Small shared-worker primitive for latency-bounded tool fan-out.

ThreadPoolExecutor limits running workers but its queue is unbounded. Tools that
return on a deadline while workers are still stuck must also bound submissions,
otherwise repeated timeouts retain an unlimited number of threads or queued jobs.
"""
from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, TypeVar

T = TypeVar("T")


class BoundedExecutor:
    def __init__(self, max_workers: int, *, thread_name_prefix: str) -> None:
        workers = max(1, int(max_workers))
        self._pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix=thread_name_prefix,
        )
        # At most one accepted task per worker. A timed-out prior batch therefore
        # cannot grow an unbounded executor queue behind stuck work.
        self._slots = threading.BoundedSemaphore(workers)

    def submit(self, fn: Callable[..., T], /, *args, **kwargs) -> Future[T] | None:
        if not self._slots.acquire(blocking=False):
            return None
        try:
            future = self._pool.submit(fn, *args, **kwargs)
        except Exception:
            self._slots.release()
            raise
        # Futures invoke callbacks for success, failure, and cancellation. Releasing
        # here also covers a queued future canceled before its wrapper ever starts.
        future.add_done_callback(lambda _future: self._slots.release())
        return future

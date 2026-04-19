"""
progress.py — Thread-Safe Progress Tracking
============================================

Provides a lightweight callback protocol and concrete implementations for
reporting simulation progress back to the UI layer.

Design
------
* **Protocol-based**: the engine accepts any callable matching the signature
  ``(percent: float, message: str) -> None``.  The UI layer can pass a Dash
  callback, a ``tqdm`` bar, or any other sink.
* **Thread-safe**: ``ProgressTracker`` uses a ``threading.Lock`` to protect
  shared counters when updated from worker threads inside
  ``concurrent.futures`` callables.
* **Multiprocessing-safe**: ``MultiprocessingProgressProxy`` uses a
  ``multiprocessing.Queue`` so worker *processes* (not just threads) can
  send updates back to the main process.

Typical use (synchronous / threaded)
-------------------------------------
>>> def my_callback(pct, msg):
...     print(f"[{pct:.1f}%] {msg}")
>>> tracker = ProgressTracker(total=500, callback=my_callback)
>>> tracker.update(50, "Monte Carlo batch 1/10 done")

Typical use (multiprocessing)
-------------------------------
>>> import multiprocessing as mp
>>> q = mp.Queue()
>>> proxy = MultiprocessingProgressProxy(q)
>>> # pass proxy.update as callback to worker processes
>>> listener = ProgressListener(q, total=500, callback=my_callback)
>>> listener.start()   # starts background thread that drains the queue
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

# ProgressCallback signature: (percent: float, message: str) -> None
ProgressCallback = Callable[[float, str], None]


# ---------------------------------------------------------------------------
# Thread-safe tracker (used with concurrent.futures / threading)
# ---------------------------------------------------------------------------


class ProgressTracker:
    """Thread-safe progress counter that fires a callback on each update.

    Parameters
    ----------
    total : int
        Total number of units of work (e.g. Monte Carlo iterations).
    callback : ProgressCallback
        Called with ``(percent, message)`` after every ``update`` call.
        The callback is invoked from whichever thread calls ``update``,
        so it must be thread-safe itself (or the caller must ensure
        single-threaded access).
    initial : int, optional
        Starting count (default 0).
    """

    def __init__(
        self,
        total: int,
        callback: ProgressCallback,
        initial: int = 0,
    ) -> None:
        if total <= 0:
            raise ValueError(f"total must be > 0; got {total}.")
        self._total = total
        self._done = initial
        self._callback = callback
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, n: int = 1, message: str = "") -> None:
        """Advance the counter by *n* and fire the callback.

        Parameters
        ----------
        n : int, optional
            Number of work units completed in this update.  Default 1.
        message : str, optional
            Human-readable status string forwarded to the callback.
        """
        with self._lock:
            self._done = min(self._done + n, self._total)
            pct = 100.0 * self._done / self._total
        self._callback(pct, message)

    def set_done(self, done: int, message: str = "") -> None:
        """Set the absolute counter and fire the callback."""
        with self._lock:
            self._done = min(max(done, 0), self._total)
            pct = 100.0 * self._done / self._total
        self._callback(pct, message)

    @property
    def percent(self) -> float:
        """Current completion percentage [0.0, 100.0]."""
        with self._lock:
            return 100.0 * self._done / self._total

    @property
    def done(self) -> int:
        """Current count of completed units."""
        with self._lock:
            return self._done

    def reset(self) -> None:
        """Reset the counter to zero (does NOT fire the callback)."""
        with self._lock:
            self._done = 0


# ---------------------------------------------------------------------------
# Multiprocessing progress proxy + listener
# ---------------------------------------------------------------------------


class MultiprocessingProgressProxy:
    """Sends progress updates from a worker *process* to a ``multiprocessing.Queue``.

    This object is picklable and safe to pass to subprocess workers.  Each
    call to ``update`` puts a ``(n, message)`` tuple on the queue.

    The main process should run a ``ProgressListener`` to drain the queue and
    forward updates to the real callback.

    Parameters
    ----------
    queue : multiprocessing.Queue
        Shared queue between the main process and workers.
    """

    def __init__(self, queue) -> None:
        self._q = queue

    def update(self, n: int = 1, message: str = "") -> None:
        """Put ``(n, message)`` on the queue.  Non-blocking best-effort."""
        try:
            self._q.put_nowait((n, message))
        except Exception:
            pass  # never crash a worker because of a progress queue failure


class ProgressListener:
    """Drains a ``multiprocessing.Queue`` in a daemon thread and fires a callback.

    Parameters
    ----------
    queue : multiprocessing.Queue
        Same queue passed to ``MultiprocessingProgressProxy`` objects.
    total : int
        Total units of work (forwarded to an internal ``ProgressTracker``).
    callback : ProgressCallback
        Called on the listener thread with ``(percent, message)``.
    """

    _SENTINEL = None  # put None on queue to stop the listener

    def __init__(self, queue, total: int, callback: ProgressCallback) -> None:
        self._q = queue
        self._tracker = ProgressTracker(total=total, callback=callback)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the background listener thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the listener to stop and wait for the thread to finish."""
        self._q.put(self._SENTINEL)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            item = self._q.get()
            if item is None:
                break
            n, message = item
            self._tracker.update(n, message)


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------


def null_callback(pct: float, message: str) -> None:  # noqa: D401
    """A no-op progress callback (discard all updates)."""
    pass


def print_callback(pct: float, message: str) -> None:  # noqa: D401
    """A simple stdout progress callback (useful for CLI tools)."""
    bar_len = 30
    filled = int(bar_len * pct / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    suffix = f" {message}" if message else ""
    print(f"\r[{bar}] {pct:5.1f}%{suffix}", end="", flush=True)
    if pct >= 100.0:
        print()  # newline on completion


def make_tqdm_callback(
    desc: str = "Cascade",
    total: int = 100,
) -> ProgressCallback:
    """Create a callback that updates a tqdm progress bar.

    Requires ``tqdm`` to be installed (``pip install tqdm``).

    Parameters
    ----------
    desc : str, optional
        Label shown next to the progress bar.
    total : int, optional
        Maximum value (default 100, representing percentage points).

    Returns
    -------
    ProgressCallback
        Call this with ``(pct, message)`` to advance the bar.
    """
    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise ImportError(
            "tqdm is required for make_tqdm_callback. "
            "Install it with: pip install tqdm"
        ) from exc

    pbar = tqdm(total=total, desc=desc, unit="%")
    last_pct = [0.0]

    def _cb(pct: float, message: str) -> None:
        delta = pct - last_pct[0]
        if delta > 0:
            pbar.update(delta)
            last_pct[0] = pct
        if message:
            pbar.set_postfix_str(message)
        if pct >= 100.0:
            pbar.close()

    return _cb

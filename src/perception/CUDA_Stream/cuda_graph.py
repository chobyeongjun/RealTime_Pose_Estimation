"""CUDA Graph capture for fixed-shape infer + preproc.

On Orin NX the CPU launch cost of hundreds of tiny kernels dominates
small-batch inference. Once shapes are fixed, we capture the
preproc+infer sub-graph and replay it with a single launch.

If capture fails (dynamic shape, disallowed op, etc.) we fall back to
regular eager execution without affecting correctness.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Callable, Optional

import torch

LOGGER = logging.getLogger(__name__)


class GraphedStep:
    """Wraps a callable into a CUDA Graph on a specific stream."""

    def __init__(
        self,
        stream: torch.cuda.Stream,
        fn: Callable[[], None],
        warmup: int = 3,
    ) -> None:
        self.stream = stream
        self.fn = fn
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._warmup_iters = warmup
        self._captured = False
        self._capture_error: Optional[str] = None
        # Diagnostic counters (Codex R11, 2026-05-06) — verify actual graph
        # replay vs eager fallback. inf 13ms vs trtexec 6.4ms could be either
        # H1 (graph not replaying) or H3 (GPU contention with ZED/bridge).
        self._replay_count = 0
        self._eager_count = 0

    def try_capture(self, max_retries: int = 3) -> bool:
        """Warm up then capture. Returns True on success.

        ROOT CAUSE FIX (2026-04-21):
          Default torch.cuda.graph() uses 'global' capture mode. ZED SDK's
          grab() runs on its own internal stream — uncoordinated with our
          pipeline streams. During capture, ZED grab issued a cudaCall on
          its stream → cudaErrorStreamCaptureInvalidated → silent fallback
          to eager → run was 40Hz instead of 80Hz, non-deterministic.

        Diagnostic from Jetson trace:
          [ZED][WARNING] [Grab] A CUDA error occurred:
            operation not permitted when stream is capturing (900)
          → cudaErrorStreamCaptureInvalidated

        Fix:
          1. capture_error_mode='thread_local' — only THIS thread's CUDA
             work is tracked; ZED's other-thread streams cannot invalidate.
          2. Retry up to max_retries — even with thread_local there can be
             transient races (driver init, etc.).
          3. Hard sync before each attempt — drain stale GPU work.
          4. On total failure, raise — silent eager fallback hides bugs and
             makes runs non-reproducible. Caller can catch if they want
             eager fallback explicitly.
        """
        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                # Drain everything before each attempt — GPU must be
                # quiescent or capture picks up stray work and fails.
                torch.cuda.synchronize()

                with torch.cuda.stream(self.stream):
                    for _ in range(self._warmup_iters):
                        self.fn()
                    self.stream.synchronize()

                    self._graph = torch.cuda.CUDAGraph()
                    # 'thread_local' = ZED's other-thread CUDA calls
                    # don't invalidate this capture (root cause of err 900).
                    with torch.cuda.graph(
                        self._graph,
                        stream=self.stream,
                        capture_error_mode="thread_local",
                    ):
                        self.fn()
                self._captured = True
                if attempt > 1:
                    LOGGER.info(
                        "CUDA graph captured on attempt %d/%d (stream=%s)",
                        attempt, max_retries, self.stream,
                    )
                else:
                    LOGGER.info("CUDA graph captured on stream %s", self.stream)
                return True
            except Exception as err:
                last_err = err
                self._capture_error = str(err)
                self._graph = None
                self._captured = False
                LOGGER.warning(
                    "CUDA graph capture attempt %d/%d failed: %s",
                    attempt, max_retries, err,
                )
        # All retries exhausted — raise so the caller sees a deterministic
        # failure rather than a silent slow path. Reproducibility > silent fallback.
        raise RuntimeError(
            f"CUDA graph capture failed after {max_retries} attempts: "
            f"{last_err}. Eager-mode silent fallback was masking this and "
            f"causing non-reproducible runs (sometimes 80Hz, sometimes 40Hz). "
            f"See cuda_graph.py docstring for root cause."
        )

    def replay(self) -> None:
        if self._captured and self._graph is not None:
            self._graph.replay()
            self._replay_count += 1
        else:
            with torch.cuda.stream(self.stream):
                self.fn()
            self._eager_count += 1

    @property
    def captured(self) -> bool:
        return self._captured

    @property
    def capture_error(self) -> Optional[str]:
        return self._capture_error

    @property
    def replay_count(self) -> int:
        """Codex R11: number of actual graph.replay() calls. If this stays
        small while frame_count grows, capture failed and we're in eager."""
        return self._replay_count

    @property
    def eager_count(self) -> int:
        """Codex R11: number of eager fallback executions. Should be 0 in
        production. If non-zero, graph capture failed silently."""
        return self._eager_count


@contextlib.contextmanager
def disabled_if(cond: bool):
    """Tiny utility — lets the caller skip graph capture under a flag."""
    if cond:
        yield None
    else:
        yield True

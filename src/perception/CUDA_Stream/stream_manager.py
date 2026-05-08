"""StreamManager — 4 CUDA streams + events + pinned host buffer pool.

Design:
    capture_stream  : ZED grab / H2D
    preproc_stream  : letterbox + normalize on GPU
    infer_stream    : TRT execute_async_v3
    post_stream     : 2D->3D + filter, final D2H to pinned host buffer

Cross-stream dependencies are expressed via torch.cuda.Event only
(stream.wait_event / event.record). No host-side synchronize() in the hot
path — only post_stream.synchronize() when consumer needs the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import torch
except ImportError as exc:  # pragma: no cover — Jetson-only hard dep
    raise RuntimeError(
        "torch is required for CUDA_Stream. On Jetson install the "
        "JetPack-matched PyTorch wheel."
    ) from exc


STAGE_NAMES = ("capture", "preproc", "infer", "post")


@dataclass
class StreamBundle:
    """Holds a torch.cuda.Stream plus its record event.

    Two event APIs (Codex R2 #11):
      Single-frame mode (current):
        ``record_done()`` + ``wait_for(other)`` use ``self.done_event`` —
        one event per bundle, re-recorded each frame.
      Multi-inflight mode (Phase 4 frame overlap):
        ``record_event(token.event)`` + ``wait_event(token.event)`` use
        per-token events held by ``PipelineToken``. Required because the
        single ``done_event`` is RACE-prone when two frames are in flight
        on the same stream (the second record overwrites the first before
        the first wait fires).
    Both APIs may coexist; current code only uses the bundle one.
    """

    name: str
    stream: "torch.cuda.Stream"
    done_event: "torch.cuda.Event" = field(init=False)

    def __post_init__(self) -> None:
        self.done_event = torch.cuda.Event(enable_timing=False, blocking=False)

    def record_done(self) -> None:
        self.done_event.record(self.stream)

    def wait_for(self, other: "StreamBundle") -> None:
        """Make this stream wait until ``other`` has recorded its event."""
        self.stream.wait_event(other.done_event)

    # ------------------------------------------------------------------
    # Per-token event API (Phase 4 frame overlap, Codex R2 #11).
    # Currently UNUSED — added in Phase 1 Day 2 so Phase 4 can wire tokens
    # without touching StreamBundle again.
    # ------------------------------------------------------------------
    def record_event(self, event: "torch.cuda.Event") -> None:
        """Record an externally-owned event on this bundle's stream.

        Caller (PipelineToken) owns the event lifecycle. This method is the
        per-token analogue of ``record_done()``.
        """
        event.record(self.stream)

    def wait_event(self, event: "torch.cuda.Event") -> None:
        """Make this stream wait on an externally-owned event.

        Per-token analogue of ``wait_for(other)`` but the event is supplied
        directly instead of being read from another bundle's ``done_event``.
        """
        self.stream.wait_event(event)

    @staticmethod
    def make_event(enable_timing: bool = False) -> "torch.cuda.Event":
        """Create a token-owned event.

        ``enable_timing=False`` for stream dependencies (cheaper —
        ``cudaEventDisableTiming`` flag). Tracer-owned events for stage
        timing should set ``enable_timing=True`` separately.
        ``blocking=False`` so ``query()`` is a non-blocking driver call.
        """
        return torch.cuda.Event(enable_timing=enable_timing, blocking=False)


class StreamManager:
    """Manages the four streams + pinned host buffers."""

    def __init__(
        self,
        device: Optional[torch.device] = None,
        pinned_bytes: int = 0,
        high_priority_stages: Optional[List[str]] = None,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available — cannot build StreamManager")
        self.device = device or torch.device("cuda:0")

        # All stages high-priority on TRT 10.x.
        # On TRT 8.x (4/18 baseline) priority asymmetry was harmless because
        # CUDA event timing showed the actual GPU work ~1ms; TRT 10.x's
        # heavier per-launch overhead caused low-prio streams (post) to
        # wait for infer to fully release SMs (measured post=7ms incl wait).
        # Equal priority lets the SM scheduler interleave overlapping work
        # across frames the way the 3-stage pipeline was designed to.
        high = set(high_priority_stages or list(STAGE_NAMES))
        lo_prio, hi_prio = torch.cuda.Stream.priority_range()
        self.streams: Dict[str, StreamBundle] = {}
        for name in STAGE_NAMES:
            prio = hi_prio if name in high else lo_prio
            try:
                stream = torch.cuda.Stream(device=self.device, priority=prio)
            except TypeError:
                # older torch may not accept priority kwarg
                stream = torch.cuda.Stream(device=self.device)
            self.streams[name] = StreamBundle(name=name, stream=stream)

        # pinned host buffer pool (bytes). Jetson iGPU benefits from
        # pinned host memory for host-visible D2H results (e.g. keypoints).
        self._pinned_pool: Optional[torch.Tensor] = None
        if pinned_bytes > 0:
            self._pinned_pool = torch.empty(
                pinned_bytes, dtype=torch.uint8, pin_memory=True
            )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def stream(self, stage: str) -> torch.cuda.Stream:
        return self.streams[stage].stream

    def bundle(self, stage: str) -> StreamBundle:
        return self.streams[stage]

    def stream_ptr(self, stage: str) -> int:
        """Raw CUDA stream handle for TensorRT execute_async_v3."""
        return int(self.streams[stage].stream.cuda_stream)

    # ------------------------------------------------------------------
    # Pinned buffers
    # ------------------------------------------------------------------
    def pinned_slice(self, offset: int, nbytes: int) -> torch.Tensor:
        if self._pinned_pool is None:
            raise RuntimeError("pinned_bytes=0 — no pinned pool allocated")
        if offset + nbytes > self._pinned_pool.numel():
            raise ValueError(
                f"pinned pool exhausted: need {offset + nbytes}B, "
                f"pool={self._pinned_pool.numel()}B"
            )
        return self._pinned_pool[offset : offset + nbytes]

    @staticmethod
    def make_pinned(shape, dtype=torch.float32) -> torch.Tensor:
        """Convenience for a standalone pinned host tensor."""
        return torch.empty(shape, dtype=dtype, pin_memory=True)

    # ------------------------------------------------------------------
    # Safety / debug
    # ------------------------------------------------------------------
    def synchronize_all(self) -> None:
        """Host-blocking sync of every stream. Use only at shutdown."""
        for bundle in self.streams.values():
            bundle.stream.synchronize()

    def query_all(self) -> Dict[str, bool]:
        """Non-blocking check — True means the stream has no pending work."""
        return {name: b.stream.query() for name, b in self.streams.items()}

    def __repr__(self) -> str:  # pragma: no cover — debug aid
        flags = self.query_all()
        return f"StreamManager(device={self.device}, idle={flags})"

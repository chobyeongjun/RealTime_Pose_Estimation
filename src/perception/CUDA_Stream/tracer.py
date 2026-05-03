"""Per-frame pipeline tracer — breadcrumbs for live debugging.

Instead of wondering "which stream stalled?" at 3 AM, enable the tracer
with ``--trace results/trace.csv`` and inspect the per-stage CUDA-event
timing offline.

Design points:
  * Every stage (capture / preproc / infer / post) gets a pair of
    ``torch.cuda.Event(enable_timing=True)`` markers. They're recorded
    on the matching stream so the elapsed time reflects actual GPU
    work, not host latency.
  * ``elapsed_time`` calls require the events to be fully recorded —
    we only read them AFTER ``post_stream.synchronize()``. This keeps
    the hot path sync-free.
  * When ``enabled=False`` the tracer is a zero-cost no-op. We still
    expose the same API so callers don't sprinkle ``if tracer:`` all
    over the place.

Emitted row schema (CSV):
    frame_id, ts_ns, e2e_ms,
    pre_ms, inf_ms, post_ms, host_overhead_ms,
    valid, occluded_count, depth_invalid_ratio,
    constraint_reason, box_conf

NB: "cap" is NOT tracked — ZED H2D runs on a stream owned by the
bridge, not a pipeline stream we can mark_start/end on. The capture
overhead is indirectly visible via host_overhead_ms.
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

LOGGER = logging.getLogger(__name__)

STAGE_NAMES = ("pre", "inf", "post")  # cap is measured indirectly via
                                        # (e2e - pre - inf - post) = host
                                        # + ZED-internal-stream overhead.


@dataclass
class FrameTrace:
    frame_id: int = 0
    ts_ns: int = 0
    t_host_start: float = 0.0
    t_host_end: float = 0.0
    stage_ms: Dict[str, float] = field(default_factory=dict)
    valid: bool = False
    occluded_count: int = 0
    depth_invalid_ratio: float = 0.0
    constraint_reason: str = ""
    box_conf: float = 0.0

    @property
    def e2e_ms(self) -> float:
        return (self.t_host_end - self.t_host_start) * 1e3

    @property
    def host_overhead_ms(self) -> float:
        gpu = sum(self.stage_ms.get(s, 0.0) for s in STAGE_NAMES)
        return max(self.e2e_ms - gpu, 0.0)

    def to_row(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "ts_ns": self.ts_ns,
            "e2e_ms": round(self.e2e_ms, 3),
            "pre_ms": round(self.stage_ms.get("pre", 0.0), 3),
            "inf_ms": round(self.stage_ms.get("inf", 0.0), 3),
            "post_ms": round(self.stage_ms.get("post", 0.0), 3),
            "host_overhead_ms": round(self.host_overhead_ms, 3),
            "valid": int(self.valid),
            "occluded_count": self.occluded_count,
            "depth_invalid_ratio": round(self.depth_invalid_ratio, 3),
            "constraint_reason": self.constraint_reason,
            "box_conf": round(self.box_conf, 3),
        }


class PipelineTracer:
    """Record per-stage GPU timings when ``enabled=True``. No-op otherwise."""

    FIELDS = (
        "frame_id", "ts_ns", "e2e_ms",
        "pre_ms", "inf_ms", "post_ms",
        "host_overhead_ms",
        "valid", "occluded_count", "depth_invalid_ratio",
        "constraint_reason", "box_conf",
    )

    def __init__(
        self,
        enabled: bool = False,
        csv_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        ring_size: int = 256,
    ) -> None:
        # ``enabled`` gates CSV writing only. GPU event timing is always on
        # so that pre/inf/post_ms appear in [SLOW] logs without --trace.
        self.enabled = enabled
        self.csv_path = Path(csv_path) if csv_path else None
        self.device = device or torch.device("cuda:0")
        self._events: Dict[str, tuple] = {}
        self._current = FrameTrace()
        self._rows: List[Dict[str, Any]] = []
        self.ring_size = ring_size
        if torch.cuda.is_available():
            for s in STAGE_NAMES:
                self._events[s] = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )

    # ------------------------------------------------------------------
    # Hot-path API
    # ------------------------------------------------------------------
    def begin(self, frame_id: int, ts_ns: int) -> None:
        self._current = FrameTrace(frame_id=frame_id, ts_ns=ts_ns)
        self._current.t_host_start = time.perf_counter()

    def mark_start(self, stage: str, stream: torch.cuda.Stream) -> None:
        if stage not in self._events:
            return
        self._events[stage][0].record(stream)

    def mark_end(self, stage: str, stream: torch.cuda.Stream) -> None:
        if stage not in self._events:
            return
        self._events[stage][1].record(stream)

    def set_result_meta(
        self,
        valid: bool,
        occluded_count: int = 0,
        depth_invalid_ratio: float = 0.0,
        constraint_reason: str = "",
        box_conf: float = 0.0,
    ) -> None:
        self._current.valid = valid
        self._current.occluded_count = occluded_count
        self._current.depth_invalid_ratio = depth_invalid_ratio
        self._current.constraint_reason = constraint_reason
        self._current.box_conf = box_conf

    def end(self) -> FrameTrace:
        """Called AFTER post_stream.synchronize() — safe to read events."""
        self._current.t_host_end = time.perf_counter()
        for stage, (a, b) in self._events.items():
            try:
                self._current.stage_ms[stage] = a.elapsed_time(b)
            except Exception as err:  # pragma: no cover
                LOGGER.debug("tracer elapsed_time %s failed: %s", stage, err)
                self._current.stage_ms[stage] = 0.0
        if self.enabled:
            row = self._current.to_row()
            self._rows.append(row)
            if len(self._rows) > self.ring_size and self.csv_path is None:
                self._rows.pop(0)
        return self._current

    # ------------------------------------------------------------------
    # Post-run API
    # ------------------------------------------------------------------
    def dump(self) -> Optional[Path]:
        if not self._rows:
            return None
        if self.csv_path is None:
            return None
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(self._rows)
        LOGGER.info("tracer dumped %d rows to %s", len(self._rows), self.csv_path)
        return self.csv_path

    def rows(self) -> List[Dict[str, Any]]:
        return list(self._rows)

    def summary(self, percentiles=(50, 95, 99)) -> Dict[str, Dict[str, float]]:
        """Quick summary — min/max/percentile per stage."""
        if not self._rows:
            return {}
        import numpy as np  # local import keeps tracer CPU-optional

        out: Dict[str, Dict[str, float]] = {}
        for key in ("e2e_ms", "pre_ms", "inf_ms", "post_ms", "host_overhead_ms"):
            values = np.asarray([r[key] for r in self._rows if r[key] > 0], dtype=float)
            if values.size == 0:
                continue
            stats = {"mean": float(values.mean()), "min": float(values.min()),
                     "max": float(values.max())}
            for p in percentiles:
                stats[f"p{p}"] = float(np.percentile(values, p))
            out[key] = stats
        return out

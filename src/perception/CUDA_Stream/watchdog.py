"""Watchdog — monitors stream health and SHM freshness; forces safe-stop.

Responsibilities:
  * time-out a stream that has been busy > ``stream_timeout_ms``
    (usually indicates a hang — log + trigger fallback)
  * alert when no SHM update arrives for > ``publish_timeout_ms``
  * keep depth-invalid ratio history and warn on sustained high values
    (past incident: ZED copy=False race caused calib 0% — see
    skiro-learnings). Flag threshold = 30% over 5s window.
  * **Safety**: when unhealthy, FORCE the publisher to emit
    ``valid=False`` and drop an estop sentinel file (``/dev/shm/
    hwalker_pose_cuda_estop``). The control loop MUST honor this
    sentinel and return the cable force command to 0N. Without this
    the stale ``valid=1`` keypoint array would keep driving AK60 up to
    the 70N limit.

The watchdog runs in its own thread; it does NOT own the CUDA context.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Optional

import numpy as np
import torch

ESTOP_FILE_DEFAULT = "/dev/shm/hwalker_pose_cuda_estop"

LOGGER = logging.getLogger(__name__)


@dataclass
class WatchdogStatus:
    healthy: bool = True
    reason: str = ""
    stream_busy_ms: dict = field(default_factory=dict)
    publish_age_ms: float = 0.0
    depth_invalid_ratio: float = 0.0
    fallback_triggered: bool = False


class StreamWatchdog:
    def __init__(
        self,
        streams: "dict[str, torch.cuda.Stream]",
        *,
        stream_timeout_ms: float = 30.0,  # ~3 SVGA frames. Trips only on
                                          # real hangs, not on a single slow
                                          # frame or occasional ZED jitter.
        publish_timeout_ms: float = 250.0,  # ≈17 consecutive slow SVGA frames.
                                            # Measured on Jetson Orin NX @68Hz
                                            # (14.5ms/frame), occasional 200ms
                                            # spikes occur from GC / thermal /
                                            # IMU retrieve bursts. 100ms tripped
                                            # false positives — 250ms covers
                                            # those without missing real hangs.
        startup_grace_s: float = 5.0,        # ZED depth init can take 1-2s; ignore
                                              # publish_timeout until the FIRST
                                              # real publish arrives.
        depth_window_s: float = 5.0,
        depth_invalid_threshold: float = 0.30,
        fallback_cb: Optional[Callable[[str], None]] = None,
        publisher: Optional[Any] = None,  # ShmPublisher — forced valid=False
        estop_file: str = ESTOP_FILE_DEFAULT,
        poll_hz: float = 200.0,  # 5ms granularity → worst-case detect ≈ 20ms
    ) -> None:
        self.streams = streams
        self.stream_timeout_ms = stream_timeout_ms
        self.publish_timeout_ms = publish_timeout_ms
        self.startup_grace_s = startup_grace_s
        self.depth_window_s = depth_window_s
        self.depth_invalid_threshold = depth_invalid_threshold
        self.fallback_cb = fallback_cb
        self.publisher = publisher
        self.estop_file = estop_file
        self.poll_interval = 1.0 / max(poll_hz, 1.0)

        self._stop = threading.Event()
        self._pause = threading.Event()  # set = paused (skip ticks)
        self._thread: Optional[threading.Thread] = None
        self._status = WatchdogStatus()
        self._start_ts = time.monotonic()
        self._last_publish_ts = time.monotonic()
        self._got_first_publish = False
        self._depth_samples: Deque[tuple[float, float]] = deque()  # (t, ratio)
        self._busy_since: dict[str, Optional[float]] = {n: None for n in streams}
        self._status_lock = threading.Lock()
        self._last_fallback_reason = ""

    # ------------------------------------------------------------------
    # Signals from the pipeline
    # ------------------------------------------------------------------
    def note_publish(self) -> None:
        self._last_publish_ts = time.monotonic()
        self._got_first_publish = True

    def note_depth_invalid(self, ratio: float) -> None:
        now = time.monotonic()
        self._depth_samples.append((now, ratio))
        cutoff = now - self.depth_window_s
        while self._depth_samples and self._depth_samples[0][0] < cutoff:
            self._depth_samples.popleft()

    # ------------------------------------------------------------------
    # Thread control
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread:
            return
        self._stop.clear()
        # Reset timing anchors so the startup grace starts NOW, not at __init__.
        self._start_ts = time.monotonic()
        self._last_publish_ts = time.monotonic()
        self._got_first_publish = False
        self._thread = threading.Thread(
            target=self._loop, name="StreamWatchdog", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def pause(self) -> None:
        """Pause stream.query() polling (use during CUDA graph capture).

        Watchdog calls torch.cuda.Stream.query() every poll_interval, which
        is a CUDA API call → forbidden during graph capture (raises
        cudaErrorStreamCaptureUnsupported and invalidates capture).
        Caller should pause() before capture, resume() after.
        """
        self._pause.set()

    def resume(self) -> None:
        """Resume stream polling after graph capture finishes."""
        self._pause.clear()

    def status(self) -> WatchdogStatus:
        with self._status_lock:
            # return a shallow copy so reader isn't racing against writer
            return WatchdogStatus(
                healthy=self._status.healthy,
                reason=self._status.reason,
                stream_busy_ms=dict(self._status.stream_busy_ms),
                publish_age_ms=self._status.publish_age_ms,
                depth_invalid_ratio=self._status.depth_invalid_ratio,
                fallback_triggered=self._status.fallback_triggered,
            )

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._pause.is_set():
                # Skip CUDA polling during graph capture etc.
                # Don't update timing anchors so we don't false-positive
                # on resume.
                time.sleep(self.poll_interval)
                continue
            try:
                self._tick()
            except Exception as err:  # pragma: no cover — never die
                LOGGER.error("watchdog tick error: %s", err)
            time.sleep(self.poll_interval)

    def _tick(self) -> None:
        now = time.monotonic()
        startup_elapsed = now - self._start_ts
        in_startup = (
            not self._got_first_publish
            and startup_elapsed < self.startup_grace_s
        )
        stream_busy_ms: dict[str, float] = {}
        reason = ""
        healthy = True

        for name, s in self.streams.items():
            idle = s.query()
            if idle:
                self._busy_since[name] = None
                stream_busy_ms[name] = 0.0
            else:
                if self._busy_since[name] is None:
                    self._busy_since[name] = now
                age_ms = (now - self._busy_since[name]) * 1e3
                stream_busy_ms[name] = age_ms
                # Startup warmup / JIT / ZED depth init can make a stream
                # transiently busy for 20-50ms on the very first frame.
                # Skip stream-timeout checks until we've seen a real publish
                # or the startup grace window elapses.
                if age_ms > self.stream_timeout_ms and not in_startup:
                    reason = f"stream {name} busy {age_ms:.1f} ms"
                    healthy = False

        publish_age_ms = (now - self._last_publish_ts) * 1e3
        startup_elapsed = now - self._start_ts
        # Don't consider "no publish" unhealthy until either (a) at least
        # one real publish has landed, or (b) the startup grace window
        # expired. Otherwise ZED depth init (1-2s) always trips a false
        # estop before the first frame.
        if (publish_age_ms > self.publish_timeout_ms
                and not reason
                and (self._got_first_publish
                     or startup_elapsed > self.startup_grace_s)):
            reason = f"no publish for {publish_age_ms:.0f} ms"
            healthy = False

        if self._depth_samples:
            avg = sum(r for _, r in self._depth_samples) / len(self._depth_samples)
        else:
            avg = 0.0
        if avg > self.depth_invalid_threshold and not reason:
            reason = f"depth invalid ratio {avg:.0%} over {self.depth_window_s:.0f}s"
            healthy = False

        with self._status_lock:
            was_healthy = self._status.healthy
            self._status.healthy = healthy
            self._status.reason = reason
            self._status.stream_busy_ms = stream_busy_ms
            self._status.publish_age_ms = publish_age_ms
            self._status.depth_invalid_ratio = avg

            # Re-arm fallback on healthy → unhealthy transition (not only once).
            became_unhealthy = was_healthy and not healthy
            became_healthy = not was_healthy and healthy

            if became_unhealthy or (not healthy and reason != self._last_fallback_reason):
                self._status.fallback_triggered = True
                self._last_fallback_reason = reason
                LOGGER.warning("watchdog → fallback: %s", reason)
                self._force_safe_stop(reason)
                if self.fallback_cb:
                    try:
                        self.fallback_cb(reason)
                    except Exception as err:  # pragma: no cover
                        LOGGER.error("fallback_cb raised: %s", err)

            if became_healthy:
                self._status.fallback_triggered = False
                self._clear_safe_stop()

    def _force_safe_stop(self, reason: str) -> None:
        """Invalidate SHM + drop estop sentinel. Never raises.

        Reads the publisher's current K at call time (not at init) so the
        zero buffers match regardless of which schema the pipeline uses.
        Previously hard-coded K=17 silently failed for lowlimb6 (K=6).
        """
        try:
            if self.publisher is not None:
                K = int(getattr(self.publisher, "K", 17))
                zeros3 = np.zeros((K, 3), dtype=np.float32)
                zeros1 = np.zeros((K,), dtype=np.float32)
                zeros2 = np.zeros((K, 2), dtype=np.float32)
                # SHM v2 (Codex review b1ky3965z P1-1 fix, 2026-05-11):
                # rgb_ts_ns 명시. valid=False + valid_mask_bits=0 → 모든 keypoint invalid.
                # valid_reason 명시 안 하면 publisher 가 INVALID_UNKNOWN default.
                self.publisher.publish(
                    frame_id=0,
                    rgb_ts_ns=time.time_ns(),
                    kpts_3d_m=zeros3,
                    kpt_conf=zeros1,
                    kpts_2d_px=zeros2,
                    box_conf=0.0,
                    valid=False,
                    depth_invalid_ratio=1.0,
                )
        except Exception as err:  # pragma: no cover
            LOGGER.error("watchdog publisher force-invalidate failed: %s", err)
        try:
            Path(self.estop_file).write_text(reason + "\n")
        except Exception as err:  # pragma: no cover
            LOGGER.error("watchdog estop file write failed: %s", err)

    def _clear_safe_stop(self) -> None:
        try:
            if os.path.exists(self.estop_file):
                os.unlink(self.estop_file)
        except Exception as err:  # pragma: no cover
            LOGGER.error("watchdog estop file unlink failed: %s", err)

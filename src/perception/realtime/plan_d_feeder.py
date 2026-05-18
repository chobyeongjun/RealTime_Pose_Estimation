"""Plan D feeder — separate Python process for EKF cascade + forecast publish.

Architecture (Sprint 1 Phase 1 A.1):
  Inference loop (pipeline_main.py with --plan-d-mode async):
    capture → predict → 3D depth → SHM v2 publish (END)
    → puts (t, q, sigma, hip_z) to multiprocessing.Queue

  This process (separate, parallel):
    while not stop:
      msg = queue.get()
      predictor.feed(t, q, sigma, hip_z)
      forecast = predictor.forecast(0.05)
      forecast_publisher.publish(forecast, hs_events, ...)

Why this design:
  - Plan D feed (439 us p50) + forecast publish (~500 us est) 둘 다 inference loop 에서 제거
  - Cross-process IPC via multiprocessing.Queue (pickle overhead ~50-100us, async)
  - Forecast 의 logical latency ~1 frame (8.3ms @ 120fps) — 50ms forecast horizon 으로 흡수
  - Inference loop p50 expected: 17.56 → 16.0-16.5 ms (1.0-1.5 ms gain)

Public API:
  start_feeder_process(forecast_shm_name, fs_hz, tau_s) → (process, queue, stop_event)
  stop_feeder(process, queue, stop_event)

이 module 은 inference loop (pipeline_main.py) 에서 import.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue as queue_module
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

LOGGER = logging.getLogger(__name__)


# ─── Message type for queue ──────────────────────────────────────────────
@dataclass
class FeedMessage:
    """One Plan D feed message (sent inference loop → feeder process)."""
    t_now: float                  # seconds (monotonic)
    q: np.ndarray                 # (6,) float64
    sigma_per_joint: np.ndarray   # (6,) float64
    hip_z_world_m: float
    frame_id: int
    rgb_ts_ns: int


# ─── Feeder process main loop ────────────────────────────────────────────
def _feeder_main(
    in_queue: "mp.Queue[FeedMessage]",
    stop_event: "mp.Event",
    forecast_shm_name: str,
    n_joints: int,
    fs_hz: float,
    tau_s: float,
    log_path: Optional[str] = None,
    pythonpath: Optional[str] = None,
) -> None:
    """Run inside child process. Consumes queue, runs Plan D, publishes forecast.

    NOTE: spawn context is used (not fork) to avoid inheriting parent's CUDA
    context. This means we need to re-setup sys.path manually.
    """
    # Re-setup sys.path (spawn does not inherit parent's path modifications)
    if pythonpath:
        for p in pythonpath.split(":"):
            if p and p not in sys.path:
                sys.path.insert(0, p)

    # Setup logging in child
    if log_path:
        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format='%(asctime)s [feeder pid=%(process)d] %(message)s',
            force=True,
        )
    log = logging.getLogger("plan_d_feeder")
    log.info("Plan D feeder process started (pid=%d, spawn context)", os.getpid())

    try:
        from perception.plan_d_prototype import PlanDPredictor
    except ImportError as e:
        log.error("PlanDPredictor import failed: %s (sys.path=%s)", e, sys.path)
        return

    try:
        from perception.realtime.forecast_publisher import ForecastPublisher
    except ImportError:
        try:
            from src.perception.realtime.forecast_publisher import ForecastPublisher
        except ImportError as e:
            log.error("ForecastPublisher import failed: %s", e)
            return

    # Force-unlink any stale forecast SHM (from prior inline run or crashed process)
    try:
        from multiprocessing import shared_memory
        stale = shared_memory.SharedMemory(name=forecast_shm_name)
        stale.close()
        stale.unlink()
        log.info("Unlinked stale /%s before create", forecast_shm_name)
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning("Stale SHM unlink failed (continuing): %s", e)

    # Initialize predictor + forecast publisher
    predictor = PlanDPredictor(n_joints=n_joints, fs_hz=fs_hz)
    forecast_pub = None
    try:
        forecast_pub = ForecastPublisher(name=forecast_shm_name, create=True)
        log.info("Forecast publisher opened: /%s", forecast_shm_name)
    except Exception as e:
        log.error("ForecastPublisher init failed: %s", e)
        return

    n_processed = 0
    n_dropped = 0
    n_errors = 0
    t_start = time.monotonic()

    while not stop_event.is_set():
        try:
            msg = in_queue.get(timeout=0.5)
        except queue_module.Empty:
            continue
        except (EOFError, BrokenPipeError):
            log.info("Queue closed by parent")
            break

        if msg is None:  # poison pill
            break

        n_processed += 1
        try:
            predictor.feed(
                t_now=float(msg.t_now),
                q=np.asarray(msg.q, dtype=np.float64),
                sigma_per_joint=np.asarray(msg.sigma_per_joint, dtype=np.float64),
                hip_z_world_m=float(msg.hip_z_world_m),
            )
            forecast = predictor.forecast(tau_s=tau_s)

            forecast_pub.publish(
                frame_id=msg.frame_id,
                publish_done_mono_ns=time.monotonic_ns(),
                tau_lookahead_s=tau_s,
                forecast=forecast,
                cascade_level=int(predictor.level),
                stride_count=int(predictor.stride_count),
                template_touched_fraction=float(predictor.template_touched_fraction),
                is_ready_for_control=predictor.is_ready_for_control(),
                hs_event_L=predictor.predict_heel_strike("L"),
                hs_event_R=predictor.predict_heel_strike("R"),
            )
        except Exception as e:
            n_errors += 1
            if n_errors < 5:
                log.warning("Feed/forecast error frame=%d: %s", msg.frame_id, e)

    elapsed = time.monotonic() - t_start
    log.info(
        "Plan D feeder shutdown: processed=%d, errors=%d, dropped=%d, elapsed=%.1fs (%.1f msg/s)",
        n_processed, n_errors, n_dropped, elapsed, n_processed / elapsed if elapsed > 0 else 0,
    )

    if forecast_pub is not None:
        try:
            forecast_pub.close()
        except Exception:
            pass


# ─── Public API ──────────────────────────────────────────────────────────
def start_feeder_process(
    forecast_shm_name: str = "hwalker_forecast",
    n_joints: int = 6,
    fs_hz: float = 60.0,
    tau_s: float = 0.05,
    queue_size: int = 200,
    log_path: Optional[str] = None,
):
    """Spawn the Plan D feeder process.

    Returns:
        (process, queue, stop_event) — for inference loop to push messages.

    Example:
        proc, q, stop = start_feeder_process()
        for frame in loop:
            ... inference ...
            try:
                q.put_nowait(FeedMessage(t_now, q_vec, sigma, hip_z, frame_id, ts_ns))
            except queue.Full:
                pass  # drop frame
        stop_feeder(proc, q, stop)
    """
    # IMPORTANT: spawn (NOT fork). fork() inherits parent's CUDA context
    # (from ZED SDK + TRT engine), which breaks in the child process.
    # spawn restarts a fresh Python interpreter — slower (~1s start) but safe.
    ctx = mp.get_context("spawn")
    in_queue: "mp.Queue[FeedMessage]" = ctx.Queue(maxsize=queue_size)
    stop_event = ctx.Event()

    # Pass current sys.path to child (spawn doesn't inherit)
    pythonpath = ":".join(p for p in sys.path if p and p != ".")

    proc = ctx.Process(
        target=_feeder_main,
        args=(in_queue, stop_event, forecast_shm_name, n_joints, fs_hz, tau_s, log_path, pythonpath),
        daemon=False,  # spawn + daemon=True can cause join issues
        name="PlanDFeeder",
    )
    proc.start()
    LOGGER.info("Plan D feeder process started (pid=%d, spawn context)", proc.pid)
    return proc, in_queue, stop_event


def stop_feeder(proc, queue, stop_event, timeout: float = 3.0) -> None:
    """Gracefully stop feeder process."""
    try:
        stop_event.set()
        try:
            queue.put(None, timeout=0.5)  # poison pill
        except Exception:
            pass
        proc.join(timeout=timeout)
        if proc.is_alive():
            LOGGER.warning("Plan D feeder did not exit, terminating")
            proc.terminate()
            proc.join(timeout=1.0)
    except Exception as e:
        LOGGER.warning("Plan D feeder stop error: %s", e)


# ─── Standalone test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    """Quick smoke test: spawn feeder, send 100 synthetic messages, shut down."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-msgs", type=int, default=100)
    ap.add_argument("--rate-hz", type=float, default=60.0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.info("Smoke test: spawning Plan D feeder")

    proc, q, stop = start_feeder_process(
        forecast_shm_name="test_forecast_feeder",
        log_path="/tmp/plan_d_feeder_test.log",
    )

    period = 1.0 / args.rate_hz
    sigma = np.full(6, 0.05, dtype=np.float64)
    for i in range(args.n_msgs):
        t_now = i * period
        phase = 2 * np.pi * t_now
        q_vec = np.array([
            0.3 * np.sin(phase),
            0.5 * max(0.0, np.sin(phase + np.pi / 4)),
            0.3 * np.sin(phase + np.pi / 2),
            0.3 * np.sin(phase + np.pi),
            0.5 * max(0.0, np.sin(phase + np.pi + np.pi / 4)),
            0.3 * np.sin(phase + np.pi + np.pi / 2),
        ], dtype=np.float64)
        hip_z = 0.5 + 0.02 * np.cos(2 * phase)

        msg = FeedMessage(
            t_now=t_now, q=q_vec, sigma_per_joint=sigma,
            hip_z_world_m=hip_z, frame_id=i, rgb_ts_ns=int(t_now * 1e9),
        )
        try:
            q.put_nowait(msg)
        except queue_module.Full:
            logging.warning("Queue full at frame %d", i)
        time.sleep(period * 0.5)  # simulate ~2x rate

    logging.info("Sent %d messages, waiting 1s for feeder to drain", args.n_msgs)
    time.sleep(1.0)
    stop_feeder(proc, q, stop)
    logging.info("Smoke test complete. Check /tmp/plan_d_feeder_test.log")

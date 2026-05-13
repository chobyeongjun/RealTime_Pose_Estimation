"""Short-burst depth hold for ZED PERFORMANCE NaN bursts.

Problem: When a single keypoint's depth flickers NaN for 1-3 frames, the
whole frame goes invalid → Plan D EKF skips feed → cascade stuck at L1.

Strategy (A+B hybrid):
  Burst ≤ MAX_HOLD_FRAMES (default 3): substitute last-good keypoint 3D.
                                       Frame becomes valid again, EKF gets
                                       a normal measurement.
  Burst > MAX_HOLD_FRAMES: stop holding, return NaN. compute_joint_state
                            then marks invalid → predictor.feed() skipped
                            → EKF predict-only (process model). Sigma grows
                            automatically. Bridge falls back to heartbeat.

Per-keypoint hold counters — each joint independent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


MAX_HOLD_FRAMES = 3   # ~50ms at 64Hz / ~25ms at 120Hz — within Plan D τ


@dataclass
class _LastGood:
    pos: Optional[np.ndarray]   # (3,) float32 last-good 3D
    conf: float                  # last-good confidence
    age_frames: int              # frames since last fresh sample


class DepthHoldLayer:
    """Per-keypoint NaN burst smoothing — short hold, long invalidate.

    Sit BETWEEN raw_3d (depth-backprojected) and compute_joint_state().
    Pure function-style, no global state — caller owns the instance.
    """

    def __init__(self, max_hold_frames: int = MAX_HOLD_FRAMES):
        self.max_hold = max_hold_frames
        self._cache: Dict[str, _LastGood] = {}
        # counters for telemetry
        self.held_total = 0
        self.dropped_total = 0
        self.fresh_total = 0

    def step(
        self,
        raw_3d: Dict[str, np.ndarray],
        confs: Dict[str, float],
        expected_joints: Optional[list] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
        """Return (filled_3d, status_per_kp).

        status_per_kp[name] ∈ {'fresh', 'held', 'dropped', 'absent'}.
        'absent' = no last-good and current is missing/NaN.

        Codex P2: _batch_2d_to_3d omits invalid keypoints from raw_3d (key
        simply absent) rather than including them with z=NaN. So we must
        iterate over the *expected* joint name set, not just raw_3d.keys().
        """
        # Build the set we must process this tick:
        #   - all currently-fresh joints
        #   - all explicitly expected joints (caller list)
        #   - all joints we are still holding (so they keep aging out)
        names = set(raw_3d.keys()) | set(self._cache.keys())
        if expected_joints:
            names.update(expected_joints)

        out: Dict[str, np.ndarray] = {}
        status: Dict[str, str] = {}
        for name in names:
            pt = raw_3d.get(name)
            fresh = False
            xyz = None
            if pt is not None:
                xyz = np.asarray(pt, dtype=np.float32)
                z = float(xyz[2]) if xyz.shape[0] >= 3 else float('nan')
                fresh = np.isfinite(z) and 0.1 <= z <= 3.0

            if fresh:
                self._cache[name] = _LastGood(pos=xyz.copy(),
                                              conf=float(confs.get(name, 1.0)),
                                              age_frames=0)
                out[name] = xyz
                status[name] = 'fresh'
                self.fresh_total += 1
                continue

            # Missing or NaN this tick — try hold
            lg = self._cache.get(name)
            if lg is not None and lg.age_frames < self.max_hold:
                lg.age_frames += 1
                out[name] = lg.pos.copy()
                status[name] = 'held'
                self.held_total += 1
            else:
                status[name] = 'dropped' if lg is not None else 'absent'
                self.dropped_total += 1
                self._cache.pop(name, None)

        return out, status

    def stats(self) -> Dict[str, int]:
        return {
            "fresh":   self.fresh_total,
            "held":    self.held_total,
            "dropped": self.dropped_total,
        }

    def reset(self) -> None:
        self._cache.clear()
        self.held_total = 0
        self.dropped_total = 0
        self.fresh_total = 0

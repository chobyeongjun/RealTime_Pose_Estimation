"""Unit tests for DepthHoldLayer."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from perception.realtime.depth_hold import DepthHoldLayer  # noqa: E402


def test_fresh_passthrough():
    layer = DepthHoldLayer()
    raw = {"left_hip": np.array([0.1, -0.2, 1.5], dtype=np.float32)}
    confs = {"left_hip": 0.8}
    out, status = layer.step(raw, confs)
    assert status["left_hip"] == "fresh"
    assert np.allclose(out["left_hip"], [0.1, -0.2, 1.5])


def test_short_burst_held_with_last_good():
    layer = DepthHoldLayer(max_hold_frames=3)
    # frame 0: fresh
    raw = {"hip": np.array([0.1, -0.2, 1.5], dtype=np.float32)}
    confs = {"hip": 0.8}
    layer.step(raw, confs)

    # frames 1..3: NaN — should be held
    nan_raw = {"hip": np.array([0.1, -0.2, float('nan')], dtype=np.float32)}
    for i in range(3):
        out, status = layer.step(nan_raw, confs)
        assert status["hip"] == "held", f"frame {i+1} expected held, got {status}"
        assert np.allclose(out["hip"], [0.1, -0.2, 1.5])


def test_long_burst_dropped_after_max_hold():
    layer = DepthHoldLayer(max_hold_frames=3)
    layer.step({"hip": np.array([0,0,1.5], np.float32)}, {"hip": 1.0})
    nan_raw = {"hip": np.array([0,0,float('nan')], np.float32)}
    # 3 frames held
    for _ in range(3):
        layer.step(nan_raw, {"hip": 1.0})
    # 4th frame must drop
    out, status = layer.step(nan_raw, {"hip": 1.0})
    assert status["hip"] == "dropped"
    assert "hip" not in out


def test_drop_resets_so_next_fresh_can_start_hold():
    layer = DepthHoldLayer(max_hold_frames=2)
    # never-fresh first
    out, status = layer.step({"hip": np.array([0,0,float('nan')], np.float32)},
                             {"hip": 1.0})
    assert status["hip"] == "absent"
    # now fresh
    layer.step({"hip": np.array([0,0,1.0], np.float32)}, {"hip": 1.0})
    # then NaN should hold (not absent)
    out, status = layer.step({"hip": np.array([0,0,float('nan')], np.float32)},
                             {"hip": 1.0})
    assert status["hip"] == "held"


def test_out_of_range_treated_as_nan():
    layer = DepthHoldLayer()
    layer.step({"hip": np.array([0,0,1.0], np.float32)}, {"hip": 1.0})
    # z=5.0 out of [0.1, 3.0]
    out, status = layer.step({"hip": np.array([0,0,5.0], np.float32)},
                             {"hip": 1.0})
    assert status["hip"] == "held"


def test_stats_counters():
    layer = DepthHoldLayer(max_hold_frames=2)
    layer.step({"hip": np.array([0,0,1.0], np.float32)}, {"hip": 1.0})
    layer.step({"hip": np.array([0,0,float('nan')], np.float32)}, {"hip": 1.0})
    layer.step({"hip": np.array([0,0,float('nan')], np.float32)}, {"hip": 1.0})
    layer.step({"hip": np.array([0,0,float('nan')], np.float32)}, {"hip": 1.0})  # drop
    s = layer.stats()
    assert s["fresh"] == 1
    assert s["held"] == 2
    assert s["dropped"] == 1


def test_per_joint_independent():
    layer = DepthHoldLayer(max_hold_frames=2)
    layer.step({
        "L": np.array([0,0,1.0], np.float32),
        "R": np.array([0,0,1.0], np.float32),
    }, {"L": 1.0, "R": 1.0})

    # L NaN, R fresh
    out, status = layer.step({
        "L": np.array([0,0,float('nan')], np.float32),
        "R": np.array([0,0,1.1], np.float32),
    }, {"L": 1.0, "R": 1.0})
    assert status["L"] == "held"
    assert status["R"] == "fresh"
    assert np.isclose(out["L"][2], 1.0)
    assert np.isclose(out["R"][2], 1.1)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))

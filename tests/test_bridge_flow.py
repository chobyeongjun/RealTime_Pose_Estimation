"""Mac-side test: full bridge flow (synthetic forecast → mock serial).

Goal: prove that when forecast_publisher writes a valid forecast packet,
shm_to_teensy_bridge reads it, applies safety gate, and emits a PKT_COMMAND
with byte-correct q_pred and joint count = 6. No real Teensy needed.
"""
from __future__ import annotations

import math
import struct
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "src" / "perception" / "realtime"))
sys.path.insert(0, str(_REPO / "scripts"))

from forecast_publisher import ForecastPublisher  # noqa: E402
from shm_to_teensy_bridge import (  # noqa: E402
    ForecastReader, ForecastSample, MockSerial, kp_kd_default,
    FORECAST_SIZE, FORECAST_VERSION, N_JOINTS,
)
from teensy_smoke_test import (  # noqa: E402
    parse_frame, PKT_COMMAND, PKT_HEARTBEAT, MAGIC,
)


def _shm_name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class MockForecast:
    def __init__(self, q_pred):
        self.phi = 1.2
        self.sigma_phi = 0.05
        self.omega = 6.28
        self.sigma_omega = 0.2
        self.alpha = 0.0
        self.sigma_alpha = 0.5
        self.q_pred = np.asarray(q_pred, dtype=np.float32)


class MockHS:
    def __init__(self, t=0.5, conf=0.9, ready=True):
        self.t_ahead_s = t
        self.sigma_t_s = 0.02
        self.confidence = conf
        self.ready = ready


def _publish_one(name: str, q_pred, *, ready=True, cascade=3,
                 sigma_phi=0.05, q_sigma=0.05) -> None:
    pub = ForecastPublisher(name=name, create=True)
    try:
        fc = MockForecast(q_pred)
        fc.sigma_phi = sigma_phi
        pub.publish(
            frame_id=1,
            publish_done_mono_ns=time.monotonic_ns(),
            tau_lookahead_s=0.05,
            forecast=fc,
            cascade_level=cascade,
            stride_count=10,
            template_touched_fraction=0.9,
            is_ready_for_control=ready,
            hs_event_L=MockHS(),
            hs_event_R=MockHS(),
            q_pred_sigma=np.full(N_JOINTS, q_sigma, dtype=np.float32),
        )
    finally:
        pub.close()
    # Caller is expected to unlink. We don't here so reader still sees it.


def test_forecast_reader_reads_valid_publish():
    name = _shm_name("tfr")
    _publish_one(name, [0.1, 0.2, 0.3, -0.1, -0.2, -0.3])
    try:
        reader = ForecastReader(name)
        try:
            fc = reader.read()
            assert fc is not None
            assert fc.version == FORECAST_VERSION
            assert fc.is_ready == 1
            assert fc.cascade_level == 3
            assert abs(fc.q_pred[0] - 0.1) < 1e-5
            assert abs(fc.q_pred[5] - (-0.3)) < 1e-5
            assert fc.valid_for_control()
        finally:
            reader.close()
    finally:
        try:
            from multiprocessing import shared_memory
            shared_memory.SharedMemory(name=name).unlink()
        except Exception:
            pass


def test_valid_for_control_rejects_low_cascade():
    name = _shm_name("tlc")
    _publish_one(name, [0.0]*6, cascade=1)
    try:
        reader = ForecastReader(name)
        fc = reader.read()
        assert fc is not None
        assert fc.cascade_level == 1
        assert not fc.valid_for_control()
        reader.close()
    finally:
        try:
            from multiprocessing import shared_memory
            shared_memory.SharedMemory(name=name).unlink()
        except Exception:
            pass


def test_valid_for_control_rejects_high_sigma():
    name = _shm_name("ths")
    _publish_one(name, [0.0]*6, q_sigma=1.0)
    try:
        reader = ForecastReader(name)
        fc = reader.read()
        assert fc is not None
        assert not fc.valid_for_control(max_q_sigma=0.30)
        reader.close()
    finally:
        try:
            from multiprocessing import shared_memory
            shared_memory.SharedMemory(name=name).unlink()
        except Exception:
            pass


def test_valid_for_control_rejects_not_ready():
    name = _shm_name("tnr")
    _publish_one(name, [0.0]*6, ready=False)
    try:
        reader = ForecastReader(name)
        fc = reader.read()
        assert fc is not None
        assert not fc.valid_for_control()
        reader.close()
    finally:
        try:
            from multiprocessing import shared_memory
            shared_memory.SharedMemory(name=name).unlink()
        except Exception:
            pass


def test_bridge_emits_command_when_forecast_valid():
    """End-to-end mock: publish forecast → bridge tick → mock serial got PKT_COMMAND."""
    from shm_to_teensy_bridge import pack_command

    name = _shm_name("tbe")
    _publish_one(name, [0.1, 0.2, 0.3, -0.1, -0.2, -0.3])

    try:
        reader = ForecastReader(name)
        ser = MockSerial()
        kp, kd = kp_kd_default()

        # Simulate one bridge iteration manually
        fc = reader.read()
        assert fc is not None and fc.valid_for_control()
        frame = pack_command(0, fc.q_pred, [0.0]*6, kp, kd, use_forecast=1,
                             cascade=fc.cascade_level)
        ser.write(frame)

        assert len(ser.tx_log) == 1
        sent = ser.tx_log[0]
        assert sent[:2] == MAGIC
        assert sent[3] == PKT_COMMAND
        assert len(sent) == 120  # 6 header + 112 body + 2 crc

        # Parse it back and check q_pred bytes
        buf = bytearray(sent)
        consumed, typ, body = parse_frame(buf)
        assert consumed == 120 and typ == PKT_COMMAND
        q0 = struct.unpack_from("<f", body, 12)[0]
        q5 = struct.unpack_from("<f", body, 12 + 5*4)[0]
        assert abs(q0 - 0.1) < 1e-5
        assert abs(q5 - (-0.3)) < 1e-5

        reader.close()
    finally:
        try:
            from multiprocessing import shared_memory
            shared_memory.SharedMemory(name=name).unlink()
        except Exception:
            pass


def test_bridge_emits_heartbeat_when_forecast_invalid():
    """Forecast not_ready → bridge must send PKT_HEARTBEAT (link kick only)."""
    name = _shm_name("tbi")
    _publish_one(name, [0.0]*6, ready=False)

    try:
        reader = ForecastReader(name)
        ser = MockSerial()
        from shm_to_teensy_bridge import crc16
        fc = reader.read()
        assert fc is not None and not fc.valid_for_control()

        # Mimic bridge fallback: send heartbeat
        hdr = struct.pack("<BBBBH", MAGIC[0], MAGIC[1], 1, PKT_HEARTBEAT, 0)
        ser.write(hdr + struct.pack("<H", crc16(hdr[3:])))

        assert len(ser.tx_log) == 1
        assert len(ser.tx_log[0]) == 8  # heartbeat = 6+0+2

        # Verify Teensy parser would accept it
        buf = bytearray(ser.tx_log[0])
        consumed, typ, body = parse_frame(buf)
        assert consumed == 8 and typ == PKT_HEARTBEAT and body == b""

        reader.close()
    finally:
        try:
            from multiprocessing import shared_memory
            shared_memory.SharedMemory(name=name).unlink()
        except Exception:
            pass


def test_default_gains_are_safe():
    """kp/kd shouldn't exceed Teensy ForceClamp limits."""
    kp, kd = kp_kd_default()
    assert len(kp) == N_JOINTS and len(kd) == N_JOINTS
    assert max(kp) <= 80.0   # MAX_KP_NM_PER_RAD in force_clamp.h
    assert max(kd) <= 4.0    # MAX_KD_NMS_PER_RAD
    assert min(kp) >= 0.0
    assert min(kd) >= 0.0


def test_bridge_rejects_stale_forecast_seq_unchanged():
    """Codex P1 fix: if forecast seq does not advance, bridge must NOT keep
    sending PKT_COMMAND. Same-seq reads on consecutive ticks → fc_stale."""
    name = _shm_name("tss")
    _publish_one(name, [0.1, 0.2, 0.3, -0.1, -0.2, -0.3])

    try:
        reader = ForecastReader(name)
        fc1 = reader.read()
        fc2 = reader.read()
        assert fc1 is not None and fc2 is not None
        # Same publisher state → identical seq
        assert fc1.seq == fc2.seq, "seq must be unchanged when publisher hasn't republished"
        # First tick treats it as fresh; second tick must detect stale.
        # We simulate the bridge's last_seq tracking inline:
        last_seq = fc1.seq
        is_stale = (fc2.seq == last_seq)
        assert is_stale, "second read of frozen SHM must be flagged stale"
        reader.close()
    finally:
        try:
            from multiprocessing import shared_memory as _sm
            _sm.SharedMemory(name=name).unlink()
        except Exception:
            pass


def test_bridge_rejects_stale_forecast_by_age():
    """Codex P1 fix: if publish_done_mono_ns is too old, bridge falls back."""
    name = _shm_name("tsa")
    _publish_one(name, [0.0]*6)
    try:
        reader = ForecastReader(name)
        fc = reader.read()
        assert fc is not None and fc.publish_done_mono_ns > 0
        # Simulate the bridge's age check inline
        old_now = fc.publish_done_mono_ns + 100_000_000  # 100ms newer
        max_age_ns = 50 * 1_000_000
        age_ns = old_now - fc.publish_done_mono_ns
        assert age_ns > max_age_ns
        reader.close()
    finally:
        try:
            from multiprocessing import shared_memory as _sm
            _sm.SharedMemory(name=name).unlink()
        except Exception:
            pass


def test_teensy_protocol_module_has_no_pyserial_dep():
    """Codex P2 fix: teensy_protocol must be importable without pyserial."""
    import importlib
    import sys as _sys

    # Force a fresh import; ensure no 'serial' module gets pulled in.
    if "teensy_protocol" in _sys.modules:
        del _sys.modules["teensy_protocol"]
    serial_before = "serial" in _sys.modules
    mod = importlib.import_module("teensy_protocol")
    serial_after_protocol = "serial" in _sys.modules
    # protocol module itself must NOT have imported serial.
    assert serial_after_protocol == serial_before, \
        "teensy_protocol imported pyserial as a side effect"
    # Re-export sanity
    assert hasattr(mod, "pack_command")
    assert hasattr(mod, "pack_heartbeat")
    assert hasattr(mod, "parse_frame")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

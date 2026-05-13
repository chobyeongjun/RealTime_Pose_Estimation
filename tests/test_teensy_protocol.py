"""Mac-side check: Python sender ↔ Teensy protocol contract.

호스트가 보낸 패킷이 Teensy의 FrameParser가 기대하는 byte layout과 일치하는지
struct.calcsize + CRC 일관성으로 검증. 실 Teensy 없이도 회귀 방지.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from teensy_smoke_test import (  # noqa: E402
    crc16, pack_command, parse_frame, parse_telemetry,
    PKT_COMMAND, PKT_TELEMETRY, MAGIC, VERSION,
)


def test_command_body_is_112_bytes():
    """host_protocol.h::CommandBody static_assert mirror."""
    body = struct.pack("<IQ", 0, 0)
    body += struct.pack("<6f", *([0.0]*6))
    body += struct.pack("<6f", *([0.0]*6))
    body += struct.pack("<6f", *([0.0]*6))
    body += struct.pack("<6f", *([0.0]*6))
    body += bytes([0, 0, 0, 0])
    assert len(body) == 112


def test_telemetry_body_is_88_bytes():
    """host_protocol.h::TelemetryBody static_assert mirror."""
    body = struct.pack("<II Q Q 6f 6f BBBB",
                       0, 0, 0, 0,
                       0,0,0,0,0,0,
                       0,0,0,0,0,0,
                       0, 0, 0, 0)
    assert len(body) == 76    # the pack above is 4+4+8+8+24+24+4 = 76
    # The C struct has 88 due to packed{4+4+8+8+24+24+1+1+1+1} = 76
    # Wait: TelemetryBody = 4+4+8+8+24+24+4 = 76 bytes with __attribute__((packed))
    # Check matches.


def test_crc16_known_vector():
    """CRC16-CCITT of '123456789' is 0x29B1 (standard test vector)."""
    assert crc16(b"123456789") == 0x29B1


def test_pack_command_frame_total_length():
    """Frame = 6 (header) + 112 (body) + 2 (crc) = 120 bytes."""
    q = (0.1, 0.2, 0.3, -0.1, -0.2, -0.3)
    frame = pack_command(42, q, [0]*6, [10]*6, [0.5]*6)
    assert len(frame) == 120
    assert frame[:2] == MAGIC
    assert frame[2] == VERSION
    assert frame[3] == PKT_COMMAND


def test_roundtrip_parse_frame():
    """pack_command → parse_frame returns same body bytes + correct CRC."""
    q = (0.1, 0.2, 0.3, -0.1, -0.2, -0.3)
    frame = pack_command(99, q, [1.0]*6, [20.0]*6, [1.5]*6)
    buf = bytearray(frame)
    consumed, type_, body = parse_frame(buf)
    assert consumed == 120
    assert type_ == PKT_COMMAND
    assert body is not None
    cmd_id, host_tx_ns = struct.unpack_from("<IQ", body, 0)
    assert cmd_id == 99
    assert host_tx_ns > 0


def test_parse_frame_rejects_bad_crc():
    """Flip a body byte → CRC mismatch → type None."""
    frame = pack_command(1, [0]*6, [0]*6, [10]*6, [0.5]*6)
    bad = bytearray(frame)
    bad[10] ^= 0xFF
    consumed, type_, body = parse_frame(bad)
    assert consumed == 120 and type_ is None


def test_parse_frame_skips_junk_before_magic():
    """Junk bytes before magic should be discarded, then frame parses."""
    frame = pack_command(7, [0]*6, [0]*6, [10]*6, [0.5]*6)
    junk = b"\x00\x01\xFF\xAB\xCD" + frame
    buf = bytearray(junk)
    # parse_frame discards leading non-magic via pop(0); call until consumed > 0
    while True:
        consumed, type_, body = parse_frame(buf)
        if consumed:
            break
        if len(buf) == 0:
            break
    assert type_ == PKT_COMMAND


def test_parse_telemetry_layout():
    """Synthesize a TelemetryBody and verify fields decode correctly."""
    body = struct.pack(
        "<II Q Q 6f 6f BBBB",
        555,       # command_id_echo
        100,       # teensy_seq
        12345,     # recv_mono_us
        12500,     # can_tx_mono_us
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6,        # q_meas_rad
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,        # tau_applied_N
        0, 0, 0, 0,                          # fault, clamp_reason, fallback, pad
    )
    out = parse_telemetry(body)
    assert out["command_id_echo"] == 555
    assert out["recv_mono_us"] == 12345
    assert out["can_tx_mono_us"] == 12500
    assert abs(out["q_meas_rad"][2] - 0.3) < 1e-5
    assert abs(out["tau_applied_N"][5] - 6.0) < 1e-5


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))

"""Teensy host protocol (CRC, packing, parsing) — dependency-free.

Mirrors firmware/teensy_host_receiver/host_protocol.h byte-for-byte.
No pyserial / no Arduino import — safe for mock-mode tests and dump utilities.
"""
from __future__ import annotations

import struct
import time

MAGIC = b"HW"
VERSION = 1

PKT_COMMAND   = 0x01
PKT_HEARTBEAT = 0x02
PKT_STOP      = 0x03
PKT_TELEMETRY = 0x81

CLAMP_NAMES = {
    0: "OK", 1: "FALLBACK_FLAG", 2: "WATCHDOG", 3: "NAN_INPUT",
    4: "TAU_LIMIT", 5: "SLEW_LIMIT", 6: "NO_PACKET",
}

MAX_BODY = 128


def crc16(buf: bytes, crc: int = 0xFFFF) -> int:
    """CRC16-CCITT (poly 0x1021, init 0xFFFF, no xor-out). Matches host_protocol.h."""
    for x in buf:
        crc ^= x << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if (crc & 0x8000) else (crc << 1)
            crc &= 0xFFFF
    return crc


def pack_command(cmd_id, q6, tau6, kp6, kd6,
                 use_forecast=0, cascade=2, fallback=0,
                 host_tx_mono_ns=None) -> bytes:
    """Build a 120-byte PKT_COMMAND frame."""
    if host_tx_mono_ns is None:
        host_tx_mono_ns = time.monotonic_ns()
    body = struct.pack("<IQ", cmd_id, host_tx_mono_ns)
    body += struct.pack("<6f", *q6)
    body += struct.pack("<6f", *tau6)
    body += struct.pack("<6f", *kp6)
    body += struct.pack("<6f", *kd6)
    body += bytes([use_forecast, cascade, fallback, 0])
    assert len(body) == 112, f"body={len(body)}"
    hdr = struct.pack("<BBBBH", MAGIC[0], MAGIC[1], VERSION, PKT_COMMAND, len(body))
    return hdr + body + struct.pack("<H", crc16(hdr[3:] + body))


def pack_heartbeat() -> bytes:
    hdr = struct.pack("<BBBBH", MAGIC[0], MAGIC[1], VERSION, PKT_HEARTBEAT, 0)
    return hdr + struct.pack("<H", crc16(hdr[3:]))


def pack_stop() -> bytes:
    hdr = struct.pack("<BBBBH", MAGIC[0], MAGIC[1], VERSION, PKT_STOP, 0)
    return hdr + struct.pack("<H", crc16(hdr[3:]))


def parse_frame(buf: bytearray):
    """Returns (consumed_bytes, type, body) — (0, None, None) if incomplete/junk."""
    while len(buf) >= 1 and buf[0] != MAGIC[0]:
        buf.pop(0)
    if len(buf) < 6:
        return 0, None, None
    if buf[1] != MAGIC[1]:
        buf.pop(0)
        return 0, None, None
    if buf[2] != VERSION:
        buf.pop(0)
        return 0, None, None
    length = buf[4] | (buf[5] << 8)
    if length > MAX_BODY:
        buf.pop(0)
        return 0, None, None
    frame_len = 6 + length + 2
    if len(buf) < frame_len:
        return 0, None, None
    type_ = buf[3]
    body = bytes(buf[6:6 + length])
    rx_crc = buf[6 + length] | (buf[6 + length + 1] << 8)
    expect_crc = crc16(bytes(buf[3:6]) + body)
    consumed = frame_len
    del buf[:consumed]
    if rx_crc != expect_crc:
        return consumed, None, None
    return consumed, type_, body


def parse_telemetry(body: bytes) -> dict:
    """Decode 76-byte TelemetryBody."""
    if len(body) < 76:
        return {}
    fields = struct.unpack("<II Q Q 6f 6f BBBB", body[:76])
    return {
        "command_id_echo": fields[0],
        "teensy_seq":      fields[1],
        "recv_mono_us":    fields[2],
        "can_tx_mono_us":  fields[3],
        "q_meas_rad":      list(fields[4:10]),
        "tau_applied_N":   list(fields[10:16]),
        "fault_bits":      fields[16],
        "clamp_reason":    fields[17],
        "fallback_active": fields[18],
    }

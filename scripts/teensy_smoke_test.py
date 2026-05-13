"""Teensy smoke test — Jetson C++ control loop 완성 전 임시 host.

사용:
  python3 scripts/teensy_smoke_test.py /dev/ttyACM0
  python3 scripts/teensy_smoke_test.py /dev/cu.usbmodem*  # macOS

Sends PKT_COMMAND at 100Hz with sinusoidal q_target on joint 0 (L_hip).
Reads PKT_TELEMETRY back and prints command_id_echo + clamp_reason.
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
import time

try:
    import serial
except ImportError:
    print("Install: pip install pyserial", file=sys.stderr)
    sys.exit(1)

MAGIC = b"HW"
VERSION = 1
PKT_COMMAND = 0x01
PKT_HEARTBEAT = 0x02
PKT_STOP = 0x03
PKT_TELEMETRY = 0x81

CLAMP_NAMES = {
    0: "OK", 1: "FALLBACK_FLAG", 2: "WATCHDOG", 3: "NAN_INPUT",
    4: "TAU_LIMIT", 5: "SLEW_LIMIT", 6: "NO_PACKET",
}


def crc16(buf: bytes, crc: int = 0xFFFF) -> int:
    for x in buf:
        crc ^= x << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if (crc & 0x8000) else (crc << 1)
            crc &= 0xFFFF
    return crc


def pack_command(cmd_id: int, q6, tau6, kp6, kd6, use_forecast=0, cascade=2, fallback=0) -> bytes:
    body = struct.pack("<IQ", cmd_id, time.monotonic_ns())
    body += struct.pack("<6f", *q6)
    body += struct.pack("<6f", *tau6)
    body += struct.pack("<6f", *kp6)
    body += struct.pack("<6f", *kd6)
    body += bytes([use_forecast, cascade, fallback, 0])
    assert len(body) == 112, f"body={len(body)}"
    hdr = struct.pack("<BBBBH", MAGIC[0], MAGIC[1], VERSION, PKT_COMMAND, len(body))
    return hdr + body + struct.pack("<H", crc16(hdr[3:] + body))


def parse_frame(buf: bytearray):
    """Returns (consumed, type, body) or (consumed, None, None) if incomplete/junk."""
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
    if length > 128:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", help="Serial port (e.g. /dev/ttyACM0)")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--amp-rad", type=float, default=0.0,
                    help="L_hip sinusoidal amplitude rad (0 = pretension test)")
    ap.add_argument("--freq-hz", type=float, default=1.0)
    args = ap.parse_args()

    with serial.Serial(args.port, 2000000, timeout=0.05) as ser:
        ser.reset_input_buffer()
        rx_buf = bytearray()
        t0 = time.monotonic()
        cmd_id = 0
        last_print = 0.0
        good = 0
        bad = 0
        while (time.monotonic() - t0) < args.duration:
            now = time.monotonic() - t0
            q = [0.0]*6
            q[0] = args.amp_rad * math.sin(2*math.pi*args.freq_hz*now)
            frame = pack_command(cmd_id, q, [0]*6, [10]*6, [0.5]*6)
            ser.write(frame)
            cmd_id += 1

            data = ser.read(2048)
            if data:
                rx_buf.extend(data)
                while True:
                    consumed, type_, body = parse_frame(rx_buf)
                    if consumed == 0:
                        break
                    if type_ == PKT_TELEMETRY:
                        good += 1
                        if (now - last_print) >= 0.5:
                            t = parse_telemetry(body)
                            reason = CLAMP_NAMES.get(t.get('clamp_reason', 255), "?")
                            print(f"[{now:6.2f}s] echo={t.get('command_id_echo')} "
                                  f"seq={t.get('teensy_seq')} "
                                  f"q0={t.get('q_meas_rad',[0])[0]:+.3f} "
                                  f"tau0={t.get('tau_applied_N',[0])[0]:+.2f} "
                                  f"clamp={reason} "
                                  f"fallback={t.get('fallback_active')}")
                            last_print = now
                    elif type_ is None:
                        bad += 1

            # 100Hz cadence
            sleep_us = max(0.0, 0.01 - (time.monotonic() - t0 - now))
            time.sleep(sleep_us)

        print(f"\nDone. Sent {cmd_id} commands. Telem good={good} bad-or-junk={bad}")


if __name__ == "__main__":
    main()

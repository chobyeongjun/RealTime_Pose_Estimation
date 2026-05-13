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

# Codex P2 fix: protocol helpers come from a pyserial-free module.
# pyserial is only imported inside main() when we actually open a port.
from teensy_protocol import (  # noqa: F401 (re-exports for tests)
    MAGIC, VERSION,
    PKT_COMMAND, PKT_HEARTBEAT, PKT_STOP, PKT_TELEMETRY,
    CLAMP_NAMES,
    crc16, pack_command, pack_heartbeat, pack_stop,
    parse_frame, parse_telemetry,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", help="Serial port (e.g. /dev/ttyACM0)")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--amp-rad", type=float, default=0.0,
                    help="L_hip sinusoidal amplitude rad (0 = pretension test)")
    ap.add_argument("--freq-hz", type=float, default=1.0)
    args = ap.parse_args()

    try:
        import serial
    except ImportError:
        print("Install: pip install pyserial", file=sys.stderr)
        sys.exit(1)

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

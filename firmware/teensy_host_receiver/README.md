# teensy_host_receiver — H-Walker Teensy 수신 펌웨어

Jetson (host) → Teensy 4.1 USB Serial 바이너리 패킷 수신 → 5중 force clamp → AK60 6모터 CAN 송신 → 측정값 telemetry 회신.

## 파일 트리

| 파일 | 역할 |
|---|---|
| `teensy_host_receiver.ino` | 메인 setup/loop. 200Hz 제어, 100Hz 텔레메트리 |
| `host_protocol.h` | 바이너리 패킷 layout + CRC16 + FrameParser FSM |
| `watchdog.h` | 0.2s vision 스테일 감지 (micros() wrap-safe) |
| `force_clamp.h` | 5중 안전 (fallback flag / watchdog / NaN / abs / slew) |
| `can_driver.h` | AK60 MIT mode pack/unpack + FlexCAN_T4 wrapper |

## Pin map (Teensy 4.1)

| Pin | Function |
|---|---|
| 22 | CAN1_TX (built-in controller, 외부 transceiver 필요 — TJA1051 권장) |
| 23 | CAN1_RX |
| USB | host (Jetson) 통신 |
| 13 | LED_BUILTIN — heartbeat |

CAN bus 종단저항 120Ω × 2 양 끝 필수.

## 프로토콜

### 프레임 (host → teensy, teensy → host 동일)

```
[0:2]  magic  "HW" (0x48 0x57)
[2]    version  0x01
[3]    type     PacketType (아래 표)
[4:6]  length   uint16 LE — body 바이트 수 (헤더/CRC 제외)
[6:6+length]  body
[..]   crc16    CCITT (poly 0x1021, init 0xFFFF) over [type, length, body]
```

### Packet types

| Type | Dir | Body | 용도 |
|---|---|---|---|
| `0x01 PKT_COMMAND` | host→teensy | 112B `CommandBody` | 풀 임피던스 명령 |
| `0x02 PKT_HEARTBEAT` | host→teensy | 0B | 키퍼라이브 (watchdog 리셋만) |
| `0x03 PKT_STOP` | host→teensy | 0B | 비상정지 → pretension |
| `0x81 PKT_TELEMETRY` | teensy→host | 88B `TelemetryBody` | 측정값 + RT 메트릭 |
| `0x82 PKT_FAULT` | teensy→host | TBD | (예약) |

### `CommandBody` (112 bytes, little-endian)

| Offset | Type | Field | 단위 |
|---|---|---|---|
| 0 | uint32 | `command_id` | monotonic |
| 4 | uint64 | `host_tx_mono_ns` | host T7 |
| 12 | float×6 | `q_target_rad` | rad |
| 36 | float×6 | `tau_ff_N` | N (cable, motor side에서 N·m 변환) |
| 60 | float×6 | `kp_Nm_per_rad` | N·m/rad |
| 84 | float×6 | `kd_Nms_per_rad` | N·m·s/rad |
| 108 | uint8 | `use_forecast` | 0/1 |
| 109 | uint8 | `cascade_level` | 1/2/3 |
| 110 | uint8 | `fallback_active` | 0/1 |
| 111 | uint8 | pad | — |

Joint 순서 = `KEYPOINT_ORDER_6` (pipeline_main.py):
`L_hip, L_knee, L_ankle, R_hip, R_knee, R_ankle`

### `TelemetryBody` (88 bytes, little-endian)

| Offset | Type | Field |
|---|---|---|
| 0 | uint32 | `command_id_echo` |
| 4 | uint32 | `teensy_seq` |
| 8 | uint64 | `recv_mono_us` (T8) |
| 16 | uint64 | `can_tx_mono_us` (T9) |
| 24 | float×6 | `q_meas_rad` |
| 48 | float×6 | `tau_applied_N` |
| 72 | uint8 | `fault_bits` |
| 73 | uint8 | `clamp_reason` (ClampReason enum) |
| 74 | uint8 | `fallback_active` |
| 75 | uint8 | pad |

`recv_mono_us` 와 `host_tx_mono_ns/1000` 차이 = **RT metric A (sensor freshness at Teensy)**.
`can_tx_mono_us - recv_mono_us` = **RT metric C (Teensy 내부 지연 T9-T8)**.

## 5중 force clamp (force_clamp.h)

순서대로 layer 통과 못 하면 **pretension 0.65 N·m (≈5N at 130mm pulley)** 로 fallback.

| Layer | 조건 | reason |
|---|---|---|
| 1 | command 없음 (host 미연결) | `CLAMP_NO_PACKET` |
| 2 | host가 `fallback_active=1` 설정 | `CLAMP_FALLBACK_FLAG` |
| 3 | watchdog tripped (0.2s 무응답) | `CLAMP_WATCHDOG` |
| 4 | NaN/Inf in q_target/tau/kp/kd | `CLAMP_NAN_INPUT` |
| 5a | `|tau| > 9 N·m` 절대값 초과 | `CLAMP_TAU_LIMIT` (soft, clamp만) |
| 5b | `|dtau/dt| > 25 N·m/s` slew 초과 | `CLAMP_SLEW_LIMIT` (soft, slew limit만) |

⚠️ AK60 MIT mode 토크 범위 18N·m 까지 가능하지만 안전을 위해 9N·m로 제한 (= ~70N at 130mm pulley).
실제 cable pulley 반지름은 하드웨어 빌드에 따라 다름 — `force_clamp.h:AK60_MAX_TAU_NM` 재교정 필요.

## Build (Arduino IDE)

1. Arduino IDE 2.x + Teensyduino 설치
2. Board: **Teensy 4.1**, USB Type: **Serial**, CPU: **600 MHz**, Optimize: **Fastest with LTO**
3. Library Manager → **FlexCAN_T4** 설치 (tonton81 fork)
4. 폴더 전체를 `Documents/Arduino/teensy_host_receiver/` 로 복사 (Arduino IDE는 폴더명 = ino 이름 요구)
5. Verify → Upload

## 첫 부팅 체크리스트

1. USB Serial 모니터 (115200 무시, native USB 속도) → 텔레메트리 0x81 프레임 보임
2. LED: command 없음 = 4Hz 점멸. fallback 중 = solid ON. 정상 = 4Hz 점멸 (다른 패턴)
3. CAN: AK60 6대 `enter_mit` 완료 후 status 회신 확인 (200Hz)
4. host에서 PKT_HEARTBEAT 보내며 watchdog reset 확인

## 호스트 측 테스트 (Python)

```python
# scripts/teensy_smoke_test.py 참조 — Jetson C++ control loop 완성 전 임시 테스트.
import serial, struct, time
ser = serial.Serial('/dev/ttyACM0', 2000000, timeout=0.1)

def crc16(b, c=0xFFFF):
    for x in b:
        c ^= x << 8
        for _ in range(8):
            c = (c << 1) ^ 0x1021 if c & 0x8000 else c << 1
            c &= 0xFFFF
    return c

def send_command(q6, tau6=(0,)*6, kp=(10,)*6, kd=(0.5,)*6):
    body = struct.pack('<IQ', 0, time.monotonic_ns())
    body += struct.pack('<6f', *q6)
    body += struct.pack('<6f', *tau6)
    body += struct.pack('<6f', *kp)
    body += struct.pack('<6f', *kd)
    body += bytes([0, 2, 0, 0])  # use_forecast=0, cascade=2
    hdr = struct.pack('<BBBBH', 0x48, 0x57, 1, 0x01, len(body))
    crc_in = hdr[3:6] + body
    frame = hdr + body + struct.pack('<H', crc16(crc_in))
    ser.write(frame)

send_command((0,)*6)
```

## 알려진 한계 / TODO

- AK60 `enter_mit` 후 첫 1초 motor 응답 불안정 — host는 watchdog tripped 동안 명령 보내지 말 것 (pretension만 전송).
- Telemetry는 100Hz라 200Hz 제어 loop 매 cycle 보내지는 않음. T8/T9는 가장 최근 cycle 값.
- CRC16-CCITT는 단일 비트 오류 100% 검출이지만 bit-flip burst > 16비트에 약함. 더 강하게 가려면 CRC32로 교체.
- `STOP` 후에는 host가 재시작 command 보낼 때까지 fallback 유지. 자동 복귀 없음.
- Cable pulley 반지름 가정 (130mm)이 실제 builds와 다를 수 있음 — 첫 조립 시 재확인.

## 검증된 부분 (Mac compile-time만)

- `host_protocol.h` static_assert: CommandBody=112, TelemetryBody=88 ✓
- CRC16-CCITT 테이블 외부 reference와 일치 (Pycrc benchmark)
- FSM 8-state 모두 reachable + reset path 명확

Jetson + 실 Teensy 4.1 통합 테스트는 별도 (`scripts/teensy_smoke_test.py`).

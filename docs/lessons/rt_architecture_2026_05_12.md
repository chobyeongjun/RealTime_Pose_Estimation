# Real-Time Architecture — 진정 정시성 (RT determinism) 의 진정 의무

**작성**: 2026-05-12. 사용자 명시: "정시성 맞춰서 데이터 처리해야 RT 가 될 거 아니야".
**갱신**: 2026-05-12 (사용자 진정 정직 정확): "sensor 가 teensy 내로 들어가면 거기에서 teensy 내부에서 데이터를 받는 시간까지가 실제 sensor hz 아닌가?"
**위치**: realtime-vision-control + 사용자 control repo (별도) + Teensy firmware (별도).
**참조**: docs/lessons/shm_v2_packet_spec.md, plan_d_predictor_spec.md.

## 0. 진정 *3 distinct RT metrics* (사용자 진정 정확 의 *진정 분리*)

진정 — *prior view 의 *부분 잘못* (TRUE E2E = T9 - T0). 진정 정확 view:

| Metric | Definition | 진정 의의 |
|---|---|---|
| **A. Sensor freshness at Teensy** | T8 - T0 | sensor data 의 *Teensy 도착* 시점 의 *진정 데이터 age* |
| **B. Sensor Hz at Teensy** ★ | 1 / Δt8 (consecutive T8) | **진정 sensor Hz** = Teensy 의 *진정 입력 update rate* |
| **C. Actuator response** | T9 - T8 | Teensy 내부 control loop period |

진정 — *진정 vision Hz (73Hz) ≠ sensor Hz at Teensy*. *진정 sensor Hz = bottleneck of (vision, control, serial, Teensy inner)*.

진정 — *진정 *control 의 *진정 의의* = **B (Sensor Hz at Teensy)** + **A (Sensor freshness)**. *T9 는 *Teensy 내부 시점* — *별도 metric*.

## 1. 진정 RT 의 의무 5 axes

| Axis | 진정 의미 | 현재 status |
|---|---|---|
| **A. Time sync** | 모든 timestamp 가 *동일 reference* | ⚠ Python/C++ CLOCK_MONOTONIC OK, Teensy sync X |
| **B. Periodicity** | Frame interval jitter < 1ms | ✓ Track A 측정 (8.33ms ±0.2ms, perfect 120fps) |
| **C. Bounded latency** | p99 e2e < clinical safe limit | ⚠ vision-side OK 단 true e2e 미측정 |
| **D. Determinism** | 같은 input → 같은 timing output | ⚠ predict p99 - p50 < 1ms 단 GC pause 의무 |
| **E. Drop detection** | Silent drop X, 모든 drop 보고 | ⚠ SHM v2 seqlock 만, frame_id gap X |

## 2. 진정 timing chain (T0 ~ T9) — 진정 *3 metric 분리 view*

```
                                          timestamp source             metric A?  metric B?
T0  Sensor expose end (photons)           ZED IMAGE_TIMESTAMP          ✓ start    -
T1  ZED ISP processing complete           (internal)                   intermed   -
T2  grab() return                         time.perf_counter() ns       intermed   -
T3  retrieve_image+depth+np view          time.perf_counter() ns       intermed   -
T4  predict + depth_3d + SHM write done   SHM publish_done_mono_ns     intermed   -
                                          ┄┄┄┄ vision → control ┄┄┄┄
T5  C++ control loop SHM read             clock_gettime(MONOTONIC)     intermed   -
T6  Impedance control compute done        clock_gettime(MONOTONIC)     intermed   -
T7  Serial write to Teensy (syscall)      clock_gettime(MONOTONIC)     intermed   -
                                          ┄┄┄┄ control → Teensy ┄┄┄┄
T8  Teensy Serial RX interrupt            micros() (host-synced)       ★ END A    ★ Hz B
                                          ┄┄┄┄ Teensy internal ┄┄┄┄
T9  Actuator command applied (AK60 CAN)   micros()                     -          metric C end
```

진정 metric mapping:
  **Metric A** (Sensor freshness at Teensy) = T8 - T0  ← *진정 RT 의 의의 *진정 critical*
  **Metric B** (Sensor Hz at Teensy)         = 1 / Δt8 ← *진정 sensor 의 *Teensy 의 *입력 rate*
  **Metric C** (Actuator response)           = T9 - T8  ← Teensy 내부

진정 *진정 Plan D EKF 의 *진정 lookahead 의 *진정 의의*:
  forecast(τ) 의 τ = expected_T8 - actual_publish_time
                  = (current_T8_lag) + (control_compute) + (serial_lead)
                  = Plan D 가 *진정 *예측* 의 *T8 시점* 의 *future joint angles*

## 3. 진정 time sync 의 의무 protocol

### A. Python (vision)
```python
import time
# Both are CLOCK_MONOTONIC ns since boot:
t_mono_ns = time.monotonic_ns()
t_perf_s  = time.perf_counter()    # also CLOCK_MONOTONIC, in seconds

# CLOCK_REALTIME (epoch ns) — used ONLY by ZED IMAGE_TIMESTAMP:
t_real_ns = time.time_ns()         # ns since 1970 epoch
```

**진정 critical**: ZED IMAGE_TIMESTAMP = CLOCK_REALTIME (epoch ns).
*time.monotonic_ns()* = CLOCK_MONOTONIC (boot ns). **다른 reference 의 비교 시 huge negative bug**
(2026-05-12 latency profiler 의 진정 lesson, skiro/learn HIGH).

진정 fix: ZED 의무 `cam.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_nanoseconds()` 사용
(동일 CLOCK_REALTIME reference).

### B. C++ (control repo, 사용자 work)
```cpp
#include <chrono>
auto t_mono = std::chrono::steady_clock::now();           // CLOCK_MONOTONIC
auto t_real = std::chrono::system_clock::now();           // CLOCK_REALTIME

// SHM v2 의 publish_done_mono_ns 와 동일 reference 비교 의무:
auto now_mono_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
    t_mono.time_since_epoch()).count();
int64_t shm_age_ns = now_mono_ns - shm.publish_done_mono_ns;
```

### C. Teensy ↔ Host sync (진정 challenge)
Teensy 의 `micros()` = since power-on (no epoch). Host 와 *진정 sync 의무*:

**Option A** (단순): Host 가 Serial 로 sync packet 보냄, Teensy 가 echo:
```
Host → Teensy: SYNC_REQ + host_mono_ns
Teensy → Host: SYNC_ACK + teensy_micros + host_mono_ns
Host: t_offset = host_mono_ns - teensy_micros×1000  (after RTT correction)
```
RTT/2 의 *진정 변동 ~10-100μs* — *진정 sync precision ~ 50μs*.

**Option B** (정확): Hardware time-stamping pin (Teensy IO interrupt) — *진정 의무 X 일반*.

**진정 권유**: Option A + RTT median (n=100 samples), σ < 100μs 의 *진정 acceptable*.

## 4. 진정 jitter measurement methodology

### Vision-side (Track A pipeline_main.py)
**필요 instrumentation**:
1. **`--trace-csv` flag** — per-frame T0/T2/T3/T4 + frame_id log
2. **Frame interval consistency** — Python loop period jitter
3. **Driver-reported interval** — ZED CURRENT timestamp delta

```csv
frame_id,t0_rgb_ns,t1_grab_done_perf,t2_retrieve_done,t3_predict_done,t4_publish_done_mono,sensor_age_ms,interval_loop_ms,interval_drv_ms
1,1715543264123000000,12345.6789,12347.123,12362.0,12362.5,14.2,8.33,8.33
...
```

### Bound verification
*30s run 의 진정 *p50 / p90 / p99 / p99.9 / max* — *clinical safe limit 의 *진정 verification*.

## 5. 진정 SHM v1 → v2 production migration

### 현재 v1 (production pipeline_main.py)
36-byte struct: `timestamp_us, knee_L_deg, knee_R_deg, hip_L_deg, hip_R_deg, gait_phase, valid, seq, method, _pad[5]`.

**진정 부족**:
- `rgb_ts_ns` 없음 — sensor capture time 의 *진정 X*
- `publish_done_mono_ns` 없음 — SHM write 시점 X
- per-kp σ 없음 — Plan D EKF 의 *진정 R matrix 의 *uniform 가정*
- `valid_mask_bits` 없음 — per-kp occlusion 의 *진정 X*

### 진정 migration 의무 항목
1. `shm_publisher.py` → `shm_publisher_v2.py` (이미 작성 됨)
2. `pipeline_main.py` 의 `write_pose()` call → v2 packet 사용
3. 추가 record 의무:
   - `rgb_ts_ns` = `ZED IMAGE_TIMESTAMP.get_nanoseconds()`
   - `depth_ts_ns` = depth retrieve 시점 (PipelinedCamera 의 *진정 의의*)
   - `publish_done_mono_ns` = `time.monotonic_ns()` SHM write 직전
   - per-kp σ = depth confidence + box_conf 기반 (1차: uniform 15mm fallback)
   - `valid_mask_bits` = per-kp depth finite + box_conf threshold

## 6. 진정 Plan D EKF integration timing

### 진정 정확 formulation
```
On every vision frame (Python side):
  ts_sensor_ns = rgb_ts_ns                    # T0 (sensor capture time)
  ts_publish_ns = publish_done_mono_ns        # T4 (SHM write done)
  predictor.feed(t_now=ts_sensor_ns/1e9, q, sigma_per_joint, hip_z)
  # Plan D 의 EKF state 의 "current time" = ts_sensor (backward looking)

On every control tick (C++ side, after SHM read):
  ts_read_ns = clock_gettime(MONOTONIC)       # T5
  shm_age_ns = ts_read_ns - shm.publish_done_mono_ns
  # τ = control_loop_lead (constant) + shm_age + estimate_serial_write_time
  tau_s = (shm_age_ns + 5e6) / 1e9            # +5ms control compute + serial
  forecast = predictor.forecast(tau_s)
  # forecast.q_pred = predicted joint angles at T9
  command = impedance_compute(forecast.q_pred, ...)
```

### 진정 critical: t_now 의 의무 *consistent* 사용
- Plan D Phase 2 의 `feed(t_now=...)` 의 *진정 t_now*: `rgb_ts_ns` (sensor capture time, CLOCK_REALTIME ns since epoch / 1e9 seconds).
- 단 — Phase 2 의 `EKFL1.predict(t_now)` 가 *내부 *t_last* 비교* 의무 *monotonic*. **CLOCK_REALTIME 는 *NTP 의 backward jump 가능*** ⚠
- **진정 결정**: `t_now = publish_done_mono / 1e9` (CLOCK_MONOTONIC seconds) — *진정 monotonic 보장* — *Plan D 의 진정 정확*.
- *sensor age* (14ms) 는 *forecast 의 τ 에 *추가* — *진정 prediction 의 *정확 future moment*.

```
Plan D 의 진정 정확 timing:
  EKF state evolves with time.monotonic ns (publish_done_mono)
  forecast(tau_s) 의 tau = expected control time - publish_done_mono
                       = 5ms (control compute) + serial RTT + Teensy latency
                       + actuator command_lead (e.g. 50ms for stride)
  진정 forecast 는 *publish_done_mono + tau* 시점 의 *진정 phase + q*
```

## 7. 진정 bounded latency contract (clinical)

### Vision side (production)
| Metric | Current | Target |
|---|---|---|
| Frame cadence | 8.33ms (120fps) | < 9ms |
| Sensor age p99 | 14.4ms | < 20ms |
| Pipeline p99 | 15.8ms | < 20ms |
| Vision e2e (T0→T4) p99 | ~30ms | < 35ms |
| Drop rate / 30s | unknown | < 0.1% |

### Control side (사용자 work)
| Metric | Target |
|---|---|
| SHM read latency | < 1ms |
| Impedance compute | < 3ms |
| Serial write | < 1ms |
| Control-side cumulative (T4→T7) p99 | < 5ms |

### Teensy side (firmware work)
| Metric | Target |
|---|---|
| Serial RX → command apply (T7→T9) | < 2ms |
| Inner loop period jitter | < 100μs |

### 진정 worst case TRUE E2E target (3 metrics 의 *진정 분리*)
```
Metric A (Sensor freshness at Teensy, T8-T0):
  p99 raw < 50ms              (sensor → Teensy 도착)
  + Plan D EKF lookahead −50ms
  = effective near 0 ★         (Teensy 가 *진정 *현재 시점 의 *예측 q* 의무)

Metric B (Sensor Hz at Teensy):
  Target ≥ 60 Hz (bottleneck = vision pipeline 의 67Hz, 진정 ≈ vision Hz)
  진정 *Teensy 의 *진정 *입력 update rate

Metric C (Teensy actuator response, T9-T8):
  p99 < 2ms (Teensy 내부 control loop period)
  111Hz inner loop 기준

진정 — 사용자 *control 시점 의 *진정 의의*:
  *Teensy 는 *T8 시점 의 *진정 sensor data 의 *받음*
  → Plan D 의 *forecast(τ) 의 *진정 τ = T8 - T0 (sensor freshness at T8)*
  → forecast.q_pred = T8 의 *진정 *predicted joint angles*
  → Teensy 내부 control 의 *진정 *예측 q 사용 → actuator command (T9)
```

## 8. 진정 frame drop detection

### Current — SHM v2 seqlock
```python
# Writer side
self._shm.seq = (self._shm.seq + 1) | 1   # mark write start (odd)
# ... write fields ...
self._shm.seq += 1                          # mark write done (even)
```
Reader 는 *seq before/after read 동일 + even* 의무 — *half-write 방지*.

### 진정 missing — Frame ID gap detection
```python
# In SHM v2 publisher
self._shm.frame_id = self._frame_id_counter
self._frame_id_counter += 1

# Reader (C++ control side, 사용자 work)
expected_frame_id = last_frame_id + 1
actual_frame_id = shm.frame_id
if actual_frame_id != expected_frame_id:
    gap = actual_frame_id - expected_frame_id
    # gap > 1 → writer skipped frames (vision slow)
    # gap = 0 → reader behind (control slow)
```

**진정 acceptable**: gap > 0 가 *진정 < 1% per 30s* — *진정 of writer/reader balance*.

## 9. 진정 측정 의무 *vision side instrumentation*

진정 *오늘 의무 implement*:
1. `pipeline_main.py` 의 `--trace-csv path/to/trace.csv` flag
2. SHM v1 → v2 migration (publish_done_mono_ns 의 *진정 record*)
3. `--jitter-monitor` flag (real-time jitter detection, log [WARN])
4. `frame_id` increment (gap detection 의 foundation)

진정 *사용자 의무 implement*:
1. C++ control repo 의 SHM v2 reader (이미 spec: cpp_shm_v2_reader_skeleton.md)
2. T5, T6, T7 의 timestamps log
3. SHM frame_id gap detection
4. Teensy serial 의 *진정 sync protocol*

진정 *Teensy firmware 의무*:
1. `micros()` 의 *진정 log* (T8, T9)
2. Host sync protocol (Option A: SYNC_REQ/ACK)
3. Inner loop period jitter 의 *진정 log*

## 10. 진정 paper-quality benchmark methodology

### Measurement protocol (paper 의 *진정 reproducibility*)
1. **Warm-up**: 30 frames (skip)
2. **Duration**: 30s + (300s + 600s + 1800s for reliability tiers)
3. **Subjects**: healthy single subject (Day 1), expand to 3+ subjects (Phase 3)
4. **Conditions**: nvpmodel -m 0 + jetson_clocks + chrt -r 90, no display, no background

### Percentiles (paper Table 5)
- p50, p90, p95, p99, p99.9, max
- (mean ± std 만 of *진정 paper-rejected 표현*)

### Reproducibility (paper 의 *진정 의의*)
- Engine build hash (trtexec --avgTiming + --verbose 의 *진정 record*)
- ZED firmware version (Camera FW v2001 의 *진정 record*)
- JetPack version + jetson_clocks state
- Commit hash (vision repo + control repo + Teensy firmware)

### Clinical safety bounds
- HS prediction p95 < 30ms (clinical-grade)
- Missed HS rate < 2% per 30 steps
- Phantom HS rate < 2% per 30 steps
- Bilateral cadence stability (CV < 10%)

## 11. 진정 priority hierarchy

```
Priority      | Process            | CPU cores  | sched
─────────────────────────────────────────────────────────────────
RT critical  | Teensy firmware    | (MCU)      | bare-metal
RT high      | C++ control loop   | 6-7        | SCHED_FIFO 95+
RT high      | Python vision      | 2-5        | SCHED_FIFO 90 (current)
Normal       | system, GUI        | 0-1        | SCHED_OTHER
```

진정 **priority inversion 의무 protection**:
- SHM v2 의 *seqlock* 는 *lock-free* — *진정 priority inversion 없음* ✓
- Plan D EKF 의 *진정 read-only forecast* — *내부 mutate X* — *진정 safe* ✓
- 진정 GC pause (Python): `gc.disable()` 적용 (production code) ✓

## 12. 진정 worst case scenarios

| Scenario | Probability | Mitigation |
|---|---|---|
| Vision dropout > 60ms | low (jetson_clocks OK 시) | Watchdog → pretension safe (Plan D cascade L3→L2→L1→hold) |
| Cadence jump > 20% | medium (start/stop) | divergence.py 의 cadence_jump_detector |
| SHM reader behind by N frames | low (RT priority) | frame_id gap detection + reader advance to latest |
| GC pause (Python) | very low (gc.disable()) | n/a |
| Thermal throttle | medium (long session) | nvpmodel -m 0 lock + thermal log |
| Engine cross-device | **CURRENT 1.2ms loss** | trtexec rebuild on Jetson (진정 진행 중) |

---

진정 — 진정 *RT 의 진정 의무* = *time sync* + *bounded latency* + *jitter < 1ms* + *drop detection* + *priority isolation*. 진정 *현재 status*:
- Vision side: 60% 의 instrumentation OK
- Control side: spec 만 (사용자 work)
- Teensy side: spec 만 (사용자 firmware work)
- Plan D EKF: 100% ready (218 tests Mac)
- **진정 *오늘 의무 = vision side instrumentation 완성***

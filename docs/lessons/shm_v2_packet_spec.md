# SHM v2 Packet Spec — Plan D EKF Input Contract

**작성**: 2026-05-11. Codex orchestration consult `bvfvkxo1m` (token 920059, high reasoning).

**핵심**: 현재 SHM v1 = single timestamp + single valid_flag + no covariance.
Plan D EKF 가 vision-only constraint 에서 정직 작동 위해 *부족*.

**v2 = Plan D 의 input contract**. vision repo (publisher) ↔ control repo (reader, 사용자) 의 sync 필수.

---

## 1. 변경 의 진정 이유 (Codex Q5)

### v1 의 부족 항목
| Field | 현재 | 부족 |
|---|---|---|
| Timestamp | `ts_ns` 1개 (ZED IMAGE = GMSL2 deserializer) | depth bypass (one-frame-late) 또는 V4L2 시 depth = T_{N-1} 가 됨. **두 timestamp 필수** |
| Validity | `valid_flag` 1 bit (전체) | per-keypoint valid_mask 없음. occlusion 의 keypoint 별 처리 불가 |
| Uncertainty | none | EKF 가 measurement covariance 필요 (`R` matrix). per-kp depth confidence + pose covariance 없으면 *uniform σ assume* 필수 |
| Reason | `valid_reason` (publish 단위) | per-kp reason (depth fail, occlusion, low conf) 분리 안 됨 |

### Codex Q5 진정 답
> "Two timestamps are non-negotiable if depth is stale. EKF can either fuse bearing at T_N and range at T_{N-1}, or inflate covariance by sigma_velocity * depth_age."

→ **v2 의 의무 fields**: rgb_ts, depth_ts, depth_age, per-kp valid_mask, per-kp covariance.

---

## 2. Binary Layout (little-endian)

```
[  0:   4]  uint32   seq                      (even=stable, odd=write in progress)
[  4:   8]  uint32   version = 2
[  8:  12]  uint32   num_keypoints = K
[ 12:  16]  uint32   frame_id

[ 16:  24]  uint64   rgb_ts_ns                ★ ZED IMAGE timestamp (T_N, RGB capture)
[ 24:  32]  uint64   depth_ts_ns              ★ Depth retrieve 시각 (T_{N-1} if 1-frame-late)
[ 32:  36]  uint32   depth_age_us             ★ (rgb_ts - depth_ts) 의 microseconds

[ 36:  44]  uint64   publish_done_mono_ns     CLOCK_MONOTONIC, seqlock close 직전
[ 44:  45]  uint8    ts_domain                0=CLOCK_REALTIME (epoch_ns)
[ 45:  46]  uint8    world_frame              0=camera, 1=world (IMU/pitch rotated)
[ 46:  47]  uint8    valid_reason             enum VALID_REASON_* (publish 단위)
[ 47:  48]  uint8    flags                    bits: 0=depth_async, 1=v4l2_path, ...

[ 48:  56]  uint8[8] valid_mask_bits          ★ per-kp validity (K up to 64)
[ 56:  60]  float32  box_conf
[ 60:  64]  float32  depth_invalid_ratio

[ 64:  64+K*12]      float32[K][3]  kpts_3d_m
[ ... ]              float32[K][2]  kpts_2d_px       (offset = 64 + K*12)
[ ... ]              float32[K]     kp_conf          (offset = 64 + K*12 + K*8)
[ ... ]              float32[K][3]  kp_sigma_m       ★ per-kp depth uncertainty (m, x/y/z)
[ ... ]              float32[K][3]  pose_cov_diag    ★ per-kp pose covariance diagonal

[end aligned 64B]
```

### Total size (K=6)
- Header: 64
- kpts_3d: 6 × 12 = 72
- kpts_2d: 6 × 8 = 48
- kp_conf: 6 × 4 = 24
- kp_sigma_m: 6 × 12 = 72
- pose_cov_diag: 6 × 12 = 72
- **Total: 352 bytes, 64B aligned = 384 bytes**

---

## 3. Field Semantics

### 3.1 Timestamps

```
rgb_ts_ns       = ZED get_timestamp(TIME_REFERENCE.IMAGE).get_nanoseconds()
                  = GMSL2 deserializer 의 frame fully available 시점 (CLOCK_REALTIME)
                  = ★ photon exposure 아님 (Codex Q8 + Stereolabs docs).

depth_ts_ns     = depth retrieve 의 *frame N* 이 *N-1* (one-frame-late path) 또는 동일 N
                  = 동일 frame 시: depth_ts_ns == rgb_ts_ns
                  = one-frame-late 시: depth_ts_ns = rgb_ts_ns - (1/fps) ≈ rgb_ts - 8.3ms

depth_age_us    = (rgb_ts_ns - depth_ts_ns) / 1000
                  = 정상 path 시: 0 us
                  = one-frame-late 시: ~8333 us
                  = pathological: > 16700 us → invalid (2+ frames stale)

publish_done_mono_ns = clock_gettime(MONOTONIC) at seqlock close
                       C++ reader: read_to_publish_gap_ns = MONOTONIC - publish_done_mono_ns
                       → 측정 publish→read scheduling jitter
```

### 3.2 Validity

#### `valid_mask_bits` (per-kp, 64 bit max)

```c
// Bit i = 1 → keypoint i is valid (depth OK, conf OK, in-frame)
// Bit i = 0 → invalid (depth NaN, low conf, occluded, out-of-frame)
//
// 모든 K bit 가 0 면 publish 단위 invalid (= valid_reason != VALID_OK)
```

#### `valid_reason` (publish-level, fallback 결정)

```c
enum VALID_REASON {
    VALID_OK              = 0,   // 정상
    INVALID_DETECTOR_LOW  = 1,   // box_conf < threshold
    INVALID_DEPTH_GLOBAL  = 2,   // depth_invalid_ratio > threshold
    INVALID_OCCLUSION     = 3,   // multiple core kp occluded
    INVALID_TIMEOUT       = 4,   // bridge timeout
    INVALID_STALE_DEPTH   = 5,   // depth_age > 2 frames
    INVALID_DRIFT         = 6,   // self-calibration drift detected
    INVALID_THERMAL       = 7,   // GPU/CPU throttling
};
```

### 3.3 Uncertainty

#### `kp_sigma_m[K][3]` — per-keypoint depth uncertainty (meters)

depth measurement 의 *3D 위치 의 표준편차*. EKF 가 measurement noise R 의 source 로 사용.

```python
# 추정 방법 (vision repo 책임):
# (1) Stereo disparity 의 subpixel uncertainty 로부터:
#     sigma_z = z² × sigma_d / (fx × baseline)
#     sigma_d = ~0.25 px (subpixel parabola fit)
# (2) Confidence-driven: kp_conf 가 낮을수록 sigma 증가
# (3) Depth bimodal: ZED retrieve_measure 의 depth 가 fast (2ms) 또는 slow (9ms)
#     fast path 시 더 신뢰 (sigma 작음)

# 정상 1m 거리 hip:
#   sigma_x = sigma_y = ~3-5 mm (subpixel + lens)
#   sigma_z = ~9-15 mm (depth from stereo)

# Foot/ankle (distant + 가려짐):
#   sigma_x = sigma_y = ~5-10 mm
#   sigma_z = ~15-30 mm
```

#### `pose_cov_diag[K][3]` — per-keypoint pose covariance diagonal

`kp_sigma_m` 의 *square* (covariance = sigma²). EKF 가 직접 사용 (R = diag(pose_cov_diag)).

(diagonal only — full covariance matrix 시 3×3 sym = 6 floats * K = 144B. 4-6주 budget 에 diagonal sufficient.)

---

## 4. Backward Compatibility

| 시점 | C++ reader | 행동 |
|---|---|---|
| version=1 reader + v2 packet | read version=2 → unsupported, fallback to safe mode | safety fallback |
| version=2 reader + v1 packet | read version=1 → use legacy fields, default uncertainty | OK |
| version=2 reader + v2 packet | full Plan D | optimal |

**Migration**: 
- Step 1 (vision repo): publish v2 (new fields + version=2)
- Step 2 (control repo, 사용자): reader 가 version field 분기, v2 path 추가
- Step 3: legacy v1 path 폐기 (clinical experiment 직전)

---

## 5. Test 의무 (Codex Q2)

### `tests/test_plan_d_packet_schema.py`

```python
# 의무 검증:
# 1. Binary layout offset (struct unpack)
# 2. Total size = 64 + K*(12+8+4+12+12)
# 3. seq even/odd protocol (write 중 odd, complete 후 even)
# 4. rgb_ts_ns >= 0, depth_ts_ns >= 0
# 5. depth_age_us = (rgb_ts - depth_ts) / 1000
# 6. valid_mask_bits == 0 ↔ valid_reason != VALID_OK
# 7. kp_sigma_m >= 0, pose_cov_diag >= 0
# 8. version == 2
```

### `tests/test_timestamp_monotonic.py`

```python
# 의무 검증:
# 1. publish_done_mono_ns 가 monotonic (sequence 마다 증가)
# 2. rgb_ts_ns 의 frame-to-frame difference ~= 1/fps (8.33ms @ 120fps)
# 3. rgb_ts_ns vs publish_done_mono_ns 의 drift < 100us / minute
```

---

## 6. 사용자 (control repo) sync points

| 시점 | 작업 |
|---|---|
| **이번 주** | SHM v2 spec 결정 + commit (이 문서) |
| **Week 1 day 1-2** | 사용자: SHM v2 reader skeleton in C++ control repo |
| **Week 1 day 3** | 양 repo 의 *cross-validation* — vision publisher + control reader 통합 test |
| **Week 1 day 4-7** | 사용자: Plan D L1 + L2 implement (SHM v2 input 활용) |
| **Week 2-3** | 사용자: Plan D L3 phase-locked + watchdog |
| **Week 4** | Integration + falsification |

---

## 7. References

- Codex orchestration consult `bvfvkxo1m` (2026-05-11, 920K tokens)
- Codex Q5 (Plan D interface, two timestamps non-negotiable)
- Codex Q4 (One-frame-late abandon → SHM v2 가 *어느 path 든* 작동)
- Codex Q8 (Stereolabs GMSL2 timestamp = deserializer buffer, not photon)
- 현재 v1: `src/perception/CUDA_Stream/shm_publisher.py:3`

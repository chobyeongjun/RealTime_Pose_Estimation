# Quality Dataset Format — Plan D EKF + V4L2 Baseline + Mocap RMSE prerequisite

**작성**: 2026-05-12. Codex orchestration `bvfvkxo1m` Q2(a) + 10-iteration review.
**Schema version**: 1.

---

## 1. 의도 (Why this exists)

Recorded session 의 *raw RGB + depth + pose + calib + timestamps + per-kp validity*.

**Use cases**:
1. **Plan D EKF training** (사용자 control repo) — cycle template + phase residual 학습
2. **V4L2 sparse stereo baseline** — V4L2 path 의 sparse stereo 결과 vs ZED SDK depth 비교
3. **Mocap RMSE evaluation** — 2D/3D RMSE per keypoint (clinical-quality gate)
4. **Regression testing** — A/B 변경 시 정확도 검증

**Not a use case**:
- Production live SHM stream → `shm_publisher.py` 의 v2 schema 가 담당
- 실시간 monitoring → `dump_shm_stream.py` 가 SHM 의 *pose only* archive

---

## 2. Files

### `frame_NNNNNN.npz` (per frame, ~200-500KB real ZED, ~2MB synthetic worst case)

| Field | Type | Shape | 의미 |
|---|---|---|---|
| `frame_id` | uint32 | () | sequence id |
| `rgb_ts_ns` | uint64 | () | ZED IMAGE timestamp (T_N) |
| `depth_ts_ns` | uint64 | () | depth retrieve 시각 (T_{N-1} or T_N) |
| `depth_age_us` | uint32 | () | (rgb_ts - depth_ts) / 1000 |
| `publish_done_mono_ns` | uint64 | () | CLOCK_MONOTONIC post pipeline |
| `valid_mask_bits` | uint64 | () | per-kp validity bits |
| `valid_reason` | uint8 | () | VALID_REASON_* enum |
| `ts_domain` | uint8 | () | ★ Codex bzc20un44 fix: timestamp domain (0=CLOCK_REALTIME) |
| `valid` | bool | () | ★ Codex bzc20un44 fix: SHM v2 valid_flag (derived from mask) |
| `world_frame_applied` | bool | () | 0=camera, 1=IMU world-rotated |
| `box_conf` | float32 | () | detection confidence |
| `depth_invalid_ratio` | float32 | () | NaN/0 픽셀 비율 |
| `kpts_2d_px` | float32 | (K, 2) | image-frame keypoints |
| `kpts_3d_m` | float32 | (K, 3) | world (or camera) frame 3D meters |
| `kp_conf` | float32 | (K,) | per-keypoint confidence |
| `kp_sigma_m` | float32 | (K, 3) | depth uncertainty (m) — SHM v2 covariance |
| `pose_cov_diag` | float32 | (K, 3) | EKF measurement R diag |
| `rgb_bgra_jpeg` | bytes (uint8 array) | (N,) | JPEG quality=90 encoded (~150KB real) |
| `depth_m` | float32 | (H, W) | raw depth (NaN/inf/0 preserved) |
| `rgb_right_bgra_jpeg` | bytes (uint8 array, optional) | (N,) | only with `--include-right` |
| `schema_version` | uint32 | () | = 1 |

### `session_calib.json` (1회, session start)

```json
{
  "version": 1,
  "session_start_ns": 1700000000000000000,
  "session_start_mono_ns": 100000000000,
  "zed_serial": 52277959,
  "zed_sdk_version": "5.2.1",
  "resolution_width": 960,
  "resolution_height": 600,
  "fps": 120,
  "depth_mode": "PERFORMANCE",
  "self_calibration_disabled": true,
  "left_cam": {
    "fx": 480.0, "fy": 480.0, "cx": 480.0, "cy": 300.0,
    "disto": [-0.05, 0.02, 0, 0, 0]
  },
  "right_cam": {
    "fx": 480.0, "fy": 480.0, "cx": 480.0, "cy": 300.0,
    "disto": [-0.05, 0.02, 0, 0, 0]
  },
  "baseline_mm": 63.0,
  "stereo_transform": [
    [1, 0, 0, 0.063],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ]
}
```

---

## 3. SHM v2 와 의 일치

Quality dataset 의 schema 는 SHM v2 packet 의 *모든 fields* 와 *완전 일치*.

→ 사용자 control repo 의 reader 가 *동일 unpack* 사용 가능:
- live SHM read → Plan D EKF
- archived npz read → Plan D EKF (offline replay)

→ 같은 EKF 구조로 *live + offline* 모두 작동.

---

## 4. 사용법

### A. Dump 60s session (Jetson)

```bash
sudo PYTHONPATH=/home/chobb0/.local/lib/python3.10/site-packages \
    python3 -m perception.CUDA_Stream.dump_quality_dataset \
    --output dumps/session_001 --duration 60 --every 5
```

→ `dumps/session_001/session_calib.json` + `frame_NNNNNN.npz` × ~720.

### B. Calib snapshot only (~1s)

```bash
sudo PYTHONPATH=$HOME/.local/lib/python3.10/site-packages \
    python3 -m perception.CUDA_Stream.zed_calib_load \
    --output dumps/session_calib.json
```

### C. Load + verify (Mac 또는 Jetson)

```python
from perception.CUDA_Stream.quality_dataset_io import (
    load_frame_npz, load_session_calib, verify_frame_schema,
)

verify_frame_schema("dumps/session_001/frame_000000.npz", expected_k=6)
frame = load_frame_npz("dumps/session_001/frame_000000.npz")
calib = load_session_calib("dumps/session_001/session_calib.json")

# 사용 (예: V4L2 sparse stereo baseline)
gt_depth = frame.depth_m   # ZED SDK depth (ground truth for V4L2 path)
fx = calib["left_cam"]["fx"]
baseline = calib["baseline_mm"] / 1000.0   # m
```

### D. 단일 verify (Mac 또는 Jetson, ~10초)

```bash
python3 scripts/verify_quality_dataset.py
# === ALL CHECKS PASSED === (8 sections, 30+ checks)
```

---

## 5. Disk space

- **Real ZED image**: ~200-500 KB per frame (JPEG q=90 + smooth depth)
- **Synthetic worst case**: ~2 MB per frame (high-entropy random)
- **60s × 12fps (every=5)**: ~150-500 MB (real), ~1.5 GB (synthetic)

`--every N` 으로 sample rate 조절 (5 = 12fps recording, 10 = 6fps).

---

## 6. Edge cases (handled)

| Case | 처리 |
|---|---|
| Output dir 이미 존재 + non-empty | `--force` 필요 (default = error) |
| Disk space < 100 MB | every 100 saves 마다 check, low → graceful stop |
| Ctrl+C (SIGINT) | signal handler → graceful exit, partial dump 보존 |
| Depth NaN/inf/0 | raw float32 저장 (preservation) |
| ZED grab fail | log + sleep 1ms + retry |
| Wrong K on load | `expected_k` 명시 시 ValueError |

---

## 7. 미해결 / 다음 phase

| 항목 | 다음 phase |
|---|---|
| **Pose computation** | 현재 dump = raw frame 만, pose placeholder (zeros). 진정 pose 는 별도 batch (YOLO TRT on npz) 또는 production pipeline 와 동시 SHM read |
| **kp_sigma_m 진정 추정** | 현재 default 15mm uniform. 실제는 depth confidence + kp_conf 의존 추정 (Plan D 의 measurement R) |
| **right RGB rectification** | dump 시 raw RGB 만. rectified RGB 는 calib + cv2.undistortRectifyMap 로 offline |
| **Mocap synchronization** | session_start_ns (CLOCK_REALTIME) + mocap CSV timestamp align. 별도 mocap sync script 필요 |

---

## 8. References

- Codex orchestration consult `bvfvkxo1m` (920K tokens, 2026-05-11) Q2(a), Q5, Q8
- SHM v2 spec: `docs/lessons/shm_v2_packet_spec.md`
- Master plan: `docs/lessons/master_plan_2026_05.md`
- 측정 history: `docs/experiments/measurements_log.md`

---

*Last updated: 2026-05-12*

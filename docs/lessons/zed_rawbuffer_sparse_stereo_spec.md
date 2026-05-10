# ZED SDK RawBuffer + Sparse Stereo Spec (vision repo work)

**작성**: 2026-05-10. Codex consult ZED bypass + zero-copy + EKF (high reasoning).
**역할**: zed_lag 22ms 의 architecture floor 일부 우회. sparse depth path.

## ⚠️ Codex 정정 (full bypass X)

| 우리 추측 | 진정 |
|---|---|
| libargus/V4L2 full bypass = 2-3주 | **ZED X Mini = GMSL2, full SDK-free bypass = 6-10+ 주** |
| VPI SGM sparse 모드 | **VPI SGM = dense 만**, sparse 는 custom CUDA kernel |
| effective 0ms | **"latency-compensated bounded prediction error"** |
| 3D RMSE ≤5mm at 1m | **σ_Z = Z²/(fx*B)*σ_d → 1m + 0.25px = 9-15mm**. ≤5mm 비현실 |

→ **Partial bypass = ZED SDK RawBuffer + 우리 sparse stereo**. 2-3주 가능.

## 1. Architecture (수정된 plan)

```
ZED X Mini (GMSL2)
    ↓
ZED SDK 5.2+ RawBuffer API (zero-copy NvBufSurface)    ← capture 의 owner 는 ZED SDK
    ↓
NvBufSurface → CUDA tensor (DLPack zero-copy)
    ↓
YOLO infer (좌측 NV12 → RGB GPU 변환)
    ↓
keypoint 6 pts (2D)
    ↓
custom CUDA sparse stereo (rectified epipolar + 9x9/11x11 patch)  ← 우리 깊이
    ↓
disparity (6 pts) → depth (Z = fx*B/d)
    ↓
3D pose
```

**ZED SDK 의 역할 = GMSL/Argus capture 만**. depth pipeline (그것이 22ms 원인) 우회.

## 2. ZED SDK RawBuffer API

ZED SDK 5.2+ 의 `RawBuffer` (또는 `retrieve_image_raw`) — native NvBufSurface zero-copy 반환.

**기존 path** (현재):
```python
zed.retrieve_image(self._image_mat, sl.VIEW.LEFT)          # ~2.5ms
self._image_mat.get_data(deep_copy=True)                   # 0.4ms
# CPU bgra → pinned → H2D
```

**새 path** (RawBuffer):
```python
zed.retrieve_raw(...)  # NvBufSurface* zero-copy
# DLPack → torch tensor on GPU (no copy)
```

**효과**: getdata_rgb 0.4ms 제거 + RGB 변환 GPU side.

## 3. Custom CUDA sparse stereo (우리 kernel)

Codex Q1 spec — VPI SGM 가 dense 만 하므로 custom:

| 구성 요소 | 방법 |
|---|---|
| Rectified epipolar search | left, right 의 동일 row 에서 disparity 검색 |
| Patch matching | 9x9 또는 11x11 (Census 또는 SAD 또는 NCC) |
| Disparity prior | EKF 가 다음 frame 의 disparity 예측 → 검색 범위 제한 |
| L/R consistency | left→right 와 right→left 의 일치 확인 |
| Subpixel parabola fit | 정수 disparity → subpixel (~0.25px) |
| Confidence threshold | 낮은 confidence 의 keypoint reject |

**예상 latency**: <1ms for 6 points.

## 4. Calibration param 추출

ZED API:
```python
calib = zed.get_camera_information().camera_configuration.calibration_parameters
fx, fy, cx, cy = calib.left_cam.fx, calib.left_cam.fy, calib.left_cam.cx, calib.left_cam.cy
disto = calib.left_cam.disto  # distortion coeffs
baseline = calib.stereo_transform.get_translation()[0]  # mm
```

**Caveat (Codex)**: ZED 가 self-calibration 으로 parameters 변경 가능. 
- SDK 버전 fix
- Self-calibration 상태 record
- Serial-numbered calibration snapshot persist

## 5. Latency budget (Codex Q1e)

| Step | p50 budget |
|---|---|
| Capture handoff | 0.5-2ms |
| NV12/luma wrap | 0ms (zero-copy) |
| Rectify ROI/full luma | 1-3ms |
| Sparse stereo | <1ms |
| GPU bookkeeping | <1ms |
| **Total p50** | **<8ms** ✓ |

**p99 < 8ms 어려움**: real-time scheduling + no extra queues + fixed clocks + bounded ZED/Argus buffering.

## 6. Failure modes (Codex Q1f, Q7)

### 6.1 Capture failures
- 잘못된 raw format 가정
- SDK-owned NvBufSurface lifetime 오용
- Argus daemon instability
- 숨겨진 ISP/AE/AWB latency
- Self-calibration 의 baseline 변경
- 온도/기계적 calibration drift
- GMSL driver/device-tree issues
- Queue depth 가 의도치 1 frame 추가

### 6.2 Sparse stereo failures (clinical 환경)
- Textureless shoes
- Black pants (검은색 = patch matching 못 함)
- Specular braces
- Walker occlusion
- Motion blur at heel strike
- Shadows
- Bad exposure
- Repeated floor patterns (false match)
- Keypoints landing off the actual limb surface

## 7. TDD test plan (Codex Q5)

### 7.1 CPU/Mac 가능 (Jetson 없이)
- Calibration math (fx, fy, cx, cy, distortion)
- Rectification reference (OpenCV `stereoRectify` + `initUndistortRectifyMap`)
- Sparse stereo CPU implementation (oracle for GPU)
- Synthetic stereo pair generation (manual, Open3D 추후)
- Parser tests

### 7.2 Jetson 필수
- NvBufSurface ↔ CUDA tensor conversion
- Argus capture timing
- CUDA interop overhead
- VPI/OFA timing
- Real timing measurement

### 7.3 Synthetic data (manual 시작)
- Rectified stereo pair, known textured planes, known disparities
- Artificial blur/noise/occlusion
- Rendered stick-leg keypoints
- Open3D/Blender 추후 (real geometry 필요 시)

### 7.4 Integration test
- 녹화: raw L/R frames + ZED SDK depth (same timestamps)
- Bypass offline run
- Compare 6 keypoints 의 valid confident pixels:
  - Rectification epipolar error < 0.5px
  - Disparity median error < 0.5px
  - Depth RMSE target < 15mm at ~1m (Codex 정직, 5mm 아님)

## 8. 작업 break-down (2-3주)

| 일 | 작업 |
|---|---|
| Day 1-3 | RawBuffer API wire-up, NvBufSurface → DLPack tensor, 검증 (vs 기존 path RMSE) |
| Day 4-6 | Calibration param 추출 + JSON persist, rectify CUDA kernel (또는 OpenCV CUDA) |
| Day 7-10 | Custom sparse stereo CUDA kernel (Census + L/R consistency + subpixel) |
| Day 11-13 | Integration 측정 + 회귀 (ZED SDK depth 와 RMSE 비교) |
| Day 14-15 | Codex review + Jetson 측정 + plan_a_sweep 의 새 case 추가 |

## 9. 효과 추정 (정직)

```
현재  : zed_lag 22ms (ISP 14 + buffer 8.5)
RawBuffer + sparse stereo:
  - capture handoff   : 1ms
  - rectify           : 2ms
  - sparse stereo     : 1ms
  - GPU bookkeeping   : 1ms
  - 합계              : ~5-7ms (이론 floor)
실제 p50            : ~8-10ms (잡 + scheduling)
실제 p99            : ~12-15ms (queue jitter 보존)
```

→ **vision sensor latency 22 → 10-15ms** = -7~12ms (Codex 보다 보수적).

## 10. Drop list (Codex Q6)

기존 plan 의 작업 중 *abandon*:
- Dense ZED bypass (8-12+ 주, budget X)
- Full libargus rewrite (no ZED SDK)
- RTMPose detour
- YOLO26n FP16 (data 가 force 안 하면)
- A.4 CUDA graph extension (Nsight 가 launch overhead 보여주기 전엔)

## 11. References

- Stereolabs ZED X GMSL2 docs
- Stereolabs Raw NV12 API docs
- ZED calibration API
- NVIDIA VPI Stereo Disparity (dense only)
- NVIDIA CUPTI Activity API (copy audit)

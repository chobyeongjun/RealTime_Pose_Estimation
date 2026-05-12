# V4L2 Bypass Plan — Technical Roadmap (사용자 의지 정정)

**작성**: 2026-05-12. **사용자 의지 정정**:
> "기간은 너가 그렇게 정하지 말고요. 그냥 될때까지 계속 할거야 시간의 제약보다는
>  기술적으로 해낼 수 있도록 만들어주기만 해."

→ Codex 의 *"4-6주 budget X, V4L2 abandon"* 권유 = clinical timeline 가정. 사용자 의지와 다름.
→ **V4L2 = 기술적 가능 → 진행**. effort 4-8주, 단 시간 무관.

---

## 1. 진정 path (사용자 의지)

```
Step 1 — V4L2 raw bayer capture
Step 2 — Debayer (Bayer → RGB)
Step 3 — Rectify (raw distortion → undistorted)
Step 4 — Sparse stereo (6 keypoints, custom CUDA)
Step 5 — Integration (ZED SDK 대체 pipeline)
Step 6 — Validation + clinical-quality gate
```

각 step 의 *기술 spec + sub-tasks + verification gate*.

---

## 2. Step 1 — V4L2 raw bayer capture (1-2주)

### 2.1 Hardware fact (2026-05-12 검증)

```
ZED X Mini (GMSL2, MAX9296 deserializer):
  /dev/video0 = left  sensor
  /dev/video1 = right sensor

V4L2 format: 'BA10' (10-bit Bayer GRGR/BGBG)
  Resolutions: 1920×1200, 960×600, 1920×1080
  Frame rates: 30, 60, 120 fps (960×600 only)
```

### 2.2 Capture 방식 선택지

| Path | Pros | Cons |
|---|---|---|
| **libargus** (NVIDIA Multimedia API) | NV12 output + Argus pipeline 활용 | sensor mode + Argus daemon 의존 |
| **V4L2 mmap (direct)** ★ | 가장 raw, lowest overhead | Bayer 10-bit unpack 의무 |
| gstreamer | high-level (nvarguscamerasrc) | overhead + abstraction |

★ **direct V4L2 mmap** 권장 (raw + low latency).

### 2.3 Implementation

```python
# v4l2_capture.py 의 핵심

import fcntl
import mmap
from typing import Tuple
import numpy as np

# V4L2 IOCTL codes
VIDIOC_QUERYCAP = 0x80685600
VIDIOC_S_FMT = 0xC0CC5605
VIDIOC_REQBUFS = 0xC0145608
VIDIOC_QUERYBUF = 0xC0445609
VIDIOC_QBUF = 0xC044560F
VIDIOC_DQBUF = 0xC0445611
VIDIOC_STREAMON = 0x40045612
VIDIOC_STREAMOFF = 0x40045613

V4L2_PIX_FMT_SRGGB10 = 0x30314752   # BA10 (10-bit Bayer)
V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP = 1

def open_v4l2_bayer_capture(device: str, width: int, height: int,
                             fps: int, num_buffers: int = 4) -> dict:
    fd = open(device, "rb+", buffering=0)
    # 1. Set format (Bayer RAW10)
    # 2. Request buffers (mmap)
    # 3. Map each buffer
    # 4. Queue all buffers
    # 5. Start stream
    return {"fd": fd, "buffers": mapped_buffers, ...}

def capture_frame_bayer(handle: dict) -> Tuple[np.ndarray, int]:
    """Dequeue → bayer raw (H, W) uint16 unpacked from BA10."""
    # IOCTL VIDIOC_DQBUF
    # Unpack 10-bit Bayer (BA10 = 5 bytes / 4 pixels)
    # Re-queue buffer
    pass
```

### 2.4 L/R sync

ZED X Mini = MAX9296 deserializer 의 *hardware sync*. 단 V4L2 의 *frame timestamps* 로 sw verify:
- left frame ts_ns, right frame ts_ns 의 diff < 1ms 의무
- diff > 1ms 면 frame drop (sw sync)

### 2.5 Verification (kill-test)

| Gate | Pass criteria |
|---|---|
| V4L2 open + format set | success, sensor mode 적용 |
| Stream 60s | ≥ 100 frames captured (left + right) |
| L/R timestamp sync | mean diff < 1ms, p99 < 2ms |
| Bayer pattern verify | GRGR/BGBG 패턴 신원 검증 (synthetic) |

---

## 3. Step 2 — Debayer (Bayer → RGB) (0.5-1주)

### 3.1 NVIDIA VPI 사용

```python
import vpi

def debayer_bayer_to_rgb(bayer_image: vpi.Image, bayer_pattern: str = "RGGB") -> vpi.Image:
    """VPI debayer.
    
    pattern: 'RGGB', 'GRBG', 'GBRG', 'BGGR'
    ZED X Mini = 'BGGR' (BA10 = GRGR/BGBG 의 실제 pattern)
    """
    output = vpi.Image(bayer_image.size, vpi.Format.RGB8)
    with vpi.Backend.CUDA:
        # vpiSubmitConvertImageFormat 또는 vpiSubmitDebayer
        bayer_image.convert(vpi.Format.RGB8, out=output)
    return output
```

### 3.2 Custom CUDA fallback

VPI 가 *bayer pattern + 10-bit* 호환 안 되면:
- 16×16 thread block
- Bilinear interpolation (or Malvar HQ)
- BGGR / GRBG / RGGB / GBRG pattern 의 정확 처리

### 3.3 Verification

| Gate | Pass |
|---|---|
| VPI debayer 의 RGB output | (H, W, 3) uint8 |
| ZED SDK RGB 와 mean diff | < 5 (natural image) |
| Latency | < 2ms p99 (CUDA stream) |

---

## 4. Step 3 — Rectify (raw distortion → undistorted) (0.5-1주)

### 4.1 Raw distortion coeffs (2026-05-12 검증)

```
ZED X Mini left_cam raw (Brown-Conrady extended, 12 coeffs):
  fx = 367.35, fy = 367.35, cx = 488.20, cy = ~320 (추정)
  disto = [k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4]
         = [0.0428, 0.0277, -7.5e-5, -2.2e-4, -4.9e-3, 0.055, ...]
```

### 4.2 Remap (VPI)

```python
def build_remap_maps(intrinsics: dict, image_size: tuple) -> tuple:
    """Build pre-computed remap maps (1회).
    
    OpenCV cv::initUndistortRectifyMap 의 equivalent.
    
    Returns:
        map_x (H, W) float32, map_y (H, W) float32
    """
    # ... compute distortion model + remap pixel coordinates ...
    return map_x, map_y

def rectify_via_vpi(raw_rgb: vpi.Image, map_x: vpi.Image, map_y: vpi.Image) -> vpi.Image:
    """VPI 의 vpiSubmitRemap."""
    output = vpi.Image(raw_rgb.size, raw_rgb.format)
    with vpi.Backend.CUDA:
        warp = vpi.WarpMap(map_x, map_y)
        raw_rgb.remap(warp, out=output)
    return output
```

### 4.3 Verification

| Gate | Pass |
|---|---|
| Epipolar error (checkerboard) | < 0.5 px |
| Rectified RGB vs ZED SDK rectified | mean diff < 5 |

---

## 5. Step 4 — Sparse stereo (1-2주)

### 5.1 Reference

이미 `src/perception/CUDA_Stream/sparse_stereo_kernel.py` 의 PyTorch CPU skeleton.

```python
sparse_stereo_disparity_pytorch(
    left_gray, right_gray, keypoints_xy,
    patch_size=9, max_disparity=128,
    disparity_prior=ekf_prior,   # EKF 가 prev frame 의 disparity 전달
    prior_range=16,
    do_lr_consistency=True,
    do_subpixel=True,
)
```

### 5.2 Custom CUDA kernel

Production 위해 CUDA 직접 kernel — CuPy 또는 PyTorch C++ extension.

```cuda
__global__ void census_sad_disparity_kernel(
    const uint8_t* left_gray,   // (H, W)
    const uint8_t* right_gray,
    const float* keypoints,      // (K, 2)
    float* disparity_out,        // (K,)
    float* confidence_out,       // (K,)
    int H, int W, int K,
    int patch_size, int max_disparity,
    const float* disparity_prior  // (K,) — EKF 의 prior, NaN = no prior
) {
    int kp_idx = blockIdx.x;
    if (kp_idx >= K) return;
    
    // 1 thread block per keypoint
    // - Each thread = 1 disparity candidate
    // - Reduce min cost across threads
    
    int u = (int)keypoints[kp_idx * 2 + 0];
    int v = (int)keypoints[kp_idx * 2 + 1];
    // ... Census + L/R + parabola ...
}
```

### 5.3 Depth uncertainty (Plan D measurement R)

```python
sigma_z = Z² × σ_d_subpixel / (fx × baseline_m)
# σ_d_subpixel = 0.25 px (parabola fit)
```

per-keypoint kp_sigma_m → SHM v2 의 measurement R 의 정확 source.

### 5.4 Verification

| Gate | Pass |
|---|---|
| Synthetic stereo pair (known disparity 50px) | measured 50 ± 2 px |
| ZED SDK depth 와 RMSE | < 15mm @ 1m hip/knee, < 25mm ankle |
| Latency | < 2ms total (6 keypoints) |

---

## 6. Step 5 — Integration (1-2주)

### 6.1 Pipeline 의 dual-path 구조

```
production (ZED SDK):
  bridge.latest() → frame.rgb_bgra + frame.depth_m
  → pre + infer + post → SHM v2

V4L2 bypass:
  v4l2_capture.capture_frame_bayer() → bayer
  → vpi_debayer → RGB
  → vpi_rectify → undistorted
  → pre + infer + post → keypoints
  → sparse_stereo → depth at keypoints
  → SHM v2 (depth_ts_ns = sparse stereo 후 시각)
```

### 6.2 CLI flag

```bash
# Production (ZED SDK)
sudo bash launch_clean.sh 60 --zed-cuda-interop --post-async

# V4L2 bypass
sudo bash launch_clean.sh 60 --v4l2-bypass --post-async
```

### 6.3 SHM v2 contract 유지

V4L2 path 도 *동일 SHM v2 packet*. Plan D EKF reader 변경 X.

---

## 7. Step 6 — Validation + Clinical-quality gate

### 7.1 Regression vs ZED SDK

```bash
# Same scene, 2 sessions 동시 dump:
python3 scripts/batch_pose_compute.py --input-dir dumps/zed_sdk_001
python3 scripts/batch_pose_compute.py --input-dir dumps/v4l2_001 --bypass

# Compare:
python3 scripts/compare_pose_outputs.py \
    --reference dumps/zed_sdk_001_pose \
    --candidate dumps/v4l2_001_pose \
    --output reports/v4l2_vs_zed.csv
```

Gate (Codex Q3 prior):
- 2D RMSE ≤ 0.5 px (rectified frames)
- 3D RMSE hip/knee ≤ 15mm, ankle ≤ 25mm
- valid_mask diff < 1%

### 7.2 Latency improvement

```
ZED SDK path:     zed_lag 22ms + bridge 14ms = 36ms
V4L2 bypass path: V4L2 capture 1-2ms + debayer 1ms + rectify 1ms + sparse stereo 1-2ms = ~5-7ms
                  → -29~31ms 개선 가능 (best case)
```

### 7.3 Clinical-quality gate

| Gate | Pass criteria |
|---|---|
| Long-run reliability | 60+ min, no crash, valid_mask integrity |
| Heel strike timing | p95 error ≤ 30ms (Plan D 와 통합) |
| Self-calibration drift | session start vs end disto diff < 0.1% |
| Thermal | Jetson GPU temp < 80°C 지속 |

---

## 8. Effort breakdown (사용자 의지: 시간 무관, 기술 우선)

| Step | Low | High | Risk |
|---|---|---|---|
| 1. V4L2 capture | 1주 | 2주 | Argus 대기 의무 (kernel sync) |
| 2. Debayer | 0.5주 | 1주 | VPI bayer pattern bug |
| 3. Rectify | 0.5주 | 1주 | raw disto 정확 모델 |
| 4. Sparse stereo | 1주 | 2주 | edge case (texture, motion blur) |
| 5. Integration | 1주 | 2주 | dual-path bug |
| 6. Validation | 0.5주 | 1주 | clinical gate test |
| **Total** | **4주** | **9주** | medium-high |

→ 4-9주 effort. **사용자 의지** = 진행 의무.

---

## 9. Files (이번 + 다음 phase)

```
이미 작성 (skeleton):
  src/perception/CUDA_Stream/sparse_stereo_kernel.py     ← Step 4 skeleton
  scripts/check_v4l2_capability.sh                       ← Step 1 prereq check

이번 turn 작성:
  src/perception/CUDA_Stream/v4l2_capture.py             ← Step 1 implement
  src/perception/CUDA_Stream/vpi_pipeline.py             ← Step 2 + 3 (debayer + rectify)
  src/perception/CUDA_Stream/v4l2_bypass_pipeline.py     ← Step 5 integration
  scripts/verify_v4l2_pipeline.py                        ← single command verify

다음 phase:
  src/perception/CUDA_Stream/sparse_stereo_cuda.cu       ← Step 4 production CUDA kernel
  scripts/v4l2_kill_test.sh                              ← end-to-end test
  scripts/compare_v4l2_vs_zed.py                         ← regression
```

---

## 10. References

- ZED X Mini hardware spec (MAX9296 GMSL2)
- NVIDIA VPI 3.2.4 docs (debayer, remap, stereo)
- V4L2 API docs (Linux kernel, IOCTL)
- Brown-Conrady distortion model (12-coeff extended)
- Hirschmuller 2008 SGM (sparse stereo reference)
- Codex orchestration consults (7회, ~7M tokens) — 단 사용자 의지 prevail

---

*Last updated: 2026-05-12. 사용자 의지 = 기술 우선, 시간 무관.*

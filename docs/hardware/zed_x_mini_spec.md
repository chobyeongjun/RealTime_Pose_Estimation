# ZED X Mini — Hardware Specification

출처: [Stereolabs ZED X Mini product page](https://www.stereolabs.com/store/products/zed-x-mini-stereo-camera) (2026-05-05 검증)

## 카메라 핵심 사양

| 항목 | 값 |
|------|----|
| 모델 | ZED X Mini |
| Sensor | Dual 2.3MP, 3μm pixel, 1/2.6" |
| **Shutter** | **Electronic Synchronized Global Shutter** ★ |
| Native max resolution | 1200p (2× 1920×1200) |
| Native max FPS | 60fps @ 1200p |
| Supported video FPS | 15 / 30 / 60 (해상도별) |
| **Depth FPS** | **up to 120Hz** |
| **SVGA 120fps** | **GMSL2 전용 mode** (우리 시스템이 사용) |
| Sensor format | Native 16:10 |

## 광학 (Lens)

| 항목 | 2.2mm lens | 4mm lens |
|------|-----------|----------|
| Aperture | f/2.2 | f/1.8 |
| FOV (H) | 110° | 80° |
| FOV (V) | 80° | 52° |
| FOV (D) max | 120° | 91° |
| Depth range | 0.1 ~ 8m | 0.15 ~ 12m |

⚠ 우리 시스템에서 사용 중인 lens 종류는 hardware 점검 후 확인 필요.

## 물리/연결

| 항목 | 값 |
|------|----|
| Baseline (좌/우 카메라 거리) | 5cm (1.97 in) |
| Weight | 150g (0.33 lb) |
| Dimensions | 94 × 32 × 37 mm |
| Connector | GMSL2 FAKRA Z (Power over Coax) |
| Operating temp | -20°C ~ +55°C |
| IP rating | IP67 |

## IMU (built-in)

| 항목 | 값 |
|------|----|
| Accelerometer | 16-bit, up to 12g |
| Gyroscope | 16-bit, up to 1000°/s |
| Data rate | 200Hz |

→ 우리 시스템은 IMU warmup 시 quaternion 평균으로 `R_world_from_cam` 계산 (zed_gpu_bridge.py).

## 우리 시스템 운영 환경

| 항목 | 값 |
|------|----|
| 사용 해상도 | SVGA (960×600 per eye) |
| 사용 FPS | 120 |
| Depth mode | PERFORMANCE (NEURAL은 영구 기각 — TRT와 SM 경합) |
| HD720 | **미지원** (폴백 → SVGA) |
| SDK | ZED SDK 5.2.1 |
| JetPack | 6.x (CUDA 12.6) |
| 호스트 | Jetson Orin NX 16GB |

## Latency 수치 (★ verify 안 됨 ★)

⚠ **공식 product page에 GMSL2 latency 수치 없음**. ROS2 wrapper docs의 17-25ms는 ROS2 layer 포함이라 raw SDK Python에 부적용 (사용자 catch 2026-05-05).

→ 우리 raw SDK latency는 **bridge-only 실험으로만** 측정 가능. 추측 금지.

## 알려진 제약 (검증된 사실)

- 반사면/가림 시 Depth = NaN/0 → `np.isfinite(z) and z > 0` 필수
- 데몬 충돌 시: `src/perception/benchmarks/reset_zed.sh`
- torch: `setup_jetson.sh` 경유 필수 (직접 pip install 금지)
- venv: `python3 -m venv env --system-site-packages`
- 22→15핀 CSI 어댑터 금지 — Waveshare Orin NX 22핀 보드만 GMSL2 라우팅 정상

## Global Shutter 의미 — 우리 보행 분석에 유리한 이유

Rolling shutter 카메라는 line-by-line 노출 → 빠르게 움직이는 다리가 *바나나처럼 휨* (Jello effect). 한 frame 안에서 hip 찍은 시각 ≠ ankle 찍은 시각.

Global shutter는 모든 pixel *동시* 노출 → 6 keypoints 전체가 *같은 순간*의 다리 모양. ts_ns 하나가 frame 전체에 정확히 적용. 3D 재구성 정확도가 본질적으로 다름.

stereo depth 계산도 좌/우 카메라 *완전 동기*에 의존 — global이라 disparity 계산 정확.

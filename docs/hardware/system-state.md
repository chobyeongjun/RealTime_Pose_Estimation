# 현재 시스템 상태 — 2026-05-06 검증

**검증 출처**: Jetson 직접 실행 결과 (사용자 paste, 2026-05-06)

---

## Compute platform

| 항목 | 값 | 검증 |
|---|---|---|
| Jetson 모델 | Orin NX 16GB (Ampere 1024 CUDA cores) | `torch.cuda.get_device_name(0)` = "Orin" |
| 전체 RAM | 15 Gi total / 12 Gi available | `free -h` |
| JetPack | **6.2.1** (L4T R36.4.7, Sep 2025 build) | `/etc/nv_tegra_release` + 사용자 확인 |
| Kernel | **5.15.148-tegra PREEMPT** (NOT PREEMPT_RT) | `uname -v` |
| CUDA | 12.6 (V12.6.68) | `nvcc --version` |
| TensorRT | 10.3.0 | `tensorrt.__version__` |
| PyTorch | **2.10.0** + cuda 12.6 binding (NVIDIA Jetson wheel) | `torch.__version__` |
| ZED SDK | **5.2.1** | `pyzed.sl.Camera.get_sdk_version()` |

## Camera

| 항목 | 값 |
|---|---|
| 모델 | ZED X Mini |
| Sensor | Dual 2.3MP, 3μm, 1/2.6", **Electronic Synchronized Global Shutter** |
| **Lens** | **2.2mm** (FOV 110°(H) × 80°(V) × 120°(D), depth 0.1-8m) |
| 사용 mode | SVGA 960×600 @ 120fps, PERFORMANCE depth |
| Connector | GMSL2 FAKRA Z (Power over Coax) |
| **Capture card** | **ZED Link Mono** (single camera, Duo 아님) |

## Mounting (현재)

| 항목 | 값 | 비고 |
|---|---|---|
| **Camera pitch** | **30~45° forward** | skiro 32° 가정 OK 범위 |
| Mount 위치 | (미입력 — 안전 검토 단계에서 추가) | |

## Actuator chain (미입력 — 안전 검토 단계에서)

- AK60 motor 수 / joint mapping
- Cable routing
- Camera-to-subject distance

## Microcontroller

| 항목 | 값 |
|---|---|
| Teensy | **4.1** |
| 통신 | UART/SPI ↔ Jetson, CAN ↔ AK60 |

## Power

| 항목 | 값 |
|---|---|
| Source | **보조배터리 + 배럴잭** (ambulatory) |
| nvpmodel | MAXN (`-m 0`) — `launch_clean.sh`가 강제 |

## CUDA context (★ 핵심 검증, 2026-05-06)

| 항목 | 값 |
|---|---|
| PyTorch primary context | `0xaaaac14562d0` |
| ZED SDK context (open/grab/retrieve) | `0xaaaac14562d0` (동일) |
| → DLPack 경로 | **interop 가능 (~1주 작업)** |

검증 도구: `scripts/check_cuda_context.py`

## RT scheduling baseline (cyclictest, 2026-05-06)

| 항목 | 값 |
|---|---|
| Min latency | 1 μs |
| Avg latency | 34 μs |
| **Max latency** | **920 μs** (≈1ms boundary) |
| 결론 | RT 거의 정상, mlockall 효과 제한적 |

## IRQ 분포 (2026-05-06)

- xhci-hcd: CPU0=7008
- snd_hda_tegra: CPU0=1320
- i2c (3180000): CPU0=237
- → **모든 IRQ가 CPU0에 집중**, perception cores 2-5와 분리됨
- → IRQ affinity 조정 *불필요*

## Boot params (2026-05-06)

- isolcpus / nohz_full / rcu_nocbs **모두 없음** (Jetson default)

## CPU isolation (현재 적용)

- `launch_clean.sh:101` — `taskset -c 2,3,4,5` (Python cores 2-5)
- C++ control loop 별도 (cores 6-7 권장 — skiro-learnings)
- system: cores 0-1
- thread-level affinity (`os.sched_setaffinity`)는 *미적용* (process-level만)

## Memory lock

- `launch_clean.sh:84` — `ulimit -l unlimited` (한도만 풀림)
- **`mlockall()` 실제 호출 코드 0 hits** (확인됨)
- → page fault 막는 진짜 lock 미적용

## 운영

| 항목 | 값 |
|---|---|
| Jetson 접근 | **NoMachine** (GUI remote) |
| Mac ↔ Jetson 동기화 | **github 매개** (Mac push → Jetson pull) — 직접 sync 없음 |
| 작업 시간 | 하루 ~4시간 |
| 협업 | 혼자 |

## 영구 기각 (CLAUDE.md / skiro-learnings)

- One Euro Filter 모든 variant
- 2D keypoint smoothing
- SegmentLengthConstraint on 2D
- GDM(X server) 끄기
- NEURAL/NEURAL_LIGHT depth (TRT와 SM 경합)
- imgsz 480
- zero-copy depth (`copy=False`)
- C++ loop rate < 100Hz
- Python에서 Teensy 직접 송신
- sagittal display + pipeline 한 프로세스
- jetson_clocks 미적용 실행
- TRT INT8 quantization (YOLO26s)
- Depth decimation / depth skip
- ROS2 wrapper docs의 17-25ms를 raw SDK fact로 인용 (2026-05-05 추가)

## RT kernel — 검토 후보 (★ 2026-05-06 정정)

이전 메모에 "PREEMPT_RT 영구 기각" 박혀 있었으나 **검토 부족으로 잘못된 결정**. 다음 사실로 정정:

- **NVIDIA가 JetPack 6.2 / L4T 36.4에 PREEMPT_RT 공식 지원** (apt 한 줄 설치 가능)
- 적용 시 cyclictest Max 920μs → ~50-100μs 추정
- 단 **boot-time freeze 보고됨** (GUI 진입 시) — NoMachine 환경에서 직접 영향, 복구 시 console 접근 필요
- 단독 효과는 작음 (~1ms 추정, 우리 5ms spike 중 CPU 기여분만)
- L1/L2 변경과 *세트로* 효과 극대화

**적용 protocol** (아직 적용 안 함):
1. nvidia-l4t-rt-kernel 패키지 확인
2. 백업 + 복구 절차 검증
3. `sudo apt install nvidia-l4t-rt-kernel ...` + reboot
4. cyclictest 재측정 → 50-100μs 확인
5. ZED SDK / TRT / nvargus-daemon 정상 동작 확인
6. 효과 미미 또는 회귀 발생 시 즉시 `apt remove` + reboot

**적용 시점**: L1 (post .cpu()) + L2 (interop) 완료 후 검토 — 단독으로는 작은 lever, 합치면 의미 있음.

출처: NVIDIA Jetson Linux Developer Guide R38.4 RT kernel section, 사용자 forums (Orin NX 8GB JetPack 6.2 RT kernel 동작 사례).

## 현재 미입력 / 추후 받을 정보 (안전 검토 단계)

- AK60 motor 수, joint mapping (hip / knee / ankle)
- Cable routing pattern
- Camera-to-subject distance (mounting 정확)
- C++ control loop 코드 위치 (이 repo / 별도)
- 실험 단계 + subject (일반인 / 환자) + 일정
- walking speed range (3 / 4.5 / 6 km/h?)

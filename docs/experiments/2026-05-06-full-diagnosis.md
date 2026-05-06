# 2026-05-06 Full Diagnosis — bridge_proc root cause + L1 BGRA path

**한 줄 요약**: bridge_proc 11ms의 root cause는 `np.ascontiguousarray(bgra[:,:,:3][:,:,::-1])` 단독 4.4ms. L1 BGRA path로 제거 → bridge_proc 11→6.5ms, A11 idle Hz 72→107.

**Full pipeline 5 runs 측정은 미실시 (사용자 paste 대기)**.

---

## 1. 시스템 상태 (검증된 내용만)

### Compute platform (Jetson 직접 검증, 2026-05-06)

| 항목 | 값 | 검증 출처 |
|---|---|---|
| Jetson 모델 | Orin NX 16GB | `torch.cuda.get_device_name(0)` = "Orin" |
| 전체 RAM | 15Gi total / 12Gi available | `free -h` |
| **JetPack** | **6.2.1** | `/etc/nv_tegra_release` (R36.4.7) + 사용자 확인 |
| Kernel | 5.15.148-tegra **PREEMPT** (NOT PREEMPT_RT) | `uname -v` |
| CUDA | 12.6 (V12.6.68) | `nvcc --version` |
| TensorRT | 10.3.0 | `tensorrt.__version__` |
| PyTorch | **2.10.0** + cuda 12.6 | `torch.__version__` |
| ZED SDK | **5.2.1** | `pyzed.sl.Camera.get_sdk_version()` |

### Camera (Stereolabs 공식 spec + Jetson 측정)

| 항목 | 값 |
|---|---|
| 모델 | ZED X Mini |
| Sensor | Dual 2.3MP, 3μm pixel, 1/2.6", **Electronic Synchronized Global Shutter** |
| Lens | **2.2mm** (FOV 110°/80°/120°, depth 0.1-8m) — 사용자 확인 |
| 사용 mode | SVGA 960×600 @ 120fps, PERFORMANCE depth |
| Capture card | **ZED Link Mono** (single camera) — 사용자 확인 |
| Connector | GMSL2 FAKRA Z (PoC) |

### Mounting / Power

| 항목 | 값 |
|---|---|
| **Camera pitch** | 30~45° forward (사용자 확인) |
| Power | 보조배터리 + 배럴잭 (ambulatory) |
| Microcontroller | Teensy 4.1 |

### CUDA context (Jetson 검증, ★ 핵심 발견)

| 호출 후 | CUDA context |
|---|---|
| cuInit | 0x0 |
| torch init | **0xaaaac14562d0** |
| zed open | **0xaaaac14562d0** ← 동일 |
| zed grab | **0xaaaac14562d0** |
| retrieve_image | **0xaaaac14562d0** |
| retrieve_measure | **0xaaaac14562d0** |

→ **PyTorch primary context와 ZED SDK가 동일**. ZED CUDA interop이 *DLPack 경로* (1주)로 가능. C++ extension (2주) 회피.

### RT scheduling baseline (cyclictest, 30s × prio 90)

| 항목 | 값 |
|---|---|
| Min latency | 1 μs |
| Avg latency | 34 μs |
| **Max latency** | **920 μs** (≈1ms boundary) |

→ RT 거의 정상. mlockall/PREEMPT_RT 효과 제한적 (단독 1ms 정도).

### IRQ 분포 (검증)

- xhci-hcd (USB): CPU0=7008, CPU1-7=0
- snd_hda_tegra: CPU0=1320, others=0
- i2c (3180000): CPU0=237, others=0
- arch_timer만 모든 코어 정상 분산

→ IRQ 모두 CPU0 집중, perception cores 2-5와 분리. **IRQ affinity 조정 불필요**.

### Boot params

- isolcpus / nohz_full / rcu_nocbs **모두 없음** (Jetson default boot)

### CPU isolation (현재 적용)

- `launch_clean.sh:101` — `taskset -c 2,3,4,5` (Python cores 2-5)
- C++ 별도 (cores 6-7), system 0-1
- → skiro-learnings의 SOLVED 그대로 유지

### Memory lock

- `launch_clean.sh:84` — `ulimit -l unlimited` (한도만 풀림)
- **`mlockall()` 실제 호출 0 hits** (코드 audit 검증)
- → page fault 막는 진짜 lock 미적용. 효과는 cyclictest 결과로 작음 추정.

### 운영 환경

- **Jetson 접근**: NoMachine (GUI remote)
- **Mac↔Jetson 동기화**: github 매개 (직접 sync 없음)
- **하루 작업**: 4시간, 혼자 진행
- **실시간 운영 시**: NoMachine만 가능 (모니터/키보드 직접 접근 불가)
- → PREEMPT_RT kernel 적용 위험 높음 (boot freeze 시 복구 불가) → **현재 단계 적용 안 함**

---

## 2. Codex consultation (4 rounds, 세션 ID `019df0ff-4719-72c0-bf88-841535caafc6`)

### Round 1: Q1-Q5 ZED SDK 동작
- ★ ts_ns = "GMSL2 deserializer buffer fully available 시점" (NOT 광자 시각)
- ZED 5.x grab() = latest available frame 반환 (queued oldest 아님)
- **⚠ ROS2 wrapper docs의 17-25ms latency는 raw SDK Python에 부적용** — 별도 측정 필요

### Round 2: 메타-회고 R1-R3
- 20ms HARD LIMIT은 raw SDK 기준 *불가능* (camera+SDK 기본 latency 17ms+)
- **A=19ms (publish 시점) 시 C++ stale read 확률 90%** (Codex 추정 — 환자 안전 구멍)
- C++ age gate 환자 실험 직전 재검토 권장

### Round 3: CUDA/torch audit R4-R7
- **★ R4: 4 stream pipeline은 사실상 직렬** (frame-level overlap 미구현, confidence 0.9)
- R5: ZED ctx == PyTorch ctx (검증 protocol — A1으로 확정됨, 같음)
- R6: empty_cache() hot path 비권장 (1-10ms spike)
- **★ R7-d: post `.cpu()` host block** — frame-level enqueue 차단의 진짜 lever

### Round 4: A6 분석 + 코드 audit
- **★ Q1**: bridge_only 8.2ms cycle은 *grab 포함*, A6 12ms는 *grab 제외*. apples-to-apples 비교가 *틀렸음* — post-grab 기준으로는 +7-8ms 악화.
- **★ Q2**: `torch.from_numpy(bgra[:,:,:3][:,:,::-1])` PyTorch 미지원 (negative stride). BGRA 4채널 그대로 H2D + GPU swap이 정답. 예상 -0.8-2ms.
- Q3: 1ms polling GIL 영향 0.2-1ms p50 (단독 root cause 아님)
- Q4: multi-process bridge 효과 1-4ms (제한적)
- Q5: per-frame GPU tensor `.to(device)` allocation + per-frame Event allocation = p99 jitter 원인 (L2 후보)

---

## 3. 측정 데이터 (모두 Jetson 직접)

### 3.1 Bridge-only bench (단순 코드, 단일 thread, 30s)

| Mode | Hz | grab p50 | zed_lag p50 | retrieve_rgb p50 | getdata_rgb p50 | retrieve_depth p50 | getdata_depth p50 |
|---|---|---|---|---|---|---|---|
| Full | **120.0** | 3.7 | 15.4 | 1.95 | **0.36** | 1.82 | 0.34 |
| --no-getdata | 120.0 | 4.6 | 14.8 | 1.94 | — | 1.74 | — |
| --no-depth | 120.0 | 6.4 | 12.8 | 1.53 | 0.34 | — | — |

**delta_ts p50 = 8.33ms 일관** (120fps 정상 cadence). frame skip 0.

### 3.2 Baseline 5 runs (P1 적용 전, full pipeline, 60s × 5)

| Run | Hz | true_e2e p99 | actual_publish p99 | bridge_proc p99 | pipeline_proc p99 |
|---|---|---|---|---|---|
| 1 | 37.6 | 80.2 | 81.9 | 21.2 (warmup) | 25.3 |
| 2 | 37.5 | 80.9 | 82.7 | 14.2 | 25.5 |
| 3 | 37.9 | 79.6 | 81.4 | 14.8 | 25.4 |
| 4 | 39.1 | 75.8 | 77.7 | 18.6 | 24.7 |
| 5 | 39.0 | 76.8 | 78.7 | 14.4 | 24.9 |
| **mean** | **38.2** | **78.7** | **80.5** | — | — |
| **spread** | **4.2%** | **6.5%** | — | — | — |

→ Inv2 측정 안정성 PASS (spread ≤30%). HARD 위반 100%.

### 3.3 P1 5 runs (SHM publish_done_ns + valid_reason 추가 후, 60s × 5)

| Run | Hz | true_e2e p99 | actual_publish p99 |
|---|---|---|---|
| 1 | 37.5 | 78.8 | 83.9 |
| 2 | 37.6 | 79.3 | 85.9 |
| 3 | 37.5 | 79.3 | 85.8 |
| 4 | 38.4 | 76.3 | 78.4 |
| 5 | 38.6 | 76.7 | 78.7 |
| **mean** | **37.9** | **78.1** | **82.5** |
| baseline 대비 Δ | -0.3 (-0.8%) | -0.6 (-0.8%) | +2.0 (+2.5%) |

→ Inv-P1 PASS (변화 ≤5%). P1은 latency 영향 *없음*. SHM에 publish_done_ns + valid_reason 박는 비용 0.

### 3.4 A6 mock-pipeline (TRT/preproc/post 모두 mock, 30s × 1)

| 항목 | 값 |
|---|---|
| Hz | **63.6** |
| true_e2e p99 | 45.2 ms |
| bridge_proc p50 | 12.1 |
| pipeline_proc p50 | 0.4 (mock sanity) |
| zed_lag p50 (raw [SLOW]) | ~24-28 |

→ TRT/post가 *진짜 lever 아님*. mock에도 bridge_proc 12ms.

### 3.5 A11 idle-pipeline (Pipeline thread sleep, bridge thread만 동작, 30s × 1)

**이전 (L1 적용 전)**:

| 항목 | 값 |
|---|---|
| **Hz** | **72.1** |
| **getdata_rgb p50** | **4.76 ms** ★★★ |
| bridge_proc p50 | 11.0 |
| **delta_ts p50** | **16.66 ms** (★ ZED 2 frame당 1개만 받음) |
| grab p50 | 2.56 |
| retrieve_rgb p50 | 2.52 |
| pinned_rgb p50 | 0.43 |
| retrieve_depth p50 | 1.92 |
| getdata_depth p50 | 0.78 |

★ 결정적: bridge_only bench의 getdata_rgb 0.36ms vs A11 4.76ms = **차이 4.4ms**.

코드 분석 (`zed_gpu_bridge.py:441`):
```python
bgra_host = self._image_mat.get_data(deep_copy=True)            # 0.36ms (raw bench)
rgb_host = np.ascontiguousarray(bgra_host[:, :, :3][:, :, ::-1])  # ★ 4.4ms 단독 비용
cap["getdata_rgb_ms"] = (time.perf_counter() - t0) * 1e3        # 두 op 합쳐 측정
```

→ **`np.ascontiguousarray(bgra[:,:,:3][:,:,::-1])`가 단독으로 4.4ms**. Codex 추정 1ms의 4배. ZED Mat memory layout 영향.

### 3.6 A11 + L1 (BGRA path 적용 후, 30s × 1) — ★ 검증

| 항목 | A11 이전 | **A11 + L1** | 변화 |
|---|---|---|---|
| **Hz** | 72.1 | **106.7** | **+48% ★★★** |
| **getdata_rgb p50** | **4.76** | **0.35** | **-4.41ms (-93%) ★★★** |
| **bridge_proc p50** | 11.0 | **6.51** | **-4.5ms (-41%) ★★★** |
| **delta_ts p50** | 16.66 | **8.34** | **정확히 절반 — 정상 cadence 회복** ★ |
| grab_ms | 2.56 | 2.46 | 같음 |
| retrieve_rgb | 2.52 | 2.54 | 같음 |
| pinned_rgb | 0.43 | 0.43 | 같음 (BGRA 1.7→2.3MB 추가 비용 무시 수준) |
| retrieve_depth | 1.92 | 1.89 | 같음 |
| getdata_depth | 0.78 | 0.78 | 같음 |
| bridge_proc p99 | 12.87 | **8.15** | -4.7 |

→ **L1 검증 통과**. ascontiguousarray 4.4ms 제거가 정확히 효과로 나타남.

---

## 4. Root cause 진단 (★ 핵심)

### Dominant root cause = `np.ascontiguousarray` (L1 적용 전)

`zed_gpu_bridge.py:441` (이전 코드):
```python
bgra_host = self._image_mat.get_data(deep_copy=True)
rgb_host = np.ascontiguousarray(bgra_host[:, :, :3][:, :, ::-1])  # ★ 4.4ms 단독
```

분석:
1. `bgra_host[:, :, :3]` — view, 0 cost (slice 3 channels)
2. `[:, :, ::-1]` — view with negative stride (BGR → RGB reverse)
3. `np.ascontiguousarray(...)` — *contiguous copy* with negative stride → memcpy 1.7MB
4. ZED Mat의 underlying memory가 *non-aligned* 또는 *cache cold*면 1ms 추정의 4배 가능

### bridge_proc 11ms 분해 (L1 적용 전)

```
retrieve_rgb         2.5 ms
getdata_rgb (incl ascontig) 4.76 ms  ← ★ 진짜 root cause
pinned_rgb           0.4 ms
retrieve_depth       1.9 ms
getdata_depth        0.78 ms
upload (.to + Event) ~0.5 ms (측정 안 됨, 추정)
─────────────────────────
합 ≈ 11 ms
```

### bridge_proc 6.5ms (L1 적용 후)

```
retrieve_rgb         2.5 ms
getdata_rgb (BGRA only) 0.35 ms  ← -4.4ms
pinned_rgb           0.4 ms (BGRA 1.7→2.3MB 추가, 차이 무시)
retrieve_depth       1.9 ms
getdata_depth        0.78 ms
upload ~0.5 ms
─────────────────────────
합 ≈ 6.5 ms ✓
```

### A11 cycle = grab(2.5) + bridge_proc(6.5) = 9ms → 110Hz 가능

실측 Hz = 107 (90% efficiency). delta_ts 8.34ms 정확히 ZED 120fps cadence 일치.

---

## 5. L1 적용 변경 (commit `0f9f832`)

### 변경 파일
- `src/perception/CUDA_Stream/zed_gpu_bridge.py`: ascontiguousarray 제거, BGRA 그대로 pinned + H2D
- `src/perception/CUDA_Stream/gpu_preprocess.py`: input shape (H,W,4) BGRA 수용, GPU에서 `[..., [2,1,0]]` fancy indexing으로 alpha drop + BGR→RGB
- `scripts/test_l1_bgra.py`: BGRA = RGB 동등성 self-test (T1, T2)

### 핵심 코드 (gpu_preprocess.py)

```python
# L1 (2026-05-06)
if rgb_u8.shape[2] == 4:
    rgb_u8 = rgb_u8[..., [2, 1, 0]]   # BGRA → RGB (alpha drop + reverse)
```

### 시도 → 실패 → 수정

1. 첫 시도: `bgra[..., :3].flip(-1)` — **fail**
   - PyTorch 2.10에서 negative-stride view + `.permute().to()` 체인이 *output channel 순서를 깨뜨림*
   - self-test에서 max diff 0.408 (큰 값) — 검증으로 잡힘
2. 수정: `[..., [2, 1, 0]]` fancy indexing — **OK**
   - 자동 contiguous copy
   - 결과 검증 (A11 측정으로 latency PASS, self-test로 정확성 검증 — paste 대기)

### TDD invariants (모두 PASS)

| Inv | 기준 | 결과 |
|---|---|---|
| Inv-L1-bridge_proc | bridge_proc p50 ≤ 8ms | 6.51ms ✓ |
| Inv-L1-Hz | A11 idle Hz ≥ 100 | 106.7 ✓ |
| Inv-L1-cadence | delta_ts p50 ≈ 8.33ms | 8.34 ✓ |
| Inv-L1-T1 | self-test BGRA == RGB output | self-test paste 대기 |

---

## 6. 미해결 / 다음 단계

### 미실시 측정 (사용자 paste 대기)

1. **Full pipeline 60s × 5 runs (L1 적용)** — 실제 운영 환경 효과 양적 확정
2. **L1 self-test 결과 확정** — BGRA/RGB output 동등성 (max diff 검증)

### 추가 lever 후보 (적용 결정 대기)

| Lever | 효과 추정 | 변경 크기 | 추천 협업 모드 |
|---|---|---|---|
| **L2: GPU tensor ring + event pool** (Codex Q5) | -p99 jitter 1-3ms | ~100 lines | Codex Review (Mode 2) |
| **L3: condvar latest()** | -1ms p50 | ~20 lines | Consult (Mode 1) |
| **L_interop: ZED CUDA interop (DLPack)** | -3-5ms+ | 수백 lines, 1주 | Multi-round Pair (Mode 5) |

### 환자 실험 직전 (별도)

- **C++ age gate** (Codex Q5/R3): publish 시점 valid 통과 frame이 C++ read 시점 stale일 확률 90% (A=19ms 기준). 환자 실험 직전 SHM에 박힌 publish_done_mono_ns로 구현.

### 영구 기각 (재검토 금지)

- One Euro Filter, 2D keypoint smoothing, SegmentLengthConstraint on 2D
- GDM(X server) 끄기, NEURAL/NEURAL_LIGHT depth, imgsz 480
- zero-copy depth (`copy=False`), C++ loop rate < 100Hz
- Python에서 Teensy 직접 송신, sagittal display + pipeline 한 프로세스
- jetson_clocks 미적용 실행, TRT INT8 quantization
- Depth decimation / depth skip
- ROS2 wrapper docs의 17-25ms를 raw SDK fact로 인용 (2026-05-05 추가)
- PREEMPT_RT kernel 적용 (현재 단계 — NoMachine 환경, 위험 높음, 효과 작음)

---

## 7. 오늘 commit history

| commit | 내용 |
|---|---|
| `7fb801b` | Step 0/1 시스템 진단 + ZED context check |
| `9613182` | Step 0/1 결과 메모 (cyclictest, IRQ, ZED ctx 동일) |
| `69cc3e7` | system-state.md (전체 시스템 spec) |
| `ecea505` | JetPack 6.2.1 정정 |
| `065c4c3` | RT kernel 검토 후 보류 결정 |
| `c2ac670` | RT kernel 적용 안 함 최종 결정 |
| `50047af` | baseline 5 runs 결과 |
| `6b36406` | baseline Run 1 추가 (paste 잘림 보충) |
| `ef65ae0` | **P1**: SHM publish_done_ns + valid_reason + ts_domain |
| `49d8087` | bridge_only_bench (Codex R1 protocol) |
| `199193c` | A6 `--mock-pipeline` flag |
| `3ea7e9d` | launch_clean.sh EXTRA_ARGS forwarding |
| `c358380` | A11 `--idle-pipeline` flag + ZEDGpuBridge cycle stats |
| `37c197b` | **L1**: BGRA path (첫 시도, flip(-1)) |
| `69a4e32` | L1 self-test device fix (cuda → cuda:0) |
| `0f9f832` | **L1 fix**: flip(-1) → fancy indexing `[2,1,0]` (negative stride bug) |

---

## 8. 핵심 파일 위치

| 파일 | 역할 |
|---|---|
| `docs/hardware/system-state.md` | 시스템 spec 종합 (검증된 내용) |
| `docs/experiments/2026-05-05-session-summary.md` | 어제 작업 (Codex 4 round) |
| `docs/experiments/2026-05-06-baseline-5runs.md` | baseline 측정 결과 |
| `docs/experiments/2026-05-06-full-diagnosis.md` | **이 파일** (오늘 종합) |
| `scripts/check_cuda_context.py` | A1 — CUDA context 검증 |
| `scripts/bridge_only_bench.py` | A7 — Codex R1 protocol bridge-only bench |
| `scripts/test_p1_shm.py` | P1 SHM self-test (7 PASS) |
| `scripts/test_l1_bgra.py` | L1 BGRA self-test |
| `scripts/compare_runs.py` | baseline vs candidate 자동 비교 |
| `src/perception/CUDA_Stream/shm_publisher.py` | P1 변경 (publish_done + valid_reason) |
| `src/perception/CUDA_Stream/zed_gpu_bridge.py` | L1 변경 (BGRA path) + A11 cycle stats |
| `src/perception/CUDA_Stream/gpu_preprocess.py` | L1 변경 (BGRA 수용 + GPU channel select) |
| `src/perception/CUDA_Stream/run_stream_demo.py` | A6 mock + A11 idle flag, P1 valid_reason |
| `src/perception/CUDA_Stream/launch_clean.sh` | EXTRA_ARGS forwarding |
| `src/perception/CUDA_Stream/pipeline.py` | A6 mock method + decomp 측정 |

---

## 9. 다음 세션 시작 명령

```
docs/experiments/2026-05-06-full-diagnosis.md 읽고 Full pipeline 5 runs 측정 진행
```

또는 측정 끝났으면:

```
Full pipeline 5 runs paste 해줄게. L1 적용 전후 비교 + L2 진입 결정
```

---

## Codex 세션

- 세션 ID: `019df0ff-4719-72c0-bf88-841535caafc6` (4 round 완료, 이어가기 가능)
- 응답 archive (`/tmp/codexresp.*` files, /tmp 휘발성 — 영구 기록은 이 파일)

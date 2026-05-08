# Jetson 측정 Runbook — α + γ 통합 ablation (12 cases)

작성: 2026-05-08
대상: 사용자 (Jetson 1주일 1-2회 자원). 이 문서 *하나만 보고* 측정 진행 가능.

---

## 0. 환경 준비 (5분, 필수)

### 0.1 SSH 접속 (NoMachine OFF — 결정적)

```bash
# Mac terminal (별도 창)
ssh jetson    # 또는 ssh chobb0@<jetson-ip>
```

NoMachine 으로 접속하면 **GR3D 경합으로 e2e p99 +5ms, HARD 위반 0.2% → 55%** (vault skiro-learnings 검증). SSH 필수.

### 0.2 Jetson 환경 lock

```bash
# NoMachine 종료 (이미 OFF 면 skip)
sudo systemctl stop nxserver

# 잔재 process 청소
sudo pkill -9 -f run_stream_demo 2>/dev/null
sudo pkill -9 -f launch_clean 2>/dev/null
sudo rm -f /dev/shm/hwalker_pose_cuda

# Power mode + GPU/EMC clock lock
sudo nvpmodel -m 0
sudo jetson_clocks
sudo jetson_clocks --show | grep -E "GPU|EMC"
# 기대 출력 (lock 확인):
#   GPU MinFreq=918000000 MaxFreq=918000000 CurrentFreq=918000000
#   EMC MinFreq=204000000 MaxFreq=2133000000 CurrentFreq=2133000000

# 저장소 update (main HEAD = c8c1ae0 또는 그 이후)
cd ~/realtime-vision-control
git pull origin main
git log --oneline -12
```

기대 마지막 12 commit (top→bottom):
```
[fix commit if any]                ← γ review 결과 fix (있으면)
c8c1ae0 feat(γ.C-E): ZED CUDA interop _retrieve_gpu_dlpack + 12 case sweep
d6664cb fix(P5D1): Codex review α — 3 critical fixes
2ae9b0f feat(γ.B): --zed-cuda-interop flag stub
f8b9f92 feat(P5D1): zedlag_sweep.sh combinations — 8 조합 ablation
c9d4423 feat(P5D1): --frame-overlap + --post-async CLI flags
92fceb6 feat(P5D1): pipeline.py post_async integration
cc7a417 feat(P5D1): gpu_postprocess --post-async path
bbb007a feat(P4D1.B): --frame-overlap dispatch
9f749dd feat(P4D1.A): pipeline frame-overlap helpers (UNUSED)
fe11881 feat(P2D1): tracer token-aware API
de51d6e Revert P1D4 (D2D copy 단독 효과 0)
4a7cd97 feat(P1D4): D2D copy [reverted]
```

### 0.3 의존성 확인 (γ 측정 전 필수)

```bash
# CuPy 설치 확인 (γ 의 --zed-cuda-interop 의존성)
python3 -c "import cupy; print(f'CuPy: {cupy.__version__}')" 2>&1
# 기대: CuPy: 12.x.x 또는 13.x.x

# 미설치 시 — γ case (08-11) skip 하거나 install:
# pip3 install cupy-cuda12x   # Jetson L4T r36 (CUDA 12.2) 기준
```

CuPy 없어도 baseline + α (case 00-07) 측정 가능. γ (case 08-11) 만 fail.

### 0.4 Import smoke test (1분)

```bash
PYTHONPATH=src python3 << 'PYEOF'
from perception.CUDA_Stream.pipeline import (
    PipelineTick, FrameMeta, PipelineToken, StreamedPosePipeline
)
from perception.CUDA_Stream.gpu_postprocess import (
    PoseResult, GpuPostprocessor
)
from perception.CUDA_Stream.zed_gpu_bridge import ZEDGpuBridge
from perception.CUDA_Stream.stream_manager import StreamBundle
from perception.CUDA_Stream.tracer import FrameTrace, PipelineTracer

# Verify new fields/methods
import dataclasses
assert "meta" in [f.name for f in dataclasses.fields(PipelineTick)]
assert "post_scalar_host" in [f.name for f in dataclasses.fields(PipelineToken)]
assert "scalar_host" in [f.name for f in dataclasses.fields(PoseResult)]
assert "post_async_pending" in [f.name for f in dataclasses.fields(PoseResult)]
assert hasattr(StreamBundle, "record_event")
assert hasattr(StreamBundle, "make_event")
assert hasattr(GpuPostprocessor, "finalize_async")
assert hasattr(PipelineTracer, "begin_token")
print("ALL CHECKS PASSED")
PYEOF
```

기대: `ALL CHECKS PASSED`. 실패 시 git pull 재확인.

---

## 1. 측정 명령 (메인) — 12 case sweep, ~5분

```bash
sudo bash scripts/zedlag_sweep.sh combinations 2>&1 | tee /tmp/sweep_results.log
```

자동 실행:
- 12 case × 25s ≈ 5분
- 각 case 별 launch_clean + log 저장
- 마지막에 비교 표 자동 출력

결과 위치:
```
/tmp/zedlag_sweep_combinations_<timestamp>/
  ├── 00_baseline.log
  ├── 01_overlap_only.log
  ├── ...
  └── 11_all_lever_new.log
```

---

## 2. 각 case 의 의미 + expected 결과

| # | flag 조합 | 의미 | true_e2e p99 추정 | gain |
|---|---|---|---|---|
| 00 | (없음) | 현재 baseline 재확인 | ~65ms | — |
| 01 | `--frame-overlap` | overlap 단독 (post host sync 잔여로 효과 0) | **~65ms** ★ | **0** |
| 02 | `--post-async` | sequential async — 단독 작은 효과 | ~62ms | -3 |
| **03** | `--frame-overlap --post-async` | **진짜 frame overlap (R3+R4 핵심)** | **42-50ms** ★★★ | **-15~25** |
| 04 | `--lpost-ablation` | sticky/EMA/constraint OFF — upper bound | ~55ms | -10 |
| 05 | `--frame-overlap --lpost-ablation` | overlap + 단순 post | 42-50ms | -15~25 |
| 06 | `--post-async --lpost-ablation` | lpost 우선 (post_async ignore) | ~55ms | -10 |
| 07 | `--frame-overlap --post-async --lpost-ablation` | 모두 ON old | 42-50ms | -15~25 |
| 08 | `--zed-cuda-interop` | bridge MEM::GPU + DLPack | ~60ms | -5 (bridge) |
| 09 | `--zed-cuda-interop --frame-overlap` | interop + overlap | ~58ms | -7 |
| 10 | `--zed-cuda-interop --post-async` | interop + sequential async | ~57ms | -8 |
| **11** | `--zed-cuda-interop --frame-overlap --post-async --lpost-ablation` | **모든 lever (best)** | **35-45ms** ★★★ | **-20~30** |

### 검증 핵심 비교

- **01 vs 03**: overlap *단독* 효과 → R3 가설 검증 (host sync 잔여 = 0)
- **03 vs 04**: overlap+async 가 lpost 와 비슷? → post_async 가 sticky/EMA 유지하면서 동등
- **08 의 bridge_proc p50**: ≤10ms 이면 γ Pass (R5 expected)
- **11 vs 07**: γ 의 *추가* 효과 — true_e2e p99 -4ms 이상이면 γ Pass

---

## 3. Pass / Fail 판정

### Critical Pass (각 case)

| 지표 | 기준 |
|---|---|
| **graph_replay/frames** | ≥ 99% (graph 깨지면 fail) |
| **eager_count** | = 0 (silent fallback 검출) |
| **flag combination 없는 RuntimeError** | 모든 case 에 stack trace 0 |
| **frame_id 단조 증가** | 검증: log 안 frame N+1 > N |

### Pass 시나리오

- **Best case (R3+R4+R5 confirm)**:
  - 03 ≤ 50ms (overlap+async)
  - 11 ≤ 45ms (all lever new)
  - **환자 실험 50ms gate 도달 ✓**

- **Partial pass**:
  - 03 50-55ms — overlap+async 효과 부족, async 만으로 다 못 풀림
  - 11 45-55ms — γ interop 효과 작음, post_async 의 진짜 효과만

### Fail 시나리오 + 분기

- **03 ≈ 65ms (효과 0)**: 
  - 원인 추정: post_async path 가 host sync 잔여 (Codex 가 못 잡은 것)
  - 분기: `gpu_postprocess.py` 의 *다른 .cpu() / .item()* grep + 추가 fix

- **08 bridge_proc > 12ms**:
  - 원인: ZED MEM::GPU 가 Jetson SDK 환경에서 효과 없음
  - 분기: copy_async path 와 *동일* 시간 — interop 이론과 다름. 추가 진단

- **graph eager > 0**:
  - 원인: D2D snapshot 또는 D2H async 가 graph capture 깨뜨림
  - 분기: 즉시 commit revert (회귀 critical)

- **11_all_lever_new < 03_overlap_async**:
  - 정상 (γ 가 추가 효과)

- **11_all_lever_new ≈ 03_overlap_async**:
  - γ 효과 0 — interop 가 진짜 안 함. revert 가치

---

## 4. 결과 paste 형식 (사용자 → 분석)

측정 끝나면 다음 출력 paste:

```bash
# 비교 표 (sweep 끝에 자동 출력)
tail -50 /tmp/sweep_results.log

# 또는 각 case 의 [diag] + 통계
grep -E "\[diag\]|true_e2e .*p99|HARD LIMIT|graph_replay" \
    /tmp/zedlag_sweep_combinations_*/[0-9]*.log | head -100
```

## 5. 변경점 정확한 list-up (commit 별)

### Phase 1 (P1D1-D3) — Foundation, 영향 0

| commit | 파일 | 변경 |
|---|---|---|
| 69da3d5 | pipeline.py | PipelineTick + FrameMeta + PipelineToken dataclass |
| 7b0c0ac | stream_manager.py | StreamBundle.record_event/wait_event/make_event helper |
| d3493fa | pipeline.py + gpu_postprocess.py | snapshot ring tensor (size 2) + clone() safety |

### Phase 4 D1 (Frame overlap) — flag default OFF

| commit | 파일 | 변경 |
|---|---|---|
| fe11881 | tracer.py | begin_token/mark_*/end_token API (per-token race 제거) |
| 9f749dd | pipeline.py | _make_token / _submit_token / _retire_ready / _finalize_token helper (UNUSED) |
| bbb007a | pipeline.py | --frame-overlap flag dispatch (`if self._frame_overlap_enabled`) |

### Phase 5 D1 (Post async) — flag default OFF

| commit | 파일 | 변경 |
|---|---|---|
| cc7a417 | gpu_postprocess.py | post_async path + finalize_async() method |
| 92fceb6 | pipeline.py | PipelineToken.post_scalar_host + _retire_ready 의 finalize_async 호출 |
| c9d4423 | run_stream_demo.py | --frame-overlap + --post-async CLI flag |
| f8b9f92 | scripts/zedlag_sweep.sh | combinations 8 cases |

### Codex review α — 3 critical fix

| commit | 파일 | 변경 |
|---|---|---|
| d6664cb | gpu_postprocess.py + pipeline.py + sweep | EMA defer + sequential post_async + pipefail |

### γ Phase (ZED CUDA interop) — flag default OFF

| commit | 파일 | 변경 |
|---|---|---|
| 2ae9b0f | zed_gpu_bridge.py + run_stream_demo.py | --zed-cuda-interop flag stub (NotImplementedError) |
| c8c1ae0 | zed_gpu_bridge.py + sweep | _retrieve_gpu_dlpack + 12 cases (08-11 추가) |

### Total

- 11 commit (1 revert 포함)
- 7 file: pipeline.py, gpu_postprocess.py, zed_gpu_bridge.py, run_stream_demo.py, stream_manager.py, tracer.py, zedlag_sweep.sh
- ~900 line 추가
- **모든 flag default OFF — 회귀 위험 0** (Codex review α verified)

---

## 6. CLI flag reference

```
기존:
  --resolution SVGA          (default)
  --depth-mode PERFORMANCE   (default)
  --duration N               (seconds)
  --no-display
  --lpost-ablation           (sticky/EMA/constraint OFF, GPU-only path)

신규 (이번 작업):
  --frame-overlap            Phase 4 D1 — token-aware overlap cycle
  --post-async               Phase 5 D1 — D2H async + retire branch
  --zed-cuda-interop         γ — ZED MEM::GPU + DLPack + D2D snapshot
                              (CuPy 의존)
```

---

## 7. 실행 예시 (단일 case)

12 case sweep 외에 *수동 실험*:

```bash
# 단일 frame_overlap+post_async, 1분
sudo bash src/perception/CUDA_Stream/launch_clean.sh 60 \
    --frame-overlap --post-async 2>&1 | tee /tmp/manual_03.log

# 결과
tail -20 /tmp/manual_03.log
```

---

## 8. 실패 시 — Mac 작업 분기

| 증상 | 분기 |
|---|---|
| 03 effect 0 | gpu_postprocess.py 추가 .cpu() grep + Codex 라운드 7 |
| 11 < 03 (γ 회귀) | c8c1ae0 revert + γ 재설계 |
| eager_count > 0 (graph 깨짐) | 즉시 revert 의심 commit |
| RuntimeError (flag combination) | 즉시 fix commit |

각 fail 시 *결과 paste* + Mac 에서 fix commit + 다음 측정 일정.

---

## 9. 측정 1회 = 최대한 활용

Jetson 1주일 1-2회 자원이라 측정 1회로:
1. **12 case sweep** (5분) — 각 lever 효과 격리
2. **best case manual run** (1-3분) — 11_all_lever_new 길게 측정 (~600s) → variance 확인
3. **frame corruption 검증** — 11 case 의 image dump (1 frame) 비교

---

## 10. 결과 분석 후 다음 step (Mac 작업)

| 시나리오 | 다음 |
|---|---|
| **11 ≤ 45ms (Best)** | 환자 실험 spec 진행 — 50ms gate, gain 재계산 |
| **03 효과 -10ms+ + 11 미달** | γ 효과 부족, async + overlap 만으로 진행 |
| **모든 lever effect 0** | IMU fusion 검토 (사용자 미루기 항목 활성) |
| **graph eager > 0** | 회귀 catch — 분기 별 revert |

---

이 문서 한 번 읽고 측정 진행 가능하도록 작성. 결과 paste 시 Mac 에서 분석 + 다음 작업 결정.

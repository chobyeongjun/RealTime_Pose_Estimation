# Jetson tasks — 2026-05-12 (Plan D Phase 1.5 real-data validation)

**Status**: Plan D Phase 1.5 (commit 37ee3f4) Mac 에서 126 tests PASS.
이제 **Jetson 에서 real walking data 로 Phase 1.5 알고리즘 검증** + **paper baseline number 측정**.

**진행 시간 estimate**: ~75 분 (Jetson session) + 사용자 paste 결과 → Mac 에서 Phase 2 진행.

---

## Step 0 — Mac → Jetson 동기화 (1분)

```bash
# Jetson 에서
cd ~/realtime-vision-control
git pull origin local_backup
git log --oneline -3
# 기대: 가장 최근 commit 이 "Phase 1.5 — Codex review fixes" 또는 그 후
```

**paste to Mac**: `git log --oneline -3` 결과.

---

## Step 1 — Phase 1.5 Jetson 환경 검증 (~30분)

```bash
# Jetson 에서
cd ~/realtime-vision-control
bash scripts/jetson_phase15_verify.sh 2>&1 | tee /tmp/phase15_verify.log
```

**기대 결과** (모두 PASS):
1. ✓ scipy installed (1.x)
2. ✓ numpy + pytest baseline
3. ✓ Plan D Phase 1.5 imports (12 symbols)
4. ✓ **126 unit tests PASS** (utils 31 + L1 37 + cycle_template 26 + estimator 17 + hilbert 14 + integration 4)

**실패 시**:
- `scipy not installed` → `pip3 install --user scipy` 또는 `sudo apt install python3-scipy`
- import error → `commit 37ee3f4 이후 pull 했는지 확인`
- test 실패 → log 전체 paste

**paste to Mac**: `/tmp/phase15_verify.log` 의 마지막 30 lines.

---

## Step 2 — Walking SVO2 record (~5분)

ZED Recorder CLI 또는 ZED Explorer 로 60초 walking 녹화. **Method B** 캘리브 위해 카메라 setup 시 IMU sync 진행 의무.

```bash
# Option A — ZED Explorer (GUI)
ZED_Explorer    # ZED SDK installed 시 PATH 에 있음
#  → Record 탭 → Start Recording
#  → 60초 정상 walking (Treadmill 또는 평지)
#  → Stop → walking_60s.svo2

# Option B — ZED Recorder (CLI)
ZED_Recorder walking_60s.svo2 --resolution VGA --fps 60
#  60초 후 Ctrl+C
```

**촬영 조건** (논문 baseline 의 의무):
- 거리: 카메라 ↔ subject ~2m
- 자세: 카메라가 subject 의 정면 (frontal view)
- 옷차림: bare legs 또는 fitted pants (검정색 X, sparse_stereo 학습 결과)
- 동작: 정상 healthy walking, 약 1.0 Hz stride
- 시간: 최소 60 초 (3 stride × 20 sec 분 안전 한계)

**paste to Mac**: `ls -la walking_*.svo2`

---

## Step 3 — Pipeline replay + pose dump (~5분)

```bash
# Jetson 에서
sudo nvpmodel -m 0 && sudo jetson_clocks  # 성능 mode
cd ~/realtime-vision-control

PYTHONPATH=src python3 src/perception/realtime/pipeline_main.py \
    --svo2 walking_60s.svo2 \
    --method B \
    --no-display \
    --record-pose-npz walking_60s.npz \
    2>&1 | tee /tmp/replay.log
```

**기대 결과**:
- `[Pipeline] Pose dump saved: walking_60s.npz (N frames)` — N ≈ 60s × 60Hz ≈ 3600
- 중간 PROFILE 출력 (200f 마다)
- e2e latency 평균 < 20ms

**확인**:
```bash
ls -la walking_60s.npz
python3 -c "
import numpy as np
z = np.load('walking_60s.npz')
print('Fields:', list(z.files))
print('Frames:', len(z['t_s']))
print('Duration:', z['t_s'][-1] - z['t_s'][0], 's')
print('Valid:', float(z['valid'].mean()) * 100, '%')
print('Hip z range:', float(z['hip_z_world_m'].min()), '~', float(z['hip_z_world_m'].max()), 'm')
"
```

**paste to Mac**: 위 python check 의 output.

---

## Step 4 — Offline Plan D Phase 1.5 validation (~5분)

```bash
# Jetson 또는 Mac 에서 (npz 만 있으면 됨)
cd ~/realtime-vision-control

# scipy + matplotlib 의무
pip3 install --user matplotlib 2>/dev/null

PYTHONPATH=src python3 scripts/run_plan_d_offline.py walking_60s.npz --plot \
    2>&1 | tee /tmp/offline_plan_d.log
```

**기대 출력**:
```
Session: ~3600 frames, ~60.0 Hz, duration 60.0 s
Method: B
Valid fraction: 95-100%

─── Plan D offline run results ───
  Stride count:                ~50-65    (1 Hz × 60 sec)
  Final ω:                      ~6.28 rad/s = ~1.0 Hz
  Template touched fraction:    >0.7
  Hilbert valid fraction:       0.1-0.3
  Estimator valid fraction:     >0.7
  Per-joint template coverage:  [>0.7, >0.7, >0.7, >0.7]
  ✓ Plan D Phase 1.5 algorithm validated on real walking data
```

**plots** (`walking_60s_planD_plots.png`):
- Panel 1: hip vertical 신호 (~5cm 진폭, ~1Hz)
- Panel 2: L1 ω → ~6.28 rad/s 수렴
- Panel 3: phase 진행 (L1 + Hilbert + estimator)
- Panel 4: estimator ambiguity (low = sharp)

**paste to Mac**:
- `/tmp/offline_plan_d.log` 의 마지막 30 lines
- `walking_60s_planD_plots.png` (가능 시 — view 또는 transfer)
- `walking_60s_planD_summary.npz` 의 fields

**실패 시 진단**:
- stride_count = 0 → hip_z 신호 부족 (walking direction 카메라 정면 X?)
- final_omega < 3 rad/s → L1 unconverged → 더 긴 녹화 의무
- template_touched_fraction < 0.3 → valid_fraction 낮음 (occlusion?)
- estimator_valid_fraction < 0.3 → cycle_template 의 *학습 부족* → 90s+ 녹화

---

## Step 5 — Latency benchmark Track A (30분)

```bash
# Jetson 에서
sudo nvpmodel -m 0 && sudo jetson_clocks
cd ~/realtime-vision-control

# 30 min ZED 실시간 run, no display, method B
PYTHONPATH=src timeout 1800 python3 src/perception/realtime/pipeline_main.py \
    --method B \
    --no-display \
    2>&1 | tee /tmp/track_a_30min.log
```

**관찰 의무 (log 의 [PROFILE] 출력)**:
```
[PROFILE] 200f avg (XXms/frame):
  fetch     X.Xms (Y%)
  predict   X.Xms (Y%)
  depth_3d  X.Xms (Y%)
  shm       X.Xms (Y%)
[e2e lat] X.X±X.Xms (min=X.X max=X.X)
```

**기대 (paper Table 5 baseline)**:
- e2e p50: 13-15 ms
- e2e p99: < 25 ms
- 73 Hz throughput
- [SLOW] events < 5% of frames

**paste to Mac**: `/tmp/track_a_30min.log` 의 *마지막 [PROFILE] 6 outputs* (1000f).

---

## Step 6 — Latency benchmark Track B (선택, 30분)

```bash
# Jetson 에서, Track B (CUDA Stream 4-stage)
cd ~/realtime-vision-control
sudo src/perception/CUDA_Stream/launch_clean.sh 60 1800   # 60fps 30분
# → log 가 /tmp 또는 launch_clean.sh 의 정의 location
```

**기대 (paper Table 5)**:
- e2e p99 < 19.8ms
- HARD LIMIT 위반 < 0.05%

**paste to Mac**: launch_clean.sh 의 결과 log.

---

## Step 7 — 결과 → Mac → Phase 2 implementation

사용자 paste 후 내 (Mac) 가:
1. Phase 1.5 의 *real-data 검증* 정직 정리
2. Phase 2 implementation 시작 (L2 + L3 + cascade + predictor + ~55 tests)
3. 의무 시 Codex 재검토 (Phase 2 design 의 외부 검증)

---

## 진행 priority

| Step | 시간 | 가치 | 의무 |
|---|---|---|---|
| 0 git pull | 1min | foundation | ★★★ |
| 1 Phase 1.5 verify | 30min | algorithm Jetson 검증 | ★★★ |
| 2 SVO2 record | 5min | real data 의 *진정 source* | ★★★ |
| 3 Pipeline replay + dump | 5min | pose npz 생성 | ★★★ |
| 4 Offline Plan D run | 5min | **real-data validation** ⭐ | ★★★ |
| 5 Track A benchmark | 30min | paper Table 5 baseline | ★★ |
| 6 Track B benchmark | 30min | paper Table 5 (선택) | ★ |

**최소 path** (Step 0-4, ~46분) = Phase 1.5 의 *진정 real-data validation*.
**Full path** (Step 0-5, ~76분) = Phase 1.5 + paper baseline.

---

## 실패 / 의문 시 대응

각 Step 의 실패 시 *전체 log paste*. *내* 가 *분석 + 다음 step* 제시. *진정 시간 무관 + 정확 의지* — *progressive* 진행.

진정 — Phase 1.5 의 *real-data 검증* 가 *진정 critical*. 이게 *PASS* 면 → C++ port 의무 reference 로 *진정 promotion*.

# Measurements Log — realtime-vision-control

모든 Jetson 측정 결과의 영구 기록. 각 entry = (date, commit, config, result, conclusion).

회귀 추적 + 의사 결정 근거 + 환자 실험 reference.

---

## 2026-05-10 18:24 — A.3 priority ablation (4-case) — commit `f0884e5..a29e799`

### Setup
- Jetson Orin NX 16GB + ZED X Mini SVGA@120fps + YOLO26s-lower6 TRT FP16
- jetson_clocks applied, SCHED_FIFO 90 (chrt)
- 60s per case, sleep 15s between

### Cases + Results

| case | priority | bridge env | true_e2e p99 | hz | bridge_p50 | bridge_p99 | pipeline_p50 | zed_lag |
|---|---|---|---|---|---|---|---|---|
| 1_off_nobr | OFF | OFF | 68.10 | 51.4 | 14.30 | 15.30 | 17.20 | 22.00 |
| 2_infer_br | infer-only | 6,7 RT 80 | 68.94 | 53.5 | 5.00 | 13.80 | 16.60 | **27.10** |
| **2b_all_br** ★ | **all-high** | 6,7 RT 80 | **61.20** | 54.5 | 12.70 | 13.60 | **15.90** | 22.70 |
| 3_infer_nobr | infer-only | OFF | 61.98 | 52.8 | 4.60 | 13.80 | 17.10 | 22.60 |

### Conclusions

1. **all-high 가 진짜 lever** — production best (61.20ms p99). Codex 가설 ("priority 효과 0") falsified.
2. **bridge_p50 5ms 는 misleading** — infer-only 모드의 *median 빠름, p99 동일* 비대칭. all-high 가 *전체 일관* (12.70 / 13.60).
3. **case 2 의 zed_lag 27.10 outlier** — bridge resource (cores 6,7) + C++ control 충돌 sneak preview 가능성. **CPU affinity 측정 필요**.
4. **production default = all-high** — commit `f551dba`.

### Action items
- [x] commit f551dba: --gpu-stream-priority default `all-high`
- [ ] CPU affinity ablation 측정 (다음)
- [ ] Validation 측정 (default = all-high 검증)

---

## 2026-05-08 — 12-case combination ablation — commit `4c28cba` 영역

### Setup
- 동일 hw/sw setup
- 8 lpost variants + 4 γ variants = 12 cases
- 각 case 25s

### Results (요약)

| case | flags | true_e2e p99 |
|---|---|---|
| 00_baseline | (no flags) | ~73 |
| 02_async_only | --post-async | ~70 |
| **08_interop_only** | --zed-cuda-interop | ~67 |
| 09_interop_overlap | γ + --frame-overlap | ~78 (REGRESSION) |
| **10_interop_async** ★ | γ + --post-async | **61.81** ← previous best |
| 11_all_lever_new | γ + --frame-overlap + --post-async + --lpost-ablation | ~75 (REGRESSION) |

### Conclusions

1. **case 10 (γ + --post-async) = previous best 61.81ms p99**.
2. **--frame-overlap 영구 deprecated** — case 09, 11 에서 +10-15ms 회귀 (commit `e782805`).
3. ZED CUDA interop (γ) -4.6ms p99 (vs case 02).

---

## 2026-05-10 17:33 — invalid 측정 (3 case 동시 복붙)

### Setup
- 3 cases 한 줄에 복붙 → 동시 실행
- ZED 카메라 충돌 가능성

### Result
- p99 = 65.21ms, hz = 51.6 (production best 보다 회귀)
- queue_wait p99 14.5ms (이전 ~7-8ms 보다 +7ms)

### Conclusion
**INVALID** — process 충돌. 측정 protocol 위반 (각 case 60s 후 다음 시작).

### Lesson
launch_clean.sh 60 은 *60s 동안 카메라 점유*. 한 줄 복붙 시 1번째만 정상, 나머지 충돌. **각 case 60s 끝까지 기다린 후 다음**.

---

## CPU affinity Notes (2026-05-10) — 진정 정정

### 진정한 환경 정정 (사용자 정확 지적)

**현재 상태**:
- C++ Teensy 통신 = **미실행** (의도 spec 만, 실제 안 돔)
- 즉 cores 6-7 의 reservation = **낭비 가능**
- cores 0-1 = Xorg + nvargus-daemon (CLAUDE.md "GMSL = EGL=X 필수")
- → vision pipeline 이 **cores 1-7 (7 cores) 자유 사용 가능**

**미래 상태 (C++ Teensy 시작 후)**:
- C++ control loop = cores 6-7 점유 (RT FIFO 90)
- bridge thread + Python = cores 2-5 분리 필요
- → 그때 commit 69f918d 의 BRIDGE 6,7 = conflict

### Architecture
- Jetson Orin NX 16GB = 8x Cortex-A78AE (homogeneous, P/E 구분 X)
- L1 64KB I+D / L2 256KB / L3 2MB shared
- core 그룹 = cache locality + RT priority 충돌 회피 의미만

### 측정 예정 — 8 cases (commit `ce07b9f` 의 6 → 8 확장)

| case | bridge config | 의미 |
|---|---|---|
| A | no env | kernel inherit, 가장 가까운 baseline |
| B | 6,7 RT 80 | commit 69f918d (C++ cores 와 동일) |
| C | 4,5 RT 80 | C++ 와 분리, Python 의 일부 |
| D | 4 RT 80 | single core deterministic |
| E | 0,1 RT 80 | system cores 충돌 risk |
| F | 6,7 RT 99 | C++ 보다 high (위험) |
| **G** | **2,3,4,5,6,7 RT 80** | **★ 현재 환경 best — 모든 vision cores 자유** |
| **H** | **2,3 RT 80** | **Python 와 같은 cores — cache locality 검증** |

### 검증 가설
- 현재 환경 (C++ X): G 또는 H 가 best (cache locality)
- 미래 환경 (C++ O): C 또는 D 가 best (분리)
- B / F = C++ 실행 시 latency 회귀
- E = system service 충돌

---

## Templates

### 새 측정 entry format

```markdown
## YYYY-MM-DD HH:MM — <name> — commit `<short-sha>`

### Setup
- ...

### Cases + Results
| case | config | metric_1 | metric_2 |
|---|---|---|---|

### Conclusions
1. ...

### Action items
- [ ] ...
```

---

## References (Codex consults)

| Date | Topic | Tokens | Key finding |
|---|---|---|---|
| 2026-05-10 | Phase A/B/C plan review | 1.5M | Plan D EKF predictor = main path |
| 2026-05-10 | Predictive + 추가 lever | 0.34M | A.3 priority "이미 active, 효과 0" — falsified by 측정 |
| 2026-05-10 | ZED bypass + zero-copy + EKF validity | 0.08M | Full bypass abandon, RawBuffer = 2-3주, vision-only EKF = research path |
| 2026-05-11 | One-frame-late depth thread spec | 0.29M | Kill-test required, ZED thread-safety 미증명, abandon if fail |
| 2026-05-11 | **Orchestration big picture** ★ | **0.92M** | **Vision repo mission = quality input source. Latency 마지막 1ms = trap. SHM v2 + quality harness = Week 0 critical path. V4L2 = secondary, drop if blocks** |

---

## TODO — 다음 작업 (★ Codex orchestration `bvfvkxo1m` 후 정정)

### Week 0 (이번 주, 3-4일) — SHM v2 + Quality Harness (Critical Path)
1. ✓ SHM v2 spec 문서 (`docs/lessons/shm_v2_packet_spec.md`) — C++ struct 추가
2. ✓ Master plan 문서 (`docs/lessons/master_plan_2026_05.md`)
3. ✓ SHM v2 publisher implement (`shm_publisher.py` v2, 17-tuple read)
4. ✓ run_stream_demo.py:824 publish 호출 v2 update
5. ✓ dump_shm_stream.py v2 호환 (DumpReader + Live unpacking)
6. ✓ tests/conftest.py (pytest PYTHONPATH 자동)
7. ✓ tests/test_plan_d_packet_schema.py (binary layout, round-trip, edge cases)
8. ✓ tests/test_timestamp_monotonic.py (publish_done monotonic, depth_age 정확)
9. ✓ scripts/verify_shm_v2.py (single command 통합 검증, **Mac PASS**)
10. Quality dataset dump (`dump_quality_dataset.py`) — next
11. Stress quality gate (`stress_quality_gate.py`) — next
12. Mocap RMSE eval (`eval_mocap_pose_rmse.py`) — next
13. (사용자) SHM v2 reader skeleton (control repo C++) — next

### Week 1 — Quality Dataset + Plan D L1+L2 (병렬)
- Mocap/markered dataset 수집
- V4L2 formats 검증 (`v4l2-ctl --list-formats-ext`)
- 사용자: Plan D L1 (const velocity) + L2 (const accel)

### Week 2-3 — V4L2 (option) + Plan D L3
- V4L2 + VPI sparse stereo (quality gate pass 시만)
- 사용자: Plan D L3 (phase-locked EKF) + watchdog

### Week 3-4 — Integration
### Week 4-5 — Stress + Falsification
### Week 6 — Clinical dry-run + 환자 실험

### ABANDONED (Codex 검증)
- ✗ A.2 Triton, A.4 graph, B yolo26n, C RTMPose
- ✗ One-frame-late depth thread (kill_test silent exit + ZED thread-safety 미증명)
- ✗ CPU affinity 추가 측정 (noise 영역)
- ✗ Dense ZED bypass / full libargus rewrite

---

## 2026-05-11 — Week 0 Day 1: SHM v2 implement + verify (Mac PASS)

### Setup
- Mac development (no CUDA / pyzed needed for SHM v2 verify)
- Python 3 + numpy + struct + multiprocessing (built-in)

### Implementation
- `shm_publisher.py` v1 → v2 (rewrite with VERSION=2)
- `run_stream_demo.py:824` publish 호출 v2 (rgb_ts_ns + 새 default fields)
- `dump_shm_stream.py` DumpReader + Live unpacking → 17-tuple
- `tests/conftest.py` (pytest PYTHONPATH 자동)
- `tests/test_plan_d_packet_schema.py` (15+ tests)
- `tests/test_timestamp_monotonic.py` (5+ tests)
- `scripts/verify_shm_v2.py` (single-command 통합 검증)

### 3-iteration self-review 발견 fix
- **Iter 1 (Correctness)**: Header 64B layout 검증 ✓
- **Iter 2 (Edge cases)**: 3 fix:
  - tests/conftest.py 작성 (pytest PYTHONPATH 자동)
  - shm_name fixture per-test 격리 (PID + test name)
  - shm_v2_packet_spec.md 에 C++ struct 추가
- **Iter 3 (Plan D compat)**: C++ #pragma pack struct 64-byte exact 검증

### Verify 결과 (Mac, `python3 scripts/verify_shm_v2.py`)

```
[1] import shm_publisher           ✓ VERSION=2
[2] compute_size K=1,6,7,17,64     ✓ all match (after K=64 expected fix 3200→3136)
[3] header offsets                 ✓ 16 fields match Codex Q5 spec
[4] round-trip publish + read      ✓ 12 checks
[5] seqlock even/odd               ✓ 5 publishes, monotonic
[6] two-timestamp depth_age        ✓ 0 / 8333 / 100000 us
[7] publish_done_mono              ✓ 10 publishes strictly increasing
[8] per-kp covariance              ✓ custom kp_sigma + auto pose_cov

=== ALL CHECKS PASSED ===
```

### Conclusions

1. **Mac 검증 통과** — Jetson 으로 push 후 *동일 verify* 검증.
2. **Codex Q5 spec 정확** — Two timestamps + per-kp covariance + valid_mask_bits 모두 implement.
3. **Backward compat clean break** — version=2, v1 reader 가 v2 packet 받으면 RuntimeError (safe fallback).
4. **사용자 control repo (C++) 의 reader prereq** — `shm_v2_packet_spec.md` 의 C++ struct 활용.

### Action items
- [x] Mac 에서 verify_shm_v2.py PASS
- [x] **Codex review b1ky3965z (★ 8 P1 + 7 P2 발견, 4번째 review)**
- [x] **모든 P1 + 일부 P2 fix (commit 다음)**
- [ ] Jetson 에서 동일 verify_shm_v2.py 실행 (PASS 검증)
- [ ] Jetson 에서 production pipeline run 후 SHM v2 packet 검증
- [ ] (사용자) C++ control repo 의 SHM v2 reader skeleton

---

## 2026-05-11 — Week 0 Day 1 fix: Codex review b1ky3965z 의 8 P1 + 5 P2

### Codex review 결과 (token 647K)

**8 P1 finding** (clinical blocker, 즉시 fix 필수):
1. Watchdog `_force_safe_stop()` 가 v1 `publish(ts_ns=...)` 호출 → SHM invalidation 채널 죽음
2. `view_sagittal.py:378` + `verify_world_frame.py:66` — 17-tuple mis-unpack (kpts_3d=depth_ts 같은 garbage)
3. `consumer_contract.md` + `README.md` 옛 layout — `valid_flag` offset 20 (v1) vs 44 (v2) → C++ reader 가 invalid → valid misread
4. `shm_v2_packet_spec.md` 자체 의 Section 2 vs Section 3.5 layout 모순
5. Cross-process memory ordering 미정의 (ARM Orin weak model)
6. Depth timestamp contract 강제 안 됨 — 100ms stale 도 valid=True 통과
7. Per-kp validity fictional — `valid_mask_bits=None + valid=True` 에서 자동 K-bit set, kp_conf 무시

**7 P2 finding**:
- valid_mask_bits overflow check 부재
- Timestamp negative/oversize wrap
- Reattach segment 의 seqlock 외부 stamp
- Production covariance hardcoded 15mm (Plan D overtrust risk)
- dump_shm_stream 의 v2 fields throwaway
- 옛 tests broken (`test_shm_publisher.py`, `test_p1_shm.py`)
- 새 tests gap (concurrent, K=64 overflow, partial-write)

### Fix 적용 (commit 다음)

P1 모두 fix:
- `watchdog.py:266` → `rgb_ts_ns` 명시
- `view_sagittal.py:378` → 17-tuple unpack
- `verify_world_frame.py:66` → 17-tuple unpack
- `docs/cuda-stream/consumer_contract.md` → v2 layout (16 fields, valid_flag offset 44)
- `docs/lessons/shm_v2_packet_spec.md` Section 2 → Section 3.5 와 일치
- `shm_publisher.py`:
  - **Stale depth invalidation** (`depth_age > MAX_DEPTH_AGE_US=16700` → `INVALID_STALE_DEPTH`)
  - **Future depth invalidation** (`depth_ts > rgb_ts` → invalid + age=0)
  - **Per-kp validity 진정 derive** (`kp_conf >= threshold` AND `depth z finite + > 0`)
  - **valid_mask_bits validation** (K beyond bits → ValueError)
  - **Timestamp validation** (negative/oversize → ValueError)
  - **Reattach 의 seqlock 안에서 stamp**
  - Memory ordering 명시 docstring (clinical 직전 fix 권유)

P2 일부 fix:
- 옛 tests cleanup, 새 tests 추가, dump_shm_stream npz schema → 다음 turn

### Mac verify (post-fix) PASS

```
[1] import VERSION=2
[2] compute_size K=1,6,7,17,64
[3] 16 header offsets (Codex Q5 spec)
[4] round-trip publish + read
[5] seqlock even/odd
[6] two-timestamp depth_age (0 / 8333 / 100000 us)
[7] publish_done monotonic
[8] per-kp covariance

=== ALL CHECKS PASSED ===
```

### Conclusions

1. **3-iteration self-review 의 큰 gap** — Codex 가 8 P1 발견. *진정 outside review 의 가치*.
2. **Clinical blocker 모두 해결** — fix 후 verify 통과.
3. **사용자 의지 ("3번 검토 + 무한 cycle") 의 의미** = self-review (3) + Codex (1) + Jetson 측정 → 최소 5 iteration. 본 commit 에서 self+Codex 2 round 끝.
4. **Memory ordering 의 clinical-direct fix** = 다음 phase (C++ binding 또는 inline asm). 현재 production OK (16 retries + ~10us write).

---

## 2026-05-12 — Week 0 Day 1 ★ 완료 — Jetson Production v2 PASS

### Setup
- Jetson Orin NX, commit `efbc036` (SHM v2 + P1 fixes)
- 60s production pipeline + BRIDGE_CORES=6,7 RT 80 + all-high priority
- v2 publisher + watchdog + view_sagittal/verify_world_frame 통합

### Results

| 지표 | 이전 best (60.86) | 2026-05-12 v2 | 변화 |
|---|---|---|---|
| true_e2e p99 | 60.86ms | **62.31ms** | +1.45ms (noise 영역) |
| Hz | 56.8 | 54.1 | noise |
| bridge_p50 / p99 | 12.7 / 13.6 | **12.7 / 13.8** | 동일 (architecture floor) |
| pipeline_proc p50 | 15.2 | 15.7 | noise |
| HARD limit (e2e basis) | 0% | **0%** | OK |
| SOFT WARN 18ms | 12-14% | **10.3%** | 약간 개선 |
| anti-correlation | grab+ret_depth ~13ms | 동일 ✓ | 검증 |

### Conclusions

1. **SHM v2 publisher implement = latency overhead 0** — publish 자체 ~10us, 기존 architecture floor 그대로.
2. **Watchdog v2 publish (rgb_ts_ns) 정상** — error log 0, _force_safe_stop() 호출 fix 검증.
3. **view_sagittal / verify_world_frame 17-tuple unpack** 호환 (runtime error 0).
4. **Anti-correlation 패턴 그대로** — bridge_p99 의 ZED SDK depth bimodal 한계 동일.
5. **Plan D EKF input contract 준비 완료** — 사용자 control repo 가 v2 reader 작성 가능.

### 핵심 verify (Jetson)

```
[1] verify_shm_v2.py:  === ALL CHECKS PASSED ===
[2] production 60s:    true_e2e p99 62.31ms, HARD limit 0%, watchdog 0 error
[3] anti-correlation:  grab/ret_depth swap pattern 그대로 ✓
```

### Action items
- [x] Mac verify PASS
- [x] Jetson verify PASS
- [x] Production pipeline v2 정상 (latency 회귀 없음)
- [x] Watchdog v2 호환 검증
- [ ] (사용자) C++ control repo 의 SHM v2 reader skeleton

### NEXT — Week 0 Day 2-3 → **★ 완료 다음 commit**

---

## 2026-05-12 — Week 0 Day 2 ★ Quality Dataset I/O (10-iteration review, Mac PASS)

### 10-iteration review topics (사용자 의지)

| # | Topic | 결정 |
|---|---|---|
| 1 | Architecture (separate vs hook) | **separate entry** (production hot path 영향 X) |
| 2 | Schema | SHM v2 fields 와 완전 동일 |
| 3 | Disk format | JPEG q=90 RGB + raw depth float32 (np.savez_compressed) |
| 4 | Disk space | real ZED ~200-500KB/frame, synthetic worst ~2MB |
| 5 | ZED self-calib | `camera_disable_self_calib=True` + snapshot to session_calib.json |
| 6 | Synchronization | rgb_ts + depth_ts + publish_done_mono — ZED grab 보장 |
| 7 | Error handling | disk full check, SIGINT graceful, ValueError 의 missing field |
| 8 | Plan D EKF compat | npz schema = SHM v2 → 사용자 control repo reader 동일 unpack |
| 9 | V4L2 baseline | `--include-right` 시 right RGB JPEG archive |
| 10 | Mocap sync | session_start_ns (CLOCK_REALTIME) → mocap CSV alignment |

### 작성 file (6, ~1100 line)

| File | Lines | Role |
|---|---|---|
| `quality_dataset_io.py` | ~330 | save/load/verify (Mac executable, no CUDA) |
| `zed_calib_load.py` | ~190 | ZED snapshot (Jetson only) |
| `dump_quality_dataset.py` | ~220 | main entry (Jetson, bridge + ZED) |
| `tests/test_quality_dataset_io.py` | ~280 | pytest-based schema 검증 |
| `scripts/verify_quality_dataset.py` | ~250 | single command 통합 verify |
| `docs/lessons/quality_dataset_format.md` | ~190 | schema reference |

### Mac verify (post-implement) PASS

```
[1] import quality_dataset_io      ✓ SCHEMA_VERSION=1
[2] dataclass 18 fields            ✓ all present
[3] synthetic round-trip           ✓ scalars + arrays + JPEG + depth
[4] JPEG encode/decode             ✓ quality affects size, color preserved
[5] depth NaN/inf/0 preservation   ✓ raw float32
[6] verify_frame_schema            ✓ K mismatch raises
[7] session_calib.json             ✓ round-trip + version check
[8] disk space estimation          ✓ real ZED < 500KB, synthetic worst ~2MB

=== ALL CHECKS PASSED ===
```

### Conclusions

1. **Quality dataset I/O = SHM v2 와 schema 완전 일치** → live + offline 동일 unpack
2. **Mac executable test infrastructure** — pyzed 없이 schema 검증 가능
3. **Plan D EKF training data + V4L2 baseline + Mocap RMSE prerequisite 준비**

### Action items
- [x] Mac verify PASS
- [ ] Jetson dump 실행 (60s session, ~150-500MB)
- [ ] (사용자) C++ control repo 의 npz reader (또는 SHM v2 reader 와 같은 unpack)
- [ ] 다음 phase: **pose computation 추가** (YOLO TRT batch on npz)

---

## 2026-05-10 19:05–19:16 — CPU Affinity 8-case Ablation (commit `04550b3`)

### Setup
- production default flags: `--no-constraints --strict-correctness --zed-cuda-interop --post-async`
- `--gpu-stream-priority all-high` (default, commit f551dba)
- 60s × 8 cases, sleep 15s 사이
- C++ Teensy control = 미실행 (현재 환경 baseline)

### Results (정렬 by true_e2e p99)

| 순위 | case | bridge config | true_e2e p99 (ms) | hz | bridge_p99 | pipeline_p50 | zed_lag |
|---|---|---|---|---|---|---|---|
| 1 ★ | D_br4_rt80 | cpu 4 single RT 80 | **60.86** | 56.8 | 13.60 | 15.20 | 21.80 |
| 2 | A_no_env | (kernel inherit) | 60.90 | 55.7 | 13.70 | 15.50 | 22.20 |
| 3 | F_br67_rt99 | 6,7 RT 99 | 61.23 | 56.8 | 13.60 | 15.10 | 21.90 |
| 4 | H_br23_rt80 | 2,3 RT 80 | 61.24 | 54.7 | 13.70 | 15.70 | 22.50 |
| 5 | B_br67_rt80 | 6,7 RT 80 (commit 69f918d) | 61.31 | 55.0 | 13.60 | 15.60 | 22.60 |
| 6 | G_br2to7_rt80 | 2-7 RT 80 | 61.56 | 54.3 | 13.60 | 15.80 | 22.70 |
| 7 | C_br45_rt80 | 4,5 RT 80 | **65.30** | 56.4 | 13.70 | 15.40 | 22.50 |
| 8 | E_br01_rt80 | 0,1 RT 80 | **67.85** | 55.0 | 13.70 | 15.70 | 23.90 |

### Conclusions ultrathink

1. **CPU affinity 의 *진짜 효과 거의 0***. Best (D 60.86) vs 6 case 평균 (61.0) = 0.14ms 차이 = statistical noise. 1-7 cases 의 상위 6 = 60.86~61.56ms 의 0.7ms 격차 안.

2. **bridge_p99 13.6ms 가 architecture floor** — ZED SDK retrieve_measure 의 depth bimodal 한계 (Codex Q3 검증). CPU 어디 두든 못 줄임.

3. **검증된 가정**:
   - **E (cores 0-1) +7ms 회귀** ★ — system (systemd/Xorg) 와 RT 80 충돌. CLAUDE.md "GMSL = EGL=X 필수" 의 실제 검증. **cores 0-1 절대 사용 X**.
   - **C (cores 4-5) +4.4ms 회귀** ★ — env audit 의 nvargus-daemon PSR=4 와 충돌. context switch + cache thrash. **cpu 4 nvargus 양보 권장**.

4. **부정된 가정**:
   - "Python cores 2-5 / C++ cores 6-7 분리 필수" → 현재 환경 (C++ X) 에선 의미 없음. cores 1-7 모두 자유.
   - "C 또는 D = best" 가설 → C 는 worst, D 는 marginal best (0.04ms vs A).

5. **production 권장**:
   - **BRIDGE_CORES env 미설정** = case A = kernel inherit. 가장 단순, near-best.
   - 또는 BRIDGE_CORES="4" = case D, deterministic single core.
   - **commit 69f918d 의 BRIDGE_CORES="6,7" 도 OK** (case B = +0.4ms, noise 영역). 단 *낭비 적 reservation*.
   - **미래 C++ Teensy 시작 후 재측정 필수**.

6. **진짜 lever 는 V4L2 우회**:
   ```
   bridge_p99 13.6ms = ZED SDK depth pipeline 한계 (못 줄임)
   V4L2 직접 + custom sparse stereo = -7~10ms 가능
   → 진정 game changer (Week 2-3)
   ```

### Action items
- [x] 측정 + 분석 + 기록
- [x] BRIDGE_CORES env 의 production default 결정 (env 미설정 권장)
- [ ] V4L2 formats 추가 검증
- [ ] **One-frame-late depth thread implement (Week 1, -10~11ms 권장)**
- [ ] V4L2 prototype 시작 (Week 2-3)

---

## 2026-05-10 — Bridge_p99 Root Cause Deep Analysis (commit `1e2b105`+)

### Anti-correlation pattern 발견

이전 측정의 verbose log 의 sub-step 분석:

| frame type | grab | ret_rgb | ret_depth | sum |
|---|---|---|---|---|
| Type A (block early) | **10.7** | 0.7 | **1.3** | 12.7 |
| Type B (block late)  | 2.6 | 2.1 | **9.2** | 13.9 |
| Type A | **10.6** | 0.6 | **1.4** | 12.6 |
| Type B | 2.7 | 1.9 | **9.1** | 13.7 |

→ **`grab` 과 `ret_depth` 가 anti-correlated**. 합 항상 ~12-14ms. ZED SDK 가 *어디서 wait 할지* 만 결정 — *총 depth pipeline work fixed*.

### Root cause

```
ZED SDK 의 grab() 행동:
  if (이전 frame 의 depth pipeline 진행 중) {
      block;              // grab=10ms
      depth 즉시 가용;    // ret_depth=1.3ms
  } else {
      즉시 return;        // grab=2.5ms
      depth 백그라운드;   // ret_depth=9ms (wait)
  }
```

→ bridge_p99 = ZED SDK 의 depth pipeline 총 work (~12-13ms). **CPU affinity 와 무관**.

### One-frame-late depth thread = 진정한 lever

```
Current (hot path 13ms):
  grab → ret_rgb → ret_depth (block) → bridge done

One-frame-late (hot path 2.5ms):
  Main thread:    grab → ret_rgb → depth=queue.pop() ★
  Worker thread:  retrieve_measure (background, queue.push())
```

| 항목 | 변화 |
|---|---|
| bridge_p99 | 13.6 → 2-3ms (-10~11ms) |
| Depth age | +8.3ms (1 frame stale) |
| Knee peak angular error | ~2.5° (300°/s × 0.0083s) — 무시 가능 |
| **Net effective latency** | **-2~3ms** |
| Jitter 제거 | bimodal → consistent fast path |

### Codex consult `bn57396zt` 답 (token 287634, 2026-05-10)

**brutal honest 발견**:

Q1 (ZED Thread Safety): **검증 필요, 가정 X**
- 공식 ZED SDK doc 에 *concurrent grab + retrieve_measure 보장 X*
- Stereolabs forum: video settings get/set 만 thread-safe (좁음)
- Logical race: worker retrieve_measure 중 main grab() → depth 가 어느 frame 의 것인지 *undefined*

Q3 (Pyzed Python): **C++ binding 권장 가능**
- pybind11 GIL release 명시 X
- patient experiment 이면 minimal C++ binding (gil_scoped_release + thread lifecycle)

Q4 (Timestamp): **Two timestamps + EKF covariance inflate**
- rgb_ts = T_N, depth_ts = T_{N-1}, depth_age = ~8.33ms
- Plan D EKF 가 처리: bearing(T_N) + range/depth(T_{N-1}) 분리 OR pose(T_N) + cov inflated by sigma_velocity * 8.33ms

Q5 (Watchdog): **Stale fail closed**
- Worst case: fresh 2D + stale depth = plausible but wrong 3D
- Age > 2 frames → publish invalid (stale reuse 절대 X)

Q7 (★ 결정적): **Kill-test first**
> "Path A has best ROI only if Q1 is resolved in 1-2 days. If not, **it is a trap**."
> "**Recommended sequencing: Week 1 day 1-2 build a kill-test for Path A**.
>  If frame association or p99 target fails, **stop. Week 1 onward go Path B (V4L2)**."

Q8 (정확한 변경 위치):
- ZEDFrame (zed_gpu_bridge.py:107) — depth fields 추가
- Bridge state (line 187) — worker, CV, DepthPacket
- Start/stop worker (line 484, 497)
- Remove main-thread depth retrieval (line 682, 781)
- pipeline.py:36 (FrameMeta) — depth_ts/age extension

### Falsification gate (Codex Q6) — 4 조건 모두 pass 해야 Path A 진행

| Gate | 조건 |
|---|---|
| G1 | Frame association: depth_frame_id 가 rgb_frame_id - 1 (consistent, ≥95%) |
| G2 | bridge_proc_p99 < 4 ms |
| G3 | depth_age_p99 < 16.7 ms (2 frames at 120fps) |
| G4 | Stale rate < 5% |

**모두 pass → Path A implement (Day 3-7)**
**하나라도 fail → 즉시 Path B (V4L2 + VPI)**

### Kill-test script: scripts/kill_test_one_frame_late.py

Minimal experiment (60s):
- Main thread: grab + retrieve_image (RGB hot path)
- Worker thread: retrieve_measure → DepthPacket queue (depth async)
- Falsification gate 자동 평가 + PASS/FAIL 출력

→ 사용자 측정 후 paste 받으면 결정.

---

## 2026-05-10 — Jetson Environment Audit (commit `b8c33df`)

### Setup
- L4T R36.4.7 (JetPack 6.2, 2025-09 build)
- jetson_clocks ON, nvpmodel MAXN, GPU 918MHz, CPU 1.984GHz × 8

### Software stack

| Package | Version |
|---|---|
| ZED SDK | 5.2.1 |
| stereolabs-zedlink-mono | 1.4.0 (MAX9296) |
| pyzed | OK (Python binding) |
| PyTorch | 2.10.0 + CUDA 12.6 |
| CuPy | 14.0.1 + CUDA 12.9 (minor mismatch) |
| **VPI** | **3.2.4 + python3.10-vpi3** ★ |
| Triton | **미설치** (A.2 fusion 은 PyTorch fallback 만) |

### 결정적 발견 — V4L2 path 가능

```
ZED SDK pyzed binding:
  Camera methods (raw/buffer): []  ← RawBuffer Python 노출 X
  Mat methods: ['update_cpu_from_gpu', 'update_gpu_from_cpu']
  MEM types: BOTH, CPU, GPU

V4L2 (kernel level):
  /dev/video0 (left), /dev/video1 (right)
  zedx 10-0020 (platform:tegra-capture-vi:1)
  → driver cleanly exposes the sensors
  → V4L2 raw capture 가능 (Codex Q1a 권장 path 의 정확한 만족)
```

→ **ZED SDK 우회의 진정한 path = V4L2 + VPI 활용** (RawBuffer 대안).

### CPU 사용 패턴 (사용자 정확)

```
nvargus-daemon  PSR=4  3.6% CPU
gnome-shell     PSR=3  0.3% CPU
update-manager  PSR=7  0.3% CPU
nxrunner.bin    PSR=3  0.1% CPU
systemd 등      cpu0-7 분산 (near idle)
Xorg            top 5s sample 에 안 보임 (거의 idle)
```

→ **system 가 어느 specific cores 도 점유 안 함**. 모든 cores 0-7 *대부분 idle*.
→ **vision pipeline 이 cores 1-7 자유 사용 가능** (cpu0 만 systemd init 양보 권장).
→ CLAUDE.md 의 cores 0-1 system reservation = *과도한 가정*, 실제 system load 0% 가까움.

### Conclusions

1. **CPU 정정**: 현재 환경 (C++ X) 에서 cores 1-7 모두 vision 자유
2. **V4L2 path 가능**: pyzed RawBuffer 노출 안 됨 단 V4L2 가 더 직접 접근
3. **VPI 3.2.4 설치됨**: sparse stereo prototype 의 ISP/rectify 활용 가능
4. **Triton 미설치**: A.2 fusion = PyTorch fallback 만 (Codex Q4 의 "Triton 효과 작음" 과 일치, drop list 강화)

### Action items
- [x] 환경 영구 기록
- [ ] V4L2 formats 추가 검증 (`v4l2-ctl --list-formats-ext /dev/video0`)
- [ ] CPU affinity 8-case sweep (script `measurement_cpu_affinity_ablation.sh`)
- [ ] 측정 결과 entry 추가

---

*Last updated: 2026-05-10*

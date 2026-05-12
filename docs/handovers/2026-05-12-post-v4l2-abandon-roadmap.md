# 2026-05-12 — Post V4L2 abandon Roadmap

**결정**: V4L2 우회 abandon (사용자 선택 C). Plan D EKF 집중.

## 현재 state (vision repo)

### PASS (Jetson 검증 완료)
- ✓ SHM v2 publisher (`shm_publisher.py`, 64+K*48 byte, seqlock OK)
- ✓ Quality Dataset I/O (`quality_dataset_io.py`, atomic write + 20 fields)
- ✓ batch_pose_compute (offline YOLO TRT, self-test PASS)
- ✓ check_v4l2_capability (BA10 detection — engineering record only)
- ✓ vpi_pipeline self-test (build_rectify_maps numpy)
- ✓ sparse_stereo_kernel self-test (3/3 correct, σ_z 8.4mm — archive 유지)

### ARCHIVED (사용자 결정 C)
- ⊗ v4l2_capture.py — tegra-capture-vi VIDIOC_S_FMT 거부
- ⊗ vpi_pipeline.py debayer/remap path (build_rectify_maps 는 별도 사용 가능)
- ⊗ sparse_stereo_kernel production CUDA (Python self-test 만 유지)

→ run_all_jetson_tests.sh: `RUN_V4L2_TESTS=1` 환경변수 시만 실행.

## 사용자 control repo 의무 (다음 phase)

### Prerequisites (이미 docs/lessons/ 작성 완료)
1. `shm_v2_packet_spec.md` — 274 lines, byte layout + invariants
2. `cpp_shm_v2_reader_skeleton.md` — 306 lines, #pragma pack struct + read pattern
3. `plan_d_predictor_spec.md` — 332 lines, EKF state + L1/L2/L3 cascade

### Implementation order (3주 estimate)

**Week 1**: SHM v2 reader (C++)
- `PlanDPacketHeaderV2` struct (#pragma pack, byte-perfect)
- seqlock read pattern (even seq → read → seq unchanged → accept)
- timestamp validation (rgb_ts > prev, depth_age_us < threshold)
- valid_mask_bits per-kp parsing
- Integration test: vision repo publisher 와 같이 실행 → 1000 packets read, 0 corruption

**Week 2**: Plan D EKF L1 (const velocity) + L2 (const accel)
- State x = [φ, ω, α] (gait phase, cadence, cadence accel)
- L1: pure constant velocity (ω̇ = 0)
- L2: constant acceleration (α̇ = 0)
- Measurement model: q (joint angles) from kpts_3d_m
- Per-kp R matrix from kp_sigma_m (SHM v2)
- Cold-start: 3 strides of measurements 의무

**Week 3**: Plan D EKF L3 (phase-locked) + integration
- Cycle template μ(φ): 128 bins, recursive update β=0.03-0.10
- Phase estimation: cross-correlation (NOT FFT/Hilbert — Codex Q1 rationale)
- Predict at τ ahead: cubic Hermite interpolation
- Cascade fallback chain: L3 → L2 → L1 → watchdog
- Falsification gate (innovation 너무 큼 → fallback)

### Vision repo 의무 (parallel, ~1주)
- Pipeline 통합: realtime/pipeline_main.py → shm_publisher v2 호출
- 현재 SHM v1 → v2 migration
- per-kp covariance 추가 (depth_uncertainty_sigma 사용, sparse_stereo_kernel.py 의 함수)
- Quality dataset dump mode (npz per-frame, optional)

## 검증 plan (Week 4-5)

### Phase 1: Latency
- Vision sensor: 60ms p99 (vendor SDK 측정, ZED X Mini)
- EKF prediction: 50-70ms ahead, error < 5° joint angle p95
- Effective latency = sensor - prediction lead = ~10ms ★

### Phase 2: HS accuracy (healthy gait, single subject)
- HS p95 error ≤ 30ms (clinical-grade gate)
- Phantom HS rate ≤ 2% per 30 steps
- Missed HS rate ≤ 2% per 30 steps

### Phase 3: Reliability
- 30 min continuous (no crash, valid_mask integrity 100%)
- 60 min (cumulative drift, ω/α stability)
- 120 min (memory leak, thermal throttle)

## Paper plan (Week 6+)

- Section III: System architecture (single-path: vendor SDK + EKF)
- Section V: SHM v2 packet contract
- Section VI: Plan D EKF predictor (main novelty)
- Section VII: Camera-agnostic interface
- Section VIII: Validation (latency + accuracy + reliability)
- Section IX.A: V4L2 investigation (engineering lesson — 정직)

## 차주 (2026-05-19) 시작 의무

1. 사용자 control repo clone + IDE setup
2. `docs/lessons/cpp_shm_v2_reader_skeleton.md` 읽기
3. C++ project skeleton (CMakeLists.txt, src/shm_reader.cpp)
4. vision repo publisher + control repo reader integration smoke test

진정 — Plan D EKF 가 *진정 game changer*. effective latency 10ms 목표 달성 가능.

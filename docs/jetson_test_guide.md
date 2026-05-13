# Jetson 단독 풀 검증 가이드

Teensy/AK60 모터 없이도 Jetson에서 실행 가능한 모든 테스트의 순서·명령·기대값.
**카메라(ZED X Mini)만 연결되어 있으면 끝까지 진행 가능.**

작성: 2026-05-13. 검증된 commit: `2c21c01` (codex 4차 review GATE PASS).

---

## 0. 사전 조건

- ZED X Mini 카메라 연결 (GMSL2)
- TRT engine 빌드 완료: `src/perception/CUDA_Stream/yolo26s-lower6-v2.engine`
  - 없으면 `git lfs pull && /usr/src/tensorrt/bin/trtexec --onnx=...` 먼저
- Python deps: `pip install -r requirements.txt` (pyserial 옵션, mock 모드면 불필요)

체크:
```bash
cd ~/realtime-vision-control
ls src/perception/CUDA_Stream/yolo26s-lower6-v2.engine || echo "MISSING ENGINE"
git rev-parse --short HEAD       # 2c21c01 이상
```

---

## 1. 한 줄 통합 테스트 (권장)

전체 7단계 자동 실행. **이게 메인 흐름이라 다른 거 다 무시해도 됨.**

```bash
bash scripts/jetson_full_test.sh
```

산출물: `recordings/jetson_full_YYYYMMDD_HHMMSS/`
- `01_pytest.log` — Mac/Jetson 공통 30개 unit 테스트
- `02_perf.log` — `nvpmodel -m 0` + `jetson_clocks`
- `03_pipeline.log` — pipeline_main 30s 부팅 + SHM 생성
- `04_dump_shm.log` — `/hwalker_pose_v2` + `/hwalker_forecast` layout 5s watch
- `05_bridge_mock.log` — SHM → MockSerial bridge 10s
- `06_analyze.log` — trace CSV latency 분석 (있으면)
- `SUMMARY.txt` — 전체 PASS/FAIL

기대 결과:
```
PASS: all unit tests passed
PASS: performance mode applied (nvpmodel + jetson_clocks)
PASS: pipeline running (pid=...)
PASS: SHM layout sane
PASS: bridge mock: NNN commands + 0 heartbeats over 10s
=== ALL TESTS PASSED ===
```

Walking session 포함하려면:
```bash
bash scripts/jetson_full_test.sh --walking
```

---

## 2. 개별 단계 (수동 디버깅)

### 2.1 Unit tests 만
```bash
PYTHONPATH=src:src/perception/benchmarks python3 -m pytest \
    tests/test_phase_b_integration.py \
    tests/test_teensy_protocol.py \
    tests/test_bridge_flow.py \
    -v
```
기대: `30 passed`

### 2.2 Performance mode
```bash
sudo nvpmodel -m 0
sudo jetson_clocks
sudo jetson_clocks --show         # GPU 918MHz, CPU 1.98GHz 확인
```

### 2.3 Pipeline 단독 + SHM v2 + Plan D
```bash
PYTHONPATH=src:src/perception/benchmarks python3 \
    src/perception/realtime/pipeline_main.py \
    --no-display --method B \
    --enable-shm-v2 --enable-plan-d
```
기대 로그 (10초 안에):
- `Method B inferred from L_hip, L_knee, ...`
- `[PROFILE] ... total: 14-20ms`
- `SHM /hwalker_pose_v2 created (...B)`
- `Plan D cascade: L1→L2 transition at frame N`

Ctrl+C로 중단.

### 2.4 SHM layout 검증 (별도 터미널)
pipeline_main 돌고 있는 동안:
```bash
PYTHONPATH=src python3 scripts/dump_shm.py
```
또는 5s 연속 watch:
```bash
PYTHONPATH=src python3 scripts/dump_shm.py --watch 5 --rate 5
```
기대:
```
── /hwalker_pose_v2 ─────
  seq=NNNN  parity=EVEN(stable)
  version=2   (expect 2)
  K=6
  → PASS: layout sanity

── /hwalker_forecast ────
  cascade_level=2  is_ready=0/1
  q_pred (rad) by joint:
    [0] L_hip     q=+0.123  σ=+0.045
    ...
  → layout PASS, valid_for_control=YES/NO
```

`is_ready=0`이면 정지 상태 또는 cascade L1 — walking 시작하면 L2로.

### 2.5 SHM → MockSerial bridge
pipeline_main 돌고 있는 동안:
```bash
PYTHONPATH=src python3 scripts/shm_to_teensy_bridge.py \
    --mock --duration 10 --rate-hz 200 --verbose
```
기대 (walking 중):
```
[ 5.00s] tx_rate=200.0Hz cmds=1000 hb=0 fc_no_data=0 telem_good=0
=== Bridge summary ===
  cmds_sent: 2000
  heartbeats_sent: 0
  fc_stale: 0
```
기대 (정지 상태):
```
  cmds_sent: 0
  heartbeats_sent: 2000
  fc_not_ready: ~2000
```

### 2.6 Walking session 60s
```bash
bash scripts/walking_session.sh 60
```
대화형 — Enter 누르고 60초 walking. 산출물: `recordings/walking_YYYYMMDD_HHMMSS/`
- `walking_*.svo2` — ZED 원본
- `walking_*.npz` — 6 keypoints × frames
- `trace_*.csv` — T0~T9 latency
- `analyze_*.txt` — RT metric A/B/C 분석
- `plan_d_*.log` — cascade transition log

---

## 3. Jetson 실험 후 결과 paste

walking 끝나면 다음 두 개만 채팅에 붙여넣기:
```bash
cat recordings/walking_*/analyze_*.txt | tail -40
cat recordings/walking_*/plan_d_*.log | tail -30
```

또는 jetson_full_test.sh 후:
```bash
cat recordings/jetson_full_*/SUMMARY.txt
```

---

## 4. Teensy 추가하기 (다음 단계)

Teensy 4.1 펌웨어 빌드 + 업로드: `firmware/teensy_host_receiver/README.md` 참조.

연결 후 풀 체인:
```bash
# 터미널 1: pipeline
PYTHONPATH=src python3 src/perception/realtime/pipeline_main.py \
    --no-display --method B --enable-shm-v2 --enable-plan-d

# 터미널 2: bridge (mock 제거)
ls /dev/ttyACM*    # Teensy 포트 확인
PYTHONPATH=src python3 scripts/shm_to_teensy_bridge.py \
    --port /dev/ttyACM0 --duration 60 --verbose

# 터미널 3: dump_shm 모니터링
PYTHONPATH=src python3 scripts/dump_shm.py --watch 60 --rate 2
```

기대: bridge `telem_rx_good` 가 늘어남 (Teensy가 100Hz 텔레메트리 송신). `clamp_reasons` dict에 `OK` 늘어나면 정상.

**모터는 단계적으로 연결.** `firmware/teensy_host_receiver/README.md`의 5단계 켜기 절차 따를 것.

---

## 5. 실패 시나리오 점검

| 증상 | 원인 | 조치 |
|---|---|---|
| pytest fail | 코드 베이스 변화 | `git log` 확인, `git diff HEAD~5` |
| SHM not found | pipeline_main 안 띄움 또는 `--enable-shm-v2` 누락 | 옵션 추가 후 재시작 |
| dump_shm `seq=ODD` | writer in progress (정상) 또는 producer dead | watch 모드로 5회 재확인 |
| bridge `fc_stale > 0` | pipeline 멈춤 / 카메라 lost | pipeline 재시작 |
| pipeline `e2e > 20ms` 빈번 | jetson_clocks 미적용 또는 NEURAL depth 설정 | `--enable-plan-d` 대신 minimal 옵션부터 |
| `valid_for_control=NO` 계속 | Plan D cascade가 L1 멈춤 | 실제 walking 시작 필요 (정지에선 L2 못 감) |

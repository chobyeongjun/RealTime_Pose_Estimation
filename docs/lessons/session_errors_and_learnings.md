# Session Errors + Learnings — 영구 학습 기록

**작성**: 2026-05-12. 사용자 의지 "모든 오류 영구 기록, 같은 실수 반복 X".

이 file = *모든 실수 + 의외 발견* 의 영구 reference. 다음 session 또는 협력자 의 prereq.

---

## 1. SSH & Environment

### 1.1 SSH username mismatch (Mac vs Jetson)
- **증상**: `ssh 192.168.0.55` → Permission denied
- **원인**: Mac user = `chobyeongjun`, Jetson user = `chobb0`. 기본 `ssh` 가 Mac user 사용.
- **해결**: `ssh chobb0@192.168.0.55` 또는 `~/.ssh/config` 의 `User chobb0`.
- **검증**: `ssh -vvv` 의 `Authentications that can continue: publickey,password`.
- **Lesson**: SSH 의 *username* 명시. publickey 인증 시 user mismatch 가 password fail 처럼 보임.

### 1.2 nvargus-daemon Permission denied (ZED X GMSL)
- **증상**: `python3 -m perception.* ` (non-sudo) → "Connecting to nvargus-daemon failed: Connection refused"
- **원인**: nvargus-daemon socket ACL = root only. user 가 video group 에 있어도 fail.
- **해결**: `sudo` 의무 (ZED X GMSL2 capture).
- **Lesson**: GMSL camera 의 *root 의무*. USB camera 와 다름.

### 1.3 PYTHONPATH 누락 (sudo + user-local install)
- **증상**: `sudo python3 -m perception.*` → `ModuleNotFoundError: No module named 'perception'` 또는 `'pyzed'`
- **원인**: `sudo` 가 user 의 PYTHONPATH inherit X. user-local 의 pyzed (`~/.local/lib/python3.10/site-packages`) + repo `src/` 모두 누락.
- **해결**:
  ```bash
  sudo PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:$PWD/src" \
      python3 -m perception.CUDA_Stream.module
  ```
- **Lesson**: `sudo -E` 만으론 부족 (Python user site-packages 는 *환경 변수* 가 아니라 *Python 의 user run 시* 자동 추가). 명시 PYTHONPATH 의무.

---

## 2. Build & Library 의 의외

### 2.1 ZED X Mini distortion model = 12 elements (not 5)
- **증상**: `ValueError: left_cam.disto must be length-5, got list len=12`
- **원인**: ZED X Mini = wide FOV 110° → Brown-Conrady extended model (12 coeffs: k1..k6, p1, p2, s1..s4). Standard pinhole 5 가 아님.
- **해결**: `4 <= len(disto) <= 12` 허용.
- **Lesson**: ZED model 가정 X. 실제 측정/문서 검증 후 코드.

### 2.2 disto 모두 0 (rectified output 의 calibration)
- **증상**: ZED X Mini 의 `left_cam.disto = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`
- **원인**: ZED SDK 가 *rectified output* 의 calib 제공. distortion 이미 적용 후.
- **해결**: V4L2 raw path 시 `calibration_parameters_raw` 사용 (Codex Q5 prior).
- **Lesson**: ZED `calibration_parameters` ≠ raw. rectified vs raw 의 구분.

### 2.3 numpy.savez_compressed 의 자동 `.npz` 추가
- **증상**: `os.replace(tmp_path, output_path)` → FileNotFoundError. tmp_path = `frame.npz.tmp` 단 numpy 가 `frame.npz.tmp.npz` 작성.
- **원인**: `np.savez_compressed` 가 *.npz 가 아닌 경로* 에 *.npz 자동 append.
- **해결**: tmp path 도 *.npz 로 끝나게* (예: `.{stem}.{uuid}.npz`).
- **Lesson**: numpy file API 의 *자동 extension* 행동. atomic write 시 의무.

### 2.4 PIL 의 JPEG encode 의 alpha channel drop
- **증상**: BGRA → JPEG round-trip 의 alpha = 255 reconstruct
- **원인**: JPEG = no alpha channel.
- **해결**: alpha 의무 reconstruction 명시 (decode 시 255).
- **Lesson**: JPEG 의 RGB-only. PNG/TIFF 가 alpha 호환. BGRA → JPEG 시 alpha 무시 의무 명시.

### 2.5 Triton 미설치 on Jetson aarch64
- **증상**: `ModuleNotFoundError: No module named 'triton'`
- **원인**: Triton 의 aarch64 (Jetson) wheel 부재. 일반 NVIDIA GPU x86 만 지원.
- **해결**: PyTorch fallback path. A.2 Triton kernel = abandon (Codex Q4 권유).
- **Lesson**: Jetson 의 라이브러리 호환 검증 (env audit 의 의무).

---

## 3. Process & Test 의 의외

### 3.1 launch_clean.sh 60 의 카메라 점유
- **증상**: 3 cases 한 줄에 복붙 → 측정 무효 (51.6Hz vs expected 67Hz)
- **원인**: launch_clean.sh 60 = 60s 동안 ZED 카메라 점유. 동시 X.
- **해결**: 각 case 60s 끝까지 기다림 후 다음.
- **Lesson**: 카메라 자원 *exclusive*. shell 의 sequential 단순 명령 의무.

### 3.2 Python logging buffer (sudo + tee 조합)
- **증상**: kill_test 의 `[main] entering loop` 직후 silent exit. 5 turns 진단 trap.
- **원인**: Python logging 의 internal handler buffer + sudo + tee 의 pipe buffer. `python3 -u` 만으론 부족.
- **해결**: `class FlushHandler(StreamHandler): emit + flush`. 또는 `print(..., flush=True)` 직접.
- **Lesson**: Python logging buffer 의 *pipe + sudo 환경* 의 fail. critical script 는 explicit flush.

### 3.3 `$?` vs `${PIPESTATUS[0]}` (sudo + tee 의 exit code)
- **증상**: `sudo ... | tee` 후 `echo "exit=$?"` 가 *tee 의 exit (0)* 만 보여줌. Python 의 진짜 exit code 다름.
- **원인**: `$?` = pipe 의 마지막 command (tee) 의 exit. Python = `${PIPESTATUS[0]}`.
- **해결**: `echo "python_exit=${PIPESTATUS[0]}"`.
- **Lesson**: pipe + `$?` = 마지막 command. PIPESTATUS array.

### 3.4 pytest 의 anyio plugin _pytest.scope (Jetson)
- **증상**: `pytest tests/` → `ModuleNotFoundError: No module named '_pytest.scope'`
- **원인**: anyio plugin 이 더 새 pytest version 의 `_pytest.scope` 의존. Jetson 의 시스템 pytest 가 오래됨.
- **해결**: `pytest -p no:anyio` 또는 `pip install --user --upgrade pytest`. 또는 *verify_*.py 직접 실행* (pytest 우회).
- **Lesson**: Jetson 의 시스템 Python ecosystem 의 old version 의무. pytest 의 plugin compat.

### 3.5 ZED Camera concurrent retrieve_measure (multi-thread, Codex Q1)
- **증상**: One-frame-late depth thread 시도 — 5 turns silent exit.
- **원인**: ZED SDK 공식 doc 에 *concurrent grab + retrieve_measure on same Camera* 보장 X. 실제 race condition 가능.
- **해결**: **abandon** (Codex 권유). 또는 minimal C++ binding + gil_scoped_release.
- **Lesson**: SDK contract 의 *명시 보장* 부재 시 abandon. *명시 doc* 의 의무.

---

## 4. Codex Review 의 가치

### 4.1 Self-review 의 진정 한계
- **3-iteration self-review 후** Codex 가 **8 P1 + 7 P2** 발견 (SHM v2 review b1ky3965z)
  - Watchdog 의 v1 publish call (clinical blocker)
  - view_sagittal / verify_world_frame mis-unpack
  - 2 layout 모순 (Section 2 vs 3.5)
  - cross-process memory ordering 미정의
  - depth timestamp contract 미강제
  - per-kp validity fictional
- **10-iteration self-review 후** Codex 가 **2 P1 + 9 P2** 발견 (Quality Dataset b5kic9w4n)
  - QualityFrame ≠ SHM v2 17-tuple (ts_domain 누락)
  - Pose placeholder contradiction (mask=0 + reason=VALID_OK)
  - atomic write 부재
  - --force 의 stale 안 지움

→ **Self-review 만으론 clinical blocker 미검출 가능**. Outside view (Codex) = *진정한 가치*.

### 4.2 Codex 의 prior art evidence (vision-only EKF)
- "Atalante 2025 → IMU on vest, NOT vision"
- "Honda Walking Assist → hip motor sensors, NOT vision"
- "Stenum 2021 OpenPose → worst HS 60ms, healthy only"
- **결론**: vision-only EKF = *research path*, NOT clinical SOTA. 사용자 academic challenge.

### 4.3 Brutal honest 의 의지
- "Stop trying to remove 30+ ms from camera stack in 4-6 weeks."
- "V4L2 = secondary upside, drop if blocks."
- "Latency 의 마지막 1ms 추구 = trap."
- → Quality + reliability > raw latency.

---

## 5. Architecture 의 의외

### 5.1 Bridge_p99 anti-correlation 패턴
- **발견**: `grab` 과 `ret_depth` anti-correlated. 합 항상 ~13ms.
- **의미**: ZED SDK 의 *depth pipeline 총 work fixed*. *어디서 wait 만 결정*.
- **함의**: CPU affinity / bridge resource / one-frame-late 으로 *직접 절감 X*. 별도 path (V4L2 또는 SHM v2 + 1-frame-late) 만 가능.
- **Lesson**: 측정 의 *aggregate p99* 보다 *sub-step distribution* 분석.

### 5.2 CPU affinity 효과 = noise (Jetson Orin NX homogeneous)
- **8-case ablation**: best 60.86ms (D, single core 4) vs 6 평균 61.0ms = 0.7ms noise.
- **원인**: Cortex-A78AE 8 cores homogeneous + L3 2MB shared + system load near 0.
- **Lesson**: 측정 후 결정. *추측 으로 affinity tuning* trap.

### 5.3 Production default 정정 사례
- `--gpu-stream-priority` default = `infer-only` (hardcoded) → 측정 검증 후 `all-high` (-7.7ms p99).
- stream_manager 의 *기존 의도* (all stages high on TRT 10.x) 가 *맞았음*. hardcoded 가 부분 적용.
- **Lesson**: hardcoded default 의 *측정 검증* 의무. flag 화 + ablation.

---

## 6. Process Learnings (사용자 의지)

### 6.1 "10번씩 검토" 의 진정 의미
= **Self-review (10 iter) + Codex outside (1+) + Mac PASS + Jetson PASS = 12+ iterations**.

이번 세션 적용 결과:
- Week 0: 16+ iterations (self 10 + Codex 6)
- Clinical blocker 10 P1 발견 (self-review 만으론 미검출 risk)

### 6.2 "한 번에 single command 로 모든 것 확인" 의지
= `scripts/verify_*.py` 의 통합 verify (8 sections, 30+ checks, ALL PASS / FAIL gate).

장점: 사용자 paste 부담 작음. 즉시 분석.

### 6.3 "결과 모두 영구 기록"
= `docs/experiments/measurements_log.md` + `docs/lessons/*.md`.

이번 세션: 측정 history, env audit, P1/P2 fix history, errors+learnings (이 file).

---

## 7. 미해결 (다음 phase 의 prereq)

| 항목 | 어디서 학습 |
|---|---|
| ZED SDK concurrent retrieve 의 진짜 contract | Stereolabs forum 또는 source code (closed) |
| Triton on Jetson aarch64 — wheel 가능성 | future PyTorch release |
| V4L2 의 ZED X 의 정확 raw format | Jetson 의 `v4l2-ctl --list-formats-ext` 측정 |
| ARM Orin 의 cross-process atomic 의 정확 ordering | clinical 직전 의무 audit (C++ binding 또는 inline asm) |
| Plan D EKF 의 phase template 의 *clinical patient* 의 cycle 깨짐 | Week 4-5 stress test |

---

*Last updated: 2026-05-12. 새 실수 발생 시 즉시 entry 추가.*

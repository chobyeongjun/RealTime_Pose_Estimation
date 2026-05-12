"""
pipeline_main.py
================
ZED + YOLO26s → KF → calibration → SHM publisher
50Hz 실시간 루프

사용법:
    python3 pipeline_main.py                    # ZED 실시간, Method A
    python3 pipeline_main.py --svo2 walk.svo2   # SVO2 파일 재생
    python3 pipeline_main.py --method A         # Standing Calibration (기본)
    python3 pipeline_main.py --method B         # ZED IMU World Frame
    python3 pipeline_main.py --no-display       # OpenCV 창 없음
    python3 pipeline_main.py --no-trt           # TensorRT 없이 PyTorch 직접 실행
"""

from __future__ import annotations

import argparse
import gc
import signal
import sys
import os
import time
from collections import deque

import numpy as np


# ─── Peak latency 방어 ────────────────────────────────────────────────────────
def _apply_latency_defenses():
    """GC + CPU affinity로 Python jitter 최소화.

    1. gc.disable(): 자동 GC가 random 프레임에서 2-5ms pause 유발
       → 수동으로 gen-0 (fastest) 주기 collect
    2. CPU affinity: OS가 다른 프로세스로 preempt 하는 걸 줄임
       Jetson Orin NX 16GB는 8 core ARM A78. core 0-1 system에 양보, 2-7 고정.
    """
    gc.disable()
    gc.collect()  # 초기 누적분 1회 정리 후 disable 상태 유지
    try:
        # Python은 cores 2-5 (4 cores 전용).
        # cores 0-1: 시스템, cores 6-7: C++ hw_control_loop 용.
        # 완전 격리로 context switch / cache pollution 차단.
        os.sched_setaffinity(0, {2, 3, 4, 5})
        print("[Pipeline] CPU affinity: cores 2-5 (전용), GC disabled")
    except (AttributeError, OSError) as e:
        print(f"[Pipeline] CPU affinity 설정 실패: {e} (GC만 적용)")

# 이 파일 기준 경로 설정
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_REALTIME_DIR = _HERE
_PERCEPTION_DIR = os.path.join(_HERE, '..')
if _PERCEPTION_DIR not in sys.path:
    sys.path.insert(0, _PERCEPTION_DIR)

from shm_publisher import ShmPublisher, FlexionAngles, SHM_NAME
from kf_smoother import GaitKalmanFilter
from calibration import StandingCalibration, ZEDIMUWorldFrame
from joint_3d import compute_joint_state
from safety_guard import DepthSafetyGuard, SafetyLevel
from bone_constraint import BoneLengthConstraint

# benchmarks 경로를 모듈 레벨에서 한 번만 추가
_BENCH_DIR = os.path.join(_HERE, '..', 'benchmarks')
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

# 모듈 레벨 import (루프 안에서 반복 import 방지)
try:
    from postprocess_accel import batch_2d_to_3d as _batch_2d_to_3d  # type: ignore
    from zed_camera import create_camera as _create_camera, PipelinedCamera  # type: ignore
    _HAS_BATCH_3D = True
except ImportError:
    _HAS_BATCH_3D = False
    _create_camera = None
    PipelinedCamera = None

try:
    from trt_pose_engine import TRTPoseEngine  # type: ignore
    _HAS_DIRECT_TRT = True
except ImportError:
    _HAS_DIRECT_TRT = False

# ─── 상수 ─────────────────────────────────────────────────────────────────────
TARGET_HZ       = 60.0
TARGET_DT       = 1.0 / TARGET_HZ          # 0.0167 s
LOOP_WARN_MS    = 25.0                      # 22ms 평균, 25ms 초과 시 경고
CAMERA_LATENCY  = 0.021                     # KF latency 보상 [s]
DEFAULT_MODEL   = "yolo26s-lower6-v2.pt"


# ─── 메인 클래스 ──────────────────────────────────────────────────────────────

class Pipeline:
    """
    ZED 카메라 → YOLO 포즈 추정 → Kalman Filter → SHM publish 전체 파이프라인.

    상태 머신:
        CALIBRATING  : StandingCalibration이 n_frames 채울 때까지 대기
        RUNNING      : 정상 루프
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args   = args
        self.pub    = ShmPublisher(SHM_NAME)
        self.kf     = GaitKalmanFilter(dt=TARGET_DT, angle_noise_deg=2.5,
                                       accel_noise_deg_s2=1000.0)
        self.guard      = DepthSafetyGuard()
        self.calibrator = None   # StandingCalibration or ZEDIMUWorldFrame
        self.camera     = None
        self.model      = None
        self._running   = False
        self._fps_count = 0
        self._fps_t0    = 0.0
        # Bone length constraint (3D outlier 제거, Method A/B 공통).
        # std_threshold: 기본 10mm. 노이즈 많은 환경이면 --bone-std 로 완화 가능.
        self.bone_bc = BoneLengthConstraint(
            tolerance=getattr(args, 'bone_tol', 0.20),
            std_threshold=getattr(args, 'bone_std', 0.010),
            min_samples=20,
        )
        self._bone_calib_target = 30
        self._bone_hit_recent = 0

        # Joint velocity bound (P0-4): 프레임간 각도 점프 검출
        # 인간 무릎 최대 각속도 ≈ 900°/s (sprint 기준). 70Hz에선 14°/frame.
        # 안전 마진 포함 20°/frame 허용 (0.9초 만에 180° 가능한 속도).
        # 이를 초과하면 outlier (keypoint 오류, depth 튐) → 이전 값 유지 + 플래그.
        self._vel_prev_flexion = {
            'left_knee': None, 'right_knee': None,
            'left_hip':  None, 'right_hip':  None,
        }
        self._vel_max_delta_deg = 20.0
        self._vel_hit_counter = 0
        self._vel_total_counter = 0

    # ── 초기화 ────────────────────────────────────────────────────────────────

    def _init_camera(self):
        """ZED 카메라 초기화 + PipelinedCamera 래핑."""
        if _create_camera is None:
            raise ImportError("zed_camera 모듈 없음. benchmarks 경로 확인.")

        svo2 = self.args.svo2
        depth_mode = self.args.depth_mode
        if svo2:
            raw = _create_camera(use_zed=True, video_path=svo2, depth_mode=depth_mode)
            print(f"[Pipeline] SVO2 파일 열기: {svo2} (depth_mode={depth_mode})")
        else:
            raw = _create_camera(use_zed=True, video_path=None,
                                 depth_mode=depth_mode, fps=120)
            print(f"[Pipeline] ZED 실시간 (depth_mode={depth_mode})")

        # PipelinedCamera: grab을 predict와 병렬화
        # skip_imu=True: Method A/B 모두 런타임 IMU 불필요 (Method B는 static R)
        # serialize_depth=True (P0-5): depth retrieve를 predict 뒤로 미뤄 GPU 순차 실행
        #                              → predict spike 제거, 평균 +1-2ms 감수
        if PipelinedCamera is not None:
            self.camera = PipelinedCamera(
                raw,
                skip_imu=True,
                serialize_depth=self.args.serialize_depth,
            )
            self._raw_camera = raw
        else:
            self.camera = raw
            self._raw_camera = raw

        self.camera.open()
        print("[Pipeline] 카메라 열기 완료 (PipelinedCamera)")

    @property
    def _inner_camera(self):
        """intrinsics 등 접근용 원본 카메라"""
        return self._raw_camera

    def _init_model(self):
        """DirectTRT 우선, 실패 시 Ultralytics 폴백."""
        _bench = os.path.join(_HERE, '..', 'benchmarks')
        if _bench not in sys.path:
            sys.path.insert(0, _bench)

        _model_dir = os.path.join(_HERE, '..', 'models')
        imgsz = self.args.imgsz

        if _HAS_DIRECT_TRT and not self.args.no_trt:
            engine_path = os.path.join(_model_dir, f'yolo26s-lower6-v2-{imgsz}.engine')
            print(f"[Pipeline] DirectTRT 모드 (imgsz={imgsz})")
            self.model = TRTPoseEngine(engine_path, imgsz=imgsz)
            self.model.load()
            self._use_direct_trt = True
        else:
            from pose_models import LowerBodyPoseModel
            model_path = self.args.model
            use_trt = not self.args.no_trt
            self.model = LowerBodyPoseModel(
                model_path=model_path, use_tensorrt=use_trt,
                imgsz=imgsz, smoothing=0.0, segment_constraint=False,
            )
            self.model.load()
            self._use_direct_trt = False
            print(f"[Pipeline] Ultralytics 모드: {model_path}")

    def _init_calibrator(self):
        """Method A 또는 B 캘리브레이터 초기화."""
        method = self.args.method.upper()
        if method == 'A':
            self.calibrator = StandingCalibration(n_frames=30)
            print("[Pipeline] Method A: Standing Calibration 시작 — 똑바로 서 있으세요")
        elif method == 'B':
            # ZEDIMUWorldFrame.init()은 raw camera의 grab()/zed를 직접 호출.
            # 이 시점에는 PipelinedCamera의 capture 스레드가 consumed.wait에서 블록됨
            # (open() 후 첫 grab만 한 상태) → raw camera 단독 사용은 thread-safe.
            self.calibrator = ZEDIMUWorldFrame(self._raw_camera)
            ok = self.calibrator.init(warm_up_frames=20)
            if not ok:
                print("[Pipeline] Method B IMU 초기화 실패 → Method A로 전환", file=sys.stderr)
                self.calibrator = StandingCalibration(n_frames=30)
                self.args.method = 'A'
            else:
                print("[Pipeline] Method B: ZED IMU World Frame 준비 완료")
        else:
            raise ValueError(f"--method 는 A 또는 B 이어야 합니다: {method}")

    def setup(self) -> None:
        """전체 초기화 순서."""
        print("[Pipeline] 초기화 시작...")
        self._init_camera()
        self._init_model()
        self._init_calibrator()
        self.pub.open()
        print("[Pipeline] SHM 오픈 완료 →", SHM_NAME)
        # 초기화 후, 루프 직전에 latency 방어 적용
        # (init에서 생긴 객체들 한번 정리 후 GC off)
        _apply_latency_defenses()
        print("[Pipeline] 초기화 완료. 루프 시작.\n")

    # ── 루프 ──────────────────────────────────────────────────────────────────

    def _process_frame(self, t0: float) -> None:
        """최소화된 프레임 처리: grab → predict → 3D → SHM.
        Safety/KF 제거. C++ 루프가 100Hz 정밀 타이밍 + 제어 담당.
        """
        _t0 = time.perf_counter()
        ts_us = t0 * 1e6

        # ① RGB만 먼저 → release 즉시 → capture가 다음 grab 시작 (predict와 병렬)
        if hasattr(self.camera, 'get_rgb') and hasattr(self.camera, 'get_depth_and_gravity'):
            rgb = self.camera.get_rgb()
            if rgb is None:
                return
            # ★ 핵심: release 즉시 호출 → capture 스레드가 다음 frame grab 시작
            #   (현재 frame의 depth retrieve는 capture 내부에서 계속 진행됨)
            self.camera.release()
            _gravity = None
            depth = None
            new_api = True
        elif hasattr(self.camera, 'get') and hasattr(self.camera, 'release'):
            rgb, depth, _gravity = self.camera.get()
            if rgb is None:
                return
            self.camera.release()
            new_api = False
        else:
            if not self.camera.grab():
                return
            rgb = self.camera.get_rgb()
            depth = self.camera.get_depth() if hasattr(self.camera, 'get_depth') else None
            new_api = False
        _t1 = time.perf_counter()

        # ② 추론 (capture 스레드가 depth retrieve — GPU 경합 허용)
        is_bgra = (rgb.ndim == 3 and rgb.shape[2] == 4)
        if is_bgra and hasattr(self.model, 'predict_bgra'):
            result = self.model.predict_bgra(rgb)
        elif is_bgra:
            import cv2
            result = self.model.predict(cv2.cvtColor(rgb, cv2.COLOR_BGRA2BGR))
        else:
            result = self.model.predict(rgb)
        _t2 = time.perf_counter()

        # ③ depth 회수 (predict와 병렬 retrieve됨, 대부분 0ms 대기) → 3D
        if new_api:
            depth, _gravity = self.camera.get_depth_and_gravity()
        if _HAS_BATCH_3D and depth is not None and result.detected:
            raw_3d = _batch_2d_to_3d(result.keypoints_2d, depth, self._inner_camera)
        else:
            raw_3d = {}

        # ③.5 Bone length constraint — 보행 중 outlier 투영 보정 (Method 무관)
        #     캘리브 중엔 샘플만 쌓고 apply 안 함. ref 확정 후 매 프레임 apply.
        if self.bone_bc.ready:
            raw_3d, self._bone_hit_recent = self.bone_bc.apply(raw_3d)
        else:
            self._bone_hit_recent = 0
        _t3 = time.perf_counter()

        # ④ JointState
        kp2d  = getattr(result, 'keypoints_2d', {})
        confs = getattr(result, 'confidences',  {})
        state = compute_joint_state(kp2d, raw_3d, confs, timestamp_us=ts_us)

        # ⑤ Calibration (시작 30프레임만)
        method = self.args.method.upper()
        if method == 'A':
            cal: StandingCalibration = self.calibrator
            if not cal.done:
                cal.update(state)
                # 뼈 길이 샘플도 함께 수집 (정자세 같은 프레임 재사용)
                if state.valid and raw_3d:
                    self.bone_bc.add_sample(raw_3d)
                prog = cal.progress
                bone_n = self.bone_bc.sample_count()
                print(f"\r  [CalibA] {prog*100:.0f}% ({int(prog*30)}/30)  "
                      f"[Bone] {bone_n}/{self._bone_calib_target} samples   ",
                      end='', flush=True)
                return
            flexion = cal.to_flexion(state)
        else:
            wf: ZEDIMUWorldFrame = self.calibrator
            flexion = wf.to_flexion(state)
            # Method B는 IMU warmup만 하고 바로 진행 → bone 캘리브 별도 phase.
            # tried_finalize=True면 시도 끝났으니 더 이상 sample/finalize 안 함.
            if (not self.bone_bc.ready
                    and not self.bone_bc.tried_finalize
                    and state.valid and raw_3d):
                self.bone_bc.add_sample(raw_3d)
                n = self.bone_bc.sample_count()
                if n >= self._bone_calib_target:
                    self.bone_bc.finalize()   # 성공/실패 단 한 번 로그
                elif n % 10 == 0 and n > 0:
                    print(f"  [Bone] {n}/{self._bone_calib_target} samples (서 계세요)")

        # Method A가 방금 캘리브 완료한 경우: bone constraint도 finalize (한 번만)
        if (method == 'A' and self.calibrator.done
                and not self.bone_bc.ready
                and not self.bone_bc.tried_finalize
                and self.bone_bc.sample_count() >= self._bone_calib_target):
            self.bone_bc.finalize()

        # ⑤.5 Joint velocity bound (P0-4): 프레임간 점프 감지 + 억제
        #     20°/frame 초과 시 이전 값 유지 (인간 생리적 불가능).
        #     valid가 False가 되도록 표시 → C++ 쪽 ILC가 이 프레임 무시.
        self._vel_total_counter += 1
        vel_hit = False
        if flexion.valid:
            for attr_name, key in [
                ('left_knee_deg',  'left_knee'),
                ('right_knee_deg', 'right_knee'),
                ('left_hip_deg',   'left_hip'),
                ('right_hip_deg',  'right_hip'),
            ]:
                cur = getattr(flexion, attr_name)
                prev = self._vel_prev_flexion[key]
                if prev is not None and abs(cur - prev) > self._vel_max_delta_deg:
                    # 점프 감지 — 이전 값으로 rollback
                    setattr(flexion, attr_name, prev)
                    vel_hit = True
                else:
                    self._vel_prev_flexion[key] = cur
            if vel_hit:
                self._vel_hit_counter += 1
                # 이 프레임은 outlier → C++이 ILC에 반영 안 하도록 valid=False
                flexion.valid = False

        # ⑥ SHM 쓰기 (raw 각도 직접 전달 — KF 없음)
        self.pub.write_pose(FlexionAngles(
            left_knee_deg  = flexion.left_knee_deg,
            right_knee_deg = flexion.right_knee_deg,
            left_hip_deg   = flexion.left_hip_deg,
            right_hip_deg  = flexion.right_hip_deg,
            gait_phase     = 0.0,
            timestamp_us   = ts_us,
            valid          = flexion.valid,
            method         = method,
        ))
        _t4 = time.perf_counter()

        # ⑦ Plan D offline validation 용 dump (record-pose-npz)
        if getattr(self.args, 'record_pose_npz', None):
            if not hasattr(self, '_dump_buf'):
                self._dump_buf = {
                    't_s': [], 'hip_z_world_m': [], 'left_hip_z': [], 'right_hip_z': [],
                    'left_knee_rad': [], 'right_knee_rad': [],
                    'left_hip_rad': [], 'right_hip_rad': [],
                    'valid': [],
                }
            l_hip_z = float(raw_3d.get('left_hip', (0.0, 0.0, np.nan))[2]) if raw_3d else np.nan
            r_hip_z = float(raw_3d.get('right_hip', (0.0, 0.0, np.nan))[2]) if raw_3d else np.nan
            hip_z_mean = np.nanmean([l_hip_z, r_hip_z]) if not (np.isnan(l_hip_z) and np.isnan(r_hip_z)) else np.nan
            self._dump_buf['t_s'].append(ts_us * 1e-6)
            self._dump_buf['hip_z_world_m'].append(hip_z_mean)
            self._dump_buf['left_hip_z'].append(l_hip_z)
            self._dump_buf['right_hip_z'].append(r_hip_z)
            self._dump_buf['left_knee_rad'].append(np.deg2rad(flexion.left_knee_deg))
            self._dump_buf['right_knee_rad'].append(np.deg2rad(flexion.right_knee_deg))
            self._dump_buf['left_hip_rad'].append(np.deg2rad(flexion.left_hip_deg))
            self._dump_buf['right_hip_rad'].append(np.deg2rad(flexion.right_hip_deg))
            self._dump_buf['valid'].append(bool(flexion.valid))

        # release는 이미 get_rgb 직후에 호출됨 (new_api). 여기서는 아무것도 안 함.

        # ── 뼈 길이 ring buffer (최근 200f) 업데이트 ──
        if not hasattr(self, '_bone_buf'):
            self._bone_buf = {k: deque(maxlen=200) for k in
                              ('L_thigh', 'L_shank', 'R_thigh', 'R_shank')}
            self._e2e_buf = deque(maxlen=200)
        if raw_3d:
            if 'left_hip' in raw_3d and 'left_knee' in raw_3d:
                self._bone_buf['L_thigh'].append(
                    float(np.linalg.norm(np.array(raw_3d['left_knee']) - np.array(raw_3d['left_hip']))))
            if 'left_knee' in raw_3d and 'left_ankle' in raw_3d:
                self._bone_buf['L_shank'].append(
                    float(np.linalg.norm(np.array(raw_3d['left_ankle']) - np.array(raw_3d['left_knee']))))
            if 'right_hip' in raw_3d and 'right_knee' in raw_3d:
                self._bone_buf['R_thigh'].append(
                    float(np.linalg.norm(np.array(raw_3d['right_knee']) - np.array(raw_3d['right_hip']))))
            if 'right_knee' in raw_3d and 'right_ankle' in raw_3d:
                self._bone_buf['R_shank'].append(
                    float(np.linalg.norm(np.array(raw_3d['right_ankle']) - np.array(raw_3d['right_knee']))))
        # e2e latency: frame 캡처 시점(t0) → SHM write 시점
        frame_ms = (_t4 - t0) * 1000.0
        self._e2e_buf.append(frame_ms)
        # 개별 프레임 slow warning (20ms 초과 시 바로 출력, 진단용)
        if frame_ms > 20.0:
            print(f"\n[SLOW] frame {getattr(self, '_prof_count', 0)}: {frame_ms:.1f}ms  "
                  f"(fetch={(_t1-_t0)*1000:.1f} predict={(_t2-_t1)*1000:.1f} "
                  f"depth={(_t3-_t2)*1000:.1f} shm={(_t4-_t3)*1000:.1f})", flush=True)

        # ── 프로파일 (200f마다) + 3D + 뼈 통계 + e2e latency ──
        self._prof_count = getattr(self, '_prof_count', 0) + 1
        self._prof_sum = getattr(self, '_prof_sum', {'fetch':0,'predict':0,'depth_3d':0,'shm':0})
        self._prof_sum['fetch']    += _t1 - _t0
        self._prof_sum['predict']  += _t2 - _t1
        self._prof_sum['depth_3d'] += _t3 - _t2
        self._prof_sum['shm']      += _t4 - _t3
        if self._prof_count % 200 == 0:
            total = sum(self._prof_sum.values())
            print(f"\n[PROFILE] {self._prof_count}f avg ({total/200*1000:.1f}ms/frame):")
            for k, v in self._prof_sum.items():
                print(f"  {k:<10} {v/200*1000:.1f}ms ({v/total*100:.0f}%)")
            # 3D 좌표 샘플 (양쪽)
            if raw_3d:
                for side in ('left', 'right'):
                    parts = []
                    for j in ('hip', 'knee', 'ankle'):
                        key = f'{side}_{j}'
                        if key in raw_3d:
                            p = raw_3d[key]
                            parts.append(f"{j}=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f})")
                    if parts:
                        print(f"  [3D {side[0].upper()}] {' '.join(parts)} [m]")
            # 뼈 길이 통계: mean±std (min~max) [정상범위]
            for name in ('L_thigh', 'L_shank', 'R_thigh', 'R_shank'):
                buf = self._bone_buf[name]
                if len(buf) >= 10:
                    arr = np.array(buf)
                    print(f"  [bone {name}] {arr.mean():.3f}±{arr.std():.3f}m  "
                          f"({arr.min():.3f}~{arr.max():.3f})  N={len(buf)}")
            # 좌우 대칭성: thigh, shank 각각의 좌우 차이
            if (len(self._bone_buf['L_thigh']) >= 10 and
                    len(self._bone_buf['R_thigh']) >= 10):
                lt = np.array(self._bone_buf['L_thigh']).mean()
                rt = np.array(self._bone_buf['R_thigh']).mean()
                ls = np.array(self._bone_buf['L_shank']).mean()
                rs = np.array(self._bone_buf['R_shank']).mean()
                print(f"  [sym] thigh |L-R|={abs(lt-rt)*100:.1f}cm  "
                      f"shank |L-R|={abs(ls-rs)*100:.1f}cm")
            # e2e latency (capture → SHM)
            if self._e2e_buf:
                arr = np.array(self._e2e_buf)
                print(f"  [e2e lat] {arr.mean():.1f}±{arr.std():.1f}ms  "
                      f"(min={arr.min():.1f}  max={arr.max():.1f})")
            # Bone constraint hit rate (outlier 얼마나 교정됐는지)
            if self.bone_bc.ready:
                hits = self.bone_bc.hit_summary()
                if hits:
                    total_ratio = sum(h['ratio'] for h in hits.values()) / 4.0
                    lines = [f"{k.split('->')[0][:1]}-{k.split('->')[1][-5:]}"
                             f":{h['ratio']*100:.1f}%" for k, h in hits.items()]
                    print(f"  [bone-hit] avg {total_ratio*100:.1f}% | " + " ".join(lines))
            # Joint velocity bound hit rate (프레임간 점프 감지)
            if self._vel_total_counter > 0:
                vel_ratio = self._vel_hit_counter / self._vel_total_counter
                print(f"  [vel-bound] hit {self._vel_hit_counter}/{self._vel_total_counter} "
                      f"({vel_ratio*100:.2f}%)  max_delta={self._vel_max_delta_deg:.0f}°/f")
            self._prof_sum = {k:0 for k in self._prof_sum}

        # ⑦ 화면 표시 (선택적): Sagittal view (world frame — Method B면 중력 기준)
        if not self.args.no_display:
            self._display_sagittal(state, flexion)

    def _display_sagittal(self, state, flexion) -> None:
        """World frame sagittal view (Y-Z 측면) — 중력 기준 보행 자세 시각화.

        Method B: state.positions (camera frame) → R 행렬로 world 변환 후 그림
        Method A: 그대로 camera frame (카메라 기울기에 영향 받음)
        """
        try:
            import cv2
        except ImportError:
            return

        # ── FPS 계산 (표시용) ──
        now = time.perf_counter()
        fps = (1.0 / max(now - getattr(self, '_disp_last', now - 0.017), 1e-6))
        self._disp_fps = 0.9 * getattr(self, '_disp_fps', fps) + 0.1 * fps
        self._disp_last = now

        width, height = 480, 600
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)

        method = self.args.method.upper()
        frame_label = "World Frame (IMU)" if method == 'B' else "Camera Frame"

        cv2.putText(canvas, f'Sagittal - {frame_label}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, f'FPS: {self._disp_fps:.1f}', (width - 120, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        if not state.valid or not state.positions:
            cv2.putText(canvas, 'No Detection', (width // 2 - 70, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 1, cv2.LINE_AA)
            cv2.imshow('Sagittal', canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self._running = False
            return

        # ── Positions: Method B면 world frame으로 변환 ──
        positions = state.positions
        if method == 'B' and hasattr(self.calibrator, '_R') and self.calibrator._R is not None:
            R = self.calibrator._R
            positions = {name: (R @ pt).astype(np.float32)
                         for name, pt in state.positions.items()}

        zs = [p[2] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        if not zs:
            cv2.imshow('Sagittal', canvas)
            cv2.waitKey(1)
            return

        z_min, z_max = min(zs), max(zs)
        y_min, y_max = min(ys), max(ys)
        margin = 60
        z_range = max(z_max - z_min, 0.01)
        y_range = max(y_max - y_min, 0.01)
        scale = min((width - 2 * margin) / z_range,
                    (height - 2 * margin) / y_range) * 0.8
        z_center = (z_min + z_max) / 2
        y_center = (y_min + y_max) / 2

        def to_screen(y3d, z3d):
            sx = int(margin + (z3d - z_center) * scale + (width - 2 * margin) / 2)
            sy = int(margin + (y3d - y_center) * scale + (height - 2 * margin) / 2)
            return (sx, sy)

        # 좌우 다리 (왼쪽=파랑, 오른쪽=빨강)
        sides = [
            ('left',  (100, 200, 255), (50, 150, 255)),
            ('right', (255, 150, 100), (255, 100, 50)),
        ]
        for side, line_color, joint_color in sides:
            hip_k, knee_k, ankle_k = f'{side}_hip', f'{side}_knee', f'{side}_ankle'
            pts = {}
            for name in (hip_k, knee_k, ankle_k):
                if name in positions:
                    y3d, z3d = positions[name][1], positions[name][2]
                    pts[name] = to_screen(y3d, z3d)
            if hip_k in pts and knee_k in pts:
                cv2.line(canvas, pts[hip_k], pts[knee_k], line_color, 3)
            if knee_k in pts and ankle_k in pts:
                cv2.line(canvas, pts[knee_k], pts[ankle_k], line_color, 3)
            for name, pt in pts.items():
                cv2.circle(canvas, pt, 7, joint_color, -1)
                cv2.circle(canvas, pt, 7, (255, 255, 255), 1)
        # waist (hip 좌우 연결)
        if 'left_hip' in positions and 'right_hip' in positions:
            lh = to_screen(positions['left_hip'][1], positions['left_hip'][2])
            rh = to_screen(positions['right_hip'][1], positions['right_hip'][2])
            cv2.line(canvas, lh, rh, (150, 150, 150), 2)

        # 중력 방향 지시 (world frame에서만 의미 있음)
        if method == 'B':
            arrow_x = width - 40
            cv2.arrowedLine(canvas, (arrow_x, 50), (arrow_x, 90),
                            (200, 200, 200), 2, tipLength=0.3)
            cv2.putText(canvas, 'g', (arrow_x + 5, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        # ── 각도 패널 ──
        info_y = height - 180
        angle_data = [
            ('L Knee flex',  flexion.left_knee_deg  if flexion.valid else None),
            ('R Knee flex',  flexion.right_knee_deg if flexion.valid else None),
            ('L Hip flex',   flexion.left_hip_deg   if flexion.valid else None),
            ('R Hip flex',   flexion.right_hip_deg  if flexion.valid else None),
        ]
        for label, val in angle_data:
            if val is not None:
                color = (100, 255, 100) if 'L ' in label else (100, 200, 255)
                cv2.putText(canvas, f'{label}: {val:+.1f} deg', (15, info_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                info_y += 20

        # 뼈 길이 (정적 참조용)
        info_y += 8
        for bone, length in sorted(state.bone_lengths.items()):
            cv2.putText(canvas, f'{bone}: {length:.3f}m', (15, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
            info_y += 16

        # 축 라벨
        cv2.putText(canvas, 'Z (depth) ->', (width - 130, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(canvas, 'Y (down)', (5, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)

        cv2.imshow('Sagittal', canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self._running = False

    def _fps_tick(self) -> None:
        """1초마다 FPS 출력."""
        self._fps_count += 1
        now = time.perf_counter()
        elapsed = now - self._fps_t0
        if elapsed >= 1.0:
            fps = self._fps_count / elapsed
            print(f"[FPS] {fps:.1f} Hz", flush=True)
            self._fps_count = 0
            self._fps_t0    = now

    def run(self) -> None:
        """50Hz 메인 루프."""
        self._running = True
        self._fps_t0  = time.perf_counter()
        self._gc_counter = 0

        print("[Pipeline] 루프 시작 (Ctrl+C 로 종료)\n")

        while self._running:
            t0      = time.perf_counter()
            loop_start = t0

            try:
                self._process_frame(t0)
            except StopIteration:
                print("\n[Pipeline] SVO2 파일 끝.")
                break
            except Exception as exc:
                print(f"\n[Pipeline][ERR] 프레임 처리 오류: {exc}", file=sys.stderr)

            self._fps_tick()
            # 주기적 gen-0 GC — 자동 GC off 상태에서 cyclic ref 방지용.
            # gen-0만: ~0.3ms 예측 가능. 500f마다 (7초마다) 한 번.
            self._gc_counter += 1
            if self._gc_counter >= 500:
                gc.collect(generation=0)
                self._gc_counter = 0
            # sleep 없음 — 최대 속도로 SHM에 쓰기. C++가 100Hz 타이밍 담당.

    def shutdown(self) -> None:
        """Graceful shutdown."""
        self._running = False
        print("\n[Pipeline] 종료 중...")

        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass

        if self.camera is not None:
            try:
                self.camera.close()
            except Exception:
                pass

        self.pub.close()
        self.pub.unlink()

        # Plan D offline validation 용 npz dump (record-pose-npz 활성화 시)
        if getattr(self.args, 'record_pose_npz', None) and hasattr(self, '_dump_buf'):
            out_path = self.args.record_pose_npz
            try:
                np.savez(
                    out_path,
                    t_s=np.array(self._dump_buf['t_s'], dtype=np.float64),
                    hip_z_world_m=np.array(self._dump_buf['hip_z_world_m'], dtype=np.float64),
                    left_hip_z=np.array(self._dump_buf['left_hip_z'], dtype=np.float64),
                    right_hip_z=np.array(self._dump_buf['right_hip_z'], dtype=np.float64),
                    left_knee_rad=np.array(self._dump_buf['left_knee_rad'], dtype=np.float64),
                    right_knee_rad=np.array(self._dump_buf['right_knee_rad'], dtype=np.float64),
                    left_hip_rad=np.array(self._dump_buf['left_hip_rad'], dtype=np.float64),
                    right_hip_rad=np.array(self._dump_buf['right_hip_rad'], dtype=np.float64),
                    valid=np.array(self._dump_buf['valid'], dtype=bool),
                    method=str(self.args.method),
                )
                n = len(self._dump_buf['t_s'])
                print(f"[Pipeline] Pose dump saved: {out_path} ({n} frames)")
            except Exception as exc:
                print(f"[Pipeline][WARN] Pose dump failed: {exc}", file=sys.stderr)

        print("[Pipeline] SHM 해제 완료. 종료.")


# ─── CLI 진입점 ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="H-Walker 보행 재활 로봇 실시간 포즈 파이프라인 (50Hz)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--method", choices=["A", "B"], default="A",
        help="캘리브레이션 방법\n  A: Standing Calibration (기본)\n  B: ZED IMU World Frame",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"YOLO 모델 경로 (기본: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--svo2", default=None, metavar="FILE",
        help="SVO2 파일 경로 (없으면 실시간 카메라)",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="OpenCV 화면 표시 비활성화",
    )
    parser.add_argument(
        "--no-trt", action="store_true",
        help="TensorRT 비활성화 (PyTorch 직접 실행)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="모델 입력 해상도 (기본 640 — CLAUDE.md 명시: 480 영구 거부)",
    )
    parser.add_argument(
        "--depth-mode", dest="depth_mode", default="PERFORMANCE",
        choices=["PERFORMANCE", "NEURAL_LIGHT", "NEURAL", "NEURAL_PLUS", "ULTRA", "QUALITY"],
        help="ZED depth mode (기본 PERFORMANCE — 74Hz 달성. NEURAL 계열은 정확도 이득 미미 + 속도 큰 희생)",
    )
    parser.add_argument(
        "--serialize-depth", dest="serialize_depth", action="store_true",
        default=False,
        help="depth retrieve를 predict 뒤로 미룸 (실험 기능, 기본 OFF).\n"
             "  ON:  GPU 순차 실행 → spike 제거되지만 grab 파이프라인 붕괴로 39Hz.\n"
             "  OFF: 기존 parallel 모드 (73Hz, 간헐 spike 있음, 대부분 background 프로세스 원인).\n"
             "  → 근본 해결은 CUDA_Stream 폴더의 stream 분리 실험 (별도 트랙).",
    )
    parser.add_argument(
        "--bone-std", dest="bone_std", type=float, default=0.010,
        help="Bone constraint 캘리브 std 허용 한계 [m] (기본 0.010 = 10mm). "
             "노이즈 많은 환경에서 ref 채택 안 되면 0.015~0.020으로 완화.",
    )
    parser.add_argument(
        "--bone-tol", dest="bone_tol", type=float, default=0.20,
        help="Bone constraint 동작 중 ref 대비 허용 편차 비율 (기본 0.20 = ±20%).",
    )
    parser.add_argument(
        "--record-pose-npz", dest="record_pose_npz", default=None, metavar="FILE",
        help="Plan D offline validation 용 pose 데이터 dump.\n"
             "  per-frame: (t_s, hip_z_world_m, joint_angles_rad, valid_mask)\n"
             "  shutdown 시 npz 저장. Plan D 의 cold-start + L1 검증 input.",
    )
    return parser.parse_args()


def main() -> None:
    args     = parse_args()
    pipeline = Pipeline(args)

    # Ctrl+C → graceful shutdown
    def _sigint_handler(sig, frame):
        pipeline.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    try:
        pipeline.setup()
        pipeline.run()
    except Exception as exc:
        print(f"\n[Pipeline][FATAL] {exc}", file=sys.stderr)
        raise
    finally:
        pipeline.shutdown()


if __name__ == "__main__":
    main()

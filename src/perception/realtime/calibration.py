"""
calibration.py
==============
관절 각도 기준점 설정 방법 비교: Method A vs Method B

Method A — Standing Calibration (상대 기준)
    피험자가 똑바로 서 있을 때 N프레임을 평균 → neutral 각도 정의
    보행 중 각도 = current - neutral → flexion

Method B — ZED IMU World Frame (절대 기준)
    ZED X Mini 내장 IMU → 카메라 기울기(quaternion) → 중력 정렬 rotation
    관절 3D 좌표를 world frame으로 변환 → 절대 각도

속도 측정:
    run_latency_benchmark() 로 A/B 각각의 추가 지연(ms) 비교
"""

import sys
import os
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict

_bench_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'benchmarks')
if _bench_dir not in sys.path:
    sys.path.insert(0, _bench_dir)

from joint_3d import JointState3D, compute_joint_state, _angle_3d
from shm_publisher import FlexionAngles


# ─────────────────────────────────────────────────────────────────────────────
# Method A — Standing Calibration
# ─────────────────────────────────────────────────────────────────────────────

class StandingCalibration:
    """
    피험자가 똑바로 서 있는 동안 N프레임 평균 → neutral 각도 저장.
    이후 보행 중 raw 3D 각도에서 neutral을 빼면 flexion 각도가 됨.

    사용법:
        cal = StandingCalibration(n_frames=30)
        while not cal.done:
            cal.update(joint_state)           # 서 있는 동안 호출
        # 이후 보행 중:
        flexion = cal.to_flexion(joint_state)
    """

    def __init__(self, n_frames: int = 30):
        self.n_frames = n_frames
        # knee_flexion만 사용 (hip은 trunk keypoint 없어 측정 불가)
        self._samples: Dict[str, list] = {
            'left_knee': [], 'right_knee': [],
        }
        self._neutral: Optional[Dict[str, float]] = None

    @property
    def done(self) -> bool:
        return self._neutral is not None

    @property
    def progress(self) -> float:
        """0.0 ~ 1.0"""
        if self.done:
            return 1.0
        n = min(len(v) for v in self._samples.values())
        return n / self.n_frames

    def update(self, state: JointState3D):
        """서 있는 동안 매 프레임 호출. done이 되면 자동으로 finalize."""
        if self.done or not state.valid:
            return

        # knee_angle property (backward compat): 180 - knee_flexion
        for key, val in [
            ('left_knee',  state.left_knee_angle),
            ('right_knee', state.right_knee_angle),
        ]:
            if val is not None:
                self._samples[key].append(val)

        # 모든 각도에서 n_frames 이상 모이면 확정
        if all(len(v) >= self.n_frames for v in self._samples.values()):
            self._finalize()

    def _finalize(self):
        self._neutral = {k: float(np.mean(v[-self.n_frames:]))
                        for k, v in self._samples.items()}
        print('[CalibA] neutral 각도 확정:')
        for k, v in self._neutral.items():
            print(f'  {k:<14} {v:.2f} deg')

    def to_flexion(self, state: JointState3D) -> FlexionAngles:
        """
        보행 중 호출 → FlexionAngles 반환.
        neutral보다 각도가 줄어들수록(굴곡) flexion이 증가.
        """
        t0 = time.perf_counter()

        out = FlexionAngles(method='A', timestamp_us=state.timestamp_us)
        if not self.done or not state.valid:
            return out

        n = self._neutral
        # 3D 각도는 신전 = 180°, 굴곡 = 감소
        # flexion = neutral - current  → 굴곡할수록 양수
        # knee_flexion = neutral(≈178°) - current(180-flexion) ≈ flexion - 2°
        # 결과: 0°=완전신전, 양수=굴곡 (Winter convention 근사)
        if state.left_knee_angle is not None:
            out.left_knee_deg  = n['left_knee']  - state.left_knee_angle
        if state.right_knee_angle is not None:
            out.right_knee_deg = n['right_knee'] - state.right_knee_angle
        # hip: trunk keypoint 없으므로 0으로 설정
        out.left_hip_deg  = 0.0
        out.right_hip_deg = 0.0

        out.valid = True
        return out

    def save(self, path: str):
        """캘리브레이션 결과 저장 (피험자별 재사용)."""
        import json
        with open(path, 'w') as f:
            json.dump(self._neutral, f, indent=2)
        print(f'[CalibA] 저장: {path}')

    def load(self, path: str):
        """저장된 캘리브레이션 불러오기."""
        import json
        with open(path) as f:
            self._neutral = json.load(f)
        print(f'[CalibA] 로드: {path}')
        for k, v in self._neutral.items():
            print(f'  {k:<14} {v:.2f} deg')


# ─────────────────────────────────────────────────────────────────────────────
# Method B — ZED IMU World Frame
# ─────────────────────────────────────────────────────────────────────────────

class ZEDIMUWorldFrame:
    """
    ZED X Mini 내장 IMU → 카메라 orientation(quaternion) 읽기
    → rotation matrix R_cam_to_world 계산
    → 관절 3D 좌표를 world frame (gravity-aligned)으로 변환
    → world frame에서 관절 각도 재계산

    World frame 정의:
        Y_world = +위 (anti-gravity)
        Z_world = +앞 (보행 전진 방향, ZED front)
        X_world = +왼쪽

    사용법:
        wf = ZEDIMUWorldFrame(zed_camera_obj)
        wf.init()                              # IMU warm-up
        flexion = wf.to_flexion(joint_state)
    """

    def __init__(self, zed_camera):
        """
        Args:
            zed_camera: ZEDCamera 인스턴스 (열려 있어야 함)
        """
        self._cam = zed_camera
        self._R   = None          # 3×3 rotation matrix (cam → world)
        self._world_up_camera = None  # world up axis expressed in camera frame
        self._imu_ok = False
        self._neutral: Optional[Dict[str, float]] = None

    def _has_pyzed(self) -> bool:
        try:
            import pyzed.sl as sl
            return True
        except ImportError:
            return False

    def init(self, warm_up_frames: int = 20):
        """
        IMU 안정화를 위해 warm_up_frames 동안 orientation 평균.
        ZED grab loop 안에서 호출.
        """
        if not self._has_pyzed():
            print('[CalibB] pyzed 없음 → Method B 사용 불가')
            return False

        import pyzed.sl as sl

        quats = []
        print(f'[CalibB] IMU warm-up {warm_up_frames}프레임...')
        for _ in range(warm_up_frames):
            self._cam.grab()
            sd = sl.SensorsData()
            self._cam.zed.get_sensors_data(sd, sl.TIME_REFERENCE.IMAGE)
            imu = sd.get_imu_data()
            o   = imu.get_pose().get_orientation().get()   # [ox, oy, oz, ow]
            quats.append(o)

        # 평균 quaternion (간단 평균, 오차 작음)
        q_mean = np.mean(quats, axis=0)
        q_mean /= np.linalg.norm(q_mean)
        self._R = self._quat_to_R(q_mean)
        self._world_up_camera = self._compute_world_up_camera(q_mean)
        self._imu_ok = True
        print('[CalibB] IMU rotation matrix 확정:')
        print(self._R.round(4))
        print(f'[CalibB] world up in camera frame: {self._world_up_camera.round(4)}')
        return True

    def refresh_R(self):
        """매 프레임 R을 갱신하고 싶을 때 (카메라 이동 시)."""
        if not self._imu_ok:
            return
        import pyzed.sl as sl
        sd = sl.SensorsData()
        self._cam.zed.get_sensors_data(sd, sl.TIME_REFERENCE.IMAGE)
        imu = sd.get_imu_data()
        q = imu.get_pose().get_orientation().get()
        q /= np.linalg.norm(q)
        self._R = self._quat_to_R(q)
        self._world_up_camera = self._compute_world_up_camera(q)

    def world_up_in_camera(self) -> Optional[np.ndarray]:
        """World up axis (gravity-opposite) expressed in camera frame.

        Returns None until init() succeeds. Used by compute_joint_state for
        thigh/shank inclination calculations (joint_3d.py:232).

        Cached during init() and refresh_R(); production runtime uses static
        R (skip_imu=True) so this is set once at startup.
        """
        if not self._imu_ok or self._world_up_camera is None:
            return None
        return self._world_up_camera

    @staticmethod
    def _compute_world_up_camera(q) -> np.ndarray:
        """ZED IMU quaternion [x, y, z, w] → world up vector in camera frame.

        Same arithmetic as ``ZEDCamera.get_gravity_vector`` in
        ``src/perception/benchmarks/zed_camera.py:553-572``. Reusing the exact
        formula keeps quaternion convention compatibility — ZED SDK's q plus
        this expression is the verified producer of a sane up direction in
        this codebase.

        Equivalent to ``_quat_to_R(q)[:, 1]`` (world Y axis in camera frame),
        kept separate so unit tests can exercise the formula without building
        a full R matrix.
        """
        x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        return np.array([
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x),
        ], dtype=np.float32)

    @staticmethod
    def _quat_to_R(q) -> np.ndarray:
        """
        quaternion [x, y, z, w] → 3×3 rotation matrix.
        ZED SDK 출력 포맷: [ox, oy, oz, ow]
        """
        x, y, z, w = q
        R = np.array([
            [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
            [    2*(x*y + z*w), 1 - 2*(x*x + z*z),   2*(y*z - x*w)],
            [    2*(x*z - y*w),   2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ], dtype=np.float32)
        return R

    def _transform_positions(self, positions: dict) -> dict:
        """camera frame 3D 좌표 → world frame."""
        return {name: (self._R @ pt).astype(np.float32)
                for name, pt in positions.items()}

    def set_standing_neutral(self, state: JointState3D):
        """
        B안에서도 neutral 기준점 설정 (선택 사항).
        world frame 각도를 neutral-referenced로 만들 때 사용.
        """
        pos_w = self._transform_positions(state.positions)
        angles = self._compute_angles_world(pos_w)
        self._neutral = angles
        print('[CalibB] world frame neutral 확정:')
        for k, v in angles.items():
            print(f'  {k:<14} {v:.2f} deg')

    @staticmethod
    def _compute_angles_world(pos_w: dict) -> dict:
        """world frame 3D 좌표에서 관절 각도 계산."""
        angles = {}
        p = pos_w

        if all(k in p for k in ('left_hip', 'left_knee', 'left_ankle')):
            angles['left_knee'] = _angle_3d(p['left_hip'], p['left_knee'], p['left_ankle'])
        if all(k in p for k in ('right_hip', 'right_knee', 'right_ankle')):
            angles['right_knee'] = _angle_3d(p['right_hip'], p['right_knee'], p['right_ankle'])
        if all(k in p for k in ('left_knee', 'left_hip', 'right_hip')):
            angles['left_hip'] = _angle_3d(p['left_knee'], p['left_hip'], p['right_hip'])
        if all(k in p for k in ('right_knee', 'right_hip', 'left_hip')):
            angles['right_hip'] = _angle_3d(p['right_knee'], p['right_hip'], p['left_hip'])

        return angles

    def to_flexion(self, state: JointState3D,
                   refresh_imu: bool = False) -> FlexionAngles:
        """
        보행 중 호출 → world frame FlexionAngles 반환.

        Args:
            state: joint_3d.compute_joint_state() 출력
            refresh_imu: True면 이번 프레임 IMU도 갱신 (카메라 이동 시)
        """
        t0 = time.perf_counter()

        out = FlexionAngles(method='B', timestamp_us=state.timestamp_us)
        if not self._imu_ok or not state.valid:
            return out

        if refresh_imu:
            self.refresh_R()  # ~0.05ms 추가

        # ① camera → world 변환
        pos_w = self._transform_positions(state.positions)

        # ② world frame에서 각도 계산
        angles_w = self._compute_angles_world(pos_w)

        # ③ neutral이 있으면 flexion으로 변환, 없으면 raw 반환
        if self._neutral:
            n = self._neutral
            lk = n.get('left_knee',  0) - angles_w.get('left_knee',  0)
            rk = n.get('right_knee', 0) - angles_w.get('right_knee', 0)
            lh = n.get('left_hip',   0) - angles_w.get('left_hip',   0)
            rh = n.get('right_hip',  0) - angles_w.get('right_hip',  0)
            out.left_knee_deg  = lk if lk != 0 else 0.0
            out.right_knee_deg = rk if rk != 0 else 0.0
            out.left_hip_deg   = lh if lh != 0 else 0.0
            out.right_hip_deg  = rh if rh != 0 else 0.0
        else:
            out.left_knee_deg  = angles_w.get('left_knee',  0.0)
            out.right_knee_deg = angles_w.get('right_knee', 0.0)
            out.left_hip_deg   = angles_w.get('left_hip',   0.0)
            out.right_hip_deg  = angles_w.get('right_hip',  0.0)

        out.valid = True
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 속도 비교 벤치마크
# ─────────────────────────────────────────────────────────────────────────────

def run_latency_benchmark(zed_camera=None, n_iter: int = 1000):
    """
    Method A vs B 속도 비교.

    Args:
        zed_camera: ZEDCamera 인스턴스 (B안 측정 시 필요. None이면 A만 측정)
        n_iter: 반복 횟수
    """
    import random

    # 더미 JointState3D 생성 (실제 값과 유사)
    dummy_positions = {
        'left_hip':    np.array([-0.10,  0.05,  0.30], np.float32),
        'right_hip':   np.array([ 0.10,  0.05,  0.30], np.float32),
        'left_knee':   np.array([-0.12, -0.02,  0.70], np.float32),
        'right_knee':  np.array([ 0.12, -0.02,  0.68], np.float32),
        'left_ankle':  np.array([-0.11, -0.05,  1.00], np.float32),
        'right_ankle': np.array([ 0.11, -0.05,  0.98], np.float32),
    }
    dummy_state = compute_joint_state(
        keypoints_2d  = {k: (0.0, 0.0) for k in dummy_positions},
        keypoints_3d  = dummy_positions,
        confidences   = {k: 1.0 for k in dummy_positions},
    )

    results = {}

    # ── Method A ──
    cal_a = StandingCalibration(n_frames=1)
    cal_a.update(dummy_state)   # 즉시 neutral 설정

    times_a = []
    for _ in range(n_iter):
        t = time.perf_counter()
        _ = cal_a.to_flexion(dummy_state)
        times_a.append((time.perf_counter() - t) * 1e6)

    results['A'] = np.array(times_a)

    # ── Method B (ZED IMU 없이 rotation matrix만 측정) ──
    wf = ZEDIMUWorldFrame(None)
    wf._R = np.eye(3, dtype=np.float32)  # 더미 R
    wf._imu_ok = True

    times_b_noIMU = []
    for _ in range(n_iter):
        t = time.perf_counter()
        _ = wf.to_flexion(dummy_state, refresh_imu=False)
        times_b_noIMU.append((time.perf_counter() - t) * 1e6)

    results['B_no_IMU_refresh'] = np.array(times_b_noIMU)

    # ── Method B (ZED IMU 매 프레임 갱신 시뮬레이션) ──
    # 실제 ZED SDK 없이 quaternion 계산만 측정
    times_b_quat = []
    for _ in range(n_iter):
        t = time.perf_counter()
        q = np.array([0.0, 0.0, 0.0, 1.0], np.float32)
        q += np.random.randn(4).astype(np.float32) * 0.001
        q /= np.linalg.norm(q)
        wf._R = ZEDIMUWorldFrame._quat_to_R(q)
        _ = wf.to_flexion(dummy_state, refresh_imu=False)
        times_b_quat.append((time.perf_counter() - t) * 1e6)

    results['B_with_quat'] = np.array(times_b_quat)

    # ── 결과 출력 ──
    pipeline_total_us = 30_000  # YOLO+ZED 합산 ~30ms

    print('\n' + '='*60)
    print(f'  Calibration Latency Benchmark  (n={n_iter})')
    print('='*60)
    print(f'  {"Method":<28} {"mean":>7} {"p95":>7} {"p99":>7}  [μs]')
    print('-'*60)

    for label, arr in results.items():
        print(f'  {label:<28} {np.mean(arr):7.2f} '
              f'{np.percentile(arr, 95):7.2f} {np.percentile(arr, 99):7.2f}')

    print('-'*60)
    a_mean  = np.mean(results['A'])
    b_mean  = np.mean(results['B_with_quat'])
    overhead = b_mean - a_mean

    print(f'\n  B - A 추가 오버헤드:  {overhead:.2f} μs')
    print(f'  전체 파이프라인 대비: {overhead / pipeline_total_us * 100:.3f} %')
    print(f'  (파이프라인 전체 ~{pipeline_total_us/1000:.0f}ms 기준)')

    print('\n  [결론]')
    if overhead < 100:
        print(f'  → {overhead:.1f}μs 차이 = 무시 가능 (파이프라인의 {overhead/pipeline_total_us*100:.3f}%)')
        print('  → 속도 관점에서 A/B 동등. B 사용 권장.')
    else:
        print(f'  → {overhead:.1f}μs 차이 = 확인 필요')
    print('='*60)

    return results


if __name__ == '__main__':
    run_latency_benchmark()

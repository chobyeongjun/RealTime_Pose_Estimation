"""
ZED X Mini 카메라 유틸리티
- RGB 이미지 + Depth Map 캡처
- Body Tracking 없이 경량 모드로 동작
- 벤치마크용 프레임 제공
- AsyncCamera: 비동기 캡처로 GPU 유휴 시간 제거
- GPU 전처리: cv2.cuda를 이용한 리사이즈/색공간 변환
"""

import math
import numpy as np
import os
import time
import threading
from collections import deque

try:
    import pyzed.sl as sl
    HAS_ZED = True
except ImportError:
    HAS_ZED = False
    print("[WARNING] pyzed 미설치 - 웹캠 폴백 모드로 동작")

import cv2

# GPU 전처리 가용성 확인
HAS_CUDA_CV = False
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        HAS_CUDA_CV = True
except (AttributeError, cv2.error):
    pass


# ============================================================================
# Joint 3D 위치 정확도 + 속도 개선 유틸리티
# ============================================================================

def _sample_depth_patch(depth_map, x, y, patch_radius=3):
    """키포인트 주변 패치에서 중앙값 depth 추출 (단일 픽셀 노이즈 제거)

    단일 픽셀 대신 (2*r+1)×(2*r+1) 영역의 유효 depth 중앙값을 사용.
    에러값(NaN, inf, 0)이 있어도 중앙값이므로 무시됨.

    Args:
        depth_map: ZED depth map (float32, meters)
        x, y: 키포인트 픽셀 좌표
        patch_radius: 패치 반경 (기본 3 → 7×7 = 49픽셀)

    Returns:
        float: 중앙값 depth (meters), 유효값 없으면 NaN
    """
    h, w = depth_map.shape[:2]
    x, y = int(x), int(y)

    y0 = max(0, y - patch_radius)
    y1 = min(h, y + patch_radius + 1)
    x0 = max(0, x - patch_radius)
    x1 = min(w, x + patch_radius + 1)

    patch = depth_map[y0:y1, x0:x1].ravel()
    valid = patch[np.isfinite(patch) & (patch > 0)]

    if len(valid) == 0:
        return float('nan')
    return float(np.median(valid))


class _OneEuroFilter1D:
    """1D One Euro Filter (3D 좌표 필터링용 경량 버전)"""

    __slots__ = ('min_cutoff', 'beta', 'd_cutoff', 'x_prev', 'dx_prev', 't_prev')

    def __init__(self, t0, x0, min_cutoff=0.5, beta=0.01, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0
        self.dx_prev = 0.0
        self.t_prev = t0

    def __call__(self, t, x):
        te = t - self.t_prev
        if te <= 0:
            return x
        a_d = self._sf(te, self.d_cutoff)
        dx = (x - self.x_prev) / te
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._sf(te, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    @staticmethod
    def _sf(te, cutoff):
        r = 2 * math.pi * cutoff * te
        return r / (r + 1)


class Joint3DFilter:
    """Joint별 3D 위치 정확도 보장 시스템

    처리 순서 (매 프레임):
      1) 패치 중앙값 depth → pixel_to_3d에서 이미 적용
      2) One Euro Filter (X, Y, Z 독립) → depth 떨림 제거
      3) depth 무효 시 이전 프레임 + 속도 기반 보간
      4) 3D Segment Length Constraint → 정자세 캘리브레이션 기반 뼈 길이 보정

    정자세 캘리브레이션:
      - 처음 calib_frames 프레임 동안 서 있는 자세에서 3D 뼈 길이 측정
      - 이 길이는 변하지 않으므로, 이후 보행 중 키포인트가 이상하면 보정
      - 예: 무릎을 ankle로 잘못 인식 → thigh 길이가 캘리브 대비 50% 짧음 → 보정
    """

    # 3D 뼈 세그먼트 정의: (부모, 자식) — 자식을 보정
    SEGMENTS_3D = [
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ]

    def __init__(self, min_cutoff=0.5, beta=0.01, d_cutoff=1.0, max_missing=5,
                 calib_frames=30, tolerance=0.20):
        """
        Args:
            min_cutoff: One Euro Filter 기본 cutoff (낮을수록 강한 필터링)
            beta: 속도 적응 계수 (높을수록 빠른 동작에 민감)
            max_missing: 연속 depth 무효 허용 프레임 수
            calib_frames: 정자세 캘리브레이션 프레임 수
            tolerance: 뼈 길이 허용 오차 (0.20 = ±20%)
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.max_missing = max_missing
        self._filters = {}    # name → (_1D_x, _1D_y, _1D_z)
        self._prev_3d = {}    # name → (X, Y, Z)
        self._prev_vel = {}   # name → (vX, vY, vZ)
        self._missing = {}    # name → 연속 무효 카운트
        self._prev_t = {}     # name → 이전 타임스탬프

        # 3D Segment Length Constraint (정자세 캘리브레이션)
        self.calib_frames = calib_frames
        self.tolerance = tolerance
        self._calib_count = 0
        self._calib_done = False
        self._seg_samples = {}  # (parent, child) → [3D 길이 리스트]
        self._seg_ref = {}      # (parent, child) → 기준 3D 길이 (meters)

    @property
    def calibrated(self):
        return self._calib_done

    @property
    def calib_progress(self):
        if self._calib_done:
            return 1.0
        return min(1.0, self._calib_count / self.calib_frames)

    def filter(self, name, pt3d, t):
        """3D 좌표를 필터링하여 반환

        Args:
            name: 키포인트 이름 (예: "left_ankle")
            pt3d: (X, Y, Z) 또는 None (depth 무효)
            t: 현재 타임스탬프 (초, time.perf_counter)

        Returns:
            (X, Y, Z) numpy array 또는 None
        """
        if pt3d is not None:
            x, y, z = float(pt3d[0]), float(pt3d[1]), float(pt3d[2])
            self._missing[name] = 0

            if name not in self._filters:
                self._filters[name] = (
                    _OneEuroFilter1D(t, x, self.min_cutoff, self.beta, self.d_cutoff),
                    _OneEuroFilter1D(t, y, self.min_cutoff, self.beta, self.d_cutoff),
                    _OneEuroFilter1D(t, z, self.min_cutoff, self.beta, self.d_cutoff),
                )
                self._prev_3d[name] = (x, y, z)
                self._prev_t[name] = t
                return np.array([x, y, z], dtype=np.float32)

            fx, fy, fz = self._filters[name]
            xf, yf, zf = fx(t, x), fy(t, y), fz(t, z)

            # 속도 기록 (보간용)
            prev = self._prev_3d[name]
            dt = t - self._prev_t.get(name, t)
            if dt > 0:
                self._prev_vel[name] = (
                    (xf - prev[0]) / dt,
                    (yf - prev[1]) / dt,
                    (zf - prev[2]) / dt,
                )

            self._prev_3d[name] = (xf, yf, zf)
            self._prev_t[name] = t
            return np.array([xf, yf, zf], dtype=np.float32)

        # pt3d is None — depth 무효: 이전 프레임 기반 보간
        miss = self._missing.get(name, 0) + 1
        self._missing[name] = miss

        if miss > self.max_missing or name not in self._prev_3d:
            return None  # 너무 오래 무효 → 신뢰 불가

        prev = self._prev_3d[name]
        vel = self._prev_vel.get(name, (0, 0, 0))
        dt = t - self._prev_t.get(name, t)

        # 속도 기반 예측 (등속 가정)
        predicted = (
            prev[0] + vel[0] * dt,
            prev[1] + vel[1] * dt,
            prev[2] + vel[2] * dt,
        )
        self._prev_3d[name] = predicted
        self._prev_t[name] = t
        return np.array(predicted, dtype=np.float32)

    def apply_segment_constraint(self, keypoints_3d):
        """3D 뼈 길이 제약 적용 (filter 호출 후, 전체 keypoints에 대해 1회 호출)

        정자세 캘리브레이션 중: 뼈 길이 샘플 수집
        캘리브레이션 완료 후: 뼈 길이가 기준 대비 ±tolerance 벗어나면 자식 joint 보정

        Args:
            keypoints_3d: dict {name → (X, Y, Z)} — in-place 수정됨

        Returns:
            bool: 제약이 적용되었으면 True
        """
        if not self._calib_done:
            return self._collect_3d_sample(keypoints_3d)
        return self._enforce_3d_constraint(keypoints_3d)

    def _collect_3d_sample(self, keypoints_3d):
        """정자세 캘리브레이션: 3D 뼈 길이 샘플 수집"""
        valid_count = 0
        for parent, child in self.SEGMENTS_3D:
            if parent in keypoints_3d and child in keypoints_3d:
                p = np.array(keypoints_3d[parent])
                c = np.array(keypoints_3d[child])
                length = float(np.linalg.norm(c - p))
                if length > 0.05:  # 최소 5cm (노이즈 필터)
                    key = (parent, child)
                    if key not in self._seg_samples:
                        self._seg_samples[key] = []
                    self._seg_samples[key].append(length)
                    valid_count += 1

        if valid_count >= 3:
            self._calib_count += 1

        if self._calib_count >= self.calib_frames:
            self._finalize_3d_calibration()

        return False

    def _finalize_3d_calibration(self):
        """캘리브레이션 완료: 중앙값으로 기준 뼈 길이 확정"""
        for key, lengths in self._seg_samples.items():
            if len(lengths) >= 5:
                sorted_l = sorted(lengths)
                self._seg_ref[key] = sorted_l[len(sorted_l) // 2]  # 중앙값

        if len(self._seg_ref) >= 3:
            self._calib_done = True
            print("[Joint3DFilter] 3D 뼈 길이 캘리브레이션 완료 "
                  f"({self._calib_count} 프레임, 정자세)")
            for (p, c), length in self._seg_ref.items():
                print(f"  {p}→{c}: {length:.3f}m")
        else:
            self._calib_count = max(0, self._calib_count - 10)

        self._seg_samples.clear()

    def _enforce_3d_constraint(self, keypoints_3d):
        """3D 뼈 길이 제약 적용 — 자식 joint를 부모→자식 방향으로 보정

        무릎을 ankle로 잘못 인식 등의 경우:
          thigh 길이가 캘리브 대비 크게 벗어남 → 자식(knee) 위치를 보정
        """
        applied = False
        for parent, child in self.SEGMENTS_3D:
            key = (parent, child)
            if key not in self._seg_ref:
                continue
            if parent not in keypoints_3d or child not in keypoints_3d:
                continue

            p = np.array(keypoints_3d[parent], dtype=np.float64)
            c = np.array(keypoints_3d[child], dtype=np.float64)
            cur_len = float(np.linalg.norm(c - p))

            if cur_len < 1e-6:
                continue

            ref_len = self._seg_ref[key]
            min_len = ref_len * (1 - self.tolerance)
            max_len = ref_len * (1 + self.tolerance)

            if min_len <= cur_len <= max_len:
                continue  # 범위 내 — OK

            # 자식 joint를 parent→child 방향으로 기준 길이에 맞게 투영
            target_len = max(min_len, min(max_len, cur_len))
            direction = (c - p) / cur_len  # 단위 벡터
            new_c = p + direction * target_len
            keypoints_3d[child] = tuple(new_c.astype(np.float32))

            # 필터 내부 상태도 업데이트 (다음 프레임 보간용)
            self._prev_3d[child] = tuple(new_c)
            applied = True

        return applied

    def get_ref_lengths_3d(self):
        """캘리브레이션된 3D 기준 뼈 길이 반환"""
        return {f"{p}→{c}": length
                for (p, c), length in self._seg_ref.items()}

    def reset(self):
        self._filters.clear()
        self._prev_3d.clear()
        self._prev_vel.clear()
        self._missing.clear()
        self._prev_t.clear()


class ZEDCamera:
    """ZED X Mini RGB + Depth 캡처 (Body Tracking OFF)

    Global Shutter 활용 전략:
      ZED X Mini는 Global Shutter 센서를 탑재하여 Rolling Shutter 대비:
      - 빠른 보행 동작에서 motion blur / rolling shutter 왜곡 없음
      - 모든 픽셀이 동시에 촬영되므로 keypoint 위치가 정확
      - 120fps까지 지원하여 고속 동작 캡처 가능
      → 별도 코드 설정 불필요 (하드웨어 레벨에서 자동 적용)
      → 최적 활용: 높은 FPS + 짧은 노출 시간으로 motion blur 최소화
    """

    def __init__(self, resolution="SVGA", fps=120, depth_mode="PERFORMANCE"):
        if not HAS_ZED:
            raise RuntimeError("ZED SDK가 설치되지 않았습니다")

        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()

        # 해상도 설정
        res_map = {
            "SVGA": sl.RESOLUTION.SVGA,         # 960x600 (ZED X Mini 최적)
            "HD1080": sl.RESOLUTION.HD1080,      # 1920x1080
            "HD1200": sl.RESOLUTION.HD1200,      # 1920x1200
            "HD720": sl.RESOLUTION.HD720,        # 1280x720 (ZED X Mini 미지원)
            "HD2K": sl.RESOLUTION.HD2K,
            "VGA": sl.RESOLUTION.VGA,
        }
        self.init_params.camera_resolution = res_map.get(resolution, sl.RESOLUTION.SVGA)
        self.init_params.camera_fps = fps

        # Depth 모드 설정 (NONE = depth OFF, 속도 최적화)
        depth_map = {
            "NONE": sl.DEPTH_MODE.NONE,
            # NEURAL_LIGHT: SDK 5.x부터. 없으면 NEURAL로 폴백
            "NEURAL_LIGHT": getattr(sl.DEPTH_MODE, 'NEURAL_LIGHT', sl.DEPTH_MODE.NEURAL),
            "NEURAL": sl.DEPTH_MODE.NEURAL,
            "NEURAL_PLUS": sl.DEPTH_MODE.NEURAL_PLUS,
            "ULTRA": sl.DEPTH_MODE.ULTRA,
            "QUALITY": sl.DEPTH_MODE.QUALITY,
            "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
        }
        self.init_params.depth_mode = depth_map.get(depth_mode.upper(), sl.DEPTH_MODE.PERFORMANCE)
        self.depth_enabled = depth_mode.upper() != "NONE"
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.depth_minimum_distance = 0.1
        self.init_params.depth_maximum_distance = 3.0

        # 이미지 품질 최적화
        self.init_params.enable_image_enhancement = True

        self.image = sl.Mat()
        self.image_gpu = sl.Mat()  # GPU retrieve용
        self.depth = sl.Mat()
        self.point_cloud = sl.Mat()

        self.runtime_params = sl.RuntimeParameters()
        self.is_open = False
        self._camera_info = None  # open() 후 채워짐

    def open(self):
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED 열기 실패: {err}")
        self.is_open = True

        info = self.zed.get_camera_information()
        cam_model = str(info.camera_model)
        res_w = info.camera_configuration.resolution.width
        res_h = info.camera_configuration.resolution.height
        actual_fps = info.camera_configuration.fps
        calib = info.camera_configuration.calibration_parameters

        # Global Shutter 판별: ZED X Mini / ZED X 시리즈는 Global Shutter
        is_gs = any(kw in cam_model for kw in ["ZED_XM", "ZED_X", "X Mini", "ZED X"])

        # 카메라 메타데이터 저장 (벤치마크 결과에 포함)
        self._camera_info = {
            "camera_model": cam_model,
            "serial_number": str(info.serial_number),
            "firmware_version": str(getattr(info, 'camera_firmware_version', 'N/A')),
            "resolution": f"{res_w}x{res_h}",
            "fps": actual_fps,
            "shutter_type": "Global Shutter" if is_gs else "Rolling Shutter",
            "is_global_shutter": is_gs,
            "depth_mode": "ON" if self.depth_enabled else "OFF",
            "calibration": {
                "fx": float(calib.left_cam.fx),
                "fy": float(calib.left_cam.fy),
                "cx": float(calib.left_cam.cx),
                "cy": float(calib.left_cam.cy),
            },
        }

        print(f"[ZED] 카메라 열림: {cam_model}")
        print(f"  해상도: {res_w}x{res_h}")
        print(f"  FPS: {actual_fps}")
        print(f"  셔터: {'Global Shutter ✓ (motion blur 없음)' if is_gs else 'Rolling Shutter'}")
        print(f"  Depth: {'ON' if self.depth_enabled else 'OFF (속도 최적화)'}")
        return True

    def configure_for_benchmark(self, exposure_us=None):
        """벤치마크 최적 카메라 설정

        Global Shutter 최적 활용:
          - ZED X Mini의 Global Shutter는 빠른 보행 동작에서
            motion blur 없이 sharp frame을 제공
          - Rolling Shutter 카메라 대비 정확한 keypoint 위치 추출 가능
          - 짧은 노출 시간 설정으로 motion blur 완전 제거 권장

        Args:
            exposure_us: 노출 시간 (마이크로초). None=자동, 2000~4000 권장
                         짧을수록 motion blur 감소하지만 어두워짐
        """
        if not self.is_open:
            return

        if exposure_us is not None:
            # 수동 노출: 보행 분석에서 motion blur 최소화
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, exposure_us // 100)
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 50)
            print(f"  [카메라] 수동 노출: {exposure_us}us, Gain: 50")
        else:
            # 자동 노출: 환경에 맞게 자동 조절
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 1)
            print(f"  [카메라] 자동 노출/게인")

        # Depth confidence 임계값 (50 = 균형잡힌 설정)
        self.runtime_params.confidence_threshold = 50

    def get_camera_info(self):
        """카메라 메타데이터 반환 (벤치마크 결과 저장용)"""
        return self._camera_info or {}

    def grab(self):
        """프레임 캡처. 성공 시 True"""
        if not self.is_open:
            return False
        return self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS

    def get_rgb(self, copy=True, raw_bgra=False):
        """BGR 이미지 반환 (OpenCV 호환)
        Args:
            copy: True면 BGR 복사본 반환, False면 내부 버퍼 참조
            raw_bgra: True면 BGRA numpy view 그대로 반환 (zero-copy, 최고 속도)
        """
        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
        if raw_bgra:
            return self.image.get_data()
        bgra = self.image.get_data()
        bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        return bgr.copy() if copy else bgr

    def get_rgb_gpu_transfer(self):
        """ZED GPU → numpy (GPU retrieve 후 최소 복사)

        기존 get_rgb: retrieve(CPU) → BGRA→BGR 변환 → copy
        이 방식:     retrieve(GPU) → get_data() → numpy (1회 복사)

        호출자가 torch.from_numpy(bgra).cuda(non_blocking=True)로 비동기 전송.
        BGRA→RGB 변환은 torch GPU에서 처리 (trt_pose_engine._preprocess_gpu).
        """
        self.zed.retrieve_image(self.image_gpu, sl.VIEW.LEFT, sl.MEM.GPU)
        # GPU Mat의 데이터를 CPU numpy로 가져옴 (sl.Mat 내부 처리)
        # 이후 torch.cuda(non_blocking=True)로 비동기 GPU 전송
        bgra = self.image_gpu.get_data()
        return bgra

    def get_depth(self, copy=False):
        """Depth map (numpy float32, meters)
        Args:
            copy: True면 안전한 복사본 반환, False면 내부 버퍼 직접 참조 (빠르지만 다음 grab에서 덮어씀)
        """
        if not self.depth_enabled:
            return None
        self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
        data = self.depth.get_data()
        return data.copy() if copy else data

    def get_point_cloud(self):
        """3D Point Cloud (numpy float32, Nx4: x,y,z,rgba)"""
        if not self.depth_enabled:
            return None
        self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
        return self.point_cloud.get_data().copy()

    def pixel_to_3d(self, x, y, depth_map=None, patch_radius=3):
        """2D 픽셀 좌표 → 3D 좌표 (패치 중앙값 depth 사용)

        단일 픽셀 대신 주변 (2r+1)×(2r+1) 패치의 중앙값으로
        depth 노이즈를 제거하여 Joint 3D 위치 정확도 향상.
        """
        if not self.depth_enabled:
            return None
        if depth_map is None:
            self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
            depth_map = self.depth.get_data()

        x_int, y_int = int(x), int(y)
        h, w = depth_map.shape[:2]
        if not (0 <= x_int < w and 0 <= y_int < h):
            return None

        z = _sample_depth_patch(depth_map, x_int, y_int, patch_radius)
        if not np.isfinite(z) or z <= 0:
            return None

        calib = self.zed.get_camera_information().camera_configuration.calibration_parameters
        fx = calib.left_cam.fx
        fy = calib.left_cam.fy
        cx = calib.left_cam.cx
        cy = calib.left_cam.cy
        x3d = (x_int - cx) * z / fx
        y3d = (y_int - cy) * z / fy
        return np.array([x3d, y3d, z], dtype=np.float32)

    def get_gravity_vector(self):
        """IMU에서 중력 방향 벡터 반환 (카메라 좌표계 기준).

        ZED IMU quaternion → rotation matrix → world Y축(up)을
        카메라 좌표계로 변환. thigh_inclination 계산에 사용.
        IMU 미지원 카메라면 None 반환.
        """
        if not self.is_open:
            return None
        sensors = sl.SensorsData()
        self.zed.get_sensors_data(sensors, sl.TIME_REFERENCE.CURRENT)
        q = sensors.get_imu_data().get_pose().get_orientation().get()
        # quaternion [x, y, z, w] → rotation matrix
        x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        # world up = [0, 1, 0] → camera frame으로 변환
        # R^T * [0,1,0] (ZED SDK: q = camera→world)
        up_x = 2.0 * (x * y - w * z)
        up_y = 1.0 - 2.0 * (x * x + z * z)
        up_z = 2.0 * (y * z + w * x)
        return np.array([up_x, up_y, up_z], dtype=np.float32)

    def close(self):
        if self.is_open:
            self.zed.close()
            self.is_open = False

    def __del__(self):
        self.close()


class WebcamFallback:
    """ZED 없을 때 일반 웹캠으로 폴백 (depth 없음)"""

    def __init__(self, camera_id=0, width=1280, height=720, **kwargs):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.frame = None
        self.is_open = False

    def open(self):
        self.is_open = self.cap.isOpened()
        if self.is_open:
            print(f"[Webcam] 카메라 열림: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                  f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        return self.is_open

    def grab(self):
        ret, self.frame = self.cap.read()
        return ret

    def get_rgb(self):
        return self.frame.copy() if self.frame is not None else None

    def get_depth(self):
        return None  # 웹캠은 depth 없음

    def pixel_to_3d(self, x, y, depth_map=None):
        return None

    def close(self):
        self.cap.release()
        self.is_open = False


class VideoFileSource:
    """동영상 파일에서 프레임 읽기 (카메라 없이 벤치마크용)"""

    def __init__(self, video_path, loop=True, **kwargs):
        self.video_path = video_path
        self.loop = loop
        self.cap = None
        self.frame = None
        self.is_open = False

    def open(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"동영상 파일 열기 실패: {self.video_path}")
        self.is_open = True
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[Video] 파일 열림: {self.video_path}")
        print(f"  해상도: {w}x{h}, FPS: {fps:.1f}, 총 프레임: {total}")
        return True

    def grab(self):
        ret, self.frame = self.cap.read()
        if not ret and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, self.frame = self.cap.read()
        return ret

    def get_rgb(self):
        return self.frame.copy() if self.frame is not None else None

    def get_depth(self):
        return None

    def pixel_to_3d(self, x, y, depth_map=None):
        return None

    def close(self):
        if self.cap:
            self.cap.release()
        self.is_open = False


class SVO2FileSource:
    """ZED SVO2 파일에서 RGB + Depth 재생 (녹화 영상으로 3D 벤치마크)"""

    def __init__(self, svo_path, loop=True, **kwargs):
        if not HAS_ZED:
            raise RuntimeError("SVO2 재생에는 ZED SDK가 필요합니다")
        self.svo_path = svo_path
        self.loop = loop
        self.zed = sl.Camera()
        self.image = sl.Mat()
        self.depth = sl.Mat()
        self.runtime_params = sl.RuntimeParameters()
        self.is_open = False
        self._total_frames = 0
        self._calib = None

    def open(self):
        init = sl.InitParameters()
        init.set_from_svo_file(self.svo_path)
        init.svo_real_time_mode = False  # 최대 속도로 재생
        init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        init.coordinate_units = sl.UNIT.METER

        err = self.zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"SVO2 열기 실패: {err}")

        self.is_open = True
        info = self.zed.get_camera_information()
        w = info.camera_configuration.resolution.width
        h = info.camera_configuration.resolution.height
        fps = info.camera_configuration.fps
        self._total_frames = self.zed.get_svo_number_of_frames()
        self._calib = info.camera_configuration.calibration_parameters

        print(f"[SVO2] 파일 열림: {self.svo_path}")
        print(f"  해상도: {w}x{h}, FPS: {fps}, 총 프레임: {self._total_frames}")
        print(f"  Depth: 활성 (3D 벤치마크 가능)")
        return True

    def grab(self):
        if not self.is_open:
            return False
        err = self.zed.grab(self.runtime_params)
        if err == sl.ERROR_CODE.SUCCESS:
            return True
        elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            if self.loop:
                self.zed.set_svo_position(0)
                return self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS
            return False
        return False

    def get_rgb(self, copy=True, raw_bgra=False):
        """API matched to ZEDCamera.get_rgb so PipelinedCamera works with SVO.

        Without this signature PipelinedCamera._loop calls
        get_rgb(copy=True, raw_bgra=True) → TypeError in the capture thread →
        ready_rgb event never sets → main blocks 1s timeout → 1Hz stuck.
        """
        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
        if raw_bgra:
            data = self.image.get_data()       # BGRA 4-channel
            return data.copy() if copy else data
        bgra = self.image.get_data()
        bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        return bgr.copy() if copy else bgr

    def get_depth(self, copy=True):
        """API matched to ZEDCamera.get_depth.
        Default copy=True preserves prior SVO2FileSource behavior.
        """
        self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
        data = self.depth.get_data()
        return data.copy() if copy else data

    def pixel_to_3d(self, x, y, depth_map=None, patch_radius=3):
        """2D 픽셀 좌표 → 3D 좌표 (패치 중앙값 depth 사용)"""
        if depth_map is None:
            self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
            depth_map = self.depth.get_data()

        x_int, y_int = int(x), int(y)
        h, w = depth_map.shape[:2]
        if not (0 <= x_int < w and 0 <= y_int < h):
            return None

        z = _sample_depth_patch(depth_map, x_int, y_int, patch_radius)
        if not np.isfinite(z) or z <= 0:
            return None

        fx = self._calib.left_cam.fx
        fy = self._calib.left_cam.fy
        cx = self._calib.left_cam.cx
        cy = self._calib.left_cam.cy
        x3d = (x_int - cx) * z / fx
        y3d = (y_int - cy) * z / fy
        return np.array([x3d, y3d, z], dtype=np.float32)

    def close(self):
        if self.is_open:
            self.zed.close()
            self.is_open = False

    def __del__(self):
        self.close()


# ============================================================================
# GPU 전처리 유틸리티
# ============================================================================
class GPUPreprocessor:
    """cv2.cuda를 이용한 GPU 전처리 (리사이즈, 색공간 변환)

    CPU 전처리 대비 2~5ms 절약 가능 (특히 고해상도에서).
    cv2.cuda 미지원 시 자동으로 CPU fallback.
    """

    def __init__(self):
        self.use_gpu = HAS_CUDA_CV
        if self.use_gpu:
            self._gpu_mat = cv2.cuda_GpuMat()
            self._gpu_resized = cv2.cuda_GpuMat()
            print("[GPU Preproc] cv2.cuda 활성화")
        else:
            print("[GPU Preproc] cv2.cuda 미지원 → CPU fallback")

    def resize(self, image, target_size):
        """이미지 리사이즈 (width, height)"""
        if self.use_gpu:
            self._gpu_mat.upload(image)
            self._gpu_resized = cv2.cuda.resize(self._gpu_mat, target_size)
            return self._gpu_resized.download()
        return cv2.resize(image, target_size)

    def bgr_to_rgb(self, image):
        """BGR → RGB 변환"""
        if self.use_gpu:
            self._gpu_mat.upload(image)
            gpu_rgb = cv2.cuda.cvtColor(self._gpu_mat, cv2.COLOR_BGR2RGB)
            return gpu_rgb.download()
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def preprocess(self, image, target_size=None, to_rgb=False):
        """리사이즈 + 색공간 변환을 한번에 (GPU 메모리 전송 최소화)"""
        if self.use_gpu:
            self._gpu_mat.upload(image)
            mat = self._gpu_mat
            if target_size is not None:
                mat = cv2.cuda.resize(mat, target_size)
            if to_rgb:
                mat = cv2.cuda.cvtColor(mat, cv2.COLOR_BGR2RGB)
            return mat.download()
        else:
            if target_size is not None:
                image = cv2.resize(image, target_size)
            if to_rgb:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image


# ============================================================================
# 비동기 카메라 캡처 래퍼
# ============================================================================
class AsyncCamera:
    """카메라를 별도 스레드에서 비동기로 캡처

    메인 스레드에서 GPU 추론하는 동안 카메라 스레드가 다음 프레임을 미리 캡처.
    → GPU 유휴 시간 제거, 파이프라인 병렬화.

    capture_depth=False (기본): grab+rgb만 비동기, depth는 main thread에서 직접 호출.
      → TRT predict와 GPU 경합 없음 (grab의 depth 계산은 겹쳐도 retrieve는 별도)
    capture_depth=True: 기존 방식 (depth도 캡처 스레드에서 가져옴, GPU 경합 주의)

    사용법:
        camera = create_camera(...)
        async_cam = AsyncCamera(camera, queue_size=2, capture_depth=False)
        async_cam.open()
        ...
        rgb = async_cam.get_latest_rgb()   # 비동기 캡처된 최신 RGB
        # ... model.predict(rgb) ...       # GPU 추론
        depth = async_cam.get_depth()      # predict 후 depth retrieve (경합 없음)
        ...
        async_cam.close()
    """

    def __init__(self, camera, queue_size=2, capture_depth=False, raw_bgra=False):
        self.camera = camera
        self.queue_size = queue_size
        self._capture_depth = capture_depth
        self._raw_bgra = raw_bgra
        self._frame_queue = deque(maxlen=queue_size)
        self._lock = threading.Lock()
        self._zed_lock = threading.Lock()  # ZED SDK thread-safety 보장
        self._running = False
        self._thread = None
        self.is_open = False
        self.depth_enabled = getattr(camera, 'depth_enabled', False)

    def open(self):
        result = self.camera.open()
        self.is_open = self.camera.is_open
        self.depth_enabled = getattr(self.camera, 'depth_enabled', False)
        if self.is_open:
            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            timeout = 2.0
            t0 = time.monotonic()
            while len(self._frame_queue) == 0 and time.monotonic() - t0 < timeout:
                time.sleep(0.001)
            fmt = "BGRA" if self._raw_bgra else "BGR"
            depth_mode = "+depth" if self._capture_depth else " (depth deferred)"
            print(f"[AsyncCamera] 비동기 캡처 시작 ({fmt}{depth_mode}, queue={self.queue_size})")
        return result

    def _capture_loop(self):
        """백그라운드 캡처 스레드 — ZED SDK 호출은 _zed_lock으로 보호"""
        while self._running:
            with self._zed_lock:
                grabbed = self.camera.grab()
                if grabbed:
                    if self._raw_bgra:
                        try:
                            frame = self.camera.get_rgb(copy=True, raw_bgra=True)
                        except TypeError:
                            frame = self.camera.get_rgb(copy=True)
                    else:
                        frame = self.camera.get_rgb(copy=True)
                    depth = None
                    if self._capture_depth and hasattr(self.camera, 'get_depth'):
                        depth = self.camera.get_depth()
            if grabbed:
                with self._lock:
                    self._frame_queue.append((frame, depth))
            else:
                time.sleep(0.001)

    def grab(self):
        """최신 프레임이 있으면 True"""
        with self._lock:
            return len(self._frame_queue) > 0

    def get_rgb(self, copy=False):
        """최신 RGB 프레임 반환"""
        with self._lock:
            if self._frame_queue:
                rgb, _ = self._frame_queue[-1]
                return rgb.copy() if copy else rgb
        return None

    def get_depth(self):
        """Depth 가져오기.
        capture_depth=True: 캡처 스레드에서 가져온 depth 반환
        capture_depth=False: 내부 camera에서 직접 retrieve (_zed_lock으로 thread-safe)
        """
        if not self._capture_depth:
            with self._zed_lock:
                if hasattr(self.camera, 'get_depth'):
                    return self.camera.get_depth()
            return None
        with self._lock:
            if self._frame_queue:
                _, depth = self._frame_queue[-1]
                return depth
        return None

    def get_latest_rgb(self):
        """최신 RGB만 소비 (큐에서 제거). depth는 get_depth()로 별도 호출.
        Returns: rgb 또는 None
        """
        with self._lock:
            if self._frame_queue:
                rgb, _ = self._frame_queue.pop()
                return rgb
        return None

    def get_latest(self):
        """최신 프레임 소비 (큐에서 제거)
        Returns: (rgb, depth) 또는 (None, None)
        """
        with self._lock:
            if self._frame_queue:
                return self._frame_queue.pop()
        return None, None

    def pixel_to_3d(self, x, y, depth_map=None):
        """3D 변환 위임"""
        if hasattr(self.camera, 'pixel_to_3d'):
            return self.camera.pixel_to_3d(x, y, depth_map)
        return None

    def get_gravity_vector(self):
        """IMU gravity vector 위임 (_zed_lock으로 thread-safe)"""
        with self._zed_lock:
            if hasattr(self.camera, 'get_gravity_vector'):
                return self.camera.get_gravity_vector()
        return None

    def close(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.camera.close()
        self.is_open = False


# ============================================================================
# 파이프라인 카메라 — grab과 predict를 겹치는 Double Buffer
# ============================================================================
class PipelinedCamera:
    """grab과 predict를 파이프라인으로 병렬화.

    캡처 스레드가 grab+rgb+depth를 한번에 가져오고,
    main 스레드가 소비할 때까지 대기. ZED SDK 호출이 한 스레드에서만
    일어나므로 segfault 위험 없음.

    타이밍:
        기존:     [grab 9ms] → [predict 15ms] → [grab 9ms] → ...  = 24ms/frame
        파이프라인: [grab 9ms]  [grab 9ms]   [grab 9ms]
                   [predict 15ms] [predict 15ms]                   = 15ms/frame

    사용법:
        raw = create_camera(...)
        cam = PipelinedCamera(raw)
        cam.open()
        while True:
            rgb, depth = cam.get()       # 비동기 캡처된 프레임
            result = model.predict(rgb)  # grab과 겹침
        cam.close()
    """

    def __init__(self, camera, skip_imu=False, serialize_depth=True):
        """
        Args:
            camera: ZEDCamera 인스턴스
            skip_imu: Method A면 True (IMU retrieve 건너뜀)
            serialize_depth: True면 P0-5 모드 — depth retrieve를 main의 predict 뒤로 미룸
                             GPU 경합 제거 (predict spike 해결). 평균 +1-2ms 지연 감수.
                             False면 기존 병렬 모드 (retrieve_depth가 predict와 겹침).
        """
        self.camera = camera
        self.frame = None
        self._depth = None
        self._gravity = None
        self.skip_imu = skip_imu
        self.serialize_depth = serialize_depth

        # Ready 이벤트 3단계
        self.ready_rgb   = threading.Event()  # grab + retrieve_image 완료
        self.ready_depth = threading.Event()  # retrieve_depth 완료
        self.consumed    = threading.Event()  # main이 release() 호출
        self.consumed.set()

        # P0-5 serialize 모드 전용:
        # main이 predict 끝내고 depth 요청 → capture가 retrieve_depth 시작
        self.depth_request = threading.Event()

        # parallel-safe (default) 모드 전용:
        # main 이 get_depth_and_gravity 로 현재 frame depth 읽고 set →
        # capture 가 다음 frame depth retrieve 진행. RGB-depth atomic 보장.
        # 초기 set: 첫 iter 의 capture 가 depth retrieve 즉시 진행 가능.
        self.depth_consumed = threading.Event()
        self.depth_consumed.set()

        # 하위 호환
        self.ready = threading.Event()

        self._running = False
        self._thread = None
        self.is_open = False
        self.depth_enabled = getattr(camera, 'depth_enabled', False)

    def open(self):
        result = self.camera.open()
        self.is_open = self.camera.is_open
        self.depth_enabled = getattr(self.camera, 'depth_enabled', False)
        if self.is_open:
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            # 첫 프레임 대기 (depth까지 준비)
            self.ready_depth.wait(timeout=3.0)
            print("[PipelinedCamera] 파이프라인 캡처 시작 (RGB/depth 분리 ready)")
        return result

    def _loop(self):
        """캡처 스레드: grab → retrieve_image → [depth 요청 대기] → retrieve_depth

        P0-5 (serialize_depth=True) 모드:
          capture: grab → retrieve_image → ready_rgb → depth_request 대기
                   → retrieve_depth → ready_depth → consumed 대기
          main:    get_rgb → release → predict(GPU 단독) → depth_request.set
                   → get_depth (0 대기) → depth_3d → shm → next

          → retrieve_depth가 predict와 절대 안 겹침 (GPU 순차).
          → 평균 +1-2ms, spike 완전 제거.

        기존 모드 (serialize_depth=False):
          capture가 retrieve_depth까지 해서 main의 predict와 병렬.
          평균 빠르지만 GPU 경합으로 spike 20-28ms 간헐.
        """
        while self._running:
            # Step 1-2: main의 release 대기 후 허가 소비
            self.consumed.wait(timeout=0.1)
            if not self._running:
                break
            self.consumed.clear()
            if self.serialize_depth:
                self.depth_request.clear()  # 다음 사이클용 reset
                self.ready_depth.clear()

            if not self.camera.grab():
                continue

            # ── Phase 1: RGB만 먼저 (main이 predict 시작 가능) ──
            self.frame = self.camera.get_rgb(copy=True, raw_bgra=True)
            self.ready_rgb.set()
            self.ready.set()

            # ── Phase 2: depth retrieve
            #   serialize_depth=True: main이 predict 끝내고 depth_request.set 할 때까지 대기
            #   False (parallel-safe): main 이 이전 frame depth 를 consume 할 때까지 대기.
            #     이 대기 가 없으면 frame K+1 의 depth 가 main 의 frame K depth 읽기 보다
            #     먼저 self._depth 를 덮어쓰는 race 발생 (RGB-depth misalignment).
            #     첫 iter 는 depth_consumed 가 init 시 set 되어 있어 즉시 진행.
            if self.serialize_depth:
                self.depth_request.wait(timeout=0.1)
                if not self._running:
                    break
            else:
                # parallel-safe: wait for main consumption of previous depth
                while self._running and not self.depth_consumed.wait(timeout=0.1):
                    pass
                if not self._running:
                    break
                self.depth_consumed.clear()

            if hasattr(self.camera, 'get_depth'):
                try:
                    self._depth = self.camera.get_depth(copy=True)
                except TypeError:
                    d = self.camera.get_depth()
                    self._depth = d.copy() if d is not None else None

            # ── Phase 3: IMU (Method A면 skip) ──
            if not self.skip_imu and hasattr(self.camera, 'get_gravity_vector'):
                self._gravity = self.camera.get_gravity_vector()
            else:
                self._gravity = None

            self.ready_depth.set()

    def get_rgb(self, timeout=1.0):
        """RGB만 먼저 가져옴 (predict 시작용). release()는 depth_3d 끝난 뒤 호출."""
        self.ready_rgb.wait(timeout=timeout)
        self.ready_rgb.clear()
        return self.frame

    def get_depth_and_gravity(self, timeout=1.0):
        """predict 끝난 뒤 호출 — 현재 frame 의 depth + gravity 반환.

        serialize_depth=True: 여기서 depth_request.set() → capture가 retrieve 시작.
                              retrieve 3ms 소요 후 ready_depth.set() 기다림.
        serialize_depth=False (parallel-safe): 현재 frame 의 retrieve 는 이미 완료 됨.
                              return 후 depth_consumed.set() 으로 capture 에 다음
                              frame 의 retrieve 허용 신호. RGB-depth atomic 보장.
        """
        if self.serialize_depth:
            self.depth_request.set()  # capture에게 "이제 retrieve 해도 됨" 신호
        self.ready_depth.wait(timeout=timeout)
        self.ready_depth.clear()
        # Snapshot CURRENT frame's depth+gravity BEFORE signaling capture, so any
        # subsequent overwrite by capture's next-iter retrieve doesn't affect us.
        depth, gravity = self._depth, self._gravity
        if not self.serialize_depth:
            self.depth_consumed.set()  # capture 에 다음 frame retrieve 허용
        return depth, gravity

    def get(self):
        """하위 호환: RGB + depth + IMU 한번에 (두 ready 모두 대기)."""
        self.ready_rgb.wait(timeout=1.0)
        self.ready_depth.wait(timeout=1.0)
        self.ready_rgb.clear()
        self.ready_depth.clear()
        self.ready.clear()
        return self.frame, self._depth, self._gravity

    def get_depth(self):
        """하위 호환용"""
        return self._depth

    def get_gravity_vector(self):
        return self._gravity

    def release(self):
        """다음 grab 시작 — depth_3d 끝난 뒤 호출."""
        self.consumed.set()

    def pixel_to_3d(self, x, y, depth_map=None):
        if hasattr(self.camera, 'pixel_to_3d'):
            return self.camera.pixel_to_3d(x, y, depth_map)
        return None

    def close(self):
        self._running = False
        # Wake the capture thread from any wait point so it can exit cleanly.
        self.consumed.set()        # consumed.wait
        self.depth_consumed.set()  # depth_consumed.wait (parallel-safe mode)
        self.depth_request.set()   # depth_request.wait (serialize mode)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.camera.close()
        self.is_open = False


def create_camera(use_zed=True, video_path=None, depth_mode="PERFORMANCE",
                   async_capture=False, queue_size=2, **kwargs):
    """카메라 팩토리 함수

    SVO2/SVO 파일은 자동으로 ZED SDK로 열어 depth 포함 재생.
    일반 동영상(mp4, avi 등)은 OpenCV로 열어 RGB만 재생.

    Args:
        depth_mode: "NONE" = depth OFF (속도 최적화), "PERFORMANCE" = 기본
        async_capture: True면 비동기 캡처 (GPU 유휴 시간 제거)
        queue_size: 비동기 캡처 시 프레임 큐 크기
    """
    if video_path:
        ext = os.path.splitext(video_path)[1].lower()
        if ext in ('.svo2', '.svo') and HAS_ZED:
            camera = SVO2FileSource(video_path, **kwargs)
        else:
            camera = VideoFileSource(video_path, **kwargs)
    elif use_zed and HAS_ZED:
        camera = ZEDCamera(depth_mode=depth_mode, **kwargs)
    else:
        camera = WebcamFallback(**kwargs)

    if async_capture:
        return AsyncCamera(camera, queue_size=queue_size)
    return camera

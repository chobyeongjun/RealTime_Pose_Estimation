"""
후처리 가속 래퍼
================
C++ 확장 모듈(pose_postprocess_cpp)이 빌드되어 있으면 사용하고,
없으면 기존 Python 구현으로 자동 폴백합니다.

사용법:
    from postprocess_accel import (
        batch_2d_to_3d,
        create_joint_3d_filter,
        compute_lower_limb_angles_fast,
        HAS_CPP_EXT,
    )

빌드 방법:
    cd benchmarks/cpp_ext
    pip install pybind11
    python setup.py build_ext --inplace
"""

import sys
import os
import numpy as np

# C++ 확장 모듈 로딩 시도
HAS_CPP_EXT = False
_cpp = None

# cpp_ext/ 디렉토리를 검색 경로에 추가
_cpp_ext_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpp_ext")
if _cpp_ext_dir not in sys.path:
    sys.path.insert(0, _cpp_ext_dir)

try:
    import pose_postprocess_cpp as _cpp
    HAS_CPP_EXT = True
    print("[postprocess_accel] C++ 확장 모듈 로드 성공 — 후처리 가속 활성화")
except ImportError:
    print("[postprocess_accel] C++ 확장 모듈 미빌드 — Python 폴백 모드")


# ============================================================================
# batch_2d_to_3d: 2D 키포인트 + depth → 3D 일괄 변환
# ============================================================================

def batch_2d_to_3d(keypoints_2d, depth_map, camera, crop_x=0, patch_radius=3):
    """2D 키포인트를 3D 좌표로 일괄 변환.

    Args:
        keypoints_2d: dict {name → (px, py)}
        depth_map: (H, W) float32 numpy array
        camera: ZEDCamera 또는 SVO2FileSource (pixel_to_3d, calibration 가능한 객체)
        crop_x: 크롭 x 오프셋
        patch_radius: depth 패치 반경

    Returns:
        dict {name → (X, Y, Z)} — depth 무효인 키포인트는 제외
    """
    if not keypoints_2d or depth_map is None:
        return {}

    if HAS_CPP_EXT:
        return _batch_2d_to_3d_cpp(keypoints_2d, depth_map, camera, crop_x, patch_radius)
    else:
        return _batch_2d_to_3d_python(keypoints_2d, depth_map, camera, crop_x, patch_radius)


def _get_camera_intrinsics(camera):
    """카메라 intrinsic 파라미터 추출."""
    if hasattr(camera, '_calib') and camera._calib is not None:
        calib = camera._calib
    elif hasattr(camera, 'zed'):
        calib = camera.zed.get_camera_information().camera_configuration.calibration_parameters
    else:
        return None
    return calib.left_cam.fx, calib.left_cam.fy, calib.left_cam.cx, calib.left_cam.cy


def _batch_2d_to_3d_cpp(keypoints_2d, depth_map, camera, crop_x=0, patch_radius=3):
    """C++ 배치 처리."""
    intrinsics = _get_camera_intrinsics(camera)
    if intrinsics is None:
        return _batch_2d_to_3d_python(keypoints_2d, depth_map, camera, crop_x, patch_radius)

    fx, fy, cx, cy = intrinsics
    names = list(keypoints_2d.keys())
    coords = np.array(
        [[px + crop_x, py] for (px, py) in keypoints_2d.values()],
        dtype=np.float32,
    )

    depth_f32 = depth_map if depth_map.dtype == np.float32 else depth_map.astype(np.float32)

    return _cpp.batch_2d_to_3d(names, coords, depth_f32, fx, fy, cx, cy, patch_radius)


def _batch_2d_to_3d_python(keypoints_2d, depth_map, camera, crop_x=0, patch_radius=3):
    """Python 폴백 — 기존 for문 방식."""
    result = {}
    for name, (px, py) in keypoints_2d.items():
        pt3d = camera.pixel_to_3d(px + crop_x, py, depth_map)
        if pt3d is not None:
            result[name] = tuple(pt3d)
    return result


# ============================================================================
# Joint3DFilter: C++ 또는 Python 자동 선택
# ============================================================================

def create_joint_3d_filter(min_cutoff=0.5, beta=0.01, d_cutoff=1.0,
                           max_missing=5, calib_frames=30, tolerance=0.20):
    """Joint3DFilter 인스턴스 생성.

    C++ 확장이 있으면 C++ 버전, 없으면 Python 버전 반환.
    두 버전 모두 동일한 인터페이스:
        .filter(name, pt3d, t) → np.array or None
        .apply_segment_constraint(keypoints_3d) → dict
        .calibrated → bool
        .calib_progress → float
        .reset()
    """
    if HAS_CPP_EXT:
        print("[postprocess_accel] Joint3DFilter: C++ 버전 사용")
        return _CppJoint3DFilterWrapper(
            min_cutoff, beta, d_cutoff, max_missing, calib_frames, tolerance)
    else:
        print("[postprocess_accel] Joint3DFilter: Python 버전 사용")
        from zed_camera import Joint3DFilter
        return Joint3DFilter(
            min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff,
            max_missing=max_missing, calib_frames=calib_frames, tolerance=tolerance)


class _CppJoint3DFilterWrapper:
    """C++ Joint3DFilter를 Python Joint3DFilter와 동일한 인터페이스로 래핑."""

    def __init__(self, min_cutoff, beta, d_cutoff, max_missing, calib_frames, tolerance):
        self._impl = _cpp.Joint3DFilter(
            min_cutoff, beta, d_cutoff, max_missing, calib_frames, tolerance)

    def filter(self, name, pt3d, t):
        """3D 좌표 필터링.

        Args:
            name: 키포인트 이름
            pt3d: np.array([X, Y, Z]) 또는 None
            t: 타임스탬프

        Returns:
            np.array([X, Y, Z]) 또는 None
        """
        if pt3d is None:
            # C++ 측은 빈 배열로 "무효" 신호
            empty = np.array([], dtype=np.float32)
            result = self._impl.filter(name, empty, t)
        else:
            arr = np.asarray(pt3d, dtype=np.float32)
            result = self._impl.filter(name, arr, t)

        if result.size == 0:
            return None
        return result

    def apply_segment_constraint(self, keypoints_3d):
        """3D 세그먼트 제약 적용.

        Args:
            keypoints_3d: dict {name → (X, Y, Z)} — in-place로 수정됨

        Returns:
            bool: 제약이 적용되었으면 True
        """
        before_keys = set(keypoints_3d.keys())
        result_dict = self._impl.apply_segment_constraint(keypoints_3d)

        # C++에서 반환된 dict로 업데이트
        changed = False
        for k, v in result_dict.items():
            if k in keypoints_3d:
                old = keypoints_3d[k]
                if old != v:
                    keypoints_3d[k] = v
                    changed = True
        return changed

    @property
    def calibrated(self):
        return self._impl.calibrated

    @property
    def calib_progress(self):
        return self._impl.calib_progress

    def get_ref_lengths_3d(self):
        return self._impl.get_ref_lengths_3d()

    def reset(self):
        self._impl.reset()


# ============================================================================
# compute_lower_limb_angles: 관절 각도 일괄 계산
# ============================================================================

def compute_lower_limb_angles_fast(pose_result, use_3d=True):
    """하체 관절 각도 일괄 계산.

    C++ 확장이 있으면 C++, 없으면 Python joint_angles.py 사용.

    Args:
        pose_result: PoseResult 객체
        use_3d: True면 keypoints_3d, False면 keypoints_2d

    Returns:
        dict {angle_name → angle_degrees}
    """
    if HAS_CPP_EXT:
        keypoints = pose_result.keypoints_3d if use_3d else pose_result.keypoints_2d
        return _cpp.compute_lower_limb_angles(
            keypoints, pose_result.confidences, use_3d)
    else:
        from joint_angles import compute_lower_limb_angles
        return compute_lower_limb_angles(pose_result, use_3d=use_3d)

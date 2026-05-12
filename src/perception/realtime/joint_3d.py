"""
joint_3d.py
===========
ZED + YOLO26s-lower6 → 6개 관절의 3D 좌표 + 관절 각도

ZED 카메라 좌표계 (카메라 기준):
    X : 이미지 오른쪽 (+)
    Y : 이미지 아래쪽 (+)
    Z : 카메라 광축 방향, 카메라에서 멀어질수록 (+) = depth

워커 장착 시 (위에서 아래로 틸트):
    Z ≈ 카메라 → 관절까지의 직선 거리 (hip이 짧고 ankle이 길다)
    X ≈ 좌우 방향
    Y ≈ 전후 방향 (카메라 틸트에 따라 달라짐)

관절 각도 계산은 3D 벡터 내적을 사용하므로
좌표계 회전(World frame 변환)과 무관하게 정확합니다.
"""

import sys
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

# benchmarks 경로 추가 (기존 코드 재사용)
_bench_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'benchmarks')
if _bench_dir not in sys.path:
    sys.path.insert(0, _bench_dir)


# ─────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────

JOINT_NAMES = ['left_hip', 'right_hip', 'left_knee', 'right_knee',
               'left_ankle', 'right_ankle']

# 해부학적 뼈 길이 범위 [m] — 성인 기준
BONE_RANGES = {
    'left_thigh':  (0.25, 0.55),   # hip → knee (depth 노이즈 고려)
    'right_thigh': (0.25, 0.55),
    'left_shank':  (0.25, 0.50),   # knee → ankle
    'right_shank': (0.25, 0.50),
}

# confidence 임계값 — 이 값 미만이면 해당 관절 무시
CONF_THRESHOLD = 0.5


# ─────────────────────────────────────────────
# 데이터 구조
# ─────────────────────────────────────────────

@dataclass
class JointState3D:
    """한 프레임의 6관절 3D 상태.

    각도 규약:
        knee_flexion_deg   : 완전 신전=0°, 굴곡 증가=양수 (Winter convention)
                             계산: 180° - (hip-knee-ankle 내각)
                             Trunk 불필요 → 항상 사용 가능
                             케이블 제어의 주 피드백 변수

        thigh_inclination_deg : hip→knee 벡터의 수직축 대비 기울기
                             World frame (ZED IMU) 필요
                             Trunk upright 가정 시 Hip Flexion 근사값
                             ZED IMU 없으면 None

        NOTE: True Hip Flexion (thigh vs trunk)은
              Trunk keypoint 없으므로 계산 불가.
    """
    # 3D 좌표 [m], 카메라 좌표계
    positions: Dict[str, np.ndarray] = field(default_factory=dict)
    # 2D 좌표 [px]
    pixels:    Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # confidence [0~1]
    confs:     Dict[str, float] = field(default_factory=dict)

    # ── 제어용 주 각도 [deg] ──
    # Knee Flexion: 완전신전=0°, peak swing≈60° (Winter convention)
    # Trunk 불필요, 항상 사용 가능
    left_knee_flexion:  Optional[float] = None
    right_knee_flexion: Optional[float] = None

    # ── 참고용 Thigh Inclination [deg] ──
    # hip→knee 벡터의 수직(중력) 대비 기울기
    # World frame 필요 (ZED IMU). 없으면 None.
    # Trunk upright 가정 시 Hip Flexion 근사값
    left_thigh_inclination:  Optional[float] = None
    right_thigh_inclination: Optional[float] = None

    # ── Shank Inclination [deg] (Plan D spec 의 *6 joints 중 ankle proxy) ──
    # knee→ankle 벡터의 수직(중력) 대비 기울기
    # World frame 필요 (ZED IMU). 없으면 None.
    # YOLO26s-lower6 은 *toe keypoint X — ankle dorsiflexion 직접 X.
    # shank_inclination 가 *진정 *ankle 의 *진정 *kinematic proxy
    # (gait analysis 의 *진정 *standard practice — Winter, Davis, etc.)
    left_shank_inclination:  Optional[float] = None
    right_shank_inclination: Optional[float] = None

    # ── Depth 품질 ──
    depth_valid_count: int = 0   # 유효 depth 가진 keypoint 수 (0~6)
    depth_quality: Dict[str, float] = field(default_factory=dict)  # {name: depth_m or nan}

    # 뼈 길이 [m]
    bone_lengths: Dict[str, float] = field(default_factory=dict)

    # 메타
    timestamp_us: float = 0.0
    valid: bool = False            # knee_flexion 계산 가능한 keypoint 충분 시 True

    # 하위 호환성 — 기존 코드가 참조하는 경우를 위한 프로퍼티
    @property
    def left_knee_angle(self) -> Optional[float]:
        """하위 호환: 180° - left_knee_flexion"""
        if self.left_knee_flexion is None:
            return None
        return 180.0 - self.left_knee_flexion

    @property
    def right_knee_angle(self) -> Optional[float]:
        if self.right_knee_flexion is None:
            return None
        return 180.0 - self.right_knee_flexion


# ─────────────────────────────────────────────
# 핵심 함수
# ─────────────────────────────────────────────

def _angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    세 점 A-B-C 에서 B의 내각을 구한다 [deg].
    B를 꼭짓점으로 BA, BC 벡터의 사잇각.
    완전 신전 무릎 ≈ 180°, 최대 굴곡 ≈ 60°~70°.
    """
    v1 = a - b
    v2 = c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _bone_length(p_from: np.ndarray, p_to: np.ndarray) -> float:
    return float(np.linalg.norm(p_to - p_from))


def compute_joint_state(
    keypoints_2d: dict,
    keypoints_3d: dict,
    confidences:  dict,
    timestamp_us: float = 0.0,
    world_up_vec: Optional[np.ndarray] = None,  # world frame의 수직(중력 반대) 단위벡터
) -> JointState3D:
    """
    YOLO26s-lower6 출력 + ZED depth → JointState3D 계산.

    Args:
        keypoints_2d  : {name: (px, py)}
        keypoints_3d  : {name: (X, Y, Z)}   — ZED depth 변환 결과
        confidences   : {name: float}
        timestamp_us  : 타임스탬프 [μs]
        world_up_vec  : world frame 수직 단위벡터 (ZED IMU 있을 때 전달)
                        있으면 thigh_inclination 계산 가능

    주요 출력:
        state.left_knee_flexion   : 제어용 주 변수 (0°=신전, 60°=peak swing)
        state.left_thigh_inclination : Hip Flexion 근사 (world_up_vec 필요)
    """
    state = JointState3D(timestamp_us=timestamp_us)

    # ── confidence + depth 유효성 필터 ──
    depth_valid = 0
    for name in JOINT_NAMES:
        conf = confidences.get(name, 0.0)
        if conf < CONF_THRESHOLD:
            continue
        if name in keypoints_2d:
            state.pixels[name] = keypoints_2d[name]
        if name in keypoints_3d:
            pt = keypoints_3d[name]
            z = float(pt[2]) if len(pt) >= 3 else float('nan')
            state.depth_quality[name] = z
            # ZED depth 유효 범위: 0.1 ~ 3.0 m, non-NaN
            if np.isfinite(z) and 0.1 <= z <= 3.0:
                state.positions[name] = np.array(pt[:3], dtype=np.float32)
                depth_valid += 1
        state.confs[name] = conf

    state.depth_valid_count = depth_valid

    # ── 유효성 판단: knee flexion 계산에 필요한 최소 keypoint ──
    # left 또는 right 중 한쪽이라도 hip+knee+ankle 3점 있으면 유효
    def _has_triplet(side: str) -> bool:
        return all(f'{side}_{j}' in state.positions
                   for j in ('hip', 'knee', 'ankle'))

    has_left  = _has_triplet('left')
    has_right = _has_triplet('right')
    state.valid = has_left or has_right

    if not state.valid:
        return state

    pos = state.positions

    # ── 뼈 길이 ──
    for side in ('left', 'right'):
        h, k, a = f'{side}_hip', f'{side}_knee', f'{side}_ankle'
        if h in pos and k in pos:
            state.bone_lengths[f'{side}_thigh'] = _bone_length(pos[h], pos[k])
        if k in pos and a in pos:
            state.bone_lengths[f'{side}_shank'] = _bone_length(pos[k], pos[a])

    # ── Knee Flexion (제어 주 변수) ──
    # Winter convention: 완전신전=0°, 굴곡=양수
    # = 180° - (hip-knee-ankle 내각)
    for side, attr in [('left', 'left_knee_flexion'), ('right', 'right_knee_flexion')]:
        h, k, a = f'{side}_hip', f'{side}_knee', f'{side}_ankle'
        if all(j in pos for j in (h, k, a)):
            raw = _angle_3d(pos[h], pos[k], pos[a])   # 180° at full extension
            setattr(state, attr, 180.0 - raw)           # → 0° at full extension

    # ── Thigh Inclination (Hip Flexion 근사, world frame 필요) ──
    # hip→knee 벡터와 수직축(world_up_vec) 사잇각
    # 90° = 수평, 0° = 수직 아래 (swing 시 앞으로 기울어짐 = 양수 flexion)
    # ── Shank Inclination (Ankle Flexion proxy, Plan D 6-joints spec) ──
    # knee→ankle 벡터와 수직축 사잇각 — gait analysis 의 *진정 *standard
    if world_up_vec is not None:
        up = np.asarray(world_up_vec, dtype=np.float32)
        up_norm = float(np.linalg.norm(up))
        if up_norm > 1e-6:
            up = up / up_norm
            # Thigh (hip → knee)
            for side, attr in [('left', 'left_thigh_inclination'),
                                ('right', 'right_thigh_inclination')]:
                h, k = f'{side}_hip', f'{side}_knee'
                if h in pos and k in pos:
                    thigh = pos[k] - pos[h]
                    thigh_n = float(np.linalg.norm(thigh))
                    if thigh_n > 1e-6:
                        thigh = thigh / thigh_n
                        cos_a = float(np.clip(np.dot(thigh, -up), -1.0, 1.0))
                        inclination = float(np.degrees(np.arccos(cos_a)))
                        setattr(state, attr, inclination)
            # Shank (knee → ankle) — Plan D ankle proxy
            for side, attr in [('left', 'left_shank_inclination'),
                                ('right', 'right_shank_inclination')]:
                k, a = f'{side}_knee', f'{side}_ankle'
                if k in pos and a in pos:
                    shank = pos[a] - pos[k]
                    shank_n = float(np.linalg.norm(shank))
                    if shank_n > 1e-6:
                        shank = shank / shank_n
                        cos_a = float(np.clip(np.dot(shank, -up), -1.0, 1.0))
                        inclination = float(np.degrees(np.arccos(cos_a)))
                        setattr(state, attr, inclination)

    return state


def compute_kp_sigma(
    positions: Dict[str, np.ndarray],
    confs: Dict[str, float],
    fx: float = 480.0,
    fy: Optional[float] = None,
    baseline_m: float = 0.063,
    sigma_d_subpixel: float = 0.25,
    default_sigma_m: float = 0.015,
) -> Dict[str, np.ndarray]:
    """Per-keypoint 3D position uncertainty σ (m) — Plan D EKF R source.

    진정 *stereo depth uncertainty formula (Plan D spec)*:
      σ_z = Z² × σ_d / (fx × baseline)        Z = depth, σ_d = stereo subpixel σ
      σ_x ≈ Z / fx                             (pixel quantization at depth Z)
      σ_y ≈ Z / fy

    Confidence weighting: low conf → larger σ (weight = 1 / max(0.1, conf)).

    Args:
        positions: dict {joint_name: (X, Y, Z)} world frame m
        confs: dict {joint_name: 0-1}
        fx, fy: focal length px (ZED X Mini SVGA ~ 480px)
        baseline_m: stereo baseline (ZED X Mini ~ 0.063m)
        sigma_d_subpixel: stereo matching subpixel σ (default 0.25 — paper)
        default_sigma_m: fallback σ when depth invalid (default 15mm)

    Returns:
        dict {joint_name: np.array([σ_x, σ_y, σ_z]) float32 m}
        Invalid joints → large σ (1.0m) for safe EKF R inflation.
    """
    if fy is None:
        fy = fx
    sigmas: Dict[str, np.ndarray] = {}
    for name, pos in positions.items():
        Z = float(pos[2]) if len(pos) >= 3 else float('nan')
        if not np.isfinite(Z) or Z <= 0:
            sigmas[name] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            continue
        # Stereo depth σ
        sigma_z = (Z * Z) * sigma_d_subpixel / (fx * baseline_m)
        # Pixel quantization at Z
        sigma_x = Z / fx
        sigma_y = Z / fy
        # Confidence weighting
        c = float(confs.get(name, 0.5))
        weight = 1.0 / max(0.1, c)
        sigmas[name] = np.array(
            [sigma_x * weight, sigma_y * weight, sigma_z * weight],
            dtype=np.float32,
        )
    return sigmas


def validate_bone_lengths(state: JointState3D) -> Dict[str, bool]:
    """
    뼈 길이가 해부학적 범위 내인지 확인.
    True = 정상, False = 비정상 (keypoint 오검출 의심).
    """
    results = {}
    for bone, (lo, hi) in BONE_RANGES.items():
        if bone in state.bone_lengths:
            length = state.bone_lengths[bone]
            results[bone] = lo <= length <= hi
    return results

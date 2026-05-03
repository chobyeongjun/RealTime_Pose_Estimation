"""GPU postprocess — YOLOv8/26-pose decode → 3D (depth sample) → optional filter.

Schema-aware: supports ``coco17`` (17 kpts, last-dim 56) and the H-Walker
custom ``lowlimb6`` (6 kpts, last-dim 23) via :mod:`keypoint_config`.

IMPORTANT defaults (from skiro-learnings / user feedback):

  * ``use_filter=False`` — OneEuro was observed to *suppress* detection on
    the custom lower-limb model; initial low-confidence keypoints get
    zero'd and the filter remembers that zero, dragging subsequent real
    detections toward 0. **Leave off by default.**
  * Bone-length constraint lives in :mod:`constraints` — **do NOT enable**
    during calibration / initial warm-up (positive feedback loop observed
    with yolo26s-lower6-v2: left-thigh collapsed to 9px vs 31px right).

Graph-capture safe: the ``int(...argmax.item())`` D2H in this module means
**do NOT** include the postprocess step in ``cuda_graph.GraphedStep``;
graph capture targets only preproc + infer.

Output last-dim layout for YOLOv8/26 pose head:
    raw: (1, N, 5 + K*3), where K = num_keypoints.
    last_dim = 5 + K*3  →  coco17 ⇒ 56,  lowlimb6 ⇒ 23.
    Legacy detect+cls layout (last_dim = 6 + K*3) is also auto-detected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .gpu_preprocess import LetterboxParams
from .keypoint_config import COCO17, KeypointSchema, get_schema


@dataclass
class PoseResult:
    kpts_2d_px: torch.Tensor   # (K, 2) original-image pixels, CUDA
    kpts_3d_m: torch.Tensor    # (K, 3) meters in camera frame, CUDA
    kpt_conf: torch.Tensor     # (K,) float, CUDA
    box_conf: float
    valid: bool                # False when no person was detected above threshold
    depth_invalid_ratio: float  # 0..1 — fraction of keypoints whose depth was bad


class OneEuroFilter1D:
    """Vectorized OneEuro filter (last-axis samples, flexible shape).

    NB: invalid samples (caller passes NaN) are skipped — this prevents
    the "zero sink" failure mode we hit with the lower-limb model.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        min_cutoff: float = 2.5,   # higher than old 1.0 → less lag
        beta: float = 0.3,         # stronger adaptation for fast walking
        d_cutoff: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.device = device or torch.device("cuda:0")
        self.x_prev: Optional[torch.Tensor] = None
        self.dx_prev = torch.zeros(shape, device=self.device)
        self.t_prev: Optional[float] = None

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev.zero_()
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt: float):
        tau = 1.0 / (2.0 * 3.141592653589793 * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: torch.Tensor, t_s: float) -> torch.Tensor:
        if self.x_prev is None or self.t_prev is None:
            self.x_prev = x.clone()
            self.t_prev = t_s
            return x
        dt = max(t_s - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * dx_hat.abs()
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t_s
        return x_hat


class GpuPostprocessor:
    """Decode pose head output, sample depth, optionally smooth."""

    def __init__(
        self,
        schema: KeypointSchema = COCO17,
        conf_threshold: float = 0.35,
        kpt_conf_threshold: float = 0.30,
        ankle_conf_threshold: float = 0.72,  # ankle often occluded — higher bar
        depth_patch: int = 3,           # sample a 3×3 patch and take nanmedian
        occluded_count: Optional[int] = None,  # None = max(2, K//3) — relative to schema
        device: Optional[torch.device] = None,
        use_filter: bool = False,        # ⚠ skiro-learnings: keep OFF by default
        filter_params: Optional[dict] = None,
    ) -> None:
        self.schema = schema
        self.K = schema.num_keypoints
        self.conf_threshold = conf_threshold
        self.kpt_conf_threshold = kpt_conf_threshold
        self.ankle_conf_threshold = ankle_conf_threshold
        self.depth_patch = max(1, int(depth_patch))

        # Determine ankle indices from schema name convention.
        ankle_set = {i for i, n in enumerate(schema.keypoints) if "ankle" in n.lower()}
        self._ankle_set = ankle_set

        # Per-joint depth threshold tensor: ankle uses ankle_conf_threshold,
        # all others use kpt_conf_threshold. Ankles are frequently occluded
        # behind the treadmill belt or walker frame — a stricter threshold
        # prevents low-confidence ankle depth from corrupting 3D lift.
        thresh_list = [
            ankle_conf_threshold if i in ankle_set else kpt_conf_threshold
            for i in range(self.K)
        ]
        self._kpt_thresh_t = torch.tensor(thresh_list, dtype=torch.float32,
                                          device=device or torch.device("cuda:0"))

        # Core mask: True for non-ankle joints. Occlusion validity check
        # uses only core (hip + knee) joints so a hidden ankle doesn't
        # immediately flip valid=False for a perfectly visible upper leg.
        self._core_mask_t = torch.tensor(
            [i not in ankle_set for i in range(self.K)],
            dtype=torch.bool, device=device or torch.device("cuda:0"),
        )

        # For COCO17 the default of 2 (12%) is strict enough, but for
        # lowlimb6 requiring ≥2 occlusions is 33% of all joints — too
        # harsh for a gait rehab workflow where a single leg occlusion
        # is common. Use max(2, K // 3) so:
        #   K=6  → 2 (same absolute bar, but better calibrated)
        #   K=17 → 5 (~30% — less chatty than 2/17 = 12%)
        # NOTE: occluded_count is checked against CORE joints only (hips+knees).
        core_k = self.K - len(ankle_set)
        self.occluded_count = (
            occluded_count if occluded_count is not None else max(2, core_k // 3)
        )
        self.device = device or torch.device("cuda:0")
        self.use_filter = use_filter
        self._filter: Optional[OneEuroFilter1D] = None
        if use_filter:
            fp = filter_params or {}
            self._filter = OneEuroFilter1D(
                (self.K, 3), device=self.device, **fp
            )

        # Simple 3D EMA — different from OneEuro (which we cannot use,
        # see skiro-learnings: 0/6 joints failure on lowlimb6). Vanilla
        # exponential moving average on the *3D output only* (never on
        # 2D keypoints — that path corrupts depth lookup). Alpha=0.7
        # means new=0.3*current + 0.7*prev: fast enough for walking
        # (5 Hz gait band) but cuts depth jitter visibly.
        self.ema_alpha: float = 0.7   # weight on PREVIOUS sample
        # EMA is a LOW-PASS only — it smooths gradual noise but does NOT
        # reject outlier teleportation (a 50cm jump just gets dampened to
        # 15cm = still wrong). For outliers use bone_length / velocity
        # constraints (constraints.py) which HARD-REJECT bad frames.
        self._ema_prev: Optional[torch.Tensor] = None  # (K, 3) on GPU

        # Sticky / hold-last-good state — when detection fails for a
        # short burst (occlusion, motion blur, single missed frame), we
        # keep emitting the LAST GOOD pose with valid=True so the
        # consumer (sagittal viewer + C++ control) sees a continuous
        # signal instead of zeros + flicker.
        # After max_sticky frames we give up and emit a true valid=False
        # so safety still kicks in if the subject is genuinely gone.
        # Default 5 frames @ ~85Hz ≈ 60ms hold — well inside C++ inner
        # control horizon and the AK60 mechanical bandwidth.
        self.max_sticky_frames: int = 5
        self._sticky_kpts_3d: Optional[torch.Tensor] = None
        self._sticky_kpts_2d: Optional[torch.Tensor] = None
        self._sticky_kpt_conf: Optional[torch.Tensor] = None
        self._sticky_box_conf: float = 0.0
        self._sticky_dir: float = 0.0
        self._sticky_age: int = 0

    @classmethod
    def from_schema_name(cls, name: str, **kwargs) -> "GpuPostprocessor":
        return cls(schema=get_schema(name), **kwargs)

    # ------------------------------------------------------------------
    def __call__(
        self,
        raw_output: torch.Tensor,
        depth_hw: Optional[torch.Tensor],
        lb_params: LetterboxParams,
        calibration: dict,
        stream: torch.cuda.Stream,
        ts_s: float,
    ) -> PoseResult:
        if raw_output.ndim != 3 or raw_output.shape[0] != 1:
            raise ValueError(
                f"raw_output must be (1,N,L), got {tuple(raw_output.shape)}"
            )
        last = raw_output.shape[2]
        expected_pose = 5 + self.K * 3
        expected_legacy = 6 + self.K * 3
        if last == expected_pose:
            kpt_offset = 5
        elif last == expected_legacy:
            kpt_offset = 6
        else:
            raise ValueError(
                f"raw_output last dim {last} doesn't match schema {self.schema.name} "
                f"(expected {expected_pose} pose or {expected_legacy} legacy)"
            )

        K = self.K
        with torch.cuda.stream(stream):
            det = raw_output[0]  # (N, last)
            conf = det[:, 4]
            # Use GPU argmax (no D2H here). gather best row directly.
            best = torch.argmax(conf)
            box_conf_t = conf[best]

            # Stage 1 — do ALL GPU work first, no .item()/.cpu() calls.
            # We collect every scalar we'll need into a single tensor and
            # sync ONCE at the end. TRT 10.x makes per-stream syncs more
            # expensive, so going from 4 syncs (old) → 1 sync (new) is
            # the dominant post-stage speedup.

            # Gather best row's keypoints (still on GPU)
            kpts = det[best, kpt_offset : kpt_offset + K * 3].view(K, 3)
            xy_letter = kpts[:, :2]
            kp_conf = kpts[:, 2]

            # Un-letterbox
            xy_src = xy_letter.clone()
            xy_src[:, 0] = (xy_src[:, 0] - lb_params.pad_x) / lb_params.scale
            xy_src[:, 1] = (xy_src[:, 1] - lb_params.pad_y) / lb_params.scale
            xy_src[:, 0].clamp_(0, lb_params.src_w - 1)
            xy_src[:, 1].clamp_(0, lb_params.src_h - 1)

            # 3D lift (vectorized — see _lift_to_3d_v); returns invalid as GPU tensor
            if depth_hw is not None:
                kpts_3d, invalid_ratio_t = self._lift_to_3d_v(
                    xy_src, depth_hw, calibration, kp_conf
                )
            else:
                kpts_3d = torch.zeros((K, 3), device=self.device)
                kpts_3d[:, :2] = xy_src
                invalid_ratio_t = torch.tensor(1.0, device=self.device)

            # Occlusion count on GPU — core joints only (hips + knees).
            # Ankles are frequently occluded and should not flip the validity
            # flag when the upper-leg chain is cleanly visible.
            num_low_conf_t = (kp_conf[self._core_mask_t] < self.kpt_conf_threshold).sum().float()

            # === Single D2H sync — combine 3 scalars into 1 transfer ===
            scalars = torch.stack([box_conf_t, num_low_conf_t, invalid_ratio_t]).cpu()
            box_conf, num_low_conf_f, invalid_ratio = scalars.tolist()
            num_low_conf = int(num_low_conf_f)

            # Branch on box_conf (CPU side) — only reached after one sync
            if box_conf < self.conf_threshold:
                return self._maybe_sticky(box_conf)

            occluded = num_low_conf >= self.occluded_count
            if occluded:
                # Don't clobber EMA / sticky on a transient occlusion —
                # _maybe_sticky decides whether to hold or fail-stop.
                return self._maybe_sticky(box_conf)

            if self._filter is not None:
                kpts_3d = self._filter(kpts_3d, ts_s)

            # 3D EMA — applied AFTER occlusion / conf gate so a bad frame
            # doesn't pollute the running average. Bootstrap with current
            # sample if there's no prev (first valid frame after reset).
            if self._ema_prev is None:
                self._ema_prev = kpts_3d.clone()
            else:
                kpts_3d = (
                    (1.0 - self.ema_alpha) * kpts_3d
                    + self.ema_alpha * self._ema_prev
                )
                self._ema_prev = kpts_3d.clone()

            # Update sticky state — this is the new "last good" anchor.
            self._sticky_kpts_3d = kpts_3d.clone()
            self._sticky_kpts_2d = xy_src.clone()
            self._sticky_kpt_conf = kp_conf.clone()
            self._sticky_box_conf = box_conf
            self._sticky_dir = invalid_ratio
            self._sticky_age = 0

            return PoseResult(
                kpts_2d_px=xy_src,
                kpts_3d_m=kpts_3d,
                kpt_conf=kp_conf,
                box_conf=box_conf,
                valid=True,
                depth_invalid_ratio=invalid_ratio,
            )

    # ------------------------------------------------------------------
    def _maybe_sticky(self, box_conf: float) -> PoseResult:
        """Detection failed (low conf or too many occluded joints).

        If we still have a recent 'last good' pose within max_sticky_frames,
        emit it as VALID so the consumer sees continuity. Otherwise emit
        a real INVALID so safety (C++ watchdog → pretension) kicks in.
        """
        K = self.K
        if (self._sticky_kpts_3d is not None
                and self._sticky_age < self.max_sticky_frames):
            self._sticky_age += 1
            return PoseResult(
                kpts_2d_px=self._sticky_kpts_2d,
                kpts_3d_m=self._sticky_kpts_3d,
                kpt_conf=self._sticky_kpt_conf,
                box_conf=self._sticky_box_conf,
                valid=True,
                depth_invalid_ratio=self._sticky_dir,
            )
        # Sticky budget exhausted — give up and reset so a future
        # recovered detection doesn't blend with the abandoned pose.
        self._ema_prev = None
        self._sticky_kpts_3d = None
        self._sticky_kpts_2d = None
        self._sticky_kpt_conf = None
        self._sticky_age = 0
        zeros2 = torch.zeros((K, 2), device=self.device)
        zeros3 = torch.zeros((K, 3), device=self.device)
        zerosc = torch.zeros((K,), device=self.device)
        return PoseResult(
            kpts_2d_px=zeros2, kpts_3d_m=zeros3, kpt_conf=zerosc,
            box_conf=box_conf, valid=False, depth_invalid_ratio=1.0,
        )

    # ------------------------------------------------------------------
    def _lift_to_3d_v(
        self,
        xy: torch.Tensor,
        depth_hw: torch.Tensor,
        calib: dict,
        kp_conf: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized 3D lift — single GPU op for all keypoints+patches.

        Replaces the old Python double-loop (9 GPU launches for 3×3 patch)
        with a single advanced-indexing op (1 launch). Also defers the
        invalid_ratio .item() — caller does ONE combined D2H sync.

        Returns:
            (xyz, invalid_ratio_gpu) — xyz is (K, 3), invalid_ratio is
            a 0-d GPU tensor that the caller stacks with other scalars
            into a single .cpu() transfer.
        """
        H, W = depth_hw.shape
        r = self.depth_patch // 2
        u = xy[:, 0].clamp(r, W - r - 1).long()  # (K,)
        v = xy[:, 1].clamp(r, H - r - 1).long()  # (K,)

        # Cache patch-offset indices: depth_patch is fixed at init, so
        # we build (P², ) offsets once and reuse across frames.
        psize = 2 * r + 1
        cached_dv = getattr(self, "_dv_cache", None)
        if cached_dv is None or cached_dv.numel() != psize * psize:
            offs = torch.arange(-r, r + 1, device=self.device)
            dv_grid, du_grid = torch.meshgrid(offs, offs, indexing="ij")
            self._dv_cache = dv_grid.reshape(-1)  # (P²,)
            self._du_cache = du_grid.reshape(-1)  # (P²,)

        # Broadcast: (P², 1) + (1, K) → (P², K) — single advanced index
        vv = v.unsqueeze(0) + self._dv_cache.unsqueeze(1)
        uu = u.unsqueeze(0) + self._du_cache.unsqueeze(1)
        patches = depth_hw[vv, uu]  # (P², K) — 1 GPU launch (was 9)

        valid = torch.isfinite(patches) & (patches > 0.0)
        patches = torch.where(valid, patches, torch.full_like(patches, float("nan")))
        z = torch.nanmedian(patches, dim=0).values  # (K,)
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

        # invalid ratio stays on GPU — caller stacks + syncs
        invalid_ratio_t = (~valid).float().mean()

        z = torch.where(kp_conf >= self._kpt_thresh_t, z, torch.zeros_like(z))

        fx, fy = calib["fx"], calib["fy"]
        cx, cy = calib["cx"], calib["cy"]
        x3 = (xy[:, 0] - cx) * z / fx
        y3 = (xy[:, 1] - cy) * z / fy
        xyz_cam = torch.stack([x3, y3, z], dim=1)  # (K, 3) camera frame

        # If the bridge provided an IMU-based rotation, project into a
        # gravity-aligned world frame. This matches mainline Method B
        # (ZEDIMUWorldFrame) so sagittal view stays vertical even when
        # the camera is tilted ~32° on the walker mount.
        R = calib.get("R_world_from_cam")
        if R is not None:
            # R: (3, 3) on same device; xyz_cam @ R.T == (R @ xyz.T).T
            xyz_world = xyz_cam @ R.t()
            return xyz_world, invalid_ratio_t
        return xyz_cam, invalid_ratio_t

"""GPU preprocess — letterbox resize + normalize on CUDA.

Input  : (H, W, 3) uint8 on CUDA (RGB, from ZEDGpuBridge)
Output : (1, 3, 640, 640) float16/float32 on CUDA, [0, 1]

We use ``torch.nn.functional.interpolate`` (bilinear) for resize and a
fused constant-pad + scale. All ops respect the caller's stream via
``torch.cuda.stream(...)`` context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass
class LetterboxParams:
    """Back-projection info to undo letterbox on output keypoints."""

    scale: float
    pad_x: int
    pad_y: int
    src_h: int
    src_w: int
    dst: int


class GpuPreprocessor:
    """Letterbox + normalize on GPU, bound to a specific stream."""

    def __init__(
        self,
        imgsz: int = 640,
        dtype: torch.dtype = torch.float16,
        device: torch.device | None = None,
        pad_value: float = 114.0 / 255.0,
    ) -> None:
        self.imgsz = imgsz
        self.dtype = dtype
        self.device = device or torch.device("cuda:0")
        self.pad_value = pad_value
        # pre-allocated output buffer so downstream (TRT) can zero-copy bind
        self.out = torch.empty(
            (1, 3, imgsz, imgsz), dtype=dtype, device=self.device
        )
        self.last_params: LetterboxParams | None = None

    def __call__(
        self, rgb_u8: torch.Tensor, stream: torch.cuda.Stream
    ) -> Tuple[torch.Tensor, LetterboxParams]:
        """Run preprocess on ``stream``.

        Parameters
        ----------
        rgb_u8 : torch.Tensor, (H, W, 3) RGB or (H, W, 4) BGRA uint8 on CUDA
                 L1 (2026-05-06): now accepts BGRA 4ch directly from ZED.
                 GPU does BGR→RGB channel select + alpha drop here, replacing
                 the costly CPU `np.ascontiguousarray(bgra[:,:,:3][:,:,::-1])`
                 in zed_gpu_bridge (was ~4ms, now sub-microsecond on GPU).
        stream : CUDA stream to execute on

        Returns
        -------
        (out_chw, params) — out_chw is self.out (ready for TRT), params
        is the letterbox metadata for post-processing.
        """
        if rgb_u8.device != self.device:
            raise ValueError(
                f"expected device={self.device}, got {rgb_u8.device}"
            )
        if rgb_u8.ndim != 3 or rgb_u8.shape[2] not in (3, 4):
            raise ValueError(
                f"expected (H,W,3) RGB or (H,W,4) BGRA, got {tuple(rgb_u8.shape)}"
            )

        # L1 (2026-05-06): BGRA → RGB on GPU. Use fancy indexing [2,1,0] which
        # selects channels R(=2), G(=1), B(=0) — alpha auto-drops + reverse in
        # one op + returns contiguous tensor (no negative-stride view bugs).
        # Tested: `flip(-1)` produced a negative-stride view that broke after
        # subsequent permute().to(dtype) on PyTorch 2.10 — output channels were
        # corrupted. fancy indexing avoids that path entirely.
        if rgb_u8.shape[2] == 4:
            rgb_u8 = rgb_u8[..., [2, 1, 0]]

        H, W, _ = rgb_u8.shape
        scale = min(self.imgsz / H, self.imgsz / W)
        nh = int(round(H * scale))
        nw = int(round(W * scale))
        pad_x = (self.imgsz - nw) // 2
        pad_y = (self.imgsz - nh) // 2

        with torch.cuda.stream(stream):
            # uint8 (H,W,3) -> float (1,3,H,W) normalized
            img = rgb_u8.permute(2, 0, 1).unsqueeze(0).to(self.dtype) / 255.0
            # bilinear resize to (1,3,nh,nw)
            resized = F.interpolate(
                img, size=(nh, nw), mode="bilinear", align_corners=False
            )
            # pad to imgsz with gray (114/255)
            self.out.fill_(self.pad_value)
            self.out[:, :, pad_y : pad_y + nh, pad_x : pad_x + nw] = resized

        self.last_params = LetterboxParams(
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            src_h=H,
            src_w=W,
            dst=self.imgsz,
        )
        return self.out, self.last_params

    def undo_letterbox(
        self, xy: torch.Tensor, params: LetterboxParams | None = None
    ) -> torch.Tensor:
        """Map keypoint pixels from 640×640 letterboxed back to original."""
        params = params or self.last_params
        if params is None:
            raise RuntimeError("No letterbox params yet; call __call__ first")
        xy = xy.clone()
        xy[..., 0] = (xy[..., 0] - params.pad_x) / params.scale
        xy[..., 1] = (xy[..., 1] - params.pad_y) / params.scale
        xy[..., 0].clamp_(0, params.src_w - 1)
        xy[..., 1].clamp_(0, params.src_h - 1)
        return xy

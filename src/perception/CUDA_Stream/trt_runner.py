"""TRTRunner — TensorRT engine loader + execute_async_v3 wrapper.

Uses the TensorRT 10.x Python API. Input/output tensors are allocated as
torch tensors on CUDA to simplify interop with the rest of the pipeline
(stream sharing, D2H to pinned host). We never call
``torch.cuda.synchronize()`` from inside this module — the caller owns
synchronization via events on :class:`StreamManager`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import tensorrt as trt
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "tensorrt is required. On JetPack use the system package "
        "(/usr/lib/python3/dist-packages/tensorrt)."
    ) from exc

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("torch is required for TRTRunner") from exc


LOGGER = logging.getLogger(__name__)


def _load_engine_bytes(path: Path) -> Tuple[bytes, Dict[str, Any]]:
    """Return (raw_trt_plan, ultralytics_metadata_or_empty).

    Ultralytics' ``model.export(format='engine')`` prepends a header:
        [4 bytes uint32 LE: json_len][json_len bytes UTF-8 JSON][TRT plan]

    Standard engines (from trtexec, custom scripts) start with the TRT
    plan directly. We auto-detect by peeking the first 4 bytes and
    trying to parse the candidate JSON — if either check fails we fall
    back to treating the whole file as a raw TRT plan.

    The JSON contains useful fields like ``kpt_shape`` (e.g. ``[6, 3]``
    for a lowlimb6 model) which inspect_engine surfaces to help the
    caller pick the right schema.
    """
    raw = path.read_bytes()
    if len(raw) < 4:
        return raw, {}
    meta_len = int.from_bytes(raw[:4], byteorder="little")
    if 4 + meta_len >= len(raw) or not (0 < meta_len < 1_000_000):
        return raw, {}
    meta_bytes = raw[4 : 4 + meta_len]
    try:
        meta = json.loads(meta_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return raw, {}
    if not isinstance(meta, dict):
        return raw, {}
    # Confirmed Ultralytics format — strip the header.
    plan = raw[4 + meta_len :]
    LOGGER.info(
        "detected ultralytics engine header (%d-byte JSON), stripping. "
        "metadata keys: %s",
        meta_len,
        sorted(meta.keys()),
    )
    return plan, meta

_TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
}
# BOOL and BF16 are optional depending on TRT version
if hasattr(trt.DataType, "BOOL"):
    _TRT_TO_TORCH[trt.DataType.BOOL] = torch.bool
if hasattr(trt.DataType, "BF16"):
    _TRT_TO_TORCH[trt.DataType.BF16] = torch.bfloat16


@dataclass
class TensorBinding:
    name: str
    shape: Tuple[int, ...]
    dtype: torch.dtype
    is_input: bool
    tensor: torch.Tensor  # pre-allocated CUDA buffer


class TRTRunner:
    """Load a .engine and run it on an explicit CUDA stream."""

    def __init__(
        self,
        engine_path: str | Path,
        device: Optional[torch.device] = None,
        trt_logger_level: int = trt.Logger.WARNING,
    ) -> None:
        self.device = device or torch.device("cuda:0")
        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TRT engine not found: {self.engine_path}")

        self._trt_logger = trt.Logger(trt_logger_level)
        self._runtime = trt.Runtime(self._trt_logger)
        engine_bytes, self.metadata = _load_engine_bytes(self.engine_path)
        self.engine = self._runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine {self.engine_path}")
        self.context = self.engine.create_execution_context()

        # resolve bindings — TRT 10 uses tensor API (get_tensor_name / mode)
        self.bindings: Dict[str, TensorBinding] = {}
        self._input_names: List[str] = []
        self._output_names: List[str] = []

        # Diagnostic counter (Codex R11, 2026-05-06) — set_tensor_address is
        # expensive and triggers TRT 10.x graph re-validation. Caller path
        # bind_input_address has a per-pointer cache so the steady-state count
        # should plateau quickly. If it grows linearly with frame_count,
        # cache is broken → that's the +6.6ms vs trtexec.
        self._set_address_count = 0

        n_io = self.engine.num_io_tensors
        for i in range(n_io):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            is_input = mode == trt.TensorIOMode.INPUT
            shape = tuple(self.engine.get_tensor_shape(name))
            trt_dtype = self.engine.get_tensor_dtype(name)
            torch_dtype = _TRT_TO_TORCH.get(trt_dtype)
            if torch_dtype is None:
                raise RuntimeError(
                    f"Unsupported TRT dtype {trt_dtype} for tensor {name}"
                )

            # dynamic shapes (any -1) must be resolved by caller before infer
            safe_shape = tuple(max(d, 1) for d in shape)
            tensor = torch.empty(safe_shape, dtype=torch_dtype, device=self.device)
            self.bindings[name] = TensorBinding(
                name=name,
                shape=shape,
                dtype=torch_dtype,
                is_input=is_input,
                tensor=tensor,
            )
            (self._input_names if is_input else self._output_names).append(name)
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
            self._set_address_count += 1

        LOGGER.info(
            "TRTRunner loaded %s : inputs=%s outputs=%s",
            self.engine_path.name,
            self._input_names,
            self._output_names,
        )

    # ------------------------------------------------------------------
    # Shape / dtype helpers
    # ------------------------------------------------------------------
    @property
    def input_names(self) -> List[str]:
        return list(self._input_names)

    @property
    def output_names(self) -> List[str]:
        return list(self._output_names)

    def set_input_shape(self, name: str, shape: Tuple[int, ...]) -> None:
        """For dynamic-shape engines. Re-allocates the input buffer if needed."""
        binding = self.bindings[name]
        if not binding.is_input:
            raise ValueError(f"{name} is not an input")
        if tuple(binding.tensor.shape) != tuple(shape):
            binding.tensor = torch.empty(
                shape, dtype=binding.dtype, device=self.device
            )
        self.context.set_input_shape(name, shape)
        self.context.set_tensor_address(name, int(binding.tensor.data_ptr()))
        binding.shape = tuple(shape)

    def set_input(self, name: str, tensor: torch.Tensor) -> None:
        """Copy data into the pre-bound input buffer (async H2D on caller stream)."""
        binding = self.bindings[name]
        if not binding.is_input:
            raise ValueError(f"{name} is not an input")
        if tensor.device != self.device:
            binding.tensor.copy_(tensor, non_blocking=True)
        else:
            # same-device shallow write (no copy if caller owns the address)
            if tensor.data_ptr() != binding.tensor.data_ptr():
                binding.tensor.copy_(tensor, non_blocking=True)

    def bind_input_address(self, name: str, tensor: torch.Tensor) -> None:
        """Directly bind caller-owned tensor as input (zero-copy).

        TRT 10.x: ``set_tensor_address`` mutates context state and adds
        non-trivial launch overhead (~0.5-1ms per call). We cache the
        last bound pointer per tensor name and only call when it changes.
        For preproc output that's reused frame-to-frame this means we
        only call set_tensor_address once across the entire run.
        """
        binding = self.bindings[name]
        if not binding.is_input:
            raise ValueError(f"{name} is not an input")
        if tensor.shape != binding.tensor.shape or tensor.dtype != binding.dtype:
            raise ValueError(
                f"{name} expects shape={binding.tensor.shape} dtype={binding.dtype}, "
                f"got shape={tensor.shape} dtype={tensor.dtype}"
            )
        new_ptr = int(tensor.data_ptr())
        cached_ptrs = getattr(self, "_bound_ptrs", None)
        cached_refs = getattr(self, "_bound_tensor_refs", None)
        if cached_ptrs is None:
            cached_ptrs = {}
            cached_refs = {}
            self._bound_ptrs = cached_ptrs
            self._bound_tensor_refs = cached_refs
        # CRITICAL: keep a strong reference to the tensor whose pointer
        # we cached. Without this, the caller's tensor can be GC'd, the
        # underlying CUDA buffer freed, and TRT will read freed memory
        # on the next inference → cudaErrorIllegalAddress on shutdown.
        # If the SAME ptr value gets re-used by a new allocation we'd
        # silently keep the stale binding — so we also re-set on tensor
        # identity change (cached_refs.get(name) is not tensor).
        if cached_ptrs.get(name) != new_ptr or cached_refs.get(name) is not tensor:
            self.context.set_tensor_address(name, new_ptr)
            self._set_address_count += 1
            cached_ptrs[name] = new_ptr
            cached_refs[name] = tensor

    def get_output(self, name: str) -> torch.Tensor:
        return self.bindings[name].tensor

    @property
    def set_address_count(self) -> int:
        """Codex R11: cumulative set_tensor_address calls. Should plateau
        at (n_inputs + n_outputs) after warmup. Linear growth = cache miss."""
        return self._set_address_count

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def infer_async(self, stream_ptr: int) -> None:
        """Kick off inference on the given CUDA stream handle (int).

        Does NOT synchronize. Caller records an event on their stream and
        propagates the event to downstream consumers.
        """
        ok = self.context.execute_async_v3(stream_handle=stream_ptr)
        if not ok:
            raise RuntimeError(
                f"TRT execute_async_v3 failed for {self.engine_path.name}"
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def describe(self) -> Dict[str, Dict]:
        out = {}
        for name, b in self.bindings.items():
            out[name] = {
                "role": "input" if b.is_input else "output",
                "shape": tuple(b.tensor.shape),
                "dtype": str(b.dtype),
            }
        return out

    def __del__(self) -> None:
        # torch tensors + TRT context/engine clean themselves up — explicit
        # drops keep teardown order deterministic during interpreter shutdown.
        try:
            self.bindings.clear()
            self.context = None  # type: ignore[assignment]
            self.engine = None  # type: ignore[assignment]
            self._runtime = None  # type: ignore[assignment]
        except Exception:  # pragma: no cover
            pass


def warmup(runner: TRTRunner, stream_ptr: int, iters: int = 10) -> None:
    """Run a few dummy inferences so TRT allocates workspace lazily."""
    for _ in range(iters):
        runner.infer_async(stream_ptr)
    # we still need a sync here because this is setup, not hot path
    torch.cuda.synchronize()

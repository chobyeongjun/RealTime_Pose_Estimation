"""Jetson 측정 전 smoke test — α + γ contract 검증.

사용법:
    cd ~/realtime-vision-control
    PYTHONPATH=src python3 scripts/jetson_smoke_test.py

기대 출력:
    === ALL CHECKS PASSED ===

paste indent 문제 회피 — file 로 git 받아 실행.
"""
from __future__ import annotations

import dataclasses


def main() -> int:
    from perception.CUDA_Stream.pipeline import (
        PipelineTick, FrameMeta, PipelineToken, StreamedPosePipeline
    )
    from perception.CUDA_Stream.gpu_postprocess import (
        PoseResult, GpuPostprocessor
    )
    from perception.CUDA_Stream.zed_gpu_bridge import ZEDGpuBridge
    from perception.CUDA_Stream.stream_manager import StreamBundle
    from perception.CUDA_Stream.tracer import FrameTrace, PipelineTracer

    # P1D1 contract — meta + trace fields
    pt_fields = [f.name for f in dataclasses.fields(PipelineTick)]
    assert "meta" in pt_fields, f"PipelineTick.meta missing: {pt_fields}"
    assert "trace" in pt_fields

    # P1D1 contract — FrameMeta 6 fields
    fm_fields = [f.name for f in dataclasses.fields(FrameMeta)]
    expected_fm = {"frame_id", "ts_ns", "bridge_start_ns", "ready_ns", "pickup_ns", "capture_ms"}
    assert set(fm_fields) == expected_fm, f"FrameMeta fields mismatch: {fm_fields}"

    # P1D1 contract — PipelineToken 14 fields + γ post_scalar_host
    tok_fields = [f.name for f in dataclasses.fields(PipelineToken)]
    assert "post_scalar_host" in tok_fields
    assert "raw_snapshot" in tok_fields
    assert "pre_done" in tok_fields
    assert "post_done" in tok_fields

    # P5D1 contract — PoseResult async fields
    pr_fields = [f.name for f in dataclasses.fields(PoseResult)]
    assert "scalar_host" in pr_fields
    assert "post_async_pending" in pr_fields
    assert "num_low_conf_t" in pr_fields

    # P1D2 contract — StreamBundle per-token API
    assert hasattr(StreamBundle, "record_event")
    assert hasattr(StreamBundle, "wait_event")
    assert hasattr(StreamBundle, "make_event")
    assert hasattr(StreamBundle, "record_done")  # 기존 유지
    assert hasattr(StreamBundle, "wait_for")     # 기존 유지

    # P5D1 contract — GpuPostprocessor.finalize_async
    assert hasattr(GpuPostprocessor, "finalize_async")

    # P2D1 contract — PipelineTracer token-aware API
    assert hasattr(PipelineTracer, "begin_token")
    assert hasattr(PipelineTracer, "mark_start_token")
    assert hasattr(PipelineTracer, "mark_end_token")
    assert hasattr(PipelineTracer, "end_token")
    assert hasattr(PipelineTracer, "set_result_meta_token")

    # γ contract — ZEDGpuBridge.zed_cuda_interop arg
    import inspect
    bridge_sig = inspect.signature(ZEDGpuBridge.__init__)
    assert "zed_cuda_interop" in bridge_sig.parameters

    print("=== ALL CHECKS PASSED ===")
    print("    Phase 1+4+5+gamma + critical fixes 모두 완료. Jetson 측정 가능.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

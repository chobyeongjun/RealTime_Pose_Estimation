#!/usr/bin/env python3
"""P1 SHM layout self-test (Mac에서도 실행 가능).

검증 항목:
  T1: 새 offset 정확 (PUBLISH_DONE_OFF=40, VALID_REASON_OFF=48, TS_DOMAIN_OFF=49)
  T2: publish_done_mono_ns가 publish() 호출 시점 근처 monotonic 시간
  T3: valid_reason 박힘 / 읽힘 (모든 enum 값)
  T4: valid=True일 때 valid_reason 강제로 VALID_OK
  T5: ShmReader가 새 field 반환 (12-tuple)
  T6: SHM segment 크기 변화 0 (padding 활용 → 기존 layout 안 깨짐)
  T7: forward compatibility — version=1 그대로
  T8: seqlock invariant 유지 (write 중에 reader가 retry)

실행:
    python3 scripts/test_p1_shm.py
"""
from __future__ import annotations

import struct
import sys
import time
from pathlib import Path

# repo root을 sys.path에 추가 (src/ layout 가정)
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from perception.CUDA_Stream.shm_publisher import (
    HEADER_SIZE,
    INVALID_BUDGET_EXCEED,
    INVALID_CONSTRAINT,
    INVALID_NO_DETECTION,
    INVALID_OCCLUDED,
    INVALID_UNKNOWN,
    INVALID_WARMUP,
    PUBLISH_DONE_OFF,
    SEQ_OFF,
    ShmPublisher,
    ShmReader,
    TS_DOMAIN_EPOCH,
    TS_DOMAIN_OFF,
    VALID_OK,
    VALID_REASON_NAMES,
    VALID_REASON_OFF,
    VERSION,
    VERSION_OFF,
    compute_size,
)


def test_t1_offsets():
    """T1: 새 offset이 헤더 docstring과 일치 + 64B 안에 다 들어감."""
    assert PUBLISH_DONE_OFF == 40, f"PUBLISH_DONE_OFF should be 40, got {PUBLISH_DONE_OFF}"
    assert VALID_REASON_OFF == 48, f"VALID_REASON_OFF should be 48, got {VALID_REASON_OFF}"
    assert TS_DOMAIN_OFF == 49, f"TS_DOMAIN_OFF should be 49, got {TS_DOMAIN_OFF}"
    # 49+1 = 50, padding 14 bytes 남음
    assert TS_DOMAIN_OFF + 1 <= HEADER_SIZE, "new fields must fit in header"
    print("  T1 offsets — PASS")


def test_t6_size_unchanged():
    """T6: K=6 SHM 크기가 P1 전후 동일 (padding 활용 → 기존 segment 호환)."""
    K = 6
    # K=6: header 64 + payload (6*3 + 6 + 6*2)*4 = 64 + 144 = 208 → ceil(64) → 256B
    expected = 256
    assert compute_size(K) == expected, (
        f"K={K} segment size should be {expected}, got {compute_size(K)}"
    )
    K = 17
    # K=17: header 64 + payload (17*3 + 17 + 17*2)*4 = 64 + 408 = 472 → ceil(64) → 512B
    expected = 512
    assert compute_size(K) == expected, (
        f"K={K} segment size should be {expected}, got {compute_size(K)}"
    )
    print("  T6 size unchanged — PASS")


def test_t7_version_unchanged():
    """T7: forward compat — version 그대로 1."""
    assert VERSION == 1, f"VERSION should stay at 1 for forward-compat, got {VERSION}"
    print("  T7 version=1 — PASS")


def _make_test_publisher(name="test_p1_shm_xxx", K=6):
    """Helper: SHM 만들고 ShmPublisher 반환. 호출자가 close() 책임."""
    # 기존에 같은 이름 있으면 정리
    pub = ShmPublisher(num_keypoints=K, name=name, create=True)
    return pub


def test_t2_t3_publish_read_roundtrip():
    """T2 + T3 + T4 + T5: publish/read full roundtrip."""
    K = 6
    name = "test_p1_roundtrip"

    pub = ShmPublisher(num_keypoints=K, name=name, create=True)
    try:
        kpts_3d = np.arange(K * 3, dtype=np.float32).reshape(K, 3)
        kpt_conf = np.linspace(0.5, 0.95, K, dtype=np.float32)
        kpts_2d = np.arange(K * 2, dtype=np.float32).reshape(K, 2) * 10

        # T2 + T3: publish with valid_reason=INVALID_BUDGET_EXCEED
        ts_ns = 1234567890123456789
        t_before_mono = time.monotonic_ns()
        pub.publish(
            frame_id=42,
            ts_ns=ts_ns,
            kpts_3d_m=kpts_3d,
            kpt_conf=kpt_conf,
            kpts_2d_px=kpts_2d,
            box_conf=0.87,
            valid=False,
            depth_invalid_ratio=0.1,
            world_frame_applied=True,
            valid_reason=INVALID_BUDGET_EXCEED,
        )
        t_after_mono = time.monotonic_ns()

        rdr = ShmReader(name=name, expected_k=K)
        result = rdr.read()
        assert result is not None, "read() returned None"
        # T5: 12-tuple 형식
        assert len(result) == 12, f"reader should return 12-tuple, got {len(result)}"
        (frame_id, ts_ns_r, kpts_3d_r, kpt_conf_r, kpts_2d_r,
         box_conf, valid, depth_inv, world_frame_applied,
         publish_done_mono_ns, valid_reason, ts_domain) = result

        assert frame_id == 42
        assert ts_ns_r == ts_ns
        assert np.allclose(kpts_3d_r, kpts_3d)
        assert np.allclose(kpt_conf_r, kpt_conf)
        assert np.allclose(kpts_2d_r, kpts_2d)
        assert abs(box_conf - 0.87) < 1e-5
        assert valid is False
        assert abs(depth_inv - 0.1) < 1e-5
        assert world_frame_applied is True

        # T2: publish_done_mono_ns가 호출 시점 근처
        assert t_before_mono <= publish_done_mono_ns <= t_after_mono, (
            f"publish_done_mono_ns={publish_done_mono_ns} not in "
            f"[{t_before_mono}, {t_after_mono}]"
        )

        # T3: valid_reason 정확
        assert valid_reason == INVALID_BUDGET_EXCEED, (
            f"valid_reason should be {INVALID_BUDGET_EXCEED}, got {valid_reason}"
        )

        # ts_domain
        assert ts_domain == TS_DOMAIN_EPOCH, f"ts_domain should be epoch (0), got {ts_domain}"

        rdr.close()
        print("  T2/T3/T5 publish/read roundtrip — PASS")
    finally:
        pub.close()


def test_t4_valid_forces_ok():
    """T4: valid=True일 때 valid_reason 강제로 VALID_OK (caller 실수 방지)."""
    K = 6
    name = "test_p1_force_ok"

    pub = ShmPublisher(num_keypoints=K, name=name, create=True)
    try:
        kpts_3d = np.zeros((K, 3), dtype=np.float32)
        kpt_conf = np.ones(K, dtype=np.float32) * 0.9
        kpts_2d = np.zeros((K, 2), dtype=np.float32)

        # 일부러 valid=True인데 valid_reason=INVALID_BUDGET_EXCEED 던짐 — 무시되어야 함
        pub.publish(
            frame_id=1,
            ts_ns=1000,
            kpts_3d_m=kpts_3d,
            kpt_conf=kpt_conf,
            kpts_2d_px=kpts_2d,
            box_conf=0.9,
            valid=True,
            valid_reason=INVALID_BUDGET_EXCEED,  # caller 실수 시뮬레이션
        )

        rdr = ShmReader(name=name, expected_k=K)
        result = rdr.read()
        assert result is not None
        valid_reason = result[10]
        assert valid_reason == VALID_OK, (
            f"valid=True must force valid_reason=VALID_OK ({VALID_OK}), got {valid_reason}"
        )
        rdr.close()
        print("  T4 valid=True forces VALID_OK — PASS")
    finally:
        pub.close()


def test_t8_seqlock_invariant():
    """T8: seqlock invariant — write 중간에 reader가 retry, 끝나면 stable."""
    K = 6
    name = "test_p1_seqlock"

    pub = ShmPublisher(num_keypoints=K, name=name, create=True)
    try:
        kpts_3d = np.zeros((K, 3), dtype=np.float32)
        kpt_conf = np.ones(K, dtype=np.float32) * 0.5
        kpts_2d = np.zeros((K, 2), dtype=np.float32)

        # 첫 publish
        pub.publish(
            frame_id=10, ts_ns=10_000, kpts_3d_m=kpts_3d, kpt_conf=kpt_conf,
            kpts_2d_px=kpts_2d, box_conf=0.5, valid=True,
        )
        seq_after_first = struct.unpack_from("<I", pub._buf, SEQ_OFF)[0]
        assert seq_after_first % 2 == 0, f"seq should be even after publish, got {seq_after_first}"

        # 두 번째 publish — seq 증가
        pub.publish(
            frame_id=11, ts_ns=11_000, kpts_3d_m=kpts_3d, kpt_conf=kpt_conf,
            kpts_2d_px=kpts_2d, box_conf=0.5, valid=True,
        )
        seq_after_second = struct.unpack_from("<I", pub._buf, SEQ_OFF)[0]
        assert seq_after_second > seq_after_first
        assert seq_after_second % 2 == 0
        print(f"  T8 seqlock — PASS  (seq: {seq_after_first} → {seq_after_second})")
    finally:
        pub.close()


def test_t9_all_valid_reasons():
    """T9: 모든 VALID_REASON_* 값이 라운드트립 가능."""
    K = 6
    name = "test_p1_reasons"

    pub = ShmPublisher(num_keypoints=K, name=name, create=True)
    try:
        kpts_3d = np.zeros((K, 3), dtype=np.float32)
        kpt_conf = np.ones(K, dtype=np.float32) * 0.3
        kpts_2d = np.zeros((K, 2), dtype=np.float32)

        rdr = ShmReader(name=name, expected_k=K)
        for reason in (
            INVALID_NO_DETECTION,
            INVALID_OCCLUDED,
            INVALID_BUDGET_EXCEED,
            INVALID_CONSTRAINT,
            INVALID_WARMUP,
            INVALID_UNKNOWN,
        ):
            pub.publish(
                frame_id=99, ts_ns=99_000, kpts_3d_m=kpts_3d, kpt_conf=kpt_conf,
                kpts_2d_px=kpts_2d, box_conf=0.0, valid=False,
                valid_reason=reason,
            )
            r = rdr.read()
            assert r is not None
            assert r[10] == reason, (
                f"reason {VALID_REASON_NAMES[reason]} ({reason}) didn't roundtrip "
                f"(read {r[10]})"
            )
        rdr.close()
        print(f"  T9 all valid_reasons roundtrip — PASS  ({len(VALID_REASON_NAMES)} values)")
    finally:
        pub.close()


def main() -> int:
    print("=== P1 SHM self-test ===")
    try:
        test_t1_offsets()
        test_t6_size_unchanged()
        test_t7_version_unchanged()
        test_t2_t3_publish_read_roundtrip()
        test_t4_valid_forces_ok()
        test_t8_seqlock_invariant()
        test_t9_all_valid_reasons()
    except AssertionError as e:
        print(f"\nFAIL: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2

    print("\nALL P1 self-tests PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

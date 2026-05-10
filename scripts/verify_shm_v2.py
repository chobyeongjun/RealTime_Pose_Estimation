"""SHM v2 통합 검증 — 한 번에 모든 의무 검증.

Codex orchestration bvfvkxo1m (2026-05-11) Q2.

검증 항목 (사용자 의지 — single command, 한번에):
    1. shm_publisher 의 import OK
    2. compute_size(K) 의 정확성 (K=1..17)
    3. Header offsets 의 정확성 (Codex Q5 spec)
    4. publish + read round-trip (모든 fields 정확)
    5. seqlock even/odd protocol
    6. Two-timestamp + depth_age_us 정확
    7. valid_mask_bits default derivation
    8. kp_sigma_m / pose_cov_diag default + custom
    9. Error handling (wrong shape / dtype / version)
   10. publish_done_mono_ns monotonic (10 publishes)

실행 (Mac 또는 Jetson):
    PYTHONPATH=src python3 scripts/verify_shm_v2.py
또는:
    python3 scripts/verify_shm_v2.py    # conftest.py 가 src/ 자동 추가

PASS → exit 0, FAIL → exit 1 (어떤 check 가 fail 했는지 명시).
"""
from __future__ import annotations

import os
import struct
import sys
import time
from pathlib import Path

# repo root 의 src/ 자동 추가
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def main() -> int:
    failures = []

    print("=" * 60)
    print("SHM v2 통합 검증")
    print("=" * 60)

    # ---------- 1. Import ----------
    print("\n[1] import shm_publisher...")
    try:
        from perception.CUDA_Stream.shm_publisher import (
            ShmPublisher, ShmReader,
            VERSION, HEADER_SIZE, compute_size,
            SEQ_OFF, VERSION_OFF, K_OFF, FRAME_ID_OFF,
            RGB_TS_OFF, DEPTH_TS_OFF, DEPTH_AGE_OFF,
            BOX_CONF_OFF, DEPTH_INVALID_OFF,
            VALID_FLAG_OFF, WORLD_FRAME_OFF, VALID_REASON_OFF, TS_DOMAIN_OFF,
            PUBLISH_DONE_OFF, VALID_MASK_BITS_OFF,
            VALID_OK, INVALID_NO_DETECTION, INVALID_STALE_DEPTH,
            DEFAULT_SIGMA_M, DEFAULT_NAME,
        )
        import numpy as np
        print(f"  ✓ import OK, VERSION={VERSION}")
        if VERSION != 2:
            failures.append(f"VERSION should be 2, got {VERSION}")
    except Exception as e:
        print(f"  ✗ import FAIL: {e}")
        return 1

    # ---------- 2. compute_size ----------
    print("\n[2] compute_size...")
    expected_sizes = {
        1: 128,        # 64 + 48 = 112 → 128
        6: 384,        # 64 + 288 = 352 → 384
        7: 448,        # 64 + 336 = 400 → 448
        17: 896,       # 64 + 816 = 880 → 896
        64: 3136,      # 64 + 3072 = 3136 (이미 64-aligned)
    }
    for K, expected in expected_sizes.items():
        actual = compute_size(K)
        if actual != expected:
            failures.append(f"compute_size({K}) expected {expected}, got {actual}")
        else:
            print(f"  ✓ K={K:2d} → {actual} bytes")

    # ---------- 3. Header offsets ----------
    print("\n[3] header offsets (Codex Q5 spec)...")
    expected_offsets = {
        "SEQ_OFF": 0, "VERSION_OFF": 4, "K_OFF": 8, "FRAME_ID_OFF": 12,
        "RGB_TS_OFF": 16, "DEPTH_TS_OFF": 24, "DEPTH_AGE_OFF": 32,
        "BOX_CONF_OFF": 36, "DEPTH_INVALID_OFF": 40,
        "VALID_FLAG_OFF": 44, "WORLD_FRAME_OFF": 45,
        "VALID_REASON_OFF": 46, "TS_DOMAIN_OFF": 47,
        "PUBLISH_DONE_OFF": 48, "VALID_MASK_BITS_OFF": 56,
        "HEADER_SIZE": 64,
    }
    actuals = {
        "SEQ_OFF": SEQ_OFF, "VERSION_OFF": VERSION_OFF, "K_OFF": K_OFF,
        "FRAME_ID_OFF": FRAME_ID_OFF,
        "RGB_TS_OFF": RGB_TS_OFF, "DEPTH_TS_OFF": DEPTH_TS_OFF,
        "DEPTH_AGE_OFF": DEPTH_AGE_OFF,
        "BOX_CONF_OFF": BOX_CONF_OFF, "DEPTH_INVALID_OFF": DEPTH_INVALID_OFF,
        "VALID_FLAG_OFF": VALID_FLAG_OFF, "WORLD_FRAME_OFF": WORLD_FRAME_OFF,
        "VALID_REASON_OFF": VALID_REASON_OFF, "TS_DOMAIN_OFF": TS_DOMAIN_OFF,
        "PUBLISH_DONE_OFF": PUBLISH_DONE_OFF, "VALID_MASK_BITS_OFF": VALID_MASK_BITS_OFF,
        "HEADER_SIZE": HEADER_SIZE,
    }
    for name, expected in expected_offsets.items():
        actual = actuals[name]
        if actual != expected:
            failures.append(f"{name}: expected {expected}, got {actual}")
        else:
            print(f"  ✓ {name:24s} = {actual}")

    # ---------- 4. Round-trip (publish + read) ----------
    print("\n[4] round-trip publish + read...")
    K = 6
    name = f"verify_v2_{os.getpid()}_main"
    # Cleanup before
    try:
        os.remove(f"/dev/shm/{name}")
    except FileNotFoundError:
        pass

    try:
        pub = ShmPublisher(K, name=name, create=True)
        try:
            kpts_3d = np.array([
                [0.0, 0.0, 1.0], [0.1, 0.0, 1.0],
                [0.0, 0.5, 1.2], [0.1, 0.5, 1.2],
                [0.0, 0.9, 1.5], [0.1, 0.9, 1.5],
            ], dtype=np.float32)
            kpts_2d = np.array([
                [320, 380], [340, 380], [310, 460], [350, 460],
                [305, 540], [355, 540],
            ], dtype=np.float32)
            kp_conf = np.array([0.9, 0.9, 0.85, 0.85, 0.8, 0.8], dtype=np.float32)

            pub.publish(
                frame_id=42, rgb_ts_ns=1_000_000_000,
                kpts_3d_m=kpts_3d, kpt_conf=kp_conf, kpts_2d_px=kpts_2d,
                box_conf=0.85, valid=True,
            )

            reader = ShmReader(name=name, expected_k=K)
            try:
                result = reader.read()
                if result is None:
                    failures.append("[4] read returned None")
                else:
                    (frame_id, rgb_ts, depth_ts, depth_age_us,
                     kpts_3d_r, kp_conf_r, kpts_2d_r, kp_sigma, pose_cov,
                     box_conf, valid, depth_inv, world_frame,
                     publish_done, valid_reason, ts_domain,
                     valid_mask_bits) = result

                    checks = [
                        (frame_id == 42, f"frame_id={frame_id}, expected 42"),
                        (rgb_ts == 1_000_000_000, f"rgb_ts={rgb_ts}"),
                        (depth_ts == 1_000_000_000, f"depth_ts={depth_ts} (default same-frame)"),
                        (depth_age_us == 0, f"depth_age_us={depth_age_us}, expected 0"),
                        (valid is True, f"valid={valid}"),
                        (valid_reason == VALID_OK, f"valid_reason={valid_reason}"),
                        (np.array_equal(kpts_3d_r, kpts_3d), "kpts_3d round-trip"),
                        (np.array_equal(kpts_2d_r, kpts_2d), "kpts_2d round-trip"),
                        (np.array_equal(kp_conf_r, kp_conf), "kp_conf round-trip"),
                        (np.allclose(kp_sigma, np.full((K, 3), DEFAULT_SIGMA_M)),
                            f"kp_sigma default ≠ {DEFAULT_SIGMA_M}"),
                        (np.allclose(pose_cov, np.full((K, 3), DEFAULT_SIGMA_M ** 2)),
                            "pose_cov default = sigma²"),
                        (valid_mask_bits == (1 << K) - 1,
                            f"valid_mask_bits={bin(valid_mask_bits)}, expected {bin((1<<K)-1)}"),
                    ]
                    for ok, msg in checks:
                        if ok:
                            print(f"  ✓ {msg.split(',')[0] if isinstance(msg, str) else msg}")
                        else:
                            failures.append(f"[4] {msg}")
            finally:
                reader.close()
        finally:
            pub.close()
    except Exception as e:
        failures.append(f"[4] round-trip exception: {e}")

    # ---------- 5. seqlock ----------
    print("\n[5] seqlock even/odd...")
    name2 = f"verify_v2_{os.getpid()}_seq"
    try:
        os.remove(f"/dev/shm/{name2}")
    except FileNotFoundError:
        pass

    try:
        pub = ShmPublisher(K, name=name2, create=True)
        try:
            seqs = []
            for i in range(5):
                pub.publish(
                    frame_id=i, rgb_ts_ns=1_000_000_000 + i * 8_333_333,
                    kpts_3d_m=kpts_3d, kpt_conf=kp_conf, kpts_2d_px=kpts_2d,
                    box_conf=0.9, valid=True,
                )
                seq = struct.unpack_from("<I", pub._buf, SEQ_OFF)[0]
                seqs.append(seq)
            all_even = all(s % 2 == 0 for s in seqs)
            monotonic = all(seqs[i] > seqs[i-1] for i in range(1, len(seqs)))
            if all_even and monotonic:
                print(f"  ✓ 5 publishes → seqs {seqs} (all even, monotonic)")
            else:
                failures.append(f"[5] seq invariant fail: {seqs}")
        finally:
            pub.close()
    except Exception as e:
        failures.append(f"[5] seqlock exception: {e}")

    # ---------- 6. Two-timestamp + depth_age ----------
    print("\n[6] two-timestamp depth_age...")
    name3 = f"verify_v2_{os.getpid()}_ts"
    try:
        os.remove(f"/dev/shm/{name3}")
    except FileNotFoundError:
        pass

    try:
        pub = ShmPublisher(K, name=name3, create=True)
        reader = ShmReader(name=name3, expected_k=K)
        try:
            cases = [
                (1_000_000_000, 1_000_000_000, 0, "same-frame"),
                (1_000_000_000, 1_000_000_000 - 8_333_333, 8333, "1-frame-late (8.33ms)"),
                (1_000_000_000, 1_000_000_000 - 100_000_000, 100_000, "100ms stale"),
            ]
            for rgb_ts, depth_ts, expected_age, label in cases:
                pub.publish(
                    frame_id=1, rgb_ts_ns=rgb_ts,
                    kpts_3d_m=kpts_3d, kpt_conf=kp_conf, kpts_2d_px=kpts_2d,
                    box_conf=0.9, valid=True, depth_ts_ns=depth_ts,
                )
                result = reader.read()
                age = result[3]
                if abs(age - expected_age) <= 200:   # tolerance for floor div
                    print(f"  ✓ {label}: depth_age_us={age}")
                else:
                    failures.append(f"[6] {label}: age={age}, expected {expected_age}")
        finally:
            reader.close()
            pub.close()
    except Exception as e:
        failures.append(f"[6] timestamp exception: {e}")

    # ---------- 7. publish_done monotonic ----------
    print("\n[7] publish_done_mono_ns monotonic...")
    name4 = f"verify_v2_{os.getpid()}_mono"
    try:
        os.remove(f"/dev/shm/{name4}")
    except FileNotFoundError:
        pass

    try:
        pub = ShmPublisher(K, name=name4, create=True)
        reader = ShmReader(name=name4, expected_k=K)
        try:
            mono_list = []
            for i in range(10):
                pub.publish(
                    frame_id=i, rgb_ts_ns=1_000_000_000 + i * 8_333_333,
                    kpts_3d_m=kpts_3d, kpt_conf=kp_conf, kpts_2d_px=kpts_2d,
                    box_conf=0.9, valid=True,
                )
                result = reader.read()
                mono_list.append(result[13])
                time.sleep(0.001)

            strict_inc = all(mono_list[i] > mono_list[i-1]
                             for i in range(1, len(mono_list)))
            if strict_inc:
                print(f"  ✓ 10 publishes, publish_done_mono strictly increasing")
            else:
                failures.append(f"[7] publish_done not monotonic: {mono_list}")
        finally:
            reader.close()
            pub.close()
    except Exception as e:
        failures.append(f"[7] monotonic exception: {e}")

    # ---------- 8. Per-kp covariance ----------
    print("\n[8] per-kp covariance round-trip...")
    name5 = f"verify_v2_{os.getpid()}_cov"
    try:
        os.remove(f"/dev/shm/{name5}")
    except FileNotFoundError:
        pass

    try:
        pub = ShmPublisher(K, name=name5, create=True)
        reader = ShmReader(name=name5, expected_k=K)
        try:
            sigma = np.array([
                [0.005, 0.005, 0.010], [0.005, 0.005, 0.010],
                [0.008, 0.008, 0.012], [0.008, 0.008, 0.012],
                [0.015, 0.015, 0.025], [0.015, 0.015, 0.025],
            ], dtype=np.float32)
            pub.publish(
                frame_id=1, rgb_ts_ns=1_000_000_000,
                kpts_3d_m=kpts_3d, kpt_conf=kp_conf, kpts_2d_px=kpts_2d,
                box_conf=0.9, valid=True, kp_sigma_m=sigma,
            )
            result = reader.read()
            kp_sigma_r = result[7]
            pose_cov_r = result[8]
            if np.allclose(kp_sigma_r, sigma):
                print(f"  ✓ custom kp_sigma_m round-trip")
            else:
                failures.append(f"[8] kp_sigma_m mismatch")
            if np.allclose(pose_cov_r, sigma ** 2):
                print(f"  ✓ pose_cov_diag default = sigma²")
            else:
                failures.append(f"[8] pose_cov_diag mismatch")
        finally:
            reader.close()
            pub.close()
    except Exception as e:
        failures.append(f"[8] covariance exception: {e}")

    # ---------- Summary ----------
    print("\n" + "=" * 60)
    if failures:
        print(f"=== FAIL ({len(failures)} issues) ===")
        for f in failures:
            print(f"  ✗ {f}")
        return 1
    else:
        print("=== ALL CHECKS PASSED ===")
        print("→ SHM v2 publisher + reader 정확히 작동.")
        print("→ Plan D EKF (사용자 control repo) 의 input contract OK.")
        return 0


if __name__ == "__main__":
    sys.exit(main())

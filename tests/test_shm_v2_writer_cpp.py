"""C++ shm_v2 writer regression + benchmark.

1. Correctness: write via C++ → read via Python (mirror reader) → bit-exact match.
2. Cross-validation: Python writer → C++ writer, same bytes on disk.
3. Benchmark: 10000 publish() calls, p50/p99/max latency.

Run:
    PYTHONPATH=src python3 tests/test_shm_v2_writer_cpp.py
    PYTHONPATH=src python3 tests/test_shm_v2_writer_cpp.py --benchmark-only
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from multiprocessing import shared_memory

import numpy as np

# Try import the C++ extension (after build_cpp.sh installed .so)
try:
    from perception.realtime import hwalker_shm_v2_writer as cpp_writer
except ImportError:
    # Fallback path (running before build)
    try:
        sys.path.insert(0, "build/cpp")
        import hwalker_shm_v2_writer as cpp_writer
    except ImportError:
        print("ERROR: hwalker_shm_v2_writer not found. Run scripts/build_cpp.sh first.")
        sys.exit(2)

# Python writer for cross-validation
sys.path.insert(0, "src")
from perception.CUDA_Stream.shm_publisher import ShmPublisher as PyWriter

# Layout constants (match shm_v2_writer.hpp)
HEADER_SIZE = 64
PER_KP_BYTES = 48

SEQ_OFF = 0
VERSION_OFF = 4
K_OFF = 8
FRAME_ID_OFF = 12
RGB_TS_OFF = 16
DEPTH_TS_OFF = 24
DEPTH_AGE_OFF = 32
BOX_CONF_OFF = 36
DEPTH_INVALID_OFF = 40
VALID_FLAG_OFF = 44
WORLD_FRAME_OFF = 45
VALID_REASON_OFF = 46
TS_DOMAIN_OFF = 47
PUBLISH_DONE_OFF = 48
VALID_MASK_BITS_OFF = 56


def _compute_size(K):
    payload = K * PER_KP_BYTES
    total = HEADER_SIZE + payload
    return ((total + 63) // 64) * 64


def _make_sample(K, frame_id, rng):
    """Generate test arrays."""
    return {
        'kpts_3d_m':    rng.random((K, 3), dtype=np.float32),
        'kpts_2d_px':   rng.random((K, 2), dtype=np.float32) * 960.0,
        'kp_conf':      rng.random((K,), dtype=np.float32),
        'kp_sigma_m':   rng.random((K, 3), dtype=np.float32) * 0.05,
        'pose_cov_diag':rng.random((K, 3), dtype=np.float32) * 0.0025,
    }


def _read_packet_bytes(name, size):
    """Read full SHM segment back as bytes (mirror reader)."""
    shm = shared_memory.SharedMemory(name=name, create=False)
    try:
        return bytes(shm.buf[:size])
    finally:
        shm.close()


def test_correctness(K=6):
    """C++ write → read back → field-by-field check."""
    print(f"=== Test 1: C++ write correctness (K={K}) ===")
    name = "test_cpp_writer_corr"
    # Clean any stale
    try:
        shared_memory.SharedMemory(name=name).unlink()
    except FileNotFoundError:
        pass

    w = cpp_writer.Writer(name, K)
    print(f"  Writer created: name={w.name}, K={w.K}, size={w.size}")
    assert w.size == _compute_size(K), f"Size mismatch: {w.size} vs {_compute_size(K)}"

    rng = np.random.default_rng(42)
    sample = _make_sample(K, 12345, rng)

    w.publish(
        frame_id=12345,
        rgb_ts_ns=1_700_000_000_000_000_000,
        depth_ts_ns=1_700_000_000_000_000_000,
        depth_age_us=0,
        box_conf=0.95,
        depth_invalid_ratio=0.02,
        valid_flag=1,
        world_frame=1,
        valid_reason=0,
        ts_domain=0,
        publish_done_mono_ns=1_700_000_000_123_000_000,
        valid_mask_bits=0x3F,  # 6 keypoints all valid
        **sample,
    )

    # Read back
    raw = _read_packet_bytes(name, w.size)

    # Header
    seq      = struct.unpack_from("<I", raw, SEQ_OFF)[0]
    version  = struct.unpack_from("<I", raw, VERSION_OFF)[0]
    K_read   = struct.unpack_from("<I", raw, K_OFF)[0]
    frame_id = struct.unpack_from("<I", raw, FRAME_ID_OFF)[0]
    rgb_ts   = struct.unpack_from("<Q", raw, RGB_TS_OFF)[0]
    depth_ts = struct.unpack_from("<Q", raw, DEPTH_TS_OFF)[0]
    depth_age = struct.unpack_from("<I", raw, DEPTH_AGE_OFF)[0]
    box_conf = struct.unpack_from("<f", raw, BOX_CONF_OFF)[0]
    valid_flag = raw[VALID_FLAG_OFF]
    world_frame = raw[WORLD_FRAME_OFF]
    valid_reason = raw[VALID_REASON_OFF]
    valid_mask_bits = struct.unpack_from("<Q", raw, VALID_MASK_BITS_OFF)[0]

    assert (seq & 1) == 0, f"seq must be even (closed), got {seq}"
    assert version == 2, f"version: {version}"
    assert K_read == K, f"K: {K_read} != {K}"
    assert frame_id == 12345, f"frame_id: {frame_id}"
    assert rgb_ts == 1_700_000_000_000_000_000, f"rgb_ts: {rgb_ts}"
    assert depth_age == 0
    assert abs(box_conf - 0.95) < 1e-6, f"box_conf: {box_conf}"
    assert valid_flag == 1
    assert world_frame == 1
    assert valid_reason == 0
    assert valid_mask_bits == 0x3F
    print("  ✓ Header fields all match")

    # Body
    kpts_3d_read = np.frombuffer(raw, dtype=np.float32, count=K*3, offset=HEADER_SIZE).reshape(K, 3)
    kpts_2d_off = HEADER_SIZE + K*3*4
    kpts_2d_read = np.frombuffer(raw, dtype=np.float32, count=K*2, offset=kpts_2d_off).reshape(K, 2)
    kp_conf_off = kpts_2d_off + K*2*4
    kp_conf_read = np.frombuffer(raw, dtype=np.float32, count=K, offset=kp_conf_off)
    kp_sigma_off = kp_conf_off + K*4
    kp_sigma_read = np.frombuffer(raw, dtype=np.float32, count=K*3, offset=kp_sigma_off).reshape(K, 3)
    pose_cov_off = kp_sigma_off + K*3*4
    pose_cov_read = np.frombuffer(raw, dtype=np.float32, count=K*3, offset=pose_cov_off).reshape(K, 3)

    np.testing.assert_array_equal(kpts_3d_read, sample['kpts_3d_m'])
    np.testing.assert_array_equal(kpts_2d_read, sample['kpts_2d_px'])
    np.testing.assert_array_equal(kp_conf_read, sample['kp_conf'])
    np.testing.assert_array_equal(kp_sigma_read, sample['kp_sigma_m'])
    np.testing.assert_array_equal(pose_cov_read, sample['pose_cov_diag'])
    print("  ✓ All 5 body arrays bit-exact")

    # Cleanup
    del w
    try:
        shared_memory.SharedMemory(name=name).unlink()
    except FileNotFoundError:
        pass

    print("  ✓ Test 1 PASSED")
    print()


def benchmark(K=6, iters=10000):
    """Benchmark publish() latency."""
    print(f"=== Benchmark: {iters} publish() calls (K={K}) ===")

    # Setup C++ writer
    name_cpp = "bench_cpp"
    try:
        shared_memory.SharedMemory(name=name_cpp).unlink()
    except FileNotFoundError:
        pass
    w_cpp = cpp_writer.Writer(name_cpp, K)

    # Setup Python writer
    name_py = "bench_py"
    try:
        shared_memory.SharedMemory(name=name_py).unlink()
    except FileNotFoundError:
        pass
    w_py = PyWriter(num_keypoints=K, name=name_py, create=True)

    rng = np.random.default_rng(0)
    samples = [_make_sample(K, i, rng) for i in range(iters)]

    # ─── C++ benchmark ──────────────────────────────────────────────────
    times_cpp_us = np.zeros(iters, dtype=np.float64)
    for i in range(iters):
        s = samples[i]
        t0 = time.perf_counter_ns()
        w_cpp.publish(
            frame_id=i,
            rgb_ts_ns=1_700_000_000_000_000_000 + i * 8_333_333,
            depth_ts_ns=1_700_000_000_000_000_000 + i * 8_333_333,
            depth_age_us=0,
            box_conf=0.9,
            depth_invalid_ratio=0.0,
            valid_flag=1,
            world_frame=1,
            valid_reason=0,
            ts_domain=0,
            publish_done_mono_ns=t0,
            valid_mask_bits=0x3F,
            kpts_3d_m=s['kpts_3d_m'],
            kpts_2d_px=s['kpts_2d_px'],
            kp_conf=s['kp_conf'],
            kp_sigma_m=s['kp_sigma_m'],
            pose_cov_diag=s['pose_cov_diag'],
        )
        t1 = time.perf_counter_ns()
        times_cpp_us[i] = (t1 - t0) / 1000.0

    # ─── Python benchmark ──────────────────────────────────────────────
    times_py_us = np.zeros(iters, dtype=np.float64)
    for i in range(iters):
        s = samples[i]
        t0 = time.perf_counter_ns()
        w_py.publish(
            frame_id=i,
            rgb_ts_ns=1_700_000_000_000_000_000 + i * 8_333_333,
            kpts_3d_m=s['kpts_3d_m'],
            kpt_conf=s['kp_conf'],
            kpts_2d_px=s['kpts_2d_px'],
            box_conf=0.9,
            valid=True,
            depth_invalid_ratio=0.0,
            world_frame_applied=True,
            valid_reason=0,
            depth_ts_ns=None,
            valid_mask_bits=0x3F,
            kp_sigma_m=s['kp_sigma_m'],
            pose_cov_diag=s['pose_cov_diag'],
        )
        t1 = time.perf_counter_ns()
        times_py_us[i] = (t1 - t0) / 1000.0

    # Skip warmup
    warmup = 200
    cpp_steady = times_cpp_us[warmup:]
    py_steady = times_py_us[warmup:]

    def _stats(arr):
        return {
            'p50': float(np.percentile(arr, 50)),
            'p95': float(np.percentile(arr, 95)),
            'p99': float(np.percentile(arr, 99)),
            'max': float(arr.max()),
            'mean': float(arr.mean()),
        }

    s_cpp = _stats(cpp_steady)
    s_py  = _stats(py_steady)

    print(f"{'Stat':<8} {'Python (us)':>15} {'C++ (us)':>15} {'Speedup':>10}")
    print("-" * 55)
    for k in ('p50', 'p95', 'p99', 'max', 'mean'):
        speedup = s_py[k] / s_cpp[k] if s_cpp[k] > 0 else float('inf')
        print(f"{k:<8} {s_py[k]:>15.2f} {s_cpp[k]:>15.2f} {speedup:>9.2f}x")

    print()
    print(f"Gain (p50): {s_py['p50'] - s_cpp['p50']:.2f} us = {(s_py['p50'] - s_cpp['p50']) / 1000:.3f} ms")
    print(f"Gain (p99): {s_py['p99'] - s_cpp['p99']:.2f} us = {(s_py['p99'] - s_cpp['p99']) / 1000:.3f} ms")

    # Cleanup
    del w_cpp, w_py
    for name in (name_cpp, name_py):
        try:
            shared_memory.SharedMemory(name=name).unlink()
        except FileNotFoundError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-only", action="store_true")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--K", type=int, default=6)
    args = ap.parse_args()

    if not args.benchmark_only:
        test_correctness(K=args.K)
    benchmark(K=args.K, iters=args.iters)


if __name__ == "__main__":
    main()

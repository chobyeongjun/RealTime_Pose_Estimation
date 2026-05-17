"""Quick NPZ inspection — invalid frame burst length distribution.

Usage:
    python3 scripts/analyze_invalid_bursts.py recordings/walking_*/walking_*.npz
"""
import sys
import numpy as np

if len(sys.argv) < 2:
    print("Usage: analyze_invalid_bursts.py <npz_path>")
    sys.exit(1)

z = np.load(sys.argv[1])
v = z["valid"].astype(int)
runs = np.split(v, np.where(np.diff(v))[0] + 1)
inv = [len(r) for r in runs if r[0] == 0]
fr  = [len(r) for r in runs if r[0] == 1]

print(f"frames total: {v.size}, valid: {v.sum()} ({v.mean()*100:.1f}%)")
print()
print("INVALID bursts (frame is invalid):")
print(f"  count        : {len(inv)}")
if inv:
    print(f"  median length: {np.median(inv):.1f} frames")
    print(f"  max length   : {max(inv)} frames")
    print(f"  > 3 frames (depth_hold 한계 초과): {sum(1 for x in inv if x > 3)}")
    print(f"  > 10 frames                    : {sum(1 for x in inv if x > 10)}")
    print(f"  > 30 frames (0.5초 이상)         : {sum(1 for x in inv if x > 30)}")
print()
print("VALID runs (continuous tracking):")
print(f"  count        : {len(fr)}")
if fr:
    print(f"  median length: {np.median(fr):.1f} frames")
    print(f"  max length   : {max(fr)} frames")

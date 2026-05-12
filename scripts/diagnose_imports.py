#!/usr/bin/env python3
"""Diagnose pipeline_main.py 의 import chain — 진정 어느 module 의무 missing.

사용 (Jetson):
    PYTHONPATH=src:src/perception/benchmarks python3 scripts/diagnose_imports.py
"""
import sys
import os
import subprocess


def main():
    print("=" * 60)
    print("  Pipeline import diagnostic")
    print("=" * 60)
    print(f"  cwd: {os.getcwd()}")
    print(f"  PYTHONPATH: {os.environ.get('PYTHONPATH', '(not set)')}")
    print(f"  sys.path[:5]: {sys.path[:5]}")
    print()

    print("── Module imports ──")
    for module in ['postprocess_accel', 'zed_camera', 'trt_pose_engine']:
        try:
            mod = __import__(module)
            mod_file = getattr(mod, '__file__', '(builtin)')
            print(f"  OK   {module:<20} → {mod_file}")
        except ImportError as e:
            print(f"  FAIL {module:<20} → {e}")
        except Exception as e:
            print(f"  ERR  {module:<20} → {type(e).__name__}: {e}")

    print()
    print("── Find missing modules on Jetson ──")
    for name in ['postprocess_accel', 'trt_pose_engine']:
        try:
            result = subprocess.run(
                ['find', '/home/chobb0', '-name', f'{name}*'],
                capture_output=True, text=True, timeout=10,
            )
            lines = [l for l in result.stdout.strip().split('\n') if l]
            if lines:
                print(f"  {name}:")
                for line in lines[:5]:
                    print(f"    {line}")
            else:
                print(f"  {name}: NOT FOUND anywhere on /home/chobb0")
        except Exception as e:
            print(f"  {name}: find error: {e}")

    print()
    print("── Current benchmarks dir ──")
    bench_dir = 'src/perception/benchmarks'
    if os.path.isdir(bench_dir):
        for f in sorted(os.listdir(bench_dir)):
            full = os.path.join(bench_dir, f)
            size = os.path.getsize(full) if os.path.isfile(full) else 0
            print(f"  {f:<30} {size:>8} bytes")
    else:
        print(f"  ({bench_dir} does not exist)")


if __name__ == "__main__":
    main()

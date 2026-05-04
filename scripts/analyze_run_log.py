#!/usr/bin/env python3
"""Analyze CUDA_Stream run log — decompose true_e2e_ms latency.

Usage:
    python3 scripts/analyze_run_log.py path/to/run.log

Reads [SLOW] frame entries (which now include the bridge/queue/pipeline
decomposition) and produces:
  1. Distribution per stage (p50/p95/p99/max)
  2. Top N slowest frames (full breakdown)
  3. Correlation hint between stages
  4. Final [STATS] summary echo

Designed for debug visibility on the Plan v3 phase 0 instrumentation.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# [SLOW] frame 128  true_e2e=24.3 ms (e2e=15.7)
#   decomp: bridge_proc=3.1  queue_wait=2.4  pipeline_proc=18.8  zed_lag=0.0 ms
#   capture : grab=2.7  ret_rgb=2.6  getdata_rgb=12.0 ...
#   pipeline: pre=1.2  inf=7.1  post=7.2  constraint=2.1
SLOW_HEAD = re.compile(
    r"\[SLOW\] frame (?P<fid>\d+)\s+true_e2e=(?P<true>[\d.]+) ms \(e2e=(?P<e2e>[\d.]+)\)"
)
DECOMP = re.compile(
    r"decomp:\s+bridge_proc=(?P<bp>[\d.]+)\s+queue_wait=(?P<qw>[\d.]+)\s+"
    r"pipeline_proc=(?P<pp>[\d.]+)\s+zed_lag=(?P<zl>[\d.]+) ms"
)
CAPTURE = re.compile(
    r"capture\s*:\s*grab=(?P<grab>[\d.]+)\s+ret_rgb=(?P<rrgb>[\d.]+)\s+"
    r"getdata_rgb=(?P<grgb>[\d.]+)\s+pinned_rgb=(?P<prgb>[\d.]+)\s+"
    r"ret_depth=(?P<rdep>[\d.]+)\s+getdata_depth=(?P<gdep>[\d.]+)"
)
PIPELINE = re.compile(
    r"pipeline:\s+pre=(?P<pre>[\d.]+)\s+inf=(?P<inf>[\d.]+)\s+"
    r"post=(?P<post>[\d.]+)(?:\s+constraint=(?P<con>[\d.]+))?"
)


def parse_log(path: Path) -> List[Dict[str, float]]:
    """Stream log, group every (head, decomp, capture, pipeline) into one frame dict."""
    frames: List[Dict[str, float]] = []
    pending: Dict[str, float] = {}
    expecting = "head"  # head → decomp → capture → pipeline
    for line in path.read_text(errors="replace").splitlines():
        if expecting == "head":
            m = SLOW_HEAD.search(line)
            if m:
                pending = {
                    "frame_id": float(m.group("fid")),
                    "true_e2e": float(m.group("true")),
                    "e2e": float(m.group("e2e")),
                }
                expecting = "decomp"
            continue
        if expecting == "decomp":
            m = DECOMP.search(line)
            if m:
                pending.update({
                    "bridge_proc": float(m.group("bp")),
                    "queue_wait": float(m.group("qw")),
                    "pipeline_proc": float(m.group("pp")),
                    "zed_lag": float(m.group("zl")),
                })
                expecting = "capture"
            elif SLOW_HEAD.search(line):
                # decomp not present (older format) — skip to head
                expecting = "head"
            continue
        if expecting == "capture":
            m = CAPTURE.search(line)
            if m:
                pending.update({
                    "grab": float(m.group("grab")),
                    "ret_rgb": float(m.group("rrgb")),
                    "getdata_rgb": float(m.group("grgb")),
                    "pinned_rgb": float(m.group("prgb")),
                    "ret_depth": float(m.group("rdep")),
                    "getdata_depth": float(m.group("gdep")),
                })
                expecting = "pipeline"
            continue
        if expecting == "pipeline":
            m = PIPELINE.search(line)
            if m:
                pending.update({
                    "pre": float(m.group("pre")),
                    "inf": float(m.group("inf")),
                    "post": float(m.group("post")),
                    "constraint": float(m.group("con")) if m.group("con") else 0.0,
                })
                frames.append(pending.copy())
                expecting = "head"
            continue
    return frames


def percentiles(values: List[float]) -> Tuple[float, float, float, float, float]:
    if not values:
        return (float("nan"),) * 5
    s = sorted(values)
    n = len(s)
    def pct(p): return s[min(n - 1, int(n * p / 100))]
    return s[0], pct(50), pct(95), pct(99), s[-1]


def fmt_dist(name: str, values: List[float], width: int = 14) -> str:
    mn, p50, p95, p99, mx = percentiles(values)
    return f"  {name:<{width}}  min={mn:5.1f}  p50={p50:5.1f}  p95={p95:5.1f}  p99={p99:5.1f}  max={mx:5.1f} ms"


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: analyze_run_log.py <run.log>", file=sys.stderr)
        return 1
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"file not found: {path}", file=sys.stderr)
        return 1
    frames = parse_log(path)
    if not frames:
        print(f"no [SLOW] frames found in {path}", file=sys.stderr)
        print("(needs new format with decomp/constraint lines — re-run after deploying instrumentation)")
        return 2

    print(f"=== {path.name} — {len(frames)} SLOW frames parsed ===\n")

    print("Top-level latency:")
    print(fmt_dist("true_e2e", [f["true_e2e"] for f in frames]))
    print(fmt_dist("e2e (gpu)", [f["e2e"] for f in frames]))
    print()

    print("Decomposition (sum ≈ true_e2e):")
    print(fmt_dist("zed_lag", [f["zed_lag"] for f in frames]))
    print(fmt_dist("bridge_proc", [f["bridge_proc"] for f in frames]))
    print(fmt_dist("queue_wait", [f["queue_wait"] for f in frames]))
    print(fmt_dist("pipeline_proc", [f["pipeline_proc"] for f in frames]))
    print()

    print("Bridge thread breakdown:")
    print(fmt_dist("grab", [f["grab"] for f in frames]))
    print(fmt_dist("ret_rgb", [f["ret_rgb"] for f in frames]))
    print(fmt_dist("getdata_rgb", [f["getdata_rgb"] for f in frames]))
    print(fmt_dist("pinned_rgb", [f["pinned_rgb"] for f in frames]))
    print(fmt_dist("ret_depth", [f["ret_depth"] for f in frames]))
    print(fmt_dist("getdata_depth", [f["getdata_depth"] for f in frames]))
    print()

    print("Pipeline thread breakdown:")
    print(fmt_dist("pre", [f["pre"] for f in frames]))
    print(fmt_dist("inf", [f["inf"] for f in frames]))
    print(fmt_dist("post", [f["post"] for f in frames]))
    print(fmt_dist("constraint", [f["constraint"] for f in frames]))
    print()

    # Top 5 slowest by true_e2e
    print("Top 5 slowest frames (by true_e2e):")
    for f in sorted(frames, key=lambda x: -x["true_e2e"])[:5]:
        print(
            f"  frame {int(f['frame_id']):5d}  true_e2e={f['true_e2e']:5.1f}  "
            f"= bridge {f['bridge_proc']:5.1f} + queue {f['queue_wait']:5.1f} "
            f"+ pipeline {f['pipeline_proc']:5.1f} ({f['zed_lag']:.1f} zed_lag)"
        )
    print()

    # Identify dominant component per frame
    print("Dominant component for slow frames (>20ms true_e2e):")
    counts = {"bridge_proc": 0, "queue_wait": 0, "pipeline_proc": 0}
    slow_frames = [f for f in frames if f["true_e2e"] > 20.0]
    for f in slow_frames:
        dom = max(counts.keys(), key=lambda k: f[k])
        counts[dom] += 1
    total = max(len(slow_frames), 1)
    for k, v in counts.items():
        print(f"  {k:<14}  {v:4d} / {total}  ({v/total*100:5.1f}%)")
    print(f"\n  → {len(slow_frames)} / {len(frames)} parsed frames had true_e2e > 20ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Plan v7 (2026-05-07) — zed_lag 측정 결과 비교 도구.

각 Round 의 launch_clean.sh 출력에서 핵심 metric 만 뽑아 표로 출력.
파일 또는 stdin 으로 여러 run 받아 한 줄씩 비교.

사용법:
    # 단일 run 파싱
    sudo bash launch_clean.sh 20 2>&1 | python3 scripts/parse_zedlag_results.py -

    # 여러 파일 비교
    python3 scripts/parse_zedlag_results.py \\
        --label "AUTO"     run_auto.log \\
        --label "MAN-5ms"  run_5ms.log \\
        --label "MAN-8ms"  run_8ms.log
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


METRIC_PATTERNS = {
    "hz":          re.compile(r"→\s+([\d.]+)\s+Hz"),
    "e2e_p99":     re.compile(r"e2e\s+\(gpu only\).*p50/95/99\s*=\s*[\d.]+/[\d.]+/([\d.]+)"),
    "true_e2e_p50": re.compile(r"true_e2e.*p50/95/99\s*=\s*([\d.]+)/"),
    "true_e2e_p99": re.compile(r"true_e2e.*p50/95/99\s*=\s*[\d.]+/[\d.]+/([\d.]+)"),
    "bridge_p50":  re.compile(r"bridge_proc=([\d.]+)/"),
    "bridge_p99":  re.compile(r"bridge_proc=[\d.]+/([\d.]+)"),
    "queue_p50":   re.compile(r"queue_wait=([\d.]+)/"),
    "pipeline_p50": re.compile(r"pipeline_proc=([\d.]+)/"),
    "decomp_sum":  re.compile(r"decomposition mean.*=\s+([\d.]+)\s+ms"),
    "true_e2e_mean": re.compile(r"true_e2e mean=([\d.]+)"),
    "violation_pct": re.compile(r"HARD LIMIT 20 ms.*true_e2e basis.*\((\d+\.\d+)%\)"),
    "graph_replay": re.compile(r"\[diag\].*replay=(\d+)"),
    "graph_eager":  re.compile(r"\[diag\].*eager=(\d+)"),
    "set_address":  re.compile(r"\[diag\].*set_address=(\d+)"),
}


@dataclass
class RunResult:
    label: str
    hz: Optional[float] = None
    e2e_p99: Optional[float] = None
    true_e2e_p50: Optional[float] = None
    true_e2e_p99: Optional[float] = None
    true_e2e_mean: Optional[float] = None
    bridge_p50: Optional[float] = None
    bridge_p99: Optional[float] = None
    queue_p50: Optional[float] = None
    pipeline_p50: Optional[float] = None
    decomp_sum: Optional[float] = None
    zed_lag: Optional[float] = None  # 계산: true_e2e_mean - decomp_sum
    violation_pct: Optional[float] = None
    graph_replay: Optional[int] = None
    graph_eager: Optional[int] = None
    set_address: Optional[int] = None

    def fmt(self, key: str) -> str:
        v = getattr(self, key)
        if v is None:
            return "—"
        if isinstance(v, int):
            return f"{v:>5d}"
        return f"{v:>6.2f}"


def parse(text: str, label: str) -> RunResult:
    r = RunResult(label=label)
    for key, pat in METRIC_PATTERNS.items():
        m = pat.search(text)
        if m:
            v = m.group(1)
            try:
                if key in ("graph_replay", "graph_eager", "set_address"):
                    setattr(r, key, int(v))
                else:
                    setattr(r, key, float(v))
            except ValueError:
                pass
    if r.true_e2e_mean is not None and r.decomp_sum is not None:
        r.zed_lag = r.true_e2e_mean - r.decomp_sum
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+",
                    help="Log files (or '-' for stdin). Use --label N times for naming.")
    ap.add_argument("--label", action="append", default=[],
                    help="Labels matching positional inputs.")
    args = ap.parse_args()

    runs: list[RunResult] = []
    for i, inp in enumerate(args.inputs):
        label = args.label[i] if i < len(args.label) else inp
        if inp == "-":
            text = sys.stdin.read()
        else:
            text = Path(inp).read_text()
        runs.append(parse(text, label))

    # Header
    cols = [
        ("label", 14),
        ("hz", 8),
        ("e2e_p99", 9),
        ("true_e2e_p50", 12),
        ("true_e2e_p99", 12),
        ("zed_lag", 9),
        ("bridge_p50", 11),
        ("bridge_p99", 11),
        ("pipeline_p50", 12),
        ("violation_pct", 9),
        ("graph_replay", 13),
        ("graph_eager", 11),
    ]
    print(" | ".join(f"{name:>{w}}" for name, w in cols))
    print("-+-".join("-" * w for _, w in cols))
    for r in runs:
        row = []
        for name, w in cols:
            if name == "label":
                row.append(f"{r.label[:w]:>{w}}")
            else:
                v = r.fmt(name)
                row.append(f"{v:>{w}}")
        print(" | ".join(row))

    return 0


if __name__ == "__main__":
    sys.exit(main())

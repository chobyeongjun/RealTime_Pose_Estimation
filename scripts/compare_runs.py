#!/usr/bin/env python3
"""Plan v6 baseline vs candidate run comparator (TDD-loop tool).

Parses N CUDA_Stream run logs per side, aggregates per-stage stats across runs,
verifies four TDD invariants, and recommends the next branch in Plan v6.

Run logs are produced by `launch_clean.sh` and contain:
  - rolling [STATS] blocks every 10s (ignored — final summary is more accurate)
  - final summary block at shutdown (parsed)
  - per-frame [SLOW] entries for true_e2e > LATENCY_SOFT_WARN_MS (used for zed_lag)

Usage:
    # Self-test (no Jetson logs needed — synthetic data)
    python3 scripts/compare_runs.py --self-test

    # Real comparison
    python3 scripts/compare_runs.py \\
        --baseline run_8fcfbcc_1.log run_8fcfbcc_2.log run_8fcfbcc_3.log \\
        --baseline-name "8fcfbcc (Phase 0 only)" \\
        --candidate run_main_1.log run_main_2.log run_main_3.log \\
        --candidate-name "main (Phase A)" \\
        [--unique-frames]   # dedupe [SLOW] entries by frame_id (Plan v6.1 fair-compare)

Invariants (from Plan v6.1):
    Inv1  decomp 등식       sum(decomp p50) ≈ true_e2e p50  ±15%
    Inv2  측정 안정성        run-to-run spread (max-min)/mean ≤ 30%
    Inv3  Phase A zed_lag   candidate zed_lag p99 ≤ baseline × 1.10
    Inv4  frame_skip 의미    candidate frame_skip > 0 (consume-once active)

Exit codes:
    0  All invariants PASS, branch recommendation printed
    1  Parse error or missing files
    2  At least one invariant FAILED (recommendation may suggest rollback)
    3  Self-test failure
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Optional, Tuple


# ─── Regex patterns (final summary, not rolling [STATS]) ──────────────────────

RE_DONE = re.compile(
    r"done:\s+(?P<ticks>\d+)\s+ticks\s+/\s+(?P<dur>[\d.]+)s\s+→\s+(?P<hz>[\d.]+)\s+Hz"
    r"(?:\s+\(skip=(?P<skip>\d+),\s+(?P<skip_pct>[\d.]+)%)?"
)
RE_E2E = re.compile(
    r"e2e \(gpu only\) p50/95/99\s*=\s*"
    r"(?P<p50>[\d.]+)/(?P<p95>[\d.]+)/(?P<p99>[\d.]+) ms\s+max=(?P<mx>[\d.]+) ms"
)
RE_TRUE_E2E = re.compile(
    r"true_e2e \(cam→gpu_done\) p50/95/99\s*=\s*"
    r"(?P<p50>[\d.]+)/(?P<p95>[\d.]+)/(?P<p99>[\d.]+) ms\s+max=(?P<mx>[\d.]+) ms"
)
RE_HARD = re.compile(
    r"HARD LIMIT [\d.]+ ms \(true_e2e basis\):\s+(?P<over>\d+)\s+/\s+(?P<total>\d+)"
    r"\s+frames violated\s+\((?P<pct>[\d.]+)%\)"
)
RE_DECOMP_DIST = re.compile(
    r"decomposition p50/p99:\s+bridge_proc=(?P<bp50>[\d.]+)/(?P<bp99>[\d.]+)\s+"
    r"queue_wait=(?P<qw50>[\d.]+)/(?P<qw99>[\d.]+)\s+"
    r"pipeline_proc=(?P<pp50>[\d.]+)/(?P<pp99>[\d.]+) ms"
)
RE_DECOMP_MEAN = re.compile(
    r"decomposition mean.*bridge=(?P<b>[\d.]+)\s+\+\s+queue=(?P<q>[\d.]+)\s+\+\s+"
    r"pipeline=(?P<p>[\d.]+)\s*=\s*(?P<sum>[\d.]+) ms\s+\(true_e2e mean=(?P<tm>[\d.]+)\)"
)
RE_AP_FINAL = re.compile(
    # Phase A only — actual_publish_e2e_ms summary line (if present)
    r"actual_publish.*p50/95/99\s*=\s*"
    r"(?P<p50>[\d.]+)/(?P<p95>[\d.]+)/(?P<p99>[\d.]+) ms\s+max=(?P<mx>[\d.]+) ms"
)
RE_SLOW_HEAD = re.compile(
    r"\[SLOW\] frame (?P<fid>\d+)\s+true_e2e=(?P<true>[\d.]+) ms \(e2e=(?P<e2e>[\d.]+)\)"
)
RE_SLOW_DECOMP = re.compile(
    r"decomp:\s+bridge_proc=(?P<bp>[\d.]+)\s+queue_wait=(?P<qw>[\d.]+)\s+"
    r"pipeline_proc=(?P<pp>[\d.]+)\s+zed_lag=(?P<zl>[\d.]+) ms"
)


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class RunStats:
    """Single run summary. Missing fields are NaN."""
    path: str = ""
    ticks: int = 0
    duration_s: float = float("nan")
    hz: float = float("nan")
    frame_skip: Optional[int] = None  # main only

    # GPU-only e2e (camera buffers → gpu done, narrow scope)
    e2e_p50: float = float("nan")
    e2e_p95: float = float("nan")
    e2e_p99: float = float("nan")
    e2e_max: float = float("nan")

    # Cam→gpu_done (true_e2e_ms — primary safety metric)
    true_e2e_p50: float = float("nan")
    true_e2e_p95: float = float("nan")
    true_e2e_p99: float = float("nan")
    true_e2e_max: float = float("nan")
    true_e2e_mean: float = float("nan")

    # HARD LIMIT compliance
    hard_violations: int = 0
    total_frames: int = 0
    hard_pct: float = float("nan")

    # Decomposition (full distribution from final summary)
    bridge_proc_p50: float = float("nan")
    bridge_proc_p99: float = float("nan")
    queue_wait_p50: float = float("nan")
    queue_wait_p99: float = float("nan")
    pipeline_proc_p50: float = float("nan")
    pipeline_proc_p99: float = float("nan")
    bridge_proc_mean: float = float("nan")
    queue_wait_mean: float = float("nan")
    pipeline_proc_mean: float = float("nan")
    decomp_sum_mean: float = float("nan")

    # zed_lag — only available from [SLOW] entries (BIASED toward slow frames)
    zed_lag_p50_slow: float = float("nan")
    zed_lag_p99_slow: float = float("nan")
    zed_lag_max_slow: float = float("nan")
    n_slow: int = 0
    n_slow_unique: int = 0  # after dedupe by frame_id (Plan v6.1)

    # Phase A only
    actual_publish_p99: Optional[float] = None


@dataclass
class Aggregate:
    """Cross-run aggregate. min/mean/max + spread for each metric."""
    name: str
    n_runs: int
    runs: List[RunStats] = field(default_factory=list)

    def metric(self, attr: str) -> Tuple[float, float, float, float]:
        """Return (min, mean, max, spread_pct) for the given metric across runs."""
        vals = [getattr(r, attr) for r in self.runs]
        vals = [v for v in vals if v is not None and not _is_nan(v)]
        if not vals:
            return float("nan"), float("nan"), float("nan"), float("nan")
        mn, mx = min(vals), max(vals)
        mu = mean(vals)
        spread_pct = ((mx - mn) / mu * 100.0) if mu > 0 else float("nan")
        return mn, mu, mx, spread_pct


@dataclass
class InvariantResult:
    name: str
    passed: bool
    explanation: str


def _is_nan(x) -> bool:
    return isinstance(x, float) and x != x


# ─── Parsing ──────────────────────────────────────────────────────────────────

def parse_run_text(text: str, unique_frames: bool = False, path: str = "") -> RunStats:
    """Parse one run's log text into a RunStats."""
    s = RunStats(path=path)

    for line in text.splitlines():
        m = RE_DONE.search(line)
        if m:
            s.ticks = int(m.group("ticks"))
            s.duration_s = float(m.group("dur"))
            s.hz = float(m.group("hz"))
            if m.group("skip") is not None:
                s.frame_skip = int(m.group("skip"))
            continue
        m = RE_E2E.search(line)
        if m:
            s.e2e_p50 = float(m.group("p50"))
            s.e2e_p95 = float(m.group("p95"))
            s.e2e_p99 = float(m.group("p99"))
            s.e2e_max = float(m.group("mx"))
            continue
        m = RE_TRUE_E2E.search(line)
        if m:
            s.true_e2e_p50 = float(m.group("p50"))
            s.true_e2e_p95 = float(m.group("p95"))
            s.true_e2e_p99 = float(m.group("p99"))
            s.true_e2e_max = float(m.group("mx"))
            continue
        m = RE_HARD.search(line)
        if m:
            s.hard_violations = int(m.group("over"))
            s.total_frames = int(m.group("total"))
            s.hard_pct = float(m.group("pct"))
            continue
        m = RE_DECOMP_DIST.search(line)
        if m:
            s.bridge_proc_p50 = float(m.group("bp50"))
            s.bridge_proc_p99 = float(m.group("bp99"))
            s.queue_wait_p50 = float(m.group("qw50"))
            s.queue_wait_p99 = float(m.group("qw99"))
            s.pipeline_proc_p50 = float(m.group("pp50"))
            s.pipeline_proc_p99 = float(m.group("pp99"))
            continue
        m = RE_DECOMP_MEAN.search(line)
        if m:
            s.bridge_proc_mean = float(m.group("b"))
            s.queue_wait_mean = float(m.group("q"))
            s.pipeline_proc_mean = float(m.group("p"))
            s.decomp_sum_mean = float(m.group("sum"))
            s.true_e2e_mean = float(m.group("tm"))
            continue
        m = RE_AP_FINAL.search(line)
        if m:
            s.actual_publish_p99 = float(m.group("p99"))
            continue

    # zed_lag from [SLOW] entries (with optional unique-frame dedupe)
    slow_zed_lags: List[float] = []
    seen_frames: set[int] = set()
    pending_fid: Optional[int] = None
    for line in text.splitlines():
        m = RE_SLOW_HEAD.search(line)
        if m:
            pending_fid = int(m.group("fid"))
            continue
        m = RE_SLOW_DECOMP.search(line)
        if m and pending_fid is not None:
            zl = float(m.group("zl"))
            if unique_frames:
                if pending_fid in seen_frames:
                    pending_fid = None
                    continue
                seen_frames.add(pending_fid)
            slow_zed_lags.append(zl)
            pending_fid = None

    s.n_slow = len(slow_zed_lags) + (len(seen_frames) - len(slow_zed_lags) if unique_frames else 0)
    s.n_slow_unique = len(slow_zed_lags) if unique_frames else len(set(seen_frames)) if seen_frames else len(slow_zed_lags)
    if slow_zed_lags:
        slow_zed_lags.sort()
        n = len(slow_zed_lags)
        s.zed_lag_p50_slow = slow_zed_lags[min(n - 1, n // 2)]
        s.zed_lag_p99_slow = slow_zed_lags[min(n - 1, int(n * 0.99))]
        s.zed_lag_max_slow = slow_zed_lags[-1]

    return s


def parse_run_file(path: Path, unique_frames: bool = False) -> RunStats:
    text = path.read_text(errors="replace")
    return parse_run_text(text, unique_frames=unique_frames, path=str(path))


# ─── Invariant checks ─────────────────────────────────────────────────────────

def check_inv1_decomp_sum(agg: Aggregate, tol_pct: float = 15.0) -> InvariantResult:
    """Inv1: bridge_mean + queue_mean + pipeline_mean ≈ true_e2e_mean (±tol%)"""
    mismatches = []
    for r in agg.runs:
        if any(_is_nan(v) for v in (r.decomp_sum_mean, r.true_e2e_mean)):
            continue
        diff_pct = abs(r.decomp_sum_mean - r.true_e2e_mean) / max(r.true_e2e_mean, 1e-6) * 100
        if diff_pct > tol_pct:
            mismatches.append((r.path, diff_pct))
    if not mismatches:
        return InvariantResult(
            "Inv1 (decomp 등식)", True,
            f"sum(decomp mean) ≈ true_e2e mean (±{tol_pct}%) for all {agg.n_runs} runs",
        )
    return InvariantResult(
        "Inv1 (decomp 등식)", False,
        f"{len(mismatches)} run(s) violate ±{tol_pct}%: " +
        ", ".join(f"{Path(p).name}({d:.1f}%)" for p, d in mismatches),
    )


def check_inv2_stability(agg: Aggregate, max_spread_pct: float = 30.0) -> InvariantResult:
    """Inv2: run-to-run spread (max-min)/mean ≤ max_spread_pct for true_e2e_p99 + Hz"""
    mn_p99, mu_p99, mx_p99, sp_p99 = agg.metric("true_e2e_p99")
    mn_hz, mu_hz, mx_hz, sp_hz = agg.metric("hz")
    fails = []
    if not _is_nan(sp_p99) and sp_p99 > max_spread_pct:
        fails.append(f"true_e2e_p99 spread {sp_p99:.1f}%")
    if not _is_nan(sp_hz) and sp_hz > max_spread_pct:
        fails.append(f"Hz spread {sp_hz:.1f}%")
    if not fails:
        return InvariantResult(
            f"Inv2 ({agg.name} 측정 안정성)", True,
            f"spread ≤ {max_spread_pct}%: true_e2e_p99={sp_p99:.1f}%  Hz={sp_hz:.1f}%",
        )
    return InvariantResult(
        f"Inv2 ({agg.name} 측정 안정성)", False,
        "spread > " + f"{max_spread_pct}%: " + ", ".join(fails),
    )


def check_inv3_phase_a_zed_lag(
    baseline: Aggregate, candidate: Aggregate, ratio: float = 1.10
) -> InvariantResult:
    """Inv3: candidate zed_lag p99 ≤ baseline × ratio (Phase A 무죄 가정 검증)"""
    _, b_mean, _, _ = baseline.metric("zed_lag_p99_slow")
    _, c_mean, _, _ = candidate.metric("zed_lag_p99_slow")
    if _is_nan(b_mean) or _is_nan(c_mean):
        return InvariantResult(
            "Inv3 (Phase A zed_lag)", False,
            f"zed_lag missing — baseline={b_mean}, candidate={c_mean}",
        )
    threshold = b_mean * ratio
    passed = c_mean <= threshold
    return InvariantResult(
        "Inv3 (Phase A zed_lag)", passed,
        f"candidate zed_lag p99 mean={c_mean:.1f}  vs  baseline×{ratio}={threshold:.1f}  "
        f"({'PASS' if passed else 'FAIL — Phase A 의심'})",
    )


def check_inv4_frame_skip(candidate: Aggregate) -> InvariantResult:
    """Inv4: candidate frame_skip > 0 (consume-once 동작 증거)"""
    skips = [r.frame_skip for r in candidate.runs if r.frame_skip is not None]
    if not skips:
        return InvariantResult(
            "Inv4 (frame_skip 의미)", False,
            "candidate runs have no frame_skip field — Phase A code missing",
        )
    nonzero = [s for s in skips if s > 0]
    passed = len(nonzero) > 0
    return InvariantResult(
        "Inv4 (frame_skip 의미)", passed,
        f"frame_skip per run: {skips}  ({'PASS' if passed else 'FAIL — consume-once inactive'})",
    )


# ─── Branch recommendation (Plan v6.1) ────────────────────────────────────────

def recommend_branch(
    invs: List[InvariantResult],
    baseline: Aggregate,
    candidate: Aggregate,
) -> str:
    """Choose next phase per Plan v6.1 logic."""
    inv = {i.name: i for i in invs}

    if not inv["Inv1 (decomp 등식)"].passed:
        return ("계측 버그 — decomp 등식 깨짐. ready_event 위치 (Codex Q3) 재검토 후 "
                "h2d_enqueue/deque_append/pickup 분리 instrumentation 추가 필요.")

    inv2_keys = [k for k in inv if k.startswith("Inv2")]
    if any(not inv[k].passed for k in inv2_keys):
        return ("측정 자체 불안정 — jetson_clocks/nvpmodel 재확인. 5회로 늘려 재측정 권장.")

    if not inv["Inv4 (frame_skip 의미)"].passed:
        return ("Phase A consume-once 비활성. main 빌드/배포 확인. "
                "consume-once 동작 안 하면 baseline과 의미적 동일 → 비교 의미 없음.")

    if not inv["Inv3 (Phase A zed_lag)"].passed:
        return ("Phase A 의심 — zed_lag p99이 baseline 대비 10% 초과 증가. "
                "Codex Q2 통찰 적용: bridge.latest()의 1ms sleep polling을 condvar로 교체 "
                "또는 consume-once를 일시 롤백하고 unique-frame 기준으로 재측정. "
                "다만 Codex 가설 — 'baseline은 duplicate frame으로 lag을 숨겼다' — 도 검토. "
                "→ --unique-frames 모드로 다시 비교 권장.")

    # All invariants PASS. Decide based on dominant stage in candidate.
    _, c_zed, _, _ = candidate.metric("zed_lag_p99_slow")
    _, c_pipe, _, _ = candidate.metric("pipeline_proc_p99")
    _, c_bridge, _, _ = candidate.metric("bridge_proc_p99")

    if not _is_nan(c_zed) and c_zed > 25.0:
        return (f"가설 H2 확정: zed_lag p99 mean={c_zed:.1f}ms ≫ ZED X Mini 정상 17-25ms. "
                "→ ZED SDK 내부 buffer 적체. Codex Q4: T1/T2 분리는 1-3ms gain만, "
                "ZED CUDA interop이 본질적 해법. "
                "권장 순서: (1) bridge-only 실험 — pipeline 떼고 grab+ts만 측정해 "
                "'pure ZED latency' 분리, (2) 결과에 따라 Phase D (CUDA interop) 직진.")

    if not _is_nan(c_pipe) and c_pipe > 18.0:
        return (f"pipeline_proc p99 mean={c_pipe:.1f}ms 우세. "
                "Codex Q3: post-stage .cpu() D2H sync (gpu_postprocess.py:266) 제거 시도 — "
                "Phase B1 진행. 예상 -3-5ms.")

    if not _is_nan(c_bridge) and c_bridge > 8.0:
        return (f"bridge_proc p99 mean={c_bridge:.1f}ms 우세. "
                "host copy (getdata + pinned + H2D) 비중 점검 → bridge breakdown 분석 후 "
                "Phase B2 또는 D 결정.")

    return ("모든 invariant PASS, 분포 양호. 5회로 늘려 분포 안정성 확정 후 다음 phase 결정.")


# ─── Reporting ────────────────────────────────────────────────────────────────

def fmt_metric(label: str, baseline: Aggregate, candidate: Aggregate, attr: str) -> str:
    b_mn, b_mu, b_mx, b_sp = baseline.metric(attr)
    c_mn, c_mu, c_mx, c_sp = candidate.metric(attr)
    if _is_nan(b_mu) or _is_nan(c_mu):
        delta_str = "  N/A"
    else:
        delta = c_mu - b_mu
        delta_pct = (delta / b_mu * 100) if b_mu > 0 else float("nan")
        sign = "+" if delta >= 0 else ""
        delta_str = f"  Δ={sign}{delta:5.1f}  ({sign}{delta_pct:5.1f}%)"
    return (
        f"  {label:<22}  "
        f"{baseline.name[:12]:>12}: {b_mu:6.1f} (sp {b_sp:4.1f}%)  | "
        f"{candidate.name[:12]:>12}: {c_mu:6.1f} (sp {c_sp:4.1f}%){delta_str}"
    )


def print_report(baseline: Aggregate, candidate: Aggregate, invs: List[InvariantResult]) -> None:
    print(f"=== Plan v6 비교 리포트 ===\n")
    print(f"Baseline:  {baseline.name}  ({baseline.n_runs} runs)")
    for r in baseline.runs:
        print(f"  - {Path(r.path).name}: ticks={r.ticks} hz={r.hz:.1f} "
              f"true_e2e p99={r.true_e2e_p99:.1f}ms zed_lag p99={r.zed_lag_p99_slow:.1f}ms "
              f"hard={r.hard_pct:.2f}% slow_n={r.n_slow}")
    print(f"\nCandidate: {candidate.name}  ({candidate.n_runs} runs)")
    for r in candidate.runs:
        skip_str = f" skip={r.frame_skip}" if r.frame_skip is not None else ""
        print(f"  - {Path(r.path).name}: ticks={r.ticks} hz={r.hz:.1f}{skip_str} "
              f"true_e2e p99={r.true_e2e_p99:.1f}ms zed_lag p99={r.zed_lag_p99_slow:.1f}ms "
              f"hard={r.hard_pct:.2f}% slow_n={r.n_slow}")

    print("\n=== Per-stage comparison (mean across runs, sp = run-to-run spread%) ===")
    for label, attr in [
        ("Hz",                      "hz"),
        ("true_e2e p50",            "true_e2e_p50"),
        ("true_e2e p95",            "true_e2e_p95"),
        ("true_e2e p99",            "true_e2e_p99"),
        ("true_e2e max",            "true_e2e_max"),
        ("HARD violation %",        "hard_pct"),
        ("zed_lag p99 (slow biased)", "zed_lag_p99_slow"),
        ("bridge_proc p99",         "bridge_proc_p99"),
        ("queue_wait p99",          "queue_wait_p99"),
        ("pipeline_proc p99",       "pipeline_proc_p99"),
        ("e2e (gpu only) p99",      "e2e_p99"),
    ]:
        print(fmt_metric(label, baseline, candidate, attr))

    print("\n=== TDD Invariants ===")
    for inv in invs:
        mark = "[PASS]" if inv.passed else "[FAIL]"
        print(f"  {mark}  {inv.name}: {inv.explanation}")

    print("\n=== Branch Recommendation (Plan v6.1) ===")
    print(f"  → {recommend_branch(invs, baseline, candidate)}")


# ─── Self-test ────────────────────────────────────────────────────────────────

def _make_synth_log(
    fps: float, true_e2e_p99: float, true_e2e_mean: float,
    bridge_p99: float, queue_p99: float, pipeline_p99: float,
    bridge_mean: float, queue_mean: float, pipeline_mean: float,
    zed_lags_slow: List[float], hard_pct: float = 5.0,
    skip_count: Optional[int] = None,
) -> str:
    skip_str = f"  (skip={skip_count}, 5.0% of polls; stats on 2400 post-warmup frames)" if skip_count is not None else "  (stats on 2400 post-warmup frames)"
    lines = [
        f"done: 2400 ticks / 60.0s → {fps:.1f} Hz{skip_str}",
        f"e2e (gpu only) p50/95/99 = 12.0/15.0/18.0 ms  max=20.0 ms",
        f"true_e2e (cam→gpu_done) p50/95/99 = 14.0/{true_e2e_p99-5:.1f}/{true_e2e_p99:.1f} ms  max={true_e2e_p99+5:.1f} ms",
        f"HARD LIMIT 20 ms (true_e2e basis): 120 / 2400 frames violated ({hard_pct:.3f}%) — published as valid=False",
        f"decomposition p50/p99: bridge_proc=2.0/{bridge_p99:.1f}  queue_wait=3.0/{queue_p99:.1f}  pipeline_proc=10.0/{pipeline_p99:.1f} ms",
        f"decomposition mean (sum check): bridge={bridge_mean:.1f} + queue={queue_mean:.1f} + pipeline={pipeline_mean:.1f} = {bridge_mean+queue_mean+pipeline_mean:.1f} ms (true_e2e mean={true_e2e_mean:.1f})",
    ]
    for i, zl in enumerate(zed_lags_slow):
        lines.append(f"[SLOW] frame {100+i}  true_e2e=85.0 ms (e2e=18.0)")
        lines.append(f"  decomp: bridge_proc=2.0  queue_wait=3.0  pipeline_proc=15.0  zed_lag={zl:.1f} ms")
        lines.append(f"  capture : grab=2.0  ret_rgb=2.0  getdata_rgb=10.0  pinned_rgb=1.0  ret_depth=2.0  getdata_depth=8.0")
        lines.append(f"  pipeline: pre=1.0  inf=7.0  post=7.0  constraint=2.0")
    return "\n".join(lines)


def self_test() -> int:
    print("=== Self-test ===")
    failures = 0

    # Test 1: parse a synthetic log
    log = _make_synth_log(
        fps=43.0, true_e2e_p99=85.0, true_e2e_mean=20.0,
        bridge_p99=4.5, queue_p99=8.2, pipeline_p99=20.3,
        bridge_mean=2.5, queue_mean=4.0, pipeline_mean=13.5,
        zed_lags_slow=[40.0, 41.0, 42.0, 43.0, 65.0],
    )
    s = parse_run_text(log)
    assert s.hz == 43.0, f"hz {s.hz}"
    assert s.true_e2e_p99 == 85.0, f"true_e2e_p99 {s.true_e2e_p99}"
    assert s.bridge_proc_p99 == 4.5, f"bridge_proc_p99 {s.bridge_proc_p99}"
    assert s.true_e2e_mean == 20.0, f"true_e2e_mean {s.true_e2e_mean}"
    assert abs(s.zed_lag_p99_slow - 65.0) < 0.1, f"zed_lag_p99_slow {s.zed_lag_p99_slow}"
    assert s.n_slow == 5, f"n_slow {s.n_slow}"
    print("  T1 parse synthetic — PASS")

    # Test 2: unique-frames dedupe (same frame_id appearing twice should count once)
    log_dup = log + "\n" + "\n".join([
        "[SLOW] frame 100  true_e2e=85.0 ms (e2e=18.0)",  # duplicate
        "  decomp: bridge_proc=2.0  queue_wait=3.0  pipeline_proc=15.0  zed_lag=99.0 ms",
        "  capture : ...",
        "  pipeline: ...",
    ])
    s_raw = parse_run_text(log_dup, unique_frames=False)
    s_uniq = parse_run_text(log_dup, unique_frames=True)
    assert s_raw.zed_lag_max_slow == 99.0, "raw mode should include duplicate"
    assert s_uniq.zed_lag_max_slow == 65.0, f"unique mode should drop duplicate, got {s_uniq.zed_lag_max_slow}"
    print("  T2 unique-frames dedupe — PASS")

    # Test 3: aggregate spread
    runs = [
        RunStats(hz=40.0, true_e2e_p99=80.0, zed_lag_p99_slow=40.0,
                 decomp_sum_mean=20.0, true_e2e_mean=20.0, frame_skip=10),
        RunStats(hz=42.0, true_e2e_p99=85.0, zed_lag_p99_slow=42.0,
                 decomp_sum_mean=20.5, true_e2e_mean=20.0, frame_skip=12),
        RunStats(hz=44.0, true_e2e_p99=90.0, zed_lag_p99_slow=44.0,
                 decomp_sum_mean=21.0, true_e2e_mean=20.0, frame_skip=15),
    ]
    agg = Aggregate(name="test", n_runs=3, runs=runs)
    mn, mu, mx, sp = agg.metric("hz")
    assert mn == 40.0 and mx == 44.0, f"hz min/max {mn}/{mx}"
    assert abs(mu - 42.0) < 1e-6, f"hz mean {mu}"
    assert abs(sp - (4.0 / 42.0 * 100)) < 0.01, f"hz spread {sp}"
    print("  T3 aggregate metric — PASS")

    # Test 4: Inv1 PASS
    inv1 = check_inv1_decomp_sum(agg, tol_pct=10.0)
    assert inv1.passed, f"Inv1 expected PASS, got: {inv1.explanation}"

    # Inv1 FAIL
    bad_runs = [RunStats(decomp_sum_mean=10.0, true_e2e_mean=20.0, path="bad.log")]
    bad_agg = Aggregate(name="bad", n_runs=1, runs=bad_runs)
    inv1_bad = check_inv1_decomp_sum(bad_agg, tol_pct=10.0)
    assert not inv1_bad.passed
    print("  T4 Inv1 (decomp 등식) — PASS")

    # Test 5: Inv2 (stability)
    inv2 = check_inv2_stability(agg, max_spread_pct=30.0)
    assert inv2.passed, inv2.explanation
    unstable = Aggregate(name="unstable", n_runs=2, runs=[
        RunStats(hz=20.0, true_e2e_p99=50.0),
        RunStats(hz=80.0, true_e2e_p99=200.0),
    ])
    inv2_bad = check_inv2_stability(unstable, max_spread_pct=30.0)
    assert not inv2_bad.passed
    print("  T5 Inv2 (측정 안정성) — PASS")

    # Test 6: Inv3 (Phase A zed_lag)
    baseline_agg = Aggregate(name="baseline", n_runs=2, runs=[
        RunStats(zed_lag_p99_slow=40.0), RunStats(zed_lag_p99_slow=42.0),
    ])
    cand_ok = Aggregate(name="cand_ok", n_runs=2, runs=[
        RunStats(zed_lag_p99_slow=43.0), RunStats(zed_lag_p99_slow=44.0),
    ])  # mean 43.5 ≤ 41 * 1.10 = 45.1 → PASS
    cand_fail = Aggregate(name="cand_fail", n_runs=2, runs=[
        RunStats(zed_lag_p99_slow=50.0), RunStats(zed_lag_p99_slow=55.0),
    ])  # mean 52.5 > 45.1 → FAIL
    assert check_inv3_phase_a_zed_lag(baseline_agg, cand_ok).passed
    assert not check_inv3_phase_a_zed_lag(baseline_agg, cand_fail).passed
    print("  T6 Inv3 (Phase A zed_lag) — PASS")

    # Test 7: Inv4 (frame_skip)
    cand_skip = Aggregate(name="x", n_runs=2, runs=[
        RunStats(frame_skip=10), RunStats(frame_skip=20),
    ])
    cand_no_skip = Aggregate(name="x", n_runs=2, runs=[
        RunStats(frame_skip=0), RunStats(frame_skip=0),
    ])
    cand_no_field = Aggregate(name="x", n_runs=1, runs=[RunStats()])
    assert check_inv4_frame_skip(cand_skip).passed
    assert not check_inv4_frame_skip(cand_no_skip).passed
    assert not check_inv4_frame_skip(cand_no_field).passed
    print("  T7 Inv4 (frame_skip) — PASS")

    # Test 8: branch recommendation — H2 (zed_lag dominant)
    base_h2 = Aggregate(name="base", n_runs=2, runs=[
        RunStats(zed_lag_p99_slow=38.0, frame_skip=None,
                 decomp_sum_mean=20.0, true_e2e_mean=20.0,
                 hz=43.0, true_e2e_p99=80.0, pipeline_proc_p99=15.0,
                 bridge_proc_p99=4.0),
        RunStats(zed_lag_p99_slow=40.0, frame_skip=None,
                 decomp_sum_mean=20.5, true_e2e_mean=20.0,
                 hz=44.0, true_e2e_p99=82.0, pipeline_proc_p99=15.0,
                 bridge_proc_p99=4.0),
    ])
    cand_h2 = Aggregate(name="cand", n_runs=2, runs=[
        RunStats(zed_lag_p99_slow=39.0, frame_skip=10,
                 decomp_sum_mean=20.0, true_e2e_mean=20.0,
                 hz=43.0, true_e2e_p99=80.0, pipeline_proc_p99=15.0,
                 bridge_proc_p99=4.0),
        RunStats(zed_lag_p99_slow=41.0, frame_skip=12,
                 decomp_sum_mean=20.5, true_e2e_mean=20.0,
                 hz=44.0, true_e2e_p99=82.0, pipeline_proc_p99=15.0,
                 bridge_proc_p99=4.0),
    ])
    invs = [
        check_inv1_decomp_sum(cand_h2),
        check_inv2_stability(base_h2),
        check_inv2_stability(cand_h2),
        check_inv3_phase_a_zed_lag(base_h2, cand_h2),
        check_inv4_frame_skip(cand_h2),
    ]
    rec = recommend_branch(invs, base_h2, cand_h2)
    assert "H2" in rec or "zed_lag" in rec, f"expected H2 branch, got: {rec}"
    print("  T8 branch recommendation H2 — PASS")

    # Test 9: branch recommendation — Phase A 의심 (Inv3 fail)
    cand_lag = Aggregate(name="cand", n_runs=2, runs=[
        RunStats(zed_lag_p99_slow=60.0, frame_skip=10,
                 decomp_sum_mean=20.0, true_e2e_mean=20.0,
                 hz=43.0, true_e2e_p99=80.0, pipeline_proc_p99=15.0,
                 bridge_proc_p99=4.0),
        RunStats(zed_lag_p99_slow=62.0, frame_skip=12,
                 decomp_sum_mean=20.5, true_e2e_mean=20.0,
                 hz=44.0, true_e2e_p99=82.0, pipeline_proc_p99=15.0,
                 bridge_proc_p99=4.0),
    ])
    invs2 = [
        check_inv1_decomp_sum(cand_lag),
        check_inv2_stability(base_h2),
        check_inv2_stability(cand_lag),
        check_inv3_phase_a_zed_lag(base_h2, cand_lag),
        check_inv4_frame_skip(cand_lag),
    ]
    rec2 = recommend_branch(invs2, base_h2, cand_lag)
    assert "Phase A 의심" in rec2 or "consume-once" in rec2, f"expected Phase A suspicion, got: {rec2}"
    print("  T9 branch recommendation Phase A 의심 — PASS")

    print("\nALL self-tests PASSED")
    return 0


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true", help="Run synthetic-data self-test and exit")
    p.add_argument("--baseline", nargs="+", help="Baseline run logs (≥1)")
    p.add_argument("--candidate", nargs="+", help="Candidate run logs (≥1)")
    p.add_argument("--baseline-name", default="baseline", help="Display name for baseline")
    p.add_argument("--candidate-name", default="candidate", help="Display name for candidate")
    p.add_argument("--unique-frames", action="store_true",
                   help="Dedupe [SLOW] entries by frame_id (Plan v6.1 fair-compare per Codex Q2)")
    args = p.parse_args(argv)

    if args.self_test:
        try:
            return self_test()
        except AssertionError as e:
            print(f"SELF-TEST FAILED: {e}", file=sys.stderr)
            return 3

    if not args.baseline or not args.candidate:
        p.error("--baseline and --candidate required (or --self-test)")

    baseline_runs = [parse_run_file(Path(p), unique_frames=args.unique_frames) for p in args.baseline]
    candidate_runs = [parse_run_file(Path(p), unique_frames=args.unique_frames) for p in args.candidate]

    baseline = Aggregate(name=args.baseline_name, n_runs=len(baseline_runs), runs=baseline_runs)
    candidate = Aggregate(name=args.candidate_name, n_runs=len(candidate_runs), runs=candidate_runs)

    invs = [
        check_inv1_decomp_sum(candidate),
        check_inv2_stability(baseline),
        check_inv2_stability(candidate),
        check_inv3_phase_a_zed_lag(baseline, candidate),
        check_inv4_frame_skip(candidate),
    ]

    print_report(baseline, candidate, invs)

    return 0 if all(i.passed for i in invs) else 2


if __name__ == "__main__":
    sys.exit(main())

# Mac-only validation workflow — Jetson trips minimised

Goal: cycle through implementation + verification on Mac only, falling back to
Jetson **once per major milestone** (not once per code change).

## What runs on Mac

| Capability | Mac OK? | Why |
|---|---|---|
| Plan D EKF logic | ✓ | pure numpy / scipy |
| Synthetic walking signal generation | ✓ | offline generator |
| NPZ replay + analysis | ✓ | NPZ already downloaded |
| EKF ω/φ visualisation | ✓ | matplotlib agg |
| Cable kinematics (geometry math) | ✓ | numpy |
| Bridge logic + MockSerial | ✓ | no Teensy needed |
| Latency budget analysis | ✓ | trace.csv from prior Jetson session |
| Forecast accuracy on synthetic | ✓ | (limited, see "What Mac CANNOT capture") |
| `/codex review` and `/codex consult` | ✓ | sandboxed read-only repo access |
| Mocap-style ground truth simulator | ✓ | synthetic + analysis |

## What Mac CANNOT capture (Jetson required for these)

| Capability | Why Jetson needed |
|---|---|
| Actual TRT YOLO inference | TensorRT engine compiled for sm_87 |
| Real ZED depth (PERFORMANCE mode) | pyzed.sl + camera hardware |
| Real-time T0~T4 latency on production hardware | matters for the paper's RT claim |
| Real biomechanical hip_vertical shape | Plan D's Hilbert envelope is calibrated to *real* gait, not idealised sinusoids — synthetic reproduction is approximate |
| Teensy USB Serial round-trip (T5~T11) | actual firmware execution |
| Mocap-vision sync | Mocap hardware |

## Workflow

### Daily cycle (Mac only)

```
1. Read docs/plan_d_explained.md when EKF concepts are unclear
2. Modify code locally
3. python3 -m pytest tests/ -q                    # full Mac regression
4. python3 scripts/visualize_ekf_learning.py <existing NPZ>
   → check PNG plots for the behaviour you changed
5. /codex review on the commit before pushing
6. git push
```

### Per-milestone Jetson session

```
1. Jetson: git pull
2. Jetson: bash scripts/jetson_full_test.sh --walking 60
3. Jetson: scp recordings/walking_<TS>/* to Mac
4. Mac: python3 scripts/analyze_walking_results.py recordings/walking_<TS>/
5. Mac: python3 scripts/visualize_ekf_learning.py recordings/walking_<TS>/walking_*.npz
   → ω trace, phi sawtooth, cascade transitions visible
6. Mac: paste analysis.md + PNGs into discussion
7. Mac: decide next iteration based on real data
```

### Realistic expectations

- **Synthetic walking signal does NOT match Plan D exactly** — Hilbert
  envelope and HS detection are calibrated to real gait biomechanics
  (non-sinusoidal stance/swing, distinct envelope shape). Synthetic
  ±cos(phi) or even our `synth_walking_signal.py` give approximate
  but not exact reproduction. Real ω learning lives on the Jetson side.

- **Tests on Mac** assert *invariants* (layout, math correctness, code
  paths) — not paper-grade ω convergence. Paper claims will be measured
  with Mocap ground truth in Phase 6.

- **Visualisation on Mac** is the bridge between abstract concepts and
  reality. Run `visualize_ekf_learning.py` on every new NPZ; the PNG
  immediately shows whether the EKF locked the cadence.

## Tools index (all Mac-runnable)

| Tool | Purpose |
|---|---|
| `scripts/synth_walking_signal.py` | generate realistic walking-shape signals |
| `scripts/visualize_ekf_learning.py` | render ω/φ/cascade trace from NPZ |
| `scripts/cable_kinematics.py` | compute cable length from joint pose |
| `scripts/analyze_walking_results.py` | NPZ + trace.csv unified analysis |
| `scripts/dump_shm.py` (Jetson-only) | live SHM inspection |
| `scripts/shm_to_teensy_bridge.py --mock` | bridge dry run, no Teensy |

## When to re-record on Jetson

Re-record only when:
1. A code change touches the **vision-to-Plan D feed** path (e.g., Phase 2 fix).
2. New geometry calibration (walker mount position, P_anchor measurement).
3. Mocap session.

Do NOT re-record for:
- Code changes that only touch C++ control loop (not yet exists).
- Cable kinematics math changes (replay through existing NPZ).
- Plot / analyzer tweaks.

## Tutorial

If EKF concepts are unclear, read `docs/plan_d_explained.md` first.
It covers state choice, process model, measurement model, Kalman gain,
cascade transitions, and failure modes with code line refs.

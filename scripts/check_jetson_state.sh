#!/usr/bin/env bash
# Print Jetson real-time-critical state so measurements are reproducible.
#
# Usage:
#     bash scripts/check_jetson_state.sh
#     bash scripts/check_jetson_state.sh > /tmp/jetson_state_$(date +%Y%m%d_%H%M).txt
#
# What it checks:
#   - nvpmodel (power mode)         : MUST be 0 (MAXN)
#   - jetson_clocks                 : MUST be applied (locks max clocks)
#   - GPU clock                     : at min/max from devfreq
#   - CPU governor                  : performance vs powersave
#   - Thermal zones                 : current temp per zone
#   - Free memory                   : RAM available
#   - GPU memory                    : (NA on Jetson — unified, see free)
#   - Background process pressure   : top users of CPU
set -u

echo "═══════════════════════════════════════════════════════════"
echo " Jetson real-time state — $(date)"
echo "═══════════════════════════════════════════════════════════"

# ── nvpmodel ────────────────────────────────────────────────────────────────
echo ""
echo "── nvpmodel (power mode) ──"
if command -v nvpmodel >/dev/null 2>&1; then
    sudo nvpmodel -q 2>&1 | head -3
else
    echo "  ⚠ nvpmodel not installed"
fi

# ── jetson_clocks ───────────────────────────────────────────────────────────
echo ""
echo "── jetson_clocks (status) ──"
if command -v jetson_clocks >/dev/null 2>&1; then
    sudo jetson_clocks --show 2>&1 | head -20
else
    echo "  ⚠ jetson_clocks not installed"
fi

# ── GPU clock ───────────────────────────────────────────────────────────────
echo ""
echo "── GPU clock (current/min/max) ──"
for d in /sys/class/devfreq/*; do
    if [ -d "$d" ] && grep -q gpu <(basename "$d") 2>/dev/null; then
        cur=$(cat "$d/cur_freq" 2>/dev/null || echo "?")
        mn=$(cat "$d/min_freq" 2>/dev/null || echo "?")
        mx=$(cat "$d/max_freq" 2>/dev/null || echo "?")
        gov=$(cat "$d/governor" 2>/dev/null || echo "?")
        echo "  $(basename $d): cur=$cur min=$mn max=$mx governor=$gov"
    fi
done
# Alternative: tegrastats (more reliable on Jetson Orin)
if command -v tegrastats >/dev/null 2>&1; then
    echo "  tegrastats 1-sample (kill after 1s):"
    timeout 1 tegrastats --interval 100 2>/dev/null | head -1 || true
fi

# ── CPU governor ────────────────────────────────────────────────────────────
echo ""
echo "── CPU governor (per core) ──"
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    if [ -f "$c" ]; then
        core=$(echo "$c" | sed -E 's|.*cpu([0-9]+)/.*|\1|')
        gov=$(cat "$c")
        cur=$(cat "${c%scaling_governor}cpuinfo_cur_freq" 2>/dev/null || echo "?")
        mx=$(cat "${c%scaling_governor}cpuinfo_max_freq" 2>/dev/null || echo "?")
        echo "  cpu${core}: ${gov}  (cur=${cur} max=${mx} kHz)"
    fi
done | head -12

# ── Thermal zones ───────────────────────────────────────────────────────────
echo ""
echo "── Thermal zones ──"
for tz in /sys/class/thermal/thermal_zone*; do
    if [ -d "$tz" ]; then
        type_name=$(cat "$tz/type" 2>/dev/null || echo "?")
        temp_raw=$(cat "$tz/temp" 2>/dev/null || echo "0")
        temp_c=$(awk -v t="$temp_raw" 'BEGIN { printf "%.1f", t/1000 }')
        echo "  $(basename $tz) ($type_name): ${temp_c} °C"
    fi
done | head -10

# ── Memory ──────────────────────────────────────────────────────────────────
echo ""
echo "── Memory (RAM) ──"
free -h | head -3

# ── Top CPU users ───────────────────────────────────────────────────────────
echo ""
echo "── Top 5 CPU consumers (instant snapshot) ──"
ps -eo pid,pcpu,pmem,comm --sort=-pcpu 2>/dev/null | head -6

# ── Verdict ─────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo " VERDICT"
echo "═══════════════════════════════════════════════════════════"

# Read nvpmodel mode (sudo'd above already)
NVP_MODE=$(sudo nvpmodel -q 2>/dev/null | grep -oE 'NV Power Mode.*' | head -1)
echo "  Power mode: $NVP_MODE"
if echo "$NVP_MODE" | grep -q "MAXN\|0$"; then
    echo "  ✓ MAXN (full power)"
else
    echo "  ✗ NOT MAXN — apply:  sudo nvpmodel -m 0"
fi

# jetson_clocks "Clocks are running" indicator (approximate check via GPU freq at max)
GPU_CUR=$(cat /sys/class/devfreq/*/cur_freq 2>/dev/null | head -1)
GPU_MAX=$(cat /sys/class/devfreq/*/max_freq 2>/dev/null | head -1)
if [ -n "$GPU_CUR" ] && [ -n "$GPU_MAX" ]; then
    if [ "$GPU_CUR" = "$GPU_MAX" ]; then
        echo "  ✓ GPU at max freq ($GPU_CUR) — jetson_clocks likely applied"
    else
        echo "  ✗ GPU not at max ($GPU_CUR / $GPU_MAX) — apply:  sudo jetson_clocks"
    fi
fi

echo ""
echo "  For reproducible real-time measurement:"
echo "    sudo nvpmodel -m 0 && sudo jetson_clocks"
echo "    bash scripts/check_jetson_state.sh   # verify"

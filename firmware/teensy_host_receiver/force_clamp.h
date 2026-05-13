// On-Teensy 5-layer force clamp. Mirrors the host clamp logic but runs as the
// LAST line of defense before CAN write. Even if host violates contract, this
// must hold AK60 within hardware limits.
//
// Layers:
//   1. fallback_active flag (host already declared fallback) → pretension
//   2. watchdog tripped                                       → pretension
//   3. NaN / inf in q_target or tau_ff                        → pretension
//   4. |tau| > AK60_MAX_TAU_NM                                → clamp to limit
//   5. |dtau/dt| > AK60_MAX_SLEW                              → slew limit
#pragma once

#include <math.h>
#include <stdint.h>

#include "host_protocol.h"

namespace hw {

// AK60 datasheet (T-Motor) — ⚠️ verify with actual gearbox before deploy.
// "70N max force" in CLAUDE.md is the CABLE-side spec; motor side is torque (N·m).
//
// Unit contract:
//   host sends tau_ff_N   in CABLE NEWTONS (mechanical force at end-effector)
//   force_clamp converts  to MOTOR N·m via   tau_Nm = tau_N * PULLEY_RADIUS_M
//   CAN MIT mode commands MOTOR N·m
//
// Cable max 70 N × pulley 130 mm = motor max ~9 N·m.
inline constexpr float PULLEY_RADIUS_M      = 0.130f;   // ⚠️ verify on physical build
inline constexpr float AK60_MAX_TAU_NM      = 9.0f;     // = 70 N cable at 130 mm pulley
inline constexpr float AK60_MAX_TAU_CABLE_N = AK60_MAX_TAU_NM / PULLEY_RADIUS_M;  // ~69.2 N
inline constexpr float AK60_MAX_SLEW_NM_S   = 25.0f;    // 9 N·m in 360 ms
inline constexpr float PRETENSION_TAU_NM    = 0.65f;    // ~5N at 130mm pulley
inline constexpr float MAX_KP_NM_PER_RAD    = 80.0f;
inline constexpr float MAX_KD_NMS_PER_RAD   = 4.0f;
inline constexpr float MAX_Q_TARGET_RAD     = 2.5f;     // ~143° — physical ROM

enum ClampReason : uint8_t {
    CLAMP_OK            = 0,
    CLAMP_FALLBACK_FLAG = 1,
    CLAMP_WATCHDOG      = 2,
    CLAMP_NAN_INPUT     = 3,
    CLAMP_TAU_LIMIT     = 4,
    CLAMP_SLEW_LIMIT    = 5,
    CLAMP_NO_PACKET     = 6,
};

struct ClampedCommand {
    float    q_target_rad[hw_proto::N_JOINTS];
    float    tau_ff_Nm[hw_proto::N_JOINTS];
    float    kp[hw_proto::N_JOINTS];
    float    kd[hw_proto::N_JOINTS];
    uint8_t  reason;
    uint8_t  pretension;       // 1 → all torques = PRETENSION_TAU_NM
};

class ForceClamp {
public:
    // Apply all 5 layers in order. dt_s is loop delta (e.g., 0.005 for 200Hz).
    void apply(const hw_proto::CommandBody* src,
               bool watchdog_tripped,
               float dt_s,
               ClampedCommand& out)
    {
        // Default: pretension. Each layer either accepts the command or trips back.
        const bool no_pkt = (src == nullptr);
        const bool fallback_flag = (!no_pkt) && (src->fallback_active != 0);

        if (no_pkt) {
            set_pretension(out, CLAMP_NO_PACKET);
            commit_prev_(out);
            return;
        }
        if (fallback_flag) {
            set_pretension(out, CLAMP_FALLBACK_FLAG);
            commit_prev_(out);
            return;
        }
        if (watchdog_tripped) {
            set_pretension(out, CLAMP_WATCHDOG);
            commit_prev_(out);
            return;
        }
        // NaN / inf check
        for (int i = 0; i < hw_proto::N_JOINTS; ++i) {
            if (!isfinite(src->q_target_rad[i]) || !isfinite(src->tau_ff_N[i])
                || !isfinite(src->kp_Nm_per_rad[i]) || !isfinite(src->kd_Nms_per_rad[i])) {
                set_pretension(out, CLAMP_NAN_INPUT);
                commit_prev_(out);
                return;
            }
        }

        // Build candidate
        uint8_t reason = CLAMP_OK;
        for (int i = 0; i < hw_proto::N_JOINTS; ++i) {
            float q  = clamp_abs(src->q_target_rad[i],     MAX_Q_TARGET_RAD);
            float kp = clamp_pos(src->kp_Nm_per_rad[i],    MAX_KP_NM_PER_RAD);
            float kd = clamp_pos(src->kd_Nms_per_rad[i],   MAX_KD_NMS_PER_RAD);

            // host sends tau in CABLE NEWTONS — convert to MOTOR N·m via pulley.
            // (Codex P1: previous code passed cable N straight to CAN as N·m → 7x over-torque.)
            const float requested_cable_N = src->tau_ff_N[i];
            const float requested_tau_Nm  = requested_cable_N * PULLEY_RADIUS_M;
            float want_tau = clamp_abs(requested_tau_Nm, AK60_MAX_TAU_NM);
            if (fabsf(requested_cable_N) > AK60_MAX_TAU_CABLE_N) reason = CLAMP_TAU_LIMIT;

            // Slew limit
            float prev = prev_tau_[i];
            float max_step = AK60_MAX_SLEW_NM_S * fmaxf(dt_s, 1e-4f);
            float delta = want_tau - prev;
            float applied = want_tau;
            if (fabsf(delta) > max_step) {
                applied = prev + (delta > 0 ? max_step : -max_step);
                if (reason == CLAMP_OK) reason = CLAMP_SLEW_LIMIT;
            }
            applied = clamp_abs(applied, AK60_MAX_TAU_NM);

            out.q_target_rad[i] = q;
            out.tau_ff_Nm[i]    = applied;
            out.kp[i]           = kp;
            out.kd[i]           = kd;
        }
        out.reason     = reason;
        out.pretension = 0;
        commit_prev_(out);
    }

    void reset_prev_tau() {
        for (int i = 0; i < hw_proto::N_JOINTS; ++i) prev_tau_[i] = 0.0f;
    }

private:
    static float clamp_abs(float v, float lim) {
        if (v >  lim) return  lim;
        if (v < -lim) return -lim;
        return v;
    }
    static float clamp_pos(float v, float lim) {
        if (v < 0.0f) return 0.0f;
        if (v > lim)  return lim;
        return v;
    }
    void set_pretension(ClampedCommand& out, uint8_t reason) {
        for (int i = 0; i < hw_proto::N_JOINTS; ++i) {
            out.q_target_rad[i] = 0.0f;
            out.tau_ff_Nm[i]    = PRETENSION_TAU_NM;
            out.kp[i]           = 0.0f;
            out.kd[i]           = 0.0f;
        }
        out.reason     = reason;
        out.pretension = 1;
    }
    void commit_prev_(const ClampedCommand& applied) {
        for (int i = 0; i < hw_proto::N_JOINTS; ++i) {
            prev_tau_[i] = applied.tau_ff_Nm[i];
        }
    }

    float prev_tau_[hw_proto::N_JOINTS]{};
};

}  // namespace hw

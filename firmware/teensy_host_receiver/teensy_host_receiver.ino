// teensy_host_receiver — Receive impedance commands from Jetson over USB
// Serial, validate (CRC + 5-layer clamp), apply to 6 AK60 motors via CAN,
// and stream measured state back as telemetry.
//
// Tested target: Teensy 4.1, FlexCAN_T4 lib, USB native at 480Mbps.
// Set Arduino IDE → Board: Teensy 4.1, USB Type: Serial, CPU 600MHz.
//
// Pin map (Teensy 4.1):
//   CAN1_TX = 22, CAN1_RX = 23   (built-in CAN controller — no transceiver chip needed
//                                  on Teensy 4.1's CAN1; use MCP2562 or TJA1051 if external)
//   USB Serial = native (object `Serial`)
//
// RT metrics emitted in TELEMETRY:
//   recv_mono_us  — T8 candidate (when COMMAND frame finished parsing)
//   can_tx_mono_us — T9 (just before CAN write completed for joint 0)
#include <Arduino.h>

#include "host_protocol.h"
#include "watchdog.h"
#include "force_clamp.h"
#include "can_driver.h"

namespace {

constexpr uint32_t LOOP_PERIOD_US     = 5000UL;   // 200 Hz control
constexpr uint32_t TELEM_PERIOD_US    = 10000UL;  // 100 Hz telemetry up
constexpr uint32_t HEARTBEAT_LED_US   = 250000UL; // 4 Hz blink while OK
constexpr uint32_t STATUS_READ_BUDGET = 6;        // CAN frames per loop

// Two separate watchdogs (Codex P1):
//   g_cmd_wd  — kicked ONLY by PKT_COMMAND (vision freshness). Gates force_clamp.
//   g_link_wd — kicked by any host frame (link liveness). Telemetry only.
// If host keeps heartbeating but vision producer dies, g_cmd_wd trips at 200ms
// while g_link_wd stays fresh, so we go to pretension correctly.
hw::Watchdog                     g_cmd_wd (/*timeout_us=*/200000UL);
hw::Watchdog                     g_link_wd(/*timeout_us=*/500000UL);
hw::ForceClamp                   g_clamp;
hw_proto::CommandBody            g_last_cmd{};
bool                             g_have_cmd = false;
volatile uint32_t                g_last_recv_us = 0;
volatile uint32_t                g_last_cmd_id  = 0;

hw::MotorStatus                  g_status[hw_proto::N_JOINTS]{};

#if defined(__IMXRT1062__)
hw::AK60Bus                      g_can;
#endif

uint32_t g_loop_seq = 0;
uint32_t g_last_telem_us = 0;
uint32_t g_last_blink_us = 0;
bool     g_led_on = false;

// ── Frame callback ─────────────────────────────────────────────────────
void on_frame(hw_proto::PacketType type, const uint8_t* body, uint16_t len, uint64_t recv_mono_us) {
    const uint32_t now32 = (uint32_t)(recv_mono_us & 0xFFFFFFFFu);
    // Every host frame refreshes link watchdog (telemetry/liveness only).
    g_link_wd.kick(now32);

    switch (type) {
        case hw_proto::PKT_COMMAND: {
            if (len != sizeof(hw_proto::CommandBody)) return;
            memcpy(&g_last_cmd, body, sizeof(hw_proto::CommandBody));
            g_have_cmd = true;
            g_last_recv_us = now32;
            g_last_cmd_id  = g_last_cmd.command_id;
            // Command watchdog kicks ONLY here — vision freshness, not link.
            g_cmd_wd.kick(now32);
            break;
        }
        case hw_proto::PKT_HEARTBEAT: {
            // Intentionally does NOT kick g_cmd_wd. If vision dies but host
            // keeps sending heartbeats, g_cmd_wd trips → pretension.
            break;
        }
        case hw_proto::PKT_STOP: {
            // Force pretension immediately. Clamp picks CLAMP_FALLBACK_FLAG.
            g_last_cmd.fallback_active = 1;
            break;
        }
        default:
            break;
    }
}

hw_proto::FrameParser g_parser(&on_frame);

// ── Telemetry up ───────────────────────────────────────────────────────
void send_telemetry(uint32_t now_us, uint32_t can_tx_us, const hw::ClampedCommand& applied) {
    hw_proto::TelemetryBody t{};
    t.command_id_echo = g_last_cmd_id;
    t.teensy_seq      = g_loop_seq;
    t.recv_mono_us    = (uint64_t)g_last_recv_us;
    t.can_tx_mono_us  = (uint64_t)can_tx_us;
    for (int i = 0; i < hw_proto::N_JOINTS; ++i) {
        t.q_meas_rad[i]    = g_status[i].pos_rad;
        t.tau_applied_N[i] = applied.tau_ff_Nm[i];   // N·m (host knows)
    }
    t.fault_bits      = 0;  // TODO: latch CAN errors here
    t.clamp_reason    = applied.reason;
    t.fallback_active = applied.pretension;
    hw_proto::write_frame(Serial,
                          hw_proto::PKT_TELEMETRY,
                          reinterpret_cast<const uint8_t*>(&t),
                          sizeof(t));
}

}  // namespace

// ───────────────────────────────────────────────────────────────────────
// Arduino setup / loop
// ───────────────────────────────────────────────────────────────────────
void setup() {
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);

    Serial.begin(hw_proto::HOST_BAUD);   // baud arg ignored on native USB
    // Wait up to 2s for host, then proceed regardless (firmware boots without host too).
    uint32_t t_start = millis();
    while (!Serial && (millis() - t_start) < 2000) {}

#if defined(__IMXRT1062__)
    g_can.begin();
    delay(50);
    for (uint8_t i = 0; i < hw_proto::N_JOINTS; ++i) {
        g_can.enter_mit(hw::AK60_MOTOR_ID[i]);
        delay(5);
    }
#endif

    g_clamp.reset_prev_tau();
}

void loop() {
    const uint32_t loop_t0 = micros();

    // ── 1. Drain Serial RX ─────────────────────────────────────────────
    while (Serial.available() > 0) {
        int b = Serial.read();
        if (b < 0) break;
        g_parser.feed(static_cast<uint8_t>(b), micros());
    }

    // ── 2. Drain CAN RX (status frames) ────────────────────────────────
#if defined(__IMXRT1062__)
    for (uint32_t i = 0; i < STATUS_READ_BUDGET; ++i) {
        hw::MotorStatus s;
        if (!g_can.read_status(s)) break;
        for (int j = 0; j < hw_proto::N_JOINTS; ++j) {
            if (s.id == hw::AK60_MOTOR_ID[j]) {
                g_status[j] = s;
                break;
            }
        }
    }
#endif

    // ── 3. Clamp + dispatch (200Hz fixed cadence) ──────────────────────
    static uint32_t next_ctrl_us = 0;
    if ((int32_t)(loop_t0 - next_ctrl_us) >= 0) {
        next_ctrl_us = loop_t0 + LOOP_PERIOD_US;
        const float dt_s = LOOP_PERIOD_US * 1e-6f;

        hw::ClampedCommand applied{};
        // ONLY command watchdog gates the clamp — link watchdog is telemetry-only.
        const bool wd_tripped = g_cmd_wd.tripped(loop_t0);
        const hw_proto::CommandBody* src = g_have_cmd ? &g_last_cmd : nullptr;
        g_clamp.apply(src, wd_tripped, dt_s, applied);

        uint32_t can_tx_us = loop_t0;
#if defined(__IMXRT1062__)
        for (int i = 0; i < hw_proto::N_JOINTS; ++i) {
            uint32_t t9 = g_can.send_mit(
                hw::AK60_MOTOR_ID[i],
                applied.q_target_rad[i],
                0.0f,                  // vel target = 0 (impedance via kp/kd)
                applied.kp[i],
                applied.kd[i],
                applied.tau_ff_Nm[i]);
            if (i == 0) can_tx_us = t9;
        }
#endif

        ++g_loop_seq;

        // ── 4. Telemetry up ────────────────────────────────────────────
        if ((uint32_t)(loop_t0 - g_last_telem_us) >= TELEM_PERIOD_US) {
            g_last_telem_us = loop_t0;
            send_telemetry(loop_t0, can_tx_us, applied);
        }

        // ── 5. LED heartbeat ───────────────────────────────────────────
        if ((uint32_t)(loop_t0 - g_last_blink_us) >= HEARTBEAT_LED_US) {
            g_last_blink_us = loop_t0;
            g_led_on = !g_led_on;
            digitalWrite(LED_BUILTIN, applied.pretension ? HIGH : (g_led_on ? HIGH : LOW));
        }
    }
}

// AK60 CAN driver (MIT mode). Uses FlexCAN_T4 on Teensy 4.x.
//
// MIT mode packet (8 bytes per motor):
//   pos_target  16-bit (4 nibbles)
//   vel_target  12-bit
//   kp          12-bit
//   kd          12-bit
//   tau_ff      12-bit
//
// IMPORTANT: ranges below match AK60 v1.1 firmware. ⚠️ Verify against your
// motor's datasheet before deploy. Wrong range = wrong torque commanded.
//
// 6 actuators, IDs 0x01..0x06 (configured per AK60 board).
#pragma once

#include <stdint.h>
#include <math.h>

#include "host_protocol.h"

// FlexCAN_T4 is a Teensy 4.x library. Wrapped in #if so this header can compile
// host-side too (for unit tests that don't link the Arduino core).
#if defined(__IMXRT1062__)
  #include <FlexCAN_T4.h>
#endif

namespace hw {

// AK60 v1.1 MIT-mode ranges
inline constexpr float AK60_P_MIN = -12.5f;
inline constexpr float AK60_P_MAX =  12.5f;
inline constexpr float AK60_V_MIN = -45.0f;
inline constexpr float AK60_V_MAX =  45.0f;
inline constexpr float AK60_KP_MIN =  0.0f;
inline constexpr float AK60_KP_MAX = 500.0f;
inline constexpr float AK60_KD_MIN =  0.0f;
inline constexpr float AK60_KD_MAX =   5.0f;
inline constexpr float AK60_T_MIN = -18.0f;
inline constexpr float AK60_T_MAX =  18.0f;

inline constexpr uint32_t AK60_BAUD = 1000000UL;
inline constexpr uint8_t  AK60_MOTOR_ID[hw_proto::N_JOINTS] = {
    0x01, 0x02, 0x03,   // L hip, knee, ankle
    0x04, 0x05, 0x06,   // R hip, knee, ankle
};

inline uint16_t float_to_uint(float x, float x_min, float x_max, int bits) {
    if (x < x_min) x = x_min;
    if (x > x_max) x = x_max;
    float span = x_max - x_min;
    uint32_t maxv = (1u << bits) - 1u;
    return static_cast<uint16_t>((x - x_min) * static_cast<float>(maxv) / span);
}

inline void pack_mit(float p, float v, float kp, float kd, float t_ff, uint8_t out[8]) {
    uint16_t p_int  = float_to_uint(p,    AK60_P_MIN,  AK60_P_MAX, 16);
    uint16_t v_int  = float_to_uint(v,    AK60_V_MIN,  AK60_V_MAX, 12);
    uint16_t kp_int = float_to_uint(kp,   AK60_KP_MIN, AK60_KP_MAX, 12);
    uint16_t kd_int = float_to_uint(kd,   AK60_KD_MIN, AK60_KD_MAX, 12);
    uint16_t t_int  = float_to_uint(t_ff, AK60_T_MIN,  AK60_T_MAX, 12);
    out[0] = p_int  >> 8;
    out[1] = p_int  & 0xFF;
    out[2] = v_int  >> 4;
    out[3] = ((v_int & 0x0F) << 4) | (kp_int >> 8);
    out[4] = kp_int & 0xFF;
    out[5] = kd_int >> 4;
    out[6] = ((kd_int & 0x0F) << 4) | (t_int >> 8);
    out[7] = t_int  & 0xFF;
}

// Decode incoming AK60 status frame (5 bytes): id, pos, vel, tau
struct MotorStatus {
    uint8_t  id;
    float    pos_rad;
    float    vel_rad_s;
    float    tau_Nm;
    uint32_t recv_us;
};

inline float uint_to_float(uint16_t v, float x_min, float x_max, int bits) {
    uint32_t maxv = (1u << bits) - 1u;
    return static_cast<float>(v) * (x_max - x_min) / static_cast<float>(maxv) + x_min;
}

inline bool unpack_status(const uint8_t* data, uint8_t dlc, MotorStatus& s) {
    if (dlc < 5) return false;
    s.id = data[0];
    uint16_t p_int = (static_cast<uint16_t>(data[1]) << 8) | data[2];
    uint16_t v_int = (static_cast<uint16_t>(data[3]) << 4) | (data[4] >> 4);
    uint16_t t_int = (static_cast<uint16_t>(data[4] & 0x0F) << 8) | data[5];
    s.pos_rad   = uint_to_float(p_int, AK60_P_MIN, AK60_P_MAX, 16);
    s.vel_rad_s = uint_to_float(v_int, AK60_V_MIN, AK60_V_MAX, 12);
    s.tau_Nm    = uint_to_float(t_int, AK60_T_MIN, AK60_T_MAX, 12);
    return true;
}

#if defined(__IMXRT1062__)
// Wrapper that owns the FlexCAN bus and sends 6-motor commands.
class AK60Bus {
public:
    void begin() {
        bus_.begin();
        bus_.setBaudRate(AK60_BAUD);
        bus_.setMaxMB(16);
        bus_.enableFIFO();
        bus_.enableFIFOInterrupt();
    }

    // Send a single motor MIT command. Returns micros() right BEFORE write.
    uint32_t send_mit(uint8_t motor_id, float p, float v, float kp, float kd, float t_ff) {
        CAN_message_t msg;
        msg.id  = motor_id;
        msg.len = 8;
        pack_mit(p, v, kp, kd, t_ff, msg.buf);
        uint32_t t9 = micros();
        bus_.write(msg);
        return t9;
    }

    // Enter MIT mode for a motor (special CAN payload).
    void enter_mit(uint8_t motor_id) {
        CAN_message_t msg;
        msg.id  = motor_id;
        msg.len = 8;
        for (int i = 0; i < 7; ++i) msg.buf[i] = 0xFF;
        msg.buf[7] = 0xFC;
        bus_.write(msg);
    }

    // Exit MIT mode (stop).
    void exit_mit(uint8_t motor_id) {
        CAN_message_t msg;
        msg.id  = motor_id;
        msg.len = 8;
        for (int i = 0; i < 7; ++i) msg.buf[i] = 0xFF;
        msg.buf[7] = 0xFD;
        bus_.write(msg);
    }

    // Read one status frame if available. Returns true on success.
    bool read_status(MotorStatus& s) {
        CAN_message_t msg;
        if (!bus_.read(msg)) return false;
        s.recv_us = micros();
        return unpack_status(msg.buf, msg.len, s);
    }

private:
    FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_64> bus_;
};
#endif  // __IMXRT1062__

}  // namespace hw

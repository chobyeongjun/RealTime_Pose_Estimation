// Host (Jetson) → Teensy USB-Serial binary protocol.
//
// Frame:
//   [0]  uint8   magic[0] = 'H' (0x48)
//   [1]  uint8   magic[1] = 'W' (0x57)
//   [2]  uint8   version = 1
//   [3]  uint8   type    (PacketType enum)
//   [4]  uint16  length  (= body bytes, not header+crc)
//   [6]  body[length]
//   [6+length]  uint16  crc16  (CCITT, poly 0x1021, init 0xFFFF, over [type..body end])
//
// Endianness: little-endian (matches Jetson aarch64).
// Max frame = 2+2+2 + MAX_BODY + 2 = 8 + MAX_BODY bytes.
//
// Joint order (matches src/perception/realtime/pipeline_main.py KEYPOINT_ORDER_6):
//   0: L hip flex   1: L knee   2: L ankle dorsiflexion
//   3: R hip flex   4: R knee   5: R ankle dorsiflexion
//
// Reviewed against forecast_publisher.py at commit b4ba43e.
#pragma once

#include <Arduino.h>
#include <stdint.h>

namespace hw_proto {

inline constexpr uint8_t  MAGIC_0  = 'H';
inline constexpr uint8_t  MAGIC_1  = 'W';
inline constexpr uint8_t  VERSION  = 1;
inline constexpr uint8_t  N_JOINTS = 6;
inline constexpr uint16_t MAX_BODY = 128;       // CommandBody = 112, room for growth
inline constexpr uint32_t HOST_BAUD = 2000000;  // 2Mbps (Teensy USB native speed ignored)

enum PacketType : uint8_t {
    PKT_COMMAND   = 0x01,  // host → teensy: full impedance command
    PKT_HEARTBEAT = 0x02,  // host → teensy: keepalive (resets watchdog only)
    PKT_STOP      = 0x03,  // host → teensy: emergency stop → pretension
    PKT_TELEMETRY = 0x81,  // teensy → host: command echo + measured state
    PKT_FAULT     = 0x82,  // teensy → host: fault code (out-of-band)
};

// Command body — 112 bytes
struct __attribute__((packed)) CommandBody {
    uint32_t command_id;
    uint64_t host_tx_mono_ns;          // host T7 (when host called write())
    float    q_target_rad[N_JOINTS];   // 24 B
    float    tau_ff_N[N_JOINTS];       // 24 B
    float    kp_Nm_per_rad[N_JOINTS];  // 24 B
    float    kd_Nms_per_rad[N_JOINTS]; // 24 B
    uint8_t  use_forecast;             // 1 = derived from /hwalker_forecast
    uint8_t  cascade_level;            // Plan D cascade (1/2/3)
    uint8_t  fallback_active;          // host already decided pretension
    uint8_t  pad;
};
static_assert(sizeof(CommandBody) == 4 + 8 + 4*6*4 + 4, "CommandBody layout drift");

// Telemetry body — 88 bytes (sent at ~200Hz)
struct __attribute__((packed)) TelemetryBody {
    uint32_t command_id_echo;
    uint32_t teensy_seq;
    uint64_t recv_mono_us;             // T8 candidate: when COMMAND finished parsing
    uint64_t can_tx_mono_us;           // T9: when CAN write enqueued
    float    q_meas_rad[N_JOINTS];     // 24 B
    float    tau_applied_N[N_JOINTS];  // 24 B
    uint8_t  fault_bits;
    uint8_t  clamp_reason;             // last ForceClamp::reason
    uint8_t  fallback_active;
    uint8_t  pad;
};
static_assert(sizeof(TelemetryBody) == 4 + 4 + 8 + 8 + 4*6 + 4*6 + 4, "TelemetryBody layout drift");

// CRC16-CCITT (poly 0x1021, init 0xFFFF, no xor-out)
inline uint16_t crc16_ccitt(const uint8_t* p, size_t n, uint16_t crc = 0xFFFF) {
    while (n--) {
        crc ^= static_cast<uint16_t>(*p++) << 8;
        for (int i = 0; i < 8; ++i) {
            crc = (crc & 0x8000) ? static_cast<uint16_t>((crc << 1) ^ 0x1021)
                                 : static_cast<uint16_t>(crc << 1);
        }
    }
    return crc;
}

// Parser state machine — fed one byte at a time from Serial.
class FrameParser {
public:
    using OnFrame = void (*)(PacketType type, const uint8_t* body, uint16_t len, uint64_t recv_mono_us);

    explicit FrameParser(OnFrame cb) : cb_(cb) {}

    // Feed one byte. Returns true if a complete valid frame was just emitted.
    bool feed(uint8_t b, uint64_t now_mono_us) {
        switch (state_) {
            case S_MAGIC0:
                if (b == MAGIC_0) state_ = S_MAGIC1;
                else              ++drop_count_;
                return false;
            case S_MAGIC1:
                if (b == MAGIC_1) state_ = S_VERSION;
                else {
                    ++drop_count_;
                    state_ = (b == MAGIC_0) ? S_MAGIC1 : S_MAGIC0;
                }
                return false;
            case S_VERSION:
                if (b != VERSION) { ++bad_version_; reset_(); return false; }
                state_ = S_TYPE;
                return false;
            case S_TYPE:
                type_ = static_cast<PacketType>(b);
                state_ = S_LEN_LO;
                return false;
            case S_LEN_LO:
                length_ = b;
                state_ = S_LEN_HI;
                return false;
            case S_LEN_HI:
                length_ |= static_cast<uint16_t>(b) << 8;
                if (length_ > MAX_BODY) { ++bad_length_; reset_(); return false; }
                body_idx_ = 0;
                state_ = (length_ == 0) ? S_CRC_LO : S_BODY;
                return false;
            case S_BODY:
                body_[body_idx_++] = b;
                if (body_idx_ >= length_) state_ = S_CRC_LO;
                return false;
            case S_CRC_LO:
                crc_rx_ = b;
                state_ = S_CRC_HI;
                return false;
            case S_CRC_HI: {
                crc_rx_ |= static_cast<uint16_t>(b) << 8;
                // CRC over [type, length_lo, length_hi, body...]
                uint8_t hdr[3] = { static_cast<uint8_t>(type_),
                                   static_cast<uint8_t>(length_ & 0xFF),
                                   static_cast<uint8_t>((length_ >> 8) & 0xFF) };
                uint16_t crc = crc16_ccitt(hdr, 3);
                if (length_) crc = crc16_ccitt(body_, length_, crc);
                bool ok = (crc == crc_rx_);
                if (ok) {
                    ++good_count_;
                    if (cb_) cb_(type_, body_, length_, now_mono_us);
                } else {
                    ++bad_crc_;
                }
                reset_();
                return ok;
            }
        }
        return false;
    }

    void reset_() {
        state_ = S_MAGIC0;
        body_idx_ = 0;
    }

    // Counters (host can query via telemetry/Serial)
    uint32_t drop_count() const { return drop_count_; }
    uint32_t bad_version() const { return bad_version_; }
    uint32_t bad_length() const { return bad_length_; }
    uint32_t bad_crc() const { return bad_crc_; }
    uint32_t good_count() const { return good_count_; }

private:
    enum State : uint8_t {
        S_MAGIC0, S_MAGIC1, S_VERSION, S_TYPE,
        S_LEN_LO, S_LEN_HI, S_BODY, S_CRC_LO, S_CRC_HI,
    } state_{S_MAGIC0};

    OnFrame  cb_;
    PacketType type_{PKT_HEARTBEAT};
    uint16_t length_{0};
    uint16_t body_idx_{0};
    uint16_t crc_rx_{0};
    uint8_t  body_[MAX_BODY]{};

    uint32_t drop_count_{0};
    uint32_t bad_version_{0};
    uint32_t bad_length_{0};
    uint32_t bad_crc_{0};
    uint32_t good_count_{0};
};

// Write a complete framed packet to a Stream (USB Serial). Returns bytes written.
inline size_t write_frame(Stream& out, PacketType type, const uint8_t* body, uint16_t length) {
    uint8_t hdr[6] = {
        MAGIC_0, MAGIC_1, VERSION, static_cast<uint8_t>(type),
        static_cast<uint8_t>(length & 0xFF),
        static_cast<uint8_t>((length >> 8) & 0xFF),
    };
    out.write(hdr, 6);
    if (body && length) out.write(body, length);
    uint8_t crc_in[3] = { hdr[3], hdr[4], hdr[5] };
    uint16_t crc = crc16_ccitt(crc_in, 3);
    if (body && length) crc = crc16_ccitt(body, length, crc);
    uint8_t crc_b[2] = { static_cast<uint8_t>(crc & 0xFF),
                         static_cast<uint8_t>((crc >> 8) & 0xFF) };
    out.write(crc_b, 2);
    return 6 + length + 2;
}

}  // namespace hw_proto

// Vision-staleness watchdog. micros() wraparound-safe.
//
// micros() returns uint32 — wraps every ~71 minutes. Use unsigned subtraction
// which works correctly across the wrap.
#pragma once

#include <Arduino.h>
#include <stdint.h>

namespace hw {

class Watchdog {
public:
    explicit Watchdog(uint32_t timeout_us = 200000UL)  // 0.2s default
        : timeout_us_(timeout_us), last_kick_us_(0), ever_kicked_(false) {}

    void kick(uint32_t now_us) {
        last_kick_us_ = now_us;
        ever_kicked_ = true;
    }

    // True if the last kick was > timeout_us ago (or no kick ever).
    bool tripped(uint32_t now_us) const {
        if (!ever_kicked_) return true;
        // unsigned modulo subtraction handles the wrap
        return (uint32_t)(now_us - last_kick_us_) > timeout_us_;
    }

    uint32_t age_us(uint32_t now_us) const {
        if (!ever_kicked_) return UINT32_MAX;
        return (uint32_t)(now_us - last_kick_us_);
    }

    uint32_t timeout_us() const { return timeout_us_; }

private:
    uint32_t timeout_us_;
    uint32_t last_kick_us_;
    bool     ever_kicked_;
};

}  // namespace hw

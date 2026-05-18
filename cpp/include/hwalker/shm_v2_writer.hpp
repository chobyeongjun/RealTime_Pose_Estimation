// SHM v2 writer — drop-in C++ replacement for shm_publisher.py.
//
// MIRROR of src/perception/CUDA_Stream/shm_publisher.py exactly (Codex review b1ky3965z).
//   HEADER_SIZE = 64
//   per-kp record (48B) = kpts_3d(12) + kpts_2d(8) + conf(4) + sigma(12) + cov_diag(12)
//   total = 64 + K*48, then 64B-aligned (K=6 → 384B)
//
// Seqlock protocol (single producer, multiple readers):
//   - Writer increments seq to odd before write, even after (release fence between).
//   - Reader checks seq before & after body — equal AND even → valid.
//
// ARM weak memory model fix (Codex review b1ky3965z P1-5):
//   Use std::atomic<uint32_t> for seq + memory_order_release on store.
//   Plus std::atomic_thread_fence(release) before seq close.
//
// Build: see cpp/CMakeLists.txt. Linked as Python extension via pybind11.
#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>

namespace hwalker {
namespace shm_v2 {

inline constexpr std::uint32_t VERSION       = 2;
inline constexpr std::size_t   HEADER_SIZE   = 64;
inline constexpr std::size_t   PER_KP_BYTES  = 48;

// Header offsets (must match shm_publisher.py exactly)
inline constexpr std::size_t SEQ_OFF              = 0;
inline constexpr std::size_t VERSION_OFF          = 4;
inline constexpr std::size_t K_OFF                = 8;
inline constexpr std::size_t FRAME_ID_OFF         = 12;
inline constexpr std::size_t RGB_TS_OFF           = 16;
inline constexpr std::size_t DEPTH_TS_OFF         = 24;
inline constexpr std::size_t DEPTH_AGE_OFF        = 32;
inline constexpr std::size_t BOX_CONF_OFF         = 36;
inline constexpr std::size_t DEPTH_INVALID_OFF    = 40;
inline constexpr std::size_t VALID_FLAG_OFF       = 44;
inline constexpr std::size_t WORLD_FRAME_OFF      = 45;
inline constexpr std::size_t VALID_REASON_OFF     = 46;
inline constexpr std::size_t TS_DOMAIN_OFF        = 47;
inline constexpr std::size_t PUBLISH_DONE_OFF     = 48;
inline constexpr std::size_t VALID_MASK_BITS_OFF  = 56;

inline std::size_t compute_size(std::size_t K) {
    const std::size_t payload = K * PER_KP_BYTES;
    const std::size_t total   = HEADER_SIZE + payload;
    return ((total + 63) / 64) * 64;
}

class Writer {
public:
    // Open or create POSIX SHM `/name`. If existing, reattach (size check).
    // Throws std::runtime_error on failure.
    Writer(const std::string& name, std::size_t K);
    ~Writer();
    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;

    // Publish a v2 packet. All arrays must be K-sized + float32 / contiguous.
    // Caller-side validation expected; this is the hot path.
    //
    // Pointers (must be valid for duration of call, no ownership transfer):
    //   kpts_3d_m:    (K, 3) float32
    //   kpts_2d_px:   (K, 2) float32
    //   kp_conf:      (K,)   float32
    //   kp_sigma_m:   (K, 3) float32
    //   pose_cov_diag:(K, 3) float32
    void publish(
        std::uint32_t frame_id,
        std::uint64_t rgb_ts_ns,
        std::uint64_t depth_ts_ns,
        std::uint32_t depth_age_us,
        float         box_conf,
        float         depth_invalid_ratio,
        std::uint8_t  valid_flag,
        std::uint8_t  world_frame,
        std::uint8_t  valid_reason,
        std::uint8_t  ts_domain,
        std::uint64_t publish_done_mono_ns,
        std::uint64_t valid_mask_bits,
        const float*  kpts_3d_m,
        const float*  kpts_2d_px,
        const float*  kp_conf,
        const float*  kp_sigma_m,
        const float*  pose_cov_diag);

    std::size_t size() const { return size_; }
    std::size_t K()    const { return K_; }
    const std::string& name() const { return name_; }

private:
    std::string  name_;
    std::size_t  K_;
    std::size_t  size_;
    int          fd_;
    std::uint8_t* buf_;

    // Cached body offsets
    std::size_t off_kpts_3d_;
    std::size_t off_kpts_2d_;
    std::size_t off_kpt_conf_;
    std::size_t off_kp_sigma_;
    std::size_t off_pose_cov_;
};

}  // namespace shm_v2
}  // namespace hwalker

// SHM v2 writer implementation. See hwalker/shm_v2_writer.hpp.
//
// Critical paths:
//   - publish() is the hot loop (called ~60-120 Hz)
//   - All writes via uintptr_t aligned reinterpret_cast (no struct.pack overhead)
//   - memcpy for body arrays (compiles to SIMD ldp/stp on ARM64)
//
// Memory ordering (Codex review b1ky3965z P1-5):
//   - seq is std::atomic<uint32_t>
//   - store(release) on odd (open) and even (close)
//   - atomic_thread_fence(release) between body writes and seq close
//
// Header writes use plain stores (intentional, performance-critical).
// The release fence guarantees they are globally visible before seq close.
#include "hwalker/shm_v2_writer.hpp"

#include <atomic>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace hwalker {
namespace shm_v2 {

namespace {

inline std::atomic<std::uint32_t>* seq_ptr(std::uint8_t* buf) {
    static_assert(sizeof(std::atomic<std::uint32_t>) == sizeof(std::uint32_t),
                  "atomic<uint32_t> must be word-sized for SHM layout");
    return reinterpret_cast<std::atomic<std::uint32_t>*>(buf + SEQ_OFF);
}

}  // namespace

Writer::Writer(const std::string& name, std::size_t K)
    : name_(name), K_(K), size_(0), fd_(-1), buf_(nullptr) {
    if (K < 1 || K > 64) {
        throw std::invalid_argument("K must be 1..64");
    }
    if (name == "hwalker_pose") {
        throw std::invalid_argument(
            "SHM name 'hwalker_pose' is reserved for v1 mainline");
    }
    size_ = compute_size(K);

    // Body offsets (sequential after header)
    off_kpts_3d_  = HEADER_SIZE;
    off_kpts_2d_  = off_kpts_3d_  + K_ * 3 * sizeof(float);
    off_kpt_conf_ = off_kpts_2d_  + K_ * 2 * sizeof(float);
    off_kp_sigma_ = off_kpt_conf_ + K_ * 1 * sizeof(float);
    off_pose_cov_ = off_kp_sigma_ + K_ * 3 * sizeof(float);

    // POSIX shm_open + ftruncate + mmap
    const std::string path = "/" + name;
    fd_ = ::shm_open(path.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd_ < 0) {
        throw std::runtime_error("shm_open failed for " + path);
    }
    // Reattach safe: check existing size
    struct stat st {};
    if (::fstat(fd_, &st) == 0 && static_cast<std::size_t>(st.st_size) >= size_) {
        // Existing SHM, no need to truncate (avoid data loss for readers)
    } else {
        if (::ftruncate(fd_, static_cast<off_t>(size_)) < 0) {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("ftruncate failed");
        }
    }
    void* p = ::mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (p == MAP_FAILED) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("mmap failed");
    }
    buf_ = static_cast<std::uint8_t*>(p);

    // Initial seqlock init (under seqlock)
    auto* seq = seq_ptr(buf_);
    std::uint32_t cur = seq->load(std::memory_order_acquire);
    std::uint32_t seq_write = ((cur & 1u) == 0u) ? cur + 1u : cur + 2u;
    seq->store(seq_write, std::memory_order_release);  // open (odd)

    // Zero body
    std::memset(buf_ + HEADER_SIZE, 0, size_ - HEADER_SIZE);

    // Stamp version + K
    *reinterpret_cast<std::uint32_t*>(buf_ + VERSION_OFF) = VERSION;
    *reinterpret_cast<std::uint32_t*>(buf_ + K_OFF)       = static_cast<std::uint32_t>(K);
    *reinterpret_cast<std::uint8_t*>(buf_ + VALID_FLAG_OFF) = 0;

    std::atomic_thread_fence(std::memory_order_release);
    seq->store(seq_write + 1u, std::memory_order_release);  // close (even)
}

Writer::~Writer() {
    if (buf_ != nullptr && buf_ != MAP_FAILED) {
        ::munmap(buf_, size_);
    }
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

void Writer::publish(
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
    const float*  pose_cov_diag) {

    auto* seq = seq_ptr(buf_);
    std::uint32_t cur = seq->load(std::memory_order_acquire);
    std::uint32_t seq_write = ((cur & 1u) == 0u) ? cur + 1u : cur + 2u;
    seq->store(seq_write, std::memory_order_release);  // open (odd)

    // ── Header writes (plain stores, release fence later) ───────────────
    *reinterpret_cast<std::uint32_t*>(buf_ + VERSION_OFF)       = VERSION;
    *reinterpret_cast<std::uint32_t*>(buf_ + K_OFF)             = static_cast<std::uint32_t>(K_);
    *reinterpret_cast<std::uint32_t*>(buf_ + FRAME_ID_OFF)      = frame_id;
    *reinterpret_cast<std::uint64_t*>(buf_ + RGB_TS_OFF)        = rgb_ts_ns;
    *reinterpret_cast<std::uint64_t*>(buf_ + DEPTH_TS_OFF)      = depth_ts_ns;
    *reinterpret_cast<std::uint32_t*>(buf_ + DEPTH_AGE_OFF)     = depth_age_us;
    *reinterpret_cast<float*>(buf_ + BOX_CONF_OFF)              = box_conf;
    *reinterpret_cast<float*>(buf_ + DEPTH_INVALID_OFF)         = depth_invalid_ratio;
    *(buf_ + VALID_FLAG_OFF)                                    = valid_flag;
    *(buf_ + WORLD_FRAME_OFF)                                   = world_frame;
    *(buf_ + VALID_REASON_OFF)                                  = valid_reason;
    *(buf_ + TS_DOMAIN_OFF)                                     = ts_domain;
    *reinterpret_cast<std::uint64_t*>(buf_ + PUBLISH_DONE_OFF)  = publish_done_mono_ns;
    *reinterpret_cast<std::uint64_t*>(buf_ + VALID_MASK_BITS_OFF) = valid_mask_bits;

    // ── Body writes (memcpy → SIMD on ARM64) ────────────────────────────
    std::memcpy(buf_ + off_kpts_3d_,  kpts_3d_m,     K_ * 3 * sizeof(float));
    std::memcpy(buf_ + off_kpts_2d_,  kpts_2d_px,    K_ * 2 * sizeof(float));
    std::memcpy(buf_ + off_kpt_conf_, kp_conf,       K_ * 1 * sizeof(float));
    std::memcpy(buf_ + off_kp_sigma_, kp_sigma_m,    K_ * 3 * sizeof(float));
    std::memcpy(buf_ + off_pose_cov_, pose_cov_diag, K_ * 3 * sizeof(float));

    // ── Release fence + seq close ───────────────────────────────────────
    std::atomic_thread_fence(std::memory_order_release);
    seq->store(seq_write + 1u, std::memory_order_release);  // close (even)
}

}  // namespace shm_v2
}  // namespace hwalker

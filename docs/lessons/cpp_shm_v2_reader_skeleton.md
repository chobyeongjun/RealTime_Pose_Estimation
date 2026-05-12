# C++ SHM v2 Reader Skeleton — 사용자 control repo 의 prerequisite

**작성**: 2026-05-12. Codex orchestration `bvfvkxo1m` Q5 + 사용자 control repo 작업 prereq.

이 문서 = **사용자 control repo (C++) 의 SHM v2 reader 작성 의무 spec**.

Plan D EKF (predictor) 의 input contract.

---

## 1. Spec 요약

| Field | 값 |
|---|---|
| Schema version | 2 |
| Header size | 64 bytes exact |
| Total size | HEADER_SIZE + K × 48 bytes (aligned 64) |
| K=6 total | 384 bytes |
| Endian | little-endian |
| Domain | CLOCK_REALTIME (rgb_ts, depth_ts) + CLOCK_MONOTONIC (publish_done) |
| Seqlock | uint32 seq, even=stable, odd=write in progress |

vision repo (제) = publisher. control repo (사용자) = reader.

---

## 2. Header Layout (#pragma pack)

```cpp
// shm_reader_v2.hpp
#pragma once
#include <cstdint>

#pragma pack(push, 1)
struct PlanDPacketHeaderV2 {
    uint32_t seq;                       // even=stable, odd=write
    uint32_t version;                   // MUST == 2
    uint32_t num_keypoints;             // K (1..64)
    uint32_t frame_id;
    uint64_t rgb_ts_ns;                 // CLOCK_REALTIME (T_N, RGB capture)
    uint64_t depth_ts_ns;               // CLOCK_REALTIME (T_{N-1} or T_N)
    uint32_t depth_age_us;              // (rgb_ts - depth_ts) / 1000
    float    box_conf;
    float    depth_invalid_ratio;
    uint8_t  valid_flag;                // 0 or 1 (derived from mask)
    uint8_t  world_frame;               // 0=cam, 1=world (IMU rotated)
    uint8_t  valid_reason;              // enum, see below
    uint8_t  ts_domain;                 // 0 = CLOCK_REALTIME
    uint64_t publish_done_mono_ns;      // CLOCK_MONOTONIC
    uint64_t valid_mask_bits;           // per-kp validity (bit i = keypoint i)
};
#pragma pack(pop)

static_assert(sizeof(PlanDPacketHeaderV2) == 64, "header MUST be 64 bytes");

enum ValidReason : uint8_t {
    VALID_OK              = 0,
    INVALID_NO_DETECTION  = 1,
    INVALID_OCCLUDED      = 2,
    INVALID_BUDGET_EXCEED = 3,
    INVALID_CONSTRAINT    = 4,
    INVALID_WARMUP        = 5,
    INVALID_STALE_DEPTH   = 6,
    INVALID_DRIFT         = 7,
    INVALID_THERMAL       = 8,
    INVALID_UNKNOWN       = 255,
};
```

## 3. Body Layout (K-dependent)

```cpp
// Body offsets (after 64-byte header):
//   float kpts_3d_m   [K][3]   offset = 64
//   float kpts_2d_px  [K][2]   offset = 64 + K*12
//   float kp_conf     [K]      offset = 64 + K*20
//   float kp_sigma_m  [K][3]   offset = 64 + K*24
//   float pose_cov_diag[K][3]  offset = 64 + K*36

struct PlanDPacketBodyOffsets {
    size_t kpts_3d_m;
    size_t kpts_2d_px;
    size_t kp_conf;
    size_t kp_sigma_m;
    size_t pose_cov_diag;
    size_t total_size;
};

inline PlanDPacketBodyOffsets compute_offsets(uint32_t K) {
    PlanDPacketBodyOffsets o;
    o.kpts_3d_m     = 64;
    o.kpts_2d_px    = 64 + K * 12;
    o.kp_conf       = 64 + K * 20;
    o.kp_sigma_m    = 64 + K * 24;
    o.pose_cov_diag = 64 + K * 36;
    size_t raw_size = 64 + K * 48;
    o.total_size = (raw_size + 63) & ~size_t(63);   // 64-aligned
    return o;
}
```

## 4. Reader (POSIX SHM + seqlock)

```cpp
// shm_reader_v2.cpp
#include "shm_reader_v2.hpp"

#include <atomic>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

class PlanDShmReader {
public:
    /// Open existing SHM segment (publisher 가 먼저 create).
    bool open(const char* shm_name = "hwalker_pose_cuda", uint32_t expected_k = 6) {
        // POSIX shm_open
        char path[256];
        snprintf(path, sizeof(path), "/dev/shm/%s", shm_name);
        fd_ = ::open(path, O_RDONLY);
        if (fd_ < 0) return false;

        struct stat st;
        if (fstat(fd_, &st) < 0) return false;
        size_ = st.st_size;

        ptr_ = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (ptr_ == MAP_FAILED) return false;

        // Verify version + K
        const auto* hdr = reinterpret_cast<const PlanDPacketHeaderV2*>(ptr_);
        if (hdr->version != 2) return false;
        if (hdr->num_keypoints != expected_k) return false;

        K_ = expected_k;
        offsets_ = compute_offsets(K_);
        return true;
    }

    /// Read latest stable packet (seqlock retry, max 16 attempts).
    /// Returns true if a stable snapshot was acquired into `out`.
    struct Snapshot {
        PlanDPacketHeaderV2 header;
        std::vector<float> kpts_3d;        // K * 3
        std::vector<float> kpts_2d;        // K * 2
        std::vector<float> kp_conf;        // K
        std::vector<float> kp_sigma_m;     // K * 3
        std::vector<float> pose_cov_diag;  // K * 3
    };

    bool read_latest(Snapshot& out, int max_retries = 16) {
        const auto* hdr_ptr = reinterpret_cast<const PlanDPacketHeaderV2*>(ptr_);
        const std::atomic<uint32_t>* seq_atomic =
            reinterpret_cast<const std::atomic<uint32_t>*>(&hdr_ptr->seq);

        for (int retry = 0; retry < max_retries; ++retry) {
            uint32_t seq0 = seq_atomic->load(std::memory_order_acquire);
            if (seq0 & 1u) continue;   // writer in progress

            // Copy header
            std::memcpy(&out.header, hdr_ptr, sizeof(PlanDPacketHeaderV2));

            // Body (copy under seqlock)
            const uint8_t* base = reinterpret_cast<const uint8_t*>(ptr_);
            const float* p_3d = reinterpret_cast<const float*>(base + offsets_.kpts_3d_m);
            const float* p_2d = reinterpret_cast<const float*>(base + offsets_.kpts_2d_px);
            const float* p_c  = reinterpret_cast<const float*>(base + offsets_.kp_conf);
            const float* p_s  = reinterpret_cast<const float*>(base + offsets_.kp_sigma_m);
            const float* p_pc = reinterpret_cast<const float*>(base + offsets_.pose_cov_diag);

            out.kpts_3d.assign(p_3d, p_3d + K_ * 3);
            out.kpts_2d.assign(p_2d, p_2d + K_ * 2);
            out.kp_conf.assign(p_c, p_c + K_);
            out.kp_sigma_m.assign(p_s, p_s + K_ * 3);
            out.pose_cov_diag.assign(p_pc, p_pc + K_ * 3);

            // Recheck seq
            std::atomic_thread_fence(std::memory_order_acquire);
            uint32_t seq1 = seq_atomic->load(std::memory_order_acquire);
            if (seq0 == seq1 && (seq0 & 1u) == 0) {
                return true;
            }
        }
        return false;   // stale (writer busy)
    }

    /// Per-keypoint validity check.
    bool is_kp_valid(uint64_t valid_mask_bits, uint32_t kp_index) const {
        return (valid_mask_bits >> kp_index) & 1u;
    }

    void close() {
        if (ptr_ && ptr_ != MAP_FAILED) munmap(ptr_, size_);
        if (fd_ >= 0) ::close(fd_);
        ptr_ = nullptr;
        fd_ = -1;
    }

    ~PlanDShmReader() { close(); }

private:
    int fd_ = -1;
    void* ptr_ = nullptr;
    size_t size_ = 0;
    uint32_t K_ = 0;
    PlanDPacketBodyOffsets offsets_;
};
```

## 5. Plan D EKF Input Integration

```cpp
// Plan D EKF state: x = [phi, omega, alpha]
// Measurement: q = [hip_flex_L, knee_L, ankle_L, hip_flex_R, knee_R, ankle_R]

void plan_d_step(PlanDShmReader& reader, EKF& ekf) {
    PlanDShmReader::Snapshot snap;
    if (!reader.read_latest(snap)) {
        ekf.no_measurement_step();
        return;
    }

    // Stale depth check (Codex Q5 contract enforcement)
    if (snap.header.depth_age_us > 16700) {
        ekf.no_measurement_step();
        return;
    }

    // 모든 keypoint invalid → no measurement
    if (snap.header.valid_mask_bits == 0 || !snap.header.valid_flag) {
        ekf.no_measurement_step();
        return;
    }

    // Convert kpts_3d → joint angles (lower-limb6 schema)
    JointAngles q;
    q.hip_L  = compute_hip_angle(snap.kpts_3d[0*3+0], ...);   // L_hip
    q.knee_L = compute_knee_angle(snap.kpts_3d[2*3+0], ...);  // L_knee
    // ...

    // Per-keypoint covariance — EKF measurement R = diag(pose_cov_diag * J^T J)
    Eigen::MatrixXd R = ekf.build_R_from_per_kp_cov(snap.pose_cov_diag);

    // Two-timestamp handling (Codex Q4):
    // Option A: use rgb_ts_ns as primary, inflate R by velocity² × depth_age²
    double depth_age_s = snap.header.depth_age_us / 1e6;
    if (depth_age_s > 0) {
        double sigma_extra = ekf.estimate_velocity() * depth_age_s;
        R += Eigen::MatrixXd::Identity(R.rows(), R.cols()) * sigma_extra * sigma_extra;
    }

    ekf.update(q, R, snap.header.rgb_ts_ns);
}
```

## 6. Failure modes (Codex Q5 + Q7)

| Reason | Action |
|---|---|
| `valid_flag = 0` | EKF no_measurement_step |
| `depth_age_us > 16700` | EKF no_measurement_step, log warning |
| Snapshot stale (16 retries fail) | use previous EKF state, increment stale_count |
| `version != 2` | safe fallback, alert (publisher upgrade required) |
| Innovation residual > 3σ | reject, log "phase divergence" |
| Stale > 50ms | EKF fallback L3 → L2 → L1 → hold → pretension |

## 7. Build (사용자 control repo)

```bash
g++ -std=c++17 -O2 -pthread \
    -I/path/to/eigen3 \
    shm_reader_v2.cpp main.cpp \
    -lrt \
    -o plan_d_control
```

## 8. Test (사용자 control repo)

```cpp
// tests/test_shm_reader_v2.cpp
TEST(PlanDReader, RoundTrip) {
    // 1. Run vision repo 의 publisher (Python or scripts/verify_shm_v2.py)
    // 2. Open + read_latest
    PlanDShmReader reader;
    ASSERT_TRUE(reader.open("hwalker_pose_cuda", 6));
    PlanDShmReader::Snapshot snap;
    ASSERT_TRUE(reader.read_latest(snap));
    EXPECT_EQ(snap.header.version, 2);
    EXPECT_EQ(snap.header.num_keypoints, 6);
}
```

---

## 9. References

- `docs/lessons/shm_v2_packet_spec.md` — full spec (vision repo)
- `src/perception/CUDA_Stream/shm_publisher.py` — Python publisher (vision repo)
- Codex orchestration `bvfvkxo1m` Q5
- Codex review `b1ky3965z` P1-5 (ARM Orin memory ordering note)

---

*Last updated: 2026-05-12. vision ↔ control repo sync 의 의무.*

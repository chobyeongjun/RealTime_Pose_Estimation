/**
 * pose_postprocess.cpp — C++ 후처리 확장 모듈
 * =============================================
 * Python 후처리 파이프라인의 성능 핵심 부분을 C++로 구현.
 *
 * 포함 기능:
 *   1. batch_sample_depth_patch: depth map에서 여러 키포인트의 패치 중앙값 일괄 추출
 *   2. batch_pixel_to_3d: 2D 픽셀 좌표 배열 → 3D 좌표 배열 (카메라 역투영)
 *   3. OneEuroFilter1D: 1D One Euro Filter (적응형 저역 통과)
 *   4. Joint3DFilter: 3D 키포인트 필터링 + 보간 + 세그먼트 제약
 *   5. compute_angle_3d: 3D 관절 각도 계산
 *   6. compute_lower_limb_angles: 하체 관절 각도 일괄 계산
 *
 * 빌드:
 *   pip install pybind11
 *   python setup.py build_ext --inplace
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <limits>
#include <tuple>

namespace py = pybind11;

// ============================================================================
// 1. Depth Patch Sampling (배치)
// ============================================================================

/**
 * 단일 키포인트의 패치 중앙값 depth 추출.
 * Python의 _sample_depth_patch와 동일한 로직.
 */
static float sample_depth_patch(const float* depth_data, int h, int w,
                                int x, int y, int patch_radius) {
    int y0 = std::max(0, y - patch_radius);
    int y1 = std::min(h, y + patch_radius + 1);
    int x0 = std::max(0, x - patch_radius);
    int x1 = std::min(w, x + patch_radius + 1);

    std::vector<float> valid;
    valid.reserve((y1 - y0) * (x1 - x0));

    for (int r = y0; r < y1; ++r) {
        for (int c = x0; c < x1; ++c) {
            float v = depth_data[r * w + c];
            if (std::isfinite(v) && v > 0.0f) {
                valid.push_back(v);
            }
        }
    }

    if (valid.empty()) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    // 중앙값 계산 (partial sort로 O(n) 평균)
    size_t mid = valid.size() / 2;
    std::nth_element(valid.begin(), valid.begin() + mid, valid.end());
    return valid[mid];
}

/**
 * 여러 키포인트의 depth를 한번에 추출.
 *
 * Args:
 *   depth_map: (H, W) float32 depth map
 *   coords: (N, 2) int32 또는 float — 각 행이 (x, y) 픽셀 좌표
 *   patch_radius: 패치 반경 (기본 3 → 7×7)
 *
 * Returns:
 *   (N,) float32 배열 — 각 키포인트의 패치 중앙값 depth (무효 시 NaN)
 */
static py::array_t<float> batch_sample_depth_patch(
        py::array_t<float, py::array::c_style | py::array::forcecast> depth_map,
        py::array_t<float, py::array::c_style | py::array::forcecast> coords,
        int patch_radius = 3) {

    auto depth_buf = depth_map.unchecked<2>();
    auto coords_buf = coords.unchecked<2>();
    int h = depth_buf.shape(0);
    int w = depth_buf.shape(1);
    int n = coords_buf.shape(0);

    const float* depth_data = depth_buf.data(0, 0);

    py::array_t<float> result(n);
    float* result_ptr = result.mutable_data();

    for (int i = 0; i < n; ++i) {
        int x = static_cast<int>(coords_buf(i, 0));
        int y = static_cast<int>(coords_buf(i, 1));

        if (x < 0 || x >= w || y < 0 || y >= h) {
            result_ptr[i] = std::numeric_limits<float>::quiet_NaN();
        } else {
            result_ptr[i] = sample_depth_patch(depth_data, h, w, x, y, patch_radius);
        }
    }

    return result;
}


// ============================================================================
// 2. Batch Pixel-to-3D
// ============================================================================

/**
 * 2D 픽셀 좌표 배열을 3D 좌표로 일괄 변환.
 *
 * Args:
 *   coords: (N, 2) float — 각 행이 (x, y) 픽셀 좌표
 *   depths: (N,) float — 각 키포인트의 depth (meters)
 *   fx, fy, cx, cy: 카메라 intrinsic 파라미터
 *
 * Returns:
 *   (N, 3) float32 배열 — 각 행이 (X, Y, Z) 3D 좌표
 *   depth가 NaN이거나 <= 0이면 해당 행은 (NaN, NaN, NaN)
 */
static py::array_t<float> batch_pixel_to_3d(
        py::array_t<float, py::array::c_style | py::array::forcecast> coords,
        py::array_t<float, py::array::c_style | py::array::forcecast> depths,
        float fx, float fy, float cx, float cy) {

    auto coords_buf = coords.unchecked<2>();
    auto depths_buf = depths.unchecked<1>();
    int n = coords_buf.shape(0);

    py::array_t<float> result({n, 3});
    auto result_buf = result.mutable_unchecked<2>();

    for (int i = 0; i < n; ++i) {
        float z = depths_buf(i);
        if (!std::isfinite(z) || z <= 0.0f) {
            result_buf(i, 0) = std::numeric_limits<float>::quiet_NaN();
            result_buf(i, 1) = std::numeric_limits<float>::quiet_NaN();
            result_buf(i, 2) = std::numeric_limits<float>::quiet_NaN();
        } else {
            float px = coords_buf(i, 0);
            float py_coord = coords_buf(i, 1);
            result_buf(i, 0) = (px - cx) * z / fx;
            result_buf(i, 1) = (py_coord - cy) * z / fy;
            result_buf(i, 2) = z;
        }
    }

    return result;
}


// ============================================================================
// 3. One Euro Filter 1D
// ============================================================================

class OneEuroFilter1D {
public:
    float min_cutoff;
    float beta;
    float d_cutoff;
    float x_prev;
    float dx_prev;
    float t_prev;

    OneEuroFilter1D(float t0, float x0,
                    float min_cutoff_ = 1.0f, float beta_ = 0.007f,
                    float d_cutoff_ = 1.0f)
        : min_cutoff(min_cutoff_), beta(beta_), d_cutoff(d_cutoff_),
          x_prev(x0), dx_prev(0.0f), t_prev(t0) {}

    float operator()(float t, float x) {
        float te = t - t_prev;
        if (te <= 0.0f) return x;

        float a_d = smoothing_factor(te, d_cutoff);
        float dx = (x - x_prev) / te;
        float dx_hat = a_d * dx + (1.0f - a_d) * dx_prev;

        float cutoff = min_cutoff + beta * std::fabs(dx_hat);
        float a = smoothing_factor(te, cutoff);
        float x_hat = a * x + (1.0f - a) * x_prev;

        x_prev = x_hat;
        dx_prev = dx_hat;
        t_prev = t;
        return x_hat;
    }

private:
    static float smoothing_factor(float te, float cutoff) {
        float r = 2.0f * static_cast<float>(M_PI) * cutoff * te;
        return r / (r + 1.0f);
    }
};


// ============================================================================
// 4. Joint 3D Filter (필터링 + 보간 + 세그먼트 제약)
// ============================================================================

struct Joint3DState {
    OneEuroFilter1D fx, fy, fz;
    float prev_x, prev_y, prev_z;
    float vel_x, vel_y, vel_z;
    float prev_t;
    int missing_count;
    bool initialized;

    Joint3DState(float t0, float x0, float y0, float z0,
                 float min_cutoff, float beta, float d_cutoff)
        : fx(t0, x0, min_cutoff, beta, d_cutoff),
          fy(t0, y0, min_cutoff, beta, d_cutoff),
          fz(t0, z0, min_cutoff, beta, d_cutoff),
          prev_x(x0), prev_y(y0), prev_z(z0),
          vel_x(0), vel_y(0), vel_z(0),
          prev_t(t0), missing_count(0), initialized(true) {}
};

// 세그먼트 정의: (parent, child)
static const std::vector<std::pair<std::string, std::string>> SEGMENTS_3D = {
    {"left_hip", "left_knee"},
    {"left_knee", "left_ankle"},
    {"right_hip", "right_knee"},
    {"right_knee", "right_ankle"},
};

class Joint3DFilter {
public:
    float min_cutoff, beta, d_cutoff;
    int max_missing;
    int calib_frames;
    float tolerance;

    Joint3DFilter(float min_cutoff_ = 0.5f, float beta_ = 0.01f,
                  float d_cutoff_ = 1.0f, int max_missing_ = 5,
                  int calib_frames_ = 30, float tolerance_ = 0.20f)
        : min_cutoff(min_cutoff_), beta(beta_), d_cutoff(d_cutoff_),
          max_missing(max_missing_), calib_frames(calib_frames_),
          tolerance(tolerance_), calib_count_(0), calib_done_(false) {}

    /**
     * 단일 키포인트 3D 필터링.
     * pt3d가 비어있으면(길이 0) depth 무효 → 보간 시도.
     *
     * Returns: (3,) float array 또는 빈 array (필터 불가 시)
     */
    py::array_t<float> filter(const std::string& name,
                              py::array_t<float> pt3d,
                              float t) {
        auto buf = pt3d.unchecked<1>();
        bool valid = (buf.shape(0) == 3) &&
                     std::isfinite(buf(0)) && std::isfinite(buf(1)) && std::isfinite(buf(2));

        if (valid) {
            float x = buf(0), y = buf(1), z = buf(2);
            auto it = states_.find(name);

            if (it == states_.end()) {
                // 첫 프레임: 필터 초기화
                states_.emplace(name, Joint3DState(
                    t, x, y, z, min_cutoff, beta, d_cutoff));
                return make_array(x, y, z);
            }

            auto& s = it->second;
            s.missing_count = 0;

            float xf = s.fx(t, x);
            float yf = s.fy(t, y);
            float zf = s.fz(t, z);

            float dt = t - s.prev_t;
            if (dt > 0) {
                s.vel_x = (xf - s.prev_x) / dt;
                s.vel_y = (yf - s.prev_y) / dt;
                s.vel_z = (zf - s.prev_z) / dt;
            }

            s.prev_x = xf;
            s.prev_y = yf;
            s.prev_z = zf;
            s.prev_t = t;

            return make_array(xf, yf, zf);
        }

        // depth 무효 → 보간
        auto it = states_.find(name);
        if (it == states_.end()) {
            return py::array_t<float>(0);  // 빈 배열
        }

        auto& s = it->second;
        s.missing_count++;

        if (s.missing_count > max_missing) {
            return py::array_t<float>(0);
        }

        float dt = t - s.prev_t;
        float px = s.prev_x + s.vel_x * dt;
        float py_val = s.prev_y + s.vel_y * dt;
        float pz = s.prev_z + s.vel_z * dt;

        s.prev_x = px;
        s.prev_y = py_val;
        s.prev_z = pz;
        s.prev_t = t;

        return make_array(px, py_val, pz);
    }

    /**
     * 3D 세그먼트 길이 제약 적용.
     * keypoints_3d: dict {name → (X, Y, Z)} — Python dict를 받아서 수정 후 반환.
     */
    py::dict apply_segment_constraint(py::dict keypoints_3d) {
        if (!calib_done_) {
            collect_3d_sample(keypoints_3d);
            return keypoints_3d;
        }

        for (auto& [parent, child] : SEGMENTS_3D) {
            auto key = std::make_pair(parent, child);
            auto ref_it = seg_ref_.find(parent + ":" + child);
            if (ref_it == seg_ref_.end()) continue;

            if (!keypoints_3d.contains(parent.c_str()) ||
                !keypoints_3d.contains(child.c_str())) continue;

            auto p_tuple = keypoints_3d[parent.c_str()].cast<py::tuple>();
            auto c_tuple = keypoints_3d[child.c_str()].cast<py::tuple>();

            float px = p_tuple[0].cast<float>();
            float py_val = p_tuple[1].cast<float>();
            float pz = p_tuple[2].cast<float>();
            float cx = c_tuple[0].cast<float>();
            float cy = c_tuple[1].cast<float>();
            float cz = c_tuple[2].cast<float>();

            float dx = cx - px, dy = cy - py_val, dz = cz - pz;
            float cur_len = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (cur_len < 1e-6f) continue;

            float ref_len = ref_it->second;
            float min_len = ref_len * (1.0f - tolerance);
            float max_len = ref_len * (1.0f + tolerance);

            if (cur_len >= min_len && cur_len <= max_len) continue;

            float target_len = std::max(min_len, std::min(max_len, cur_len));
            float scale = target_len / cur_len;
            float nx = px + dx * scale;
            float ny = py_val + dy * scale;
            float nz = pz + dz * scale;

            keypoints_3d[child.c_str()] = py::make_tuple(nx, ny, nz);

            // 내부 상태 업데이트
            auto state_it = states_.find(child);
            if (state_it != states_.end()) {
                state_it->second.prev_x = nx;
                state_it->second.prev_y = ny;
                state_it->second.prev_z = nz;
            }
        }

        return keypoints_3d;
    }

    bool calibrated() const { return calib_done_; }

    float calib_progress() const {
        if (calib_done_) return 1.0f;
        return std::min(1.0f, static_cast<float>(calib_count_) / calib_frames);
    }

    py::dict get_ref_lengths_3d() const {
        py::dict result;
        for (auto& [key, length] : seg_ref_) {
            result[key.c_str()] = length;
        }
        return result;
    }

    void reset() {
        states_.clear();
    }

private:
    std::unordered_map<std::string, Joint3DState> states_;
    int calib_count_;
    bool calib_done_;
    // key = "parent:child", value = [lengths]
    std::unordered_map<std::string, std::vector<float>> seg_samples_;
    std::unordered_map<std::string, float> seg_ref_;

    static py::array_t<float> make_array(float x, float y, float z) {
        py::array_t<float> arr(3);
        float* ptr = arr.mutable_data();
        ptr[0] = x; ptr[1] = y; ptr[2] = z;
        return arr;
    }

    void collect_3d_sample(py::dict keypoints_3d) {
        int valid_count = 0;
        for (auto& [parent, child] : SEGMENTS_3D) {
            if (!keypoints_3d.contains(parent.c_str()) ||
                !keypoints_3d.contains(child.c_str())) continue;

            auto p_tuple = keypoints_3d[parent.c_str()].cast<py::tuple>();
            auto c_tuple = keypoints_3d[child.c_str()].cast<py::tuple>();

            float px = p_tuple[0].cast<float>();
            float py_val = p_tuple[1].cast<float>();
            float pz = p_tuple[2].cast<float>();
            float cx = c_tuple[0].cast<float>();
            float cy = c_tuple[1].cast<float>();
            float cz = c_tuple[2].cast<float>();

            float dx = cx - px, dy = cy - py_val, dz = cz - pz;
            float length = std::sqrt(dx*dx + dy*dy + dz*dz);

            if (length > 0.05f) {
                std::string key = parent + ":" + child;
                seg_samples_[key].push_back(length);
                valid_count++;
            }
        }

        if (valid_count >= 3) {
            calib_count_++;
        }

        if (calib_count_ >= calib_frames) {
            finalize_3d_calibration();
        }
    }

    void finalize_3d_calibration() {
        for (auto& [key, lengths] : seg_samples_) {
            if (lengths.size() >= 5) {
                std::sort(lengths.begin(), lengths.end());
                seg_ref_[key] = lengths[lengths.size() / 2];
            }
        }

        if (seg_ref_.size() >= 3) {
            calib_done_ = true;
            py::print("[Joint3DFilter C++] 3D 뼈 길이 캘리브레이션 완료 (",
                      calib_count_, "프레임)");
            for (auto& [key, length] : seg_ref_) {
                py::print("  ", key, ":", length, "m");
            }
        } else {
            calib_count_ = std::max(0, calib_count_ - 10);
        }

        seg_samples_.clear();
    }
};


// ============================================================================
// 5. Joint Angle Computation
// ============================================================================

/**
 * 3개의 3D 점에서 관절 각도 계산 (라디안).
 * pose_processor_node.cpp:108-118의 computeAngle과 동일.
 */
static float compute_angle_3d_impl(float px, float py_val, float pz,
                                    float jx, float jy, float jz,
                                    float cx, float cy, float cz) {
    float v1x = px - jx, v1y = py_val - jy, v1z = pz - jz;
    float v2x = cx - jx, v2y = cy - jy, v2z = cz - jz;

    float n1 = std::sqrt(v1x*v1x + v1y*v1y + v1z*v1z);
    float n2 = std::sqrt(v2x*v2x + v2y*v2y + v2z*v2z);

    if (n1 < 1e-9f || n2 < 1e-9f) return 0.0f;

    float dot = (v1x*v2x + v1y*v2y + v1z*v2z) / (n1 * n2);
    dot = std::max(-1.0f, std::min(1.0f, dot));
    return std::acos(dot);
}

/**
 * Python에서 호출 가능한 3D 각도 계산.
 */
static float compute_angle_3d(py::array_t<float> parent,
                               py::array_t<float> joint,
                               py::array_t<float> child) {
    auto p = parent.unchecked<1>();
    auto j = joint.unchecked<1>();
    auto c = child.unchecked<1>();
    return compute_angle_3d_impl(
        p(0), p(1), p(2), j(0), j(1), j(2), c(0), c(1), c(2));
}


// ============================================================================
// 6. Lower Limb Angles (일괄 계산)
// ============================================================================

struct AngleDef {
    std::string name;
    std::string parent;
    std::string joint;
    std::string child;
};

static const std::vector<AngleDef> LOWER_LIMB_ANGLE_DEFS = {
    {"left_knee_flexion",  "left_hip",   "left_knee",  "left_ankle"},
    {"right_knee_flexion", "right_hip",  "right_knee", "right_ankle"},
    {"left_hip_flexion",   "left_knee",  "left_hip",   "right_hip"},
    {"right_hip_flexion",  "right_knee", "right_hip",  "left_hip"},
    {"left_ankle_dorsiflexion",  "left_knee",  "left_ankle", "left_toe"},
    {"right_ankle_dorsiflexion", "right_knee", "right_ankle", "right_toe"},
};

/**
 * PoseResult의 keypoints와 confidences에서 하체 관절 각도를 일괄 계산.
 *
 * Args:
 *   keypoints: dict {name → tuple(x,y,z) 또는 tuple(x,y)}
 *   confidences: dict {name → float}
 *   use_3d: True면 3D, False면 2D
 *
 * Returns:
 *   dict {angle_name → angle_degrees}
 */
static py::dict compute_lower_limb_angles(py::dict keypoints,
                                           py::dict confidences,
                                           bool use_3d = true) {
    py::dict angles;
    constexpr float PI = static_cast<float>(M_PI);

    for (auto& def : LOWER_LIMB_ANGLE_DEFS) {
        if (!keypoints.contains(def.parent.c_str()) ||
            !keypoints.contains(def.joint.c_str()) ||
            !keypoints.contains(def.child.c_str())) {
            continue;
        }

        // confidence 체크
        float min_conf = 1.0f;
        for (auto& kp_name : {def.parent, def.joint, def.child}) {
            float c = 0.0f;
            if (confidences.contains(kp_name.c_str())) {
                c = confidences[kp_name.c_str()].cast<float>();
            }
            min_conf = std::min(min_conf, c);
        }
        if (min_conf < 0.3f) continue;

        auto p_tuple = keypoints[def.parent.c_str()].cast<py::tuple>();
        auto j_tuple = keypoints[def.joint.c_str()].cast<py::tuple>();
        auto c_tuple = keypoints[def.child.c_str()].cast<py::tuple>();

        float angle_rad;

        if (use_3d) {
            angle_rad = compute_angle_3d_impl(
                p_tuple[0].cast<float>(), p_tuple[1].cast<float>(), p_tuple[2].cast<float>(),
                j_tuple[0].cast<float>(), j_tuple[1].cast<float>(), j_tuple[2].cast<float>(),
                c_tuple[0].cast<float>(), c_tuple[1].cast<float>(), c_tuple[2].cast<float>());
        } else {
            // 2D: z = 0
            angle_rad = compute_angle_3d_impl(
                p_tuple[0].cast<float>(), p_tuple[1].cast<float>(), 0.0f,
                j_tuple[0].cast<float>(), j_tuple[1].cast<float>(), 0.0f,
                c_tuple[0].cast<float>(), c_tuple[1].cast<float>(), 0.0f);
        }

        float angle_deg;
        if (def.name.find("knee_flexion") != std::string::npos) {
            angle_deg = (PI - angle_rad) * 180.0f / PI;
        } else if (def.name.find("ankle_dorsiflexion") != std::string::npos) {
            angle_deg = (angle_rad - PI / 2.0f) * 180.0f / PI;
        } else {
            angle_deg = angle_rad * 180.0f / PI;
        }

        angles[def.name.c_str()] = angle_deg;
    }

    return angles;
}


// ============================================================================
// 7. Full Post-processing Pipeline (한번에 처리)
// ============================================================================

/**
 * 2D 키포인트 + depth map → 3D 좌표 일괄 변환.
 * Python for문 17회 대신 C++ 단일 호출.
 *
 * Args:
 *   keypoint_names: 키포인트 이름 리스트
 *   coords: (N, 2) float — 각 행이 (px, py) (크롭 오프셋 적용 후)
 *   depth_map: (H, W) float32
 *   fx, fy, cx, cy: 카메라 intrinsics
 *   patch_radius: depth 패치 반경
 *
 * Returns:
 *   dict {name → tuple(X, Y, Z)} — depth 무효인 키포인트는 제외
 */
static py::dict batch_2d_to_3d(
        std::vector<std::string> keypoint_names,
        py::array_t<float, py::array::c_style | py::array::forcecast> coords,
        py::array_t<float, py::array::c_style | py::array::forcecast> depth_map,
        float fx, float fy, float cx, float cy,
        int patch_radius = 3) {

    auto depth_buf = depth_map.unchecked<2>();
    auto coords_buf = coords.unchecked<2>();
    int h = depth_buf.shape(0);
    int w = depth_buf.shape(1);
    int n = coords_buf.shape(0);
    const float* depth_data = depth_buf.data(0, 0);

    py::dict result;

    for (int i = 0; i < n && i < static_cast<int>(keypoint_names.size()); ++i) {
        int x = static_cast<int>(coords_buf(i, 0));
        int y = static_cast<int>(coords_buf(i, 1));

        if (x < 0 || x >= w || y < 0 || y >= h) continue;

        float z = sample_depth_patch(depth_data, h, w, x, y, patch_radius);
        if (!std::isfinite(z) || z <= 0.0f) continue;

        float x3d = (static_cast<float>(x) - cx) * z / fx;
        float y3d = (static_cast<float>(y) - cy) * z / fy;

        result[keypoint_names[i].c_str()] = py::make_tuple(x3d, y3d, z);
    }

    return result;
}


// ============================================================================
// pybind11 Module Definition
// ============================================================================

PYBIND11_MODULE(pose_postprocess_cpp, m) {
    m.doc() = "C++ 후처리 확장 모듈 — depth 패치 샘플링, 3D 변환, "
              "One Euro Filter, Joint 3D Filter, 관절 각도 계산";

    // Batch functions
    m.def("batch_sample_depth_patch", &batch_sample_depth_patch,
          py::arg("depth_map"), py::arg("coords"), py::arg("patch_radius") = 3,
          "depth map에서 여러 키포인트의 패치 중앙값 일괄 추출");

    m.def("batch_pixel_to_3d", &batch_pixel_to_3d,
          py::arg("coords"), py::arg("depths"),
          py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
          "2D 픽셀 좌표 배열 → 3D 좌표 배열 (카메라 역투영)");

    m.def("batch_2d_to_3d", &batch_2d_to_3d,
          py::arg("keypoint_names"), py::arg("coords"), py::arg("depth_map"),
          py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
          py::arg("patch_radius") = 3,
          "2D 키포인트 + depth → 3D dict 일괄 변환");

    // Angle computation
    m.def("compute_angle_3d", &compute_angle_3d,
          py::arg("parent"), py::arg("joint"), py::arg("child"),
          "3D 관절 각도 계산 (라디안)");

    m.def("compute_lower_limb_angles", &compute_lower_limb_angles,
          py::arg("keypoints"), py::arg("confidences"),
          py::arg("use_3d") = true,
          "하체 관절 각도 일괄 계산 (도)");

    // Joint3DFilter class
    py::class_<Joint3DFilter>(m, "Joint3DFilter",
        "3D 키포인트 필터링 + 보간 + 세그먼트 제약")
        .def(py::init<float, float, float, int, int, float>(),
             py::arg("min_cutoff") = 0.5f, py::arg("beta") = 0.01f,
             py::arg("d_cutoff") = 1.0f, py::arg("max_missing") = 5,
             py::arg("calib_frames") = 30, py::arg("tolerance") = 0.20f)
        .def("filter", &Joint3DFilter::filter,
             py::arg("name"), py::arg("pt3d"), py::arg("t"),
             "단일 키포인트 3D 필터링")
        .def("apply_segment_constraint", &Joint3DFilter::apply_segment_constraint,
             py::arg("keypoints_3d"),
             "3D 뼈 길이 제약 적용")
        .def_property_readonly("calibrated", &Joint3DFilter::calibrated)
        .def_property_readonly("calib_progress", &Joint3DFilter::calib_progress)
        .def("get_ref_lengths_3d", &Joint3DFilter::get_ref_lengths_3d)
        .def("reset", &Joint3DFilter::reset);
}

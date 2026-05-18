// pybind11 binding: hwalker_shm_v2_writer Python module.
//
// Usage from Python:
//   import hwalker_shm_v2_writer as cpp_writer
//   w = cpp_writer.Writer("hwalker_pose_cuda", K=6)
//   w.publish(
//       frame_id, rgb_ts_ns, depth_ts_ns, depth_age_us,
//       box_conf, depth_invalid_ratio,
//       valid_flag, world_frame, valid_reason, ts_domain,
//       publish_done_mono_ns, valid_mask_bits,
//       kpts_3d_m,    # (K, 3) float32
//       kpts_2d_px,   # (K, 2) float32
//       kp_conf,      # (K,)   float32
//       kp_sigma_m,   # (K, 3) float32
//       pose_cov_diag # (K, 3) float32
//   )
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "hwalker/shm_v2_writer.hpp"

namespace py = pybind11;
using hwalker::shm_v2::Writer;

namespace {

void check_array_f32(const py::array_t<float, py::array::c_style | py::array::forcecast>& arr,
                     std::size_t expected_dim0, std::size_t expected_dim1,
                     const char* name) {
    if (arr.ndim() != (expected_dim1 == 0 ? 1 : 2)) {
        throw std::invalid_argument(std::string(name) + " must be 1D or 2D");
    }
    if (static_cast<std::size_t>(arr.shape(0)) != expected_dim0) {
        throw std::invalid_argument(std::string(name) + " dim0 mismatch");
    }
    if (expected_dim1 > 0 && static_cast<std::size_t>(arr.shape(1)) != expected_dim1) {
        throw std::invalid_argument(std::string(name) + " dim1 mismatch");
    }
}

void publish_wrapper(
    Writer& w,
    std::uint32_t frame_id,
    std::uint64_t rgb_ts_ns,
    std::uint64_t depth_ts_ns,
    std::uint32_t depth_age_us,
    float box_conf,
    float depth_invalid_ratio,
    std::uint8_t valid_flag,
    std::uint8_t world_frame,
    std::uint8_t valid_reason,
    std::uint8_t ts_domain,
    std::uint64_t publish_done_mono_ns,
    std::uint64_t valid_mask_bits,
    py::array_t<float, py::array::c_style | py::array::forcecast> kpts_3d_m,
    py::array_t<float, py::array::c_style | py::array::forcecast> kpts_2d_px,
    py::array_t<float, py::array::c_style | py::array::forcecast> kp_conf,
    py::array_t<float, py::array::c_style | py::array::forcecast> kp_sigma_m,
    py::array_t<float, py::array::c_style | py::array::forcecast> pose_cov_diag) {

    const std::size_t K = w.K();
    check_array_f32(kpts_3d_m,     K, 3, "kpts_3d_m");
    check_array_f32(kpts_2d_px,    K, 2, "kpts_2d_px");
    check_array_f32(kp_conf,       K, 0, "kp_conf");
    check_array_f32(kp_sigma_m,    K, 3, "kp_sigma_m");
    check_array_f32(pose_cov_diag, K, 3, "pose_cov_diag");

    {
        py::gil_scoped_release release;  // hot path — no Python state needed
        w.publish(
            frame_id, rgb_ts_ns, depth_ts_ns, depth_age_us,
            box_conf, depth_invalid_ratio,
            valid_flag, world_frame, valid_reason, ts_domain,
            publish_done_mono_ns, valid_mask_bits,
            kpts_3d_m.data(), kpts_2d_px.data(), kp_conf.data(),
            kp_sigma_m.data(), pose_cov_diag.data());
    }
}

}  // namespace

PYBIND11_MODULE(hwalker_shm_v2_writer, m) {
    m.doc() = "C++ SHM v2 writer — drop-in replacement for shm_publisher.py hot path.";

    py::class_<Writer>(m, "Writer")
        .def(py::init<const std::string&, std::size_t>(),
             py::arg("name"), py::arg("K"))
        .def_property_readonly("name", &Writer::name)
        .def_property_readonly("K", &Writer::K)
        .def_property_readonly("size", &Writer::size)
        .def("publish", &publish_wrapper,
             py::arg("frame_id"),
             py::arg("rgb_ts_ns"),
             py::arg("depth_ts_ns"),
             py::arg("depth_age_us"),
             py::arg("box_conf"),
             py::arg("depth_invalid_ratio"),
             py::arg("valid_flag"),
             py::arg("world_frame"),
             py::arg("valid_reason"),
             py::arg("ts_domain"),
             py::arg("publish_done_mono_ns"),
             py::arg("valid_mask_bits"),
             py::arg("kpts_3d_m"),
             py::arg("kpts_2d_px"),
             py::arg("kp_conf"),
             py::arg("kp_sigma_m"),
             py::arg("pose_cov_diag"));
}

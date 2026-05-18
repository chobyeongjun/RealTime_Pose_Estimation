// pybind11 binding for TrtRunner — Python drop-in replacement.
//
// Usage from Python:
//   import hwalker_trt_runner
//   runner = hwalker_trt_runner.TrtRunner("yolo26s-lower6-v2.engine")
//   # input: pre-allocated GPU tensor (e.g. torch.empty(shape, device='cuda'))
//   # output: pre-allocated GPU tensor
//   runner.infer(input.data_ptr(), output.data_ptr(), stream_ptr_or_0)
//   # or sync (for smoke test):
//   runner.infer_sync(input.data_ptr(), output.data_ptr())
//
// Note: data_ptr() returns uint64 CUDA pointer. Stream similarly.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hwalker/trt_runner.hpp"

namespace py = pybind11;
using hwalker::TrtRunner;

PYBIND11_MODULE(hwalker_trt_runner, m) {
    m.doc() = "C++ TensorRT inference runner — Sprint 1 Phase 2 Week 1.";

    py::class_<TrtRunner>(m, "TrtRunner")
        .def(py::init<const std::string&>(), py::arg("engine_path"),
             "Load .engine file and create execution context. Throws on failure.")
        .def("infer",
             [](TrtRunner& self,
                std::uint64_t input_ptr,
                std::uint64_t output_ptr,
                std::uint64_t stream) {
                 // Release GIL during TRT execution (CUDA async)
                 bool ok;
                 {
                     py::gil_scoped_release release;
                     ok = self.infer(input_ptr, output_ptr, stream);
                 }
                 return ok;
             },
             py::arg("input_gpu_ptr"),
             py::arg("output_gpu_ptr"),
             py::arg("stream") = 0,
             "Async inference on given CUDA stream. Returns True on success.")
        .def("infer_sync",
             [](TrtRunner& self,
                std::uint64_t input_ptr,
                std::uint64_t output_ptr) {
                 bool ok;
                 {
                     py::gil_scoped_release release;
                     ok = self.infer_sync(input_ptr, output_ptr);
                 }
                 return ok;
             },
             py::arg("input_gpu_ptr"),
             py::arg("output_gpu_ptr"),
             "Synchronous inference (waits on default stream). For smoke test.")
        .def_property_readonly("input_name",   &TrtRunner::input_name)
        .def_property_readonly("output_name",  &TrtRunner::output_name)
        .def_property_readonly("input_shape",  &TrtRunner::input_shape)
        .def_property_readonly("output_shape", &TrtRunner::output_shape)
        .def_property_readonly("input_bytes",  &TrtRunner::input_bytes)
        .def_property_readonly("output_bytes", &TrtRunner::output_bytes)
        .def_property_readonly("engine_path",  &TrtRunner::engine_path);
}

// pybind11 binding for CudaPreprocessor (Sprint 1 Phase 2 Week 3).
#include "hwalker/cuda_preprocess.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

PYBIND11_MODULE(hwalker_cuda_preprocess, m) {
    m.doc() = "CUDA letterbox preprocessing for YOLO inference (H-Walker Sprint 1 Phase 2 Week 3)";

    py::class_<hwalker::CudaPreprocessor>(m, "CudaPreprocessor")
        .def(py::init<int, int, int, int>(),
             py::arg("imgsz"),
             py::arg("max_h") = 1200,
             py::arg("max_w") = 1920,
             py::arg("max_channels") = 4,
             "Allocate GPU staging buffer for input upload.\n\n"
             "Args:\n"
             "    imgsz: output size (square, e.g. 640)\n"
             "    max_h: max expected input height (default 1200)\n"
             "    max_w: max expected input width (default 1920)\n"
             "    max_channels: 3 (BGR) or 4 (BGRA), default 4 to cover both"
        )
        .def("process",
             [](hwalker::CudaPreprocessor& self,
                py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> image,
                std::uint64_t output_gpu_ptr,
                bool is_bgra,
                std::uint64_t stream_handle) -> bool
             {
                 if (image.ndim() != 3) {
                     throw std::invalid_argument("image must be HxWxC (ndim=3)");
                 }
                 const int h = static_cast<int>(image.shape(0));
                 const int w = static_cast<int>(image.shape(1));
                 const int c = static_cast<int>(image.shape(2));
                 const int expected_c = is_bgra ? 4 : 3;
                 if (c != expected_c) {
                     throw std::invalid_argument(
                         std::string("image channels=") + std::to_string(c) +
                         " but is_bgra=" + (is_bgra ? "True (expects 4)" : "False (expects 3)"));
                 }
                 const std::uint8_t* input_ptr = static_cast<const std::uint8_t*>(image.data());
                 return self.process(
                     input_ptr, h, w,
                     reinterpret_cast<float*>(output_gpu_ptr),
                     is_bgra,
                     stream_handle
                 );
             },
             py::arg("image"),
             py::arg("output_gpu_ptr"),
             py::arg("is_bgra") = false,
             py::arg("stream_handle") = 0,
             py::call_guard<py::gil_scoped_release>(),
             "Process one frame: H2D + kernel launch on stream.\n\n"
             "Args:\n"
             "    image: numpy uint8 array HxWxC (BGR if is_bgra=False, BGRA if True)\n"
             "    output_gpu_ptr: pre-allocated GPU float32 buffer, NCHW (1,3,imgsz,imgsz)\n"
             "        e.g. torch_tensor.data_ptr()\n"
             "    is_bgra: True for 4-channel BGRA input\n"
             "    stream_handle: cudaStream_t cast to int\n"
             "        e.g. torch.cuda.Stream().cuda_stream\n\n"
             "Returns:\n"
             "    True on success. Caller must synchronize stream before reading output."
        )
        .def_property_readonly("imgsz", &hwalker::CudaPreprocessor::imgsz)
        .def_property_readonly("staging_bytes", &hwalker::CudaPreprocessor::staging_bytes)
        ;
}

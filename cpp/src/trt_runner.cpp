// TRT C++ inference runner implementation (TensorRT 10.3+).
//
// Migrated from Python TRT call to remove ~2ms Python API wrap overhead.
// Memory ownership: caller provides input/output GPU pointers.
//
// TRT 10.x API differences from 8.x:
//   - getNbBindings() → getNbIOTensors()
//   - getBindingName(i) → getIOTensorName(i)
//   - bindingIsInput() → getTensorIOMode() == kINPUT
//   - getBindingDimensions() → getTensorShape()
//   - enqueueV2() → enqueueV3() (uses tensor addresses set via setTensorAddress)
#include "hwalker/trt_runner.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace hwalker {

namespace {

// Simple TRT logger — print warnings and errors to stderr.
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            const char* level =
                (severity == Severity::kERROR) ? "ERROR" :
                (severity == Severity::kWARNING) ? "WARN" :
                (severity == Severity::kINTERNAL_ERROR) ? "INTERNAL_ERROR" :
                "INFO";
            std::cerr << "[TRT-" << level << "] " << msg << std::endl;
        }
    }
};

std::vector<char> read_engine_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error("Cannot open engine file: " + path);
    }
    std::streamsize size = f.tellg();
    f.seekg(0);
    std::vector<char> buf(static_cast<std::size_t>(size));
    if (!f.read(buf.data(), size)) {
        throw std::runtime_error("Failed to read engine file: " + path);
    }
    return buf;
}

std::size_t dtype_bytes(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kBOOL:  return 1;
        case nvinfer1::DataType::kUINT8: return 1;
        default:                         return 4;
    }
}

std::size_t shape_bytes(const nvinfer1::Dims& dims, nvinfer1::DataType dt) {
    std::size_t n = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] > 0) n *= static_cast<std::size_t>(dims.d[i]);
    }
    return n * dtype_bytes(dt);
}

}  // namespace

// ──────────────────────────────────────────────────────────────────────────
TrtRunner::TrtRunner(const std::string& engine_path)
    : engine_path_(engine_path),
      logger_(std::make_unique<TrtLogger>()) {

    // Read engine bytes
    auto engine_bytes = read_engine_file(engine_path);

    // Create runtime
    runtime_.reset(nvinfer1::createInferRuntime(*logger_));
    if (!runtime_) {
        throw std::runtime_error("Failed to create TRT runtime");
    }

    // Deserialize engine
    engine_.reset(runtime_->deserializeCudaEngine(
        engine_bytes.data(), engine_bytes.size()));
    if (!engine_) {
        throw std::runtime_error("Failed to deserialize engine: " + engine_path);
    }

    // Create execution context
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        throw std::runtime_error("Failed to create execution context");
    }

    // Enumerate I/O tensors (TRT 10 API)
    const int n_io = engine_->getNbIOTensors();
    if (n_io < 2) {
        std::ostringstream os;
        os << "Expected at least 2 I/O tensors, got " << n_io;
        throw std::runtime_error(os.str());
    }

    bool found_input = false, found_output = false;
    for (int i = 0; i < n_io; ++i) {
        const char* name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        auto shape = engine_->getTensorShape(name);
        auto dtype = engine_->getTensorDataType(name);

        std::vector<int64_t> shape_vec;
        for (int d = 0; d < shape.nbDims; ++d) {
            shape_vec.push_back(static_cast<int64_t>(shape.d[d]));
        }
        const std::size_t n_bytes = shape_bytes(shape, dtype);

        if (mode == nvinfer1::TensorIOMode::kINPUT && !found_input) {
            input_name_ = name;
            input_shape_ = std::move(shape_vec);
            input_bytes_ = n_bytes;
            found_input = true;
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT && !found_output) {
            output_name_ = name;
            output_shape_ = std::move(shape_vec);
            output_bytes_ = n_bytes;
            found_output = true;
        }
    }

    if (!found_input || !found_output) {
        throw std::runtime_error("Engine must have at least 1 input + 1 output");
    }

    std::cerr << "[TrtRunner] loaded: " << engine_path
              << "  input='" << input_name_ << "' (" << input_bytes_ << "B)"
              << "  output='" << output_name_ << "' (" << output_bytes_ << "B)"
              << std::endl;
}

TrtRunner::~TrtRunner() = default;

// ──────────────────────────────────────────────────────────────────────────
bool TrtRunner::infer(std::uint64_t input_gpu_ptr,
                       std::uint64_t output_gpu_ptr,
                       std::uint64_t stream_handle) {
    void* in_addr = reinterpret_cast<void*>(input_gpu_ptr);
    void* out_addr = reinterpret_cast<void*>(output_gpu_ptr);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);

    // Set per-tensor addresses (TRT 10 way)
    if (!context_->setTensorAddress(input_name_.c_str(), in_addr)) {
        std::cerr << "[TrtRunner] setTensorAddress(input) failed" << std::endl;
        return false;
    }
    if (!context_->setTensorAddress(output_name_.c_str(), out_addr)) {
        std::cerr << "[TrtRunner] setTensorAddress(output) failed" << std::endl;
        return false;
    }

    // Enqueue on stream
    if (!context_->enqueueV3(stream)) {
        std::cerr << "[TrtRunner] enqueueV3 failed" << std::endl;
        return false;
    }
    return true;
}

bool TrtRunner::infer_sync(std::uint64_t input_gpu_ptr,
                            std::uint64_t output_gpu_ptr) {
    if (!infer(input_gpu_ptr, output_gpu_ptr, 0)) {
        return false;
    }
    cudaError_t err = cudaStreamSynchronize(0);
    if (err != cudaSuccess) {
        std::cerr << "[TrtRunner] cudaStreamSynchronize failed: "
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return true;
}

}  // namespace hwalker

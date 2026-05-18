// TRT C++ inference runner — Sprint 1 Phase 2 Week 1.
//
// Drop-in replacement for Python TRT call (pipeline_main.py:T1→T2 stage).
// Today async measurement:
//   T1→T2 predict: 15.18 ms p50
//     ├─ Engine floor:  8.48 ms (hardware)
//     └─ Python wrap:  ~6.70 ms ← C++ migration target
//
// Expected gain: 1.5-2 ms (Python TRT API wrap 제거)
//
// API design:
//   - Engine path (.engine file) → load + execute context
//   - infer(input_ptr, output_ptr, stream) — pure GPU pointers
//   - Caller manages input/output allocation + CUDA stream
//   - No torch dependency (uint64 CUDA pointer interface)
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Forward declare to keep TRT headers out of public interface
namespace nvinfer1 {
class ILogger;
class IRuntime;
class ICudaEngine;
class IExecutionContext;
}

namespace hwalker {

class TrtRunner {
public:
    // Load engine from .engine file. Throws std::runtime_error on failure.
    explicit TrtRunner(const std::string& engine_path);
    ~TrtRunner();
    TrtRunner(const TrtRunner&) = delete;
    TrtRunner& operator=(const TrtRunner&) = delete;

    // Set per-tensor address (caller's GPU memory) and execute.
    // input_gpu_ptr, output_gpu_ptr: CUDA device pointers (cudaMalloc'd or torch GPU tensor data_ptr)
    // stream_handle: cudaStream_t cast to uint64_t (0 = default stream)
    // Returns: true on success, false on TRT internal failure (queue full, etc.)
    bool infer(std::uint64_t input_gpu_ptr,
               std::uint64_t output_gpu_ptr,
               std::uint64_t stream_handle);

    // Synchronous inference: runs on stream, then cudaStreamSynchronize.
    // Convenience wrapper for benchmark / smoke test.
    bool infer_sync(std::uint64_t input_gpu_ptr,
                    std::uint64_t output_gpu_ptr);

    // Tensor info (introspection)
    std::string input_name() const  { return input_name_; }
    std::string output_name() const { return output_name_; }
    std::vector<int64_t> input_shape() const  { return input_shape_; }
    std::vector<int64_t> output_shape() const { return output_shape_; }
    std::size_t input_bytes() const  { return input_bytes_; }
    std::size_t output_bytes() const { return output_bytes_; }

    // Engine + context handles for advanced caller use
    const std::string& engine_path() const { return engine_path_; }

private:
    std::string engine_path_;
    std::unique_ptr<nvinfer1::ILogger> logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    std::string input_name_;
    std::string output_name_;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
    std::size_t input_bytes_ = 0;
    std::size_t output_bytes_ = 0;
};

}  // namespace hwalker

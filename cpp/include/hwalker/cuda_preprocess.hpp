// CUDA letterbox preprocessing kernel for YOLO inference (Sprint 1 Phase 2 Week 3).
//
// Single fused kernel: H2D copy + BGR/BGRA → RGB + bilinear resize + letterbox + normalize.
// Replaces torch GPU preprocess sequence in trt_pose_engine.py:_preprocess_gpu(),
// expected to drop preprocess p50 from 1.62 ms → ~0.3 ms.
//
// Output: float32 NCHW (1, 3, imgsz, imgsz), normalized [0, 1], RGB order, pad value 114/255.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace hwalker {

class CudaPreprocessor {
public:
    // Pre-allocates GPU staging buffer for input upload.
    //   imgsz:    output size (square, e.g. 640)
    //   max_h, max_w: maximum expected input dimensions for staging buffer
    //   max_channels: 3 (BGR) or 4 (BGRA)
    CudaPreprocessor(int imgsz, int max_h, int max_w, int max_channels);
    ~CudaPreprocessor();

    CudaPreprocessor(const CudaPreprocessor&) = delete;
    CudaPreprocessor& operator=(const CudaPreprocessor&) = delete;

    // Process one frame:
    //   1. cudaMemcpyAsync(host input → GPU staging) on stream
    //   2. Launch letterbox kernel on stream (writes to caller's output_gpu_ptr)
    //
    // input_host_ptr:  CPU uint8 buffer, layout = HxWxC interleaved (BGR or BGRA)
    // h, w:            input height/width
    // output_gpu_ptr:  pre-allocated GPU float32 buffer, layout = 1x3ximgsz×imgsz NCHW
    // is_bgra:         true if input has 4 channels (BGRA), false for 3 (BGR)
    // stream_handle:   cudaStream_t cast to uint64
    //
    // Returns true on success. On false, the kernel may have partial state.
    // Caller is responsible for stream synchronization before reading output.
    bool process(const std::uint8_t* input_host_ptr,
                 int h, int w,
                 float* output_gpu_ptr,
                 bool is_bgra,
                 std::uint64_t stream_handle);

    int imgsz() const { return imgsz_; }
    std::size_t staging_bytes() const { return staging_bytes_; }

private:
    int imgsz_;
    int max_channels_;
    std::uint8_t* d_staging_;       // GPU staging buffer for H2D
    std::size_t staging_bytes_;
};

}  // namespace hwalker

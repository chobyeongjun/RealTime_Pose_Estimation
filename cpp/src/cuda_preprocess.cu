// CUDA letterbox preprocessing kernel (Sprint 1 Phase 2 Week 3).
//
// Replaces torch GPU preprocess chain:
//   numpy → cuda → channel_swap → permute → float → div → interpolate
//   → input_tensor.copy_(pad) → slice assignment
// with a single fused kernel that produces float32 NCHW RGB letterboxed output
// in one pass.
//
// Letterbox geometry:
//   scale  = min(imgsz/h, imgsz/w)
//   new_h, new_w = round(h * scale), round(w * scale)
//   pad_h, pad_w = (imgsz - new_h) / 2, (imgsz - new_w) / 2
//   Pixels outside letterbox region → 114/255 (matches torch _pad_tensor).
//
// Bilinear interpolation uses the "half-pixel center" convention to match
// torch.nn.functional.interpolate(..., mode='bilinear', align_corners=False):
//   src = (out + 0.5) / scale - 0.5
#include "hwalker/cuda_preprocess.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace hwalker {

namespace {

constexpr float PAD_VALUE = 114.0f / 255.0f;

// One thread per output (channel, y, x) position.
// Block: 16x16x1 = 256 threads; Grid: (S/16, S/16, 3).
__global__ void preprocess_letterbox_kernel(
    const std::uint8_t* __restrict__ input,
    int in_h, int in_w, int in_channels,
    float* __restrict__ output,
    int out_size,
    int pad_h, int pad_w,
    int new_h, int new_w,
    float inv_scale)   // 1.0 / scale
{
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_c = blockIdx.z;     // 0=R, 1=G, 2=B (output channel)

    if (out_x >= out_size || out_y >= out_size) return;

    // NCHW index: batch=0, ch=out_c, h=out_y, w=out_x
    const int out_idx = ((out_c) * out_size + out_y) * out_size + out_x;

    // Inside letterbox region?
    const int region_y = out_y - pad_h;
    const int region_x = out_x - pad_w;
    if (region_y < 0 || region_y >= new_h || region_x < 0 || region_x >= new_w) {
        output[out_idx] = PAD_VALUE;
        return;
    }

    // Half-pixel center bilinear coord (matches torch align_corners=False)
    const float src_y = (region_y + 0.5f) * inv_scale - 0.5f;
    const float src_x = (region_x + 0.5f) * inv_scale - 0.5f;

    // Clamped neighbor indices
    int y0 = (int)floorf(src_y);
    int x0 = (int)floorf(src_x);
    if (y0 < 0) y0 = 0;
    if (x0 < 0) x0 = 0;
    if (y0 >= in_h) y0 = in_h - 1;
    if (x0 >= in_w) x0 = in_w - 1;
    int y1 = y0 + 1; if (y1 >= in_h) y1 = in_h - 1;
    int x1 = x0 + 1; if (x1 >= in_w) x1 = in_w - 1;

    const float dy = src_y - floorf(src_y);
    const float dx = src_x - floorf(src_x);

    // Input layout: row-major H x W x C (BGR or BGRA).
    // Channel swap: output_c R(0) = input_c 2, G(1) = 1, B(2) = 0  ⇒  in_c = 2 - out_c
    const int in_c = 2 - out_c;
    const int stride_x = in_channels;
    const int stride_y = in_w * in_channels;

    const int idx00 = y0 * stride_y + x0 * stride_x + in_c;
    const int idx01 = y0 * stride_y + x1 * stride_x + in_c;
    const int idx10 = y1 * stride_y + x0 * stride_x + in_c;
    const int idx11 = y1 * stride_y + x1 * stride_x + in_c;

    const float p00 = (float)input[idx00];
    const float p01 = (float)input[idx01];
    const float p10 = (float)input[idx10];
    const float p11 = (float)input[idx11];

    const float top = p00 + dx * (p01 - p00);
    const float bot = p10 + dx * (p11 - p10);
    const float val = top + dy * (bot - top);

    output[out_idx] = val * (1.0f / 255.0f);
}

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CudaPreprocessor] %s: %s\n", what, cudaGetErrorString(err));
    }
}

}  // namespace

// ────────────────────────────────────────────────────────────────────────────
CudaPreprocessor::CudaPreprocessor(int imgsz, int max_h, int max_w, int max_channels)
    : imgsz_(imgsz),
      max_channels_(max_channels),
      d_staging_(nullptr),
      staging_bytes_(0)
{
    if (imgsz <= 0 || max_h <= 0 || max_w <= 0) {
        throw std::invalid_argument("CudaPreprocessor: imgsz/max_h/max_w must be > 0");
    }
    if (max_channels != 3 && max_channels != 4) {
        throw std::invalid_argument("CudaPreprocessor: max_channels must be 3 or 4");
    }

    staging_bytes_ = static_cast<std::size_t>(max_h) * max_w * max_channels;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_staging_), staging_bytes_);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CudaPreprocessor] cudaMalloc(%zu) failed: %s\n",
                     staging_bytes_, cudaGetErrorString(err));
        d_staging_ = nullptr;
        staging_bytes_ = 0;
        throw std::runtime_error("CudaPreprocessor: staging alloc failed");
    }

    std::fprintf(stderr,
        "[CudaPreprocessor] init imgsz=%d staging=%zu B (max %dx%dx%d)\n",
        imgsz_, staging_bytes_, max_h, max_w, max_channels);
}

CudaPreprocessor::~CudaPreprocessor() {
    if (d_staging_) {
        cudaFree(d_staging_);
        d_staging_ = nullptr;
    }
}

bool CudaPreprocessor::process(
    const std::uint8_t* input_host_ptr,
    int h, int w,
    float* output_gpu_ptr,
    bool is_bgra,
    std::uint64_t stream_handle)
{
    if (input_host_ptr == nullptr || output_gpu_ptr == nullptr) {
        std::fprintf(stderr, "[CudaPreprocessor] null pointer\n");
        return false;
    }
    if (h <= 0 || w <= 0) {
        std::fprintf(stderr, "[CudaPreprocessor] invalid h/w: %d/%d\n", h, w);
        return false;
    }
    const int channels = is_bgra ? 4 : 3;
    const std::size_t input_bytes = static_cast<std::size_t>(h) * w * channels;
    if (input_bytes > staging_bytes_) {
        std::fprintf(stderr,
            "[CudaPreprocessor] input %zu B exceeds staging %zu B\n",
            input_bytes, staging_bytes_);
        return false;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);

    // H2D copy on stream (pageable memory → blocks host briefly but enqueues stream op)
    cudaError_t err = cudaMemcpyAsync(
        d_staging_, input_host_ptr, input_bytes,
        cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cuda_check(err, "cudaMemcpyAsync");
        return false;
    }

    // Letterbox parameters (must match torch reference exactly)
    const float scale = std::min(static_cast<float>(imgsz_) / h,
                                  static_cast<float>(imgsz_) / w);
    const int new_h = static_cast<int>(h * scale);
    const int new_w = static_cast<int>(w * scale);
    const int pad_h = (imgsz_ - new_h) / 2;
    const int pad_w = (imgsz_ - new_w) / 2;
    const float inv_scale = 1.0f / scale;

    // Grid: 16x16 threads/block × (S/16, S/16, 3) blocks
    const dim3 block(16, 16, 1);
    const dim3 grid(
        (imgsz_ + block.x - 1) / block.x,
        (imgsz_ + block.y - 1) / block.y,
        3
    );

    preprocess_letterbox_kernel<<<grid, block, 0, stream>>>(
        d_staging_, h, w, channels,
        output_gpu_ptr, imgsz_,
        pad_h, pad_w, new_h, new_w,
        inv_scale
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cuda_check(err, "kernel launch");
        return false;
    }
    return true;
}

}  // namespace hwalker

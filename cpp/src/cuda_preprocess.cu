// CUDA letterbox preprocessing kernel (Sprint 1 Phase 2 Week 3).
//
// Replaces torch GPU preprocess chain:
//   numpy → cuda → channel_swap → permute → float → div → interpolate
//   → input_tensor.copy_(pad) → slice assignment
// with a single fused kernel that produces float32 NCHW RGB letterboxed output
// in one pass.
//
// Letterbox geometry (MUST match torch reference bit-equally):
//   scale  = min(imgsz/h, imgsz/w)            ← computed in DOUBLE on host
//   new_h, new_w = int(h * scale), int(w * scale)
//   pad_h, pad_w = (imgsz - new_h) / 2, (imgsz - new_w) / 2
//   Pixels outside letterbox region → 114/255 (matches torch _pad_tensor).
//
// Bilinear interpolation matches torch.nn.functional.interpolate(
//   size=(new_h, new_w), mode='bilinear', align_corners=False) which uses:
//   - Per-axis effective ratio (NOT the letterbox scale): scale_y = in_h / new_h,
//     scale_x = in_w / new_w (after truncation).
//   - Half-pixel center: src = (out + 0.5) * scale - 0.5
//   - Source coord CLAMPED to [0, in-1] BEFORE computing weights (dy, dx).
//     Without clamping, upsizes (in < out) miscompute the top/left border.
//
// Threading: one thread per output (channel, y, x). Block (16, 16, 1) = 256.
//
// Concurrency contract: CudaPreprocessor owns a single d_staging_ buffer.
// Caller MUST serialize process() calls on a single stream OR synchronize
// the previous call's stream before the next process() call. Multi-stream
// concurrent use is NOT supported (would race on d_staging_).
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
    float scale_y,    // per-axis: (float)in_h / (float)new_h
    float scale_x)    // per-axis: (float)in_w / (float)new_w
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

    // Per-axis half-pixel center bilinear coord (matches torch align_corners=False).
    // CRITICAL: clamp src_y/src_x to [0, in-1] BEFORE computing weights, so
    // dy/dx are zero at the border (matches torch behavior for upsize).
    float src_y = (region_y + 0.5f) * scale_y - 0.5f;
    float src_x = (region_x + 0.5f) * scale_x - 0.5f;
    if (src_y < 0.0f) src_y = 0.0f;
    if (src_x < 0.0f) src_x = 0.0f;
    const float in_h_f = (float)(in_h - 1);
    const float in_w_f = (float)(in_w - 1);
    if (src_y > in_h_f) src_y = in_h_f;
    if (src_x > in_w_f) src_x = in_w_f;

    int y0 = (int)floorf(src_y);
    int x0 = (int)floorf(src_x);
    int y1 = y0 + 1; if (y1 > in_h - 1) y1 = in_h - 1;
    int x1 = x0 + 1; if (x1 > in_w - 1) x1 = in_w - 1;

    const float dy = src_y - (float)y0;
    const float dx = src_x - (float)x0;

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

    // Lifetime contract: input_host_ptr must remain valid until this function
    // returns. For pageable memory + cudaMemcpyAsync, CUDA stages internally
    // before returning to host, so post-return modification is safe. For PINNED
    // memory, caller must keep buffer alive until stream sync.
    cudaError_t err = cudaMemcpyAsync(
        d_staging_, input_host_ptr, input_bytes,
        cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cuda_check(err, "cudaMemcpyAsync");
        return false;
    }

    // P1.3 fix: Letterbox geometry in DOUBLE (matches Python double).
    // Float32 diverges on edge cases like 300x301 → (637,639) vs Python (637,640).
    const double scale_d = std::min(
        static_cast<double>(imgsz_) / static_cast<double>(h),
        static_cast<double>(imgsz_) / static_cast<double>(w));
    const int new_h = static_cast<int>(static_cast<double>(h) * scale_d);
    const int new_w = static_cast<int>(static_cast<double>(w) * scale_d);
    const int pad_h = (imgsz_ - new_h) / 2;
    const int pad_w = (imgsz_ - new_w) / 2;

    // P1.1 fix: Per-axis effective scale (NOT 1/letterbox_scale).
    // Matches torch F.interpolate(size=(new_h, new_w)) behavior.
    const float scale_y = static_cast<float>(static_cast<double>(h) / static_cast<double>(new_h));
    const float scale_x = static_cast<float>(static_cast<double>(w) / static_cast<double>(new_w));

    // Defensive: new_h/new_w should fit in imgsz. If not, parameters are wrong.
    if (new_h <= 0 || new_w <= 0 || pad_h < 0 || pad_w < 0 ||
        new_h + pad_h > imgsz_ || new_w + pad_w > imgsz_) {
        std::fprintf(stderr,
            "[CudaPreprocessor] letterbox geometry invalid: "
            "h=%d w=%d imgsz=%d new=(%d,%d) pad=(%d,%d)\n",
            h, w, imgsz_, new_h, new_w, pad_h, pad_w);
        return false;
    }

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
        scale_y, scale_x
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cuda_check(err, "kernel launch");
        return false;
    }
    return true;
}

}  // namespace hwalker

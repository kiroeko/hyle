// SGLang Apache-2.0 device fragment at 06992c92ad4494fe9f9636e75be3e35040512447.
// See LICENSE, NOTICE and provenance.json for extraction/adaptations.
#pragma once

#include "norm_device.cuh"

namespace hyle::compute::cuda_native::sglang_vendor {

struct RMSNormParams {
  const void* input;
  const void* __restrict__ weight;
  void* output;
  int64_t input_stride;
  int64_t output_stride;
  uint32_t num_tokens;
  float eps;
};

template <int64_t kDim, bool kUsePDL, typename Float>
__global__ void rmsnorm_cta(const RMSNormParams __grid_constant__ params) {
  using namespace device;
  using Storage = norm::StorageType<Float, kDim>;

  constexpr auto kNumThreads = host::norm::get_cta_threads<Float, kDim>();
  constexpr auto kNumWarps = kNumThreads / kWarpThreads;

  const auto& [input, weight_ptr, output, input_stride, output_stride, num_tokens, eps] = params;
  const auto gmem = tile::Memory<Storage>::cta(kNumThreads);
  __shared__ float smem[norm::kSmemBufferSize];

  PDLWaitPrimary<kUsePDL>();  // wait for primary kernel

  for (uint32_t i = blockIdx.x; i < num_tokens; i += gridDim.x) {
    const auto input_ptr = pointer::offset<Float>(input, i * input_stride);
    const auto output_ptr = pointer::offset<Float>(output, i * output_stride);
    const auto input_vec = gmem.load(input_ptr);
    const auto weight_vec = gmem.load(weight_ptr);
    const auto output_vec = norm::apply_norm_cta<kDim>(input_vec, weight_vec, eps, smem, kNumWarps);
    gmem.store(output_ptr, output_vec);
  }

  PDLTriggerSecondary<kUsePDL>();  // launch secondary kernel
}

// Pre-Blackwell: 16B vector, each thread loads/stores twice
template <int64_t kDim, bool kUsePDL, typename Float>
__global__ __launch_bounds__(kDim / 16) void rmsnorm_cta_double(const RMSNormParams __grid_constant__ params) {
  using namespace device;
  using Float2 = packed_t<Float>;
  using Storage = AlignedVector<Float2, 4>;

  constexpr auto kNumThreads = kDim / 16;
  constexpr auto kNumWarps = kNumThreads / kWarpThreads;

  const auto& [input, weight_ptr, output, input_stride, output_stride, num_tokens, eps] = params;
  const auto gmem = tile::Memory<Storage>::cta(kNumThreads);
  __shared__ float smem[32];

  PDLWaitPrimary<kUsePDL>();

  const auto input_ptr = pointer::offset<Float>(input, blockIdx.x * input_stride);
  const auto output_ptr = pointer::offset<Float>(output, blockIdx.x * output_stride);

  const auto input_first = gmem.load(input_ptr, 0);
  const auto input_second = gmem.load(input_ptr, 1);
  const auto weight_first = gmem.load(weight_ptr, 0);
  const auto weight_second = gmem.load(weight_ptr, 1);

  float sum_of_squares = 0.0f;
#pragma unroll
  for (auto j = 0u; j < 4u; ++j) {
    const auto [x, y] = cast<fp32x2_t>(input_first[j]);
    sum_of_squares += x * x + y * y;
  }
#pragma unroll
  for (auto j = 0u; j < 4u; ++j) {
    const auto [x, y] = cast<fp32x2_t>(input_second[j]);
    sum_of_squares += x * x + y * y;
  }

  sum_of_squares = warp::reduce_sum(sum_of_squares);
  float norm_factor;
  if constexpr (kNumWarps == 1) {
    norm_factor = math::rsqrt(sum_of_squares / kDim + eps);
  } else {
    const auto warp_id = threadIdx.x / kWarpThreads;
    smem[warp_id] = sum_of_squares;
    __syncthreads();
    if (warp_id == 0) {
      const auto tx = threadIdx.x;
      const auto local_sum = tx < kNumWarps ? smem[tx] : 0.0f;
      sum_of_squares = warp::reduce_sum(local_sum);
      smem[tx] = math::rsqrt(sum_of_squares / kDim + eps);
    }
    __syncthreads();
    norm_factor = smem[warp_id];
  }

  Storage output_first, output_second;
#pragma unroll
  for (auto j = 0u; j < 4u; ++j) {
    const auto [ix, iy] = cast<fp32x2_t>(input_first[j]);
    const auto [wx, wy] = cast<fp32x2_t>(weight_first[j]);
    output_first[j] = cast<Float2>(fp32x2_t{ix * norm_factor * wx, iy * norm_factor * wy});
  }
#pragma unroll
  for (auto j = 0u; j < 4u; ++j) {
    const auto [ix, iy] = cast<fp32x2_t>(input_second[j]);
    const auto [wx, wy] = cast<fp32x2_t>(weight_second[j]);
    output_second[j] = cast<Float2>(fp32x2_t{ix * norm_factor * wx, iy * norm_factor * wy});
  }

  gmem.store(output_ptr, output_first, 0);
  gmem.store(output_ptr, output_second, 1);

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kDim, bool kUsePDL, typename Float>
__global__ void rmsnorm_warp(const RMSNormParams __grid_constant__ params) {
  using namespace device;
  using Storage = norm::StorageType<Float, kDim>;

  const auto& [input, weight_ptr, output, input_stride, output_stride, num_tokens, eps] = params;
  const auto gmem = tile::Memory<Storage>::warp();

  PDLWaitPrimary<kUsePDL>();  // wait for primary kernel

  for (uint32_t i = blockIdx.x; i < num_tokens; i += gridDim.x) {
    const auto input_ptr = pointer::offset<Float>(input, i * input_stride);
    const auto output_ptr = pointer::offset<Float>(output, i * output_stride);
    const auto input_vec = gmem.load(input_ptr);
    const auto weight_vec = gmem.load(weight_ptr);
    const auto output_vec = norm::apply_norm_warp<kDim>(input_vec, weight_vec, eps);
    gmem.store(output_ptr, output_vec);
  }

  PDLTriggerSecondary<kUsePDL>();  // launch secondary kernel
}


} // namespace hyle::compute::cuda_native::sglang_vendor

#undef HYLE_SGL_DEVICE

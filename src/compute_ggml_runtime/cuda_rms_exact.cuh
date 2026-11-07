#pragma once

// One warp handles a weighted F32 RMSNorm row of at most 128 elements. The
// four partial warp reductions and their final reduction reproduce the exact
// order of the existing 256-thread kernel, including its inactive warps.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
static __global__ void hyle_cuda_rms_norm_small_exact(
    const float *x, float *dst, const int ncols,
    const int64_t stride_row, const int64_t stride_channel,
    const int64_t stride_sample, const float eps, const float *mul,
    const int64_t mul_stride_row, const int64_t mul_stride_channel,
    const int64_t mul_stride_sample, const uint3 mul_ncols_packed,
    const uint3 mul_nrows_packed, const uint3 mul_nchannels_packed,
    const uint3 mul_nsamples_packed) {
  ggml_cuda_pdl_lc();
  const int nrows = gridDim.x, nchannels = gridDim.y;
  const int row = blockIdx.x, channel = blockIdx.y, sample = blockIdx.z;
  const int lane = threadIdx.x;
  x += sample * stride_sample + channel * stride_channel + row * stride_row;
  dst += ((sample * nchannels + channel) * nrows + row) * ncols;
  mul += fastmodulo(sample, mul_nsamples_packed) * mul_stride_sample +
         fastmodulo(channel, mul_nchannels_packed) * mul_stride_channel +
         fastmodulo(row, mul_nrows_packed) * mul_stride_row;
  ggml_cuda_pdl_sync();

  float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
  if (lane < ncols) { const float v = x[lane]; p0 += v * v; }
  if (lane + 32 < ncols) { const float v = x[lane + 32]; p1 += v * v; }
  if (lane + 64 < ncols) { const float v = x[lane + 64]; p2 += v * v; }
  if (lane + 96 < ncols) { const float v = x[lane + 96]; p3 += v * v; }
  // The original block reduction publishes lane zero of each partial warp.
  p0 = __shfl_sync(0xffffffff, warp_reduce_sum(p0), 0);
  p1 = __shfl_sync(0xffffffff, warp_reduce_sum(p1), 0);
  p2 = __shfl_sync(0xffffffff, warp_reduce_sum(p2), 0);
  p3 = __shfl_sync(0xffffffff, warp_reduce_sum(p3), 0);
  float sum = lane == 0 ? p0 : lane == 1 ? p1 : lane == 2 ? p2 : lane == 3 ? p3 : 0.f;
  sum = warp_reduce_sum(sum);
  const float scale = rsqrtf(sum / ncols + eps);
  for (int col = lane; col < ncols; col += 32)
    dst[col] = scale * x[col] * mul[fastmodulo(col, mul_ncols_packed)];
}
#endif

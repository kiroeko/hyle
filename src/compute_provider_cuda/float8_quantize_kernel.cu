#include "float8_quantize_kernel.h"

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "device_status_kernel.h"

namespace hyle::compute::cuda_native::float8_quantize::native {
namespace {
constexpr unsigned kThreads = 128, kGroupsPerBlock = kThreads / 32;

template <class T> __device__ float decode(T value) { return value; }
template <> __device__ float decode(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

// Each warp owns one contiguous group128 and its independent F32 scale.
// Explicit rounded instructions retain the declared F32 boundaries instead
// of allowing reciprocal substitution or multiply/add contraction.
template <class T>
__global__ void quantize(Program p, const T *input, unsigned char *codes,
                         float *scales, unsigned *status) {
  const unsigned lane = threadIdx.x & 31;
  for (auto group = std::uint64_t(blockIdx.x) * kGroupsPerBlock +
                    threadIdx.x / 32;
       group < p.groups;
       group += std::uint64_t(gridDim.x) * kGroupsPerBlock) {
    float values[4];
    float maximum = 0.F;
    bool finite = true;
#pragma unroll
    for (unsigned part = 0; part < 4; ++part) {
      const float value = decode(input[group * 128 + part * 32 + lane]);
      values[part] = value;
      finite = finite && isfinite(value);
      maximum = fmaxf(maximum, fabsf(value));
    }
    if (__any_sync(0xffffffffU, !finite)) {
      if (lane == 0)
        atomicCAS(status, 0U, static_cast<unsigned>(
            cuda::detail::DeviceStatusCode::kInvalidNumericInput));
      // Failed invocations expose no usable outputs; never convert a NaN or
      // infinity into an apparently successful saturated finite group.
      continue;
    }
    for (unsigned offset = 16; offset; offset /= 2)
      maximum = fmaxf(maximum,
                     __shfl_xor_sync(0xffffffffU, maximum, offset));
    maximum = fmaxf(maximum, p.floor);
    float scale = p.power2 || p.multiply_reciprocal
                      ? __fmul_rn(maximum, __uint_as_float(0x3b124925U))
                      : __fdiv_rn(maximum, 448.F);
    if (p.power2) {
      const unsigned bits = __float_as_uint(scale);
      scale = __uint_as_float(((bits >> 23) +
                               ((bits & 0x7fffffU) != 0)) << 23);
    }
    if (lane == 0)
      scales[group] = scale;
    const float reciprocal = p.multiply_reciprocal
                                 ? __fdiv_rn(1.F, scale) : 0.F;
#pragma unroll
    for (unsigned part = 0; part < 4; ++part) {
      const float normalized = p.multiply_reciprocal
          ? __fmul_rn(values[part], reciprocal)
          : __fdiv_rn(values[part], scale);
      // CUDA's E4M3 conversion is SATFINITE/RNE and retains signed zero.
      codes[group * 128 + part * 32 + lane] =
          __nv_cvt_float_to_fp8(normalized, __NV_SATFINITE, __NV_E4M3);
    }
  }
}
} // namespace

cudaError_t prepare(Program program) noexcept {
  cudaFuncAttributes attributes{};
  return program.bfloat
      ? cudaFuncGetAttributes(&attributes, quantize<__nv_bfloat16>)
      : cudaFuncGetAttributes(&attributes, quantize<float>);
}

cudaError_t launch(Program program, const void *input, unsigned char *codes,
                   float *scales, unsigned *status,
                   cudaStream_t stream) noexcept {
  if (!program.groups)
    return cudaSuccess;
  const auto blocks = unsigned((std::min)(std::uint64_t{65535},
      (program.groups + kGroupsPerBlock - 1) / kGroupsPerBlock));
  if (program.bfloat)
    quantize<<<blocks, kThreads, 0, stream>>>(
        program, static_cast<const __nv_bfloat16 *>(input), codes, scales,
        status);
  else
    quantize<<<blocks, kThreads, 0, stream>>>(
        program, static_cast<const float *>(input), codes, scales, status);
  return cudaGetLastError();
}
} // namespace hyle::compute::cuda_native::float8_quantize::native

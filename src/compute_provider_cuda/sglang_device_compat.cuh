#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

#define HYLE_SGL_DEVICE __device__ __forceinline__

namespace hyle::compute::cuda_native::sglang_vendor {
using fp16_t = __half;
using bf16_t = __nv_bfloat16;
using fp32x2_t = float2;
template <class T>
using packed_t =
    std::conditional_t<std::is_same_v<T, fp16_t>, __half2, __nv_bfloat162>;
namespace host {
template <class> inline constexpr bool dependent_false_v = false;
} // namespace host
namespace device {
inline constexpr unsigned kMaxVecBytes = 16;
inline constexpr unsigned kWarpThreads = 32;
using fp32_t = float;

template <class T, std::size_t N>
struct alignas(sizeof(T) * N) AlignedVector final {
  T values[N];
  HYLE_SGL_DEVICE T &operator[](unsigned index) { return values[index]; }
  HYLE_SGL_DEVICE const T &operator[](unsigned index) const {
    return values[index];
  }
};
template <class T, class U> HYLE_SGL_DEVICE T cast(U value) {
  if constexpr (std::is_same_v<T, float2> && std::is_same_v<U, __half2>)
    return __half22float2(value);
  else if constexpr (std::is_same_v<T, float2> &&
                     std::is_same_v<U, __nv_bfloat162>)
    return __bfloat1622float2(value);
  else if constexpr (std::is_same_v<T, __half2> && std::is_same_v<U, float2>)
    return __float22half2_rn(value);
  else if constexpr (std::is_same_v<T, __nv_bfloat162> &&
                     std::is_same_v<U, float2>)
    return __float22bfloat162_rn(value);
  else
    return static_cast<T>(value);
}

namespace pointer {
template <class T, class P>
HYLE_SGL_DEVICE auto offset(P *pointer, std::int64_t elements) {
  using Value = std::conditional_t<std::is_const_v<P>, const T, T>;
  return static_cast<Value *>(pointer) + elements;
}
} // namespace pointer
namespace math {
HYLE_SGL_DEVICE float rsqrt(float value) { return rsqrtf(value); }
} // namespace math
namespace warp {
// Pinned SGLang float SUM specialization, with the original subgroup width.
template <unsigned Width = kWarpThreads>
HYLE_SGL_DEVICE float reduce_sum(float value) {
  static_assert(Width > 0 && Width <= kWarpThreads && (Width & (Width - 1)) == 0);
#pragma unroll
  for (unsigned mask = Width / 2; mask >= 1; mask >>= 1)
    value += __shfl_xor_sync(0xffffffffU, value, mask, 32);
  return value;
}
} // namespace warp
template <class T>
HYLE_SGL_DEVICE T load_as(const void *input, unsigned index) {
  return static_cast<const T *>(input)[index];
}
template <class T>
HYLE_SGL_DEVICE void store_as(void *output, T value, unsigned index) {
  static_cast<T *>(output)[index] = value;
}
template <bool Enabled> HYLE_SGL_DEVICE void PDLWaitPrimary() {
  static_assert(!Enabled, "precise CUDA activation does not use PDL");
}
template <bool Enabled> HYLE_SGL_DEVICE void PDLTriggerSecondary() {
  static_assert(!Enabled, "precise CUDA activation does not use PDL");
}
} // namespace device
} // namespace hyle::compute::cuda_native::sglang_vendor

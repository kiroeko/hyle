#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

#define SGL_DEVICE __device__ __forceinline__

namespace sglang {
namespace host {
template <class> inline constexpr bool dependent_false_v = false;
} // namespace host
namespace device {
inline constexpr unsigned kMaxVecBytes = 16;
using fp32_t = float;

template <class T, unsigned N> struct alignas(16) AlignedVector final {
  T values[N];
  SGL_DEVICE T &operator[](unsigned index) { return values[index]; }
  SGL_DEVICE const T &operator[](unsigned index) const { return values[index]; }
};
template <class T, class U> SGL_DEVICE T cast(U value) {
  return static_cast<T>(value);
}
template <class T> SGL_DEVICE T load_as(const void *input, unsigned index) {
  return static_cast<const T *>(input)[index];
}
template <class T>
SGL_DEVICE void store_as(void *output, T value, unsigned index) {
  static_cast<T *>(output)[index] = value;
}
template <bool Enabled> SGL_DEVICE void PDLWaitPrimary() {
  static_assert(!Enabled, "SM89 reference does not use PDL");
}
template <bool Enabled> SGL_DEVICE void PDLTriggerSecondary() {
  static_assert(!Enabled, "SM89 reference does not use PDL");
}
} // namespace device
} // namespace sglang

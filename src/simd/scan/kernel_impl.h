// Shared scan kernel bodies (Kogge-Stone within vector + carry across vectors).
// Each TU defines HYLE_SIMD_TARGET_* + HYLE_KERNEL_SUFFIX then includes this.

#include <cstddef>
#include <cstdint>

#include "hyle/simd/vec.h"
#include "src/simd/scan/kernels.h"

#ifndef HYLE_KERNEL_SUFFIX
#  error "define HYLE_KERNEL_SUFFIX before including scan/kernel_impl.h"
#endif

#define HYLE_SIMD_CAT_(a, b) a##b
#define HYLE_SIMD_CAT(a, b) HYLE_SIMD_CAT_(a, b)

namespace hyle::simd::detail {
namespace {

// Inclusive prefix within one vector: log2(w) Kogge-Stone steps via SlideUpLanes.
// NOTE: do NOT use ShiftLeftLanes here — on AVX2/AVX-512, ShiftLeftBytes is
// per-128-bit-half (slli_si256), so it does not move data across the mid-vector
// boundary and silently corrupts the scan. SlideUpLanes is full-register correct.
template <class D, class V>
HYLE_FORCE_INLINE V inclusive_scan_vec(D d, V v) noexcept {
  const std::size_t w = Lanes(d);
  for (std::size_t k = 1; k < w; k *= 2) {
    v = Add(v, SlideUpLanes(d, v, k));
  }
  return v;
}

template <typename T>
HYLE_FORCE_INLINE void inclusive_scan_kernel(T* data, std::size_t n) noexcept {
  if (n == 0) return;
  const ScalableTag<T> d;
  const std::size_t w = Lanes(d);
  T carry = T{0};
  std::size_t i = 0;
  for (; i + w <= n; i += w) {
    auto v = inclusive_scan_vec(d, LoadU(d, data + i));
    v = Add(v, Set(d, carry));
    StoreU(v, d, data + i);
    carry = ExtractLane(v, w - 1);
  }
  for (; i < n; ++i) {
    carry = static_cast<T>(carry + data[i]);
    data[i] = carry;
  }
}

template <typename T>
HYLE_FORCE_INLINE void exclusive_scan_kernel(T* data, std::size_t n) noexcept {
  if (n == 0) return;
  const ScalableTag<T> d;
  const std::size_t w = Lanes(d);
  T carry = T{0};
  std::size_t i = 0;
  for (; i + w <= n; i += w) {
    const auto raw = LoadU(d, data + i);
    const auto inc = inclusive_scan_vec(d, raw);
    // exclusive within block: [0, inc0, inc1, ...] then add incoming carry to every lane.
    auto exc = Add(SlideUpLanes(d, inc, 1), Set(d, carry));
    StoreU(exc, d, data + i);
    // total through this block = last exclusive + last original element.
    carry = static_cast<T>(ExtractLane(exc, w - 1) + ExtractLane(raw, w - 1));
  }
  for (; i < n; ++i) {
    const T v = data[i];
    data[i] = carry;
    carry = static_cast<T>(carry + v);
  }
}

}  // namespace

#define HYLE_SCAN_DEF(T, tag)                                                                  \
  void HYLE_SIMD_CAT(inclusive_scan_##tag##_, HYLE_KERNEL_SUFFIX)(T * data,                    \
                                                                  std::size_t n) noexcept {    \
    inclusive_scan_kernel<T>(data, n);                                                         \
  }                                                                                            \
  void HYLE_SIMD_CAT(exclusive_scan_##tag##_, HYLE_KERNEL_SUFFIX)(T * data,                    \
                                                                  std::size_t n) noexcept {    \
    exclusive_scan_kernel<T>(data, n);                                                         \
  }

HYLE_SCAN_DEF(std::uint32_t, u32)
HYLE_SCAN_DEF(std::int32_t, i32)
HYLE_SCAN_DEF(std::uint64_t, u64)
HYLE_SCAN_DEF(std::int64_t, i64)
HYLE_SCAN_DEF(float, f32)
HYLE_SCAN_DEF(double, f64)

#undef HYLE_SCAN_DEF

}  // namespace hyle::simd::detail

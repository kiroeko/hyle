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

// Full-register slide-up by power-of-two lane count via Slide1Up composition.
// Used when SlideUpLanes would store+reload (non-32-bit, or narrow vectors).
// Slide1Up is the portable FULL-register 1-lane up (AVX2: permute2x128+alignr;
// AVX-512 32/64: valignd/q; never per-128 slli_si256).
template <int k, class D, class V>
HYLE_FORCE_INLINE V slide_up_pow2(D d, V v) noexcept {
  static_assert(k >= 1 && (k & (k - 1)) == 0, "k must be power of two");
  if constexpr (k == 1) {
    return Slide1Up(d, v);
  } else {
    return slide_up_pow2<k / 2>(d, slide_up_pow2<k / 2>(d, v));
  }
}

// One KS step: shift current partial sums up by k lanes (full register) and add.
// 32-bit × full native width: SlideUpLanes uses vpermd (AVX2) / valignd (AVX-512).
// Else: Slide1Up tree (correct across 128b boundaries on every backend).
template <int k, class D, class V>
HYLE_FORCE_INLINE V ks_step(D d, V v) noexcept {
  using T = TFromD<D>;
  if constexpr (k >= D::kMaxLanes) {
    return v;
  } else if constexpr (sizeof(T) == 4 && D::kMaxLanes >= 8) {
    return Add(v, SlideUpLanes(d, v, static_cast<std::size_t>(k)));
  } else {
    return Add(v, slide_up_pow2<k>(d, v));
  }
}

// Inclusive prefix within one vector: log2(N) Kogge-Stone steps.
template <class D, class V>
HYLE_FORCE_INLINE V inclusive_scan_vec(D d, V v) noexcept {
  v = ks_step<1>(d, v);
  v = ks_step<2>(d, v);
  v = ks_step<4>(d, v);
  v = ks_step<8>(d, v);
  v = ks_step<16>(d, v);
  v = ks_step<32>(d, v);
  return v;
}

// Last lane without ExtractLane store+reload: BroadcastLane(last) then GetLane (low).
template <class D, class V>
HYLE_FORCE_INLINE TFromD<D> last_lane(D /*d*/, V v) noexcept {
  constexpr int kLast = static_cast<int>(D::kMaxLanes) - 1;
  return GetLane(BroadcastLane<kLast>(v));
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
    carry = last_lane(d, v);
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
    // exclusive: [0, inc0, ...] via full-register Slide1Up, then + carry on all lanes.
    auto exc = Add(Slide1Up(d, inc), Set(d, carry));
    StoreU(exc, d, data + i);
    carry = static_cast<T>(last_lane(d, exc) + last_lane(d, raw));
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

#include <cstddef>
#include <cstdint>

#include "hyle/simd/native/scan.h"
#include "src/simd/scan/kernels.h"

#ifndef HYLE_SIMD_VARIANT_SUFFIX
#error "define HYLE_SIMD_VARIANT_SUFFIX before including scan/kernel_impl.h"
#endif

#define HYLE_SCAN_CAT_(a, b) a##b
#define HYLE_SCAN_CAT(a, b) HYLE_SCAN_CAT_(a, b)

namespace hyle::simd::detail {

#define HYLE_SCAN_DEF(T, tag)                                                        \
  void HYLE_SCAN_CAT(inclusive_scan_##tag##_, HYLE_SIMD_VARIANT_SUFFIX)(            \
      T* data, std::size_t n) noexcept { native::inclusive_scan(data, n); }          \
  void HYLE_SCAN_CAT(exclusive_scan_##tag##_, HYLE_SIMD_VARIANT_SUFFIX)(            \
      T* data, std::size_t n) noexcept { native::exclusive_scan(data, n); }

HYLE_SCAN_DEF(std::uint32_t, u32)
HYLE_SCAN_DEF(std::int32_t, i32)
HYLE_SCAN_DEF(std::uint64_t, u64)
HYLE_SCAN_DEF(std::int64_t, i64)
HYLE_SCAN_DEF(float, f32)
HYLE_SCAN_DEF(double, f64)

#undef HYLE_SCAN_DEF
}  // namespace hyle::simd::detail

#undef HYLE_SCAN_CAT
#undef HYLE_SCAN_CAT_

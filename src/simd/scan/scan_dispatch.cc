#include "hyle/simd/scan.h"

#include <cstddef>
#include <cstdint>

#include "src/simd/dispatch_internal.h"
#include "src/simd/runtime_internal.h"
#include "src/simd/scan/kernels.h"

#if defined(HYLE_SIMD_EXACT)
#define HYLE_SCAN_CALL(name, ...) HYLE_SIMD_EXACT_SYMBOL(name)(__VA_ARGS__)
#else
#define HYLE_SCAN_CALL(name, ...) runtime().scan_ops.name(__VA_ARGS__)
#endif

namespace hyle::simd {

#define HYLE_SCAN_API(T, tag)                                               \
  void inclusive_scan(const T* src, T* dst, std::size_t n) noexcept {      \
    HYLE_SCAN_CALL(inclusive_scan_##tag, src, dst, n);                      \
  }                                                                         \
  void exclusive_scan(const T* src, T* dst, std::size_t n, T init)         \
      noexcept {                                                            \
    HYLE_SCAN_CALL(exclusive_scan_##tag, src, dst, n, init);                \
  }

HYLE_SCAN_API(std::uint32_t, u32)
HYLE_SCAN_API(std::int32_t, i32)
HYLE_SCAN_API(std::uint64_t, u64)
HYLE_SCAN_API(std::int64_t, i64)
HYLE_SCAN_API(float, f32)
HYLE_SCAN_API(double, f64)

#undef HYLE_SCAN_API
}  // namespace hyle::simd

#undef HYLE_SCAN_CALL

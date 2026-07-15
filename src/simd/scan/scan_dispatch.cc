#include "hyle/simd/scan.h"

#include <cstddef>
#include <cstdint>

#include "hyle/platform.h"
#include "src/simd/scan/kernels.h"

// Baseline dispatcher (no -march). Same selected_tier model as ops/dispatch_ops.cc.

namespace hyle::simd {
namespace {

Isa selected_tier() noexcept {
#if defined(HYLE_ARCH_X64)
  const auto v = static_cast<std::uint32_t>(best_isa());
  if (v >= static_cast<std::uint32_t>(Isa::kAvx512)) return Isa::kAvx512;
  if (v >= static_cast<std::uint32_t>(Isa::kAvx2)) return Isa::kAvx2;
  if (v >= static_cast<std::uint32_t>(Isa::kSse42)) return Isa::kSse42;
  return Isa::kScalar;
#elif defined(HYLE_ARCH_ARM64)
  if (static_cast<std::uint32_t>(best_isa()) >= static_cast<std::uint32_t>(Isa::kNeon))
    return Isa::kNeon;
  return Isa::kScalar;
#else
  return Isa::kScalar;
#endif
}

template <typename T>
using ScanFn = void (*)(T*, std::size_t) noexcept;

template <typename T>
ScanFn<T> resolve_inclusive() noexcept;

template <typename T>
ScanFn<T> resolve_exclusive() noexcept;

#define HYLE_RESOLVE(T, tag)                                                                   \
  template <>                                                                                  \
  ScanFn<T> resolve_inclusive<T>() noexcept {                                                  \
    switch (selected_tier()) {                                                                 \
      HYLE_ARCH_CASES(inclusive_scan_##tag)                                                    \
    }                                                                                          \
  }                                                                                            \
  template <>                                                                                  \
  ScanFn<T> resolve_exclusive<T>() noexcept {                                                  \
    switch (selected_tier()) {                                                                 \
      HYLE_ARCH_CASES(exclusive_scan_##tag)                                                    \
    }                                                                                          \
  }

#if defined(HYLE_ARCH_X64)
#  define HYLE_ARCH_CASES(fnbase)                                                              \
    case Isa::kAvx512:                                                                         \
      return &detail::fnbase##_avx512;                                                         \
    case Isa::kAvx2:                                                                           \
      return &detail::fnbase##_avx2;                                                           \
    case Isa::kSse42:                                                                          \
      return &detail::fnbase##_sse4;                                                           \
    default:                                                                                   \
      return &detail::fnbase##_scalar;
#elif defined(HYLE_ARCH_ARM64)
#  define HYLE_ARCH_CASES(fnbase)                                                              \
    case Isa::kNeon:                                                                           \
      return &detail::fnbase##_neon;                                                           \
    default:                                                                                   \
      return &detail::fnbase##_scalar;
#else
#  define HYLE_ARCH_CASES(fnbase)                                                              \
    default:                                                                                   \
      return &detail::fnbase##_scalar;
#endif

HYLE_RESOLVE(std::uint32_t, u32)
HYLE_RESOLVE(std::int32_t, i32)
HYLE_RESOLVE(std::uint64_t, u64)
HYLE_RESOLVE(std::int64_t, i64)
HYLE_RESOLVE(float, f32)
HYLE_RESOLVE(double, f64)

#undef HYLE_RESOLVE
#undef HYLE_ARCH_CASES

}  // namespace

#define HYLE_SCAN_API(T)                                                                       \
  void inclusive_scan(T* data, std::size_t n) noexcept {                                       \
    static const ScanFn<T> fn = resolve_inclusive<T>();                                        \
    fn(data, n);                                                                               \
  }                                                                                            \
  void exclusive_scan(T* data, std::size_t n) noexcept {                                       \
    static const ScanFn<T> fn = resolve_exclusive<T>();                                        \
    fn(data, n);                                                                               \
  }

HYLE_SCAN_API(std::uint32_t)
HYLE_SCAN_API(std::int32_t)
HYLE_SCAN_API(std::uint64_t)
HYLE_SCAN_API(std::int64_t)
HYLE_SCAN_API(float)
HYLE_SCAN_API(double)

#undef HYLE_SCAN_API

Isa active_scan_variant() noexcept {
  static const Isa tier = selected_tier();
  return tier;
}

}  // namespace hyle::simd

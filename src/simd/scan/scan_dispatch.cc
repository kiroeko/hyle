#include "hyle/simd/scan.h"

#include <cstddef>
#include <cstdint>

#include "hyle/platform.h"
#include "src/simd/scan/kernels.h"

// Baseline dispatcher (no -march). Same selected_tier model as ops/dispatch_ops.cc.
//
// PLATFORM SPLIT (2026-07-15, scan_wsl_probe):
// On Linux x86-64, a plain `a[i]+=a[i-1]` chain is already ~1 cyc/elem under GCC; single-
// stream SIMD Kogge-Stone + carry measured ~0.70–0.84× that (slower). So Linux x64 scan
// always uses a pure scalar loop. Windows MSVC scalar is much slower → SIMD wins ~3.8×.
// Arm64 keeps NEON (not the Linux-x64 pathology). Mac follows the arch path (no Linux pin).

namespace hyle::simd {
namespace {

Isa selected_tier() noexcept {
#if defined(HYLE_OS_LINUX) && defined(HYLE_ARCH_X64)
  // See file header: SIMD single-stream prefix loses to GCC's scalar chain on this OS/arch.
  return Isa::kScalar;
#elif defined(HYLE_ARCH_X64)
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

// True scalar chain (no soft-SIMD KS). Used on Linux x64 and as the portable floor.
template <typename T>
void inclusive_scan_pure(T* data, std::size_t n) noexcept {
  if (n == 0) return;
  for (std::size_t i = 1; i < n; ++i) data[i] = static_cast<T>(data[i - 1] + data[i]);
}

template <typename T>
void exclusive_scan_pure(T* data, std::size_t n) noexcept {
  if (n == 0) return;
  T acc = T{0};
  for (std::size_t i = 0; i < n; ++i) {
    const T v = data[i];
    data[i] = acc;
    acc = static_cast<T>(acc + v);
  }
}

template <typename T>
ScanFn<T> resolve_inclusive() noexcept;

template <typename T>
ScanFn<T> resolve_exclusive() noexcept;

#define HYLE_RESOLVE(T, tag)                                                                   \
  template <>                                                                                  \
  ScanFn<T> resolve_inclusive<T>() noexcept {                                                  \
    switch (selected_tier()) {                                                                 \
      HYLE_ARCH_CASES(inclusive_scan_##tag, inclusive_scan_pure<T>)                            \
    }                                                                                          \
  }                                                                                            \
  template <>                                                                                  \
  ScanFn<T> resolve_exclusive<T>() noexcept {                                                  \
    switch (selected_tier()) {                                                                 \
      HYLE_ARCH_CASES(exclusive_scan_##tag, exclusive_scan_pure<T>)                            \
    }                                                                                          \
  }

#if defined(HYLE_OS_LINUX) && defined(HYLE_ARCH_X64)
// Force pure scalar regardless of CPUID tier (selected_tier already returns kScalar, but
// keep the pure loop — not the soft-SIMD scalar backend KS).
#  define HYLE_ARCH_CASES(fnbase, pure_fn)                                                     \
    default:                                                                                   \
      return &pure_fn;
#elif defined(HYLE_ARCH_X64)
#  define HYLE_ARCH_CASES(fnbase, pure_fn)                                                     \
    case Isa::kAvx512:                                                                         \
      return &detail::fnbase##_avx512;                                                         \
    case Isa::kAvx2:                                                                           \
      return &detail::fnbase##_avx2;                                                           \
    case Isa::kSse42:                                                                          \
      return &detail::fnbase##_sse4;                                                           \
    default:                                                                                   \
      return &pure_fn;
#elif defined(HYLE_ARCH_ARM64)
#  define HYLE_ARCH_CASES(fnbase, pure_fn)                                                     \
    case Isa::kNeon:                                                                           \
      return &detail::fnbase##_neon;                                                           \
    default:                                                                                   \
      return &pure_fn;
#else
#  define HYLE_ARCH_CASES(fnbase, pure_fn)                                                     \
    default:                                                                                   \
      return &pure_fn;
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

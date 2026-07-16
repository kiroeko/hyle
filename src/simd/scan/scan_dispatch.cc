#include "hyle/simd/scan.h"

#include <cstddef>
#include <cstdint>

#include "src/simd/dispatch_internal.h"
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

#if defined(HYLE_OS_LINUX) && defined(HYLE_ARCH_X64)
// Force pure scalar regardless of CPUID tier (selected_tier already returns kScalar, but
// keep the pure loop — not the soft-SIMD scalar backend KS).
#  define HYLE_SIMD_SCAN_RESOLVE(T, tag)                                                       \
    template <>                                                                                \
    ScanFn<T> resolve_inclusive<T>() noexcept { return &inclusive_scan_pure<T>; }              \
    template <>                                                                                \
    ScanFn<T> resolve_exclusive<T>() noexcept { return &exclusive_scan_pure<T>; }
#elif defined(HYLE_ARCH_X64)
#  define HYLE_SIMD_SCAN_RESOLVE(T, tag)                                                       \
    template <>                                                                                \
    ScanFn<T> resolve_inclusive<T>() noexcept {                                                \
      return detail::resolve_default_variant<ScanFn<T>>(                                       \
          &inclusive_scan_pure<T>, &detail::inclusive_scan_##tag##_sse4,                      \
          &detail::inclusive_scan_##tag##_avx2, &detail::inclusive_scan_##tag##_avx512);      \
    }                                                                                           \
    template <>                                                                                \
    ScanFn<T> resolve_exclusive<T>() noexcept {                                                \
      return detail::resolve_default_variant<ScanFn<T>>(                                       \
          &exclusive_scan_pure<T>, &detail::exclusive_scan_##tag##_sse4,                      \
          &detail::exclusive_scan_##tag##_avx2, &detail::exclusive_scan_##tag##_avx512);      \
    }
#elif defined(HYLE_ARCH_ARM64)
#if defined(HYLE_SIMD_HAS_SVE_VARIANT)
#  define HYLE_SIMD_SCAN_RESOLVE(T, tag)                                                       \
    template <> ScanFn<T> resolve_inclusive<T>() noexcept {                                   \
      return detail::resolve_default_variant<ScanFn<T>>(                                      \
          &inclusive_scan_pure<T>, &detail::inclusive_scan_##tag##_neon,                     \
          &detail::inclusive_scan_##tag##_sve);                                               \
    }                                                                                          \
    template <> ScanFn<T> resolve_exclusive<T>() noexcept {                                   \
      return detail::resolve_default_variant<ScanFn<T>>(                                      \
          &exclusive_scan_pure<T>, &detail::exclusive_scan_##tag##_neon,                     \
          &detail::exclusive_scan_##tag##_sve);                                               \
    }
#else
#  define HYLE_SIMD_SCAN_RESOLVE(T, tag)                                                       \
    template <>                                                                                \
    ScanFn<T> resolve_inclusive<T>() noexcept {                                                \
      return detail::resolve_default_variant<ScanFn<T>>(                                       \
          &inclusive_scan_pure<T>, &detail::inclusive_scan_##tag##_neon);                     \
    }                                                                                           \
    template <>                                                                                \
    ScanFn<T> resolve_exclusive<T>() noexcept {                                                \
      return detail::resolve_default_variant<ScanFn<T>>(                                       \
          &exclusive_scan_pure<T>, &detail::exclusive_scan_##tag##_neon);                     \
    }
#endif
#elif defined(HYLE_ARCH_RISCV64) && defined(HYLE_SIMD_HAS_RVV_VARIANT)
#  define HYLE_SIMD_SCAN_RESOLVE(T, tag)                                                       \
    template <> ScanFn<T> resolve_inclusive<T>() noexcept {                                   \
      return detail::resolve_default_variant<ScanFn<T>>(                                      \
          &inclusive_scan_pure<T>, &detail::inclusive_scan_##tag##_rvv);                      \
    }                                                                                          \
    template <> ScanFn<T> resolve_exclusive<T>() noexcept {                                   \
      return detail::resolve_default_variant<ScanFn<T>>(                                      \
          &exclusive_scan_pure<T>, &detail::exclusive_scan_##tag##_rvv);                      \
    }
#else
#  define HYLE_SIMD_SCAN_RESOLVE(T, tag)                                                       \
    template <>                                                                                \
    ScanFn<T> resolve_inclusive<T>() noexcept { return &inclusive_scan_pure<T>; }              \
    template <>                                                                                \
    ScanFn<T> resolve_exclusive<T>() noexcept { return &exclusive_scan_pure<T>; }
#endif

HYLE_SIMD_SCAN_RESOLVE(std::uint32_t, u32)
HYLE_SIMD_SCAN_RESOLVE(std::int32_t, i32)
HYLE_SIMD_SCAN_RESOLVE(std::uint64_t, u64)
HYLE_SIMD_SCAN_RESOLVE(std::int64_t, i64)
HYLE_SIMD_SCAN_RESOLVE(float, f32)
HYLE_SIMD_SCAN_RESOLVE(double, f64)

#undef HYLE_SIMD_SCAN_RESOLVE

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

}  // namespace hyle::simd

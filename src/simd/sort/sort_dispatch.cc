#include "hyle/simd/sort.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "hyle/macros.h"  // HYLE_FATAL_ASSERT (fail-fast on size-contract violation)
#include "hyle/platform.h"
#include "hyle/simd/dispatch.h"
#include "src/simd/sort/kernels.h"

// sort_dispatch.cc -- baseline dispatcher for the vqsort borrow ops. Mirrors math_dispatch.cc:
// per-type it routes to the widest LIVE tier -> x86 avx512 -> avx2 -> scalar; arm64 NEON
// (unconditional -- ARMv8 baseline); everything else the std:: scalar floor. The borrow cores are
// vendored header-only (can't be missing) + compiled per tier; this dispatcher only enters a wide
// tier when the CPU supports it (cpu_features()), so an old CPU never SIGILLs.
//
// Unlike math (per-OP best-of-breed across two libs), sort borrows ONE lib (Highway -- only it
// ships a vectorized sort), so there is no per-op library pick, just the tier ladder. vqsort's
// AVX-512 target is NOT MSVC-broken (integer/compare SIMD, not FP transcendentals), so the 512
// tier is used on all compilers -- no _MSC_VER exclusion (contrast the math AVX3 log core).

namespace hyle::simd {
namespace {

bool have_avx2() noexcept {
#if defined(HYLE_ARCH_X64)
  return cpu_features().avx2;
#else
  return false;
#endif
}

// AVX-512 group (f+bw+vl+dq+cd) -- same condition as math_dispatch's have_avx512 / best_isa()>=
// kAvx512. vqsort's sorting networks + Compress widen to 512-bit here.
bool have_avx512() noexcept {
#if defined(HYLE_ARCH_X64)
  const auto& f = cpu_features();
  return f.avx512f && f.avx512bw && f.avx512vl && f.avx512dq && f.avx512cd;
#else
  return false;
#endif
}

// P1-2: large integer arrays use stable LSD radix. Template so float/double discard cleanly.
template <typename T>
bool try_radix_sort(T* keys, std::size_t n, SortOrder order) noexcept {
  if constexpr (std::is_integral_v<T>) {
    if (n >= kRadixSortMinN) {
      radix_sort(keys, n, order);
      return true;
    }
  }
  return false;
}

}  // namespace

// Generates the three public overloads (sort/partial_sort/select) for one key type. The tier
// ladder is identical for all types; only the detail:: symbol suffix (tag) differs. `ascending`
// collapses SortOrder to the bool the borrow cores take. Size contract (fail-fast): n==0 is a
// no-op; partial_sort needs k<=n; select needs k<n (matches VQSelectStatic's k<num precondition).
#if defined(HYLE_ARCH_X64)
#define HYLE_SORT_TIER_SORT(tag) \
  do { \
    static const bool hw512 = have_avx512(); \
    if (hw512) { detail::sort_##tag##_avx512(keys, n, asc); return; } \
    static const bool hw2 = have_avx2(); \
    if (hw2) { detail::sort_##tag##_avx2(keys, n, asc); return; } \
    detail::sort_##tag##_scalar(keys, n, asc); \
  } while (0)
#define HYLE_SORT_TIER_PSORT(tag) \
  do { \
    static const bool hw512 = have_avx512(); \
    if (hw512) { detail::psort_##tag##_avx512(keys, n, k, asc); return; } \
    static const bool hw2 = have_avx2(); \
    if (hw2) { detail::psort_##tag##_avx2(keys, n, k, asc); return; } \
    detail::psort_##tag##_scalar(keys, n, k, asc); \
  } while (0)
#define HYLE_SORT_TIER_SELECT(tag) \
  do { \
    static const bool hw512 = have_avx512(); \
    if (hw512) { detail::select_##tag##_avx512(keys, n, k, asc); return; } \
    static const bool hw2 = have_avx2(); \
    if (hw2) { detail::select_##tag##_avx2(keys, n, k, asc); return; } \
    detail::select_##tag##_scalar(keys, n, k, asc); \
  } while (0)
#elif defined(HYLE_ARCH_ARM64)
#define HYLE_SORT_TIER_SORT(tag)   detail::sort_##tag##_neon(keys, n, asc)
#define HYLE_SORT_TIER_PSORT(tag)  detail::psort_##tag##_neon(keys, n, k, asc)
#define HYLE_SORT_TIER_SELECT(tag) detail::select_##tag##_neon(keys, n, k, asc)
#else
#define HYLE_SORT_TIER_SORT(tag)   detail::sort_##tag##_scalar(keys, n, asc)
#define HYLE_SORT_TIER_PSORT(tag)  detail::psort_##tag##_scalar(keys, n, k, asc)
#define HYLE_SORT_TIER_SELECT(tag) detail::select_##tag##_scalar(keys, n, k, asc)
#endif

// P1-3: k ≪ n uses streaming threshold partial (sorted prefix + single-compare reject).
// Beats VQPartialSort on small-k large-n (VQ still partitions/writes the whole array).
// Float included: StreamOrder matches NaN-after-finite contract (same as scalar floor).
#define HYLE_SORT_DISPATCH(T, tag)                                                         \
  void sort(T* keys, std::size_t n, SortOrder order) noexcept {                            \
    if (n == 0) return;                                                                    \
    if (try_radix_sort(keys, n, order)) return;                                            \
    const bool asc = (order == SortOrder::kAscending);                                     \
    HYLE_SORT_TIER_SORT(tag);                                                              \
  }                                                                                        \
  void partial_sort(T* keys, std::size_t n, std::size_t k, SortOrder order) noexcept {     \
    HYLE_FATAL_ASSERT(k <= n);                                                             \
    if (n == 0 || k == 0) return;                                                          \
    const bool asc = (order == SortOrder::kAscending);                                     \
    if (detail::stream_partial_##tag(keys, n, k, asc)) return;                             \
    HYLE_SORT_TIER_PSORT(tag);                                                             \
  }                                                                                        \
  void select(T* keys, std::size_t n, std::size_t k, SortOrder order) noexcept {           \
    HYLE_FATAL_ASSERT(k < n);                                                              \
    const bool asc = (order == SortOrder::kAscending);                                     \
    HYLE_SORT_TIER_SELECT(tag);                                                            \
  }

HYLE_SORT_KEY_TYPES(HYLE_SORT_DISPATCH)
#undef HYLE_SORT_DISPATCH

// KV (top-k) dispatch. Same tier ladder as the scalar-T ops, but over the KV symbol suffix
// (kv32/kv64). The public KeyValue32/64 are byte-identical to the internal KV32Bits/KV64Bits
// ({value,key}), so we reinterpret_cast the public pointer to the detail struct the kernels take.
#if defined(HYLE_ARCH_X64)
#define HYLE_SORT_KV_LADDER_SORT(kvtag, h) \
  do { \
    static const bool hw512 = have_avx512(); \
    if (hw512) { detail::sort_##kvtag##_avx512(h, n, asc); return; } \
    static const bool hw2 = have_avx2(); \
    if (hw2) { detail::sort_##kvtag##_avx2(h, n, asc); return; } \
    detail::sort_##kvtag##_scalar(h, n, asc); \
  } while (0)
#define HYLE_SORT_KV_LADDER_PSORT(kvtag, h) \
  do { \
    static const bool hw512 = have_avx512(); \
    if (hw512) { detail::psort_##kvtag##_avx512(h, n, k, asc); return; } \
    static const bool hw2 = have_avx2(); \
    if (hw2) { detail::psort_##kvtag##_avx2(h, n, k, asc); return; } \
    detail::psort_##kvtag##_scalar(h, n, k, asc); \
  } while (0)
#define HYLE_SORT_KV_LADDER_SELECT(kvtag, h) \
  do { \
    static const bool hw512 = have_avx512(); \
    if (hw512) { detail::select_##kvtag##_avx512(h, n, k, asc); return; } \
    static const bool hw2 = have_avx2(); \
    if (hw2) { detail::select_##kvtag##_avx2(h, n, k, asc); return; } \
    detail::select_##kvtag##_scalar(h, n, k, asc); \
  } while (0)
#elif defined(HYLE_ARCH_ARM64)
#define HYLE_SORT_KV_LADDER_SORT(kvtag, h)   detail::sort_##kvtag##_neon(h, n, asc)
#define HYLE_SORT_KV_LADDER_PSORT(kvtag, h)  detail::psort_##kvtag##_neon(h, n, k, asc)
#define HYLE_SORT_KV_LADDER_SELECT(kvtag, h) detail::select_##kvtag##_neon(h, n, k, asc)
#else
#define HYLE_SORT_KV_LADDER_SORT(kvtag, h)   detail::sort_##kvtag##_scalar(h, n, asc)
#define HYLE_SORT_KV_LADDER_PSORT(kvtag, h)  detail::psort_##kvtag##_scalar(h, n, k, asc)
#define HYLE_SORT_KV_LADDER_SELECT(kvtag, h) detail::select_##kvtag##_scalar(h, n, k, asc)
#endif

#define HYLE_SORT_KV_DISPATCH(PubT, DetT, kvtag)                                           \
  void sort(PubT* kv, std::size_t n, SortOrder order) noexcept {                           \
    if (n == 0) return;                                                                    \
    const bool asc = (order == SortOrder::kAscending);                                     \
    auto* h = reinterpret_cast<detail::DetT*>(kv);                                         \
    HYLE_SORT_KV_LADDER_SORT(kvtag, h);                                                    \
  }                                                                                        \
  void partial_sort(PubT* kv, std::size_t n, std::size_t k, SortOrder order) noexcept {    \
    HYLE_FATAL_ASSERT(k <= n);                                                             \
    if (n == 0 || k == 0) return;                                                          \
    const bool asc = (order == SortOrder::kAscending);                                     \
    auto* h = reinterpret_cast<detail::DetT*>(kv);                                         \
    if (detail::stream_partial_##kvtag(h, n, k, asc)) return;                              \
    HYLE_SORT_KV_LADDER_PSORT(kvtag, h);                                                   \
  }                                                                                        \
  void select(PubT* kv, std::size_t n, std::size_t k, SortOrder order) noexcept {          \
    HYLE_FATAL_ASSERT(k < n);                                                              \
    const bool asc = (order == SortOrder::kAscending);                                     \
    auto* h = reinterpret_cast<detail::DetT*>(kv);                                         \
    HYLE_SORT_KV_LADDER_SELECT(kvtag, h);                                                  \
  }

HYLE_SORT_KV_DISPATCH(KeyValue32, KV32Bits, kv32)
HYLE_SORT_KV_DISPATCH(KeyValue64, KV64Bits, kv64)
#undef HYLE_SORT_KV_DISPATCH

Isa active_sort_variant() noexcept {
#if defined(HYLE_ARCH_X64)
  static const Isa tier =
      have_avx512() ? Isa::kAvx512 : (have_avx2() ? Isa::kAvx2 : Isa::kScalar);
  return tier;
#elif defined(HYLE_ARCH_ARM64)
  return Isa::kNeon;  // Highway NEON vqsort core always compiled (ARMv8 baseline)
#else
  return Isa::kScalar;
#endif
}

}  // namespace hyle::simd

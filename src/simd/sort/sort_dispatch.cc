#include "hyle/simd/sort.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "hyle/core/compiler.h"  // HYLE_FATAL_ASSERT (fail-fast on size-contract violation)
#include "hyle/platform.h"
#include "src/simd/runtime_internal.h"
#include "src/simd/dispatch_internal.h"
#include "src/simd/sort/kernels.h"

// sort_dispatch.cc -- baseline dispatcher for the vqsort borrow ops. Mirrors math_dispatch.cc:
// per-type it routes to the widest compiler-safe LIVE tier -> x86 avx512 -> avx2 -> scalar;
// arm64 SVE full-VL partition + capped network -> NEON; riscv64 RVV -> scalar. The right-hand tier
// is the fallback when a scalable variant is unavailable, not an unconditional module clamp. The
// borrow cores are
// vendored header-only (can't be missing) + compiled per tier; this dispatcher only enters a wide
// tier when the CPU supports it (cpu_features()), so an old CPU never SIGILLs.
//
// Unlike math (per-OP best-of-breed across two libs), sort has one vectorized implementation,
// so there is no per-op library pick, just the tier ladder. Highway marks all AVX3 targets broken
// under the classic MSVC frontend; HYLE's port shares that codegen boundary. cl.exe therefore
// binds AVX-512 product builds to the AVX2 sort TU. clang-cl and GNU-like frontends retain AVX-512.

namespace hyle::simd {
namespace {

#define HYLE_SORT_SYMBOL_I(name, tag) name##_##tag
#define HYLE_SORT_SYMBOL(name, tag) HYLE_SORT_SYMBOL_I(name, tag)
#if defined(HYLE_COMPILER_MSVC)
#define HYLE_SORT_AVX512_SYMBOL(name) detail::HYLE_SIMD_SUFFIXED_SYMBOL(name, avx2)
#else
#define HYLE_SORT_AVX512_SYMBOL(name) detail::HYLE_SIMD_SUFFIXED_SYMBOL(name, avx512)
#endif
#if defined(HYLE_SIMD_EXACT_TARGET_SSE4) || defined(HYLE_SIMD_EXACT_TARGET_SCALAR)
#define HYLE_SORT_EXACT_SYMBOL(name) detail::HYLE_SIMD_SUFFIXED_SYMBOL(name, scalar)
#elif defined(HYLE_SIMD_EXACT_TARGET_AVX512) && defined(HYLE_COMPILER_MSVC)
#define HYLE_SORT_EXACT_SYMBOL(name) HYLE_SORT_AVX512_SYMBOL(name)
#elif defined(HYLE_SIMD_EXACT)
#define HYLE_SORT_EXACT_SYMBOL(name) HYLE_SIMD_EXACT_SYMBOL(name)
#endif

#if defined(HYLE_ARCH_X64) && !defined(HYLE_SIMD_EXACT)
bool have_avx2() noexcept {
#if defined(HYLE_SIMD_EXACT_TARGET_AVX2) || defined(HYLE_SIMD_EXACT_TARGET_AVX512)
  return true;
#elif defined(HYLE_SIMD_EXACT)
  return false;
#elif defined(HYLE_ARCH_X64)
  return runtime().sort.isa == Isa::kAvx2 || runtime().sort.isa == Isa::kAvx512;
#else
  return false;
#endif
}

// AVX-512 group (f+bw+vl+dq+cd), additionally constrained by the compiler policy encoded in
// runtime().sort.isa. On cl.exe the product runtime remains AVX-512 but this module reports AVX2.
bool have_avx512() noexcept {
#if defined(HYLE_SIMD_EXACT_TARGET_AVX512)
  return true;
#elif defined(HYLE_SIMD_EXACT)
  return false;
#elif defined(HYLE_ARCH_X64)
  return runtime().sort.isa == Isa::kAvx512;
#else
  return false;
#endif
}
#endif

// P1-2: large integer arrays use stable LSD radix. Template so float/double discard cleanly.
template <typename T>
bool try_radix_sort(T* keys, std::size_t n, SortOrder order) noexcept {
  if constexpr (std::is_same_v<T, std::uint32_t> || std::is_same_v<T, std::int32_t> ||
                std::is_same_v<T, std::uint64_t> || std::is_same_v<T, std::int64_t>) {
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
#if defined(HYLE_SIMD_EXACT)
#define HYLE_SORT_TIER_SORT(tag) \
  HYLE_SORT_EXACT_SYMBOL(HYLE_SORT_SYMBOL(sort, tag))(keys, n, asc)
#define HYLE_SORT_TIER_PSORT(tag) \
  HYLE_SORT_EXACT_SYMBOL(HYLE_SORT_SYMBOL(psort, tag))(keys, n, k, asc)
#define HYLE_SORT_TIER_SELECT(tag) \
  HYLE_SORT_EXACT_SYMBOL(HYLE_SORT_SYMBOL(select, tag))(keys, n, k, asc)
#elif defined(HYLE_ARCH_X64)
#define HYLE_SORT_TIER_SORT(tag) \
  do { \
    static const bool hw512 = have_avx512(); \
    if (hw512) { HYLE_SORT_AVX512_SYMBOL(HYLE_SORT_SYMBOL(sort, tag))(keys, n, asc); return; } \
    static const bool hw2 = have_avx2(); \
    if (hw2) { detail::sort_##tag##_avx2(keys, n, asc); return; } \
    detail::sort_##tag##_scalar(keys, n, asc); \
  } while (0)
#define HYLE_SORT_TIER_PSORT(tag) \
  do { \
    static const bool hw512 = have_avx512(); \
    if (hw512) { HYLE_SORT_AVX512_SYMBOL(HYLE_SORT_SYMBOL(psort, tag))(keys, n, k, asc); return; } \
    static const bool hw2 = have_avx2(); \
    if (hw2) { detail::psort_##tag##_avx2(keys, n, k, asc); return; } \
    detail::psort_##tag##_scalar(keys, n, k, asc); \
  } while (0)
#define HYLE_SORT_TIER_SELECT(tag) \
  do { \
    static const bool hw512 = have_avx512(); \
    if (hw512) { HYLE_SORT_AVX512_SYMBOL(HYLE_SORT_SYMBOL(select, tag))(keys, n, k, asc); return; } \
    static const bool hw2 = have_avx2(); \
    if (hw2) { detail::select_##tag##_avx2(keys, n, k, asc); return; } \
    detail::select_##tag##_scalar(keys, n, k, asc); \
  } while (0)
#elif defined(HYLE_ARCH_ARM64) && defined(HYLE_SIMD_HAS_SVE_VARIANT)
#if defined(HYLE_SIMD_EXACT_TARGET_SVE)
#define HYLE_SORT_USE_SVE() true
#elif defined(HYLE_SIMD_EXACT)
#define HYLE_SORT_USE_SVE() false
#else
#define HYLE_SORT_USE_SVE() (runtime().sort.isa == Isa::kSve)
#endif
#define HYLE_SORT_TIER_SORT(tag) do { if (HYLE_SORT_USE_SVE()) {                    \
  detail::sort_##tag##_sve(keys, n, asc); return; }                                 \
  detail::sort_##tag##_neon(keys, n, asc); } while (0)
#define HYLE_SORT_TIER_PSORT(tag) do { if (HYLE_SORT_USE_SVE()) {                   \
  detail::psort_##tag##_sve(keys, n, k, asc); return; }                             \
  detail::psort_##tag##_neon(keys, n, k, asc); } while (0)
#define HYLE_SORT_TIER_SELECT(tag) do { if (HYLE_SORT_USE_SVE()) {                  \
  detail::select_##tag##_sve(keys, n, k, asc); return; }                            \
  detail::select_##tag##_neon(keys, n, k, asc); } while (0)
#elif defined(HYLE_ARCH_ARM64)
#define HYLE_SORT_TIER_SORT(tag)   detail::sort_##tag##_neon(keys, n, asc)
#define HYLE_SORT_TIER_PSORT(tag)  detail::psort_##tag##_neon(keys, n, k, asc)
#define HYLE_SORT_TIER_SELECT(tag) detail::select_##tag##_neon(keys, n, k, asc)
#elif defined(HYLE_ARCH_RISCV64) && defined(HYLE_SIMD_HAS_RVV_VARIANT)
#if defined(HYLE_SIMD_EXACT_TARGET_RVV)
#define HYLE_SORT_USE_RVV() true
#elif defined(HYLE_SIMD_EXACT)
#define HYLE_SORT_USE_RVV() false
#else
#define HYLE_SORT_USE_RVV() (runtime().sort.isa == Isa::kRvv)
#endif
#define HYLE_SORT_TIER_SORT(tag) do { if (HYLE_SORT_USE_RVV()) {                    \
  detail::sort_##tag##_rvv(keys, n, asc); return; }                                 \
  detail::sort_##tag##_scalar(keys, n, asc); } while (0)
#define HYLE_SORT_TIER_PSORT(tag) do { if (HYLE_SORT_USE_RVV()) {                   \
  detail::psort_##tag##_rvv(keys, n, k, asc); return; }                             \
  detail::psort_##tag##_scalar(keys, n, k, asc); } while (0)
#define HYLE_SORT_TIER_SELECT(tag) do { if (HYLE_SORT_USE_RVV()) {                  \
  detail::select_##tag##_rvv(keys, n, k, asc); return; }                            \
  detail::select_##tag##_scalar(keys, n, k, asc); } while (0)
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
#if defined(HYLE_SIMD_EXACT)
#define HYLE_SORT_KV_LADDER_SORT(kvtag, h) \
  HYLE_SORT_EXACT_SYMBOL(HYLE_SORT_SYMBOL(sort, kvtag))(h, n, asc)
#define HYLE_SORT_KV_LADDER_PSORT(kvtag, h) \
  HYLE_SORT_EXACT_SYMBOL(HYLE_SORT_SYMBOL(psort, kvtag))(h, n, k, asc)
#define HYLE_SORT_KV_LADDER_SELECT(kvtag, h) \
  HYLE_SORT_EXACT_SYMBOL(HYLE_SORT_SYMBOL(select, kvtag))(h, n, k, asc)
#elif defined(HYLE_ARCH_X64)
#define HYLE_SORT_KV_LADDER_SORT(kvtag, h) \
  do { \
    static const bool hw512 = have_avx512(); \
    if (hw512) { HYLE_SORT_AVX512_SYMBOL(HYLE_SORT_SYMBOL(sort, kvtag))(h, n, asc); return; } \
    static const bool hw2 = have_avx2(); \
    if (hw2) { detail::sort_##kvtag##_avx2(h, n, asc); return; } \
    detail::sort_##kvtag##_scalar(h, n, asc); \
  } while (0)
#define HYLE_SORT_KV_LADDER_PSORT(kvtag, h) \
  do { \
    static const bool hw512 = have_avx512(); \
    if (hw512) { HYLE_SORT_AVX512_SYMBOL(HYLE_SORT_SYMBOL(psort, kvtag))(h, n, k, asc); return; } \
    static const bool hw2 = have_avx2(); \
    if (hw2) { detail::psort_##kvtag##_avx2(h, n, k, asc); return; } \
    detail::psort_##kvtag##_scalar(h, n, k, asc); \
  } while (0)
#define HYLE_SORT_KV_LADDER_SELECT(kvtag, h) \
  do { \
    static const bool hw512 = have_avx512(); \
    if (hw512) { HYLE_SORT_AVX512_SYMBOL(HYLE_SORT_SYMBOL(select, kvtag))(h, n, k, asc); return; } \
    static const bool hw2 = have_avx2(); \
    if (hw2) { detail::select_##kvtag##_avx2(h, n, k, asc); return; } \
    detail::select_##kvtag##_scalar(h, n, k, asc); \
  } while (0)
#elif defined(HYLE_ARCH_ARM64) && defined(HYLE_SIMD_HAS_SVE_VARIANT)
#define HYLE_SORT_KV_LADDER_SORT(kvtag, h) do { if (HYLE_SORT_USE_SVE()) {          \
  detail::sort_##kvtag##_sve(h, n, asc); return; }                                  \
  detail::sort_##kvtag##_neon(h, n, asc); } while (0)
#define HYLE_SORT_KV_LADDER_PSORT(kvtag, h) do { if (HYLE_SORT_USE_SVE()) {         \
  detail::psort_##kvtag##_sve(h, n, k, asc); return; }                              \
  detail::psort_##kvtag##_neon(h, n, k, asc); } while (0)
#define HYLE_SORT_KV_LADDER_SELECT(kvtag, h) do { if (HYLE_SORT_USE_SVE()) {        \
  detail::select_##kvtag##_sve(h, n, k, asc); return; }                             \
  detail::select_##kvtag##_neon(h, n, k, asc); } while (0)
#elif defined(HYLE_ARCH_ARM64)
#define HYLE_SORT_KV_LADDER_SORT(kvtag, h)   detail::sort_##kvtag##_neon(h, n, asc)
#define HYLE_SORT_KV_LADDER_PSORT(kvtag, h)  detail::psort_##kvtag##_neon(h, n, k, asc)
#define HYLE_SORT_KV_LADDER_SELECT(kvtag, h) detail::select_##kvtag##_neon(h, n, k, asc)
#elif defined(HYLE_ARCH_RISCV64) && defined(HYLE_SIMD_HAS_RVV_VARIANT)
#define HYLE_SORT_KV_LADDER_SORT(kvtag, h) do { if (HYLE_SORT_USE_RVV()) {          \
  detail::sort_##kvtag##_rvv(h, n, asc); return; }                                  \
  detail::sort_##kvtag##_scalar(h, n, asc); } while (0)
#define HYLE_SORT_KV_LADDER_PSORT(kvtag, h) do { if (HYLE_SORT_USE_RVV()) {         \
  detail::psort_##kvtag##_rvv(h, n, k, asc); return; }                              \
  detail::psort_##kvtag##_scalar(h, n, k, asc); } while (0)
#define HYLE_SORT_KV_LADDER_SELECT(kvtag, h) do { if (HYLE_SORT_USE_RVV()) {        \
  detail::select_##kvtag##_rvv(h, n, k, asc); return; }                             \
  detail::select_##kvtag##_scalar(h, n, k, asc); } while (0)
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

static_assert(sizeof(UInt128) == sizeof(detail::U128Bits));
static_assert(alignof(UInt128) == alignof(detail::U128Bits));
static_assert(std::is_standard_layout_v<UInt128>);
static_assert(std::is_standard_layout_v<detail::U128Bits>);

void sort(UInt128* keys, std::size_t n, SortOrder order) noexcept {
  if (n == 0) return;
  const bool asc = (order == SortOrder::kAscending);
  auto* h = reinterpret_cast<detail::U128Bits*>(keys);
  HYLE_SORT_KV_LADDER_SORT(u128, h);
}

void partial_sort(UInt128* keys, std::size_t n, std::size_t k, SortOrder order) noexcept {
  HYLE_FATAL_ASSERT(k <= n);
  if (n == 0 || k == 0) return;
  const bool asc = (order == SortOrder::kAscending);
  auto* h = reinterpret_cast<detail::U128Bits*>(keys);
  HYLE_SORT_KV_LADDER_PSORT(u128, h);
}

void select(UInt128* keys, std::size_t n, std::size_t k, SortOrder order) noexcept {
  HYLE_FATAL_ASSERT(k < n);
  const bool asc = (order == SortOrder::kAscending);
  auto* h = reinterpret_cast<detail::U128Bits*>(keys);
  HYLE_SORT_KV_LADDER_SELECT(u128, h);
}

#undef HYLE_SORT_TIER_SORT
#undef HYLE_SORT_TIER_PSORT
#undef HYLE_SORT_TIER_SELECT
#undef HYLE_SORT_KV_LADDER_SORT
#undef HYLE_SORT_KV_LADDER_PSORT
#undef HYLE_SORT_KV_LADDER_SELECT
#if defined(HYLE_SORT_USE_RVV)
#undef HYLE_SORT_USE_RVV
#endif
#if defined(HYLE_SORT_USE_SVE)
#undef HYLE_SORT_USE_SVE
#endif
#if defined(HYLE_SORT_EXACT_SYMBOL)
#undef HYLE_SORT_EXACT_SYMBOL
#endif
#undef HYLE_SORT_AVX512_SYMBOL
#undef HYLE_SORT_SYMBOL
#undef HYLE_SORT_SYMBOL_I

}  // namespace hyle::simd

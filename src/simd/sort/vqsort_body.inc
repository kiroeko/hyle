// vqsort_body.inc -- shared body of the HYLE vqsort port cores. Included by each
// ISA tier TU (vqsort_hyle_avx2.cc / _avx512.cc / _neon.cc) AFTER it defines
// HYLE_SORT_TIER (symbol suffix: avx2 / avx512 / neon) and HYLE_SIMD_TARGET_*.
// This is NOT a standalone TU (no include guard -- textually pasted once per tier).
//
// Origin algorithm: Highway hwy/contrib/sort/vqsort-inl.h VQSortStatic path
// (SHA 83eff39371e5e6494b305b253f8a3c06f831faf4). Ported to HYLE D-tag primitives
// under src/simd/sort/vqsort_hyle/.

#include "src/simd/sort/kernels.h"

#include <cstddef>
#include <cstdint>

#include "src/simd/sort/vqsort_hyle/vqsort-inl.h"

namespace hyle::simd::detail {

#define HYLE_SORT_PASTE2(a, b) a##b
#define HYLE_SORT_PASTE(a, b) HYLE_SORT_PASTE2(a, b)
// Builds e.g. sort_u32_avx2 -- HYLE_SORT_TIER is a macro so it needs two-level paste.
#define HYLE_SORT_SYM(stem, tag) HYLE_SORT_PASTE(stem##_##tag##_, HYLE_SORT_TIER)

#define HYLE_SORT_DEFINE(T, tag)                                                              \
  void HYLE_SORT_SYM(sort, tag)(T* keys, std::size_t n, bool ascending) noexcept {            \
    if (ascending)                                                                            \
      vqsort::VQSortStatic(keys, n, vqsort::SortAscending{});                                 \
    else                                                                                      \
      vqsort::VQSortStatic(keys, n, vqsort::SortDescending{});                                \
  }                                                                                           \
  void HYLE_SORT_SYM(psort, tag)(T* keys, std::size_t n, std::size_t k, bool ascending)       \
      noexcept {                                                                              \
    if (ascending)                                                                            \
      vqsort::VQPartialSortStatic(keys, n, k, vqsort::SortAscending{});                       \
    else                                                                                      \
      vqsort::VQPartialSortStatic(keys, n, k, vqsort::SortDescending{});                      \
  }                                                                                           \
  void HYLE_SORT_SYM(select, tag)(T* keys, std::size_t n, std::size_t k, bool ascending)      \
      noexcept {                                                                              \
    if (ascending)                                                                            \
      vqsort::VQSelectStatic(keys, n, k, vqsort::SortAscending{});                            \
    else                                                                                      \
      vqsort::VQSelectStatic(keys, n, k, vqsort::SortDescending{});                           \
  }

HYLE_SORT_KEY_TYPES(HYLE_SORT_DEFINE)

#undef HYLE_SORT_DEFINE

// Unsigned 128-bit numeric keys: two u64 words, low word first. The vqsort traits compare high
// then low and already provide the paired-lane sorting networks.
#define HYLE_SORT_U128_DEFINE(KT, tag, tier)                                                \
  void HYLE_SORT_SYM(sort, tag)(KT* keys, std::size_t n, bool ascending) noexcept {         \
    auto* h = reinterpret_cast<vqsort::uint128_t*>(keys);                                  \
    if (ascending) vqsort::VQSortStatic(h, n, vqsort::SortAscending{});                    \
    else vqsort::VQSortStatic(h, n, vqsort::SortDescending{});                             \
  }                                                                                        \
  void HYLE_SORT_SYM(psort, tag)(KT* keys, std::size_t n, std::size_t k, bool ascending)   \
      noexcept {                                                                           \
    auto* h = reinterpret_cast<vqsort::uint128_t*>(keys);                                  \
    if (ascending) vqsort::VQPartialSortStatic(h, n, k, vqsort::SortAscending{});          \
    else vqsort::VQPartialSortStatic(h, n, k, vqsort::SortDescending{});                   \
  }                                                                                        \
  void HYLE_SORT_SYM(select, tag)(KT* keys, std::size_t n, std::size_t k, bool ascending)  \
      noexcept {                                                                           \
    auto* h = reinterpret_cast<vqsort::uint128_t*>(keys);                                  \
    if (ascending) vqsort::VQSelectStatic(h, n, k, vqsort::SortAscending{});               \
    else vqsort::VQSelectStatic(h, n, k, vqsort::SortDescending{});                        \
  }
HYLE_SORT_U128_TYPES(HYLE_SORT_U128_DEFINE, ignored)
#undef HYLE_SORT_U128_DEFINE

// KV (top-k): HYLE KV32Bits/KV64Bits are byte-identical to the port's K32V32/K64V64
// ({value, key}, same alignas), so reinterpret_cast is a zero-copy view.
#define HYLE_SORT_KV_DEFINE(KVT, PortT, kvtag)                                                \
  void HYLE_SORT_SYM(sort, kvtag)(KVT* kv, std::size_t n, bool ascending) noexcept {          \
    auto* h = reinterpret_cast<vqsort::PortT*>(kv);                                           \
    if (ascending) vqsort::VQSortStatic(h, n, vqsort::SortAscending{});                       \
    else vqsort::VQSortStatic(h, n, vqsort::SortDescending{});                                \
  }                                                                                           \
  void HYLE_SORT_SYM(psort, kvtag)(KVT* kv, std::size_t n, std::size_t k, bool ascending)     \
      noexcept {                                                                              \
    auto* h = reinterpret_cast<vqsort::PortT*>(kv);                                           \
    if (ascending) vqsort::VQPartialSortStatic(h, n, k, vqsort::SortAscending{});             \
    else vqsort::VQPartialSortStatic(h, n, k, vqsort::SortDescending{});                      \
  }                                                                                           \
  void HYLE_SORT_SYM(select, kvtag)(KVT* kv, std::size_t n, std::size_t k, bool ascending)    \
      noexcept {                                                                              \
    auto* h = reinterpret_cast<vqsort::PortT*>(kv);                                           \
    if (ascending) vqsort::VQSelectStatic(h, n, k, vqsort::SortAscending{});                  \
    else vqsort::VQSelectStatic(h, n, k, vqsort::SortDescending{});                           \
  }
HYLE_SORT_KV_DEFINE(KV32Bits, K32V32, kv32)
HYLE_SORT_KV_DEFINE(KV64Bits, K64V64, kv64)
#undef HYLE_SORT_KV_DEFINE

#undef HYLE_SORT_SYM
#undef HYLE_SORT_PASTE
#undef HYLE_SORT_PASTE2

}  // namespace hyle::simd::detail

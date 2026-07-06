// str_search_impl.h -- the SHARED count/rfind kernel bodies, written ONCE against the vec.h
// D-tag abstraction (no raw intrinsics). Each per-ISA TU (str_search_{scalar,sse4,avx2,neon}
// .cc) #defines HYLE_SIMD_TARGET_* + HYLE_STRSEARCH_SUFFIX then includes this, so the SAME
// source compiles to str_count_scalar / str_count_avx2 / str_count_neon / ... The backend +
// -march make the loop 16-wide (SSE/NEON), 32-wide (AVX2), or 64-wide (AVX512); the kernel
// author writes one loop. The dispatcher (string_ref.cc) picks among them by best_isa().
//
// These REPLACE the old hand-rolled string_ref_avx2.cc (256b) + the SSE2 core in
// string_ref.cc: one source, all ISAs, and NEON/AVX-512 for free (previously x86-only ->
// aarch64 fell to scalar). Count keeps the psadbw-style SAD accumulate (SumsOf8) with FOUR
// independent accumulators -- the single-accumulator loop lost std::count ~1.7x; rfind uses
// the reverse-scan FindLastTrue primitive.

#include <cstddef>
#include <cstdint>

#include "hyle/simd/vec.h"

#ifndef HYLE_STRSEARCH_SUFFIX
#error "define HYLE_STRSEARCH_SUFFIX (scalar|sse4|avx2|avx512|neon) before including this"
#endif

#define HYLE_STRSEARCH_CAT_(a, b) a##b
#define HYLE_STRSEARCH_CAT(a, b) HYLE_STRSEARCH_CAT_(a, b)
#define STR_COUNT_FN HYLE_STRSEARCH_CAT(str_count_, HYLE_STRSEARCH_SUFFIX)
#define STR_RFIND_FN HYLE_STRSEARCH_CAT(str_rfind_char_, HYLE_STRSEARCH_SUFFIX)

namespace hyle::container::detail {

namespace hs = ::hyle::simd;

// Count occurrences of `c` in data[0,size). SAD accumulate: Eq marks matches, VecFromMask
// turns them into 0xFF bytes, SumsOf8 horizontally sums each 8-byte group (a 0xFF match = 255)
// into u64 lanes; accumulate across vectors, sum the lanes, divide by 255. FOUR independent
// accumulators so the per-iter SumsOf8+Add chains don't serialize (single-acc lost ~1.7x).
std::uint32_t STR_COUNT_FN(const char* data, std::uint32_t size, char c) noexcept {
  const hs::ScalableTag<std::uint8_t> d;
  const hs::ScalableTag<std::uint64_t> d64;
  const std::size_t w = hs::Lanes(d);
  const auto needle = hs::Set(d, static_cast<std::uint8_t>(c));
  const auto* p = reinterpret_cast<const std::uint8_t*>(data);

  auto a0 = hs::Zero(d64), a1 = hs::Zero(d64), a2 = hs::Zero(d64), a3 = hs::Zero(d64);
  std::size_t i = 0;
  const std::size_t w4 = w * 4;
  for (; i + w4 <= size; i += w4) {
    a0 = hs::Add(a0, hs::SumsOf8(hs::VecFromMask(d, hs::Eq(hs::LoadU(d, p + i), needle))));
    a1 = hs::Add(a1, hs::SumsOf8(hs::VecFromMask(d, hs::Eq(hs::LoadU(d, p + i + w), needle))));
    a2 = hs::Add(a2, hs::SumsOf8(hs::VecFromMask(d, hs::Eq(hs::LoadU(d, p + i + w * 2), needle))));
    a3 = hs::Add(a3, hs::SumsOf8(hs::VecFromMask(d, hs::Eq(hs::LoadU(d, p + i + w * 3), needle))));
  }
  auto acc = hs::Add(hs::Add(a0, a1), hs::Add(a2, a3));
  for (; i + w <= size; i += w)  // whole-vector remainder
    acc = hs::Add(acc, hs::SumsOf8(hs::VecFromMask(d, hs::Eq(hs::LoadU(d, p + i), needle))));
  std::uint64_t byte_sum = hs::ReduceSum(d64, acc);
  std::uint32_t n = static_cast<std::uint32_t>(byte_sum / 255u);
  for (; i < size; ++i) n += (data[i] == c);  // scalar tail (< w bytes)
  return n;
}

// Index of the last `c` in data[0,size), or 0xFFFFFFFF if absent. Backward w-byte blocks from
// the high end; in each, Eq + FindLastTrue gives the highest matching lane. A sub-w head runs
// scalar. (find(char) has memchr; rfind has no portable reverse-memchr, so this is the win.)
std::uint32_t STR_RFIND_FN(const char* data, std::uint32_t size, char c) noexcept {
  constexpr std::uint32_t kNpos = 0xFFFFFFFFu;
  const hs::ScalableTag<std::uint8_t> d;
  const std::size_t w = hs::Lanes(d);
  const auto needle = hs::Set(d, static_cast<std::uint8_t>(c));
  const auto* p = reinterpret_cast<const std::uint8_t*>(data);
  std::size_t i = size;
  while (i >= w) {
    i -= w;
    const auto m = hs::Eq(hs::LoadU(d, p + i), needle);
    const std::ptrdiff_t lane = hs::FindLastTrue(d, m);
    if (lane >= 0) return static_cast<std::uint32_t>(i) + static_cast<std::uint32_t>(lane);
  }
  for (std::uint32_t j = static_cast<std::uint32_t>(i); j-- > 0;)  // scalar head (< w bytes)
    if (data[j] == c) return j;
  return kNpos;
}

}  // namespace hyle::container::detail

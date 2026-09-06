// Shared Container string kernels. HyleDispatch generates one per-ISA TU and
// freezes Highway's static target before including this file. HYLE owns the
// pointer/count behavior; Highway owns register types and lowering.
//
// One implementation source is generated for every supported ISA. Count uses
// SumsOf8 with four independent accumulators; rfind uses the reverse-scan
// FindLastTrue primitive.

#include <cstddef>
#include <cstdint>

#include "hwy/highway.h"

#ifndef HYLE_SIMD_VARIANT_SUFFIX
#error "define HYLE_SIMD_VARIANT_SUFFIX before including str_search_impl.h"
#endif

#define HYLE_STRSEARCH_CAT_(a, b) a##b
#define HYLE_STRSEARCH_CAT(a, b) HYLE_STRSEARCH_CAT_(a, b)
#define STR_COUNT_FN HYLE_STRSEARCH_CAT(str_count_, HYLE_SIMD_VARIANT_SUFFIX)
#define STR_RFIND_FN \
  HYLE_STRSEARCH_CAT(str_rfind_char_, HYLE_SIMD_VARIANT_SUFFIX)
#define STR_LOWER_FN \
  HYLE_STRSEARCH_CAT(str_ascii_lower_, HYLE_SIMD_VARIANT_SUFFIX)
#define STR_UPPER_FN \
  HYLE_STRSEARCH_CAT(str_ascii_upper_, HYLE_SIMD_VARIANT_SUFFIX)

namespace hyle::container::detail {

namespace hn = hwy::HWY_NAMESPACE;

// Count occurrences of `c` in data[0,size). SAD accumulate: Eq marks matches, VecFromMask
// turns them into 0xFF bytes, SumsOf8 horizontally sums each 8-byte group (a 0xFF match = 255)
// into u64 lanes; accumulate across vectors, sum the lanes, divide by 255. FOUR independent
// accumulators so the per-iteration SumsOf8+Add chains do not serialize.
std::uint32_t STR_COUNT_FN(const char* data, std::uint32_t size, char c) noexcept {
  const hn::ScalableTag<std::uint8_t> d;
  const hn::ScalableTag<std::uint64_t> d64;
  const std::size_t w = hn::Lanes(d);
  const auto needle = hn::Set(d, static_cast<std::uint8_t>(c));
  const auto* p = reinterpret_cast<const std::uint8_t*>(data);

  auto a0 = hn::Zero(d64), a1 = hn::Zero(d64), a2 = hn::Zero(d64),
       a3 = hn::Zero(d64);
  std::size_t i = 0;
  const std::size_t w4 = w * 4;
  for (; i + w4 <= size; i += w4) {
    a0 = hn::Add(a0, hn::SumsOf8(hn::VecFromMask(
                         d, hn::Eq(hn::LoadU(d, p + i), needle))));
    a1 = hn::Add(a1, hn::SumsOf8(hn::VecFromMask(
                         d, hn::Eq(hn::LoadU(d, p + i + w), needle))));
    a2 = hn::Add(a2, hn::SumsOf8(hn::VecFromMask(
                         d, hn::Eq(hn::LoadU(d, p + i + w * 2), needle))));
    a3 = hn::Add(a3, hn::SumsOf8(hn::VecFromMask(
                         d, hn::Eq(hn::LoadU(d, p + i + w * 3), needle))));
  }
  auto acc = hn::Add(hn::Add(a0, a1), hn::Add(a2, a3));
  for (; i + w <= size; i += w)  // whole-vector remainder
    acc = hn::Add(acc, hn::SumsOf8(hn::VecFromMask(
                           d, hn::Eq(hn::LoadU(d, p + i), needle))));
  std::uint64_t byte_sum = hn::ReduceSum(d64, acc);
  std::uint32_t n = static_cast<std::uint32_t>(byte_sum / 255u);
  for (; i < size; ++i) n += (data[i] == c);  // scalar tail (< w bytes)
  return n;
}

// Index of the last `c` in data[0,size), or 0xFFFFFFFF if absent. Backward w-byte blocks from
// the high end; in each, Eq + FindLastTrue gives the highest matching lane. A sub-w head runs
// scalar because C++ has no portable reverse-memchr primitive.
std::uint32_t STR_RFIND_FN(const char* data, std::uint32_t size, char c) noexcept {
  constexpr std::uint32_t kNpos = 0xFFFFFFFFu;
  const hn::ScalableTag<std::uint8_t> d;
  const std::size_t w = hn::Lanes(d);
  const auto needle = hn::Set(d, static_cast<std::uint8_t>(c));
  const auto* p = reinterpret_cast<const std::uint8_t*>(data);
  std::size_t i = size;
  while (i >= w) {
    i -= w;
    const auto m = hn::Eq(hn::LoadU(d, p + i), needle);
    const std::ptrdiff_t lane = hn::FindLastTrue(d, m);
    if (lane >= 0) return static_cast<std::uint32_t>(i) + static_cast<std::uint32_t>(lane);
  }
  for (std::uint32_t j = static_cast<std::uint32_t>(i); j-- > 0;)  // scalar head (< w bytes)
    if (data[j] == c) return j;
  return kNpos;
}

std::uint32_t STR_LOWER_FN(const char* data, std::uint32_t size, char* out,
                           std::uint32_t cap) noexcept {
  const std::uint32_t n =
      cap == 0 ? 0 : (size < cap - 1 ? size : cap - 1);
  const hn::ScalableTag<std::uint8_t> d;
  const std::size_t width = hn::Lanes(d);
  const auto lower_a = hn::Set(d, static_cast<std::uint8_t>('A'));
  const auto upper_z = hn::Set(d, static_cast<std::uint8_t>('Z'));
  const auto delta = hn::Set(d, static_cast<std::uint8_t>('a' - 'A'));
  const auto* input = reinterpret_cast<const std::uint8_t*>(data);
  auto* output = reinterpret_cast<std::uint8_t*>(out);
  std::size_t i = 0;
  for (; i + width <= n; i += width) {
    const auto value = hn::LoadU(d, input + i);
    const auto ascii_upper =
        hn::And(hn::Ge(value, lower_a), hn::Le(value, upper_z));
    hn::StoreU(hn::Add(value, hn::IfThenElseZero(ascii_upper, delta)), d,
               output + i);
  }
  for (; i < n; ++i) {
    const char c = data[i];
    out[i] = c >= 'A' && c <= 'Z' ? static_cast<char>(c + ('a' - 'A')) : c;
  }
  if (cap != 0) out[n] = '\0';
  return size;
}

std::uint32_t STR_UPPER_FN(const char* data, std::uint32_t size, char* out,
                           std::uint32_t cap) noexcept {
  const std::uint32_t n =
      cap == 0 ? 0 : (size < cap - 1 ? size : cap - 1);
  const hn::ScalableTag<std::uint8_t> d;
  const std::size_t width = hn::Lanes(d);
  const auto lower_a = hn::Set(d, static_cast<std::uint8_t>('a'));
  const auto lower_z = hn::Set(d, static_cast<std::uint8_t>('z'));
  const auto delta = hn::Set(d, static_cast<std::uint8_t>('a' - 'A'));
  const auto* input = reinterpret_cast<const std::uint8_t*>(data);
  auto* output = reinterpret_cast<std::uint8_t*>(out);
  std::size_t i = 0;
  for (; i + width <= n; i += width) {
    const auto value = hn::LoadU(d, input + i);
    const auto ascii_lower =
        hn::And(hn::Ge(value, lower_a), hn::Le(value, lower_z));
    hn::StoreU(hn::Sub(value, hn::IfThenElseZero(ascii_lower, delta)), d,
               output + i);
  }
  for (; i < n; ++i) {
    const char c = data[i];
    out[i] = c >= 'a' && c <= 'z' ? static_cast<char>(c - ('a' - 'A')) : c;
  }
  if (cap != 0) out[n] = '\0';
  return size;
}

}  // namespace hyle::container::detail

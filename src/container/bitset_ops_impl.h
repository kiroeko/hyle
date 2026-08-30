#include <bit>
#include <cstddef>
#include <cstdint>

#include "hwy/highway.h"

#ifndef HYLE_SIMD_VARIANT_SUFFIX
#error "define HYLE_SIMD_VARIANT_SUFFIX before including bitset_ops_impl.h"
#endif

#define HYLE_BITSET_CAT_(a, b) a##b
#define HYLE_BITSET_CAT(a, b) HYLE_BITSET_CAT_(a, b)
#define BITSET_AND_FN HYLE_BITSET_CAT(bitset_and_, HYLE_SIMD_VARIANT_SUFFIX)
#define BITSET_OR_FN HYLE_BITSET_CAT(bitset_or_, HYLE_SIMD_VARIANT_SUFFIX)
#define BITSET_XOR_FN HYLE_BITSET_CAT(bitset_xor_, HYLE_SIMD_VARIANT_SUFFIX)
#define BITSET_AND_NOT_FN \
  HYLE_BITSET_CAT(bitset_and_not_, HYLE_SIMD_VARIANT_SUFFIX)
#define BITSET_NOT_FN HYLE_BITSET_CAT(bitset_not_, HYLE_SIMD_VARIANT_SUFFIX)
#define BITSET_COUNT_FN HYLE_BITSET_CAT(bitset_count_, HYLE_SIMD_VARIANT_SUFFIX)

namespace hyle::container::detail {
namespace hn = hwy::HWY_NAMESPACE;

void BITSET_AND_FN(std::uint64_t* destination, const std::uint64_t* source,
                   std::size_t count) noexcept {
  const hn::ScalableTag<std::uint64_t> d;
  const std::size_t width = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + width <= count; i += width)
    hn::StoreU(hn::And(hn::LoadU(d, destination + i),
                       hn::LoadU(d, source + i)),
               d, destination + i);
  for (; i < count; ++i) destination[i] &= source[i];
}

void BITSET_OR_FN(std::uint64_t* destination, const std::uint64_t* source,
                  std::size_t count) noexcept {
  const hn::ScalableTag<std::uint64_t> d;
  const std::size_t width = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + width <= count; i += width)
    hn::StoreU(hn::Or(hn::LoadU(d, destination + i),
                      hn::LoadU(d, source + i)),
               d, destination + i);
  for (; i < count; ++i) destination[i] |= source[i];
}

void BITSET_XOR_FN(std::uint64_t* destination, const std::uint64_t* source,
                   std::size_t count) noexcept {
  const hn::ScalableTag<std::uint64_t> d;
  const std::size_t width = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + width <= count; i += width)
    hn::StoreU(hn::Xor(hn::LoadU(d, destination + i),
                       hn::LoadU(d, source + i)),
               d, destination + i);
  for (; i < count; ++i) destination[i] ^= source[i];
}

void BITSET_AND_NOT_FN(std::uint64_t* destination,
                       const std::uint64_t* source,
                       std::size_t count) noexcept {
  const hn::ScalableTag<std::uint64_t> d;
  const std::size_t width = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + width <= count; i += width)
    hn::StoreU(hn::AndNot(hn::LoadU(d, source + i),
                          hn::LoadU(d, destination + i)),
               d, destination + i);
  for (; i < count; ++i) destination[i] &= ~source[i];
}

void BITSET_NOT_FN(std::uint64_t* words, std::size_t count) noexcept {
  const hn::ScalableTag<std::uint64_t> d;
  const std::size_t width = hn::Lanes(d);
  const auto ones = hn::Set(d, ~std::uint64_t{0});
  std::size_t i = 0;
  for (; i + width <= count; i += width)
    hn::StoreU(hn::Xor(hn::LoadU(d, words + i), ones), d, words + i);
  for (; i < count; ++i) words[i] = ~words[i];
}

#if !defined(HYLE_SIMD_TARGET_NEON)
std::size_t BITSET_COUNT_FN(const std::uint64_t* words,
                            std::size_t count) noexcept {
  const hn::ScalableTag<std::uint64_t> d;
  const std::size_t width = hn::Lanes(d);
  auto sum0 = hn::Zero(d);
  auto sum1 = hn::Zero(d);
  auto sum2 = hn::Zero(d);
  auto sum3 = hn::Zero(d);
  std::size_t i = 0;
  const std::size_t width4 = width * 4;
  for (; i + width4 <= count; i += width4) {
    sum0 = hn::Add(sum0, hn::PopulationCount(hn::LoadU(d, words + i)));
    sum1 = hn::Add(sum1,
                   hn::PopulationCount(hn::LoadU(d, words + i + width)));
    sum2 = hn::Add(sum2,
                   hn::PopulationCount(hn::LoadU(d, words + i + width * 2)));
    sum3 = hn::Add(sum3,
                   hn::PopulationCount(hn::LoadU(d, words + i + width * 3)));
  }
  auto sum = hn::Add(hn::Add(sum0, sum1), hn::Add(sum2, sum3));
  for (; i + width <= count; i += width)
    sum = hn::Add(sum, hn::PopulationCount(hn::LoadU(d, words + i)));
  std::uint64_t result = hn::ReduceSum(d, sum);
  for (; i < count; ++i) result += std::popcount(words[i]);
  return static_cast<std::size_t>(result);
}
#endif

}  // namespace hyle::container::detail

#undef BITSET_AND_FN
#undef BITSET_OR_FN
#undef BITSET_XOR_FN
#undef BITSET_AND_NOT_FN
#undef BITSET_NOT_FN
#undef BITSET_COUNT_FN
#undef HYLE_BITSET_CAT
#undef HYLE_BITSET_CAT_

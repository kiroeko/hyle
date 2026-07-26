#include <bit>
#include <cstddef>
#include <cstdint>

#include "hyle/simd/vec.h"

#ifndef HYLE_BITSET_SUFFIX
#error "define HYLE_BITSET_SUFFIX before including bitset_ops_impl.h"
#endif

#define HYLE_BITSET_CAT_(a, b) a##b
#define HYLE_BITSET_CAT(a, b) HYLE_BITSET_CAT_(a, b)
#define BITSET_AND_FN HYLE_BITSET_CAT(bitset_and_, HYLE_BITSET_SUFFIX)
#define BITSET_OR_FN HYLE_BITSET_CAT(bitset_or_, HYLE_BITSET_SUFFIX)
#define BITSET_XOR_FN HYLE_BITSET_CAT(bitset_xor_, HYLE_BITSET_SUFFIX)
#define BITSET_AND_NOT_FN HYLE_BITSET_CAT(bitset_and_not_, HYLE_BITSET_SUFFIX)
#define BITSET_NOT_FN HYLE_BITSET_CAT(bitset_not_, HYLE_BITSET_SUFFIX)
#define BITSET_COUNT_FN HYLE_BITSET_CAT(bitset_count_, HYLE_BITSET_SUFFIX)

namespace hyle::container::detail {
namespace hs = ::hyle::simd;

void BITSET_AND_FN(std::uint64_t* destination, const std::uint64_t* source,
                   std::size_t count) noexcept {
  const hs::ScalableTag<std::uint64_t> d;
  const std::size_t width = hs::Lanes(d);
  std::size_t i = 0;
  for (; i + width <= count; i += width)
    hs::StoreU(hs::And(hs::LoadU(d, destination + i),
                       hs::LoadU(d, source + i)),
               d, destination + i);
  for (; i < count; ++i) destination[i] &= source[i];
}

void BITSET_OR_FN(std::uint64_t* destination, const std::uint64_t* source,
                  std::size_t count) noexcept {
  const hs::ScalableTag<std::uint64_t> d;
  const std::size_t width = hs::Lanes(d);
  std::size_t i = 0;
  for (; i + width <= count; i += width)
    hs::StoreU(hs::Or(hs::LoadU(d, destination + i),
                      hs::LoadU(d, source + i)),
               d, destination + i);
  for (; i < count; ++i) destination[i] |= source[i];
}

void BITSET_XOR_FN(std::uint64_t* destination, const std::uint64_t* source,
                   std::size_t count) noexcept {
  const hs::ScalableTag<std::uint64_t> d;
  const std::size_t width = hs::Lanes(d);
  std::size_t i = 0;
  for (; i + width <= count; i += width)
    hs::StoreU(hs::Xor(hs::LoadU(d, destination + i),
                       hs::LoadU(d, source + i)),
               d, destination + i);
  for (; i < count; ++i) destination[i] ^= source[i];
}

void BITSET_AND_NOT_FN(std::uint64_t* destination,
                       const std::uint64_t* source,
                       std::size_t count) noexcept {
  const hs::ScalableTag<std::uint64_t> d;
  const std::size_t width = hs::Lanes(d);
  std::size_t i = 0;
  for (; i + width <= count; i += width)
    hs::StoreU(hs::AndNot(hs::LoadU(d, source + i),
                          hs::LoadU(d, destination + i)),
               d, destination + i);
  for (; i < count; ++i) destination[i] &= ~source[i];
}

void BITSET_NOT_FN(std::uint64_t* words, std::size_t count) noexcept {
  const hs::ScalableTag<std::uint64_t> d;
  const std::size_t width = hs::Lanes(d);
  const auto ones = hs::Set(d, ~std::uint64_t{0});
  std::size_t i = 0;
  for (; i + width <= count; i += width)
    hs::StoreU(hs::Xor(hs::LoadU(d, words + i), ones), d, words + i);
  for (; i < count; ++i) words[i] = ~words[i];
}

std::size_t BITSET_COUNT_FN(const std::uint64_t* words,
                            std::size_t count) noexcept {
  const hs::ScalableTag<std::uint64_t> d;
  const std::size_t width = hs::Lanes(d);
  auto sum0 = hs::Zero(d);
  auto sum1 = hs::Zero(d);
  auto sum2 = hs::Zero(d);
  auto sum3 = hs::Zero(d);
  std::size_t i = 0;
  const std::size_t width4 = width * 4;
  for (; i + width4 <= count; i += width4) {
    sum0 = hs::Add(sum0, hs::PopulationCount(hs::LoadU(d, words + i)));
    sum1 = hs::Add(sum1,
                   hs::PopulationCount(hs::LoadU(d, words + i + width)));
    sum2 = hs::Add(sum2,
                   hs::PopulationCount(hs::LoadU(d, words + i + width * 2)));
    sum3 = hs::Add(sum3,
                   hs::PopulationCount(hs::LoadU(d, words + i + width * 3)));
  }
  auto sum = hs::Add(hs::Add(sum0, sum1), hs::Add(sum2, sum3));
  for (; i + width <= count; i += width)
    sum = hs::Add(sum, hs::PopulationCount(hs::LoadU(d, words + i)));
  std::uint64_t result = hs::ReduceSum(d, sum);
  for (; i < count; ++i) result += std::popcount(words[i]);
  return static_cast<std::size_t>(result);
}

}  // namespace hyle::container::detail

#undef BITSET_AND_FN
#undef BITSET_OR_FN
#undef BITSET_XOR_FN
#undef BITSET_AND_NOT_FN
#undef BITSET_NOT_FN
#undef BITSET_COUNT_FN
#undef HYLE_BITSET_CAT
#undef HYLE_BITSET_CAT_

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "hyle/simd/narrow_ops.h"
#include "src/simd/runtime_internal.h"

namespace hs = hyle::simd;

namespace {

template <typename T, typename FromFloat>
std::array<T, 9> sample(FromFloat from_float, std::uint16_t nan_bits) {
  return {T::from_bits(nan_bits), from_float(-2.0f),      T::from_bits(0x8000u),
          T::from_bits(0x0000u),  from_float(3.0f),       from_float(2.0f),
          from_float(2.0f),       T::from_bits(nan_bits), from_float(1.0f)};
}

template <typename T, typename FromFloat, typename NarrowTable>
void check_table(const NarrowTable &table, FromFloat from_float,
                  std::uint16_t nan_bits, hs::KernelTarget isa) {
  const auto values = sample<T>(from_float, nan_bits);
  const T plus_zero = T::from_bits(0x0000u);
  const T nan = T::from_bits(nan_bits);
  EXPECT_EQ(table.count_equal(values.data(), values.size(), plus_zero), 2u);
  EXPECT_EQ(table.find_equal(values.data(), values.size(), plus_zero), 2u);
  EXPECT_EQ(table.count_equal_bits(values.data(), values.size(), plus_zero),
            1u);
  EXPECT_EQ(table.find_equal_bits(values.data(), values.size(), plus_zero), 3u);
  EXPECT_EQ(table.count_equal(values.data(), values.size(), nan), 0u);
  EXPECT_EQ(table.find_equal(values.data(), values.size(), nan), values.size());
  EXPECT_EQ(table.count_equal_bits(values.data(), values.size(), nan), 2u);
  EXPECT_EQ(table.find_equal_bits(values.data(), values.size(), nan), 0u);
  const float sum = table.sum(values.data(), values.size());
  EXPECT_TRUE(std::isnan(sum))
      << "isa=" << static_cast<int>(isa)
      << " type=" << (std::is_same_v<T, hs::float16_t> ? "f16" : "bf16")
      << " sum=" << sum;
  for (std::size_t i = 0; i < values.size(); ++i) {
    const float singleton = table.sum(values.data() + i, 1);
    const float expected = std::is_same_v<T, hs::float16_t>
                               ? hs::f16_to_f32(
                                     hs::float16_t::from_bits(values[i].bits))
                               : hs::bf16_to_f32(
                                     hs::bfloat16_t::from_bits(values[i].bits));
    if (std::isnan(expected))
      EXPECT_TRUE(std::isnan(singleton)) << "isa=" << static_cast<int>(isa)
                                         << " singleton=" << i;
    else
      EXPECT_FLOAT_EQ(singleton, expected) << "isa=" << static_cast<int>(isa)
                                            << " singleton=" << i;
  }
  for (std::size_t prefix = 1; prefix <= values.size(); ++prefix) {
    float expected = 0.0f;
    for (std::size_t i = 0; i < prefix; ++i) {
      expected += std::is_same_v<T, hs::float16_t>
                      ? hs::f16_to_f32(
                            hs::float16_t::from_bits(values[i].bits))
                      : hs::bf16_to_f32(
                            hs::bfloat16_t::from_bits(values[i].bits));
    }
    const float actual = table.sum(values.data(), prefix);
    if (std::isnan(expected)) {
      EXPECT_TRUE(std::isnan(actual)) << "isa=" << static_cast<int>(isa)
                                      << " prefix=" << prefix;
    }
  }

  auto ordered = values;
  std::sort(ordered.begin(), ordered.end(), [](T a, T b) {
    return hs::narrow_total_order_key(a) < hs::narrow_total_order_key(b);
  });
  EXPECT_TRUE(table.sorted_asc(ordered.data(), ordered.size()));
  EXPECT_FALSE(table.sorted_desc(ordered.data(), ordered.size()));
  std::reverse(ordered.begin(), ordered.end());
  EXPECT_TRUE(table.sorted_desc(ordered.data(), ordered.size()));
}

struct F16Table {
  const hs::Runtime::Narrow &n;
  std::size_t count_equal(const hs::float16_t *p, std::size_t size,
                          hs::float16_t v) const {
    return n.count_equal_f16(p, size, v);
  }
  std::size_t find_equal(const hs::float16_t *p, std::size_t size,
                         hs::float16_t v) const {
    return n.find_equal_f16(p, size, v);
  }
  std::size_t count_equal_bits(const hs::float16_t *p, std::size_t size,
                               hs::float16_t v) const {
    return n.count_equal_bits_f16(p, size, v);
  }
  std::size_t find_equal_bits(const hs::float16_t *p, std::size_t size,
                              hs::float16_t v) const {
    return n.find_equal_bits_f16(p, size, v);
  }
  float sum(const hs::float16_t *p, std::size_t size) const {
    return n.sum_f16(p, size);
  }
  bool sorted_asc(const hs::float16_t *p, std::size_t size) const {
    return n.is_sorted_total_asc_f16(p, size);
  }
  bool sorted_desc(const hs::float16_t *p, std::size_t size) const {
    return n.is_sorted_total_desc_f16(p, size);
  }
};

struct BF16Table {
  const hs::Runtime::Narrow &n;
  std::size_t count_equal(const hs::bfloat16_t *p, std::size_t size,
                          hs::bfloat16_t v) const {
    return n.count_equal_bf16(p, size, v);
  }
  std::size_t find_equal(const hs::bfloat16_t *p, std::size_t size,
                         hs::bfloat16_t v) const {
    return n.find_equal_bf16(p, size, v);
  }
  std::size_t count_equal_bits(const hs::bfloat16_t *p, std::size_t size,
                               hs::bfloat16_t v) const {
    return n.count_equal_bits_bf16(p, size, v);
  }
  std::size_t find_equal_bits(const hs::bfloat16_t *p, std::size_t size,
                              hs::bfloat16_t v) const {
    return n.find_equal_bits_bf16(p, size, v);
  }
  float sum(const hs::bfloat16_t *p, std::size_t size) const {
    return n.sum_bf16(p, size);
  }
  bool sorted_asc(const hs::bfloat16_t *p, std::size_t size) const {
    return n.is_sorted_total_asc_bf16(p, size);
  }
  bool sorted_desc(const hs::bfloat16_t *p, std::size_t size) const {
    return n.is_sorted_total_desc_bf16(p, size);
  }
};

} // namespace

TEST(NarrowOps, TotalOrderKeyIsBijectiveForEveryEncoding) {
  for (std::uint32_t bits = 0; bits < 65536u; ++bits) {
    const auto raw = static_cast<std::uint16_t>(bits);
    const auto f16 = hs::float16_t::from_bits(raw);
    const auto bf16 = hs::bfloat16_t::from_bits(raw);
    EXPECT_EQ(hs::narrow_from_total_order_key<hs::float16_t>(
                  hs::narrow_total_order_key(f16))
                  .bits,
              raw);
    EXPECT_EQ(hs::narrow_from_total_order_key<hs::bfloat16_t>(
                  hs::narrow_total_order_key(bf16))
                  .bits,
              raw);
  }
}

TEST(NarrowOps, PublicDispatchedSurface) {
  const auto f16 = sample<hs::float16_t>(hs::f32_to_f16, 0x7E11u);
  EXPECT_EQ(hs::count_equal(f16.data(), f16.size(), hs::f32_to_f16(2.0f)), 2u);
  EXPECT_EQ(hs::find_equal_bits(f16.data(), f16.size(),
                                hs::float16_t::from_bits(0x7E11u)),
            0u);
  EXPECT_TRUE(std::isnan(hs::sum(f16.data(), f16.size())));
  EXPECT_EQ(hs::count_equal(static_cast<const hs::float16_t *>(nullptr), 0,
                            hs::float16_t::from_bits(0)),
            0u);
  EXPECT_EQ(hs::sum(static_cast<const hs::bfloat16_t *>(nullptr), 0), 0.0f);
  EXPECT_TRUE(
      hs::is_sorted_total(static_cast<const hs::bfloat16_t *>(nullptr), 0));
}

TEST(NarrowOps, EverySupportedRuntimeVariant) {
  for (const hs::KernelTarget isa :
       {hs::KernelTarget::kScalar, hs::KernelTarget::kSse4, hs::KernelTarget::kAvx2, hs::KernelTarget::kAvx512,
        hs::KernelTarget::kNeon, hs::KernelTarget::kSve, hs::KernelTarget::kRvv}) {
    const hs::Runtime *rt = hs::runtime_for(isa);
    if (rt == nullptr)
      continue;
    check_table<hs::float16_t>(F16Table{rt->narrow}, hs::f32_to_f16, 0x7E11u,
                               isa);
    check_table<hs::bfloat16_t>(BF16Table{rt->narrow}, hs::f32_to_bf16,
                                 0x7FC1u, isa);
  }
}

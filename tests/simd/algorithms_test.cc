#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "hyle/simd/algorithms.h"

namespace hs = hyle::simd;

TEST(AlgorithmsFacade, OpsRandomAndScan) {
  std::vector<float> x{1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> y{5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> out(4);
  hs::add(x, y, out);
  EXPECT_EQ(out, (std::vector<float>{6.0f, 8.0f, 10.0f, 12.0f}));
  EXPECT_FLOAT_EQ(hs::dot(x, y), 70.0f);
  hs::saxpy(2.0f, x, y);
  EXPECT_EQ(y, (std::vector<float>{7.0f, 10.0f, 13.0f, 16.0f}));

  hs::RandomState a{12345}, b{12345};
  std::vector<std::uint64_t> ra(19), rb(19);
  hs::random_fill(a, ra);
  hs::random_fill(b, rb);
  EXPECT_EQ(ra, rb);

  std::vector<std::int32_t> scan{1, 2, 3, 4};
  hs::inclusive_scan(scan);
  EXPECT_EQ(scan, (std::vector<std::int32_t>{1, 3, 6, 10}));
  hs::exclusive_scan(scan);
  EXPECT_EQ(scan, (std::vector<std::int32_t>{0, 1, 4, 10}));
}

TEST(AlgorithmsFacade, DtypeMathAndSort) {
  std::vector<float> src{0.0f, 0.5f, 1.0f, 2.0f};
  std::vector<hs::float16_t> half(src.size());
  std::vector<float> roundtrip(src.size());
  hs::convert(src, half);
  hs::convert(half, roundtrip);
  for (std::size_t i = 0; i < src.size(); ++i) EXPECT_FLOAT_EQ(roundtrip[i], src[i]);

  std::vector<float> exp_out(src.size());
  hs::exp(src, exp_out);
  for (std::size_t i = 0; i < src.size(); ++i)
    EXPECT_NEAR(exp_out[i], std::exp(src[i]), 2e-5f);

  std::vector<std::int32_t> keys{9, -2, 7, 7, 1};
  hs::sort(keys);
  EXPECT_TRUE(std::is_sorted(keys.begin(), keys.end()));
  std::vector<std::uint32_t> arg_keys{30, 10, 20};
  std::vector<std::uint32_t> indices(arg_keys.size());
  hs::argsort(arg_keys, indices);
  EXPECT_EQ(indices, (std::vector<std::uint32_t>{1, 2, 0}));
}

TEST(AlgorithmsFacade, BitpackSizesAndRoundTrip) {
  std::vector<std::uint16_t> raw{0, 1, 3, 7, 15, 17, 31, 5, 9};
  constexpr unsigned bits = 5;
  std::vector<std::uint16_t> packed(hs::packed_size<std::uint16_t>(raw.size(), bits));
  std::vector<std::uint16_t> out(raw.size());
  hs::pack(raw, packed, bits);
  hs::unpack(packed, out, bits);
  EXPECT_EQ(out, raw);
}

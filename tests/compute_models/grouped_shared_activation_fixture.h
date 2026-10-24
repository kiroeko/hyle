#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace grouped_shared_test {

// Independent logical data, shared by the public backend consumers. Test K is
// a positive multiple of32. Sparse +/-127 inputs survive the original native
// Q8 activation conversion exactly; the small weights/products/sums are F32
// dyadics. This makes exact comparisons useful without claiming general native
// reduction-order or cross-backend bit equivalence.
struct Fixture final {
  std::int64_t tokens;
  std::int64_t slots;
  std::int64_t columns;
  std::int64_t rows;
  std::int64_t experts;
  std::vector<float> weights;

  Fixture(std::int64_t t, std::int64_t s, std::int64_t k = 64,
          std::int64_t n = 5, std::int64_t e = 3)
      : tokens(t), slots(s), columns(k), rows(n), experts(e),
        weights(static_cast<std::size_t>(e * n * k)) {
    for (std::int64_t expert = 0; expert < e; ++expert)
      for (std::int64_t row = 0; row < n; ++row)
        for (std::int64_t column = 0; column < k; ++column)
          weights[static_cast<std::size_t>((expert * n + row) * k + column)] =
              static_cast<float>((expert * 11 + row * 3 + column * 5) % 17 - 8) * 0.25F;
  }

  std::vector<float> activations(bool shared, int pass = 0) const {
    const auto a = shared ? std::int64_t{1} : slots;
    std::vector<float> result(static_cast<std::size_t>(tokens * a * columns));
    for (std::int64_t token = 0; token < tokens; ++token)
      for (std::int64_t slot = 0; slot < a; ++slot)
        for (std::int64_t block = 0; block < columns / 32; ++block) {
          const auto column = block * 32 + (token * 3 + slot * 5 + pass * 7 + block) % 32;
          result[static_cast<std::size_t>((token * a + slot) * columns + column)] =
              ((token + slot + pass + block) & 1) != 0 ? -127.0F : 127.0F;
        }
    return result;
  }

  std::vector<std::int32_t> ids(bool distinct, int round = 0) const {
    std::vector<std::int32_t> result(static_cast<std::size_t>(tokens * slots));
    for (std::int64_t token = 0; token < tokens; ++token)
      for (std::int64_t slot = 0; slot < slots; ++slot)
        result[static_cast<std::size_t>(token * slots + slot)] =
            static_cast<std::int32_t>((token + round + (distinct ? slot : 0)) % experts);
    return result;
  }

  std::vector<float> expected(bool shared, std::span<const float> input,
                              std::span<const std::int32_t> indices) const {
    const auto a = shared ? std::int64_t{1} : slots;
    std::vector<float> result(static_cast<std::size_t>(tokens * slots * rows));
    for (std::int64_t token = 0; token < tokens; ++token)
      for (std::int64_t slot = 0; slot < slots; ++slot)
        for (std::int64_t row = 0; row < rows; ++row) {
          const auto expert = indices[static_cast<std::size_t>(token * slots + slot)];
          double sum = 0;
          for (std::int64_t k = 0; k < columns; ++k)
            sum += static_cast<double>(input[static_cast<std::size_t>(
                       (token * a + (shared ? 0 : slot)) * columns + k)]) *
                weights[static_cast<std::size_t>((expert * rows + row) * columns + k)];
          result[static_cast<std::size_t>((token * slots + slot) * rows + row)] =
              static_cast<float>(sum);
        }
    return result;
  }
};

}  // namespace grouped_shared_test

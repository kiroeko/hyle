#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "hyle/compute/model_ops.h"

namespace sparse_fragment_test {
namespace c = hyle::compute;

struct Fixture final {
  c::Operation operation = c::model_ops::lightning_indexer();
  std::array<c::TensorDesc, 4> inputs;
  std::array<std::vector<std::byte>, 4> payloads;
  c::TensorDesc output;
  std::array<float, 8> expected{0, 0, 1, -2, 0.25F, 0.5F, 1, 0.25F};

  Fixture() {
    inputs = {
        c::TensorDesc::dense(c::DataType::kFloat32,
                             std::array<std::int64_t, 4>{1, 2, 64, 128})
            .value(),
        c::TensorDesc::dense(c::DataType::kFloat32,
                             std::array<std::int64_t, 4>{1, 4, 1, 128})
            .value(),
        c::TensorDesc::dense(c::DataType::kFloat32,
                             std::array<std::int64_t, 4>{1, 1, 2, 64})
            .value(),
        c::TensorDesc::dense(c::DataType::kFloat16,
                             std::array<std::int64_t, 4>{1, 1, 2, 4})
            .value()};
    output = c::infer_single_output(operation, inputs).value();

    std::vector<float> query(2 * 64 * 128);
    for (std::size_t token = 0; token < 2; ++token)
      for (std::size_t i = token * 64 * 128;
           i < (token + 1) * 64 * 128; ++i)
        query[i] = token == 0 ? 1.0F : -1.0F;
    std::vector<float> key(4 * 128);
    constexpr std::array scales{1.0F / 128, -1.0F / 128, 0.5F / 128,
                                -0.25F / 128};
    for (std::size_t row = 0; row < scales.size(); ++row)
      for (std::size_t i = 0; i < 128; ++i)
        key[row * 128 + i] = scales[row];
    std::vector<float> weights(2 * 64, 1.0F / 64);
    // Exact IEEE F16 encodings: -1, 0, 0.5, -2, 0.25, -0.5, 1, 0.
    const std::array<std::uint16_t, 8> mask{0xbc00, 0x0000, 0x3800, 0xc000,
                                            0x3400, 0xb800, 0x3c00, 0x0000};
    payloads = {bytes<float>(query), bytes<float>(key), bytes<float>(weights),
                bytes<std::uint16_t>(mask)};
  }

private:
  template <class T>
  static std::vector<std::byte> bytes(std::span<const T> values) {
    std::vector<std::byte> result(values.size_bytes());
    std::memcpy(result.data(), values.data(), result.size());
    return result;
  }
};
} // namespace sparse_fragment_test

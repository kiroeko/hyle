#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <vector>

#include "hyle/compute/operation.h"

namespace gdn_snapshot_fixture {
namespace c = hyle::compute;

// Unequal cyclic heads and two distinct batches expose both native snapshot
// slot order and accidental slot/batch interchange. No provider is the oracle.
struct Fixture final {
  std::int64_t batches, tokens, key_heads, value_heads, depth;
  std::array<c::TensorDesc, 6> inputs;
  std::array<std::vector<float>, 6> data;

  explicit Fixture(int seed = 0, std::int64_t batches = 2,
                   std::int64_t tokens = 5, std::int64_t key_heads = 2,
                   std::int64_t value_heads = 6, std::int64_t depth = 32)
      : batches(batches), tokens(tokens), key_heads(key_heads),
        value_heads(value_heads), depth(depth) {
    const auto dense = [](std::initializer_list<std::int64_t> shape) {
      return c::TensorDesc::dense(c::DataType::kFloat32,
                                  {shape.begin(), shape.size()}).value();
    };
    inputs = {dense({batches, tokens, key_heads, depth}),
              dense({batches, tokens, key_heads, depth}),
              dense({batches, tokens, value_heads, depth}),
              dense({batches, tokens, value_heads, 1}),
              dense({batches, tokens, value_heads, 1}),
              dense({batches, value_heads, depth, depth})};
    for (std::size_t port = 0; port < data.size(); ++port) {
      data[port].resize(inputs[port].element_count().value());
      for (std::size_t i = 0; i < data[port].size(); ++i) {
        const auto wave = static_cast<int>((i * 7 + port * 11 + seed) % 31) - 15;
        data[port][i] = port == 3 ? -0.015F * float(1 + i % 7)
                       : port == 4 ? 0.1F + 0.02F * float(i % 5)
                       : float(wave) / (port == 5 ? 1024.F : 128.F);
      }
    }
  }

  std::array<c::TensorDesc, 2> outputs(std::uint32_t snapshots) const {
    if (snapshots == 1)
      return {inputs[2], inputs[5]};
    const std::array shape{batches, static_cast<std::int64_t>(snapshots),
                           value_heads, depth, depth};
    return {inputs[2], c::TensorDesc::dense(c::DataType::kFloat32, shape).value()};
  }

  std::array<std::vector<double>, 2> oracle(std::uint32_t snapshots) const {
    const auto state_width = value_heads * depth * depth;
    std::array<std::vector<double>, 2> result{
        std::vector<double>(data[2].size()),
        std::vector<double>(static_cast<std::size_t>(batches * snapshots *
                                                     state_width))};
    std::vector<double> state(data[5].begin(), data[5].end());
    for (std::int64_t batch = 0; batch < batches; ++batch) {
      for (std::int64_t token = 0; token < tokens; ++token) {
        for (std::int64_t head = 0; head < value_heads; ++head) {
          const auto qk = ((batch * tokens + token) * key_heads +
                           head % key_heads) * depth;
          const auto value = ((batch * tokens + token) * value_heads + head) * depth;
          const auto gate = (batch * tokens + token) * value_heads + head;
          const auto matrix = (batch * value_heads + head) * depth * depth;
          const auto decay = std::exp(double(data[3][gate]));
          for (std::int64_t row = 0; row < depth; ++row) {
            const auto offset = matrix + row * depth;
            double projected = 0;
            for (std::int64_t key = 0; key < depth; ++key) {
              state[offset + key] *= decay;
              projected += state[offset + key] * data[1][qk + key];
            }
            const auto delta = data[4][gate] * (data[2][value + row] - projected);
            double score = 0;
            for (std::int64_t key = 0; key < depth; ++key) {
              state[offset + key] += delta * data[1][qk + key];
              score += state[offset + key] * data[0][qk + key];
            }
            result[0][value + row] = score / std::sqrt(double(depth));
          }
        }
        if (token >= tokens - snapshots) {
          const auto slot = token - (tokens - snapshots);
          std::copy_n(state.begin() + batch * state_width, state_width,
                      result[1].begin() + (batch * snapshots + slot) * state_width);
        }
      }
    }
    return result;
  }
};

struct RecurrentFixture final {
  Fixture activated{3, 1, 3, 16, 16, 128};
  std::array<c::TensorDesc, 8> inputs;
  std::array<std::vector<float>, 8> data;

  RecurrentFixture() {
    const std::array<std::int64_t, 4> shape{1, 1, 16, 1};
    const auto parameter = c::TensorDesc::dense(c::DataType::kFloat32, shape).value();
    inputs = {activated.inputs[0], activated.inputs[1], activated.inputs[2],
              activated.inputs[3], activated.inputs[4], parameter, parameter,
              activated.inputs[5]};
    data[0] = activated.data[0];
    data[1] = activated.data[1];
    data[2] = activated.data[2];
    data[3].resize(48);
    data[4].resize(48);
    data[5].resize(16);
    data[6].resize(16);
    data[7] = activated.data[5];
    for (std::size_t head = 0; head < 16; ++head) {
      data[5][head] = -0.025F * float(head + 1);
      data[6][head] = 0.01F * float(head);
    }
    for (std::size_t i = 0; i < 48; ++i) {
      data[3][i] = float(int(i % 11) - 5) / 16.F;
      data[4][i] = float(int(i % 7) - 3) / 8.F;
      activated.data[3][i] = float(
          std::log1p(std::exp(double(data[3][i]) + data[6][i % 16])) *
          data[5][i % 16]);
      activated.data[4][i] = float(1.0 / (1.0 + std::exp(-double(data[4][i]))));
    }
  }
};
} // namespace gdn_snapshot_fixture

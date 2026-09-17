#pragma once

#include "tensor_test_support.h"

namespace apple_tensor_test {
// Independent scalar updates in logical row-major order. No sorting or MLX
// arithmetic is used by the oracle; each addition rounds to storage.
inline std::vector<double> scatter_reference(std::vector<double> state,
                                             std::span<const double> updates,
                                             std::span<const std::int64_t> ids,
                                             const mx::Shape &shape, int axis,
                                             mx::Dtype type) {
  if (state.empty() || ids.empty())
    return state;
  if (axis < 0)
    axis += static_cast<int>(shape.size());
  std::size_t inner = 1;
  for (std::size_t i = axis + 1; i < shape.size(); ++i)
    inner *= shape[i];
  const std::size_t extent = shape[axis];
  for (std::size_t outer = 0; outer < state.size() / extent / inner; ++outer)
    for (std::size_t index = 0; index < ids.size(); ++index)
      for (std::size_t lane = 0; lane < inner; ++lane) {
        const auto to = (outer * extent + ids[index]) * inner + lane;
        const auto from = (outer * ids.size() + index) * inner + lane;
        const float sum =
            static_cast<float>(state[to]) + static_cast<float>(updates[from]);
        state[to] = rounded(sum, type);
      }
  return state;
}
inline void expect_ordered(const mx::array &actual,
                           std::span<const double> expected) {
  const auto observed = values(actual);
  ASSERT_EQ(observed.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (std::isnan(expected[i]))
      EXPECT_TRUE(std::isnan(observed[i])) << i;
    else {
      EXPECT_EQ(observed[i], expected[i]) << i;
      if (expected[i] == 0)
        EXPECT_EQ(std::signbit(observed[i]), std::signbit(expected[i])) << i;
    }
  }
}
} // namespace apple_tensor_test

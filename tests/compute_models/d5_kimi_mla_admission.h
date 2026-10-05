#pragma once

#include <array>
#include <cstdint>
#include <initializer_list>

#include <gtest/gtest.h>

#include "hyle/compute/operation.h"

namespace d5_kimi_mla_admission {
namespace c = hyle::compute;

inline c::TensorDesc dense(std::initializer_list<std::int64_t> shape,
                           c::DataType type = c::DataType::kFloat32) {
  return c::TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
}

template <class Accept> void check(Accept accept, bool general_value_depth = false) {
  const c::Operation operation(c::ScaledDotProductAttention{
      .scale = 0.072168782F, .causal = true,
      .accumulation_type = c::DataType::kFloat32});
  const std::array inputs{dense({1, 1, 96, 576}), dense({1, 4, 1, 576}),
                          dense({1, 4, 1, 512})};
  const std::array outputs{dense({1, 1, 96, 512})};
  EXPECT_TRUE(accept(operation, inputs, outputs));
  // GLM DSA has 64 query heads with the same compressed latent and RoPE
  // dimensions. It consumes the same original native attention program.
  auto glm_inputs = inputs;
  glm_inputs[0] = dense({1, 1, 64, 576});
  const std::array glm_outputs{dense({1, 1, 64, 512})};
  EXPECT_TRUE(accept(operation, glm_inputs, glm_outputs));
  auto bad_glm = glm_inputs;
  bad_glm[2] = dense({1, 3, 1, 512});
  EXPECT_FALSE(accept(operation, bad_glm, glm_outputs));
  EXPECT_FALSE(accept(operation, glm_inputs, outputs));
  for (const auto &bad_value : {dense({1, 3, 1, 512}),
                                 dense({1, 4, 2, 512}),
                                 dense({2, 4, 1, 512})}) {
    auto changed = inputs;
    changed[2] = bad_value;
    EXPECT_FALSE(accept(operation, changed, outputs));
  }
  auto changed = inputs;
  changed[1] = dense({1, 4, 1, 575});
  EXPECT_FALSE(accept(operation, changed, outputs));
  EXPECT_FALSE(accept(operation, inputs, std::array{inputs[0]}));
  // CPU's general SDPA intake also qualifies independent value depth; other
  // routes retain their existing narrow admission until independently tested.
  changed = inputs;
  changed[0] = dense({1, 1, 48, 576});
  EXPECT_EQ(accept(operation, changed, std::array{dense({1, 1, 48, 512})}), general_value_depth);
  for (std::size_t i = 0; i < changed.size(); ++i)
    changed[i] = c::TensorDesc::dense(c::DataType::kBFloat16,
                                      inputs[i].dimensions()).value();
  EXPECT_EQ(accept(operation, changed,
                      std::array{dense({1, 1, 96, 512}, c::DataType::kBFloat16)}), general_value_depth);
}
} // namespace d5_kimi_mla_admission

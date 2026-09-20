#pragma once

#include <hyle/compute/apple.h>

#include <array>
#include <cmath>
#include <vector>

namespace sample_apple_recurrent {
namespace c = hyle::compute;
namespace a = c::apple;
namespace mx = mlx::core;

// A small, nonzero-state consumer of the original ggml Metal region. The
// constant input has a closed-form oracle independent of backend code.
inline std::array<mx::array, 6> inputs() {
  const std::vector<float> q(32, 1.0f / 8), v(32, 1.0f / 4);
  const std::vector<float> state(32 * 32, 1.0f / 64);
  const float decay = 0, beta = 0.5f;
  return {mx::array(q.data(), {1, 1, 1, 32}),
          mx::array(q.data(), {1, 1, 1, 32}),
          mx::array(v.data(), {1, 1, 1, 32}),
          mx::array(&decay, {1, 1, 1, 1}),
          mx::array(&beta, {1, 1, 1, 1}),
          mx::array(state.data(), {1, 1, 32, 32})};
}

inline hyle::Status check(const a::Results &result, bool composed) {
  const auto arrays = result.arrays();
  if (arrays.size() != 2 || arrays[0].shape() != mx::Shape{1, 1, 1, 32} ||
      arrays[1].shape() != mx::Shape{1, 1, 32, 32})
    return hyle::Status::Internal("Apple recurrent output shapes differ");
  const double score = (7.0 / 64) / std::sqrt(32.0);
  const double expected = composed ? 0.25 + 2 * score : score;
  for (std::size_t i = 0; i < 32; ++i) {
    if (std::abs(arrays[0].data<float>()[i] - expected) > 2e-6)
      return hyle::Status::Internal("Apple recurrent score oracle mismatch");
  }
  for (std::size_t i = 0; i < 32 * 32; ++i) {
    if (std::abs(arrays[1].data<float>()[i] - 7.0 / 256) > 2e-6)
      return hyle::Status::Internal("Apple recurrent state oracle mismatch");
  }
  return hyle::Status::Ok();
}

inline hyle::Status replay(const a::Program &program, bool composed) {
  const auto values = inputs();
  HYLE_ASSIGN_OR_RETURN(auto invocation, program.create_invocation());
  HYLE_RETURN_IF_ERROR(invocation.bind(values));
  for (int repeat = 0; repeat < 3; ++repeat) {
    HYLE_ASSIGN_OR_RETURN(auto work, invocation.submit());
    HYLE_ASSIGN_OR_RETURN(auto result, work.wait());
    HYLE_RETURN_IF_ERROR(check(*result.get(), composed));
  }
  for (std::size_t i = 0; i < 32 * 32; ++i) {
    if (values[5].data<float>()[i] != 1.0f / 64)
      return hyle::Status::Internal("Apple recurrent modified caller state");
  }
  return hyle::Status::Ok();
}

inline hyle::Status direct(const a::Context &context) {
  const auto examples = inputs();
  HYLE_ASSIGN_OR_RETURN(auto program, context.prepare(c::GatedDeltaNet{}, examples));
  return replay(program, false);
}
} // namespace sample_apple_recurrent

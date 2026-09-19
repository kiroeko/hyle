#pragma once

#include "workload.h"

#include <hyle/compute/apple.h>
#include <mlx/fast.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

namespace sample_apple_direct {
namespace c = hyle::compute;
namespace a = c::apple;
namespace mx = mlx::core;

// RMSNorm uses a kernel from MLX's base metallib, so this native-region example
// also exercises the delivered resource instead of testing only JIT kernels.
inline hyle::Status verify_base_kernel(const a::Context &context) {
  constexpr float epsilon = 1e-5f;
  const std::array specs{a::InputSpec{{1, 64}, mx::float32}};
  HYLE_ASSIGN_OR_RETURN(auto program,
                        context.compile(
                            [](const auto &inputs, mx::Stream stream) {
                              return std::vector{mx::fast::rms_norm(
                                  inputs[0], std::nullopt, epsilon, stream)};
                            },
                            specs));
  std::array<float, 64> host{};
  double mean_square = 0;
  for (std::size_t i = 0; i < host.size(); ++i) {
    host[i] = float(static_cast<int>(i) % 13 - 6) * 0.125f;
    mean_square += double(host[i]) * host[i] / host.size();
  }
  const std::array inputs{mx::array(host.data(), {1, 64})};
  HYLE_ASSIGN_OR_RETURN(auto invocation, program.create_invocation());
  HYLE_RETURN_IF_ERROR(invocation.bind(inputs));
  HYLE_ASSIGN_OR_RETURN(auto work, invocation.submit());
  HYLE_ASSIGN_OR_RETURN(auto result, work.wait());
  const auto &output = result->arrays()[0];
  const double divisor = std::sqrt(mean_square + epsilon);
  for (std::size_t i = 0; i < host.size(); ++i) {
    if (std::abs(output.data<float>()[i] - host[i] / divisor) > 1e-5)
      return hyle::Status::Internal("Apple native RMSNorm oracle mismatch");
  }
  return hyle::Status::Ok();
}

// Native immutable weights, independent of model/file-format policy. Real
// applications may instead use MLX quantize and keep its checked-ready arrays.
struct Weights final {
  static constexpr int k = 64, hidden = 64, output = 16;
  std::vector<mx::array> constants;
  std::vector<float> gate, up, down;

  Weights() : gate(k * hidden), up(k * hidden), down(hidden * output) {
    for (int projection = 0; projection < 2; ++projection) {
      std::vector<std::uint32_t> packed(hidden * k / 8, 0);
      std::vector<float> scales(hidden), biases(hidden);
      for (int n = 0; n < hidden; ++n) {
        scales[n] = 0.01f + 0.002f * float(n % 3);
        biases[n] = -0.07f - 0.003f * float(n % 5);
        for (int col = 0; col < k; ++col) {
          auto q = std::uint32_t((n * 5 + col * 3 + projection * 7) % 16);
          packed[(n * k + col) / 8] |= q << (4 * (col % 8));
          (projection == 0 ? gate : up)[n * k + col] =
              float(q) * scales[n] + biases[n];
        }
      }
      constants.emplace_back(packed.data(), mx::Shape{hidden, k / 8},
                             mx::uint32);
      constants.emplace_back(scales.data(), mx::Shape{hidden, 1}, mx::float32);
      constants.emplace_back(biases.data(), mx::Shape{hidden, 1}, mx::float32);
    }
    for (int i = 0; i < hidden * output; ++i)
      down[i] = float(i % 17 - 8) * 0.015f;
    constants.emplace_back(down.data(), mx::Shape{hidden, output}, mx::float32);
  }

  hyle::Status check(const a::Results &result, std::span<const float> x,
                     int rows) const {
    if (result.arrays().size() != 1 ||
        result.arrays()[0].shape() != mx::Shape{rows, output})
      return hyle::Status::Internal("Apple FFN result shape mismatch");
    const float *actual = result.arrays()[0].data<float>();
    for (int row = 0; row < rows; ++row) {
      for (int n = 0; n < output; ++n) {
        double expected = 0;
        for (int h = 0; h < hidden; ++h) {
          double g = 0, u = 0;
          for (int i = 0; i < k; ++i) {
            g += double(x[row * k + i]) * gate[h * k + i];
            u += double(x[row * k + i]) * up[h * k + i];
          }
          expected += g / (1.0 + std::exp(-g)) * u * down[h * output + n];
        }
        if (std::abs(actual[row * output + n] - expected) > 2e-5)
          return hyle::Status::Internal(
              "Apple FFN differs from independent affine Q4 oracle");
      }
    }
    return hyle::Status::Ok();
  }
};

inline hyle::Result<a::Program> prepare(const a::Context &context,
                                        const Weights &weights, int rows) {
  HYLE_ASSIGN_OR_RETURN(auto workload, sample_workload::create());
  const std::array specs{a::InputSpec{{rows, Weights::k}, mx::float32}};
  auto program = context.compile(
      [constants = weights.constants, workload](const auto &inputs,
                                                mx::Stream stream)
          -> hyle::Result<std::vector<mx::array>> {
        const c::QuantizedMatMul qmm{
            .mode = c::QuantizedWeightMode::kUnsignedAffine4};
        const std::array gate_inputs{inputs[0], constants[0], constants[1],
                                     constants[2]};
        const std::array up_inputs{inputs[0], constants[3], constants[4],
                                   constants[5]};
        HYLE_ASSIGN_OR_RETURN(
            auto gate, a::lower(c::Operation(qmm), gate_inputs, stream, workload));
        HYLE_ASSIGN_OR_RETURN(
            auto up, a::lower(c::Operation(qmm), up_inputs, stream, workload));
        const std::array gated_inputs{gate[0], up[0]};
        HYLE_ASSIGN_OR_RETURN(
            auto gated, a::lower(c::Operation(c::GatedActivation{
                                     .inputs = c::GatedInputKind::kSeparate}),
                                 gated_inputs, stream, workload));
        const std::array down_inputs{gated[0], constants[6]};
        return a::lower(c::Operation(c::MatMul{}), down_inputs, stream, workload);
      },
      specs, workload);
  if (program && program->workload() != workload)
    return std::unexpected(
        hyle::Status::Internal("Apple workload was not retained"));
  return program;
}

inline hyle::Status replay(const a::Program &program, const Weights &weights,
                           int rows) {
  HYLE_ASSIGN_OR_RETURN(auto invocation, program.create_invocation());
  for (int run = 0; run < 3; ++run) {
    std::vector<float> host(rows * Weights::k);
    for (std::size_t i = 0; i < host.size(); ++i)
      host[i] = float((static_cast<int>(i) * 7 + run * 3) % 23 - 11) * 0.09f;
    const std::array inputs{mx::array(host.data(), {rows, Weights::k})};
    HYLE_RETURN_IF_ERROR(invocation.bind(inputs));
    HYLE_ASSIGN_OR_RETURN(auto work, invocation.submit());
    HYLE_ASSIGN_OR_RETURN(auto result, work.wait());
    HYLE_RETURN_IF_ERROR(weights.check(*result.get(), host, rows));
    // The lease and completion leave scope before this slot is reused.
  }
  return hyle::Status::Ok();
}

inline hyle::Status run() {
  HYLE_ASSIGN_OR_RETURN(auto context, a::Context::create());
  HYLE_RETURN_IF_ERROR(verify_base_kernel(context));
  Weights weights;
  for (int rows : {1, 4}) {
    HYLE_ASSIGN_OR_RETURN(auto program, prepare(context, weights, rows));
    HYLE_RETURN_IF_ERROR(replay(program, weights, rows));
  }
  return context.close();
}
} // namespace sample_apple_direct

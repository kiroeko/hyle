#pragma once

#include "workload.h"

#include <hyle/compute/vulkan.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <span>
#include <vector>

#include "vulkan_affine.h"
#include "vulkan_mature.h"
#include "vulkan_topk.h"

namespace sample_vulkan_direct {
namespace c = hyle::compute;
namespace v = c::vulkan;

inline c::TensorDesc dense(std::initializer_list<std::int64_t> dimensions) {
  return *c::TensorDesc::dense(c::DataType::kFloat32,
                               {dimensions.begin(), dimensions.size()});
}

inline std::vector<float> data(std::size_t count, std::uint32_t seed) {
  std::vector<float> result(count);
  for (std::size_t i = 0; i < count; ++i) {
    auto bits = static_cast<std::uint32_t>(i) + seed * 0x9e3779b9u;
    bits = (bits ^ (bits >> 16)) * 0x7feb352du;
    bits = (bits ^ (bits >> 15)) * 0x846ca68bu;
    bits ^= bits >> 16;
    result[i] = static_cast<float>(static_cast<int>(bits % 103u) - 51) / 64.f;
  }
  return result;
}

inline std::vector<float> matmul(std::span<const float> a,
                                 std::span<const float> b, int m, int k,
                                 int n) {
  std::vector<float> result(static_cast<std::size_t>(m) * n);
  for (int row = 0; row < m; ++row)
    for (int col = 0; col < n; ++col) {
      double value = 0;
      for (int inner = 0; inner < k; ++inner)
        value += double(a[row * k + inner]) * b[inner * n + col];
      result[row * n + col] = static_cast<float>(value);
    }
  return result;
}

inline hyle::Status check(const v::Context &context, const v::Tensor &tensor,
                          std::span<const float> expected) {
  std::vector<float> actual(expected.size());
  HYLE_RETURN_IF_ERROR(
      context.download(tensor, std::as_writable_bytes(std::span(actual))));
  for (std::size_t i = 0; i < actual.size(); ++i)
    if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) >
                                         1e-4f + 4e-5f * std::abs(expected[i]))
      return hyle::Status::Internal(
          "Vulkan sample independent numerical oracle mismatch");
  return hyle::Status::Ok();
}

inline std::vector<std::byte> count_push(std::uint32_t count) {
  std::vector<std::byte> bytes(sizeof(count));
  std::memcpy(bytes.data(), &count, sizeof(count));
  return bytes;
}

inline hyle::Result<v::Kernel> affine(const v::Context &context) {
  return v::Kernel::create(context,
                           {affine_spirv, 2, sizeof(std::uint32_t), {}});
}

struct Weights final {
  static constexpr int k = 67, hidden = 37, output = 13;
  std::vector<float> gate = data(k * hidden, 1);
  std::vector<float> up = data(k * hidden, 2);
  std::vector<float> down = data(hidden * output, 3);

  hyle::Result<std::array<v::Tensor, 3>>
  constants(const v::Context &context) const {
    HYLE_ASSIGN_OR_RETURN(
        auto g,
        context.upload(dense({k, hidden}), std::as_bytes(std::span(gate))));
    HYLE_ASSIGN_OR_RETURN(auto u, context.upload(dense({k, hidden}),
                                                 std::as_bytes(std::span(up))));
    HYLE_ASSIGN_OR_RETURN(auto d,
                          context.upload(dense({hidden, output}),
                                         std::as_bytes(std::span(down))));
    return std::array{std::move(g), std::move(u), std::move(d)};
  }

  std::vector<float> oracle(std::span<const float> x, int rows) const {
    auto g = matmul(x, gate, rows, k, hidden);
    const auto u = matmul(x, up, rows, k, hidden);
    for (std::size_t i = 0; i < g.size(); ++i)
      g[i] = g[i] / (1.f + std::exp(-g[i])) * u[i];
    auto result = matmul(g, down, rows, hidden, output);
    for (auto &value : result)
      value = 2.f * value + 1.f; // Application's native custom fragment.
    return result;
  }
};

// Ordinary typed Direct: changing ready inputs, automatic output ownership.
inline hyle::Status ordinary(const v::Context &context) {
  const auto a_desc = dense({3, 7}), b_desc = dense({7, 5});
  const auto b_host = data(7 * 5, 6);
  HYLE_ASSIGN_OR_RETURN(
      auto b, context.upload(b_desc, std::as_bytes(std::span(b_host))));
  HYLE_ASSIGN_OR_RETURN(auto program,
                        v::prepare(context, c::MatMul{}, a_desc, b));
  HYLE_ASSIGN_OR_RETURN(auto slot, program.create_invocation());
  for (std::uint32_t iteration = 1; iteration <= 3; ++iteration) {
    const auto host = data(3 * 7, iteration);
    HYLE_ASSIGN_OR_RETURN(
        auto a, context.upload(a_desc, std::as_bytes(std::span(host))));
    HYLE_RETURN_IF_ERROR(slot.bind(a));
    HYLE_ASSIGN_OR_RETURN(auto work, slot.submit());
    HYLE_ASSIGN_OR_RETURN(auto result, work.wait());
    HYLE_RETURN_IF_ERROR(
        check(context, result->tensors()[0], matmul(host, b_host, 3, 7, 5)));
    // This result lease ends before the slot is reused.
  }
  return hyle::Status::Ok();
}

// Native composition keeps caller-owned buffers and records the full FFN plus
// application SPIR-V once. The caller waits before overwriting these outputs.
inline hyle::Status recorded_ffn(const v::Context &context,
                                 const Weights &weights, int rows) {
  HYLE_ASSIGN_OR_RETURN(auto constants, weights.constants(context));
  const auto input_desc = dense({rows, Weights::k});
  const auto hidden_desc = dense({rows, Weights::hidden});
  const auto output_desc = dense({rows, Weights::output});
  const auto first = data(static_cast<std::size_t>(rows) * Weights::k, 1);
  HYLE_ASSIGN_OR_RETURN(
      auto input, context.upload(input_desc, std::as_bytes(std::span(first))));
  HYLE_ASSIGN_OR_RETURN(auto borrowed,
                        context.borrow_ready(input_desc, input.native_buffer(),
                                             input.native_offset(),
                                             *input_desc.required_bytes(),
                                             input.native_owner()));
  HYLE_ASSIGN_OR_RETURN(auto gate, context.allocate(hidden_desc));
  HYLE_ASSIGN_OR_RETURN(auto up, context.allocate(hidden_desc));
  HYLE_ASSIGN_OR_RETURN(auto product, context.allocate(hidden_desc));
  HYLE_ASSIGN_OR_RETURN(auto down, context.allocate(output_desc));
  HYLE_ASSIGN_OR_RETURN(auto output, context.allocate(output_desc));
  HYLE_ASSIGN_OR_RETURN(auto workload, sample_workload::create());
  HYLE_ASSIGN_OR_RETURN(
      auto gate_op,
      v::prepare(context, workload, c::MatMul{}, input_desc, constants[0]));
  HYLE_ASSIGN_OR_RETURN(
      auto up_op,
      v::prepare(context, workload, c::MatMul{}, input_desc, constants[1]));
  HYLE_ASSIGN_OR_RETURN(
      auto silu_op,
      v::prepare(context, workload,
                 c::GatedActivation{.inputs = c::GatedInputKind::kSeparate},
                 hidden_desc, hidden_desc));
  HYLE_ASSIGN_OR_RETURN(auto down_op, v::prepare(context, workload, c::MatMul{},
                                                 hidden_desc, constants[2]));
  if (gate_op.workload() != workload || down_op.workload() != workload)
    return hyle::Status::Internal("Vulkan workload was not retained");
  HYLE_ASSIGN_OR_RETURN(auto custom, affine(context));
  HYLE_ASSIGN_OR_RETURN(auto recorder, v::Recorder::create(context));
  HYLE_RETURN_IF_ERROR(
      recorder.append(gate_op, std::array{borrowed}, std::array{gate}));
  HYLE_RETURN_IF_ERROR(
      recorder.append(up_op, std::array{borrowed}, std::array{up}));
  HYLE_RETURN_IF_ERROR(
      recorder.append(silu_op, std::array{gate, up}, std::array{product}));
  HYLE_RETURN_IF_ERROR(
      recorder.append(down_op, std::array{product}, std::array{down}));
  const auto count = static_cast<std::uint32_t>(rows * Weights::output);
  HYLE_RETURN_IF_ERROR(recorder.dispatch(custom, std::array{down, output},
                                         {(count + 63) / 64, 1, 1},
                                         count_push(count)));
  HYLE_ASSIGN_OR_RETURN(auto program, recorder.finish());
  for (std::uint32_t iteration = 1; iteration <= 3; ++iteration) {
    const auto host =
        data(static_cast<std::size_t>(rows) * Weights::k, iteration);
    HYLE_RETURN_IF_ERROR(context.upload(input, std::as_bytes(std::span(host))));
    HYLE_ASSIGN_OR_RETURN(auto work, program.submit());
    HYLE_RETURN_IF_ERROR(work.wait());
    HYLE_RETURN_IF_ERROR(check(context, output, weights.oracle(host, rows)));
  }
  return hyle::Status::Ok();
}

inline hyle::Status run() {
  HYLE_ASSIGN_OR_RETURN(auto context, v::Context::create());
  HYLE_RETURN_IF_ERROR(sample_vulkan_mature::direct(context));
  HYLE_RETURN_IF_ERROR(ordinary(context));
  HYLE_RETURN_IF_ERROR(sample_vulkan_topk::direct(context));
  const Weights weights;
  for (int rows : {1, 16})
    HYLE_RETURN_IF_ERROR(recorded_ffn(context, weights, rows));
  return context.close();
}
} // namespace sample_vulkan_direct

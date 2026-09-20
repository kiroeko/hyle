#pragma once

#include <hyle/compute/low_precision.h>
#include <hyle/compute/vulkan.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

namespace sample_vulkan_mature {
namespace c = hyle::compute;
namespace v = c::vulkan;

template <class T> std::vector<std::byte> bytes(const std::vector<T> &values) {
  const auto view = std::as_bytes(std::span(values));
  return {view.begin(), view.end()};
}

struct Example final {
  c::Operation operation;
  std::vector<v::Tensor> inputs;
  std::vector<std::vector<std::byte>> expected;
  bool floating;
};

// Closed-form, nonzero-state recurrence; no provider math is used as an oracle.
inline hyle::Result<Example> recurrent(const v::Context &context) {
  Example example{c::Operation(c::GatedDeltaNet{}), {}, {}, true};
  const std::array<std::vector<float>, 6> data{
      std::vector<float>(32, 1.f / 8), std::vector<float>(32, 1.f / 8),
      std::vector<float>(32, 1.f / 4), std::vector<float>{0},
      std::vector<float>{0.5f},        std::vector<float>(32 * 32, 1.f / 64)};
  for (std::size_t port = 0; port < data.size(); ++port) {
    const std::array<std::int64_t, 4> shape =
        port == 5 ? std::array<std::int64_t, 4>{1, 1, 32, 32}
                  : std::array<std::int64_t, 4>{1, 1, 1, port < 3 ? 32 : 1};
    HYLE_ASSIGN_OR_RETURN(auto desc,
                          c::TensorDesc::dense(c::DataType::kFloat32, shape));
    HYLE_ASSIGN_OR_RETURN(auto input, context.upload(desc, bytes(data[port])));
    example.inputs.push_back(std::move(input));
  }
  example.expected = {
      bytes(std::vector<float>(32, float((7.0 / 64) / std::sqrt(32.0)))),
      bytes(std::vector<float>(32 * 32, 7.f / 256))};
  return example;
}

inline hyle::Result<Example> scan(const v::Context &context) {
  HYLE_ASSIGN_OR_RETURN(
      auto desc, c::TensorDesc::dense(c::DataType::kInt32,
                                      std::array<std::int64_t, 2>{2, 4}));
  const std::vector<std::int32_t> data{1, 2, -1, 3, -2, 1, 3, -1};
  HYLE_ASSIGN_OR_RETURN(auto input, context.upload(desc, bytes(data)));
  return Example{c::Operation(c::Scan{}),
                 {std::move(input)},
                 {bytes(std::vector<std::int32_t>{1, 3, 2, 5, -2, -1, 2, 1})},
                 false};
}

inline hyle::Result<Example> float8(const v::Context &context,
                                    c::Float8ScaleMode mode) {
  const bool block = mode == c::Float8ScaleMode::kFloat32Block128;
  const std::int64_t k = block ? 128 : 32, n = block ? 128 : 5;
  HYLE_ASSIGN_OR_RETURN(auto a_desc,
                        c::float8_activation_description(1, k, mode));
  HYLE_ASSIGN_OR_RETURN(auto w_desc, c::float8_weight_description(k, n, mode));
  HYLE_ASSIGN_OR_RETURN(auto a_bytes, a_desc.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto w_bytes, w_desc.required_bytes());
  std::vector<std::byte> a(a_bytes), w(w_bytes);
  // E4M3 1 and 2, activation scale 1/2, weight scale 2: each product is 2.
  for (std::size_t i = 0; i < std::size_t(k); ++i)
    a[i] = std::byte{0x38};
  for (std::size_t i = 0; i < std::size_t(n * k); ++i)
    w[i] = std::byte{0x40};
  if (block) {
    const float a_scale = 0.5f, w_scale = 2.f;
    std::memcpy(a.data() + k, &a_scale, sizeof(a_scale));
    std::memcpy(w.data() + n * k, &w_scale, sizeof(w_scale));
  } else {
    a[std::size_t(k)] = std::byte{126};
    for (std::size_t i = std::size_t(n * k); i < w.size(); ++i)
      w[i] = std::byte{128};
  }
  HYLE_ASSIGN_OR_RETURN(auto activation, context.upload(a_desc, a));
  HYLE_ASSIGN_OR_RETURN(auto weight, context.upload(w_desc, w));
  return Example{c::Operation(c::MatMul{}),
                 {std::move(activation), std::move(weight)},
                 {bytes(std::vector<float>(std::size_t(n), float(2 * k)))},
                 true};
}

inline hyle::Result<Example> mxfp4(const v::Context &context) {
  HYLE_ASSIGN_OR_RETURN(
      auto a_desc, c::TensorDesc::dense(c::DataType::kFloat16,
                                        std::array<std::int64_t, 2>{1, 32}));
  HYLE_ASSIGN_OR_RETURN(auto w_desc, c::mxfp4_weight_description(32, 3));
  const std::vector<std::uint16_t> a(32, 0x3c00);    // F16 1.
  std::vector<std::byte> w(3 * 17, std::byte{0x22}); // Two E2M1 1s per byte.
  for (std::size_t column = 0; column < 3; ++column)
    w[column * 17] = std::byte{127}; // E8M0 scale 1.
  HYLE_ASSIGN_OR_RETURN(auto activation, context.upload(a_desc, bytes(a)));
  HYLE_ASSIGN_OR_RETURN(auto weight, context.upload(w_desc, w));
  return Example{c::Operation(c::MatMul{}),
                 {std::move(activation), std::move(weight)},
                 {bytes(std::vector<std::uint16_t>(3, 0x5000))},
                 false}; // F16 32.
}

inline hyle::Result<std::array<Example, 5>>
examples(const v::Context &context) {
  HYLE_ASSIGN_OR_RETURN(auto r, recurrent(context));
  HYLE_ASSIGN_OR_RETURN(auto s, scan(context));
  HYLE_ASSIGN_OR_RETURN(auto f,
                        float8(context, c::Float8ScaleMode::kFloat32Block128));
  HYLE_ASSIGN_OR_RETURN(
      auto u, float8(context, c::Float8ScaleMode::kUnsignedExponent8Group32));
  HYLE_ASSIGN_OR_RETURN(auto m, mxfp4(context));
  return std::array{std::move(r), std::move(s), std::move(f), std::move(u),
                    std::move(m)};
}

inline hyle::Status check(const v::Context &context, const Example &example,
                          std::span<const v::Tensor> outputs) {
  if (outputs.size() != example.expected.size())
    return hyle::Status::Internal("Vulkan mature sample output count");
  for (std::size_t port = 0; port < outputs.size(); ++port) {
    const auto &expected = example.expected[port];
    std::vector<std::byte> actual(expected.size());
    HYLE_RETURN_IF_ERROR(context.download(outputs[port], actual));
    if (!example.floating) {
      if (actual != expected)
        return hyle::Status::Internal(
            "Vulkan mature sample integer/F16 mismatch");
    } else {
      for (std::size_t offset = 0; offset < actual.size();
           offset += sizeof(float)) {
        float value, reference;
        std::memcpy(&value, actual.data() + offset, sizeof(value));
        std::memcpy(&reference, expected.data() + offset, sizeof(reference));
        if (!std::isfinite(value) || std::abs(value - reference) > 2e-6f)
          return hyle::Status::Internal("Vulkan mature sample F32 mismatch");
      }
    }
  }
  return hyle::Status::Ok();
}

inline hyle::Status replay(const v::Context &context, const Example &example,
                           const v::Program &program) {
  HYLE_ASSIGN_OR_RETURN(auto first, program.create_invocation());
  HYLE_ASSIGN_OR_RETURN(auto second, program.create_invocation());
  std::vector<v::Tensor> retained;
  for (auto *slot : {&first, &second}) {
    HYLE_RETURN_IF_ERROR(slot->bind(example.inputs));
    HYLE_ASSIGN_OR_RETURN(auto work, slot->submit());
    HYLE_ASSIGN_OR_RETURN(auto result, work.wait());
    HYLE_RETURN_IF_ERROR(check(context, example, result->tensors()));
    if (retained.empty())
      retained.assign(result->tensors().begin(), result->tensors().end());
    else
      HYLE_RETURN_IF_ERROR(check(context, example, retained));
  }
  return hyle::Status::Ok();
}

inline hyle::Status direct(const v::Context &context) {
  HYLE_ASSIGN_OR_RETURN(auto cases, examples(context));
  for (const auto &example : cases) {
    std::vector<v::Argument> arguments;
    for (const auto &input : example.inputs)
      arguments.emplace_back(input.description());
    HYLE_ASSIGN_OR_RETURN(auto program,
                          v::prepare(context, example.operation, arguments));
    HYLE_RETURN_IF_ERROR(replay(context, example, program));
  }
  return hyle::Status::Ok();
}
} // namespace sample_vulkan_mature

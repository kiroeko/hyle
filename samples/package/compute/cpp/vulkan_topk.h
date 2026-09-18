#pragma once

#include <hyle/compute/vulkan.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <numeric>
#include <span>
#include <vector>

namespace sample_vulkan_topk {
namespace c = hyle::compute;
namespace v = c::vulkan;

inline constexpr std::int64_t rows = 2, experts = 288, selected = 8;

inline hyle::Result<c::TensorDesc> description() {
  return c::TensorDesc::dense(c::DataType::kFloat32, std::array{rows, experts});
}

inline std::vector<float> scores(int phase) {
  std::vector<float> result(rows * experts);
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = static_cast<float>((i * 13 + phase * 7) % 31) - 15.f;
  // Stable ties and non-finite input are part of the ordinary contract.
  result[0] = std::bit_cast<float>(std::uint32_t{0x7fc00015});
  result[1] = -0.f;
  result[2] = 0.f;
  return result;
}

inline hyle::Status check(const v::Context &context,
                          std::span<const v::Tensor> outputs,
                          std::span<const float> input, bool largest,
                          bool negate_values = false) {
  if (outputs.size() != 2 ||
      outputs[0].description().data_type() != c::DataType::kFloat32 ||
      outputs[1].description().data_type() != c::DataType::kInt32)
    return hyle::Status::Internal("TopK sample output ports differ");
  std::array<float, rows * selected> values{};
  std::array<std::int32_t, rows * selected> indices{};
  HYLE_RETURN_IF_ERROR(
      context.download(outputs[0], std::as_writable_bytes(std::span(values))));
  HYLE_RETURN_IF_ERROR(
      context.download(outputs[1], std::as_writable_bytes(std::span(indices))));
  for (std::int64_t row = 0; row < rows; ++row) {
    std::array<std::int32_t, experts> order{};
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
      const auto left = input[row * experts + a];
      const auto right = input[row * experts + b];
      if (std::isnan(left))
        return false;
      if (std::isnan(right))
        return true;
      return largest ? left > right : left < right;
    });
    for (std::int64_t col = 0; col < selected; ++col) {
      const auto position = row * selected + col;
      const auto source = input[row * experts + order[col]];
      const auto expected = negate_values ? -source : source;
      if (indices[position] != order[col] ||
          std::bit_cast<std::uint32_t>(values[position]) !=
              std::bit_cast<std::uint32_t>(expected))
        return hyle::Status::Internal(
            "TopK sample stable values/indices differ");
    }
  }
  return hyle::Status::Ok();
}

inline hyle::Status direct(const v::Context &context) {
  HYLE_ASSIGN_OR_RETURN(auto desc, description());
  for (bool largest : {false, true}) {
    HYLE_ASSIGN_OR_RETURN(
        auto program,
        v::prepare(context, c::TopK{.count = selected, .largest = largest},
                   desc));
    HYLE_ASSIGN_OR_RETURN(auto slot, program.create_invocation());
    for (int phase = 0; phase < 3; ++phase) {
      const auto host = scores(phase);
      HYLE_ASSIGN_OR_RETURN(
          auto input, context.upload(desc, std::as_bytes(std::span(host))));
      HYLE_RETURN_IF_ERROR(slot.bind(input));
      HYLE_ASSIGN_OR_RETURN(auto work, slot.submit());
      HYLE_ASSIGN_OR_RETURN(auto result, work.wait());
      HYLE_RETURN_IF_ERROR(check(context, result->tensors(), host, largest));
    }
  }
  return hyle::Status::Ok();
}
} // namespace sample_vulkan_topk

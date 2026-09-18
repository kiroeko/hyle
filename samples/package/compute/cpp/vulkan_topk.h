#pragma once

#include <hyle/compute/vulkan.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
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

// A dense logical view can still begin at an unaligned native byte address.
inline hyle::Result<v::Tensor> captured_scores(const v::Context &context,
                                               std::span<const float> input,
                                               std::int64_t origin) {
  HYLE_ASSIGN_OR_RETURN(
      auto storage_desc,
      c::TensorDesc::strided_bytes(
          c::DataType::kFloat32,
          std::array<std::int64_t, 8>{rows, 1, 1, 1, 1, 1, 1, experts * 4 + 8},
          std::array<std::int64_t, 8>{experts * 4, experts * 4, experts * 4,
                                      experts * 4, experts * 4, experts * 4,
                                      experts * 4, 1}));
  HYLE_ASSIGN_OR_RETURN(auto capacity, storage_desc.required_bytes());
  std::vector<std::byte> host(capacity, std::byte{0x5a});
  std::memcpy(host.data() + origin, input.data(), input.size_bytes());
  HYLE_ASSIGN_OR_RETURN(auto storage, context.upload(storage_desc, host));
  HYLE_ASSIGN_OR_RETURN(auto view, c::View::slice(7, origin, experts, 4));
  return storage.view(view);
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
  const auto host = scores(1);
  for (std::int64_t origin : {1, 4, 7}) {
    HYLE_ASSIGN_OR_RETURN(auto input, captured_scores(context, host, origin));
    HYLE_ASSIGN_OR_RETURN(
        auto program, v::prepare(context, c::TopK{.count = selected}, input));
    HYLE_ASSIGN_OR_RETURN(auto slot, program.create_invocation());
    HYLE_RETURN_IF_ERROR(slot.bind(std::span<const v::Tensor>{}));
    HYLE_ASSIGN_OR_RETURN(auto work, slot.submit());
    HYLE_ASSIGN_OR_RETURN(auto result, work.wait());
    HYLE_RETURN_IF_ERROR(check(context, result->tensors(), host, true));
  }
  return hyle::Status::Ok();
}
} // namespace sample_vulkan_topk

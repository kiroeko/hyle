#pragma once

#include <algorithm>
#include <numeric>

#include "tensor_test_support.h"

// Independent numeric oracle for the admitted F32 router selection contract.
namespace vulkan_topk_test {
using namespace vulkan_tensor_test;

inline std::vector<std::uint64_t> pattern(std::size_t count, int phase = 0) {
  constexpr std::array<std::uint32_t, 12> special{
      0x7fc00001U, 0xffc01234U, 0x7fc05678U, 0x80000000U,
      0U,          0x7f800000U, 0xff800000U, 1U,
      0x80000001U, 0x007fffffU, 0x7f800001U, 0xff800001U};
  std::vector<std::uint64_t> raw(count);
  for (std::size_t i = 0; i < count; ++i) {
    const auto n = i + static_cast<std::size_t>(phase) * 19;
    raw[i] = n % 23 < special.size()
                 ? special[n % 23]
                 : encode((static_cast<int>(n % 61) - 30) * 0.125,
                          c::DataType::kFloat32);
  }
  return raw;
}

inline hyle::Result<std::array<v::Tensor, 2>>
execute_all(const v::Program &program, const v::Tensor &input) {
  HYLE_ASSIGN_OR_RETURN(auto slot, program.create_invocation());
  HYLE_RETURN_IF_ERROR(slot.bind(input));
  HYLE_ASSIGN_OR_RETURN(auto work, slot.submit());
  HYLE_ASSIGN_OR_RETURN(auto result, work.wait());
  if (result.get()->tensors().size() != 2)
    return std::unexpected(
        hyle::Status::Internal("TopK must return two tensors"));
  return std::array{result.get()->tensors()[0], result.get()->tensors()[1]};
}

struct Reference final {
  std::vector<std::uint64_t> values;
  std::vector<std::uint64_t> indices;
};

inline Reference reference(const c::TensorDesc &input,
                           std::span<const std::uint64_t> raw,
                           const c::TopK &operation) {
  const auto axis = static_cast<std::size_t>(
      operation.axis < 0 ? operation.axis + static_cast<int>(input.rank())
                         : operation.axis);
  const auto width = static_cast<std::size_t>(input.dimensions()[axis]);
  std::size_t outer = 1, inner = 1;
  for (std::size_t i = 0; i < axis; ++i)
    outer *= static_cast<std::size_t>(input.dimensions()[i]);
  for (std::size_t i = axis + 1; i < input.rank(); ++i)
    inner *= static_cast<std::size_t>(input.dimensions()[i]);
  Reference result;
  result.values.resize(outer * operation.count * inner);
  result.indices.resize(result.values.size());
  for (std::size_t row = 0; row < outer; ++row) {
    for (std::size_t lane = 0; lane < inner; ++lane) {
      std::vector<std::size_t> order(width);
      std::iota(order.begin(), order.end(), 0);
      // Independent host numeric ordering: no shader ordered-key encoding,
      // bitonic network, vendor sorting or device output feeds the oracle.
      std::stable_sort(order.begin(), order.end(), [&](auto left, auto right) {
        const auto x = decode(raw[(row * width + left) * inner + lane],
                              c::DataType::kFloat32);
        const auto y = decode(raw[(row * width + right) * inner + lane],
                              c::DataType::kFloat32);
        if (std::isnan(x) != std::isnan(y))
          return !std::isnan(x);
        if (std::isnan(x) || x == y)
          return false;
        return operation.largest ? x > y : x < y;
      });
      for (std::size_t column = 0; column < operation.count; ++column) {
        const auto destination =
            (row * operation.count + column) * inner + lane;
        result.indices[destination] = order[column];
        result.values[destination] =
            raw[(row * width + order[column]) * inner + lane];
      }
    }
  }
  return result;
}

inline void expect_pair(const v::Context &context,
                        std::span<const v::Tensor> outputs,
                        const Reference &expected) {
  ASSERT_EQ(outputs.size(), 2U);
  EXPECT_EQ(outputs[0].description().data_type(), c::DataType::kFloat32);
  EXPECT_EQ(outputs[1].description().data_type(), c::DataType::kInt32);
  EXPECT_EQ(bits(context, outputs[0]), expected.values);
  EXPECT_EQ(bits(context, outputs[1]), expected.indices);
}
} // namespace vulkan_topk_test

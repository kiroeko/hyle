#pragma once

#include "gather_test_support.h"

namespace vulkan_scatter_test {
using namespace vulkan_tensor_test;
using vulkan_gather_test::raw_pattern;

inline std::vector<std::uint64_t> logical(const c::TensorDesc &desc,
                                          std::span<const std::byte> host) {
  std::vector<std::uint64_t> result(*desc.element_count(), 0);
  for (std::size_t i = 0; i < result.size(); ++i)
    std::memcpy(&result[i], host.data() + offset(desc, i),
                c::data_type_size(desc.data_type()));
  return result;
}

inline std::vector<std::uint64_t> numeric(const c::TensorDesc &desc,
                                          int phase = 0) {
  std::vector<std::uint64_t> result(*desc.element_count());
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = encode((static_cast<int>((i * 7) % 29) - 13 + phase) * 0.03125,
                       desc.data_type());
  return result;
}

inline std::vector<std::uint64_t>
reference(const c::TensorDesc &base_desc, std::span<const std::uint64_t> base,
          std::span<const std::uint64_t> indices,
          std::span<const std::uint64_t> updates, c::Scatter operation) {
  std::vector<std::uint64_t> result(base.begin(), base.end());
  if (result.empty())
    return result;
  const auto axis = static_cast<std::size_t>(
      operation.axis < 0
          ? operation.axis + static_cast<std::int32_t>(base_desc.rank())
          : operation.axis);
  const auto extent = static_cast<std::size_t>(base_desc.dimensions()[axis]);
  std::size_t inner = 1;
  for (std::size_t i = axis + 1; i < base_desc.rank(); ++i)
    inner *= static_cast<std::size_t>(base_desc.dimensions()[i]);
  for (std::size_t outer = 0; outer < result.size() / extent / inner; ++outer)
    for (std::size_t i = 0; i < indices.size(); ++i)
      for (std::size_t column = 0; column < inner; ++column) {
        const auto target =
            (outer * extent + static_cast<std::size_t>(indices[i])) * inner +
            column;
        const auto update =
            updates[(outer * indices.size() + i) * inner + column];
        if (operation.kind == c::ScatterKind::kAssign)
          result[target] = update;
        else {
          const float sum =
              static_cast<float>(
                  decode(result[target], base_desc.data_type())) +
              static_cast<float>(decode(update, base_desc.data_type()));
          result[target] = encode(sum, base_desc.data_type());
        }
      }
  return result;
}

inline void expect_scatter(const v::Context &context, const v::Tensor &output,
                           std::span<const std::uint64_t> expected,
                           bool adding) {
  const auto actual = bits(context, output);
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (adding &&
        std::isnan(decode(expected[i], output.description().data_type())))
      ASSERT_TRUE(
          std::isnan(decode(actual[i], output.description().data_type())))
          << i;
    else
      ASSERT_EQ(actual[i], expected[i]) << i;
  }
}

inline c::TensorDesc odd_layout(c::DataType type,
                                std::span<const std::int64_t> shape,
                                bool broadcast = false) {
  std::vector<std::int64_t> strides(shape.size());
  std::int64_t stride = static_cast<std::int64_t>(c::data_type_size(type)) + 1;
  for (std::size_t axis = shape.size(); axis-- > 0;) {
    strides[axis] = stride;
    stride = stride * shape[axis] + 3;
  }
  if (broadcast && shape.size() > 1)
    strides[0] = 0;
  return c::TensorDesc::strided_bytes(type, shape, strides).value();
}

struct OriginTensor final {
  v::Tensor backing, tensor;
  std::vector<std::byte> host;
};
inline OriginTensor with_origin(const v::Context &context,
                                const c::TensorDesc &desc,
                                std::span<const std::uint64_t> raw,
                                std::int64_t origin) {
  const auto size =
      static_cast<std::int64_t>(c::data_type_size(desc.data_type()));
  const auto count = static_cast<std::int64_t>(*desc.element_count());
  const auto storage =
      c::TensorDesc::strided_bytes(
          desc.data_type(), std::array<std::int64_t, 1>{count * size + origin},
          std::array<std::int64_t, 1>{1})
          .value();
  OriginTensor result;
  result.host.assign(*storage.required_bytes(), std::byte{0xcd});
  const auto packed = payload(desc, raw);
  std::ranges::copy(packed, result.host.begin() + origin);
  result.backing = context.upload(storage, result.host).value();
  auto flat =
      result.backing.view(c::View::slice(0, origin, count, size).value())
          .value();
  result.tensor =
      flat.view(c::View::reshape(
                    c::TensorShape::create(desc.dimensions()).value())
                    .value())
          .value();
  return result;
}
} // namespace vulkan_scatter_test

#include <gtest/gtest.h>

#include "concat_test_support.h"
#include "vulkan_internal.h"

namespace {
using namespace vulkan_concat_test;

TEST(VulkanConcat, VariadicExactStorageAcrossEveryAxisAndStridedStorage) {
  auto context = vulkan_test::context().value();
  for (auto type :
       {c::DataType::kFloat32, c::DataType::kFloat16, c::DataType::kBFloat16,
        c::DataType::kInt32, c::DataType::kInt64, c::DataType::kBool}) {
    const auto size = static_cast<std::int64_t>(c::data_type_size(type));
    for (auto arity : {1, 2, 3, 4, 5, 9}) {
      for (int axis = 0; axis < 3; ++axis) {
        for (int pattern = 0; pattern < 2; ++pattern) {
          SCOPED_TRACE(::testing::Message()
                       << static_cast<int>(type) << ':' << arity << ':' << axis
                       << ':' << pattern);
          std::vector<v::Tensor> inputs;
          std::vector<std::vector<std::uint64_t>> parts;
          std::vector<std::int64_t> widths;
          const std::array<std::int64_t, 3> base{2, 3, 5};
          for (int port = 0; port < arity; ++port) {
            const auto length = pattern == 0 ? 1 : (port * 3 + 1) % 4;
            auto shape = base;
            shape[axis] = length;
            const auto view_desc =
                c::TensorDesc::strided_bytes(
                    type, shape,
                    std::array<std::int64_t, 3>{
                        size + 1, shape[2] * (shape[0] * (size + 1) + 3) + 5,
                        shape[0] * (size + 1) + 3})
                    .value();
            std::vector<std::uint64_t> raw(*view_desc.element_count());
            for (std::size_t i = 0; i < raw.size(); ++i)
              raw[i] = raw_value(i, port, type);
            const auto body = payload(view_desc, raw);
            inputs.push_back(context.upload(view_desc, body).value());
            parts.push_back(std::move(raw));
            widths.push_back(length);
          }
          std::size_t outer = 1, inner = 1;
          for (int dim = 0; dim < axis; ++dim)
            outer *= static_cast<std::size_t>(base[dim]);
          for (int dim = axis + 1; dim < 3; ++dim)
            inner *= static_cast<std::size_t>(base[dim]);
          auto result = execute(
              context, c::Operation(c::Concat{pattern ? axis - 3 : axis}),
              inputs);
          ASSERT_TRUE(result) << result.error().message();
          EXPECT_EQ(bits(context, *result),
                    concatenate(parts, widths, outer, inner));
          for (std::size_t port = 0; port < inputs.size(); ++port)
            EXPECT_EQ(bits(context, inputs[port]), parts[port]);
        }
      }
    }
  }
}

TEST(VulkanConcat, EmptyGroupsHugeEmptyBroadcastAndRepeatedInputs) {
  auto context = vulkan_test::context().value();
  for (auto widths : {std::vector<std::int64_t>{0, 0, 0, 1, 2, 0, 0, 0, 0},
                      std::vector<std::int64_t>{0, 0, 0, 0, 0}}) {
    std::vector<v::Tensor> inputs;
    std::vector<std::vector<std::uint64_t>> parts;
    for (auto width : widths) {
      const auto desc = dense(c::DataType::kInt64, {7, width, 3});
      std::vector<std::uint64_t> raw(*desc.element_count());
      for (std::size_t i = 0; i < raw.size(); ++i)
        raw[i] = raw_value(i, inputs.size(), c::DataType::kInt64);
      inputs.push_back(context.upload(desc, payload(desc, raw)).value());
      parts.push_back(std::move(raw));
    }
    auto result = execute(context, c::Operation(c::Concat{1}), inputs);
    ASSERT_TRUE(result) << result.error().message();
    EXPECT_EQ(bits(context, *result), concatenate(parts, widths, 7, 3));
  }
  const auto huge = dense(c::DataType::kBool,
                          {0, (std::numeric_limits<std::int64_t>::max)() / 2});
  const auto empty = context.allocate(huge).value();
  auto result =
      execute(context, c::Operation(c::Concat{-1}), std::array{empty, empty});
  ASSERT_TRUE(result) << result.error().message();
  EXPECT_EQ(*result->description().element_count(), 0U);
  const auto one_desc = dense(c::DataType::kBFloat16, {1});
  const auto one =
      context
          .upload(one_desc,
                  payload(one_desc, std::array<std::uint64_t, 1>{0x7fc5}))
          .value();
  const auto shape = c::TensorShape::create(
                         std::array<std::int64_t, 8>{1, 1, 1, 2, 1, 3, 1, 5})
                         .value();
  const auto broadcast = one.view(c::View::broadcast_to(shape).value()).value();
  std::array<v::Tensor, 9> repeated;
  repeated.fill(broadcast);
  result = execute(context, c::Operation(c::Concat{-1}), repeated);
  ASSERT_TRUE(result) << result.error().message();
  EXPECT_EQ(bits(context, *result), std::vector<std::uint64_t>(270, 0x7fc5));
}

TEST(VulkanConcat,
     SharedWordBoundariesPreserveBorrowedNeighborsAndRejectAlias) {
  auto context = vulkan_test::context().value();
  const auto alignment =
      std::max(VkDeviceSize{4},
               v::detail::Access::runtime(context)
                   ->device->properties.limits.minStorageBufferOffsetAlignment);
  for (auto type :
       {c::DataType::kBool, c::DataType::kFloat16, c::DataType::kBFloat16}) {
    std::vector<v::Tensor> inputs;
    std::vector<std::vector<std::uint64_t>> parts;
    std::vector<v::Argument> arguments;
    for (int port = 0; port < 5; ++port) {
      const auto desc = dense(type, {7, 1});
      std::vector<std::uint64_t> raw(7);
      for (std::size_t row = 0; row < raw.size(); ++row)
        raw[row] = raw_value(row, port, type);
      inputs.push_back(context.upload(desc, payload(desc, raw)).value());
      parts.push_back(std::move(raw));
      arguments.emplace_back(desc);
    }
    const auto output_desc = dense(type, {7, 5});
    const auto capacity = (*output_desc.required_bytes() + 3) & ~std::size_t{3};
    std::vector<std::byte> guard(alignment + capacity + 8, std::byte{0x5a});
    const auto backing =
        context
            .upload(dense(c::DataType::kBool,
                          {static_cast<std::int64_t>(guard.size())}),
                    guard)
            .value();
    const auto output =
        context
            .borrow_ready(output_desc, backing.native_buffer(), alignment,
                          capacity, backing.native_owner())
            .value();
    const auto program = v::prepare(context, c::Operation(c::Concat{-1}),
                                    std::span<const v::Argument>(arguments))
                             .value();
    auto recorder = v::Recorder::create(context).value();
    ASSERT_TRUE(recorder.append(program, inputs, std::array{output}).ok());
    auto recording = recorder.finish().value();
    auto work = recording.submit().value();
    ASSERT_TRUE(work.wait().ok());
    EXPECT_EQ(
        bits(context, output),
        concatenate(parts, std::array<std::int64_t, 5>{1, 1, 1, 1, 1}, 7, 1));
    const auto observed = bytes(context, backing);
    for (std::size_t i = 0; i < observed.size(); ++i)
      if (i < alignment || i >= alignment + capacity)
        EXPECT_EQ(observed[i], guard[i]) << i;
    const auto copy =
        v::prepare(context, c::Concat{}, inputs[0].description()).value();
    auto overlapping = v::Recorder::create(context).value();
    EXPECT_FALSE(
        overlapping.append(copy, std::array{inputs[0]}, std::array{inputs[0]})
            .ok());
  }
}
} // namespace

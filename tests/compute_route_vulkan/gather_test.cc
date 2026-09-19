#include <gtest/gtest.h>

#include <algorithm>

#include "gather_test_support.h"
#include "vulkan_program_internal.h"

namespace {
using namespace vulkan_gather_test;

TEST(VulkanGather, AllScalarStorageAxesAndRawPayloadsUseOriginalGetRows) {
  auto context = vulkan_test::context().value();
  for (const auto type :
       {c::DataType::kFloat32, c::DataType::kFloat16, c::DataType::kBFloat16,
        c::DataType::kInt32, c::DataType::kInt64, c::DataType::kBool}) {
    const auto desc = dense(type, {3, 5, 7});
    const auto host = payload(desc, raw_pattern(desc));
    const auto source = context.upload(desc, host).value();
    for (int axis = 0; axis < 3; ++axis) {
      const std::vector<std::uint64_t> positions{
          static_cast<std::uint64_t>(desc.dimensions()[axis] - 1), 0, 1, 1};
      for (const auto index_type : {c::DataType::kInt32, c::DataType::kInt64}) {
        SCOPED_TRACE(::testing::Message()
                     << static_cast<int>(type) << '/' << axis << '/'
                     << static_cast<int>(index_type));
        const auto index_desc = dense(index_type, {2, 2});
        const auto indices =
            context.upload(index_desc, payload(index_desc, positions)).value();
        for (const auto bounds : {c::IndexBoundsPolicy::kChecked,
                                  c::IndexBoundsPolicy::kUnchecked}) {
          const c::Operation operation(c::Gather{axis - 3, bounds});
          auto program = v::prepare(
              context, operation,
              std::array{v::Argument(desc), v::Argument(index_desc)});
          ASSERT_TRUE(program) << program.error().message();
          const auto &state = v::detail::Access::program(*program);
          EXPECT_TRUE(std::ranges::any_of(state->dispatches, [](const auto &d) {
            return d.region &&
                   d.region->nodes[0].operation.template get_if<c::Gather>();
          }));
          EXPECT_EQ(state->status_value.has_value(),
                    bounds == c::IndexBoundsPolicy::kChecked);
          auto slot = program->create_invocation();
          ASSERT_TRUE(slot) << slot.error().message();
          ASSERT_TRUE(slot->bind(std::array{source, indices}).ok());
          auto work = slot->submit().value();
          auto output = work.wait();
          ASSERT_TRUE(output) << output.error().message();
          EXPECT_EQ(
              bits(context, output->get()->tensors()[0]),
              reference(desc, host, positions, static_cast<std::size_t>(axis)));
        }
      }
    }
    EXPECT_EQ(bytes(context, source), host);
  }
}

TEST(VulkanGather, OddStridesBroadcastIndicesScalarAndRankEight) {
  auto context = vulkan_test::context().value();
  const std::array shape{std::int64_t{2}, std::int64_t{5}, std::int64_t{3}};
  for (const auto type :
       {c::DataType::kFloat16, c::DataType::kInt64, c::DataType::kBool}) {
    const auto size = static_cast<std::int64_t>(c::data_type_size(type));
    for (const auto strides :
         {std::array<std::int64_t, 3>{3 * 5 * (size + 1), 3 * (size + 1),
                                      size + 1},
          std::array<std::int64_t, 3>{0, 3 * size, size}}) {
      const auto desc =
          c::TensorDesc::strided_bytes(type, shape, strides).value();
      const auto host = payload(desc, raw_pattern(desc));
      const auto source = context.upload(desc, host).value();
      for (const auto &index_desc :
           {dense(c::DataType::kInt64, {}),
            c::TensorDesc::strided_bytes(c::DataType::kInt32,
                                         std::array<std::int64_t, 2>{3, 2},
                                         std::array<std::int64_t, 2>{0, 5})
                .value()}) {
        std::vector<std::uint64_t> positions(*index_desc.element_count(), 1);
        auto indices =
            context.upload(index_desc, payload(index_desc, positions)).value();
        auto output = execute(context, c::Operation(c::Gather{1}),
                              std::array{source, indices});
        ASSERT_TRUE(output) << output.error().message();
        EXPECT_EQ(bits(context, *output), reference(desc, host, positions, 1));
        EXPECT_EQ(bytes(context, source), host);
      }
    }
  }
  const auto desc = dense(c::DataType::kBFloat16, {2, 1, 1, 1, 1, 1, 1, 129});
  const auto host = payload(desc, raw_pattern(desc));
  const auto source = context.upload(desc, host).value();
  for (const auto &index_desc :
       {dense(c::DataType::kInt32, {}), dense(c::DataType::kInt64, {129})}) {
    std::vector<std::uint64_t> positions(*index_desc.element_count(), 1);
    const auto indices =
        context.upload(index_desc, payload(index_desc, positions)).value();
    auto output = execute(context, c::Operation(c::Gather{0}),
                          std::array{source, indices});
    ASSERT_TRUE(output) << output.error().message();
    EXPECT_EQ(bits(context, *output), reference(desc, host, positions, 0));
  }
  const auto scalar_source_desc = dense(c::DataType::kBool, {3});
  const auto scalar_source =
      context
          .upload(scalar_source_desc,
                  payload(scalar_source_desc,
                          std::array<std::uint64_t, 3>{1, 0, 1}))
          .value();
  const auto scalar_index_desc = dense(c::DataType::kInt32, {});
  const auto scalar_index =
      context
          .upload(scalar_index_desc,
                  payload(scalar_index_desc, std::array<std::uint64_t, 1>{1}))
          .value();
  auto scalar = execute(context, c::Operation(c::Gather{0}),
                        std::array{scalar_source, scalar_index});
  ASSERT_TRUE(scalar) << scalar.error().message();
  EXPECT_EQ(scalar->description().rank(), 0U);
  EXPECT_EQ(bits(context, *scalar), (std::vector<std::uint64_t>{0}));
}

TEST(VulkanGather, BadIndicesPreserveRecordedOutputAndRetireOnlyFailedSlot) {
  auto context = vulkan_test::context().value();
  const auto desc = dense(c::DataType::kBFloat16, {5, 7});
  const auto host = payload(desc, raw_pattern(desc));
  const auto source = context.upload(desc, host).value();
  for (const auto index_type : {c::DataType::kInt32, c::DataType::kInt64}) {
    const auto index_desc = dense(index_type, {129});
    auto program =
        v::prepare(context, c::Gather{0}, source, index_desc).value();
    const auto output_desc = program.output_descriptions()[0];
    const std::vector<std::uint64_t> sentinels(*output_desc.element_count(),
                                               0x55aa);
    for (const auto bad :
         {UINT64_MAX, std::uint64_t{5}, std::uint64_t{1} << 32}) {
      if (index_type == c::DataType::kInt32 && bad == (std::uint64_t{1} << 32))
        continue;
      std::vector<std::uint64_t> positions(129, 2);
      positions.back() = bad;
      const auto indices =
          context.upload(index_desc, payload(index_desc, positions)).value();
      const auto output =
          context.upload(output_desc, payload(output_desc, sentinels)).value();
      auto recorder = v::Recorder::create(context).value();
      ASSERT_TRUE(
          recorder.append(program, std::array{indices}, std::array{output})
              .ok());
      auto recorded = recorder.finish();
      ASSERT_TRUE(recorded) << recorded.error().message();
      auto completion = recorded->submit().value();
      const auto status = completion.wait();
      EXPECT_TRUE(status.IsOutOfRange()) << status.message();
      EXPECT_EQ(bits(context, output), sentinels);
      EXPECT_FALSE(recorded->submit());
      auto slot = program.create_invocation().value();
      ASSERT_TRUE(slot.bind(indices).ok());
      auto work = slot.submit().value();
      auto result = work.wait();
      ASSERT_FALSE(result);
      EXPECT_TRUE(result.error().IsOutOfRange());
      EXPECT_TRUE(slot.bind(indices).IsOutOfRange());
    }
    std::vector<std::uint64_t> positions(129, 4);
    const auto good =
        context.upload(index_desc, payload(index_desc, positions)).value();
    auto fresh = program.create_invocation().value();
    for (int repeat = 0; repeat < 2; ++repeat) {
      ASSERT_TRUE(fresh.bind(good).ok());
      auto work = fresh.submit().value();
      auto result = work.wait();
      ASSERT_TRUE(result) << result.error().message();
      EXPECT_EQ(bits(context, result->get()->tensors()[0]),
                reference(desc, host, positions, 0));
    }
  }
}

TEST(VulkanGather, ByteOriginsWorkForCapturedAndDynamicTablesAndIndices) {
  auto context = vulkan_test::context().value();
  for (const auto type : {c::DataType::kBFloat16, c::DataType::kInt64}) {
    const auto size = static_cast<std::int64_t>(c::data_type_size(type));
    const auto desc = dense(type, {5, 7});
    const auto table = payload(desc, raw_pattern(desc));
    const auto index_desc = dense(c::DataType::kInt64, {3});
    const std::vector<std::uint64_t> positions{4, 0, 2};
    for (const std::int64_t origin : {1, 3, 4, 7}) {
      const auto backing_desc =
          c::TensorDesc::strided_bytes(
              type, std::array<std::int64_t, 2>{5, 7 * size + origin},
              std::array<std::int64_t, 2>{7 * size, 1})
              .value();
      std::vector<std::byte> host(*backing_desc.required_bytes(),
                                  std::byte{0xcd});
      std::ranges::copy(table, host.begin() + origin);
      const auto backing = context.upload(backing_desc, host).value();
      const auto source =
          backing.view(c::View::slice(1, origin, 7, size).value()).value();
      ASSERT_EQ(source.description(), desc);
      ASSERT_EQ(source.native_offset(), static_cast<VkDeviceSize>(origin));
      const auto index_backing_desc =
          c::TensorDesc::strided_bytes(
              c::DataType::kInt64, std::array<std::int64_t, 1>{3 * 8 + origin},
              std::array<std::int64_t, 1>{1})
              .value();
      std::vector<std::byte> index_host(*index_backing_desc.required_bytes(),
                                        std::byte{0x5a});
      const auto index_payload = payload(index_desc, positions);
      std::ranges::copy(index_payload, index_host.begin() + origin);
      const auto index_backing =
          context.upload(index_backing_desc, index_host).value();
      const auto indices =
          index_backing.view(c::View::slice(0, origin, 3, 8).value()).value();
      for (bool captured : {false, true}) {
        const std::array arguments{captured ? v::Argument(source)
                                            : v::Argument(desc),
                                   v::Argument(index_desc)};
        auto program =
            v::prepare(context, c::Operation(c::Gather{0}), arguments);
        ASSERT_TRUE(program) << program.error().message();
        auto slot = program->create_invocation().value();
        const auto bound = captured ? slot.bind(indices)
                                    : slot.bind(std::array{source, indices});
        ASSERT_TRUE(bound.ok()) << bound.message();
        auto work = slot.submit().value();
        auto result = work.wait();
        ASSERT_TRUE(result) << result.error().message();
        EXPECT_EQ(bits(context, result->get()->tensors()[0]),
                  reference(desc, table, positions, 0));
      }
      EXPECT_EQ(bytes(context, backing), host);
      EXPECT_EQ(bytes(context, index_backing), index_host);
    }
  }
}

TEST(VulkanGather, EmptyIndicesAndEmptyResultStillRespectCheckedDomain) {
  auto context = vulkan_test::context().value();
  for (const auto &desc : {dense(c::DataType::kFloat32, {0, 7}),
                          dense(c::DataType::kFloat32, {3, 0})}) {
    const auto source = context.allocate(desc).value();
    const auto index_desc = dense(c::DataType::kInt64, {1});
    const auto bad =
        context
            .upload(index_desc,
                    payload(index_desc, std::array<std::uint64_t, 1>{3}))
            .value();
    auto failed =
        execute(context, c::Operation(c::Gather{0}), std::array{source, bad});
    ASSERT_FALSE(failed);
    EXPECT_TRUE(failed.error().IsOutOfRange());
    const auto empty =
        context.allocate(dense(c::DataType::kInt32, {0})).value();
    auto result =
        execute(context, c::Operation(c::Gather{0}), std::array{source, empty});
    ASSERT_TRUE(result) << result.error().message();
    EXPECT_EQ(*result->description().element_count(), 0U);
  }
}

TEST(VulkanGather, RecorderRetainsErrorsFromEveryAppendedCheckedProgram) {
  auto context = vulkan_test::context().value();
  const auto desc = dense(c::DataType::kInt32, {5, 7});
  const auto host = payload(desc, raw_pattern(desc));
  const auto table = context.upload(desc, host).value();
  const auto index_desc = dense(c::DataType::kInt32, {1});
  const auto bad =
      context
          .upload(index_desc,
                  payload(index_desc, std::array<std::uint64_t, 1>{5}))
          .value();
  const std::vector<std::uint64_t> positions{4};
  const auto good =
      context.upload(index_desc, payload(index_desc, positions)).value();
  auto program = v::prepare(context, c::Gather{0}, table, index_desc).value();
  const auto output_desc = program.output_descriptions()[0];
  const std::vector<std::uint64_t> sentinel(7, 0x55aa);
  const auto first =
      context.upload(output_desc, payload(output_desc, sentinel)).value();
  const auto second = context.allocate(output_desc).value();
  auto recorder = v::Recorder::create(context).value();
  ASSERT_TRUE(
      recorder.append(program, std::array{bad}, std::array{first}).ok());
  ASSERT_TRUE(
      recorder.append(program, std::array{good}, std::array{second}).ok());
  auto recorded = recorder.finish().value();
  auto completion = recorded.submit().value();
  EXPECT_TRUE(completion.wait().IsOutOfRange());
  EXPECT_EQ(bits(context, first), sentinel);
  EXPECT_EQ(bits(context, second), reference(desc, host, positions, 0));
}
} // namespace

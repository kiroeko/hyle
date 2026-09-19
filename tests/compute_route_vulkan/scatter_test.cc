#include <gtest/gtest.h>

#include <algorithm>
#include <optional>

#include "scatter_test_support.h"
#include "vulkan_program_internal.h"

namespace {
using namespace vulkan_scatter_test;

TEST(VulkanScatter, AssignPreservesAllStorageBitsAndLastLogicalUpdate) {
  auto context = vulkan_test::context().value();
  for (const auto type :
       {c::DataType::kFloat32, c::DataType::kFloat16, c::DataType::kBFloat16,
        c::DataType::kInt32, c::DataType::kInt64, c::DataType::kBool})
    for (const auto index_type : {c::DataType::kInt32, c::DataType::kInt64})
      for (const std::int64_t width : {1, 3, 129}) {
        const auto desc = dense(type, {7, width}),
                   update_desc = dense(type, {9, width});
        const auto index_desc = dense(index_type, {9});
        const std::vector<std::uint64_t> positions{3, 0, 3, 6, 1, 6, 3, 0, 2};
        const auto base = raw_pattern(desc), update = raw_pattern(update_desc);
        const auto source = context.upload(desc, payload(desc, base)).value();
        const auto indices =
            context.upload(index_desc, payload(index_desc, positions)).value();
        const auto updates =
            context.upload(update_desc, payload(update_desc, update)).value();
        for (const auto bounds : {c::IndexBoundsPolicy::kChecked,
                                  c::IndexBoundsPolicy::kUnchecked}) {
          SCOPED_TRACE(::testing::Message()
                       << static_cast<int>(type) << '/' << width);
          const c::Scatter operation{0, c::ScatterKind::kAssign, bounds};
          auto output = execute(context, c::Operation(operation),
                                std::array{source, indices, updates});
          ASSERT_TRUE(output) << output.error().message();
          expect_scatter(context, *output,
                         reference(desc, base, positions, update, operation),
                         false);
        }
        EXPECT_EQ(bits(context, source), base);
        EXPECT_EQ(bits(context, updates), update);
      }
}

TEST(VulkanScatter, AddKeepsBaseFirstAndRoundsEachHalfUpdate) {
  auto context = vulkan_test::context().value();
  const auto index_desc = dense(c::DataType::kInt32, {7});
  const std::vector<std::uint64_t> positions{3, 1, 3, 3, 2, 2, 4};
  const auto indices =
      context.upload(index_desc, payload(index_desc, positions)).value();
  for (const auto type :
       {c::DataType::kFloat32, c::DataType::kFloat16, c::DataType::kBFloat16})
    for (const std::int64_t width : {1, 3, 129}) {
      const auto desc = dense(type, {7, width});
      auto base = numeric(desc), update = numeric(desc, 1);
      const double large = type == c::DataType::kFloat32   ? 16777216
                           : type == c::DataType::kFloat16 ? 2048
                                                           : 256;
      for (std::size_t col = 0; col < static_cast<std::size_t>(width); ++col) {
        base[3 * width + col] = encode(large, type);
        update[col] = encode(1, type);
        update[2 * width + col] = encode(-large, type);
        update[3 * width + col] = encode(1, type);
      }
      update[4 * width] = encode(INFINITY, type);
      update[5 * width] = encode(-INFINITY, type);
      update[6 * width] = encode(NAN, type);
      const auto source = context.upload(desc, payload(desc, base)).value();
      const auto updates = context.upload(desc, payload(desc, update)).value();
      for (const auto bounds :
           {c::IndexBoundsPolicy::kChecked, c::IndexBoundsPolicy::kUnchecked}) {
        const c::Scatter operation{0, c::ScatterKind::kAdd, bounds};
        auto output = execute(context, c::Operation(operation),
                              std::array{source, indices, updates});
        ASSERT_TRUE(output) << output.error().message();
        expect_scatter(context, *output,
                       reference(desc, base, positions, update, operation),
                       true);
      }
    }
}

TEST(VulkanScatter, AxesRankEightMultidimensionalIndicesAndByteStrides) {
  auto context = vulkan_test::context().value();
  for (int layout = 0; layout < 4; ++layout)
    for (const bool adding : {false, true}) {
      const auto type = adding ? c::DataType::kBFloat16 : c::DataType::kInt64;
      const auto dense_base = layout == 3
                                  ? dense(type, {1, 2, 1, 3, 1, 5, 1, 3})
                                  : dense(type, {2, 5, 3});
      const auto base_desc =
          layout == 0 ? dense_base
                      : odd_layout(type, dense_base.dimensions(), layout == 2);
      const auto dense_indices = layout == 0 ? dense(c::DataType::kInt64, {})
                                 : layout == 3
                                     ? dense(c::DataType::kInt64, {4})
                                     : dense(c::DataType::kInt64, {2, 2});
      const auto index_desc =
          layout <= 1 ? dense_indices
                      : odd_layout(c::DataType::kInt64,
                                   dense_indices.dimensions(), true);
      const auto axis = layout == 3 ? 5 : 1;
      const c::Scatter operation{
          axis - static_cast<std::int32_t>(base_desc.rank()),
          adding ? c::ScatterKind::kAdd : c::ScatterKind::kAssign};
      const auto dense_updates =
          c::infer_single_output(c::Operation(c::Gather{axis}),
                                 std::array{base_desc, index_desc})
              .value();
      const auto update_desc =
          layout == 0
              ? dense_updates
              : odd_layout(type, dense_updates.dimensions(), layout == 2);
      const auto base_host = payload(
          base_desc, adding ? numeric(base_desc) : raw_pattern(base_desc));
      const auto update_host =
          payload(update_desc,
                  adding ? numeric(update_desc, 1) : raw_pattern(update_desc));
      const std::vector<std::uint64_t> positions(*index_desc.element_count(),
                                                 3);
      const auto index_host = payload(index_desc, positions);
      const auto base = context.upload(base_desc, base_host).value();
      const auto indices = context.upload(index_desc, index_host).value();
      const auto updates = context.upload(update_desc, update_host).value();
      auto output = execute(context, c::Operation(operation),
                            std::array{base, indices, updates});
      ASSERT_TRUE(output) << output.error().message();
      expect_scatter(context, *output,
                     reference(base_desc, logical(base_desc, base_host),
                               logical(index_desc, index_host),
                               logical(update_desc, update_host), operation),
                     adding);
      EXPECT_EQ(bytes(context, base), base_host);
      EXPECT_EQ(bytes(context, indices), index_host);
      EXPECT_EQ(bytes(context, updates), update_host);
    }
}

TEST(VulkanScatter, EmptyAndLongUnsortedIndexListsUseOriginalStableSort) {
  auto context = vulkan_test::context().value();
  const auto desc = dense(c::DataType::kFloat32, {37, 3});
  const auto base = numeric(desc);
  const auto source = context.upload(desc, payload(desc, base)).value();
  for (const std::int64_t count : {0, 1, 127, 128, 129, 1025, 4097}) {
    const auto index_desc = dense(c::DataType::kInt32, {count}),
               update_desc = dense(c::DataType::kFloat32, {count, 3});
    std::vector<std::uint64_t> positions(static_cast<std::size_t>(count));
    for (std::size_t i = 0; i < positions.size(); ++i)
      positions[i] = i * 7 % 37;
    const auto update = numeric(update_desc);
    const auto indices =
        context.upload(index_desc, payload(index_desc, positions)).value();
    const auto updates =
        context.upload(update_desc, payload(update_desc, update)).value();
    for (const bool adding : {false, true}) {
      const c::Scatter operation{0, adding ? c::ScatterKind::kAdd
                                           : c::ScatterKind::kAssign};
      auto program =
          v::prepare(context, operation, desc, index_desc, update_desc);
      ASSERT_TRUE(program) << program.error().message();
      if (count != 0) {
        const auto &state = v::detail::Access::program(*program);
        const auto found =
            std::ranges::find_if(state->dispatches, [](const auto &d) {
              return d.region != nullptr;
            });
        ASSERT_NE(found, state->dispatches.end());
        EXPECT_EQ(found->region->outputs.size(),
                  1U); // Only Argsort indices leave this region.
      }
      auto slot = program->create_invocation().value();
      ASSERT_TRUE(slot.bind(std::array{source, indices, updates}).ok());
      auto work = slot.submit().value();
      auto output = work.wait();
      ASSERT_TRUE(output) << output.error().message();
      expect_scatter(context, output->get()->tensors()[0],
                     reference(desc, base, positions, update, operation),
                     adding);
    }
  }
}

TEST(VulkanScatter, CheckedFailurePreservesAllOutputAndAllowsFreshSlot) {
  auto context = vulkan_test::context().value();
  const auto desc = dense(c::DataType::kFloat32, {5, 7}),
             update_desc = dense(c::DataType::kFloat32, {129, 7});
  const auto base = numeric(desc), update = numeric(update_desc);
  const auto source = context.upload(desc, payload(desc, base)).value();
  const auto updates =
      context.upload(update_desc, payload(update_desc, update)).value();
  for (const auto index_type : {c::DataType::kInt32, c::DataType::kInt64})
    for (const bool adding : {false, true}) {
      const auto index_desc = dense(index_type, {129});
      const c::Scatter operation{0, adding ? c::ScatterKind::kAdd
                                           : c::ScatterKind::kAssign};
      auto program =
          v::prepare(context, operation, source, index_desc, updates).value();
      for (const auto bad :
           {UINT64_MAX, std::uint64_t{5}, std::uint64_t{1} << 32}) {
        if (index_type == c::DataType::kInt32 &&
            bad == (std::uint64_t{1} << 32))
          continue;
        std::vector<std::uint64_t> positions(129, 2);
        positions.back() = bad;
        const auto indices =
            context.upload(index_desc, payload(index_desc, positions)).value();
        const std::vector<std::uint64_t> sentinel(35, 0x55aa55aa);
        const auto output =
            context.upload(desc, payload(desc, sentinel)).value();
        auto recorder = v::Recorder::create(context).value();
        ASSERT_TRUE(
            recorder.append(program, std::array{indices}, std::array{output})
                .ok());
        auto recorded = recorder.finish().value();
        auto completion = recorded.submit().value();
        EXPECT_TRUE(completion.wait().IsOutOfRange());
        EXPECT_EQ(bits(context, output), sentinel);
        EXPECT_FALSE(recorded.submit());
        auto slot = program.create_invocation().value();
        ASSERT_TRUE(slot.bind(indices).ok());
        auto work = slot.submit().value();
        auto failed = work.wait();
        ASSERT_FALSE(failed);
        EXPECT_TRUE(failed.error().IsOutOfRange());
        EXPECT_TRUE(slot.bind(indices).IsOutOfRange());
      }
      std::vector<std::uint64_t> positions(129, 3);
      const auto indices =
          context.upload(index_desc, payload(index_desc, positions)).value();
      auto fresh = program.create_invocation().value();
      ASSERT_TRUE(fresh.bind(indices).ok());
      auto work = fresh.submit().value();
      auto output = work.wait();
      ASSERT_TRUE(output) << output.error().message();
      expect_scatter(context, output->get()->tensors()[0],
                     reference(desc, base, positions, update, operation),
                     adding);
      auto recorder = v::Recorder::create(context).value();
      EXPECT_TRUE(
          recorder.append(program, std::array{indices}, std::array{source})
              .IsInvalidArgument());
    }
  for (const auto &empty_desc : {dense(c::DataType::kFloat32, {0, 3}),
                                 dense(c::DataType::kFloat32, {3, 0})}) {
    const auto index_desc = dense(c::DataType::kInt64, {1});
    const auto empty_updates =
        c::infer_single_output(c::Operation(c::Gather{0}),
                               std::array{empty_desc, index_desc})
            .value();
    const auto empty = context.allocate(empty_desc).value(),
               values = context.allocate(empty_updates).value();
    const auto bad =
        context
            .upload(index_desc,
                    payload(index_desc, std::array<std::uint64_t, 1>{3}))
            .value();
    for (const bool adding : {false, true}) {
      auto output =
          execute(context,
                  c::Operation(c::Scatter{0, adding ? c::ScatterKind::kAdd
                                                    : c::ScatterKind::kAssign}),
                  std::array{empty, bad, values});
      ASSERT_FALSE(output);
      EXPECT_TRUE(output.error().IsOutOfRange());
    }
  }
}

TEST(VulkanScatter, ByteOriginsIndependentReplaysAndRetainedResults) {
  auto context = vulkan_test::context().value();
  const auto desc = dense(c::DataType::kFloat32, {7, 3}),
             index_desc = dense(c::DataType::kInt64, {5}),
             update_desc = dense(c::DataType::kFloat32, {5, 3});
  for (const bool captured : {false, true})
    for (const bool adding : {false, true}) {
      const c::Scatter operation{0, adding ? c::ScatterKind::kAdd
                                           : c::ScatterKind::kAssign};
      std::optional<v::Program> program;
      std::optional<v::Invocation> first, second;
      v::Tensor retained;
      std::vector<std::uint64_t> saved;
      for (int phase = 0; phase < 5; ++phase) {
        const auto base = numeric(desc, phase),
                   update = numeric(update_desc, phase + 1);
        const std::vector<std::uint64_t> positions{
            3, static_cast<std::uint64_t>(phase), 3, 1, 6};
        auto source = with_origin(context, desc, base, phase);
        auto indices = with_origin(context, index_desc, positions, phase);
        auto updates = with_origin(context, update_desc, update, phase);
        if (!program || captured) {
          program = v::prepare(context, operation,
                               captured ? v::Argument(source.tensor)
                                        : v::Argument(desc),
                               index_desc, update_desc)
                        .value();
          first = program->create_invocation().value();
          second = program->create_invocation().value();
        }
        auto &slot = phase == 0 ? *first : *second;
        const auto bound =
            captured ? slot.bind(std::array{indices.tensor, updates.tensor})
                     : slot.bind(std::array{source.tensor, indices.tensor,
                                            updates.tensor});
        ASSERT_TRUE(bound.ok()) << bound.message();
        auto work = slot.submit().value();
        auto output = work.wait();
        ASSERT_TRUE(output) << output.error().message();
        expect_scatter(context, output->get()->tensors()[0],
                       reference(desc, base, positions, update, operation),
                       adding);
        if (phase == 0) {
          retained = output->get()->tensors()[0];
          saved = bits(context, retained);
        } else
          EXPECT_EQ(bits(context, retained), saved);
        EXPECT_EQ(bytes(context, source.backing), source.host);
        EXPECT_EQ(bytes(context, indices.backing), indices.host);
        EXPECT_EQ(bytes(context, updates.backing), updates.host);
      }
    }
  EXPECT_TRUE(context.close().ok());
}

TEST(VulkanScatter, ColdMetadataAndNativeLimitsRejectWithoutSubmission) {
  auto context = vulkan_test::context().value();
  const auto desc = dense(c::DataType::kFloat32, {129}),
             index_desc = dense(c::DataType::kInt32, {129});
  auto &limits = v::detail::Access::runtime(context)->device->properties.limits;
  const auto saved = limits;
  for (int mode = 0; mode < 4; ++mode) {
    limits = saved;
    if (mode == 0)
      limits.maxStorageBufferRange = 600;
    else if (mode == 1)
      limits.maxComputeWorkGroupInvocations = 64;
    else if (mode == 2)
      limits.maxComputeSharedMemorySize = 16;
    else
      limits.maxComputeWorkGroupCount[0] = 0;
    for (const bool adding : {false, true}) {
      auto rejected =
          v::prepare(context,
                     c::Scatter{0, adding ? c::ScatterKind::kAdd
                                          : c::ScatterKind::kAssign},
                     desc, index_desc, desc);
      EXPECT_FALSE(rejected);
      if (!rejected)
        EXPECT_TRUE(rejected.error().IsUnimplemented() ||
                    rejected.error().IsOutOfRange());
    }
  }
  limits = saved;
  for (const auto type :
       {c::DataType::kInt32, c::DataType::kInt64, c::DataType::kBool}) {
    const auto integer = dense(type, {129});
    auto rejected = v::prepare(context, c::Scatter{0, c::ScatterKind::kAdd},
                               integer, index_desc, integer);
    ASSERT_FALSE(rejected);
    EXPECT_TRUE(rejected.error().IsUnimplemented());
  }
}

TEST(VulkanScatter, QuotaFailureRollsBackAndCloseKeepsCompletedOutputs) {
  constexpr std::size_t limit = 64 * 1024 * 1024;
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), limit);
  {
    auto context = vulkan_test::context(&quota).value();
    const auto desc = dense(c::DataType::kFloat32, {7, 129}),
               index_desc = dense(c::DataType::kInt32, {7});
    const auto base = numeric(desc), update = numeric(desc, 1);
    const std::vector<std::uint64_t> positions{3, 0, 3, 1, 6, 3, 6};
    const auto source = context.upload(desc, payload(desc, base)).value();
    const auto indices =
        context.upload(index_desc, payload(index_desc, positions)).value();
    const auto updates = context.upload(desc, payload(desc, update)).value();
    std::vector<v::Tensor> retained_outputs;
    std::vector<std::vector<std::uint64_t>> expected_outputs;
    for (const bool adding : {false, true}) {
      const c::Scatter operation{0, adding ? c::ScatterKind::kAdd
                                           : c::ScatterKind::kAssign};
      auto program =
          v::prepare(context, operation, source, index_desc, updates).value();
      const auto before = quota.used();
      auto reservation = quota.try_reserve_bytes(limit - before);
      ASSERT_TRUE(reservation);
      auto rejected = program.create_invocation();
      ASSERT_FALSE(rejected);
      EXPECT_TRUE(rejected.error().IsResourceExhausted());
      reservation.reset();
      EXPECT_EQ(quota.used(), before);
      auto slot = program.create_invocation().value();
      program = {};
      ASSERT_TRUE(slot.bind(indices).ok());
      std::size_t warmed = 0;
      v::Tensor retained;
      for (int phase = 0; phase < 5; ++phase) {
        auto work = slot.submit().value();
        auto output = work.wait();
        ASSERT_TRUE(output) << output.error().message();
        if (phase == 0)
          warmed = quota.used();
        else
          EXPECT_EQ(quota.used(), warmed);
        expect_scatter(context, output->get()->tensors()[0],
                       reference(desc, base, positions, update, operation),
                       adding);
        if (phase == 4)
          retained = output->get()->tensors()[0];
      }
      expect_scatter(context, retained,
                     reference(desc, base, positions, update, operation),
                     adding);
      retained_outputs.push_back(retained);
      expected_outputs.push_back(
          reference(desc, base, positions, update, operation));
    }
    EXPECT_TRUE(context.close().ok());
    for (std::size_t i = 0; i < retained_outputs.size(); ++i)
      expect_scatter(context, retained_outputs[i], expected_outputs[i], i != 0);
  }
  EXPECT_EQ(quota.used(), 0U);
}
} // namespace

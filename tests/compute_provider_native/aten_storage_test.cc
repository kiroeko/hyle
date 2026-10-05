#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/native.h"
#include "hyle/mem/stats.h"

namespace {
using namespace hyle::compute;
using hyle::mem::Buffer;
using Bytes = std::vector<std::byte>;

TensorDesc dense(DataType type, std::span<const std::int64_t> shape) {
  return TensorDesc::dense(type, shape).value();
}
TensorDesc odd(DataType type, std::span<const std::int64_t> shape) {
  std::vector<std::int64_t> strides(shape.size());
  auto stride = static_cast<std::int64_t>(data_type_size(type)) + 1;
  for (std::size_t reverse = 0; reverse < shape.size(); ++reverse) {
    const auto axis = shape.size() - 1 - reverse;
    strides[axis] = stride;
    stride = stride * shape[axis] + 1;
  }
  return TensorDesc::strided_bytes(type, shape, strides).value();
}
std::size_t offset(const TensorDesc &desc, std::size_t linear) {
  std::size_t result = 0;
  for (std::size_t reverse = 0; reverse < desc.rank(); ++reverse) {
    const auto axis = desc.rank() - 1 - reverse;
    const auto extent = static_cast<std::size_t>(desc.dimensions()[axis]);
    result +=
        linear % extent * static_cast<std::size_t>(desc.byte_strides()[axis]);
    linear /= extent;
  }
  return result;
}
Bytes logical(const TensorDesc &desc, int seed) {
  Bytes result(desc.element_count().value() * data_type_size(desc.data_type()));
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = std::byte((i * 71 + seed * 37) & 255);
  return result;
}
Bytes physical(const TensorDesc &desc, const Bytes &values) {
  Bytes result(desc.required_bytes().value(), std::byte{0xda});
  const auto width = data_type_size(desc.data_type());
  for (std::size_t i = 0; i < desc.element_count().value(); ++i)
    std::memcpy(result.data() + offset(desc, i), values.data() + i * width,
                width);
  return result;
}
Buffer buffer(const TensorDesc &desc) {
  return Buffer::create(hyle::mem::system_resource(),
                        desc.required_bytes().value(), 64)
      .value();
}
Bytes read(const Buffer &buffer) {
  Bytes values(buffer.size());
  EXPECT_TRUE(buffer.view().read(values).ok());
  return values;
}
Bytes index_bytes(const TensorDesc &desc, std::span<const std::int64_t> ids) {
  const auto width = data_type_size(desc.data_type());
  Bytes values(ids.size() * width);
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (width == 4) {
      const auto value = static_cast<std::int32_t>(ids[i]);
      std::memcpy(values.data() + i * width, &value, width);
    } else
      std::memcpy(values.data() + i * width, &ids[i], width);
  }
  return physical(desc, values);
}

template <class Fill, class Expected>
void replays(Context &context, Queue &queue, Operation op,
             const std::vector<TensorDesc> &descs, Fill fill,
             Expected expected) {
  const auto out_desc = infer_single_output(op, descs).value();
  auto prepared = prepare(context, op, descs, std::span(&out_desc, 1),
                          hyle::mem::system_resource());
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, aten_cpu::kProviderId);
  EXPECT_EQ(prepared->runtime_properties().replay_class,
            ProviderReplayClass::kManaged);
  auto invocation = prepared->create_invocation();
  ASSERT_TRUE(invocation) << invocation.error().message();
  std::vector<Buffer> buffers;
  std::vector<ConstTensorView> inputs;
  for (const auto &desc : descs) {
    buffers.push_back(buffer(desc));
    inputs.push_back(
        ConstTensorView::create(desc, buffers.back().view()).value());
  }
  auto output = buffer(out_desc);
  const auto destination = TensorView::create(out_desc, output.view()).value();
  for (int seed : {1, 7}) {
    fill(buffers, seed);
    auto done = invocation->submit(queue, inputs, std::span(&destination, 1));
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    EXPECT_EQ(read(output), expected(seed));
  }
}

TEST(AtenCpuStorage, FactoryMatchesLockedSdkAvailability) {
#if defined(HYLE_TEST_TORCH_CPU)
  EXPECT_TRUE(aten_cpu::provider().valid());
  EXPECT_EQ(aten_cpu::provider().id(), aten_cpu::kProviderId);
#else
  EXPECT_FALSE(aten_cpu::provider().valid());
#endif
}

TEST(AtenCpuStorage,
     SelectCopyConcatAndFillPreserveEveryScalarBitWithOddStrides) {
  if (!aten_cpu::provider())
    GTEST_SKIP() << "locked Torch CPU SDK disabled";
  auto context = Context::create(cpu::device(), aten_cpu::provider()).value();
  auto queue = context.create_queue().value();
  const std::array<std::int64_t, 2> shape{2, 3}, cond_shape{1, 3};
  for (int raw = int(DataType::kBool); raw <= int(DataType::kUInt64); ++raw) {
    const auto type = static_cast<DataType>(raw);
    const auto width = data_type_size(type);
    SCOPED_TRACE(raw);
    const auto branch = odd(type, shape),
               condition = odd(DataType::kBool, cond_shape);
    replays(
        context, queue, Operation(Select{}), {condition, branch, branch},
        [&](auto &buffers, int seed) {
          const Bytes truth{std::byte{0}, std::byte{0x80}, std::byte{0xff}};
          ASSERT_TRUE(buffers[0].view().write(physical(condition, truth)).ok());
          ASSERT_TRUE(buffers[1]
                          .view()
                          .write(physical(branch, logical(branch, seed)))
                          .ok());
          ASSERT_TRUE(buffers[2]
                          .view()
                          .write(physical(branch, logical(branch, seed + 3)))
                          .ok());
        },
        [&](int seed) {
          auto a = logical(branch, seed), b = logical(branch, seed + 3);
          for (std::size_t element = 0; element < 6; ++element)
            if (element % 3 == 0)
              std::memcpy(a.data() + element * width,
                          b.data() + element * width, width);
          return a;
        });
    replays(
        context, queue, Operation(Copy{}), {branch},
        [&](auto &buffers, int seed) {
          ASSERT_TRUE(buffers[0]
                          .view()
                          .write(physical(branch, logical(branch, seed)))
                          .ok());
        },
        [&](int seed) { return logical(branch, seed); });
    const auto empty = dense(type, std::array<std::int64_t, 2>{2, 0});
    replays(
        context, queue, Operation(Concat{.axis = -1}), {branch, empty, branch},
        [&](auto &buffers, int seed) {
          ASSERT_TRUE(buffers[0]
                          .view()
                          .write(physical(branch, logical(branch, seed)))
                          .ok());
          ASSERT_TRUE(buffers[2]
                          .view()
                          .write(physical(branch, logical(branch, seed + 3)))
                          .ok());
        },
        [&](int seed) {
          const auto a = logical(branch, seed), b = logical(branch, seed + 3);
          Bytes result(12 * width);
          for (std::size_t row = 0; row < 2; ++row) {
            std::memcpy(result.data() + row * 6 * width,
                        a.data() + row * 3 * width, 3 * width);
            std::memcpy(result.data() + (row * 6 + 3) * width,
                        b.data() + row * 3 * width, 3 * width);
          }
          return result;
        });
    for (const auto &fill_shape :
         {std::vector<std::int64_t>{}, {1, 1, 1, 1, 1, 1, 2, 3}}) {
      const auto bits = type == DataType::kBool
                            ? std::uint64_t{1}
                            : std::uint64_t{0x7ff8a5038001d7ffULL};
      const auto op =
          Fill{.output_shape = TensorShape::create(fill_shape).value(),
               .output_type = type,
               .value = ScalarValue::from_bits(type, bits)};
      replays(
          context, queue, Operation(op), {}, [](auto &, int) {},
          [&](int) {
            Bytes result(dense(type, fill_shape).required_bytes().value());
            const auto scalar = std::as_bytes(std::span(&bits, 1)).first(width);
            for (std::size_t i = 0; i < result.size(); i += width)
              std::memcpy(result.data() + i, scalar.data(), width);
            return result;
          });
    }
  }
  EXPECT_TRUE(context.close().ok());
}

TEST(AtenCpuStorage,
     GatherAndAssignCoverBothIndexWidthsEveryAxisAndLastLogicalWrite) {
  if (!aten_cpu::provider())
    GTEST_SKIP() << "locked Torch CPU SDK disabled";
  auto context = Context::create(cpu::device(), aten_cpu::provider()).value();
  auto queue = context.create_queue().value();
  const std::array<std::int64_t, 3> shape{2, 3, 2};
  for (int raw = int(DataType::kBool); raw <= int(DataType::kUInt64); ++raw)
    for (std::size_t axis = 0; axis < 3; ++axis)
      for (const auto index_type : {DataType::kInt32, DataType::kInt64})
        for (const auto &index_shape :
             {std::vector<std::int64_t>{}, {4}, {2, 2}}) {
          SCOPED_TRACE(raw);
          SCOPED_TRACE(axis);
          SCOPED_TRACE(int(index_type));
          SCOPED_TRACE(index_shape.size());
          const auto type = static_cast<DataType>(raw);
          const auto width = data_type_size(type);
          const auto table = odd(type, shape),
                     index = odd(index_type, index_shape);
          const auto gather =
              Operation(Gather{.axis = static_cast<std::int32_t>(axis)});
          const auto update_dense =
              infer_single_output(gather, std::array{table, index}).value();
          const auto updates = odd(type, update_dense.dimensions());
          const auto count = index.element_count().value();
          std::vector<std::int64_t> ids(count);
          for (std::size_t i = 0; i < count; ++i)
            ids[i] = i % 2 == 0 ? shape[axis] - 1 : 0;
          std::size_t prefix = 1, suffix = 1;
          for (std::size_t i = 0; i < axis; ++i)
            prefix *= static_cast<std::size_t>(shape[i]);
          for (std::size_t i = axis + 1; i < 3; ++i)
            suffix *= static_cast<std::size_t>(shape[i]);
          replays(
              context, queue, gather, {table, index},
              [&](auto &buffers, int seed) {
                ASSERT_TRUE(buffers[0]
                                .view()
                                .write(physical(table, logical(table, seed)))
                                .ok());
                ASSERT_TRUE(
                    buffers[1].view().write(index_bytes(index, ids)).ok());
              },
              [&](int seed) {
                const auto source = logical(table, seed);
                Bytes result(prefix * count * suffix * width);
                for (std::size_t outer = 0; outer < prefix; ++outer)
                  for (std::size_t i = 0; i < count; ++i)
                    std::memcpy(result.data() +
                                    (outer * count + i) * suffix * width,
                                source.data() + (outer * shape[axis] + ids[i]) *
                                                    suffix * width,
                                suffix * width);
                return result;
              });
          replays(
              context, queue,
              Operation(Scatter{.axis = static_cast<std::int32_t>(axis)}),
              {table, index, updates},
              [&](auto &buffers, int seed) {
                ASSERT_TRUE(buffers[0]
                                .view()
                                .write(physical(table, logical(table, seed)))
                                .ok());
                ASSERT_TRUE(
                    buffers[1].view().write(index_bytes(index, ids)).ok());
                ASSERT_TRUE(
                    buffers[2]
                        .view()
                        .write(physical(updates, logical(updates, seed + 3)))
                        .ok());
              },
              [&](int seed) {
                auto result = logical(table, seed);
                const auto values = logical(updates, seed + 3);
                for (std::size_t outer = 0; outer < prefix; ++outer)
                  for (std::size_t i = 0; i < count; ++i)
                    std::memcpy(result.data() + (outer * shape[axis] + ids[i]) *
                                                    suffix * width,
                                values.data() +
                                    (outer * count + i) * suffix * width,
                                suffix * width);
                return result;
              });
        }
  EXPECT_TRUE(context.close().ok());
}

TEST(AtenCpuStorage,
     CheckedLateIndexFailurePreservesOutputAndReplaysAfterRecovery) {
  if (!aten_cpu::provider())
    GTEST_SKIP() << "locked Torch CPU SDK disabled";
  auto context = Context::create(cpu::device(), aten_cpu::provider()).value();
  auto queue = context.create_queue().value();
  const auto table =
      dense(DataType::kUInt64, std::array<std::int64_t, 2>{3, 2});
  const auto index = odd(DataType::kInt64, std::array<std::int64_t, 2>{1, 3});
  const auto updates =
      infer_single_output(Operation(Gather{}), std::array{table, index})
          .value();
  for (bool assign : {false, true}) {
    const auto op = assign ? Operation(Scatter{}) : Operation(Gather{});
    std::vector<TensorDesc> descs{table, index};
    if (assign)
      descs.push_back(updates);
    const auto out_desc = infer_single_output(op, descs).value();
    auto prepared = prepare(context, op, descs, std::span(&out_desc, 1),
                            hyle::mem::system_resource())
                        .value();
    auto invocation = prepared.create_invocation().value();
    std::vector<Buffer> buffers;
    std::vector<ConstTensorView> inputs;
    for (const auto &desc : descs) {
      buffers.push_back(buffer(desc));
      inputs.push_back(
          ConstTensorView::create(desc, buffers.back().view()).value());
      ASSERT_TRUE(
          buffers.back().view().write(physical(desc, logical(desc, 3))).ok());
    }
    auto output = buffer(out_desc);
    const auto destination =
        TensorView::create(out_desc, output.view()).value();
    const Bytes poison(output.size(), std::byte{0xa5});
    for (const auto invalid : {std::int64_t{-1}, std::int64_t{3}, INT64_MAX}) {
      const std::array<std::int64_t, 3> ids{2, 0, invalid};
      ASSERT_TRUE(buffers[1].view().write(index_bytes(index, ids)).ok());
      ASSERT_TRUE(output.view().write(poison).ok());
      auto done = invocation.submit(queue, inputs, std::span(&destination, 1));
      ASSERT_TRUE(done) << done.error().message();
      EXPECT_TRUE(done->wait().IsOutOfRange());
      EXPECT_EQ(read(output), poison);
      auto stale = invocation.submit(queue, inputs, std::span(&destination, 1));
      ASSERT_FALSE(stale);
      EXPECT_TRUE(stale.error().IsFailedPrecondition());
      invocation = prepared.create_invocation().value();
    }
    const std::array<std::int64_t, 3> ids{2, 0, 2};
    ASSERT_TRUE(buffers[1].view().write(index_bytes(index, ids)).ok());
    auto recovered =
        invocation.submit(queue, inputs, std::span(&destination, 1));
    ASSERT_TRUE(recovered);
    EXPECT_TRUE(recovered->wait().ok());
    EXPECT_NE(read(output), poison);
  }
}

TEST(AtenCpuStorage,
     EmptyIndicesAndRankEightByteCarriersKeepPublicDescriptions) {
  if (!aten_cpu::provider())
    GTEST_SKIP() << "locked Torch CPU SDK disabled";
  auto context = Context::create(cpu::device(), aten_cpu::provider()).value();
  auto queue = context.create_queue().value();
  const auto table = odd(DataType::kUInt64,
                         std::array<std::int64_t, 8>{1, 1, 1, 1, 1, 1, 2, 1});
  const auto index = dense(DataType::kInt64, std::array<std::int64_t, 1>{1});
  replays(
      context, queue, Operation(Gather{.axis = -2}), {table, index},
      [&](auto &buffers, int seed) {
        ASSERT_TRUE(buffers[0]
                        .view()
                        .write(physical(table, logical(table, seed)))
                        .ok());
        ASSERT_TRUE(
            buffers[1]
                .view()
                .write(index_bytes(index, std::array<std::int64_t, 1>{1}))
                .ok());
      },
      [&](int seed) {
        const auto source = logical(table, seed);
        return Bytes(source.begin() + 8, source.end());
      });
  const auto empty = dense(DataType::kInt64, std::array<std::int64_t, 1>{0});
  const auto updates = infer_single_output(Operation(Gather{.axis = -2}),
                                           std::array{table, empty})
                           .value();
  replays(
      context, queue, Operation(Scatter{.axis = -2}), {table, empty, updates},
      [&](auto &buffers, int seed) {
        ASSERT_TRUE(buffers[0]
                        .view()
                        .write(physical(table, logical(table, seed)))
                        .ok());
      },
      [&](int seed) { return logical(table, seed); });
  replays(
      context, queue, Operation(Gather{.axis = -2}), {table, empty},
      [](auto &, int) {}, [](int) { return Bytes{}; });
  const auto zero =
      dense(DataType::kFloat64, std::array<std::int64_t, 2>{2, 0});
  replays(
      context, queue, Operation(Concat{.axis = 1}), {zero, zero},
      [](auto &, int) {}, [](int) { return Bytes{}; });
  // Variadic inputs have no adapter-imposed small fixed arity.
  const auto scalar = dense(DataType::kUInt8, std::array<std::int64_t, 1>{1});
  replays(
      context, queue, Operation(Concat{}), std::vector<TensorDesc>(33, scalar),
      [](auto &buffers, int seed) {
        for (auto &b : buffers)
          ASSERT_TRUE(b.view().write(Bytes{std::byte(seed)}).ok());
      },
      [](int seed) { return Bytes(33, std::byte(seed)); });
}

TEST(AtenCpuStorage, CopyRejectsAliasWhileAssignAcceptsExactInputZero) {
  if (!aten_cpu::provider())
    GTEST_SKIP() << "locked Torch CPU SDK disabled";
  auto context = Context::create(cpu::device(), aten_cpu::provider()).value();
  auto queue = context.create_queue().value();
  const auto table =
      dense(DataType::kFloat64, std::array<std::int64_t, 2>{3, 2});
  auto storage = buffer(table);
  ASSERT_TRUE(storage.view().write(logical(table, 1)).ok());
  const auto alias = TensorView::create(table, storage.view()).value();
  auto copied = prepare(context, Copy{}, table).value();
  EXPECT_FALSE(copied.alias_support(0, 0)->supported);
  auto ci = copied.create_invocation().value();
  auto denied = ci.submit(queue, alias.as_const(), alias);
  ASSERT_FALSE(denied);
  EXPECT_TRUE(denied.error().IsInvalidArgument());
  EXPECT_EQ(read(storage), logical(table, 1));
  const auto index = dense(DataType::kInt32, std::array<std::int64_t, 1>{3});
  const auto updates =
      infer_single_output(Operation(Gather{}), std::array{table, index})
          .value();
  auto ib = buffer(index), ub = buffer(updates);
  ASSERT_TRUE(
      ib.view()
          .write(index_bytes(index, std::array<std::int64_t, 3>{2, 0, 2}))
          .ok());
  ASSERT_TRUE(ub.view().write(logical(updates, 7)).ok());
  auto assigned = prepare(context, Scatter{}, table, index, updates).value();
  EXPECT_TRUE(assigned.alias_support(0, 0)->supported);
  EXPECT_FALSE(assigned.alias_support(0, 2)->supported);
  const std::array inputs{alias.as_const(),
                          ConstTensorView::create(index, ib.view()).value(),
                          ConstTensorView::create(updates, ub.view()).value()};
  auto ai = assigned.create_invocation().value();
  ASSERT_TRUE(ai.submit(queue, inputs, alias)->wait().ok());
  auto expected = logical(table, 1);
  const auto values = logical(updates, 7);
  std::memcpy(expected.data(), values.data() + 16, 16);
  std::memcpy(expected.data() + 32, values.data() + 32, 16);
  EXPECT_EQ(read(storage), expected);
}

TEST(AtenCpuStorage, PrivateInvocationsAndCloseReleaseTrackedMetadata) {
  if (!aten_cpu::provider())
    GTEST_SKIP() << "locked Torch CPU SDK disabled";
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  {
    const std::array providers{ProviderBinding{aten_cpu::provider(), 100}};
    auto context =
        Context::create(cpu::device(), {.metadata_allocator = stats.get(),
                                        .providers = providers})
            .value();
    auto queue = context.create_queue().value();
    const auto table = dense(DataType::kInt32, std::array<std::int64_t, 1>{3});
    const auto index = dense(DataType::kInt64, std::array<std::int64_t, 1>{2});
    auto prepared = prepare(context, Gather{}, table, index).value();
    auto a = prepared.create_invocation().value(),
         b = prepared.create_invocation().value();
    auto source = buffer(table), ia = buffer(index), ib = buffer(index);
    const auto od =
        infer_single_output(Operation(Gather{}), std::array{table, index})
            .value();
    auto oa = buffer(od), ob = buffer(od);
    ASSERT_TRUE(source.view().write(logical(table, 1)).ok());
    ASSERT_TRUE(
        ia.view()
            .write(index_bytes(index, std::array<std::int64_t, 2>{2, 0}))
            .ok());
    ASSERT_TRUE(
        ib.view()
            .write(index_bytes(index, std::array<std::int64_t, 2>{1, 1}))
            .ok());
    const auto sv = ConstTensorView::create(table, source.view()).value();
    for (int round = 0; round < 3; ++round) {
      ASSERT_TRUE(
          a.submit(queue,
                   std::array{
                       sv, ConstTensorView::create(index, ia.view()).value()},
                   TensorView::create(od, oa.view()).value())
              ->wait()
              .ok());
      ASSERT_TRUE(
          b.submit(queue,
                   std::array{
                       sv, ConstTensorView::create(index, ib.view()).value()},
                   TensorView::create(od, ob.view()).value())
              ->wait()
              .ok());
      EXPECT_NE(read(oa), read(ob));
    }
    auto second_queue = context.create_queue().value();
    const auto expected_a = read(oa), expected_b = read(ob);
    std::atomic<bool> good_a{false}, good_b{false};
    auto concurrent = [&](auto &invocation, auto &own_queue, const auto &ids,
                          auto &output, const auto &expected, auto &good) {
      const std::array inputs{
          sv, ConstTensorView::create(index, ids.view()).value()};
      const auto destination = TensorView::create(od, output.view()).value();
      for (int round = 0; round < 12; ++round) {
        auto done = invocation.submit(own_queue, inputs, destination);
        if (!done || !done->wait().ok())
          return;
        Bytes actual(output.size());
        if (!output.view().read(actual).ok() || actual != expected)
          return;
      }
      good.store(true);
    };
    std::thread first(
        [&] { concurrent(a, queue, ia, oa, expected_a, good_a); });
    std::thread second(
        [&] { concurrent(b, second_queue, ib, ob, expected_b, good_b); });
    first.join();
    second.join();
    EXPECT_TRUE(good_a.load());
    EXPECT_TRUE(good_b.load());
    EXPECT_TRUE(context.close().ok());
    auto rejected = a.submit(
        queue,
        std::array{sv, ConstTensorView::create(index, ia.view()).value()},
        TensorView::create(od, oa.view()).value());
    EXPECT_FALSE(rejected);
  }
  EXPECT_TRUE(stats.balanced());
}

struct FaultAllocator final {
  hyle::mem::AllocatorRef backing{};
  std::size_t attempts = 0, fail = SIZE_MAX;
  hyle::mem::AllocatorRef get() noexcept {
    return {
        [](void *self, std::size_t bytes,
           std::size_t alignment) noexcept -> void * {
          auto &owner = *static_cast<FaultAllocator *>(self);
          return owner.attempts++ == owner.fail
                     ? nullptr
                     : owner.backing.allocate(bytes, alignment);
        },
        [](void *self, void *data, std::size_t bytes,
           std::size_t alignment) noexcept {
          static_cast<FaultAllocator *>(self)->backing.deallocate(data, bytes,
                                                                  alignment);
        },
        [](void *self, void *data, std::size_t old_bytes, std::size_t new_bytes,
           std::size_t alignment) noexcept -> void * {
          auto &owner = *static_cast<FaultAllocator *>(self);
          if (new_bytes != 0 && owner.attempts++ == owner.fail)
            return nullptr;
          return owner.backing.reallocate(data, old_bytes, new_bytes,
                                          alignment);
        },
        this};
  }
};
TEST(AtenCpuStorage, EveryTrackedColdAllocationFailureIsRecoverable) {
  if (!aten_cpu::provider())
    GTEST_SKIP() << "locked Torch CPU SDK disabled";
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  FaultAllocator owner{.backing = stats.get()};
  {
    const std::array providers{ProviderBinding{aten_cpu::provider(), 100}};
    auto context =
        Context::create(cpu::device(), {.metadata_allocator = owner.get(),
                                        .providers = providers})
            .value();
    const auto table =
        dense(DataType::kFloat64, std::array<std::int64_t, 2>{3, 2});
    const auto index = odd(DataType::kInt64, std::array<std::int64_t, 1>{3});
    const auto start = owner.attempts;
    {
      auto prepared = prepare(context, Gather{}, table, index);
      ASSERT_TRUE(prepared);
    }
    const auto allocations = owner.attempts - start;
    ASSERT_GT(allocations, 0U);
    for (std::size_t i = 0; i < allocations; ++i) {
      SCOPED_TRACE(i);
      owner.fail = owner.attempts + i;
      auto rejected = prepare(context, Gather{}, table, index);
      ASSERT_FALSE(rejected);
      EXPECT_TRUE(rejected.error().IsResourceExhausted());
      owner.fail = SIZE_MAX;
    }
    auto prepared = prepare(context, Gather{}, table, index).value();
    const auto before = owner.attempts;
    {
      auto invocation = prepared.create_invocation();
      ASSERT_TRUE(invocation);
    }
    const auto inv_allocations = owner.attempts - before;
    ASSERT_GT(inv_allocations, 0U);
    for (std::size_t i = 0; i < inv_allocations; ++i) {
      SCOPED_TRACE(i);
      owner.fail = owner.attempts + i;
      auto rejected = prepared.create_invocation();
      ASSERT_FALSE(rejected);
      EXPECT_TRUE(rejected.error().IsResourceExhausted());
      owner.fail = SIZE_MAX;
    }
    EXPECT_TRUE(prepared.create_invocation());
  }
  EXPECT_TRUE(stats.balanced());
}
} // namespace

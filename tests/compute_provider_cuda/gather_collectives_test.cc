#include "collectives_test_support.h"

#include <atomic>
#include <chrono>
#include <thread>

namespace {
using namespace cccl_test;
Context context() {
  return Context::create(cuda::device(0).value(), cuda_native::cccl_provider())
      .value();
}
template <class Bits, class Index> void matrix(Context &owner, DataType dtype) {
  for (std::int32_t axis : {0, 1, -1})
    for (int source_mode : {0, 1})
      for (int index_mode : {0, 1})
        gather<Bits, Index>(owner, dtype, std::array<std::int64_t, 3>{2, 7, 3},
                            axis, std::array<std::int64_t, 2>{2, 3},
                            source_mode, index_mode);
}
template <class Bits> void both_indices(Context &owner, DataType dtype) {
  matrix<Bits, std::int32_t>(owner, dtype);
  matrix<Bits, std::int64_t>(owner, dtype);
}

template <class Bits, class Index>
void wide_dense_rows(Context &owner, DataType dtype, std::int64_t columns,
                     bool source_offset, bool output_offset, int index_mode,
                     IndexBoundsPolicy bounds) {
  SCOPED_TRACE(::testing::Message() << sizeof(Bits) << '/' << sizeof(Index)
      << '/' << columns << '/' << source_offset << '/' << output_offset
      << '/' << index_mode << '/' << static_cast<int>(bounds));
  auto memory = cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto queue = owner.create_queue().value();
  const auto desc = TensorDesc::dense(dtype,
      std::array<std::int64_t, 3>{2, 3, columns}).value();
  const auto idesc = description(sizeof(Index) == 8 ? DataType::kInt64 : DataType::kInt32,
      std::array<std::int64_t, 1>{3}, 0, index_mode);
  Tensor source(desc, memory.ref(), source_offset), indices(idesc, memory.ref(), true);
  std::vector<Bits> data(desc.element_count().value());
  for (std::size_t i = 0; i < data.size(); ++i)
    data[i] = static_cast<Bits>(0x7fc000018123ULL + i * 0x9e3779b1ULL);
  const std::array<Index, 3> selected{2, 0, 2}; // Repeated source rows are legal.
  source.write<Bits>(data);
  indices.write<Index>(selected);
  auto prepared = prepare(owner, Gather{.axis = 1, .bounds = bounds},
      desc, idesc, memory.ref());
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->workspace_requirement().size, 0u);
  Tensor output(prepared->output_description().value(), memory.ref(), output_offset);
  auto slot = prepared->create_invocation().value();
  auto completion = slot.submit(queue, source.view.as_const(),
      indices.view.as_const(), output.view);
  ASSERT_TRUE(completion) << completion.error().message();
  ASSERT_TRUE(completion->wait().ok());
  std::vector<Bits> expected(data.size());
  for (std::size_t outer = 0; outer < 2; ++outer)
    for (std::size_t row = 0; row < selected.size(); ++row)
      std::copy_n(data.begin() + (outer * 3 + selected[row]) * columns, columns,
          expected.begin() + (outer * selected.size() + row) * columns);
  EXPECT_EQ(output.read<Bits>(), expected);
  EXPECT_EQ(source.read<Bits>(), data);
  EXPECT_EQ(indices.read<Index>(), (std::vector<Index>(selected.begin(), selected.end())));
  source.guard(); indices.guard(); output.guard();
}

template <class Index> void wide_checked_failure(Context &owner) {
  auto memory = cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto queue = owner.create_queue().value();
  const auto desc = TensorDesc::dense(DataType::kFloat32,
      std::array<std::int64_t, 2>{3, 262144}).value();
  const auto idesc = TensorDesc::dense(sizeof(Index) == 8 ? DataType::kInt64 : DataType::kInt32,
      std::array<std::int64_t, 1>{3}).value();
  Tensor source(desc, memory.ref()), indices(idesc, memory.ref());
  std::vector<std::uint32_t> data(desc.element_count().value());
  std::iota(data.begin(), data.end(), 0x7fc00001u);
  source.write<std::uint32_t>(data);
  auto program = prepare(owner, Gather{}, desc, idesc, memory.ref()).value();
  Tensor output(program.output_description().value(), memory.ref());
  const std::vector<std::uint32_t> poison(data.size(), 0xdeadbeefu);
  for (const auto invalid : {Index{-1}, Index{3}, (std::numeric_limits<Index>::max)()}) {
    SCOPED_TRACE(invalid);
    indices.write<Index>(std::array<Index, 3>{2, 0, invalid});
    output.write<std::uint32_t>(poison);
    auto slot = program.create_invocation().value();
    auto completion = slot.submit(queue, source.view.as_const(), indices.view.as_const(), output.view);
    ASSERT_TRUE(completion) << completion.error().message();
    EXPECT_TRUE(completion->wait().IsOutOfRange());
    EXPECT_EQ(output.read<std::uint32_t>(), poison);
    output.guard();
  }
  // A new invocation uses the original fresh device-status reservation.
  indices.write<Index>(std::array<Index, 3>{2, 0, 2});
  auto fresh = program.create_invocation().value();
  auto completion = fresh.submit(queue, source.view.as_const(), indices.view.as_const(), output.view);
  ASSERT_TRUE(completion) << completion.error().message();
  ASSERT_TRUE(completion->wait().ok());
  const auto actual = output.read<std::uint32_t>();
  for (std::size_t row = 0; row < 3; ++row) {
    const auto selected = row == 1 ? 0u : 2u;
    EXPECT_TRUE(std::equal(actual.begin() + row * 262144,
        actual.begin() + (row + 1) * 262144, data.begin() + selected * 262144));
  }
  output.guard();
}

TEST(CudaCcclGather, DenseMegabyteRowsPreserveBitsWithOuterGroupsDuplicatesAndTails) {
  auto owner = context();
  wide_dense_rows<std::uint32_t, std::int32_t>(owner, DataType::kFloat32,
      262144, false, false, 0, IndexBoundsPolicy::kChecked);
  wide_dense_rows<std::uint64_t, std::int64_t>(owner, DataType::kFloat64,
      131072, false, false, 1, IndexBoundsPolicy::kChecked);
  wide_dense_rows<std::uint8_t, std::int64_t>(owner, DataType::kUInt8,
      1048579, true, true, 0, IndexBoundsPolicy::kChecked);
  wide_dense_rows<std::uint32_t, std::int32_t>(owner, DataType::kUInt32,
      262145, true, true, 1, IndexBoundsPolicy::kUnchecked);
}

TEST(CudaCcclGather, CheckedWideRowFailurePreventsEveryOutputByteAndFreshSlotRecovers) {
  auto owner = context();
  wide_checked_failure<std::int32_t>(owner);
  wide_checked_failure<std::int64_t>(owner);
}

TEST(CudaCcclGather, LargeSourceAddressSpansRemainPreparedWithoutAllocatingTheirPayload) {
  auto owner = context();
  auto memory = cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  const auto indices = TensorDesc::dense(DataType::kInt64,
      std::array<std::int64_t, 1>{1}).value();
  constexpr std::int64_t large = (std::int64_t{1} << 32) + 16;
  const std::array sources{
      TensorDesc::dense(DataType::kUInt8, std::array<std::int64_t, 2>{large, 1}).value(),
      TensorDesc::strided_bytes(DataType::kUInt8,
          std::array<std::int64_t, 2>{2, 7}, std::array<std::int64_t, 2>{large, 1}).value()};
  for (const auto &source : sources) {
    auto prepared = prepare(owner, Gather{}, source, indices, memory.ref());
    ASSERT_TRUE(prepared) << prepared.error().message();
    EXPECT_EQ(prepared->workspace_requirement().size, 0u);
    EXPECT_TRUE(prepared->create_invocation());
  }
}

TEST(CudaCcclGather,
     EveryStorageTypeAndIndexWidthPreserveBitsAcrossAxesAndStrides) {
  auto owner = context();
  for (auto dtype : {DataType::kBool, DataType::kFloat8E4M3FN,
                     DataType::kFloat8E5M2, DataType::kInt8, DataType::kUInt8})
    both_indices<std::uint8_t>(owner, dtype);
  for (auto dtype : {DataType::kFloat16, DataType::kBFloat16, DataType::kInt16,
                     DataType::kUInt16})
    both_indices<std::uint16_t>(owner, dtype);
  for (auto dtype : {DataType::kFloat32, DataType::kInt32, DataType::kUInt32})
    both_indices<std::uint32_t>(owner, dtype);
  for (auto dtype : {DataType::kFloat64, DataType::kInt64, DataType::kUInt64})
    both_indices<std::uint64_t>(owner, dtype);
  gather<std::uint64_t, std::int64_t>(
      owner, DataType::kFloat64,
      std::array<std::int64_t, 8>{1, 2, 1, 3, 1, 5, 2, 1}, 5,
      std::array<std::int64_t, 1>{4}, 1, 1);
  gather<std::uint16_t, std::int32_t>(
      owner, DataType::kBFloat16, std::array<std::int64_t, 6>{2, 1, 7, 3, 1, 2},
      2, std::span<const std::int64_t>{}, 2, 0);
  gather<std::uint8_t, std::int64_t>(owner, DataType::kUInt8,
                                     std::array<std::int64_t, 2>{4, 3}, 0,
                                     std::array<std::int64_t, 2>{2, 3}, 0, 2);
}

TEST(CudaCcclGather, CheckedFailurePreservesAllOutputAndFreshSlotRecovers) {
  auto owner = context();
  auto memory =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto queue = owner.create_queue().value();
  const auto desc =
      TensorDesc::dense(DataType::kFloat64, std::array<std::int64_t, 2>{4, 3})
          .value();
  const auto idesc =
      TensorDesc::dense(DataType::kInt64, std::array<std::int64_t, 1>{2})
          .value();
  Tensor input(desc, memory.ref()), indices(idesc, memory.ref());
  input.write<std::uint64_t>(
      std::array<std::uint64_t, 12>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  auto program = prepare(owner, Gather{}, desc, idesc, memory.ref()).value();
  Tensor output(program.output_description().value(), memory.ref());
  auto scratch = workspace(program, memory.ref());
  EXPECT_EQ(program.workspace_requirement().size, 0u);
  EXPECT_TRUE(program.create_invocation());
  for (auto invalid : {std::int64_t{-1}, std::int64_t{4},
                       std::numeric_limits<std::int64_t>::max(),
                       std::numeric_limits<std::int64_t>::min()}) {
    indices.write<std::int64_t>(std::array{std::int64_t{0}, invalid});
    output.write<std::uint64_t>(std::array<std::uint64_t, 6>{8, 7, 6, 5, 4, 3});
    auto slot = program.create_invocation(scratch.view()).value();
    auto done = slot.submit(queue, input.view.as_const(),
                            indices.view.as_const(), output.view);
    ASSERT_TRUE(done) << done.error().message();
    EXPECT_TRUE(done->wait().IsOutOfRange());
    EXPECT_EQ(output.read<std::uint64_t>(),
              (std::vector<std::uint64_t>{8, 7, 6, 5, 4, 3}));
    EXPECT_FALSE(slot.submit(queue, input.view.as_const(),
                             indices.view.as_const(), output.view));
  }
  indices.write<std::int64_t>(std::array<std::int64_t, 2>{3, 1});
  auto fresh = program.create_invocation(scratch.view()).value();
  ASSERT_TRUE(fresh
                  .submit(queue, input.view.as_const(), indices.view.as_const(),
                          output.view)
                  ->wait()
                  .ok());
  EXPECT_EQ(output.read<std::uint64_t>(),
            (std::vector<std::uint64_t>{10, 11, 12, 4, 5, 6}));
  output.guard();
}

TEST(CudaCcclGather, CheckedSubmitStaysPendingBehindProducerAndReportsDeviceError) {
  auto owner = context();
  auto memory =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto queue = owner.create_queue().value();
  ASSERT_TRUE(queue.reserve_completions(4).ok());
  const auto desc =
      TensorDesc::dense(DataType::kInt32, std::array<std::int64_t, 2>{4, 3})
          .value();
  const auto idesc =
      TensorDesc::dense(DataType::kInt32, std::array<std::int64_t, 1>{2})
          .value();
  Tensor input(desc, memory.ref()), indices(idesc, memory.ref());
  input.write<std::int32_t>(std::array{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  indices.write<std::int32_t>(std::array{3, 1});
  auto program = prepare(owner, Gather{}, desc, idesc, memory.ref()).value();
  EXPECT_FALSE(program.runtime_properties().blocks_host);
  Tensor output(program.output_description().value(), memory.ref());
  auto scratch = workspace(program, memory.ref());
  // Resolve native lazy loading before holding the producer gate.
  {
    auto warm = program.create_invocation(scratch.view()).value();
    ASSERT_TRUE(warm.submit(queue, input.view.as_const(), indices.view.as_const(),
                            output.view)->wait().ok());
  }
  for (bool invalid : {false, true}) {
    indices.write<std::int32_t>(std::array{3, invalid ? 4 : 1});
    output.write<std::int32_t>(std::array{-1, -2, -3, -4, -5, -6});
    auto slot = program.create_invocation(scratch.view()).value();
    struct Gate {
      std::atomic<bool> released{false}, timed_out{false};
      static void CUDART_CB wait(void *raw) {
        auto &gate = *static_cast<Gate *>(raw);
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!gate.released.load(std::memory_order_acquire)) {
          if (std::chrono::steady_clock::now() >= deadline) {
            gate.timed_out = true;
            return;
          }
          std::this_thread::yield();
        }
      }
    } gate;
    const auto stream = cuda::stream(queue).value();
    struct Release {
      Gate &gate;
      cudaStream_t stream;
      ~Release() {
        gate.released.store(true, std::memory_order_release);
        (void)cudaStreamSynchronize(stream);
      }
    } release{gate, stream};
    ASSERT_EQ(cudaLaunchHostFunc(stream, &Gate::wait, &gate), cudaSuccess);
    auto done = slot.submit(queue, input.view.as_const(), indices.view.as_const(),
                            output.view);
    ASSERT_TRUE(done) << done.error().message();
    const auto polled = done->poll();
    ASSERT_TRUE(polled) << polled.error().message();
    EXPECT_EQ(*polled, CompletionPoll::kPending);
    EXPECT_FALSE(gate.timed_out.load());
    // The original accepted-command completion retains prepared handles.
    slot = {};
    gate.released.store(true, std::memory_order_release);
    const auto terminal = done->wait();
    EXPECT_TRUE(invalid ? terminal.IsOutOfRange() : terminal.ok());
    EXPECT_EQ(output.read<std::int32_t>(), invalid
        ? (std::vector{-1, -2, -3, -4, -5, -6})
        : (std::vector{10, 11, 12, 4, 5, 6}));
    output.guard();
  }
}

TEST(CudaCcclGather, EmptyUncheckedAndAliasesUseThePublicContract) {
  auto owner = context();
  auto memory =
      cuda::MemoryResource::create(owner, cuda::MemoryKind::kDevice).value();
  auto queue = owner.create_queue().value();
  const auto desc =
      TensorDesc::dense(DataType::kInt32, std::array<std::int64_t, 1>{4})
          .value();
  Tensor input(desc, memory.ref()), indices(desc, memory.ref()),
      output(desc, memory.ref());
  input.write<std::int32_t>(std::array{5, 6, 7, 8});
  indices.write<std::int32_t>(std::array{3, 2, 1, 0});
  auto program = prepare(owner, Gather{.bounds = IndexBoundsPolicy::kUnchecked},
                         desc, desc, memory.ref())
                     .value();
  EXPECT_EQ(program.workspace_requirement().size, 0u);
  EXPECT_FALSE(program.runtime_properties().blocks_host);
  auto slot = program.create_invocation().value();
  for (auto alias : {input.view, indices.view})
    EXPECT_FALSE(slot.submit(queue, input.view.as_const(),
                             indices.view.as_const(), alias));
  const auto partial =
      TensorView::create(desc, input.buffer.view().subview(4, 16).value())
          .value();
  EXPECT_FALSE(slot.submit(queue, input.view.as_const(),
                           indices.view.as_const(), partial));
  ASSERT_TRUE(slot.submit(queue, input.view.as_const(), indices.view.as_const(),
                          output.view)
                  ->wait()
                  .ok());
  EXPECT_EQ(output.read<std::int32_t>(), (std::vector{8, 7, 6, 5}));
  for (bool empty_indices : {false, true}) {
    const auto empty =
        TensorDesc::dense(DataType::kInt32, std::array<std::int64_t, 2>{4, 0})
            .value();
    const auto zero_indices =
        TensorDesc::dense(DataType::kInt32, std::array<std::int64_t, 1>{0})
            .value();
    const auto source_desc = empty_indices ? desc : empty;
    const auto index_desc = empty_indices ? zero_indices : desc;
    Tensor source(source_desc, memory.ref()),
        selected(index_desc, memory.ref());
    if (!empty_indices)
      selected.write<std::int32_t>(std::array{-1, INT_MIN, INT_MAX, 7});
    auto no_op =
        prepare(owner, Gather{}, source_desc, index_desc, memory.ref()).value();
    EXPECT_EQ(no_op.workspace_requirement().size, 0u);
    Tensor result(no_op.output_description().value(), memory.ref());
    auto empty_slot = no_op.create_invocation().value();
    ASSERT_TRUE(empty_slot
                    .submit(queue, source.view.as_const(),
                            selected.view.as_const(), result.view)
                    ->wait()
                    .ok());
  }
  EXPECT_FALSE(prepare(owner, Gather{}, desc, desc, mem::system_resource()));
}
} // namespace

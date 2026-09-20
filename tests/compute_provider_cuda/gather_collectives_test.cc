#include "collectives_test_support.h"

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
  EXPECT_FALSE(program.create_invocation());
  EXPECT_FALSE(program.create_invocation(
      scratch.view().subview(0, scratch.size() - 1).value()));
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

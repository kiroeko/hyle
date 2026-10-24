#include "hyle/graph/providers/ggml_cuda.h"

#include "../compute_provider_ggml_cuda/tensor_test_support.h"
#include "hyle/graph/executable.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace mem = hyle::mem;
using mem::Buffer;
using BF16 = hyle::simd::bfloat16_t;
using ggml_cuda_tensor_test::dense;
using ggml_cuda_tensor_test::output_buffer;
using ggml_cuda_tensor_test::upload;
constexpr std::int64_t kRows = 17, kInner = 37, kColumns = 19;

template <class Value> struct Inputs {
  std::vector<Value> left, right, bias;
  std::vector<Value> product, added;
};

template <class Value>
Inputs<Value> inputs(int phase, int slot, bool transposed) {
  Inputs<Value> result;
  result.left.resize(kRows * kInner);
  result.right.resize(kInner * kColumns);
  result.bias.resize(kColumns);
  const auto seed = phase * 3 + slot + 1;
  for (std::size_t i = 0; i < result.left.size(); ++i) {
    const double value = std::is_same_v<Value, float>
        ? (int(i % 19) - 9) * 0.03713 + seed * 0.01317
        : (int(i % 19) - 9) / 16.0 + seed / 128.0;
    result.left[i] = half_test::encode<Value>(value);
  }
  for (std::int64_t inner = 0; inner < kInner; ++inner)
    for (std::int64_t column = 0; column < kColumns; ++column) {
      const auto i = inner * kColumns + column;
      const double value = std::is_same_v<Value, float>
          ? (int(i % 23) - 11) * 0.02919 - seed * 0.00713
          : (int(i % 23) - 11) / 32.0 - seed / 256.0;
      result.right[transposed ? column * kInner + inner : i] =
          half_test::encode<Value>(value);
    }
  for (auto &value : result.bias)
    value = half_test::encode<Value>(1.0 / 256.0);
  std::size_t rounding_witnesses = 0;
  for (std::int64_t row = 0; row < kRows; ++row)
    for (std::int64_t column = 0; column < kColumns; ++column) {
      double sum = 0;
      for (std::int64_t inner = 0; inner < kInner; ++inner)
        sum += half_test::decode(result.left[row * kInner + inner]) *
               half_test::decode(result.right[transposed
                   ? column * kInner + inner : inner * kColumns + column]);
      const auto rounded = half_test::encode<Value>(sum);
      const auto added = half_test::encode<Value>(
          half_test::decode(rounded) + half_test::decode(result.bias[column]));
      result.product.push_back(rounded);
      result.added.push_back(added);
      if constexpr (std::is_same_v<Value, BF16>)
        rounding_witnesses += added.bits != half_test::encode<Value>(
            sum + half_test::decode(result.bias[column])).bits;
    }
  // These binary-rational BF16 products/sums fit exactly in F32. The oracle
  // therefore checks RNE at the public MatMul boundary, not a loose BF16 error.
  if constexpr (std::is_same_v<Value, BF16>)
    EXPECT_GT(rounding_witnesses, 0U);
  return result;
}

template <class Value>
void check(const Buffer &output, const std::vector<Value> &expected) {
  const auto actual = half_test::read<Value>(output);
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if constexpr (std::is_same_v<Value, BF16>)
      EXPECT_EQ(actual[i].bits, expected[i].bits) << i;
    else
      EXPECT_NEAR(actual[i], expected[i],
                  2e-6 + 6e-6 * std::abs(expected[i])) << i;
  }
}

template <class Value>
void unchanged(const Buffer &output, const std::vector<Value> &snapshot) {
  const auto actual = half_test::read<Value>(output);
  ASSERT_EQ(actual.size(), snapshot.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if constexpr (std::is_same_v<Value, BF16>)
      EXPECT_EQ(actual[i].bits, snapshot[i].bits) << i;
    else
      EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]),
                std::bit_cast<std::uint32_t>(snapshot[i])) << i;
  }
}

// Capability is independent of default route priority: an explicitly selected
// provider must handle both matrices and batches of matrices.
c::TensorDesc matrix_desc(c::DataType type, std::int64_t rows,
                          std::int64_t columns, bool batched) {
  return batched ? dense(type, {1, rows, columns})
                 : dense(type, {rows, columns});
}

template <class Value> void direct(bool transposed, bool batched) {
  SCOPED_TRACE(::testing::Message() << "dtype=" << int(half_test::data_type<Value>())
                                  << ", transposed=" << transposed
                                  << ", batched=" << batched);
  auto context = c::Context::create(c::cuda::device(0).value(),
                                   c::ggml_cuda::provider()).value();
  auto memory = c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kDevice).value();
  const auto type = half_test::data_type<Value>();
  const auto left_desc = matrix_desc(type, kRows, kInner, batched);
  const auto right_desc = transposed
      ? matrix_desc(type, kColumns, kInner, batched).transposed().value()
      : matrix_desc(type, kInner, kColumns, batched);
  const auto output_desc = matrix_desc(type, kRows, kColumns, batched);
  const std::array descriptions{left_desc, right_desc};
  auto prepared = c::prepare(context, c::Operation(c::MatMul{}), descriptions,
                             std::span(&output_desc, 1), memory.ref());
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, c::ggml_cuda::kProviderId);
  std::vector<Buffer> retained;
  std::array workspaces{
      ggml_cuda_test::workspace_buffer(*prepared, memory.ref()),
      ggml_cuda_test::workspace_buffer(*prepared, memory.ref())};
  std::array runs{prepared->create_invocation(workspaces[0].view()).value(),
                  prepared->create_invocation(workspaces[1].view()).value()};
  std::array queues{context.create_queue().value(), context.create_queue().value()};
  std::vector<Value> old;
  for (int phase = 0; phase < 2; ++phase) {
    std::array<Buffer, 2> left, right, output;
    std::array<Inputs<Value>, 2> host;
    std::array<c::Completion, 2> done;
    for (int slot = 0; slot < 2; ++slot) {
      host[slot] = inputs<Value>(phase, slot, transposed);
      left[slot] = upload<Value>(memory.ref(), host[slot].left);
      right[slot] = upload<Value>(memory.ref(), host[slot].right);
      output[slot] = output_buffer(memory.ref(), output_desc);
      const std::array bindings{
          c::ConstTensorView::create(left_desc, left[slot].view()).value(),
          c::ConstTensorView::create(right_desc, right[slot].view()).value()};
      auto destination =
          c::TensorView::create(output_desc, output[slot].view()).value();
      auto submitted = runs[slot].submit(queues[slot], bindings, destination);
      ASSERT_TRUE(submitted) << submitted.error().message();
      done[slot] = std::move(*submitted);
    }
    ASSERT_TRUE(done[1].wait().ok());
    ASSERT_TRUE(done[0].wait().ok());
    for (int slot = 0; slot < 2; ++slot) {
      check(output[slot], host[slot].product);
      if (phase == 0 && slot == 0)
        old = half_test::read<Value>(output[slot]);
      retained.push_back(std::move(left[slot]));
      retained.push_back(std::move(right[slot]));
      retained.push_back(std::move(output[slot]));
    }
    // Keep every old allocation alive, forcing genuinely new addresses on
    // the second binding of both independent native invocation workspaces.
    if (phase == 1)
      unchanged(retained[2], old);
  }
}

template <class Value> void region(bool transposed, bool batched) {
  SCOPED_TRACE(::testing::Message() << "dtype=" << int(half_test::data_type<Value>())
                                  << ", transposed=" << transposed
                                  << ", batched=" << batched);
  auto context = c::Context::create(c::cuda::device(0).value(),
                                   g::ggml_cuda::provider()).value();
  auto memory = c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kDevice).value();
  const auto type = half_test::data_type<Value>();
  const auto left_desc = matrix_desc(type, kRows, kInner, batched);
  const auto stored_right_desc = transposed
      ? matrix_desc(type, kColumns, kInner, batched)
      : matrix_desc(type, kInner, kColumns, batched);
  const auto bias_desc = dense(type, {kColumns});
  const auto output_desc = matrix_desc(type, kRows, kColumns, batched);
  auto builder = g::GraphBuilder::create().value();
  const auto x = builder.input(left_desc).value();
  const auto weight = builder.input(stored_right_desc).value();
  const auto bias = builder.input(bias_desc).value();
  auto rhs = weight;
  if (transposed) {
    // Matrix axes expose [K,N] while keeping stored rows [N,K].
    const auto transpose = batched
        ? c::View::permute(std::array<std::int32_t, 3>{0, 2, 1}).value()
        : c::View::permute(std::array<std::int32_t, 2>{1, 0}).value();
    rhs = builder.add(transpose, weight).value();
  }
  const auto product = builder.add(c::MatMul{}, x, rhs).value();
  const auto added = builder.add(c::Binary{}, product, bias).value();
  ASSERT_TRUE(builder.publish(product).ok());
  ASSERT_TRUE(builder.publish(added).ok());
  const std::array domains{g::StorageDomain{.resource = memory.ref()}};
  const std::array targets{g::ExecutionTarget{
      .source = g::ExecutionSource::acquire(context).value(),
      .default_storage = g::StorageDomainId{0}}};
  const std::array lanes{g::ExecutionLane{.target = g::ExecutionTargetId{0}}};
  auto compiled = g::compile(std::move(builder).finish().value(),
      {.storage_domains = domains, .targets = targets, .lanes = lanes});
  ASSERT_TRUE(compiled) << compiled.error().message();
  // The public View is Graph-owned metadata, not another mathematical
  // dispatch or a materialized transpose. MatMul/Add stay in one region.
  ASSERT_EQ(compiled->report.step_count(), transposed ? 2U : 1U);
  const auto region_index = transposed ? 1U : 0U;
  if (transposed) {
    const auto view = compiled->report.step(0);
    ASSERT_TRUE(view);
    ASSERT_EQ(view->kind, g::CompileStepKind::kView);
    EXPECT_EQ(view->dispatch_count, 0U);
    EXPECT_EQ(view->copy_bytes, 0U);
  }
  const auto region_step = compiled->report.step(region_index);
  ASSERT_TRUE(region_step);
  EXPECT_EQ(region_step->kind, g::CompileStepKind::kProviderRegion);
  EXPECT_EQ(region_step->provider.id, c::ggml_cuda::kProviderId);
  EXPECT_EQ(region_step->node_count, 2U);
  std::vector<Buffer> retained;
  std::array runs{compiled->executable.create_invocation().value(),
                  compiled->executable.create_invocation().value()};
  std::array queues{context.create_queue().value(), context.create_queue().value()};
  std::vector<Value> old_product, old_added;
  for (int phase = 0; phase < 2; ++phase) {
    std::array<Buffer, 2> left, right, biases, products, sums;
    std::array<Inputs<Value>, 2> host;
    std::array<c::Completion, 2> done;
    for (int slot = 0; slot < 2; ++slot) {
      host[slot] = inputs<Value>(phase, slot, transposed);
      left[slot] = upload<Value>(memory.ref(), host[slot].left);
      right[slot] = upload<Value>(memory.ref(), host[slot].right);
      biases[slot] = upload<Value>(memory.ref(), host[slot].bias);
      products[slot] = output_buffer(memory.ref(), output_desc);
      sums[slot] = output_buffer(memory.ref(), output_desc);
      auto &run = runs[slot];
      ASSERT_TRUE(run.bind(x, c::ConstTensorView::create(
          left_desc, left[slot].view()).value()).ok());
      ASSERT_TRUE(run.bind(weight, c::ConstTensorView::create(
          stored_right_desc, right[slot].view()).value()).ok());
      ASSERT_TRUE(run.bind(bias, c::ConstTensorView::create(
          bias_desc, biases[slot].view()).value()).ok());
      ASSERT_TRUE(run.bind(product, c::TensorView::create(
          output_desc, products[slot].view()).value()).ok());
      ASSERT_TRUE(run.bind(added, c::TensorView::create(
          output_desc, sums[slot].view()).value()).ok());
      const std::array bindings{g::QueueBinding{g::ExecutionLaneId{0}, &queues[slot]}};
      ASSERT_TRUE(run.prepare_replay(bindings).ok());
      auto submitted = run.submit(queues[slot]);
      ASSERT_TRUE(submitted) << submitted.error().message();
      done[slot] = std::move(*submitted);
    }
    ASSERT_TRUE(done[1].wait().ok());
    ASSERT_TRUE(done[0].wait().ok());
    for (int slot = 0; slot < 2; ++slot) {
      check(products[slot], host[slot].product);
      check(sums[slot], host[slot].added);
      if (phase == 0 && slot == 0) {
        old_product = half_test::read<Value>(products[slot]);
        old_added = half_test::read<Value>(sums[slot]);
      }
      retained.push_back(std::move(left[slot]));
      retained.push_back(std::move(right[slot]));
      retained.push_back(std::move(biases[slot]));
      retained.push_back(std::move(products[slot]));
      retained.push_back(std::move(sums[slot]));
    }
    if (phase == 1) {
      unchanged(retained[3], old_product);
      unchanged(retained[4], old_added);
    }
  }
}

TEST(GgmlCudaDenseProjection, DirectDenseAndTransposedRhsKeepPrecisionAndSlots) {
  for (const bool batched : {false, true})
    for (const bool transposed : {false, true}) {
      direct<float>(transposed, batched);
      direct<BF16>(transposed, batched);
    }
}

TEST(GgmlCudaDenseProjection, WeightViewsKeepPublishedRoundingAndRebindBothSlots) {
  for (const bool batched : {false, true})
    for (const bool transposed : {false, true}) {
      region<float>(transposed, batched);
      region<BF16>(transposed, batched);
    }
}

TEST(GgmlCudaDenseProjection, PaddedOrOtherStridedRhsAndStridedLhsStayUnsupported) {
  auto context = c::Context::create(c::cuda::device(0).value(),
                                   c::ggml_cuda::provider()).value();
  auto memory = c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kDevice).value();
  for (const auto type : {c::DataType::kFloat32, c::DataType::kBFloat16}) {
    const auto bytes = std::int64_t(c::data_type_size(type));
    const auto left = dense(type, {1, kRows, kInner});
    const auto right = dense(type, {1, kInner, kColumns});
    const auto output = dense(type, {1, kRows, kColumns});
    const std::array shape{std::int64_t{1}, kInner, kColumns};
    const std::array bad_rights{
        c::TensorDesc::strided_bytes(type, shape,
            std::array{kInner * (kColumns + 1) * bytes,
                       (kColumns + 1) * bytes, bytes}).value(),
        c::TensorDesc::strided_bytes(type, shape,
            std::array{kColumns * (kInner + 1) * bytes,
                       bytes, (kInner + 1) * bytes}).value(),
        c::TensorDesc::strided_bytes(type, shape,
            std::array{kInner * kColumns * 2 * bytes,
                       kColumns * 2 * bytes, 2 * bytes}).value()};
    for (const auto &bad : bad_rights) {
      const std::array descriptions{left, bad};
      auto prepared = c::prepare(context, c::Operation(c::MatMul{}), descriptions,
                                 std::span(&output, 1), memory.ref());
      ASSERT_FALSE(prepared);
      EXPECT_TRUE(prepared.error().IsUnimplemented());
    }
    const auto bad_left = c::TensorDesc::strided_bytes(type,
        std::array{std::int64_t{1}, kRows, kInner},
        std::array{kRows * (kInner + 1) * bytes,
                   (kInner + 1) * bytes, bytes}).value();
    const std::array descriptions{bad_left, right};
    auto prepared = c::prepare(context, c::Operation(c::MatMul{}), descriptions,
                               std::span(&output, 1), memory.ref());
    ASSERT_FALSE(prepared);
    EXPECT_TRUE(prepared.error().IsUnimplemented());
  }
}
} // namespace

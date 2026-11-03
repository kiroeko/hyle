#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/detail/packed_row_matmul.h"
#include "hyle/compute/detail/queue_sequence.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/cublas.h"
#include "hyle/compute/sequence_ops.h"

namespace {
namespace c = hyle::compute;
namespace cd = c::detail;
namespace m = hyle::mem;
constexpr std::int64_t C = 128, K = 64, N = 32;

void need(hyle::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
c::TensorDesc dense(c::DataType type, std::initializer_list<std::int64_t> dimensions) {
  return c::TensorDesc::dense(type, {dimensions.begin(), dimensions.size()}).value();
}
std::uint16_t bits(c::DataType type, double value) {
  return type == c::DataType::kFloat16
      ? half_test::encode<hyle::simd::float16_t>(value).bits
      : half_test::encode<hyle::simd::bfloat16_t>(value).bits;
}
double value(std::size_t row, std::size_t column) {
  return double(int((row * 7 + column * 3) % 29) - 14) / 4.0;
}

struct Environment final {
  // A GGML or generic CUDA fallback cannot satisfy these tests.
  c::Context context = c::Context::create(c::cuda::device(0).value(), c::cublas::provider()).value();
  c::cuda::MemoryResource memory = c::cuda::MemoryResource::create(context,
      c::cuda::MemoryKind::kDevice).value();
  c::Queue queue = context.create_queue().value();
  Environment() { need(queue.reserve_completions(16)); }
  ~Environment() { context.drain().IgnoreError(); }
};

struct Case final {
  Environment &environment;
  c::DataType input_type, output_type;
  std::array<c::TensorDesc, 3> descriptions;
  c::TensorDesc result_description;
  std::array<m::Buffer, 3> storage;
  m::Buffer result, workspace;
  std::array<c::ConstTensorView, 3> inputs;
  std::array<c::TensorView, 1> outputs;
  c::PreparedOperation prepared;
  c::OperationInvocation lane;

  Case(Environment &owner, c::DataType type, c::DataType output, bool transposed)
      : environment(owner), input_type(type), output_type(output) {
    descriptions = {dense(type, {C, K}), transposed
        ? dense(type, {N, K}).transposed().value() : dense(type, {K, N}),
        dense(c::DataType::kInt32, {2})};
    const auto operation = c::sequence_ops::packed_row_matmul(output);
    std::array<c::TensorDesc, 1> inferred;
    need(c::infer_output_descriptions(operation, descriptions, inferred));
    result_description = inferred[0];
    for (std::size_t port = 0; port < inputs.size(); ++port) {
      storage[port] = m::Buffer::create(owner.memory.ref(),
          descriptions[port].required_bytes().value(), 256).value();
      inputs[port] = c::ConstTensorView::create(descriptions[port], storage[port].view()).value();
    }
    const auto bytes = result_description.required_bytes().value();
    result = m::Buffer::create(owner.memory.ref(), bytes + 32, 256).value();
    need(result.view().fill(std::byte{0xa5}));
    outputs[0] = c::TensorView::create(result_description,
        result.view().subview(0, bytes).value()).value();
    // The axes differ and each output selects column+1, so a mistaken RHS
    // transpose or prefix dimension cannot pass an identity-only oracle.
    std::vector<std::uint16_t> weights(K * N, 0);
    for (std::size_t column = 0; column < std::size_t(N); ++column)
      weights[transposed ? column * K + column + 1 : (column + 1) * N + column] = bits(type, .5);
    need(storage[1].view().write(std::as_bytes(std::span(weights))));
    prepared = c::prepare(owner.context, operation, descriptions,
        std::array{result_description}, owner.memory.ref()).value();
    EXPECT_EQ(prepared.provider().id, c::cublas::kProviderId);
    EXPECT_EQ(prepared.implementation(), c::ProviderImplementationId{2});
    EXPECT_TRUE(prepared.capture_compatible());
    const auto needed = prepared.workspace_requirement();
    workspace = m::Buffer::create(owner.memory.ref(), needed.size + 32,
        (std::max)(needed.alignment, std::size_t{256})).value();
    need(workspace.view().fill(std::byte{0xa5}));
    fresh();
  }
  ~Case() { environment.queue.drain().IgnoreError(); lane = {}; }
  void fresh() {
    need(environment.queue.drain());
    lane = {};
    lane = prepared.create_invocation(workspace.view().subview(0,
        prepared.workspace_requirement().size).value()).value();
  }
  void upload(std::size_t active) {
    std::vector<std::uint16_t> activation(C * K, input_type == c::DataType::kFloat16
        ? std::uint16_t{0x7e55} : std::uint16_t{0x7fc5});
    for (std::size_t row = 0; row < active; ++row)
      for (std::size_t column = 0; column < std::size_t(K); ++column)
        activation[row * K + column] = bits(input_type, value(row, column));
    need(storage[0].view().write(std::as_bytes(std::span(activation))));
    offsets({0, static_cast<std::int32_t>(active)});
  }
  void offsets(std::array<std::int32_t, 2> offsets) {
    need(storage[2].view().write(std::as_bytes(std::span(offsets))));
  }
  hyle::Status configure(std::size_t rows, bool clear = false) {
    // This isolates the selected Compute callback. The original Batch-owned
    // layout/generation capability is tested by the separate GraphBatch suite.
    const cd::PackedRowMatMulRowsConfiguration configuration{inputs[2], rows, clear};
    const auto configured = cd::OperationInvocationAccess::configure_cold(lane,
        cd::kPackedRowMatMulRowsExtensionId, &configuration);
    if (!configured) return configured.error();
    return *configured ? hyle::Status::Ok()
        : hyle::Status::Internal("selected cuBLAS rows configuration was not applied");
  }
  hyle::Status submit() {
    auto done = lane.submit(environment.queue, inputs, outputs);
    if (!done) return done.error();
    const auto status = done->wait();
    *done = {};
    return status;
  }
  void poison() { need(outputs[0].storage().fill(std::byte{0x5a})); }
  void guards() const {
    std::array<std::byte, 32> actual, expected;
    expected.fill(std::byte{0xa5});
    ASSERT_TRUE(result.view().subview(result_description.required_bytes().value(), 32).value().read(actual).ok());
    EXPECT_EQ(actual, expected);
    ASSERT_TRUE(workspace.view().subview(prepared.workspace_requirement().size, 32).value().read(actual).ok());
    EXPECT_EQ(actual, expected);
  }
  void expect_poison() const {
    std::vector<std::byte> actual(outputs[0].storage().size());
    ASSERT_TRUE(outputs[0].storage().read(actual).ok());
    EXPECT_EQ(actual, std::vector<std::byte>(actual.size(), std::byte{0x5a}));
    guards();
  }
  void check(std::size_t active) const {
    std::vector<std::byte> actual(outputs[0].storage().size()), expected(actual.size(), std::byte{0});
    ASSERT_TRUE(outputs[0].storage().read(actual).ok());
    for (std::size_t row = 0; row < active; ++row)
      for (std::size_t column = 0; column < std::size_t(N); ++column) {
        const auto at = row * N + column;
        const auto exact = value(row, column + 1) * .5;
        if (output_type == c::DataType::kFloat32) {
          const auto word = std::bit_cast<std::array<std::byte, 4>>(static_cast<float>(exact));
          std::copy(word.begin(), word.end(), expected.begin() + at * 4);
        } else {
          const auto word = std::bit_cast<std::array<std::byte, 2>>(bits(output_type, exact));
          std::copy(word.begin(), word.end(), expected.begin() + at * 2);
        }
      }
    EXPECT_EQ(actual, expected);
    guards();
  }
};

TEST(CublasPackedRowMatMul, PrefixTailAndLengthChangesKeepOriginalCallerWorkspace) {
  Environment environment;
  for (const auto type : {c::DataType::kFloat16, c::DataType::kBFloat16}) {
    SCOPED_TRACE(static_cast<int>(type));
    // Lt's original same-source contract supports the same 16-bit result or
    // F32. It does not imply cross FP16/BF16 destination conversion.
    for (const auto output : {c::DataType::kFloat32, type}) {
      SCOPED_TRACE(static_cast<int>(output));
      for (const bool transposed : {false, true}) {
        SCOPED_TRACE(transposed);
        Case test(environment, type, output, transposed);
        const auto original = test.workspace.view();
        const auto bytes = test.prepared.workspace_requirement().size;
        for (const auto active : {0U, 1U, 17U, 128U, 1U}) {
          SCOPED_TRACE(active);
          test.upload(active); test.poison();
          ASSERT_TRUE(test.configure(active).ok());
          const auto status = test.submit();
          ASSERT_TRUE(status.ok()) << status.message();
          test.check(active);
          EXPECT_EQ(test.workspace.view().native_handle(), original.native_handle());
          EXPECT_EQ(test.workspace.view().offset(), original.offset());
          EXPECT_EQ(test.prepared.workspace_requirement().size, bytes);
        }
      }
    }
  }
}

TEST(CublasPackedRowMatMul, MissingForeignOrInvalidControlPublishesNothingAndCanRetry) {
  Environment environment;
  Case test(environment, c::DataType::kBFloat16, c::DataType::kFloat32, true);
  test.upload(17); test.poison();
  EXPECT_TRUE(test.submit().IsUnimplemented()); test.expect_poison();
  EXPECT_FALSE(test.configure(C + 1).ok());
  ASSERT_TRUE(test.configure(17).ok());
  ASSERT_TRUE(test.submit().ok()); test.check(17);
  ASSERT_TRUE(test.configure(0, true).ok());
  test.poison();
  EXPECT_TRUE(test.submit().IsUnimplemented()); test.expect_poison();
  ASSERT_TRUE(test.configure(17).ok());
  auto foreign = m::Buffer::create(environment.memory.ref(), 8, 256).value();
  need(foreign.view().fill(std::byte{0}));
  const auto original_offsets = test.inputs[2];
  test.inputs[2] = c::ConstTensorView::create(test.descriptions[2], foreign.view()).value();
  const auto wrong_tuple = test.submit();
  EXPECT_TRUE(wrong_tuple.IsFailedPrecondition()) << wrong_tuple.message();
  test.expect_poison();
  test.inputs[2] = original_offsets;
  ASSERT_TRUE(test.submit().ok()); test.check(17);
  for (const auto offsets : {std::array<std::int32_t, 2>{1, 17}, std::array<std::int32_t, 2>{-1, 17},
      std::array<std::int32_t, 2>{0, -1}, std::array<std::int32_t, 2>{0, 129},
      std::array<std::int32_t, 2>{0, 16}}) {
    SCOPED_TRACE(offsets[0]);
    SCOPED_TRACE(offsets[1]);
    test.upload(17); test.offsets(offsets); test.poison();
    ASSERT_TRUE(test.configure(17).ok());
    const auto bad = test.submit();
    EXPECT_TRUE(bad.IsInvalidArgument()) << bad.message(); test.expect_poison();
    test.fresh(); test.upload(17);
    ASSERT_TRUE(test.configure(17).ok());
    ASSERT_TRUE(test.submit().ok()); test.check(17);
  }
  Case good(environment, c::DataType::kFloat16, c::DataType::kFloat16, false);
  test.upload(17); test.offsets({0, 129}); test.poison();
  ASSERT_TRUE(test.configure(17).ok());
  good.upload(1); good.poison(); ASSERT_TRUE(good.configure(1).ok());
  auto sequence = cd::QueueSequence::begin(environment.queue).value();
  const auto first = cd::OperationInvocationAccess::enqueue(test.lane, sequence, test.inputs, test.outputs);
  EXPECT_TRUE(first.accepted); EXPECT_TRUE(first.status.ok()) << first.status.message();
  sequence.record(first);
  const auto second = cd::OperationInvocationAccess::enqueue(good.lane, sequence, good.inputs, good.outputs);
  EXPECT_TRUE(second.accepted); EXPECT_TRUE(second.status.ok()) << second.status.message();
  sequence.record(second);
  auto done = sequence.finish().value();
  EXPECT_TRUE(done.wait().IsInvalidArgument()); done = {};
  test.expect_poison(); good.expect_poison();
}
} // namespace

namespace {
TEST(CublasPackedRowMatMul, IndependentWeightsAndControlsSurviveManyPreparedRowShapes) {
  Environment environment;
  Case first(environment, c::DataType::kBFloat16, c::DataType::kFloat32, false);
  Case second(environment, c::DataType::kBFloat16, c::DataType::kFloat32, false);
  // Identical layouts and algorithm eligibility never imply shared payloads.
  need(second.storage[1].view().fill(std::byte{0}));
  const std::vector<std::byte> zero(second.outputs[0].storage().size(),std::byte{0});
  const auto exercise=[&](std::size_t active) {
    first.upload(active);second.upload(active);
    first.poison();second.poison();
    ASSERT_TRUE(first.configure(active).ok());
    ASSERT_TRUE(second.configure(active).ok());
    ASSERT_TRUE(first.submit().ok());
    ASSERT_TRUE(second.submit().ok());
    first.check(active);
    std::vector<std::byte> actual(zero.size());
    ASSERT_TRUE(second.outputs[0].storage().read(actual).ok());
    EXPECT_EQ(actual,zero);second.guards();
    // A remembered vendor qualification grants no executable row control.
    ASSERT_TRUE(second.configure(0,true).ok());
    second.poison();EXPECT_FALSE(second.submit().ok());second.expect_poison();
  };
  for(std::size_t rows=1;rows<=96;++rows){SCOPED_TRACE(rows);exercise(rows);}
  for(const auto rows:{17u,1u,96u}){SCOPED_TRACE(rows);exercise(rows);}
}
} // namespace

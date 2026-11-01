#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "../compute_provider_native/half_test_support.h"
#include "test_support.h"
#include "hyle/compute/detail/packed_row_matmul.h"
#include "hyle/compute/detail/provider_extension.h"
#include "hyle/compute/detail/queue_sequence.h"
#include "hyle/compute/sequence_ops.h"
#include "../../src/compute_provider_ggml_cuda/active_rows.h"

namespace {
namespace c = hyle::compute;
namespace cd = c::detail;
namespace native = c::ggml_cuda::internal;
namespace m = hyle::mem;
constexpr std::int64_t C = 128, K = 64, N = 32;

c::TensorDesc dense(c::DataType type, std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
}
void need(hyle::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
std::uint16_t bits(c::DataType type, double value) {
  return type == c::DataType::kFloat16 ? half_test::encode<hyle::simd::float16_t>(value).bits
      : half_test::encode<hyle::simd::bfloat16_t>(value).bits;
}
double input_value(std::size_t row, std::size_t column) {
  return double(int((row * 7 + column * 3) % 29) - 14) / 4.0;
}

struct Environment final {
  // The new operation cannot silently use a different CUDA provider.
  c::Context context = c::Context::create(c::cuda::device(0).value(), c::ggml_cuda::provider()).value();
  c::cuda::MemoryResource memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  c::Queue queue = context.create_queue().value();
  cd::ProviderExtensionRef source = cd::ProviderExtensionAccess::query(
      context, 0, cd::kPackedRowMatMulSourceExtensionId).value();
  ~Environment() { context.drain().IgnoreError(); }
};

// Uses the original prepared artifact, QueueSequence, checked publication and
// real Completion. It does not emulate Graph's owner/generation qualification.
// Those separate typed-bridge controls are exercised by the GraphBatch tests.
struct NativeProgram final {
  Environment &environment;
  cd::ProviderPrepared prepared;
  m::Buffer workspace;
  void *invocation = nullptr;
  explicit NativeProgram(Environment &value) : environment(value) {}
  ~NativeProgram() {
    environment.context.drain().IgnoreError();
    if (invocation) prepared.vtable->destroy_invocation(invocation, m::system_allocator());
    if (prepared.vtable) prepared.vtable->destroy_prepared(prepared.state, m::system_allocator());
  }
  void compile(native::Recipe recipe) {
    struct Call { native::Recipe *recipe; cd::ProviderPrepared *prepared; } call{&recipe, &prepared};
    need(cd::ProviderExtensionAccess::call(environment.source,
        [](void *context, const void *table, void *opaque) noexcept -> hyle::Status {
          auto &call = *static_cast<Call *>(opaque);
          const auto *capability = static_cast<const cd::PackedRowMatMulSourceVTable *>(table);
          if (!capability || !capability->available || !capability->available(context))
            return hyle::Status::Unimplemented("linked packed row MatMul contract is unavailable");
          return native::compile(context, std::move(*call.recipe), m::system_allocator(), call.prepared).status;
        }, &call));
    workspace = m::Buffer::create(environment.memory.ref(), prepared.workspace.size + 32,
        (std::max)(std::size_t{256}, prepared.workspace.alignment)).value();
    need(workspace.view().fill(std::byte{0xa5}));
    fresh();
  }
  void fresh() {
    need(environment.queue.drain());
    if (invocation) {
      prepared.vtable->destroy_invocation(invocation, m::system_allocator());
      invocation = nullptr;
    }
    need(prepared.vtable->create_invocation(prepared.state, m::system_allocator(),
        workspace.view().subview(0, prepared.workspace.size).value(), &invocation));
  }
  cd::EnqueueResult enqueue(cd::QueueSequence &sequence,
      std::span<const c::ConstTensorView> inputs, std::span<const c::TensorView> outputs) {
    struct Call { NativeProgram *program; std::span<const c::ConstTensorView> inputs;
      std::span<const c::TensorView> outputs; } call{this, inputs, outputs};
    return cd::ProviderExtensionAccess::enqueue_admitted(environment.source, sequence, inputs, outputs,
        [](void *, const void *, void *opaque, cd::QueueSequence &admitted) noexcept {
          auto &call = *static_cast<Call *>(opaque);
          auto &p = *call.program;
          return p.prepared.vtable->shared_submit(p.prepared.state, p.invocation,
              admitted, call.inputs, call.outputs);
        }, &call);
  }
  hyle::Status submit(std::span<const c::ConstTensorView> inputs,
      std::span<const c::TensorView> outputs) {
    auto sequence = cd::QueueSequence::begin(environment.queue);
    if (!sequence) return sequence.error();
    sequence->record(enqueue(*sequence, inputs, outputs));
    auto done = sequence->finish();
    if (!done) return done.error();
    const auto status = done->wait(); *done = {};
    return status;
  }
  void check_workspace_guard() const {
    std::array<std::byte, 32> actual, expected;
    expected.fill(std::byte{0xa5});
    ASSERT_TRUE(workspace.view().subview(prepared.workspace.size, 32).value().read(actual).ok());
    EXPECT_EQ(actual, expected);
  }
};

struct Case final {
  Environment &environment;
  c::DataType type, output_type;
  std::array<c::TensorDesc, 3> descriptions;
  c::TensorDesc result_description;
  std::array<m::Buffer, 3> storage;
  m::Buffer result;
  std::array<c::ConstTensorView, 3> inputs;
  std::array<c::TensorView, 1> outputs;
  // Destroy the invocation and actually drain before any raw caller storage.
  NativeProgram program;
  Case(Environment &e, c::DataType input_type, c::DataType result_type, bool transposed)
      : environment(e), type(input_type), output_type(result_type), program(e) {
    descriptions = {dense(type, {C, K}), transposed ? dense(type, {N, K}).transposed().value()
        : dense(type, {K, N}), dense(c::DataType::kInt32, {2})};
    const auto operation = c::sequence_ops::packed_row_matmul(output_type);
    std::array<c::TensorDesc, 1> inferred;
    need(c::infer_output_descriptions(operation, descriptions, inferred));
    result_description = inferred[0];
    for (std::size_t port = 0; port < inputs.size(); ++port) {
      storage[port] = m::Buffer::create(e.memory.ref(), descriptions[port].required_bytes().value(), 256).value();
      inputs[port] = c::ConstTensorView::create(descriptions[port], storage[port].view()).value();
    }
    result = m::Buffer::create(e.memory.ref(), result_description.required_bytes().value() + 32, 256).value();
    need(result.view().fill(std::byte{0xa5}));
    outputs[0] = c::TensorView::create(result_description,
        result.view().subview(0, result_description.required_bytes().value()).value()).value();
    std::vector<std::uint16_t> weights(std::size_t(K * N), 0);
    for (std::size_t column = 0; column < std::size_t(N); ++column)
      weights[transposed ? column * K + column + 1 : (column + 1) * N + column] = bits(type, .5);
    need(storage[1].view().write(std::as_bytes(std::span(weights))));
    native::Recipe recipe;
    recipe.values = {descriptions[0], descriptions[1], descriptions[2], result_description};
    recipe.inputs = {0, 1, 2}; recipe.outputs = {3};
    recipe.nodes.push_back({operation, {0, 1, 2}, {3}});
    program.compile(std::move(recipe));
  }
  void upload(std::size_t active) {
    std::vector<std::uint16_t> activation(std::size_t(C * K),
        type == c::DataType::kFloat16 ? std::uint16_t{0x7e55} : std::uint16_t{0x7fc5});
    for (std::size_t row = 0; row < active; ++row)
      for (std::size_t column = 0; column < std::size_t(K); ++column)
        activation[row * K + column] = bits(type, input_value(row, column));
    need(storage[0].view().write(std::as_bytes(std::span(activation))));
    offsets({0, static_cast<std::int32_t>(active)});
  }
  void offsets(std::array<std::int32_t, 2> value) {
    need(storage[2].view().write(std::as_bytes(std::span(value))));
  }
  hyle::Status configure(std::size_t active) {
    return native::configure_rows(program.prepared, program.invocation, 2, inputs[2], active);
  }
  void poison() { need(outputs[0].storage().fill(std::byte{0x5a})); }
  void expect_poison() const {
    std::vector<std::byte> actual(outputs[0].storage().size());
    ASSERT_TRUE(outputs[0].storage().read(actual).ok());
    EXPECT_EQ(actual, std::vector<std::byte>(actual.size(), std::byte{0x5a}));
    check_guards();
  }
  void check(std::size_t active) const {
    std::vector<std::byte> actual(outputs[0].storage().size()), expected(actual.size(), std::byte{0});
    ASSERT_TRUE(outputs[0].storage().read(actual).ok());
    for (std::size_t row = 0; row < active; ++row) for (std::size_t column = 0; column < std::size_t(N); ++column) {
      const auto at = row * N + column;
      const auto value = input_value(row, column + 1) * .5;
      if (output_type == c::DataType::kFloat32) {
        const auto word = std::bit_cast<std::array<std::byte, sizeof(float)>>(static_cast<float>(value));
        std::copy(word.begin(), word.end(), expected.begin() + at * sizeof(float));
      } else {
        const auto word = std::bit_cast<std::array<std::byte, 2>>(bits(output_type, value));
        std::copy(word.begin(), word.end(), expected.begin() + at * 2);
      }
    }
    EXPECT_EQ(actual, expected);
    check_guards();
  }
  void check_guards() const {
    std::array<std::byte, 32> actual, expected;
    expected.fill(std::byte{0xa5});
    ASSERT_TRUE(result.view().subview(result_description.required_bytes().value(), 32).value().read(actual).ok());
    EXPECT_EQ(actual, expected); program.check_workspace_guard();
  }
};

TEST(GgmlCudaPackedRowMatMul, InferenceAcceptsOnlyExplicitDenseOrTransposedMatrixDomains) {
  const auto operation = c::sequence_ops::packed_row_matmul(c::DataType::kFloat32);
  std::array<c::TensorDesc, 3> inputs{dense(c::DataType::kBFloat16, {C, K}),
      dense(c::DataType::kBFloat16, {K, N}), dense(c::DataType::kInt32, {2})};
  std::array<c::TensorDesc, 1> output;
  ASSERT_TRUE(c::infer_output_descriptions(operation, inputs, output).ok());
  EXPECT_EQ(output[0], dense(c::DataType::kFloat32, {C, N}));
  inputs[1] = dense(c::DataType::kBFloat16, {N, K}).transposed().value();
  EXPECT_TRUE(c::infer_output_descriptions(operation, inputs, output).ok());
  const std::array<std::int64_t, 2> shape{K, N}, strides{N * 2 + 16, 2};
  inputs[1] = c::TensorDesc::strided_bytes(c::DataType::kBFloat16, shape, strides).value();
  EXPECT_FALSE(c::infer_output_descriptions(operation, inputs, output).ok());
  inputs[1] = dense(c::DataType::kFloat16, {K, N});
  EXPECT_FALSE(c::infer_output_descriptions(operation, inputs, output).ok());
}

TEST(GgmlCudaPackedRowMatMul, DirectAndUnconfiguredNativeCannotInventHostProof) {
  Environment environment;
  Case test(environment, c::DataType::kBFloat16, c::DataType::kFloat32, true);
  auto direct = c::prepare(environment.context, c::sequence_ops::packed_row_matmul(test.output_type),
      test.descriptions, std::array{test.result_description}, environment.memory.ref());
  EXPECT_FALSE(direct);
  if (!direct) EXPECT_TRUE(direct.error().IsUnimplemented()) << direct.error().message();
  test.upload(17); test.poison();
  auto unconfigured = test.program.submit(test.inputs, test.outputs);
  EXPECT_TRUE(unconfigured.IsUnimplemented()) << unconfigured.message();
  test.expect_poison();
  EXPECT_FALSE(test.configure(C + 1).ok());
  ASSERT_TRUE(test.configure(17).ok());
  ASSERT_TRUE(test.program.submit(test.inputs, test.outputs).ok()); test.check(17);
  ASSERT_TRUE(native::clear_rows(test.program.prepared, test.program.invocation).ok());
  test.poison();
  EXPECT_TRUE(test.program.submit(test.inputs, test.outputs).IsUnimplemented()); test.expect_poison();
}

TEST(GgmlCudaPackedRowMatMul, PrefixTailAndSameAddressLengthChangesKeepOriginalWorkspace) {
  Environment environment;
  for (const auto type : {c::DataType::kFloat16, c::DataType::kBFloat16}) {
    SCOPED_TRACE(static_cast<int>(type));
    for (const auto output : {c::DataType::kFloat32, c::DataType::kFloat16, c::DataType::kBFloat16}) {
      SCOPED_TRACE(static_cast<int>(output));
      for (const bool transposed : {false, true}) {
        SCOPED_TRACE(transposed);
        Case test(environment, type, output, transposed);
        const auto workspace = test.program.workspace.view();
        const auto bytes = test.program.prepared.workspace.size;
        for (const auto active : {0U, 1U, 17U, 128U, 1U}) {
          SCOPED_TRACE(active);
          test.upload(active); test.poison(); ASSERT_TRUE(test.configure(active).ok());
          const auto status = test.program.submit(test.inputs, test.outputs);
          ASSERT_TRUE(status.ok()) << status.message(); test.check(active);
          EXPECT_EQ(test.program.workspace.view().native_handle(), workspace.native_handle());
          EXPECT_EQ(test.program.workspace.view().offset(), workspace.offset());
          EXPECT_EQ(test.program.prepared.workspace.size, bytes);
        }
      }
    }
  }
}

TEST(GgmlCudaPackedRowMatMul, InvalidOffsetsHostMismatchAndInheritedErrorPublishNothing) {
  Environment environment;
  Case test(environment, c::DataType::kBFloat16, c::DataType::kFloat32, true);
  for (const auto offsets : {std::array<std::int32_t, 2>{1, 17}, std::array<std::int32_t, 2>{-1, 17},
      std::array<std::int32_t, 2>{0, -1}, std::array<std::int32_t, 2>{0, 129},
      std::array<std::int32_t, 2>{0, 16}}) {
    SCOPED_TRACE(offsets[0]);
    SCOPED_TRACE(offsets[1]);
    test.upload(17); test.offsets(offsets); test.poison(); ASSERT_TRUE(test.configure(17).ok());
    const auto bad = test.program.submit(test.inputs, test.outputs);
    EXPECT_TRUE(bad.IsInvalidArgument()) << bad.message(); test.expect_poison();
    test.program.fresh(); test.upload(17); ASSERT_TRUE(test.configure(17).ok());
    ASSERT_TRUE(test.program.submit(test.inputs, test.outputs).ok()); test.check(17);
  }
  Case good(environment, c::DataType::kFloat16, c::DataType::kFloat16, false);
  test.upload(17); test.offsets({0, 129}); test.poison(); ASSERT_TRUE(test.configure(17).ok());
  good.upload(1); good.poison(); ASSERT_TRUE(good.configure(1).ok());
  auto sequence = cd::QueueSequence::begin(environment.queue).value();
  sequence.record(test.program.enqueue(sequence, test.inputs, test.outputs));
  sequence.record(good.program.enqueue(sequence, good.inputs, good.outputs));
  auto done = sequence.finish().value(); EXPECT_TRUE(done.wait().IsInvalidArgument()); done = {};
  test.expect_poison(); good.expect_poison();
}
} // namespace

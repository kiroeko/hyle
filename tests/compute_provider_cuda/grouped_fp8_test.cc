#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/cuda.h"
#include "hyle/graph/executable.h"
#include "../compute_provider_native/half_test_support.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
using BF16 = hyle::simd::bfloat16_t;

struct Shape final {
  c::DataType type;
  int experts, tokens, slots;
  bool shared;
};

// These cross-cases exercise both alignment paths without a Cartesian matrix.
constexpr std::array kDirect{
    Shape{c::DataType::kFloat32, 9, 3, 3, true},
    Shape{c::DataType::kBFloat16, 9, 3, 6, false},
    Shape{c::DataType::kFloat32, 65, 2, 10, false},
    Shape{c::DataType::kBFloat16, 65, 1, 16, true},
    Shape{c::DataType::kFloat32, 9, 65, 16, true}};
constexpr std::array kGraph{
    Shape{c::DataType::kBFloat16, 65, 3, 10, true},
    Shape{c::DataType::kFloat32, 9, 2, 6, false}};

struct Tensor final {
  m::Buffer owner;
  c::TensorView view;
  Tensor(c::TensorDesc description, m::MemoryResourceRef resource)
      : owner(m::Buffer::create(resource, description.required_bytes().value(),
                                256).value()),
        view(c::TensorView::create(description, owner.view()).value()) {}
  template <class T> void write(const std::vector<T> &values) {
    const auto status = view.storage().write(std::as_bytes(std::span(values)));
    ASSERT_TRUE(status.ok()) << status.message();
  }
  std::vector<double> read() const {
    const auto size = view.description().element_count().value();
    std::vector<double> result(size);
    if (view.description().data_type() == c::DataType::kFloat32) {
      std::vector<float> raw(size);
      const auto status = view.storage().read(std::as_writable_bytes(std::span(raw)));
      EXPECT_TRUE(status.ok()) << status.message();
      std::copy(raw.begin(), raw.end(), result.begin());
    } else {
      std::vector<BF16> raw(size);
      const auto status = view.storage().read(std::as_writable_bytes(std::span(raw)));
      EXPECT_TRUE(status.ok()) << status.message();
      std::transform(raw.begin(), raw.end(), result.begin(),
                     [](BF16 x) { return half_test::decode(x); });
    }
    return result;
  }
};

// Mathematical E4M3FN decoding, independent of the native quantizer/kernel.
double decode_fp8(std::uint8_t bits) {
  const auto exponent = (bits >> 3) & 15U;
  const auto mantissa = bits & 7U;
  const double sign = (bits & 128U) ? -1.0 : 1.0;
  if (exponent == 15 && mantissa == 7)
    return std::numeric_limits<double>::quiet_NaN();
  return sign * (exponent == 0
      ? std::ldexp(double(mantissa), -9)
      : std::ldexp(1.0 + double(mantissa) / 8.0, int(exponent) - 7));
}
bool exact_fp8(double value) {
  for (unsigned code = 0; code < 256; ++code)
    if (decode_fp8(static_cast<std::uint8_t>(code)) == value) return true;
  return false;
}

struct Fixture final {
  static constexpr int k = 128, n = 129;
  Shape shape;
  int activations;
  std::array<c::TensorDesc, 3> descriptions;
  c::TensorDesc output;
  std::vector<std::byte> weights;
  std::vector<double> activation;
  explicit Fixture(Shape s)
      : shape(s), activations(s.shared ? 1 : s.slots),
        descriptions{
            c::float8_grouped_weight_description(s.experts, n, k).value(),
            c::TensorDesc::dense(s.type, std::array<std::int64_t, 3>{
                s.tokens, activations, k}).value(),
            c::TensorDesc::dense(c::DataType::kInt32,
                std::array<std::int64_t, 2>{s.tokens, s.slots}).value()},
        output(c::infer_single_output(c::model_ops::grouped_float8_linear(),
                                      descriptions).value()),
        weights(descriptions[0].required_bytes().value(), std::byte{0}),
        activation(std::size_t(s.tokens) * activations * k, 0) {
    const auto payload = std::size_t(s.experts) * n * k;
    for (int expert = 0; expert < s.experts; ++expert) {
      for (int column = 0; column < n; ++column) {
        const int feature = column % 3;
        constexpr std::array<std::uint8_t, 3> code{0x38, 0xb8, 0x30};
        weights[(std::size_t(expert) * n + column) * k + feature] =
            std::byte{code[feature]};
      }
      for (int block = 0; block < 2; ++block) {
        const float scale = float((expert + 1) * (block + 1));
        std::memcpy(weights.data() + payload +
                        (std::size_t(expert) * 2 + block) * sizeof(float),
                    &scale, sizeof(scale));
      }
    }
    for (int token = 0; token < s.tokens; ++token)
      for (int slot = 0; slot < activations; ++slot) {
        auto *row = activation.data() +
            (std::size_t(token) * activations + slot) * k;
        row[0] = double(token % 4 + 1);
        row[1] = double(slot % 16 + 1);
        row[2] = -double((token % 4 + 1) * (slot % 4 + 1));
        // Each group has absmax 448, so its native quantization scale is 1.
        // The witness weights ignore this maximum; all dot terms are exact.
        row[127] = 448.0;
      }
  }
  std::vector<std::int32_t> ids(int phase) const {
    std::vector<std::int32_t> result(std::size_t(shape.tokens) * shape.slots);
    for (int token = 0; token < shape.tokens; ++token)
      for (int slot = 0; slot < shape.slots; ++slot) {
        // Adjacent slots repeat an expert, with distinct AS activation rows.
        const int id = (token * (phase ? 7 : 11) + slot / 2) % shape.experts;
        result[std::size_t(token) * shape.slots + slot] =
            phase ? shape.experts - 1 - id : id;
      }
    return result;
  }
  std::vector<double> expected(const std::vector<std::int32_t> &selected) const {
    std::vector<double> result(selected.size() * n);
    const auto payload = std::size_t(shape.experts) * n * k;
    for (int token = 0; token < shape.tokens; ++token)
      for (int slot = 0; slot < shape.slots; ++slot)
        for (int column = 0; column < n; ++column) {
          const auto index = std::size_t(token) * shape.slots + slot;
          const auto expert = selected[index];
          const auto feature = column % 3;
          const auto code = std::to_integer<std::uint8_t>(weights[
              (std::size_t(expert) * n + column) * k + feature]);
          float scale = 0;
          std::memcpy(&scale, weights.data() + payload +
                          (std::size_t(expert) * 2 + column / 128) * sizeof(float),
                      sizeof(scale));
          double value = decode_fp8(code) * double(scale) * activation[
              (std::size_t(token) * activations + (shape.shared ? 0 : slot)) * k +
              feature];
          if (shape.type == c::DataType::kBFloat16)
            value = half_test::decode(half_test::encode<BF16>(value));
          result[index * n + column] = value;
        }
    return result;
  }
  void check_exact_inputs() const {
    for (const double x : activation) ASSERT_TRUE(exact_fp8(x)) << x;
    EXPECT_NE(ids(0), ids(1));
    if (!shape.shared) {
      const auto values = expected(ids(0));
      EXPECT_NE(values[1], values[n + 1]);
    }
  }
};

struct Inputs final {
  Tensor weight, activation;
  std::array<Tensor, 2> ids;
  Inputs(const Fixture &f, m::MemoryResourceRef resource)
      : weight(f.descriptions[0], resource), activation(f.descriptions[1], resource),
        ids{Tensor(f.descriptions[2], resource), Tensor(f.descriptions[2], resource)} {
    weight.write(f.weights);
    if (f.shape.type == c::DataType::kFloat32) {
      std::vector<float> values(f.activation.size());
      std::transform(f.activation.begin(), f.activation.end(), values.begin(),
                     [](double x) { return static_cast<float>(x); });
      activation.write(values);
    } else {
      std::vector<BF16> values(f.activation.size());
      std::transform(f.activation.begin(), f.activation.end(), values.begin(),
                     [](double x) { return half_test::encode<BF16>(x); });
      activation.write(values);
    }
    ids[0].write(f.ids(0)); ids[1].write(f.ids(1));
  }
  auto views(int phase) const {
    return std::array{weight.view.as_const(), activation.view.as_const(),
                      ids[phase].view.as_const()};
  }
};

c::Context context() {
  return c::Context::create(c::cuda::device(0).value(),
                            c::cuda_native::sglang_provider()).value();
}
void expect(const Tensor &output, const std::vector<double> &expected) {
  const auto actual = output.read();
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i)
    EXPECT_EQ(actual[i], expected[i]) << "element=" << i;
}
void direct(Shape shape, bool invalid_ids = false) {
  SCOPED_TRACE(::testing::Message() << "dtype=" << int(shape.type)
      << " E=" << shape.experts << " T=" << shape.tokens
      << " S=" << shape.slots << " A=" << (shape.shared ? 1 : shape.slots));
  Fixture f(shape); f.check_exact_inputs();
  auto owner = context();
  auto memory = c::cuda::MemoryResource::create(owner, c::cuda::MemoryKind::kDevice).value();
  Inputs inputs(f, memory.ref());
  std::array outputs{Tensor(f.output, memory.ref()), Tensor(f.output, memory.ref())};
  auto prepared = c::prepare(owner, c::model_ops::grouped_float8_linear(),
      f.descriptions, std::array{f.output}, memory.ref());
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, c::cuda_native::kSglangProviderId);
  const auto need = prepared->workspace_requirement();
  auto workspace = m::Buffer::create(memory.ref(), need.size, need.alignment).value();
  auto invocation = prepared->create_invocation(workspace.view()).value();
  auto queue = owner.create_queue().value();
  for (int phase = 0; phase < 2; ++phase) {
    auto submitted = invocation.submit(queue, inputs.views(phase),
                                       std::array{outputs[phase].view});
    ASSERT_TRUE(submitted) << submitted.error().message();
    const auto status = submitted->wait();
    ASSERT_TRUE(status.ok()) << status.message();
    expect(outputs[phase], f.expected(f.ids(phase)));
  }
  // Direct outputs are caller-owned; using separate buffers preserves the old
  // value without claiming GraphOutput's lease protection for Direct buffers.
  expect(outputs[0], f.expected(f.ids(0)));
  if (!invalid_ids) return;
  auto bad = f.ids(1);
  bad[0] = -1; bad[1] = shape.experts;
  bad[2] = std::numeric_limits<std::int32_t>::min();
  bad[3] = std::numeric_limits<std::int32_t>::max();
  inputs.ids[1].write(bad);
  auto accepted = invocation.submit(queue, inputs.views(1), std::array{outputs[1].view});
  ASSERT_TRUE(accepted) << accepted.error().message();
  EXPECT_TRUE(accepted->wait().IsOutOfRange());
  // Failed outputs are undefined by the public contract and are not observed.
  invocation = {};
  inputs.ids[1].write(f.ids(1));
  auto recovered = prepared->create_invocation(workspace.view()).value();
  auto recovery_queue = owner.create_queue().value();
  auto done = recovered.submit(recovery_queue, inputs.views(1), std::array{outputs[1].view});
  ASSERT_TRUE(done) << done.error().message();
  const auto status = done->wait();
  ASSERT_TRUE(status.ok()) << status.message();
  expect(outputs[1], f.expected(f.ids(1)));
  expect(outputs[0], f.expected(f.ids(0)));
}

void graph(Shape shape) {
  SCOPED_TRACE(::testing::Message() << "dtype=" << int(shape.type)
      << " E=" << shape.experts << " T=" << shape.tokens
      << " S=" << shape.slots << " A=" << (shape.shared ? 1 : shape.slots));
  Fixture f(shape); f.check_exact_inputs();
  auto owner = context();
  auto memory = c::cuda::MemoryResource::create(owner, c::cuda::MemoryKind::kDevice).value();
  Inputs inputs(f, memory.ref());
  auto builder = g::GraphBuilder::create().value();
  std::array<g::ValueId, 3> ports;
  for (std::size_t i = 0; i < ports.size(); ++i)
    ports[i] = builder.input(f.descriptions[i]).value();
  const auto output = builder.add(c::model_ops::grouped_float8_linear(),
                                  ports[0], ports[1], ports[2]).value();
  ASSERT_TRUE(builder.publish(output).ok());
  const auto program = std::move(builder).finish().value();
  const std::array storage{g::StorageDomain{memory.ref()}};
  const std::array targets{g::ExecutionTarget{
      g::ExecutionSource::acquire(owner).value(), g::StorageDomainId{0}}};
  auto compiled = g::compile(program, {storage, targets, {}});
  ASSERT_TRUE(compiled) << compiled.error().message();
  bool selected = false;
  for (std::size_t i = 0; i < compiled->report.step_count(); ++i)
    selected |= compiled->report.step(i)->provider.id == c::cuda_native::kSglangProviderId;
  EXPECT_TRUE(selected);
  std::array outputs{Tensor(f.output, memory.ref()), Tensor(f.output, memory.ref())};
  std::array lanes{compiled->executable.create_invocation().value(),
                   compiled->executable.create_invocation().value()};
  auto queue = owner.create_queue().value();
  const auto bind = [&](int lane, int phase) {
    const auto views = inputs.views(phase);
    for (std::size_t i = 0; i < ports.size(); ++i)
      ASSERT_TRUE(lanes[lane].bind(ports[i], views[i]).ok());
    ASSERT_TRUE(lanes[lane].bind(output, outputs[lane].view).ok());
  };
  bind(0, 0);
  auto first = lanes[0].submit(queue);
  ASSERT_TRUE(first) << first.error().message();
  ASSERT_TRUE(first->wait().ok());
  *first = c::Completion{};
  auto held = lanes[0].result(output);
  ASSERT_TRUE(held) << held.error().message();
  EXPECT_TRUE(lanes[0].bind(ports[2], inputs.ids[1].view.as_const()).IsUnavailable());
  bind(1, 1);
  auto second = lanes[1].submit(queue);
  ASSERT_TRUE(second) << second.error().message();
  ASSERT_TRUE(second->wait().ok());
  *second = c::Completion{};
  expect(outputs[0], f.expected(f.ids(0)));
  expect(outputs[1], f.expected(f.ids(1)));
  held->reset();
  ASSERT_TRUE(lanes[0].reset().ok());
  bind(0, 1);
  auto rebound = lanes[0].submit(queue);
  ASSERT_TRUE(rebound) << rebound.error().message();
  const auto status = rebound->wait();
  ASSERT_TRUE(status.ok()) << status.message();
  expect(outputs[0], f.expected(f.ids(1)));
}

TEST(CudaGroupedFp8, RuntimeSlotsAndExactRowsDirect) {
  for (const auto shape : kDirect) direct(shape);
}
TEST(CudaGroupedFp8, RuntimeSlotsAndExactRowsGraph) {
  for (const auto shape : kGraph) graph(shape);
}
TEST(CudaGroupedFp8, CheckedIdsFailCompletionAndFreshInvocationRecovers) {
  for (const auto type : {c::DataType::kFloat32, c::DataType::kBFloat16})
    direct({type, 65, 2, 10, false}, true);
}

// This is a semantic witness, not a tolerance comparison with the product
// quantizer or a second invocation of the same GPU implementation. Every
// encoded weight is positive zero, so its real scaled value is exactly zero.
// A=896 and As=896/448=2 are exact in F32/BF16/E4M3FN after quantization.
TEST(CudaGroupedFp8, ZeroCodesAndFiniteScalesKeepZeroProducts) {
  const float weight_scale = std::ldexp(1.0F, 127);
  const float activation_scale = 896.0F / 448.0F;
  ASSERT_TRUE(std::isfinite(weight_scale));
  ASSERT_EQ(activation_scale, 2.0F);
  volatile float dot = 0.0F;
  volatile float scaled_dot = dot * activation_scale;
  volatile float sequential_result = scaled_dot * weight_scale;
  volatile float combined_scale = activation_scale * weight_scale;
  volatile float prescaled_result = dot * combined_scale;
  ASSERT_EQ(sequential_result, 0.0F);
  ASSERT_TRUE(std::isinf(combined_scale));
  ASSERT_TRUE(std::isnan(prescaled_result));

  auto context = c::Context::create(c::cuda::device(0).value(),
                                   c::cuda_native::sglang_provider()).value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  auto queue = context.create_queue().value();
  constexpr std::int64_t e = 1, n = 128, k = 128;
  for (const auto type : {c::DataType::kFloat32, c::DataType::kBFloat16}) {
    const bool bfloat = type == c::DataType::kBFloat16;
    const std::string label = bfloat ? "bf16" : "f32";
    SCOPED_TRACE(label);
    const std::array descriptions{
        c::float8_grouped_weight_description(e, n, k).value(),
        c::TensorDesc::dense(type, std::array<std::int64_t, 3>{1, 1, k}).value(),
        c::TensorDesc::dense(c::DataType::kInt32,
                             std::array<std::int64_t, 2>{1, 1}).value()};
    const auto operation = c::model_ops::grouped_float8_linear();
    auto inferred = c::infer_single_output(operation, descriptions);
    ASSERT_TRUE(inferred) << inferred.error().message();
    const auto output_description = *inferred;
    Tensor weights(descriptions[0], memory.ref());
    Tensor activation(descriptions[1], memory.ref());
    Tensor ids(descriptions[2], memory.ref());
    Tensor output(output_description, memory.ref());
    const auto payload = static_cast<std::size_t>(e * n * k);
    std::vector<std::byte> encoded(descriptions[0].required_bytes().value(), std::byte{0});
    ASSERT_EQ(encoded.size(), payload + sizeof(float));
    std::memcpy(encoded.data() + payload, &weight_scale, sizeof(weight_scale));
    weights.write(encoded);
    if (bfloat) {
      std::vector<std::uint16_t> values(static_cast<std::size_t>(k), 0);
      const auto bits = std::bit_cast<std::uint32_t>(896.0F);
      ASSERT_EQ(bits & 0xffffU, 0U);
      values[0] = static_cast<std::uint16_t>(bits >> 16);
      activation.write(values);
    } else {
      std::vector<float> values(static_cast<std::size_t>(k), 0.0F);
      values[0] = 896.0F;
      activation.write(values);
    }
    ids.write(std::vector<std::int32_t>{0});
    auto prepared = c::prepare(context, operation, descriptions,
                               std::array{output_description}, memory.ref());
    ASSERT_TRUE(prepared) << prepared.error().message();
    EXPECT_EQ(prepared->provider().id, c::cuda_native::kSglangProviderId);
    const auto need = prepared->workspace_requirement();
    auto workspace = m::Buffer::create(memory.ref(), need.size, need.alignment).value();
    auto invocation = prepared->create_invocation(workspace.view()).value();
    const std::array inputs{weights.view.as_const(), activation.view.as_const(),
                            ids.view.as_const()};
    auto submitted = invocation.submit(queue, inputs, std::array{output.view});
    ASSERT_TRUE(submitted) << submitted.error().message();
    const auto status = submitted->wait();
    ASSERT_TRUE(status.ok()) << status.message();
    const auto actual = output.read();
    ASSERT_EQ(actual.size(), static_cast<std::size_t>(n));
    int nan_count = 0, infinite_count = 0, finite_nonzero_count = 0;
    for (const auto value : actual) {
      if (std::isnan(value)) ++nan_count;
      else if (std::isinf(value)) ++infinite_count;
      else if (value != sequential_result) ++finite_nonzero_count;
    }
    RecordProperty(label + "_nan_count", nan_count);
    RecordProperty(label + "_infinite_count", infinite_count);
    RecordProperty(label + "_finite_nonzero_count", finite_nonzero_count);
    RecordProperty(label + "_first_f32_bits",
                   std::to_string(std::bit_cast<std::uint32_t>(static_cast<float>(actual.front()))));
    EXPECT_EQ(nan_count, 0) << "Finite input and finite scaled zero weights produced NaN";
    EXPECT_EQ(infinite_count, 0);
    EXPECT_EQ(finite_nonzero_count, 0);
  }
}
} // namespace

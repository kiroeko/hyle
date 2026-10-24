#include "hyle/graph/cuda.h"
#include "hyle/graph/providers/ggml_cuda.h"

#include "../compute_provider_ggml_cuda/tensor_test_support.h"
#include "../graph_provider_ggml_cuda/region_test_support.h"
#include "hyle/compute/providers/cublas.h"
#include "hyle/compute/providers/cuda.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"

#include "../../src/compute_ggml_runtime/explicit_rope.h"
#include "../../src/compute_provider_ggml_cuda/native_bridge.h"

#include <ggml-alloc.h>
#include <ggml-impl.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
using BF16 = hyle::simd::bfloat16_t;
using hyle::mem::Buffer;
using ggml_cuda_tensor_test::dense;
using ggml_cuda_tensor_test::output_buffer;
using ggml_cuda_tensor_test::upload;
constexpr std::int64_t kWidth = 32, kInner = 8;

// Independent FP64 reference: in particular, exp(+100) is never evaluated,
// and ordinary Unary's limiting value at -infinity is positive zero.
double stable_silu(double x) {
  if (x == -std::numeric_limits<double>::infinity())
    return 0.0;
  const double e = std::exp(-std::abs(x));
  return x * (x >= 0 ? 1.0 / (1.0 + e) : e / (1.0 + e));
}

template <class T> T activated(T x) {
  return half_test::encode<T>(static_cast<float>(stable_silu(half_test::decode(x))));
}

template <class T> T multiplied(T x, T y) {
  return half_test::encode<T>(static_cast<float>(half_test::decode(x)) *
                              static_cast<float>(half_test::decode(y)));
}

template <class T> void expect_value(T actual, T expected) {
  const auto a = half_test::decode(actual), e = half_test::decode(expected);
  if (std::isnan(e)) {
    EXPECT_TRUE(std::isnan(a));
  } else if constexpr (std::is_same_v<T, BF16>) {
    EXPECT_EQ(actual.bits, expected.bits);
  } else if (std::isinf(e) || e == 0.0) {
    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual),
              std::bit_cast<std::uint32_t>(expected));
  } else {
    // expf's subnormal rounding is magnified by |x| in the negative tail.
    // This budget covers it while still rejecting zero for our -80..-100
    // samples; the independent SGLang comparison below is bit-exact.
    EXPECT_NEAR(a, e, std::abs(e) * 3e-6 +
                          256.0 * std::numeric_limits<float>::denorm_min());
  }
}

template <class T>
void check(const Buffer &buffer, const std::vector<T> &expected) {
  const auto actual = half_test::read<T>(buffer);
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    SCOPED_TRACE(i);
    expect_value(actual[i], expected[i]);
  }
}

template <class T>
void unchanged(const Buffer &buffer, const std::vector<T> &snapshot) {
  const auto actual = half_test::read<T>(buffer);
  ASSERT_EQ(actual.size(), snapshot.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if constexpr (std::is_same_v<T, BF16>)
      EXPECT_EQ(actual[i].bits, snapshot[i].bits) << i;
    else
      EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]),
                std::bit_cast<std::uint32_t>(snapshot[i])) << i;
  }
}

template <class T>
void direct_result(c::Provider provider, const std::vector<T> &host,
                   std::vector<T> &observed) {
  auto context = c::Context::create(c::cuda::device(0).value(), provider).value();
  auto memory = c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kDevice).value();
  const auto description = dense(half_test::data_type<T>(), {1, kWidth});
  auto input = upload<T>(memory.ref(), host);
  auto output = output_buffer(memory.ref(), description);
  const auto source = c::ConstTensorView::create(description, input.view()).value();
  const auto destination = c::TensorView::create(description, output.view()).value();
  auto prepared = c::prepare(context, c::Unary{c::UnaryKind::kSilu},
                             description, memory.ref());
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, provider.identity().id);
  auto workspace = ggml_cuda_test::workspace_buffer(*prepared, memory.ref());
  auto queue = context.create_queue().value();
  auto invocation = prepared->create_invocation(workspace.view()).value();
  auto done = invocation.submit(queue, source, destination);
  ASSERT_TRUE(done) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  observed = half_test::read<T>(output);
}

template <class T> void direct() {
  SCOPED_TRACE(int(half_test::data_type<T>()));
  const float inf = std::numeric_limits<float>::infinity();
  const std::array<float, kWidth> values{
      -100, -96, -94, -92, -90, -88, -86, -84, -82, -80,
      -inf, inf, std::numeric_limits<float>::quiet_NaN(), 0.0F, -0.0F,
      -7.8125F, -4.125F, -3.5F, -2.25F, -1.0F, -0.25F, 0.125F,
      0.5F, 0.875F, 1.125F, 2.25F, 4.125F, 8.0F, 20.0F, 32.0F, 80, 100};
  std::vector<T> host;
  for (const auto value : values)
    host.push_back(half_test::encode<T>(value));
  std::vector<T> native, mature;
  direct_result(c::ggml_cuda::provider(), host, native);
  direct_result(c::cuda_native::sglang_provider(), host, mature);
  ASSERT_EQ(native.size(), host.size());
  ASSERT_EQ(mature.size(), host.size());
  for (std::size_t i = 0; i < host.size(); ++i) {
    SCOPED_TRACE(i);
    expect_value(native[i], activated(host[i]));
    // Compare the new marked native path with the existing precise SGLang
    // ordinary Unary branch, never its separately specified packed gate.
    if (std::isnan(half_test::decode(mature[i]))) {
      EXPECT_TRUE(std::isnan(half_test::decode(native[i])));
    } else if constexpr (std::is_same_v<T, BF16>) {
      EXPECT_EQ(native[i].bits, mature[i].bits);
    } else {
      EXPECT_EQ(std::bit_cast<std::uint32_t>(native[i]),
                std::bit_cast<std::uint32_t>(mature[i]));
    }
  }
}

template <class T> struct Inputs {
  std::vector<T> gate, left, weight, silu, product;
};

template <class T> Inputs<T> inputs(int phase, int slot) {
  Inputs<T> result;
  const auto seed = 1 + slot + phase * 3;
  for (std::int64_t k = 0; k < kInner; ++k)
    result.left.push_back(half_test::encode<T>(
        (k % 2 == 0 ? 1.0 : -0.5) + seed / 16.0));
  for (std::int64_t k = 0; k < kInner; ++k)
    for (std::int64_t col = 0; col < kWidth; ++col)
      result.weight.push_back(half_test::encode<T>(
          ((k * 7 + col * 3) % 23 - 11) / 16.0 + seed / 32.0));
  std::size_t rounding_witnesses = 0;
  for (std::int64_t col = 0; col < kWidth; ++col) {
    const auto gate = half_test::encode<T>(col < 5 ? -80.0 - 5.0 * col
        : (col % 19 - 9) / 4.0 + seed / 64.0);
    result.gate.push_back(gate);
    double sum = 0;
    for (std::int64_t k = 0; k < kInner; ++k)
      sum += half_test::decode(result.left[k]) *
             half_test::decode(result.weight[k * kWidth + col]);
    // Every dot product is a short sum of binary rationals, exact in F32.
    const auto up = half_test::encode<T>(static_cast<float>(sum));
    const auto silu = activated(gate);
    const auto product = multiplied(silu, up);
    result.silu.push_back(silu);
    result.product.push_back(product);
    if constexpr (std::is_same_v<T, BF16>)
      rounding_witnesses += product.bits != half_test::encode<T>(
          static_cast<float>(stable_silu(half_test::decode(gate))) *
          static_cast<float>(half_test::decode(up))).bits;
  }
  if constexpr (std::is_same_v<T, BF16>)
    EXPECT_GT(rounding_witnesses, 0U);
  return result;
}

template <class T> void noncontiguous_region() {
  SCOPED_TRACE(int(half_test::data_type<T>()));
  auto context = g::cuda::create_context().value();
  auto memory = c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kDevice).value();
  const auto type = half_test::data_type<T>();
  const auto description = dense(type, {1, kWidth});
  const auto left_description = dense(type, {1, kInner});
  const auto weight_description = dense(type, {kInner, kWidth});
  auto builder = g::GraphBuilder::create().value();
  const auto gate = builder.input(description).value();
  const auto left = builder.input(left_description).value();
  const auto weight = builder.input(weight_description).value();
  const auto silu = builder.add(c::Unary{c::UnaryKind::kSilu}, gate).value();
  // Preserve the real insertion order. The up projection is independent of
  // SiLU, but its output is an external input of the proposed two-node region.
  const auto up = builder.add(c::MatMul{}, left, weight).value();
  const auto product = builder.add(
      c::Binary{c::BinaryKind::kMultiply}, silu, up).value();
  ASSERT_TRUE(builder.publish(product).ok());
  auto logical = std::move(builder).finish().value();
  const auto silu_node = logical.producer(silu).value();
  const auto up_node = logical.producer(up).value();
  const auto product_node = logical.producer(product).value();
  ASSERT_LT(silu_node.ordinal(), up_node.ordinal());
  ASSERT_LT(up_node.ordinal(), product_node.ordinal());
  const std::array required{
      g::NodeProviderConstraint{up_node, c::cublas::kProviderId}};
  auto compiled = g::cuda::compile(logical, context, memory.ref(),
                                    {.node_providers = required});
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.step_count(), 2U);
  ASSERT_TRUE(compiled->report.mixed_providers());
  const auto projection = compiled->report.step(0).value();
  const auto region = compiled->report.step(1).value();
  EXPECT_EQ(projection.provider.id, c::cublas::kProviderId);
  ASSERT_EQ(projection.node_count, 1U);
  EXPECT_EQ(compiled->report.step_node(0, 0).value(), up_node);
  EXPECT_EQ(region.kind, g::CompileStepKind::kProviderRegion);
  EXPECT_EQ(region.provider.id, c::ggml_cuda::kProviderId);
  EXPECT_EQ(region.implementation, c::ProviderImplementationId{3});
  const c::RoutePolicyId silu_multiply_policy{0x48594c4547474355ULL, 2, 1};
  EXPECT_EQ(region.policy, silu_multiply_policy);
  ASSERT_EQ(region.node_count, 2U);
  EXPECT_FALSE(region.nodes_contiguous);
  EXPECT_EQ(compiled->report.step_node(1, 0).value(), silu_node);
  EXPECT_EQ(compiled->report.step_node(1, 1).value(), product_node);
  ASSERT_TRUE(projection.runtime.capture_compatible);
  ASSERT_TRUE(region.runtime.capture_compatible);

  // Owners precede queues/invocations: all borrowed bindings and retained
  // capture addresses survive native invocation teardown.
  std::vector<Buffer> retained;
  std::array queues{context.create_queue().value(), context.create_queue().value()};
  std::array runs{compiled->executable.create_invocation().value(),
                  compiled->executable.create_invocation().value()};
  std::vector<T> old;
  for (int phase = 0; phase < 2; ++phase) {
    std::array<Buffer, 2> gates, lefts, weights, outputs;
    std::array<Inputs<T>, 2> host;
    for (int slot = 0; slot < 2; ++slot) {
      host[slot] = inputs<T>(phase, slot);
      gates[slot] = upload<T>(memory.ref(), host[slot].gate);
      lefts[slot] = upload<T>(memory.ref(), host[slot].left);
      weights[slot] = upload<T>(memory.ref(), host[slot].weight);
      outputs[slot] = output_buffer(memory.ref(), description);
      ASSERT_TRUE(runs[slot].bind(gate, c::ConstTensorView::create(
          description, gates[slot].view()).value()).ok());
      ASSERT_TRUE(runs[slot].bind(left, c::ConstTensorView::create(
          left_description, lefts[slot].view()).value()).ok());
      ASSERT_TRUE(runs[slot].bind(weight, c::ConstTensorView::create(
          weight_description, weights[slot].view()).value()).ok());
      ASSERT_TRUE(runs[slot].bind(product, c::TensorView::create(
          description, outputs[slot].view()).value()).ok());
    }
    for (const auto mode : {g::ReplayMode::kCommands,
                            g::ReplayMode::kRequireCapture}) {
      std::array<c::Completion, 2> done;
      for (int slot = 0; slot < 2; ++slot) {
        const auto prepared = g::cuda::prepare_replay(runs[slot], queues[slot],
            {.mode = mode, .host_wait = g::HostWaitPolicy::kReject});
        ASSERT_TRUE(prepared.ok()) << prepared.message();
        EXPECT_EQ(runs[slot].replay_route(), mode == g::ReplayMode::kCommands
            ? g::ReplayRoute::kCommands : g::ReplayRoute::kCaptured);
      }
      for (int slot = 0; slot < 2; ++slot) {
        auto submitted = runs[slot].submit(queues[slot]);
        ASSERT_TRUE(submitted) << submitted.error().message();
        done[slot] = std::move(*submitted);
      }
      ASSERT_TRUE(done[1].wait().ok());
      ASSERT_TRUE(done[0].wait().ok());
      for (int slot = 0; slot < 2; ++slot)
        check(outputs[slot], host[slot].product);
    }
    for (int slot = 0; slot < 2; ++slot) {
      if (phase == 0 && slot == 0)
        old = half_test::read<T>(outputs[slot]);
      retained.push_back(std::move(gates[slot]));
      retained.push_back(std::move(lefts[slot]));
      retained.push_back(std::move(weights[slot]));
      retained.push_back(std::move(outputs[slot]));
    }
    // All phase-zero allocations remain live, so both slots really bind new
    // addresses. The old externally held output must retain its exact bits.
    if (phase == 1)
      unchanged(retained[3], old);
  }
}

enum class Escape { kPublished, kExtraReader, kBothMultiplyInputs };

void escaped_intermediate(Escape escape) {
  SCOPED_TRACE(int(escape));
  auto context = g::cuda::create_context().value();
  auto memory = c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kDevice).value();
  const auto description = dense(c::DataType::kBFloat16, {1, kWidth});
  const auto host = inputs<BF16>(0, 1);
  std::vector<BF16> other, expected, observed;
  for (std::size_t i = 0; i < host.gate.size(); ++i) {
    other.push_back(half_test::encode<BF16>((int(i % 7) - 3) / 4.0));
    expected.push_back(multiplied(host.silu[i],
        escape == Escape::kBothMultiplyInputs ? host.silu[i] : other.back()));
    observed.push_back(escape == Escape::kExtraReader
        ? half_test::encode<BF16>(-half_test::decode(host.silu[i])) : host.silu[i]);
  }
  auto gate_buffer = upload<BF16>(memory.ref(), host.gate);
  auto other_buffer = upload<BF16>(memory.ref(), other);
  auto output = output_buffer(memory.ref(), description);
  auto intermediate = output_buffer(memory.ref(), description);
  auto builder = g::GraphBuilder::create().value();
  const auto gate = builder.input(description).value();
  const auto second = escape == Escape::kBothMultiplyInputs
      ? gate : builder.input(description).value();
  const auto silu = builder.add(c::Unary{c::UnaryKind::kSilu}, gate).value();
  const auto product = builder.add(c::Binary{c::BinaryKind::kMultiply}, silu,
      escape == Escape::kBothMultiplyInputs ? silu : second).value();
  auto visible = silu;
  if (escape == Escape::kExtraReader)
    visible = builder.add(c::Unary{c::UnaryKind::kNeg}, silu).value();
  if (escape != Escape::kBothMultiplyInputs)
    ASSERT_TRUE(builder.publish(visible).ok());
  ASSERT_TRUE(builder.publish(product).ok());
  auto logical = std::move(builder).finish().value();
  auto compiled = g::cuda::compile(logical, context, memory.ref());
  ASSERT_TRUE(compiled) << compiled.error().message();
  // A generic region may still compute this graph. Correct published/extra
  // readers and the self-use are the contract; do not forbid all regions.
  for (std::size_t i = 0; i < compiled->report.step_count(); ++i) {
    const auto step = compiled->report.step(i).value();
    if (step.provider.id == c::ggml_cuda::kProviderId)
      EXPECT_NE(step.implementation, c::ProviderImplementationId{3});
  }
  auto queue = context.create_queue().value();
  auto run = compiled->executable.create_invocation().value();
  ASSERT_TRUE(run.bind(gate, c::ConstTensorView::create(
      description, gate_buffer.view()).value()).ok());
  if (escape != Escape::kBothMultiplyInputs)
    ASSERT_TRUE(run.bind(second, c::ConstTensorView::create(
        description, other_buffer.view()).value()).ok());
  ASSERT_TRUE(run.bind(product, c::TensorView::create(
      description, output.view()).value()).ok());
  if (escape != Escape::kBothMultiplyInputs)
    ASSERT_TRUE(run.bind(visible, c::TensorView::create(
        description, intermediate.view()).value()).ok());
  ASSERT_TRUE(g::cuda::prepare_replay(run, queue).ok());
  auto done = run.submit(queue);
  ASSERT_TRUE(done) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  check(output, expected);
  if (escape != Escape::kBothMultiplyInputs)
    check(intermediate, observed);
}

template <class T> void rank4_round_trip(bool published) {
  SCOPED_TRACE(::testing::Message() << "dtype=" << int(half_test::data_type<T>())
                                  << ", published=" << published);
  // Keep both branches on the native GGML composition. Default placement may
  // keep a published ordinary SiLU on SGLang and miss this reshape boundary.
  auto context = c::Context::create(c::cuda::device(0).value(),
                                    g::ggml_cuda::provider()).value();
  auto memory = c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kDevice).value();
  const auto description = dense(half_test::data_type<T>(), {1, 2, 3, 128});
  auto builder = g::GraphBuilder::create().value();
  const auto input = builder.input(description).value();
  const auto silu = builder.add(c::Unary{c::UnaryKind::kSilu}, input).value();
  // The same external input has two readers. Only SiLU's private result may
  // disappear; neither reuse of the input nor rank restoration is a Copy.
  const auto product = builder.add(
      c::Binary{c::BinaryKind::kMultiply}, silu, input).value();
  auto extra = silu;
  if (published) {
    extra = builder.add(c::Unary{c::UnaryKind::kNeg}, silu).value();
    ASSERT_TRUE(builder.publish(silu).ok());
    ASSERT_TRUE(builder.publish(extra).ok());
  }
  ASSERT_TRUE(builder.publish(product).ok());
  const std::array domains{g::StorageDomain{.resource = memory.ref()}};
  const std::array targets{g::ExecutionTarget{
      .source = g::ExecutionSource::acquire(context).value(),
      .default_storage = g::StorageDomainId{0}}};
  const std::array lanes{g::ExecutionLane{.target = g::ExecutionTargetId{0}}};
  auto compiled = g::compile(std::move(builder).finish().value(),
      {.storage_domains = domains, .targets = targets, .lanes = lanes});
  ASSERT_TRUE(compiled) << compiled.error().message();

  std::vector<Buffer> retained;
  std::array queues{context.create_queue().value(), context.create_queue().value()};
  std::array runs{compiled->executable.create_invocation().value(),
                  compiled->executable.create_invocation().value()};
  std::vector<T> old_product, old_silu, old_extra;
  std::size_t old_output_index = 0;
  for (int phase = 0; phase < 2; ++phase) {
    std::array<Buffer, 2> sources, outputs, intermediates, readers;
    std::array<std::vector<T>, 2> host, expected, activated_values, negated;
    std::array<c::Completion, 2> done;
    for (int slot = 0; slot < 2; ++slot) {
      const auto seed = 1 + slot + phase * 3;
      std::size_t witnesses = 0;
      for (std::size_t i = 0; i < description.element_count().value(); ++i) {
        const auto x = half_test::encode<T>((int(i % 41) - 20) / 8.0 + seed / 64.0);
        const auto rounded_silu = activated(x);
        const auto rounded_product = multiplied(rounded_silu, x);
        host[slot].push_back(x);
        activated_values[slot].push_back(rounded_silu);
        expected[slot].push_back(rounded_product);
        negated[slot].push_back(half_test::encode<T>(-half_test::decode(rounded_silu)));
        if constexpr (std::is_same_v<T, BF16>)
          witnesses += rounded_product.bits != half_test::encode<T>(
              static_cast<float>(stable_silu(half_test::decode(x))) *
              static_cast<float>(half_test::decode(x))).bits;
      }
      if constexpr (std::is_same_v<T, BF16>)
        EXPECT_GT(witnesses, 0U);
      sources[slot] = upload<T>(memory.ref(), host[slot]);
      outputs[slot] = output_buffer(memory.ref(), description);
      ASSERT_TRUE(runs[slot].bind(input, c::ConstTensorView::create(
          description, sources[slot].view()).value()).ok());
      ASSERT_TRUE(runs[slot].bind(product, c::TensorView::create(
          description, outputs[slot].view()).value()).ok());
      if (published) {
        intermediates[slot] = output_buffer(memory.ref(), description);
        readers[slot] = output_buffer(memory.ref(), description);
        ASSERT_TRUE(runs[slot].bind(silu, c::TensorView::create(
            description, intermediates[slot].view()).value()).ok());
        ASSERT_TRUE(runs[slot].bind(extra, c::TensorView::create(
            description, readers[slot].view()).value()).ok());
      }
      const std::array bindings{g::QueueBinding{g::ExecutionLaneId{0}, &queues[slot]}};
      ASSERT_TRUE(runs[slot].prepare_replay(bindings).ok());
    }
    for (int slot = 0; slot < 2; ++slot) {
      auto submitted = runs[slot].submit(queues[slot]);
      ASSERT_TRUE(submitted) << submitted.error().message();
      done[slot] = std::move(*submitted);
    }
    ASSERT_TRUE(done[1].wait().ok());
    ASSERT_TRUE(done[0].wait().ok());
    for (int slot = 0; slot < 2; ++slot) {
      check(outputs[slot], expected[slot]);
      if (published) {
        check(intermediates[slot], activated_values[slot]);
        check(readers[slot], negated[slot]);
      }
      if (phase == 0 && slot == 0) {
        old_output_index = retained.size() + 1;
        old_product = half_test::read<T>(outputs[slot]);
        if (published) {
          old_silu = half_test::read<T>(intermediates[slot]);
          old_extra = half_test::read<T>(readers[slot]);
        }
      }
      retained.push_back(std::move(sources[slot]));
      retained.push_back(std::move(outputs[slot]));
      if (published) {
        retained.push_back(std::move(intermediates[slot]));
        retained.push_back(std::move(readers[slot]));
      }
    }
    // Retaining every old allocation forces new addresses in both slots and
    // keeps every previously published output independently observable.
    if (phase == 1) {
      unchanged(retained[old_output_index], old_product);
      if (published) {
        unchanged(retained[old_output_index + 1], old_silu);
        unchanged(retained[old_output_index + 2], old_extra);
      }
    }
  }
}

TEST(CudaStableSilu, DirectOrdinarySiluKeepsStableTailsAndNonfiniteContract) {
  direct<float>();
  direct<BF16>();
}

TEST(CudaStableSilu, NoncontiguousSiluMulRegionCrossesCublasAndRebindsBothSlots) {
  noncontiguous_region<float>();
  noncontiguous_region<BF16>();
}

TEST(CudaStableSilu, PublishedExtraReaderAndSelfMultiplyKeepRoundedIntermediate) {
  for (const auto escape : {Escape::kPublished, Escape::kExtraReader,
                             Escape::kBothMultiplyInputs})
    escaped_intermediate(escape);
}

TEST(CudaStableSilu, DefaultRank2ProjectionKeepsBlasLeafPreference) {
  auto context = g::cuda::create_context().value();
  // No provider constraint: capability for plain rank2 MatMul must not make
  // the generic norm/add region consume the default cuBLAS projection.
  ggml_cuda_graph_test::exercise(context, true,
      [&](const g::Graph &logical, hyle::mem::MemoryResourceRef resource) {
        return g::cuda::compile(logical, context, resource);
      });
  // The ordinary full island would tie at preference400 and consume MatMul,
  // even though that leaf has a better cuBLAS300 implementation. Exercise
  // expert rankings through common Graph compilation, without route policy.
  const std::array providers{
      c::ProviderBinding{g::ggml_cuda::provider(), 400},
      c::ProviderBinding{c::cublas::provider(), 300}};
  auto expert = c::Context::create(c::cuda::device(0).value(),
                                  {.providers = providers}).value();
  ggml_cuda_graph_test::exercise(expert, true,
      [&](const g::Graph &logical, hyle::mem::MemoryResourceRef resource) {
        const std::array domains{g::StorageDomain{.resource = resource}};
        const std::array targets{g::ExecutionTarget{
            .source = g::ExecutionSource::acquire(expert).value(),
            .default_storage = g::StorageDomainId{0}}};
        const std::array lanes{
            g::ExecutionLane{.target = g::ExecutionTargetId{0}}};
        return g::compile(logical,
            {.storage_domains = domains, .targets = targets, .lanes = lanes});
      });
}

TEST(CudaStableSilu, Rank4RoundTripsKeepPublishedRoundingAndBothSlots) {
  for (const bool published : {false, true}) {
    rank4_round_trip<float>(published);
    rank4_round_trip<BF16>(published);
  }
}
} // namespace

// BF16 gate/up fusion numerical, route and invocation-slot regressions.
namespace {
namespace dual_projection_test {
constexpr std::int64_t kColumns = 33;
constexpr c::ProviderImplementationId kImplementation{4};
constexpr c::RoutePolicyId kPolicy{0x48594c4547474355ULL, 3, 1};

struct Data {
  std::int64_t inner = 0;
  std::vector<BF16> x, gate_weight, up_weight;
  std::vector<double> gate_dot, up_dot, gate_absolute, up_absolute;
};

Data data(std::int64_t inner, int seed, bool random = false) {
  Data d;
  d.inner = inner;
  d.x.resize(inner);
  d.gate_weight.resize(kColumns * inner);
  d.up_weight.resize(kColumns * inner);
  std::mt19937 generator(static_cast<std::uint32_t>(seed));
  const auto sample = [&]() {
    return (static_cast<int>(generator() % 8191) - 4095) / 4096.0;
  };
  for (std::int64_t k = 0; k < inner; ++k)
    d.x[k] = half_test::encode<BF16>(random ? sample()
        : (static_cast<int>(k % 19) - 9) / 16.0 + seed / 128.0);
  if (!random) d.x[0] = d.x[1] = half_test::encode<BF16>(1);
  for (std::int64_t n = 0; n < kColumns; ++n)
    for (std::int64_t k = 0; k < inner; ++k) {
      const auto i = n * inner + k;
      d.gate_weight[i] = half_test::encode<BF16>(random ? sample() / 32.0
          : (static_cast<int>((n * 7 + k * 3) % 23) - 11) / 64.0 + seed / 512.0);
      d.up_weight[i] = half_test::encode<BF16>(random ? sample() / 32.0
          : (static_cast<int>((n * 11 + k * 5) % 29) - 14) / 64.0 - seed / 512.0);
      if (!random && n < 2) {
        // Two opposite BF16 halfway parities, with exact F32 products/sums.
        d.gate_weight[i] = half_test::encode<BF16>(k == 0 ? 1.0 + n / 128.0
            : k == 1 ? 1.0 / 256.0 : 0.0);
        d.up_weight[i] = half_test::encode<BF16>(k == 0 ? 1.5 + n / 128.0
            : k == 1 ? 1.0 / 256.0 : 0.0);
      }
      if (!random && n >= kColumns - 3) {
        d.gate_weight[i] = half_test::encode<BF16>(k == 0
            ? -80.0 - 10.0 * (n - (kColumns - 3)) : 0.0);
      }
    }
  for (std::int64_t n = 0; n < kColumns; ++n) {
    double gate = 0, up = 0, gate_absolute = 0, up_absolute = 0;
    for (std::int64_t k = 0; k < inner; ++k) {
      const auto x = half_test::decode(d.x[k]);
      const auto g = x * half_test::decode(d.gate_weight[n * inner + k]);
      const auto u = x * half_test::decode(d.up_weight[n * inner + k]);
      gate += g; up += u;
      gate_absolute += std::abs(g); up_absolute += std::abs(u);
      if (!random) {
        EXPECT_EQ(static_cast<double>(static_cast<float>(gate)), gate);
        EXPECT_EQ(static_cast<double>(static_cast<float>(up)), up);
      }
    }
    d.gate_dot.push_back(gate); d.up_dot.push_back(up);
    d.gate_absolute.push_back(gate_absolute); d.up_absolute.push_back(up_absolute);
  }
  return d;
}

// These isolated single-operation native programs cannot match the new
// four-node graph. exp/SiLU is always evaluated by the old GPU ordinary Unary,
// never by a CPU exp reference, and every Cast/Multiply remains explicit.
struct Direct {
  c::Context context = c::Context::create(c::cuda::device(0).value(),
                                          c::ggml_cuda::provider()).value();
  c::cuda::MemoryResource memory = c::cuda::MemoryResource::create(
      context, c::cuda::MemoryKind::kDevice).value();
};

template <class Input, class Output, class Operation>
std::vector<Output> unary(Direct &direct, Operation operation,
                          const std::vector<Input> &host) {
  const auto input_description = dense(half_test::data_type<Input>(), {1, kColumns});
  const auto output_description = dense(half_test::data_type<Output>(), {1, kColumns});
  auto input = upload<Input>(direct.memory.ref(), host);
  auto output = output_buffer(direct.memory.ref(), output_description);
  const std::array bindings{c::ConstTensorView::create(
      input_description, input.view()).value()};
  const auto status = ggml_cuda_tensor_test::execute(direct.context, operation,
      bindings, c::TensorView::create(output_description, output.view()).value());
  EXPECT_TRUE(status.ok()) << status.message();
  return half_test::read<Output>(output);
}

template <class T>
std::vector<T> multiply(Direct &direct, const std::vector<T> &left,
                        const std::vector<T> &right) {
  const auto description = dense(half_test::data_type<T>(), {1, kColumns});
  auto x = upload<T>(direct.memory.ref(), left);
  auto y = upload<T>(direct.memory.ref(), right);
  auto output = output_buffer(direct.memory.ref(), description);
  const std::array bindings{c::ConstTensorView::create(description, x.view()).value(),
                            c::ConstTensorView::create(description, y.view()).value()};
  const auto status = ggml_cuda_tensor_test::execute(direct.context,
      c::Binary{c::BinaryKind::kMultiply}, bindings,
      c::TensorView::create(description, output.view()).value());
  EXPECT_TRUE(status.ok()) << status.message();
  return half_test::read<T>(output);
}

struct Oracle {
  std::vector<BF16> gate, up, activation, output;
  std::vector<float> activation_f32;
};

Oracle oracle(Direct &direct, const Data &host, bool witnesses) {
  Oracle o;
  std::vector<float> gate_f32, up_f32;
  for (std::int64_t n = 0; n < kColumns; ++n) {
    gate_f32.push_back(static_cast<float>(host.gate_dot[n]));
    up_f32.push_back(static_cast<float>(host.up_dot[n]));
    o.gate.push_back(half_test::encode<BF16>(gate_f32.back()));
    o.up.push_back(half_test::encode<BF16>(up_f32.back()));
  }
  o.activation = unary<BF16, BF16>(direct, c::Unary{c::UnaryKind::kSilu}, o.gate);
  o.output = multiply(direct, o.activation, o.up);
  const auto gate_wide = unary<BF16, float>(direct, c::Cast{c::DataType::kFloat32}, o.gate);
  const auto up_wide = unary<BF16, float>(direct, c::Cast{c::DataType::kFloat32}, o.up);
  o.activation_f32 = unary<float, float>(direct, c::Unary{c::UnaryKind::kSilu}, gate_wide);
  if (witnesses) {
    const auto activation_wide = unary<BF16, float>(direct, c::Cast{c::DataType::kFloat32}, o.activation);
    const auto raw_gate_activation = unary<float, float>(direct, c::Unary{c::UnaryKind::kSilu}, gate_f32);
    const auto omitted_gate_round = multiply(direct,
        unary<float, BF16>(direct, c::Cast{c::DataType::kBFloat16}, raw_gate_activation), o.up);
    const auto omitted_up_round = unary<float, BF16>(direct, c::Cast{c::DataType::kBFloat16},
        multiply(direct, activation_wide, up_f32));
    const auto omitted_activation_round = unary<float, BF16>(direct, c::Cast{c::DataType::kBFloat16},
        multiply(direct, o.activation_f32, up_wide));
    std::size_t gate = 0, up = 0, activation = 0;
    for (std::size_t n = 0; n < o.output.size(); ++n) {
      gate += o.output[n].bits != omitted_gate_round[n].bits;
      up += o.output[n].bits != omitted_up_round[n].bits;
      activation += o.output[n].bits != omitted_activation_round[n].bits;
    }
    EXPECT_GT(gate, 0U) << "data must distinguish lost G RNE";
    EXPECT_GT(up, 0U) << "data must distinguish lost U RNE";
    EXPECT_GT(activation, 0U) << "data must distinguish lost A RNE";
    EXPECT_EQ(o.gate[0].bits, 0x3f80U);
    EXPECT_EQ(o.gate[1].bits, 0x3f82U);
  }
  return o;
}

enum class Escape { kNone, kPublished, kExtraReader, kForeignActivation, kOddInner, kDenseRhs };
struct Graph {
  g::Graph logical;
  g::ValueId x, foreign_x, gate_weight, up_weight, gate, up, activation, output, reader;
  std::array<g::NodeId, 4> members;
  c::TensorDesc input_description, weight_description, output_description;
  Escape escape = Escape::kNone;
};

Graph graph(std::int64_t inner, Escape escape = Escape::kNone, bool up_first = false) {
  Graph result;
  result.escape = escape;
  result.input_description = dense(c::DataType::kBFloat16, {1, inner});
  result.weight_description = escape == Escape::kDenseRhs
      ? dense(c::DataType::kBFloat16, {inner, kColumns})
      : dense(c::DataType::kBFloat16, {kColumns, inner});
  result.output_description = dense(c::DataType::kBFloat16, {1, kColumns});
  auto builder = g::GraphBuilder::create().value();
  result.x = builder.input(result.input_description).value();
  result.foreign_x = escape == Escape::kForeignActivation
      ? builder.input(result.input_description).value() : result.x;
  result.gate_weight = builder.input(result.weight_description).value();
  result.up_weight = builder.input(result.weight_description).value();
  auto gate_rhs = result.gate_weight, up_rhs = result.up_weight;
  if (escape != Escape::kDenseRhs) {
    const auto transpose = c::View::permute(std::array<std::int32_t, 2>{1, 0}).value();
    gate_rhs = builder.add(transpose, gate_rhs).value();
    up_rhs = builder.add(transpose, up_rhs).value();
  }
  const c::MatMul projection{.accumulation_type = c::DataType::kFloat32};
  if (up_first) result.up = builder.add(projection, result.foreign_x, up_rhs).value();
  result.gate = builder.add(projection, result.x, gate_rhs).value();
  result.activation = builder.add(c::Unary{c::UnaryKind::kSilu}, result.gate).value();
  if (!up_first) result.up = builder.add(projection, result.foreign_x, up_rhs).value();
  result.output = builder.add(c::Binary{c::BinaryKind::kMultiply}, result.activation, result.up).value();
  if (escape == Escape::kPublished)
    for (const auto value : {result.gate, result.up, result.activation})
      EXPECT_TRUE(builder.publish(value).ok());
  if (escape == Escape::kExtraReader) {
    result.reader = builder.add(c::Unary{c::UnaryKind::kNeg}, result.gate).value();
    EXPECT_TRUE(builder.publish(result.reader).ok());
  }
  EXPECT_TRUE(builder.publish(result.output).ok());
  result.logical = std::move(builder).finish().value();
  const std::array values{result.gate, result.up, result.activation, result.output};
  for (std::size_t i = 0; i < values.size(); ++i)
    result.members[i] = result.logical.producer(values[i]).value();
  std::sort(result.members.begin(), result.members.end(), [](auto a, auto b) { return a.ordinal() < b.ordinal(); });
  return result;
}

void route(const g::CompiledGraph &compiled, const Graph &graph, bool fused,
           bool allow_real_pair = false) {
  std::size_t matches = 0, retained_projections = 0;
  for (std::size_t i = 0; i < compiled.report.step_count(); ++i) {
    const auto step = compiled.report.step(i).value();
    constexpr c::RoutePolicyId pair_policy{0x48594c4547474355ULL, 5, 1};
    const bool real_pair = step.provider.id == c::ggml_cuda::kProviderId &&
        step.implementation == c::ProviderImplementationId{6} && step.policy == pair_policy;
    if (real_pair) {
      EXPECT_TRUE(allow_real_pair);
      EXPECT_EQ(step.copy_bytes, 0U);
      ASSERT_EQ(step.node_count, 2U);
      EXPECT_EQ(step.output_count, 2U);
      for (const auto projection : {graph.logical.producer(graph.gate).value(),
                                    graph.logical.producer(graph.up).value()}) {
        bool found = false;
        for (std::size_t j = 0; j < step.node_count; ++j)
          found |= compiled.report.step_node(i, j).value() == projection;
        EXPECT_TRUE(found);
      }
    }
    if (step.provider.id == c::ggml_cuda::kProviderId && step.implementation == kImplementation) {
      ++matches;
      EXPECT_EQ(step.policy, kPolicy);
      EXPECT_EQ(step.copy_bytes, 0U);
      ASSERT_EQ(step.node_count, 4U);
      std::array<g::NodeId, 4> observed;
      for (std::size_t j = 0; j < observed.size(); ++j)
        observed[j] = compiled.report.step_node(i, j).value();
      std::sort(observed.begin(), observed.end(), [](auto a, auto b) { return a.ordinal() < b.ordinal(); });
      EXPECT_EQ(observed, graph.members);
    }
    for (std::size_t j = 0; j < step.node_count; ++j) {
      const auto node = compiled.report.step_node(i, j).value();
      if (node != graph.logical.producer(graph.gate).value() &&
          node != graph.logical.producer(graph.up).value()) continue;
      if (!fused) {
        EXPECT_TRUE(step.provider.id == c::cublas::kProviderId ||
                    (allow_real_pair && real_pair));
        ++retained_projections;
      }
    }
  }
  EXPECT_EQ(matches, fused ? 1U : 0U);
  if (!fused) EXPECT_EQ(retained_projections, 2U);
  // This proves selected region membership, not a physical kernel launch.
}

std::vector<BF16> stored(const Data &d, const std::vector<BF16> &weight, Escape escape) {
  if (escape != Escape::kDenseRhs) return weight;
  std::vector<BF16> result(weight.size());
  for (std::int64_t n = 0; n < kColumns; ++n)
    for (std::int64_t k = 0; k < d.inner; ++k)
      result[k * kColumns + n] = weight[n * d.inner + k];
  return result;
}

void bind(g::Invocation &run, const Graph &graph, std::array<Buffer, 4> &buffers) {
  const auto x = c::ConstTensorView::create(graph.input_description, buffers[0].view()).value();
  ASSERT_TRUE(run.bind(graph.x, x).ok());
  if (graph.foreign_x != graph.x) ASSERT_TRUE(run.bind(graph.foreign_x, x).ok());
  ASSERT_TRUE(run.bind(graph.gate_weight, c::ConstTensorView::create(
      graph.weight_description, buffers[1].view()).value()).ok());
  ASSERT_TRUE(run.bind(graph.up_weight, c::ConstTensorView::create(
      graph.weight_description, buffers[2].view()).value()).ok());
  ASSERT_TRUE(run.bind(graph.output, c::TensorView::create(
      graph.output_description, buffers[3].view()).value()).ok());
}

TEST(GgmlCudaDualProjection, ExactDotsPreserveGateUpAndSiluRoundingAcrossSlots) {
  Direct direct;
  for (const bool up_first : {false, true}) {
    SCOPED_TRACE(up_first);
    auto context = g::cuda::create_context().value();
    auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
    const auto logical = graph(128, Escape::kNone, up_first);
    auto compiled = g::cuda::compile(logical.logical, context, memory.ref());
    ASSERT_TRUE(compiled) << compiled.error().message();
    route(*compiled, logical, true);
    std::vector<Buffer> retained;
    std::array queues{context.create_queue().value(), context.create_queue().value()};
    std::array runs{compiled->executable.create_invocation().value(),
                    compiled->executable.create_invocation().value()};
    std::array<std::vector<BF16>, 2> old;
    for (int round = 0; round < 2; ++round) {
      std::array<std::array<Buffer, 4>, 2> buffers;
      std::array<Oracle, 2> expected;
      for (std::size_t slot = 0; slot < 2; ++slot) {
        SCOPED_TRACE(::testing::Message() << round << ":" << slot);
        const auto host = data(128, round * 3 + static_cast<int>(slot) + 1);
        expected[slot] = oracle(direct, host, true);
        buffers[slot] = {upload<BF16>(memory.ref(), host.x),
                         upload<BF16>(memory.ref(), host.gate_weight),
                         upload<BF16>(memory.ref(), host.up_weight),
                         output_buffer(memory.ref(), logical.output_description)};
        if (round == 1)
          for (std::size_t i = 0; i < 4; ++i)
            EXPECT_NE(buffers[slot][i].native_handle(), retained[slot * 4 + i].native_handle());
        bind(runs[slot], logical, buffers[slot]);
      }
      for (const auto mode : {g::ReplayMode::kCommands, g::ReplayMode::kRequireCapture}) {
        std::array<c::Completion, 2> done;
        for (std::size_t slot = 0; slot < 2; ++slot) {
          const auto prepared = g::cuda::prepare_replay(runs[slot], queues[slot],
              {.mode = mode, .host_wait = g::HostWaitPolicy::kReject});
          ASSERT_TRUE(prepared.ok()) << prepared.message();
          EXPECT_EQ(runs[slot].replay_route(), mode == g::ReplayMode::kCommands
              ? g::ReplayRoute::kCommands : g::ReplayRoute::kCaptured);
          auto submitted = runs[slot].submit(queues[slot]);
          ASSERT_TRUE(submitted) << submitted.error().message();
          done[slot] = std::move(*submitted);
        }
        ASSERT_TRUE(done[1].wait().ok()); ASSERT_TRUE(done[0].wait().ok());
        for (std::size_t slot = 0; slot < 2; ++slot) check(buffers[slot][3], expected[slot].output);
      }
      if (round == 0) {
        auto held = runs[0].result(logical.output).value();
        EXPECT_TRUE(runs[0].reset().IsUnavailable());
        auto other = runs[1].submit(queues[1]);
        ASSERT_TRUE(other) << other.error().message();
        ASSERT_TRUE(other->wait().ok());
        EXPECT_EQ(held.tensor().value().storage().native_handle(), buffers[0][3].native_handle());
        check(buffers[0][3], expected[0].output);
      }
      for (std::size_t slot = 0; slot < 2; ++slot) {
        if (round == 0) old[slot] = half_test::read<BF16>(buffers[slot][3]);
        for (auto &buffer : buffers[slot]) retained.push_back(std::move(buffer));
      }
      if (round == 1)
        for (std::size_t slot = 0; slot < 2; ++slot) unchanged(retained[slot * 4 + 3], old[slot]);
    }
  }
}

TEST(GgmlCudaDualProjection, EscapedValuesAndUnsupportedDomainsKeepOriginalChain) {
  Direct direct;
  auto context = g::cuda::create_context().value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  for (const auto escape : {Escape::kPublished, Escape::kExtraReader,
                           Escape::kForeignActivation, Escape::kOddInner, Escape::kDenseRhs}) {
    SCOPED_TRACE(int(escape));
    const auto inner = escape == Escape::kOddInner ? 127 : 128;
    const auto host = data(inner, 1);
    const auto expected = oracle(direct, host, false);
    const auto logical = graph(inner, escape);
    auto compiled = g::cuda::compile(logical.logical, context, memory.ref());
    ASSERT_TRUE(compiled) << compiled.error().message();
    route(*compiled, logical, false,
          escape == Escape::kPublished || escape == Escape::kExtraReader);
    std::array<Buffer, 4> buffers{upload<BF16>(memory.ref(), host.x),
        upload<BF16>(memory.ref(), stored(host, host.gate_weight, escape)),
        upload<BF16>(memory.ref(), stored(host, host.up_weight, escape)),
        output_buffer(memory.ref(), logical.output_description)};
    std::array<Buffer, 3> published;
    auto reader = output_buffer(memory.ref(), logical.output_description);
    auto queue = context.create_queue().value();
    auto run = compiled->executable.create_invocation().value();
    bind(run, logical, buffers);
    if (escape == Escape::kPublished) {
      const std::array values{logical.gate, logical.up, logical.activation};
      for (std::size_t i = 0; i < values.size(); ++i) {
        published[i] = output_buffer(memory.ref(), logical.output_description);
        ASSERT_TRUE(run.bind(values[i], c::TensorView::create(
            logical.output_description, published[i].view()).value()).ok());
      }
    }
    if (escape == Escape::kExtraReader)
      ASSERT_TRUE(run.bind(logical.reader, c::TensorView::create(
          logical.output_description, reader.view()).value()).ok());
    ASSERT_TRUE(g::cuda::prepare_replay(run, queue).ok());
    auto done = run.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    check(buffers[3], expected.output);
    if (escape == Escape::kPublished) {
      check(published[0], expected.gate); check(published[1], expected.up);
      check(published[2], expected.activation);
    }
    if (escape == Escape::kExtraReader) {
      auto negated = expected.gate;
      for (auto &value : negated) value.bits ^= 0x8000U;
      check(reader, negated);
    }
  }
}

double ulp(BF16 value) {
  const auto magnitude = std::abs(half_test::decode(value));
  return std::ldexp(1.0, magnitude == 0 ? -133 : std::max(-133, std::ilogb(magnitude) - 7));
}

TEST(GgmlCudaDualProjection, LongDotsMeetIndependentReductionBudget) {
  Direct direct;
  auto context = g::cuda::create_context().value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  constexpr std::int64_t inner = 2048;
  const auto logical = graph(inner);
  auto compiled = g::cuda::compile(logical.logical, context, memory.ref());
  ASSERT_TRUE(compiled) << compiled.error().message();
  route(*compiled, logical, true);
  std::vector<Buffer> retained;
  auto queue = context.create_queue().value();
  auto run = compiled->executable.create_invocation().value();
  double max_absolute = 0, max_relative = 0, max_budget_fraction = 0;
  unsigned max_ulps = 0;
  // This separate test permits declared F32 dot-reduction variation. It does
  // not replace the exact G/U/A RNE witnesses in the first test, nor demand
  // equality with cuBLASLt's different F32 reduction tree.
  constexpr double unit_roundoff = 0x1p-24;
  constexpr double gamma = (inner + 64) * unit_roundoff / (1 - (inner + 64) * unit_roundoff);
  for (const int seed : {71, 193}) {
    const auto host = data(inner, seed, true);
    const auto expected = oracle(direct, host, false);
    std::array<Buffer, 4> buffers{upload<BF16>(memory.ref(), host.x),
        upload<BF16>(memory.ref(), host.gate_weight), upload<BF16>(memory.ref(), host.up_weight),
        output_buffer(memory.ref(), logical.output_description)};
    bind(run, logical, buffers);
    ASSERT_TRUE(g::cuda::prepare_replay(run, queue).ok());
    auto done = run.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = half_test::read<BF16>(buffers[3]);
    ASSERT_EQ(actual.size(), expected.output.size());
    for (std::size_t n = 0; n < actual.size(); ++n) {
      // BF16 products are exact F32. Bound a sequential F32 sum plus reduction,
      // both projection roundings, SiLU's global derivative bound1.1, its two
      // output quantizations, then product/final RNE. CPU exp is not used.
      const double eg = gamma * host.gate_absolute[n] + 2 * ulp(expected.gate[n]);
      const double eu = gamma * host.up_absolute[n] + 2 * ulp(expected.up[n]);
      const double a = std::abs(half_test::decode(expected.activation[n]));
      const double u = std::abs(half_test::decode(expected.up[n]));
      const double ea = 1.1 * eg + 2 * ulp(expected.activation[n]) +
          4e-6 * std::max(a, 1e-30);
      const double budget = u * ea + a * eu + ea * eu + 2 * ulp(expected.output[n]);
      const double error = std::abs(half_test::decode(actual[n]) - half_test::decode(expected.output[n]));
      EXPECT_LE(error, budget) << "column=" << n << " seed=" << seed;
      max_absolute = std::max(max_absolute, error);
      const auto reference = std::abs(half_test::decode(expected.output[n]));
      if (reference != 0) max_relative = std::max(max_relative, error / reference);
      const auto ordered = [](BF16 value) {
        return (value.bits & 0x8000U) ? 0x8000 - int(value.bits & 0x7fffU)
                                     : 0x8000 + int(value.bits);
      };
      max_ulps = std::max(max_ulps, static_cast<unsigned>(std::abs(
          ordered(actual[n]) - ordered(expected.output[n]))));
      max_budget_fraction = std::max(max_budget_fraction, error / budget);
    }
    for (auto &buffer : buffers) retained.push_back(std::move(buffer));
  }
  RecordProperty("random_max_absolute_error", std::to_string(max_absolute));
  RecordProperty("random_max_relative_error_nonzero_reference", std::to_string(max_relative));
  RecordProperty("random_max_bf16_ulps", std::to_string(max_ulps));
  RecordProperty("random_max_budget_fraction", std::to_string(max_budget_fraction));
}
} // namespace dual_projection_test
} // namespace

namespace {
namespace explicit_frequency_rope_test {
using F16 = hyle::simd::float16_t;
namespace bridge = c::ggml_cuda::bridge;

struct Shape {
  std::int64_t batch, sequence, heads, depth, rotary;
  c::RopeLayout layout;
  bool batched_positions, sequence_frequencies, exceptional;
  bool producer = false, publish_producer = false;
};

template <class T> constexpr ggml_type native_type() {
  if constexpr (std::is_same_v<T, float>) return GGML_TYPE_F32;
  else if constexpr (std::is_same_v<T, BF16>) return GGML_TYPE_BF16;
  else return GGML_TYPE_F16;
}

template <class T> std::uint32_t bits(T value) {
  if constexpr (std::is_same_v<T, float>)
    return std::bit_cast<std::uint32_t>(value);
  else return value.bits;
}

template <class T>
void compare(const Buffer &buffer, const std::vector<T> &expected,
             bool classify_nan = true) {
  const auto actual = half_test::read<T>(buffer);
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (classify_nan && std::isnan(half_test::decode(expected[i])))
      EXPECT_TRUE(std::isnan(half_test::decode(actual[i]))) << i;
    else EXPECT_EQ(bits(actual[i]), bits(expected[i])) << i;
  }
}

template <class T> struct Host {
  std::vector<T> data;
  std::vector<std::int32_t> positions;
  std::vector<float> frequencies;
};

template <class T> Host<T> host(const Shape &shape, int round, int slot) {
  Host<T> result;
  const auto count = static_cast<std::size_t>(
      shape.batch * shape.sequence * shape.heads * shape.depth);
  for (std::size_t i = 0; i < count; ++i) {
    T value = half_test::encode<T>((int((i * 7 + round * 5 + slot) % 43) - 21) / 8.0);
    if (shape.exceptional) {
      switch (i % 19) {
      case 0: value = half_test::encode<T>(0.0); break;
      case 1: value = half_test::encode<T>(-0.0); break;
      case 2:
      case 3:
        if constexpr (std::is_same_v<T, float>)
          value = std::bit_cast<float>(i % 19 == 2 ? 1U : 0x80000001U);
        else value = T::from_bits(static_cast<std::uint16_t>(i % 19 == 2 ? 1 : 0x8001));
        break;
      case 4:
      case 5:
        if constexpr (std::is_same_v<T, float>)
          value = std::bit_cast<float>(i % 19 == 4 ? 0x00800000U : 0x80800000U);
        else {
          constexpr std::uint16_t minimum = std::is_same_v<T, BF16> ? 0x0080 : 0x0400;
          value = T::from_bits(static_cast<std::uint16_t>(
              minimum | (i % 19 == 5 ? 0x8000 : 0)));
        }
        break;
      case 6: value = half_test::encode<T>(0x1p-12); break;
      case 7: value = half_test::encode<T>(std::numeric_limits<double>::infinity()); break;
      case 8: value = half_test::encode<T>(-std::numeric_limits<double>::infinity()); break;
      case 9: value = half_test::encode<T>(std::numeric_limits<double>::quiet_NaN()); break;
      default: break;
      }
    }
    result.data.push_back(value);
  }
  constexpr std::array<std::int32_t, 8> small{0, 3, -7, 19, 31, -11, 2, 5};
  constexpr std::array<std::int32_t, 8> large{
      0, -7, 16777217, INT32_MAX, -16777217, INT32_MIN, 33554435, 3};
  const auto positions = static_cast<std::size_t>(
      shape.sequence * (shape.batched_positions ? shape.batch : 1));
  for (std::size_t i = 0; i < positions; ++i)
    result.positions.push_back((shape.exceptional ? large : small)[
        (i + static_cast<std::size_t>(round + slot)) % small.size()]);
  const auto frequencies = static_cast<std::size_t>(
      (shape.rotary / 2) * (shape.sequence_frequencies ? shape.sequence : 1));
  const std::array<float, 8> edge_frequency{
      0.3F, -0.71F, std::numeric_limits<float>::denorm_min(), 0.0F,
      std::numeric_limits<float>::min(), -0.0F,
      std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};
  for (std::size_t i = 0; i < frequencies; ++i) {
    float value = shape.exceptional ? edge_frequency[i % edge_frequency.size()]
        : float(1 + (i * 3 + round + slot) % 5) / 8.0F;
    if (shape.exceptional && ((round + slot) & 1)) value = -value;
    result.frequencies.push_back(value);
  }
  return result;
}

// Original private primitive graph, not the new public RoPE carrier. This
// also executes its real I32->F32 CPY; no host trig or new RoPE is the oracle.
// Every primitive is an output so allocation/native fusion cannot erase the
// four products or their separate add/subtract rounding boundaries.
struct NativeOwners {
  ggml_context *metadata = nullptr;
  ggml_backend_t backend = nullptr;
  ggml_gallocr_t allocation = nullptr;
  ~NativeOwners() {
    if (backend) ggml_backend_synchronize(backend);
    if (allocation) ggml_gallocr_free(allocation);
    if (backend) ggml_backend_free(backend);
    if (metadata) ggml_free(metadata);
  }
};

template <class T>
std::vector<T> primitive_reference(const Shape &shape, const Host<T> &input) {
  ggml_time_init();
  NativeOwners owners;
  constexpr std::size_t slots = 128;
  owners.metadata = ggml_init({slots * ggml_tensor_overhead() +
      ggml_graph_overhead_custom(slots, false), nullptr, true});
  if (!owners.metadata) { ADD_FAILURE() << "oracle metadata allocation"; return {}; }
  const int device = bridge::device_index(0);
  if (device < 0) { ADD_FAILURE() << "oracle physical CUDA device absent"; return {}; }
  owners.backend = ggml_backend_cuda_init(device);
  if (!owners.backend) { ADD_FAILURE() << "oracle CUDA backend"; return {}; }
  bridge::initialize_backend(owners.backend);
  auto *data = ggml_new_tensor_4d(owners.metadata, native_type<T>(),
      shape.depth, shape.heads, shape.sequence, shape.batch);
  auto *positions = ggml_new_tensor_2d(owners.metadata, GGML_TYPE_I32,
      shape.sequence, shape.batched_positions ? shape.batch : 1);
  auto *frequencies = ggml_new_tensor_2d(owners.metadata, GGML_TYPE_F32,
      shape.rotary / 2, shape.sequence_frequencies ? shape.sequence : 1);
  ggml_set_input(data); ggml_set_input(positions); ggml_set_input(frequencies);
  auto *source = shape.producer ? ggml_dup(owners.metadata, data) : data;
  auto *output = c::ggml_runtime::explicit_rope(owners.metadata, source,
      positions, frequencies, shape.rotary,
      shape.layout == c::RopeLayout::kInterleaved);
  auto *graph = ggml_new_graph_custom(owners.metadata, slots, false);
  ggml_build_forward_expand(graph, output);
  int products = 0, cosine = 0, sine = 0;
  for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
    auto *node = ggml_graph_node(graph, i);
    if (node->op == GGML_OP_ROPE) {
      ADD_FAILURE() << "original helper became a RoPE carrier; oracle is no longer independent";
      return {};
    }
    products += node->op == GGML_OP_MUL;
    cosine += node->op == GGML_OP_COS;
    sine += node->op == GGML_OP_SIN;
    ggml_set_output(node);
  }
  EXPECT_EQ(products, 5); // phase plus four independently rounded products
  EXPECT_EQ(cosine, 1); EXPECT_EQ(sine, 1);
  owners.allocation = ggml_gallocr_new(ggml_backend_get_default_buffer_type(owners.backend));
  if (!owners.allocation || !ggml_gallocr_alloc_graph(owners.allocation, graph)) {
    ADD_FAILURE() << "oracle graph allocation"; return {};
  }
  ggml_backend_tensor_set(data, input.data.data(), 0, input.data.size() * sizeof(T));
  ggml_backend_tensor_set(positions, input.positions.data(), 0,
      input.positions.size() * sizeof(std::int32_t));
  ggml_backend_tensor_set(frequencies, input.frequencies.data(), 0,
      input.frequencies.size() * sizeof(float));
  const auto status = ggml_backend_graph_compute(owners.backend, graph);
  if (status != GGML_STATUS_SUCCESS) {
    ADD_FAILURE() << "oracle original graph status=" << int(status); return {};
  }
  std::vector<T> result(input.data.size());
  ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(T));
  return result;
}

// Independent row/pair indexing check only for the ordinary bounded-phase
// tuples. Host cos/sin do not claim CUDA bit parity for long/nonfinite/FTZ
// domains: those are covered by the exact primitive GPU comparison above.
template <class T>
void check_host_indexing(const Shape &shape, const Host<T> &input,
                         const Buffer &buffer) {
  const auto actual = half_test::read<T>(buffer);
  const auto pairs = shape.rotary / 2;
  std::size_t checked = 0;
  for (std::int64_t b = 0; b < shape.batch; ++b)
    for (std::int64_t s = 0; s < shape.sequence; ++s)
      for (std::int64_t h = 0; h < shape.heads; ++h) {
        const auto base = static_cast<std::size_t>(
            ((b * shape.sequence + s) * shape.heads + h) * shape.depth);
        for (std::int64_t p = 0; p < pairs; ++p) {
          const auto position_index = static_cast<std::size_t>(
              (shape.batched_positions ? b * shape.sequence : 0) + s);
          const auto frequency_index = static_cast<std::size_t>(
              (shape.sequence_frequencies ? s * pairs : 0) + p);
          const float position = static_cast<float>(input.positions[position_index]);
          const float phase = position * input.frequencies[frequency_index];
          ASSERT_LE(std::abs(phase), 32.0F);
          const auto first = base + static_cast<std::size_t>(
              shape.layout == c::RopeLayout::kInterleaved ? 2 * p : p);
          const auto second = first + static_cast<std::size_t>(
              shape.layout == c::RopeLayout::kInterleaved ? 1 : pairs);
          const double x = half_test::decode(input.data[first]);
          const double y = half_test::decode(input.data[second]);
          const double co = std::cos(double(phase)), si = std::sin(double(phase));
          const std::array<double, 2> expected{x * co - y * si, x * si + y * co};
          const std::array<std::size_t, 2> indices{first, second};
          for (std::size_t side = 0; side < 2; ++side) {
            const T rounded = half_test::encode<T>(expected[side]);
            double ulp;
            if constexpr (std::is_same_v<T, float>) {
              const float magnitude = std::abs(rounded);
              ulp = std::nextafter(magnitude, std::numeric_limits<float>::infinity()) - magnitude;
            } else {
              const auto magnitude = static_cast<std::uint16_t>(rounded.bits & 0x7fffU);
              ulp = half_test::decode_bits<T>(static_cast<std::uint16_t>(magnitude + 1)) -
                    half_test::decode_bits<T>(magnitude);
            }
            // Indexing budget: one storage ULP plus bounded-phase F32
            // trig/product error. This never relaxes strict GPU parity.
            EXPECT_NEAR(half_test::decode(actual[indices[side]]), expected[side],
                        ulp + 4e-6 * (std::abs(x) + std::abs(y)))
                << b << ',' << s << ',' << h << ',' << p << ',' << side;
            ++checked;
          }
        }
        for (std::int64_t d = shape.rotary; d < shape.depth; ++d)
          EXPECT_EQ(bits(actual[base + static_cast<std::size_t>(d)]),
                    bits(input.data[base + static_cast<std::size_t>(d)]));
      }
  EXPECT_GT(checked, 0U);
}

template <class T> void exercise(const Shape &shape) {
  SCOPED_TRACE(int(half_test::data_type<T>()));
  SCOPED_TRACE(int(shape.layout));
  auto context = c::Context::create(c::cuda::device(0).value(), g::ggml_cuda::provider()).value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  const auto data_description = dense(half_test::data_type<T>(),
      {shape.batch, shape.sequence, shape.heads, shape.depth});
  const auto position_description = shape.batched_positions
      ? dense(c::DataType::kInt32, {shape.batch, shape.sequence})
      : dense(c::DataType::kInt32, {shape.sequence});
  const auto frequency_description = shape.sequence_frequencies
      ? dense(c::DataType::kFloat32, {shape.sequence, shape.rotary / 2})
      : dense(c::DataType::kFloat32, {shape.rotary / 2});
  auto builder = g::GraphBuilder::create().value();
  const auto x = builder.input(data_description).value();
  const auto positions = builder.input(position_description).value();
  const auto frequencies = builder.input(frequency_description).value();
  const auto source = shape.producer ? builder.add(c::Copy{}, x).value() : x;
  const auto output = builder.add(c::RoPE{
      .rotary_dimensions = static_cast<std::size_t>(shape.rotary),
      .sequence_axis = 1, .layout = shape.layout}, source, positions, frequencies).value();
  if (shape.publish_producer) ASSERT_TRUE(builder.publish(source).ok());
  ASSERT_TRUE(builder.publish(output).ok());
  const auto logical = std::move(builder).finish().value();
  const std::array domains{g::StorageDomain{.resource = memory.ref()}};
  const std::array targets{g::ExecutionTarget{
      .source = g::ExecutionSource::acquire(context).value(), .default_storage = g::StorageDomainId{0}}};
  const std::array lanes{g::ExecutionLane{.target = g::ExecutionTargetId{0}}};
  auto compiled = g::compile(logical, {.storage_domains = domains, .targets = targets, .lanes = lanes});
  ASSERT_TRUE(compiled) << compiled.error().message();
  for (std::size_t step = 0; step < compiled->report.step_count(); ++step) {
    EXPECT_EQ(compiled->report.step(step)->provider.id, c::ggml_cuda::kProviderId);
    EXPECT_TRUE(compiled->report.step(step)->runtime.capture_compatible);
  }
  // All old allocations outlive both queues and Invocations. The unpublished
  // Copy producer is single-use, permitting native allocator exact in-place;
  // the separately published producer must keep its original values instead.
  std::vector<Buffer> retained;
  std::array queues{context.create_queue().value(), context.create_queue().value()};
  std::array runs{compiled->executable.create_invocation().value(),
                  compiled->executable.create_invocation().value()};
  std::vector<T> old_result, old_producer;
  for (int round = 0; round < 2; ++round) {
    std::array<Host<T>, 2> inputs;
    std::array<std::vector<T>, 2> expected;
    std::array<std::array<Buffer, 5>, 2> buffers;
    for (int slot = 0; slot < 2; ++slot) {
      inputs[slot] = host<T>(shape, round, slot);
      expected[slot] = primitive_reference(shape, inputs[slot]);
      ASSERT_EQ(expected[slot].size(), inputs[slot].data.size());
      buffers[slot][0] = upload<T>(memory.ref(), inputs[slot].data);
      buffers[slot][1] = upload<std::int32_t>(memory.ref(), inputs[slot].positions);
      buffers[slot][2] = upload<float>(memory.ref(), inputs[slot].frequencies);
      buffers[slot][3] = output_buffer(memory.ref(), data_description);
      ASSERT_TRUE(runs[slot].bind(x, c::ConstTensorView::create(
          data_description, buffers[slot][0].view()).value()).ok());
      ASSERT_TRUE(runs[slot].bind(positions, c::ConstTensorView::create(
          position_description, buffers[slot][1].view()).value()).ok());
      ASSERT_TRUE(runs[slot].bind(frequencies, c::ConstTensorView::create(
          frequency_description, buffers[slot][2].view()).value()).ok());
      ASSERT_TRUE(runs[slot].bind(output, c::TensorView::create(
          data_description, buffers[slot][3].view()).value()).ok());
      if (shape.publish_producer) {
        buffers[slot][4] = output_buffer(memory.ref(), data_description);
        ASSERT_TRUE(runs[slot].bind(source, c::TensorView::create(
            data_description, buffers[slot][4].view()).value()).ok());
      }
    }
    for (const auto mode : {g::ReplayMode::kCommands, g::ReplayMode::kRequireCapture}) {
      std::array<c::Completion, 2> completions;
      for (int slot = 0; slot < 2; ++slot) {
        const auto prepared = g::cuda::prepare_replay(runs[slot], queues[slot],
            {.mode = mode, .host_wait = g::HostWaitPolicy::kReject});
        ASSERT_TRUE(prepared.ok()) << prepared.message();
        EXPECT_EQ(runs[slot].replay_route(), mode == g::ReplayMode::kCommands
            ? g::ReplayRoute::kCommands : g::ReplayRoute::kCaptured);
      }
      for (int slot = 0; slot < 2; ++slot) {
        auto done = runs[slot].submit(queues[slot]);
        ASSERT_TRUE(done) << done.error().message();
        completions[slot] = std::move(*done);
      }
      ASSERT_TRUE(completions[1].wait().ok());
      ASSERT_TRUE(completions[0].wait().ok());
      for (int slot = 0; slot < 2; ++slot) {
        compare(buffers[slot][3], expected[slot]);
        compare(buffers[slot][0], inputs[slot].data, false);
        if (shape.publish_producer) compare(buffers[slot][4], inputs[slot].data, false);
        if (!shape.exceptional) check_host_indexing(shape, inputs[slot], buffers[slot][3]);
      }
    }
    for (int slot = 0; slot < 2; ++slot) {
      if (round == 0 && slot == 0) {
        old_result = half_test::read<T>(buffers[slot][3]);
        if (shape.publish_producer) old_producer = half_test::read<T>(buffers[slot][4]);
      }
      for (auto &buffer : buffers[slot]) retained.push_back(std::move(buffer));
    }
    if (round == 1) {
      compare(retained[3], old_result, false);
      if (shape.publish_producer) compare(retained[4], old_producer, false);
    }
  }
}

TEST(GgmlCudaExplicitRope, PrimitiveParityCoversTypesLayoutsAndBroadcasts) {
  exercise<float>({2, 3, 2, 10, 6, c::RopeLayout::kHalfSplit,
      false, false, false, true, true});
  exercise<BF16>({2, 3, 3, 12, 8, c::RopeLayout::kInterleaved,
      true, true, false, true, false});
  exercise<F16>({2, 2, 2, 10, 4, c::RopeLayout::kHalfSplit,
      true, false, false});
}

TEST(GgmlCudaExplicitRope, LongPositionsAndExceptionalValuesKeepOriginalStages) {
  exercise<float>({2, 4, 2, 12, 8, c::RopeLayout::kInterleaved,
      true, true, true});
  exercise<BF16>({2, 4, 2, 10, 8, c::RopeLayout::kHalfSplit,
      false, true, true});
  exercise<F16>({2, 4, 2, 10, 8, c::RopeLayout::kInterleaved,
      false, false, true});
}
} // namespace explicit_frequency_rope_test
} // namespace

namespace {
namespace native_fusion_layout_test {
namespace bridge = c::ggml_cuda::bridge;
constexpr int k = 32, n = 8, experts = 3;
enum class Layout { dense_offset, strided, zero, expert_padding };
enum class Kind { plain, glu, selected };

struct Owners {
  ggml_context *ctx = nullptr;
  ggml_backend_t backend = nullptr;
  ggml_gallocr_t allocation = nullptr;
  std::vector<ggml_backend_buffer_t> buffers;
  ~Owners() {
    if (backend) ggml_backend_synchronize(backend);
    if (allocation) ggml_gallocr_free(allocation);
    for (auto *buffer : buffers) ggml_backend_buffer_free(buffer);
    if (backend) ggml_backend_free(backend);
    if (ctx) ggml_free(ctx);
  }
  bool initialize() {
    ggml_time_init();
    constexpr std::size_t capacity = 96;
    ctx = ggml_init({capacity * ggml_tensor_overhead() +
        ggml_graph_overhead_custom(capacity, false), nullptr, true});
    const int device = bridge::device_index(0);
    if (!ctx || device < 0) return false;
    backend = ggml_backend_cuda_init(device);
    if (!backend) return false;
    bridge::initialize_backend(backend);
    return true;
  }
  // External buffer capacity is deliberate: old row-bound errors must write
  // a checked sentinel, rather than escape an allocation. No fake strides or
  // data pointers are passed: the tensor is bound through the native API.
  bool bind(ggml_tensor *tensor, std::size_t bytes, bool weights = false) {
    auto *buffer = ggml_backend_alloc_buffer(backend, std::max(bytes,
        ggml_backend_buft_get_alloc_size(
            ggml_backend_get_default_buffer_type(backend), tensor)));
    if (!buffer) return false;
    buffers.push_back(buffer);
    if (weights) ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    return ggml_backend_tensor_alloc(buffer, tensor,
        ggml_backend_buffer_get_base(buffer)) == GGML_STATUS_SUCCESS;
  }
  bool allocate(ggml_cgraph *graph) {
    allocation = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    return allocation && ggml_gallocr_alloc_graph(allocation, graph);
  }
};

std::vector<float> read(ggml_tensor *tensor) {
  std::vector<float> result(static_cast<std::size_t>(ggml_nelements(tensor)));
  // All callers here read either contiguous tensors or one contiguous row.
  ggml_backend_tensor_get(tensor, result.data(), 0, result.size() * sizeof(float));
  return result;
}

void same_bits(const std::vector<float> &actual, const std::vector<float> &expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i)
    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]),
        std::bit_cast<std::uint32_t>(expected[i])) << "element " << i;
}

struct Bias {
  ggml_tensor *root = nullptr, *value = nullptr;
  std::vector<float> storage, logical;
};

Bias bias(Owners &o, Layout layout, int rows, int channels, int seed) {
  Bias b;
  b.storage.resize(static_cast<std::size_t>(2 * rows * channels + 16));
  for (std::size_t i = 0; i < b.storage.size(); ++i)
    b.storage[i] = 64.0F + float(i) / 8.0F;
  b.root = ggml_new_tensor_1d(o.ctx, GGML_TYPE_F32,
      static_cast<std::int64_t>(b.storage.size()));
  ggml_set_input(b.root);
  const std::size_t row_stride = static_cast<std::size_t>(
      rows * (layout == Layout::expert_padding ? 2 : 1));
  b.value = ggml_view_2d(o.ctx, b.root, rows, channels,
      row_stride * sizeof(float), sizeof(float));
  if (layout == Layout::strided) b.value->nb[0] = 2 * sizeof(float);
  if (layout == Layout::zero) b.value->nb[0] = 0;
  // Unused singleton strides are intentionally noncanonical in the positive.
  if (channels == 1) {
    b.value->nb[1] = (2 * rows + 3) * sizeof(float);
    b.value->nb[2] = (2 * rows + 5) * sizeof(float);
    b.value->nb[3] = (2 * rows + 7) * sizeof(float);
  }
  for (int e = 0; e < channels; ++e)
    for (int r = 0; r < rows; ++r) {
      const float value = float(seed + 2 * e +
          (layout == Layout::zero ? 0 : r) - 5) / 16.0F;
      const auto index = 1 + static_cast<std::size_t>(e) * row_stride +
          static_cast<std::size_t>(r) * b.value->nb[0] / sizeof(float);
      b.storage[index] = value;
      b.logical.push_back(value);
    }
  return b;
}

std::vector<float> execute_float(Kind kind, Layout layout, int seed, bool oracle) {
  Owners o;
  if (!o.initialize()) { ADD_FAILURE() << "native CUDA initialization"; return {}; }
  const int count = kind == Kind::selected ? experts : 1;
  auto *weight = ggml_new_tensor_3d(o.ctx, GGML_TYPE_F32, k, n, count);
  auto *input = ggml_new_tensor_3d(o.ctx, GGML_TYPE_F32, k, 1, 1);
  ggml_set_input(weight); ggml_set_input(input);
  auto b = bias(o, layout, n, count, seed);
  // The post-execution root check is a real external reader. INPUT only
  // allocates early; OUTPUT retains this owned backing through the graph.
  ggml_set_output(b.root);
  auto *ids = kind == Kind::selected
      ? ggml_new_tensor_2d(o.ctx, GGML_TYPE_I32, 2, 1) : nullptr;
  if (ids) ggml_set_input(ids);
  auto *mm = ids ? ggml_mul_mat_id(o.ctx, weight, input, ids)
                 : ggml_mul_mat(o.ctx, weight, input);
  ggml_prec_set_acc(mm, GGML_PREC_F32);
  auto *added = ids ? ggml_add_id(o.ctx, mm, b.value, ids)
                   : ggml_add(o.ctx, mm, b.value);
  auto *output = added;
  ggml_tensor *up_weight = nullptr, *up_mm = nullptr, *up_added = nullptr;
  Bias up;
  if (kind == Kind::glu) {
    up_weight = ggml_new_tensor_2d(o.ctx, GGML_TYPE_F32, k, n);
    ggml_set_input(up_weight);
    up = bias(o, Layout::dense_offset, n, 1, seed + 3);
    ggml_set_output(up.root);
    up_mm = ggml_mul_mat(o.ctx, up_weight, input);
    ggml_prec_set_acc(up_mm, GGML_PREC_F32);
    up_added = ggml_add(o.ctx, up_mm, up.value);
    output = ggml_swiglu_split(o.ctx, added, up_added);
  }
  if (oracle) {
    ggml_set_output(mm); ggml_set_output(added);
    if (up_mm) { ggml_set_output(up_mm); ggml_set_output(up_added); }
  }
  ggml_set_output(output);
  auto *graph = ggml_new_graph_custom(o.ctx, 96, false);
  // The actual MM+ADD(/MM+ADD+GLU) suffix must be contiguous. Expanding
  // metadata first prevents an interposed VIEW from silently disabling it.
  ggml_build_forward_expand(graph, b.value);
  if (up.value) ggml_build_forward_expand(graph, up.value);
  ggml_build_forward_expand(graph, output);
  std::vector<ggml_op> suffix;
  bool began = false;
  for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
    auto *node = ggml_graph_node(graph, i);
    began |= node == mm;
    if (began) suffix.push_back(node->op);
  }
  const std::vector<ggml_op> expected = kind == Kind::glu
      ? std::vector<ggml_op>{GGML_OP_MUL_MAT, GGML_OP_ADD,
          GGML_OP_MUL_MAT, GGML_OP_ADD, GGML_OP_GLU}
      : std::vector<ggml_op>{ids ? GGML_OP_MUL_MAT_ID : GGML_OP_MUL_MAT,
          ids ? GGML_OP_ADD_ID : GGML_OP_ADD};
  EXPECT_EQ(suffix, expected);
  if (!o.allocate(graph)) { ADD_FAILURE() << "native graph allocation"; return {}; }
  std::vector<float> w(static_cast<std::size_t>(k * n * count), 0.0F);
  for (int e = 0; e < count; ++e)
    for (int r = 0; r < n; ++r)
      w[static_cast<std::size_t>((e * n + r) * k)] = float(r - 3 + e + seed) / 8.0F;
  std::vector<float> x(k, 0.0F); x[0] = (seed & 1) ? 0.5F : 1.0F;
  const std::array<std::int32_t, 2> selected = seed & 1
      ? std::array<std::int32_t, 2>{1, 2} : std::array<std::int32_t, 2>{2, 1};
  ggml_backend_tensor_set(weight, w.data(), 0, w.size() * sizeof(float));
  ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
  ggml_backend_tensor_set(b.root, b.storage.data(), 0, b.storage.size() * sizeof(float));
  if (ids) ggml_backend_tensor_set(ids, selected.data(), 0, sizeof(selected));
  if (up_weight) {
    std::vector<float> u(k * n, 0.0F);
    for (int r = 0; r < n; ++r) u[static_cast<std::size_t>(r * k)] = float(r + seed + 1) / 16.0F;
    ggml_backend_tensor_set(up_weight, u.data(), 0, u.size() * sizeof(float));
    ggml_backend_tensor_set(up.root, up.storage.data(), 0, up.storage.size() * sizeof(float));
  }
  if (ggml_backend_graph_compute(o.backend, graph) != GGML_STATUS_SUCCESS) {
    ADD_FAILURE() << "native graph compute"; return {};
  }
  same_bits(read(b.root), b.storage);
  if (up.root) same_bits(read(up.root), up.storage);
  if (oracle) {
    std::vector<float> linear, biased;
    for (int slot = 0; slot < (ids ? 2 : 1); ++slot) {
      const int e = ids ? selected[static_cast<std::size_t>(slot)] : 0;
      for (int r = 0; r < n; ++r) {
        const float dot = w[static_cast<std::size_t>((e * n + r) * k)] * x[0];
        linear.push_back(dot);
        biased.push_back(dot + b.logical[static_cast<std::size_t>(e * n + r)]);
      }
    }
    same_bits(read(mm), linear);
    same_bits(read(added), biased);
  }
  return read(output);
}

// Independent quantized oracle computes all expert rows as one ordinary MM
// with a dense destination. It never executes the old odd-N MUL_MAT_ID path.
// The same Q8_0 bytes feed both graphs; no host exp or re-quantization oracle.
std::vector<float> execute_quantized(bool selected, int seed, bool oracle) {
  constexpr int rows = 7, pitch = 16, extra_rows = 8;
  const int count = selected ? experts : 1;
  Owners o;
  if (!o.initialize()) { ADD_FAILURE() << "quantized CUDA initialization"; return {}; }
  const int logical_rows = rows * count;
  auto *wroot = ggml_new_tensor_2d(o.ctx, GGML_TYPE_Q8_0, k, logical_rows + extra_rows);
  if (!o.bind(wroot, ggml_nbytes(wroot), true)) {
    ADD_FAILURE() << "quantized backing allocation"; return {};
  }
  ggml_set_input(wroot);
  const auto quant_row = ggml_row_size(GGML_TYPE_Q8_0, k);
  auto *weight = ggml_view_3d(o.ctx, wroot, k,
      oracle ? logical_rows : rows, oracle ? 1 : count,
      quant_row, static_cast<std::size_t>(oracle ? logical_rows : rows) * quant_row, 0);
  auto *input = ggml_new_tensor_3d(o.ctx, GGML_TYPE_F32, k, 1, 1);
  ggml_set_input(input);
  auto *ids = selected && !oracle
      ? ggml_new_tensor_2d(o.ctx, GGML_TYPE_I32, 2, 1) : nullptr;
  if (ids) ggml_set_input(ids);
  auto *mm = ids ? ggml_mul_mat_id(o.ctx, weight, input, ids)
                 : ggml_mul_mat(o.ctx, weight, input);
  ggml_prec_set_acc(mm, GGML_PREC_F32);
  // Biases have the exact active layout used by the candidate destination.
  auto *broot = selected ? nullptr : ggml_new_tensor_1d(o.ctx, GGML_TYPE_F32, pitch + 16);
  if (broot) ggml_set_input(broot);
  auto *b = broot ? ggml_view_2d(o.ctx, broot, rows, 1,
      pitch * sizeof(float), 0) : nullptr;
  auto *out = b ? ggml_add(o.ctx, mm, b) : mm;
  ggml_tensor *sentinel = nullptr;
  std::vector<float> before;
  if (oracle) {
    ggml_set_output(mm); // no MM+bias fusion in the independent GPU oracle
  } else {
    const int output_pitch = selected ? rows : pitch;
    out->nb[1] = output_pitch * sizeof(float);
    out->nb[2] = output_pitch * (selected ? 2 : 1) * sizeof(float);
    out->nb[3] = out->nb[2];
    sentinel = ggml_new_tensor_1d(o.ctx, GGML_TYPE_F32,
        output_pitch * (selected ? 2 : 1) + 16);
    if (!o.bind(out, ggml_nbytes(sentinel))) {
      ADD_FAILURE() << "padded destination allocation"; return {};
    }
    // A second honest tensor binding observes the entire actual allocation,
    // including row seven and trailing storage outside the logical result.
    auto *buffer = out->buffer;
    if (ggml_backend_tensor_alloc(buffer, sentinel,
            ggml_backend_buffer_get_base(buffer)) != GGML_STATUS_SUCCESS) {
      ADD_FAILURE() << "destination sentinel binding"; return {};
    }
    before.resize(static_cast<std::size_t>(ggml_nelements(sentinel)));
    for (std::size_t i = 0; i < before.size(); ++i) before[i] = -256.0F - float(i) / 8.0F;
    ggml_backend_tensor_set(sentinel, before.data(), 0, before.size() * sizeof(float));
  }
  ggml_set_output(out);
  auto *graph = ggml_new_graph_custom(o.ctx, 96, false);
  ggml_build_forward_expand(graph, weight);
  if (b) ggml_build_forward_expand(graph, b);
  ggml_build_forward_expand(graph, out);
  if (b) EXPECT_EQ(ggml_graph_node(graph, ggml_graph_n_nodes(graph) - 2), mm);
  EXPECT_EQ(ggml_graph_node(graph, ggml_graph_n_nodes(graph) - 1), out);
  if (!o.allocate(graph)) { ADD_FAILURE() << "quantized graph allocation"; return {}; }
  std::vector<float> unpacked(static_cast<std::size_t>(k * (logical_rows + extra_rows)));
  for (int r = 0; r < logical_rows + extra_rows; ++r)
    for (int j = 0; j < k; ++j)
      unpacked[static_cast<std::size_t>(r * k + j)] = float((r + seed) % 9 - 4) / 8.0F;
  std::vector<std::byte> packed(ggml_nbytes(wroot));
  EXPECT_EQ(ggml_quantize_chunk(GGML_TYPE_Q8_0, unpacked.data(), packed.data(),
      0, logical_rows + extra_rows, k, nullptr), packed.size());
  std::vector<float> x(k, 0.0F); x[0] = seed & 1 ? 0.5F : 1.0F;
  std::vector<float> biases(broot ? static_cast<std::size_t>(ggml_nelements(broot)) : 0, 128.0F);
  for (int e = 0; e < (broot ? count : 0); ++e)
    for (int r = 0; r < rows; ++r)
      biases[static_cast<std::size_t>((oracle ? e * rows : e * pitch) + r)] =
          float(e + r + seed - 3) / 16.0F;
  const std::array<std::int32_t, 2> indices = seed & 1
      ? std::array<std::int32_t, 2>{1, 2} : std::array<std::int32_t, 2>{2, 1};
  ggml_backend_tensor_set(wroot, packed.data(), 0, packed.size());
  ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
  if (broot) ggml_backend_tensor_set(broot, biases.data(), 0, biases.size() * sizeof(float));
  if (ids) ggml_backend_tensor_set(ids, indices.data(), 0, sizeof(indices));
  if (ggml_backend_graph_compute(o.backend, graph) != GGML_STATUS_SUCCESS) {
    ADD_FAILURE() << "quantized graph compute"; return {};
  }
  if (broot) same_bits(read(broot), biases);
  std::vector<float> result(static_cast<std::size_t>(rows * (selected ? 2 : 1)));
  if (oracle) {
    const auto all = read(out);
    for (int slot = 0; slot < (selected ? 2 : 1); ++slot) {
      const int expert = selected ? indices[static_cast<std::size_t>(slot)] : 0;
      std::copy_n(all.begin() + expert * rows, rows, result.begin() + slot * rows);
    }
  } else {
    const auto after = read(sentinel);
    for (std::size_t i = 0; i < after.size(); ++i) {
      const auto output_pitch = static_cast<std::size_t>(selected ? rows : pitch);
      const auto slot = i / output_pitch;
      const auto row = i % output_pitch;
      if (slot < static_cast<std::size_t>(selected ? 2 : 1) &&
          row < static_cast<std::size_t>(rows))
        result[slot * rows + row] = after[i];
      else EXPECT_EQ(std::bit_cast<std::uint32_t>(after[i]),
          std::bit_cast<std::uint32_t>(before[i])) << "padded destination element " << i;
    }
  }
  return result;
}

TEST(GgmlCudaFusionLayout, BiasViewsUseActualActiveStrides) {
  for (int seed : {0, 1})
    for (auto layout : {Layout::dense_offset, Layout::strided, Layout::zero}) {
      SCOPED_TRACE("seed=" + std::to_string(seed) + " layout=" + std::to_string(int(layout)));
      same_bits(execute_float(Kind::plain, layout, seed, false),
          execute_float(Kind::plain, layout, seed, true));
    }
}

TEST(GgmlCudaFusionLayout, GluAndSelectedBiasKeepTheirPhysicalLayouts) {
  for (int seed : {0, 1}) {
    for (auto layout : {Layout::dense_offset, Layout::strided}) {
      SCOPED_TRACE("GLU seed=" + std::to_string(seed) + " layout=" + std::to_string(int(layout)));
      same_bits(execute_float(Kind::glu, layout, seed, false),
          execute_float(Kind::glu, layout, seed, true));
    }
    for (auto layout : {Layout::dense_offset, Layout::expert_padding}) {
      SCOPED_TRACE("ID seed=" + std::to_string(seed) + " layout=" + std::to_string(int(layout)));
      same_bits(execute_float(Kind::selected, layout, seed, false),
          execute_float(Kind::selected, layout, seed, true));
    }
  }
}

TEST(GgmlCudaFusionLayout, QuantizedOddRowsPreservePaddedDestinationsAndSelectedSlots) {
  for (int seed : {0, 1})
    for (bool selected : {false, true}) {
      SCOPED_TRACE("Q8_0 seed=" + std::to_string(seed) + " selected=" + std::to_string(selected));
      same_bits(execute_quantized(selected, seed, false),
          execute_quantized(selected, seed, true));
    }
}
} // namespace native_fusion_layout_test
} // namespace

namespace {
namespace native_fusion_precision_test {
using native_fusion_layout_test::Owners;
using native_fusion_layout_test::read;
using native_fusion_layout_test::same_bits;

// Both witnesses use inputs exactly representable in binary16. The overflow
// tuple has a finite F32 product but an infinite F16 product. The bounded
// tuple has the exact F32 product 1 + 2^-9 + 2^-20: F16 multiplication loses
// its last term. Neither witness depends on a particular reduction tree.
std::vector<float> execute(bool overflow, bool glu, bool oracle, bool mixed = false) {
  constexpr int reduction = 32, columns = 8;
  Owners o;
  if (!o.initialize()) { ADD_FAILURE() << "precision CUDA initialization"; return {}; }
  auto *w = ggml_new_tensor_2d(o.ctx, GGML_TYPE_F16, reduction, columns);
  auto *x = ggml_new_tensor_2d(o.ctx, GGML_TYPE_F32, reduction, 1);
  auto *b = ggml_new_tensor_2d(o.ctx, GGML_TYPE_F32, columns, 1);
  ggml_set_input(w); ggml_set_input(x); ggml_set_input(b);
  for (auto *input : {w, x, b}) ggml_set_output(input);
  auto *mm = ggml_mul_mat(o.ctx, w, x);
  const auto precision = mixed ? GGML_PREC_DEFAULT : GGML_PREC_F32;
  EXPECT_TRUE(ggml_prec_set_acc(mm, precision));
  auto *added = ggml_add(o.ctx, mm, b);
  auto *out = added;
  ggml_tensor *wu = nullptr, *bu = nullptr, *up = nullptr, *up_added = nullptr;
  if (glu) {
    wu = ggml_new_tensor_2d(o.ctx, GGML_TYPE_F16, reduction, columns);
    bu = ggml_new_tensor_2d(o.ctx, GGML_TYPE_F32, columns, 1);
    ggml_set_input(wu); ggml_set_input(bu);
    ggml_set_output(wu); ggml_set_output(bu);
    up = ggml_mul_mat(o.ctx, wu, x);
    EXPECT_TRUE(ggml_prec_set_acc(up, GGML_PREC_F32));
    up_added = ggml_add(o.ctx, up, bu);
    out = ggml_swiglu_split(o.ctx, added, up_added);
  }
  if (oracle) {
    ggml_set_output(mm); ggml_set_output(added);
    if (up) { ggml_set_output(up); ggml_set_output(up_added); }
  }
  ggml_set_output(out);
  auto *graph = ggml_new_graph_custom(o.ctx, 96, false);
  ggml_build_forward_expand(graph, out);
  const std::vector<ggml_op> expected = glu
      ? std::vector<ggml_op>{GGML_OP_MUL_MAT, GGML_OP_ADD,
          GGML_OP_MUL_MAT, GGML_OP_ADD, GGML_OP_GLU}
      : std::vector<ggml_op>{GGML_OP_MUL_MAT, GGML_OP_ADD};
  std::vector<ggml_op> actual;
  for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
    actual.push_back(ggml_graph_node(graph, i)->op);
  EXPECT_EQ(actual, expected);
  EXPECT_EQ(mm->type, GGML_TYPE_F32);
  EXPECT_EQ(out->type, GGML_TYPE_F32);
  EXPECT_EQ(mm->op_params[0], precision);
  EXPECT_EQ(added->op_params[0], 0); // the ADD has no MatMul precision field
  if (!o.allocate(graph)) { ADD_FAILURE() << "precision graph allocation"; return {}; }
  const float activation = overflow ? 2.0F : 1.0F + 0x1p-10F;
  const float weight = overflow ? 65504.0F : 1.0F + 0x1p-10F;
  std::vector<ggml_fp16_t> weights(reduction * columns, ggml_fp32_to_fp16(0.0F));
  for (int col = 0; col < columns; ++col)
    weights[static_cast<std::size_t>(col * reduction)] = ggml_fp32_to_fp16(weight);
  std::vector<float> input(reduction, 0.0F); input[0] = activation;
  std::vector<float> bias(columns);
  for (int col = 0; col < columns; ++col)
    bias[static_cast<std::size_t>(col)] = overflow ? float(col + 1) / 2.0F : 0.0F;
  ggml_backend_tensor_set(w, weights.data(), 0, weights.size() * sizeof(ggml_fp16_t));
  ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
  ggml_backend_tensor_set(b, bias.data(), 0, bias.size() * sizeof(float));
  if (up) {
    std::vector<ggml_fp16_t> up_weights(reduction * columns, ggml_fp32_to_fp16(0.0F));
    for (int col = 0; col < columns; ++col)
      up_weights[static_cast<std::size_t>(col * reduction)] = ggml_fp32_to_fp16(0.5F);
    std::vector<float> up_bias(columns, 0.25F);
    ggml_backend_tensor_set(wu, up_weights.data(), 0, up_weights.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(bu, up_bias.data(), 0, up_bias.size() * sizeof(float));
  }
  if (ggml_backend_graph_compute(o.backend, graph) != GGML_STATUS_SUCCESS) {
    ADD_FAILURE() << "precision graph compute"; return {};
  }
  // The original MatMul remains explicitly F32-accumulating after execution;
  // the fused epilogue must not acquire precision from its final operation.
  EXPECT_EQ(mm->op_params[0], precision);
  EXPECT_EQ(added->op_params[0], 0);
  const float exact_dot = static_cast<float>(double(weight) * double(activation));
  const float half_product = ggml_fp16_to_fp32(ggml_fp32_to_fp16(exact_dot));
  EXPECT_NE(std::bit_cast<std::uint32_t>(exact_dot),
      std::bit_cast<std::uint32_t>(half_product));
  if (overflow) { EXPECT_TRUE(std::isfinite(exact_dot)); EXPECT_TRUE(std::isinf(half_product)); }
  if (!mixed && (oracle || !glu)) {
    std::vector<float> reference(columns);
    for (int col = 0; col < columns; ++col)
      reference[static_cast<std::size_t>(col)] = exact_dot + bias[static_cast<std::size_t>(col)];
    if (oracle) same_bits(read(mm), std::vector<float>(columns, exact_dot));
    // No CPU nonlinear function is used. SwiGLU is compared only with the
    // original GPU graph whose two MM and two ADD results are published.
    if (oracle || !glu) same_bits(read(added), reference);
  }
  return read(out);
}

TEST(GgmlCudaFusionPrecision, F16WeightsKeepOriginalFloatAccumulationThroughEpilogues) {
  for (bool overflow : {false, true})
    for (bool glu : {false, true}) {
      SCOPED_TRACE("overflow=" + std::to_string(overflow) + " glu=" + std::to_string(glu));
      const auto reference = execute(overflow, glu, true);
      const auto fused = execute(overflow, glu, false);
      for (float value : reference) EXPECT_TRUE(std::isfinite(value));
      same_bits(fused, reference);
    }
}

// A dual carrier has one accumulation domain. DEFAULT gate/F32 up must
// retain the two original domains on hardware that distinguishes them.
TEST(GgmlCudaFusionPrecision, DifferentProjectionAccumulationDomainsKeepOriginalMath) {
  for (bool overflow : {false, true}) {
    SCOPED_TRACE("mixed overflow=" + std::to_string(overflow));
    const auto reference = execute(overflow, true, true, true);
    same_bits(execute(overflow, true, false, true), reference);
  }
}
} // namespace native_fusion_precision_test
} // namespace

namespace {
namespace native_fusion_oai_test {
using native_fusion_layout_test::Owners;
using native_fusion_layout_test::read;
using native_fusion_layout_test::same_bits;

// alpha=0 is a legitimate, exact witness: exp(0)=1 and up=1 make the
// result the clamped gate itself. Changing only limit then witnesses its
// separate propagation without a CPU exponential or a new error budget.
std::vector<float> execute(int route, float limit, int seed, bool oracle) {
  constexpr int k = 32, n = 8, experts = 3, slots = 2, tokens = 2;
  const bool quantized = route != 0, indexed = route == 2;
  const int banks = indexed ? experts : 1;
  Owners o;
  if (!o.initialize()) { ADD_FAILURE() << "OAI CUDA initialization"; return {}; }
  const auto type = quantized ? GGML_TYPE_Q8_0 : GGML_TYPE_F32;
  auto *w = ggml_new_tensor_3d(o.ctx, type, k, n, banks);
  auto *wu = ggml_new_tensor_3d(o.ctx, type, k, n, banks);
  auto *x = ggml_new_tensor_3d(o.ctx, GGML_TYPE_F32, k, 1, indexed ? tokens : 1);
  auto *b = ggml_new_tensor_2d(o.ctx, GGML_TYPE_F32, n, banks);
  auto *bu = ggml_new_tensor_2d(o.ctx, GGML_TYPE_F32, n, banks);
  auto *ids = indexed ? ggml_new_tensor_2d(o.ctx, GGML_TYPE_I32, slots, tokens) : nullptr;
  // These are persistent immutable inputs. Retaining their real allocations
  // prevents the old unfused allocation schedule from recycling W/x under
  // the final fused GLU; no mathematical intermediate is published here.
  for (auto *input : {w, wu, x, b, bu}) {
    ggml_set_input(input); ggml_set_output(input);
  }
  if (ids) { ggml_set_input(ids); ggml_set_output(ids); }
  auto *mm = indexed ? ggml_mul_mat_id(o.ctx, w, x, ids) : ggml_mul_mat(o.ctx, w, x);
  auto *added = indexed ? ggml_add_id(o.ctx, mm, b, ids) : ggml_add(o.ctx, mm, b);
  auto *up = indexed ? ggml_mul_mat_id(o.ctx, wu, x, ids) : ggml_mul_mat(o.ctx, wu, x);
  auto *up_added = indexed ? ggml_add_id(o.ctx, up, bu, ids) : ggml_add(o.ctx, up, bu);
  EXPECT_TRUE(ggml_prec_set_acc(mm, GGML_PREC_F32));
  EXPECT_TRUE(ggml_prec_set_acc(up, GGML_PREC_F32));
  auto *out = ggml_swiglu_oai(o.ctx, added, up_added, 0.0F, limit);
  if (oracle) for (auto *value : {mm, added, up, up_added}) ggml_set_output(value);
  ggml_set_output(out);
  auto *graph = ggml_new_graph_custom(o.ctx, 96, false);
  ggml_build_forward_expand(graph, out);
  const std::vector<ggml_op> expected{indexed ? GGML_OP_MUL_MAT_ID : GGML_OP_MUL_MAT,
      indexed ? GGML_OP_ADD_ID : GGML_OP_ADD,
      indexed ? GGML_OP_MUL_MAT_ID : GGML_OP_MUL_MAT,
      indexed ? GGML_OP_ADD_ID : GGML_OP_ADD, GGML_OP_GLU};
  std::vector<ggml_op> actual;
  for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) actual.push_back(ggml_graph_node(graph, i)->op);
  EXPECT_EQ(actual, expected);
  if (!o.allocate(graph)) { ADD_FAILURE() << "OAI graph allocation"; return {}; }
  const auto rows = static_cast<std::size_t>(n * banks);
  std::vector<float> weights(k * rows, 0.0F), up_weights(k * rows, 0.0F);
  for (std::size_t row = 0; row < rows; ++row)
    weights[row * k] = (row & 1 ? -1.0F : 1.0F) * float(4 + (int(row) + seed) % 6);
  const auto upload_weight = [&](ggml_tensor *tensor, const std::vector<float> &values) {
    if (!quantized) {
      ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
      return;
    }
    std::vector<std::byte> packed(ggml_nbytes(tensor));
    EXPECT_EQ(ggml_quantize_chunk(GGML_TYPE_Q8_0, values.data(), packed.data(),
        0, static_cast<std::int64_t>(rows), k, nullptr), packed.size());
    ggml_backend_tensor_set(tensor, packed.data(), 0, packed.size());
  };
  upload_weight(w, weights); upload_weight(wu, up_weights);
  std::vector<float> activation(static_cast<std::size_t>(k * (indexed ? tokens : 1)), 0.0F);
  for (int token = 0; token < (indexed ? tokens : 1); ++token)
    activation[static_cast<std::size_t>(token * k)] = token ? 0.5F : 1.0F;
  const std::vector<float> bias(rows, 0.0F), up_bias(rows, 1.0F);
  const std::array<std::int32_t, 4> indices = seed & 1
      ? std::array<std::int32_t, 4>{2, 0, 1, 2}
      : std::array<std::int32_t, 4>{1, 2, 2, 0};
  ggml_backend_tensor_set(x, activation.data(), 0, activation.size() * sizeof(float));
  ggml_backend_tensor_set(b, bias.data(), 0, bias.size() * sizeof(float));
  ggml_backend_tensor_set(bu, up_bias.data(), 0, up_bias.size() * sizeof(float));
  if (ids) ggml_backend_tensor_set(ids, indices.data(), 0, sizeof(indices));
  if (ggml_backend_graph_compute(o.backend, graph) != GGML_STATUS_SUCCESS) {
    ADD_FAILURE() << "OAI graph compute"; return {};
  }
  const auto result = read(out);
  if (oracle) {
    auto clamped = read(added);
    for (auto &value : clamped) value = std::min(value, limit);
    same_bits(result, clamped);
  }
  for (float value : result) EXPECT_TRUE(std::isfinite(value));
  return result;
}

TEST(GgmlCudaFusionPrecision, OaiParametersReachFloatingQuantizedAndMoeEpilogues) {
  for (int route : {0, 1, 2})
    for (float limit : {7.0F, 2.0F})
      for (int seed : {0, 1}) {
        SCOPED_TRACE("route=" + std::to_string(route) + " limit=" + std::to_string(limit) +
            " seed=" + std::to_string(seed));
        const auto reference = execute(route, limit, seed, true);
        same_bits(execute(route, limit, seed, false), reference);
      }
}
} // namespace native_fusion_oai_test
} // namespace

namespace {
namespace native_fusion_allocation_test {
using native_fusion_layout_test::Owners;
using native_fusion_layout_test::read;
using native_fusion_layout_test::same_bits;
namespace bridge = c::ggml_cuda::bridge;
constexpr int k = 32, n = 8;
enum class ViewKind { dense, ordinary, none_alias };
struct Dependency { ggml_tensor *root; int until; };

// This is an allocator-only consumer of the private bridge contract. The
// original graph, flags, src edges and reader counts are never edited. None
// of the mathematical inputs is published just to obtain a fused kernel.
struct AllocationGraph {
  ggml_cgraph graph{};
  std::vector<Dependency> dependencies;
  std::vector<ggml_tensor *> nodes, leafs;
  void build(Owners &o, ggml_cgraph *original) {
    bridge::collect_allocation_dependencies(o.backend, original,
        [](void *state, ggml_tensor *root, int until) {
          static_cast<AllocationGraph *>(state)->dependencies.push_back({root, until});
        }, this);
    std::stable_sort(dependencies.begin(), dependencies.end(),
        [](const auto &a, const auto &b) { return a.until < b.until; });
    leafs.assign(original->leafs, original->leafs + original->n_leafs);
    for (const auto &dep : dependencies) {
      EXPECT_NE(dep.root, nullptr);
      EXPECT_EQ(dep.root->view_src, nullptr);
      EXPECT_GE(dep.until, 0); EXPECT_LT(dep.until, original->n_nodes);
      const bool is_node = std::find(original->nodes, original->nodes + original->n_nodes, dep.root)
          != original->nodes + original->n_nodes;
      if (!is_node && std::find(leafs.begin(), leafs.end(), dep.root) == leafs.end())
        leafs.insert(leafs.begin(), dep.root);
    }
    std::size_t next = 0;
    for (int i = 0; i < original->n_nodes; ++i) {
      nodes.push_back(original->nodes[i]);
      while (next < dependencies.size() && dependencies[next].until == i) {
        auto *dependency = ggml_view_tensor(o.ctx, dependencies[next].root);
        for (int source = 0; source < GGML_MAX_SRC && next < dependencies.size()
             && dependencies[next].until == i; ++source, ++next)
          dependency->src[source] = dependencies[next].root;
        nodes.push_back(dependency);
      }
    }
    graph = *original;
    graph.nodes = nodes.data(); graph.n_nodes = static_cast<int>(nodes.size());
    graph.leafs = leafs.data(); graph.n_leafs = static_cast<int>(leafs.size());
    graph.size = static_cast<int>(std::max(nodes.size(), leafs.size()));
  }
};

std::vector<float> raw(bool glu, ViewKind view, int seed, bool oracle, bool explicit_alias = false) {
  Owners o;
  if (!o.initialize()) { ADD_FAILURE() << "allocation dependency CUDA setup"; return {}; }
  const int extra = view == ViewKind::dense ? 0 : 1;
  auto *wroot = ggml_new_tensor_2d(o.ctx, GGML_TYPE_F32, k, n + extra);
  auto *uroot = ggml_new_tensor_2d(o.ctx, GGML_TYPE_F32, k, n + extra);
  const auto weight_view = [&](ggml_tensor *root) {
    if (view == ViewKind::dense) return root;
    if (view == ViewKind::ordinary)
      return ggml_view_2d(o.ctx, root, k, n, k * sizeof(float), k * sizeof(float));
    auto *alias = ggml_new_tensor_2d(o.ctx, GGML_TYPE_F32, k, n);
    alias->view_src = root;
    alias->view_offs = k * sizeof(float);
    // A legitimate NONE alias has no source edges. Its storage root is not
    // visited by ordinary forward traversal and must be counted separately.
    return alias;
  };
  auto *w = weight_view(wroot), *u = weight_view(uroot);
  auto *x = ggml_new_tensor_1d(o.ctx, GGML_TYPE_F32, k);
  auto *b = ggml_new_tensor_1d(o.ctx, GGML_TYPE_F32, n);
  auto *bu = ggml_new_tensor_1d(o.ctx, GGML_TYPE_F32, n);
  for (auto *input : {wroot, uroot, x, b, bu}) ggml_set_input(input);
  auto *mm = ggml_mul_mat(o.ctx, w, x);
  auto *added = ggml_add(o.ctx, mm, b);
  auto *up = glu ? ggml_mul_mat(o.ctx, u, x) : nullptr;
  auto *up_added = glu ? ggml_add(o.ctx, up, bu) : nullptr;
  auto *out = glu ? ggml_swiglu_oai(o.ctx, added, up_added, 0.0F, 2.0F) : added;
  EXPECT_TRUE(ggml_prec_set_acc(mm, GGML_PREC_F32));
  if (up) EXPECT_TRUE(ggml_prec_set_acc(up, GGML_PREC_F32));
  if (oracle) {
    ggml_set_output(mm);
    if (glu) for (auto *value : {added, up, up_added}) ggml_set_output(value);
  }
  ggml_set_output(out);
  auto *original = ggml_new_graph_custom(o.ctx, 96, false);
  // Metadata views precede the mathematical span; they do not accidentally
  // create a false fallback by splitting MM+ADD+MM+ADD+GLU adjacency.
  if (view == ViewKind::ordinary) {
    ggml_build_forward_expand(original, w);
    if (glu) ggml_build_forward_expand(original, u);
  }
  ggml_build_forward_expand(original, out);
  std::vector<ggml_tensor *> original_nodes(original->nodes, original->nodes + original->n_nodes);
  std::vector<std::array<ggml_tensor *, GGML_MAX_SRC>> original_sources;
  for (auto *node : original_nodes) {
    std::array<ggml_tensor *, GGML_MAX_SRC> sources{};
    std::copy_n(node->src, GGML_MAX_SRC, sources.begin()); original_sources.push_back(sources);
  }
  AllocationGraph allocation;
  allocation.build(o, original);
  if (!oracle) {
    const auto has_root = [&](ggml_tensor *root) {
      return std::any_of(allocation.dependencies.begin(), allocation.dependencies.end(),
          [&](const auto &dep) { return dep.root == root && dep.until == original->n_nodes - 1; });
    };
    EXPECT_TRUE(has_root(wroot)); EXPECT_TRUE(has_root(x)); EXPECT_TRUE(has_root(b));
    if (glu) { EXPECT_TRUE(has_root(uroot)); EXPECT_TRUE(has_root(bu)); }
  }
  for (auto *input : {wroot, uroot, x, b, bu})
    EXPECT_EQ(input->flags & GGML_TENSOR_FLAG_OUTPUT, 0);
  if (view == ViewKind::none_alias) {
    // NONE views conventionally alias externally owned roots. Keep that
    // original ownership rather than depending on ordinary VIEW n_views
    // accounting that this metadata spelling deliberately does not have.
    if (!o.bind(wroot, ggml_nbytes(wroot)) || (glu && !o.bind(uroot, ggml_nbytes(uroot)))) {
      ADD_FAILURE() << "NONE view external root binding"; return {};
    }
  }
  if (explicit_alias) {
    if (!o.bind(b, n * sizeof(float)) || ggml_backend_tensor_alloc(b->buffer, out, b->data) != GGML_STATUS_SUCCESS) {
      ADD_FAILURE() << "explicit native alias binding"; return {};
    }
  }
  o.allocation = ggml_gallocr_new(ggml_backend_get_default_buffer_type(o.backend));
  if (!o.allocation || !ggml_gallocr_reserve(o.allocation, &allocation.graph)) {
    ADD_FAILURE() << "allocation-only reserve"; return {};
  }
  const auto measured = ggml_gallocr_get_buffer_size(o.allocation, 0);
  if (!ggml_gallocr_alloc_graph(o.allocation, &allocation.graph)) {
    ADD_FAILURE() << "allocation-only bind"; return {};
  }
  EXPECT_EQ(measured, ggml_gallocr_get_buffer_size(o.allocation, 0));
  for (std::size_t i = 0; i < original_nodes.size(); ++i) {
    EXPECT_EQ(original->nodes[i], original_nodes[i]);
    EXPECT_TRUE(std::equal(original_sources[i].begin(), original_sources[i].end(), original_nodes[i]->src));
  }
  const auto overlaps = [](const ggml_tensor *a, const ggml_tensor *other) {
    const auto begin = reinterpret_cast<std::uintptr_t>(a->data);
    const auto end = begin + ggml_nbytes(a);
    const auto second = reinterpret_cast<std::uintptr_t>(other->data);
    return begin < second + ggml_nbytes(other) && second < end;
  };
  if (!oracle && !explicit_alias)
    for (auto *root : {wroot, x, b}) EXPECT_FALSE(overlaps(out, root));
  if (!oracle && !explicit_alias && glu)
    for (auto *root : {uroot, bu}) EXPECT_FALSE(overlaps(out, root));
  std::vector<float> weights(static_cast<std::size_t>(k * (n + extra)), 0.0F), up_weights(weights.size(), 0.0F);
  for (int row = 0; row < n + extra; ++row)
    weights[static_cast<std::size_t>(row * k)] = row < extra ? 64.0F : float(((row - extra + seed) % 7) - 3);
  std::vector<float> activation(k, 0.0F), bias(n, 0.25F), up_bias(n, 1.0F);
  activation[0] = seed ? 0.5F : 1.0F;
  ggml_backend_tensor_set(wroot, weights.data(), 0, weights.size() * sizeof(float));
  // U is not part of the ordinary MM+ADD graph, hence is intentionally absent
  // from its allocation graph rather than given an artificial lifetime.
  if (glu) ggml_backend_tensor_set(uroot, up_weights.data(), 0, up_weights.size() * sizeof(float));
  ggml_backend_tensor_set(x, activation.data(), 0, activation.size() * sizeof(float));
  ggml_backend_tensor_set(b, bias.data(), 0, bias.size() * sizeof(float));
  if (glu) ggml_backend_tensor_set(bu, up_bias.data(), 0, up_bias.size() * sizeof(float));
  if (ggml_backend_graph_compute(o.backend, original) != GGML_STATUS_SUCCESS) {
    ADD_FAILURE() << "unmodified native graph compute"; return {};
  }
  const auto result = read(out);
  std::vector<float> expected(n);
  for (int row = 0; row < n; ++row) {
    const float gate = weights[static_cast<std::size_t>((row + extra) * k)] * activation[0] + bias[row];
    expected[row] = glu ? std::min(gate, 2.0F) : gate;
  }
  same_bits(result, expected);
  // Inputs and temporary roots may be dead after final execution. Only the
  // output is observed; no test asks a dead root to retain its old contents.
  return result;
}

TEST(GgmlCudaFusionAllocation, MatmulBiasUsesUnpublishedStorageRoots) {
  for (auto view : {ViewKind::dense, ViewKind::ordinary, ViewKind::none_alias})
    for (int seed : {0, 1}) {
      SCOPED_TRACE(int(view));
      SCOPED_TRACE(seed);
      same_bits(raw(false, view, seed, false), raw(false, view, seed, true));
    }
}
TEST(GgmlCudaFusionAllocation, FullGluUsesUnpublishedStorageRoots) {
  for (auto view : {ViewKind::dense, ViewKind::ordinary, ViewKind::none_alias})
    for (int seed : {0, 1}) {
      SCOPED_TRACE(int(view));
      SCOPED_TRACE(seed);
      same_bits(raw(true, view, seed, false), raw(true, view, seed, true));
    }
}
TEST(GgmlCudaFusionAllocation, ExplicitOutputOverlapKeepsOriginalRangeGuards) {
  for (int seed : {0, 1})
    same_bits(raw(true, ViewKind::dense, seed, false, true), raw(true, ViewKind::dense, seed, true));
}

TEST(GgmlCudaFusionAllocation, ProviderMeasuredWorkspaceBindsInternalProducerAndReplays) {
  auto context = c::Context::create(c::cuda::device(0).value(), g::ggml_cuda::provider()).value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  const auto input_desc = dense(c::DataType::kFloat32, {1, k});
  const auto weight_desc = dense(c::DataType::kFloat32, {k, n});
  const auto output_desc = dense(c::DataType::kFloat32, {1, n});
  auto builder = g::GraphBuilder::create().value();
  const auto x = builder.input(input_desc).value();
  const auto w = builder.input(weight_desc).value(), u = builder.input(weight_desc).value();
  const auto b = builder.input(output_desc).value(), bu = builder.input(output_desc).value();
  const auto internal_x = builder.add(c::Copy{}, x).value();
  const c::MatMul mm{.accumulation_type = c::DataType::kFloat32};
  const auto gate = builder.add(mm, internal_x, w).value();
  const auto gate_bias = builder.add(c::Binary{c::BinaryKind::kAdd}, gate, b).value();
  const auto up = builder.add(mm, internal_x, u).value();
  const auto up_bias = builder.add(c::Binary{c::BinaryKind::kAdd}, up, bu).value();
  const auto output = builder.add(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate}, gate_bias, up_bias).value();
  ASSERT_TRUE(builder.publish(output).ok());
  const auto logical = std::move(builder).finish().value();
  const auto compile = [&](std::size_t budget) {
    const std::array domains{g::StorageDomain{.resource = memory.ref(), .workspace_budget_bytes = budget}};
    const std::array targets{g::ExecutionTarget{.source = g::ExecutionSource::acquire(context).value(),
        .default_storage = g::StorageDomainId{0}}};
    const std::array lanes{g::ExecutionLane{.target = g::ExecutionTargetId{0}}};
    return g::compile(logical, {.storage_domains = domains, .targets = targets, .lanes = lanes},
        {.operation_fallback = g::OperationFallbackPolicy::kReject});
  };
  auto measured = compile(SIZE_MAX);
  ASSERT_TRUE(measured) << measured.error().message();
  const auto bytes = measured->executable.workspace_bytes();
  ASSERT_GT(bytes, 0U);
  EXPECT_EQ(measured->report.workspace_bytes(), bytes);
  auto exact = compile(bytes);
  ASSERT_TRUE(exact) << exact.error().message();
  EXPECT_EQ(exact->executable.workspace_bytes(), bytes);
  EXPECT_FALSE(compile(bytes - 1));
  RecordProperty("measured_workspace_bytes", std::to_string(bytes));
  // The short control exercises the original Graph storage budget. Actual
  // invocations below use exactly the successful measured/real native layout;
  // no native chunk count is guessed or accepted by a wider tolerance.
  std::vector<Buffer> retained;
  std::array queues{context.create_queue().value(), context.create_queue().value()};
  std::array invocations{exact->executable.create_invocation().value(), exact->executable.create_invocation().value()};
  std::vector<float> old_output;
  for (int round = 0; round < 2; ++round) {
    std::array<std::array<Buffer, 6>, 2> buffers;
    std::array<std::vector<float>, 2> expected;
    for (int slot = 0; slot < 2; ++slot) {
      const int seed = 1 + round * 2 + slot;
      std::vector<float> activation(k, 0.0F), weight(k * n, 0.0F), up_weight(k * n, 0.0F);
      std::vector<float> bias(n, 0.25F), up_bias_values(n, 1.0F), gate_values(n), ones(n, 1.0F);
      activation[0] = seed % 2 ? 0.5F : 1.0F;
      for (int col = 0; col < n; ++col) {
        weight[col] = float((col + seed) % 7 - 3);
        gate_values[col] = weight[col] * activation[0] + bias[col];
      }
      buffers[slot][0] = upload<float>(memory.ref(), activation);
      buffers[slot][1] = upload<float>(memory.ref(), weight);
      buffers[slot][2] = upload<float>(memory.ref(), up_weight);
      buffers[slot][3] = upload<float>(memory.ref(), bias);
      buffers[slot][4] = upload<float>(memory.ref(), up_bias_values);
      buffers[slot][5] = output_buffer(memory.ref(), output_desc);
      const std::array ids{x, w, u, b, bu};
      const std::array descriptions{input_desc, weight_desc, weight_desc, output_desc, output_desc};
      for (std::size_t i = 0; i < ids.size(); ++i)
        ASSERT_TRUE(invocations[slot].bind(ids[i], c::ConstTensorView::create(descriptions[i], buffers[slot][i].view()).value()).ok());
      ASSERT_TRUE(invocations[slot].bind(output, c::TensorView::create(output_desc, buffers[slot][5].view()).value()).ok());
      // A separate ordinary one-op native GLU is the numerical oracle. It
      // contains no MatMul span and therefore cannot share the new fuser.
      auto gate_buffer = upload<float>(memory.ref(), gate_values);
      auto one_buffer = upload<float>(memory.ref(), ones);
      auto oracle_buffer = output_buffer(memory.ref(), output_desc);
      const std::array oracle_inputs{
          c::ConstTensorView::create(output_desc, gate_buffer.view()).value(),
          c::ConstTensorView::create(output_desc, one_buffer.view()).value()};
      ASSERT_TRUE(ggml_cuda_tensor_test::execute(context,
          c::GatedActivation{.inputs = c::GatedInputKind::kSeparate}, oracle_inputs,
          c::TensorView::create(output_desc, oracle_buffer.view()).value()).ok());
      expected[slot] = half_test::read<float>(oracle_buffer);
    }
    for (auto mode : {g::ReplayMode::kCommands, g::ReplayMode::kRequireCapture}) {
      std::array<c::Completion, 2> done;
      for (int slot = 0; slot < 2; ++slot) {
        ASSERT_TRUE(g::cuda::prepare_replay(invocations[slot], queues[slot],
            {.mode = mode, .host_wait = g::HostWaitPolicy::kReject}).ok());
        auto submitted = invocations[slot].submit(queues[slot]);
        ASSERT_TRUE(submitted) << submitted.error().message();
        done[slot] = std::move(*submitted);
      }
      ASSERT_TRUE(done[1].wait().ok()); ASSERT_TRUE(done[0].wait().ok());
      for (int slot = 0; slot < 2; ++slot)
        same_bits(half_test::read<float>(buffers[slot][5]), expected[slot]);
    }
    // An explicitly overlapping public output remains invalid. Allocation
    // dependency extension is not an expanded public alias capability.
    auto rejected = exact->executable.create_invocation().value();
    const std::array ids{x, w, u, b, bu};
    const std::array descriptions{input_desc, weight_desc, weight_desc, output_desc, output_desc};
    for (std::size_t i = 0; i < ids.size(); ++i)
      ASSERT_TRUE(rejected.bind(ids[i], c::ConstTensorView::create(descriptions[i], buffers[0][i].view()).value()).ok());
    const auto aliased = rejected.bind(output,
        c::TensorView::create(output_desc, buffers[0][0].view()).value());
    if (aliased.ok())
      EXPECT_FALSE(g::cuda::prepare_replay(rejected, queues[0],
          {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kReject}).ok());
    if (round == 0) old_output = half_test::read<float>(buffers[0][5]);
    for (auto &slot_buffers : buffers) for (auto &buffer : slot_buffers) retained.push_back(std::move(buffer));
    if (round == 1) same_bits(half_test::read<float>(retained[5]), old_output);
  }
}
} // namespace native_fusion_allocation_test
} // namespace

#include <cuda_runtime_api.h>

namespace {
namespace grouped_projection_fusion_test {
constexpr std::int64_t reduction=32,columns=8,experts=12,slots=10;
struct Bank final {std::vector<std::byte> bytes;std::vector<float> decoded;};

Bank bank(c::GgufQuantizedType type,int seed){
  const auto native=type==c::GgufQuantizedType::kQ8_0?GGML_TYPE_Q8_0:GGML_TYPE_MXFP4;
  const auto description=c::gguf_grouped_weight_description(type,c::DataType::kFloat32,experts,columns,reduction).value();
  std::vector<float> source(static_cast<std::size_t>(experts*columns*reduction),0.0F);
  for(std::int64_t row=0;row<experts*columns;++row)
    source[static_cast<std::size_t>(row*reduction)]=
        float((row+seed)%7+1)*((row+seed)%2?-1.0F:1.0F)*(127.0F/512.0F);
  Bank result{std::vector<std::byte>(description.required_bytes().value()),std::vector<float>(source.size())};
  EXPECT_EQ(ggml_quantize_chunk(native,source.data(),result.bytes.data(),0,experts*columns,reduction,nullptr),result.bytes.size());
  const auto *traits=ggml_get_type_traits(native);
  if(!traits || !traits->to_float){ADD_FAILURE()<<"GGUF decoder unavailable";return result;}
  const auto row_bytes=ggml_row_size(native,reduction);
  for(std::int64_t row=0;row<experts*columns;++row)
    traits->to_float(result.bytes.data()+static_cast<std::size_t>(row)*row_bytes,
                     result.decoded.data()+row*reduction,reduction);
  return result;
}

std::vector<float> projection(const Bank&weights,std::span<const float> input,
                              std::span<const std::int32_t> ids){
  std::vector<float> result(ids.size()*columns);
  for(std::size_t row=0;row<ids.size();++row)for(std::int64_t col=0;col<columns;++col){
    double value=0.0;
    for(std::int64_t k=0;k<reduction;++k)
      value+=double(weights.decoded[(static_cast<std::size_t>(ids[row])*columns+col)*reduction+k])*
             double(input[row*reduction+k]);
    result[row*columns+col]=static_cast<float>(value);
  }
  return result;
}

std::vector<float> separate_glu(c::Context&context,hyle::mem::MemoryResourceRef resource,
                               const c::TensorDesc&description,std::span<const float> gate,
                               std::span<const float> up){
  auto gate_buffer=upload<float>(resource,gate),up_buffer=upload<float>(resource,up);
  auto output=output_buffer(resource,description);
  const std::array inputs{
      c::ConstTensorView::create(description,gate_buffer.view()).value(),
      c::ConstTensorView::create(description,up_buffer.view()).value()};
  const auto status=ggml_cuda_tensor_test::execute(context,
      c::GatedActivation{.inputs=c::GatedInputKind::kSeparate},inputs,
      c::TensorView::create(description,output.view()).value());
  EXPECT_TRUE(status.ok())<<status.message();
  return half_test::read<float>(output);
}

void same_glu(std::span<const float> actual,std::span<const float> expected){
  ASSERT_EQ(actual.size(),expected.size());
  // Reuse the existing F32 SiLU comparison domain in this TU. The new
  // grouped carrier does not receive a new or wider numerical tolerance.
  for(std::size_t i=0;i<actual.size();++i){SCOPED_TRACE(i);expect_value(actual[i],expected[i]);}
}

void exercise(c::GgufQuantizedType gate_type,c::GgufQuantizedType up_type,
              std::int64_t tokens,bool diagnostic,bool different_extent=false){
  const std::int64_t gate_experts=different_extent?experts-1:experts;
  const auto gate_desc=c::gguf_grouped_weight_description(gate_type,c::DataType::kFloat32,gate_experts,columns,reduction).value();
  const auto up_desc=c::gguf_grouped_weight_description(up_type,c::DataType::kFloat32,experts,columns,reduction).value();
  const auto input_desc=dense(c::DataType::kFloat32,{tokens,slots,reduction});
  const auto ids_desc=dense(c::DataType::kInt32,{tokens,slots});
  const auto output_desc=dense(c::DataType::kFloat32,{tokens,slots,columns});
  auto context=g::cuda::create_context().value();
  auto memory=c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice).value();
  const auto gate_bank=bank(gate_type,3),up_bank=bank(up_type,7);
  auto gate_weights=upload<std::byte>(memory.ref(),gate_bank.bytes);
  auto up_weights=upload<std::byte>(memory.ref(),up_bank.bytes);
  auto builder=g::GraphBuilder::create().value();
  // The negative extent control uses true dynamic external roots, whose real
  // backing has twelve banks while the first public descriptor permits eleven.
  // This avoids requiring a device fault even if a future regression ignores
  // that smaller extent. No mathematical input is marked as an output.
  const auto wg=(different_extent?builder.input(gate_desc):builder.constant(gate_desc)).value();
  const auto wu=(different_extent?builder.input(up_desc):builder.constant(up_desc)).value();
  const auto x=builder.input(input_desc).value(),indices=builder.input(ids_desc).value();
  const std::array gate_ports{wg,x,indices},up_ports{wu,x,indices};
  const auto gate=builder.result(builder.add_node(c::model_ops::grouped_matmul(),gate_ports).value()).value();
  const auto up=builder.result(builder.add_node(c::model_ops::grouped_matmul(),up_ports).value()).value();
  const auto output=builder.add(c::GatedActivation{.inputs=c::GatedInputKind::kSeparate},gate,up).value();
  std::vector<g::ValueId> published{output};
  g::ValueId neg;
  if(diagnostic){
    neg=builder.add(c::Unary{c::UnaryKind::kNeg},gate).value();
    published.insert(published.end(),{gate,up,neg});
  }
  for(const auto value:published)ASSERT_TRUE(builder.publish(value).ok());
  const auto logical=std::move(builder).finish().value();
  const std::array constants{
      g::ConstantBinding{wg,c::ConstTensorView::create(gate_desc,gate_weights.view()).value()},
      g::ConstantBinding{wu,c::ConstTensorView::create(up_desc,up_weights.view()).value()}};
  auto compiled=g::cuda::compile(logical,context,memory.ref(),
      {.constants=different_extent?std::span<const g::ConstantBinding>{}:std::span<const g::ConstantBinding>(constants)});
  ASSERT_TRUE(compiled)<<compiled.error().message();
  ASSERT_EQ(compiled->report.step_count(),1U);
  EXPECT_EQ(compiled->report.step(0)->provider.id,c::ggml_cuda::kProviderId);
  std::vector<Buffer> retained,retained_inputs;
  std::vector<std::vector<float>> retained_values;
  std::array queues{context.create_queue().value(),context.create_queue().value()};
  std::array calls{compiled->executable.create_invocation().value(),compiled->executable.create_invocation().value()};
  for(int round=0;round<2;++round){
    std::array<Buffer,2> activation_buffer,ids_buffer;
    std::array<std::vector<Buffer>,2> outputs;
    std::array<std::vector<float>,2> input_values,expected_gate,expected_up,expected_output;
    std::array<std::vector<std::int32_t>,2> selected;
    for(std::size_t lane=0;lane<calls.size();++lane){
      SCOPED_TRACE(::testing::Message()<<"T="<<tokens<<" round="<<round<<" lane="<<lane<<" diagnostic="<<diagnostic);
      input_values[lane].resize(static_cast<std::size_t>(tokens*slots*reduction),0.0F);
      selected[lane].resize(static_cast<std::size_t>(tokens*slots));
      for(std::size_t row=0;row<selected[lane].size();++row){
        // A deliberately repeated expert remains legal per token. 127 times
        // a binary power gives an exactly representable native Q8_1 row.
        selected[lane][row]=static_cast<std::int32_t>((row/2+lane+static_cast<std::size_t>(round))%static_cast<std::size_t>(gate_experts));
        input_values[lane][row*reduction]=(127.0F/128.0F)*((row+lane+static_cast<std::size_t>(round))%2?-.5F:1.0F);
      }
      expected_gate[lane]=projection(gate_bank,input_values[lane],selected[lane]);
      expected_up[lane]=projection(up_bank,input_values[lane],selected[lane]);
      expected_output[lane]=separate_glu(context,memory.ref(),output_desc,expected_gate[lane],expected_up[lane]);
      activation_buffer[lane]=upload<float>(memory.ref(),input_values[lane]);
      ids_buffer[lane]=upload<std::int32_t>(memory.ref(),selected[lane]);
      ASSERT_TRUE(calls[lane].bind(x,c::ConstTensorView::create(input_desc,activation_buffer[lane].view()).value()).ok());
      ASSERT_TRUE(calls[lane].bind(indices,c::ConstTensorView::create(ids_desc,ids_buffer[lane].view()).value()).ok());
      if(different_extent){
        ASSERT_TRUE(calls[lane].bind(wg,c::ConstTensorView::create(gate_desc,gate_weights.view()).value()).ok());
        ASSERT_TRUE(calls[lane].bind(wu,c::ConstTensorView::create(up_desc,up_weights.view()).value()).ok());
      }
      for(const auto value:published){
        outputs[lane].push_back(output_buffer(memory.ref(),output_desc));
        ASSERT_TRUE(calls[lane].bind(value,c::TensorView::create(output_desc,outputs[lane].back().view()).value()).ok());
      }
      ASSERT_TRUE(g::cuda::prepare_replay(calls[lane],queues[lane],
          {.mode=g::ReplayMode::kRequireCapture,.host_wait=g::HostWaitPolicy::kReject}).ok());
    }
    std::array<c::Completion,2> completions;
    for(std::size_t lane=0;lane<calls.size();++lane){
      auto submitted=calls[lane].submit(queues[lane]);ASSERT_TRUE(submitted)<<submitted.error().message();
      completions[lane]=std::move(*submitted);
    }
    ASSERT_TRUE(completions[1].wait().ok());ASSERT_TRUE(completions[0].wait().ok());completions={};
    for(std::size_t lane=0;lane<calls.size();++lane){
      same_glu(half_test::read<float>(outputs[lane][0]),expected_output[lane]);
      if(diagnostic){
        native_fusion_layout_test::same_bits(half_test::read<float>(outputs[lane][1]),expected_gate[lane]);
        native_fusion_layout_test::same_bits(half_test::read<float>(outputs[lane][2]),expected_up[lane]);
        auto negative=expected_gate[lane];for(auto&value:negative)value=-value;
        native_fusion_layout_test::same_bits(half_test::read<float>(outputs[lane][3]),negative);
      }
      EXPECT_EQ(half_test::read<std::int32_t>(ids_buffer[lane]),selected[lane]);
      if(different_extent){
        std::vector<std::vector<float>> previous;
        for(const auto&buffer:outputs[lane])previous.push_back(half_test::read<float>(buffer));
        for(const std::int32_t bad:{-1,static_cast<std::int32_t>(gate_experts),static_cast<std::int32_t>(experts)}){
          auto invalid=selected[lane];invalid.back()=bad;
          ASSERT_TRUE(ids_buffer[lane].view().write(std::as_bytes(std::span(invalid))).ok());
          auto failed=calls[lane].submit(queues[lane]);ASSERT_TRUE(failed)<<failed.error().message();
          EXPECT_TRUE(failed->wait().IsOutOfRange());*failed={};
          for(std::size_t port=0;port<outputs[lane].size();++port)
            native_fusion_layout_test::same_bits(half_test::read<float>(outputs[lane][port]),previous[port]);
          EXPECT_EQ(half_test::read<std::int32_t>(ids_buffer[lane]),invalid);
          ASSERT_TRUE(ids_buffer[lane].view().write(std::as_bytes(std::span(selected[lane]))).ok());
          // Accepted failure retires this slot. Recovery creates a new
          // invocation from the same immutable executable, preserving buffers.
          EXPECT_FALSE(calls[lane].reset().ok());
          EXPECT_FALSE(calls[lane].submit(queues[lane]));
          calls[lane]=compiled->executable.create_invocation().value();
          ASSERT_TRUE(calls[lane].bind(wg,c::ConstTensorView::create(gate_desc,gate_weights.view()).value()).ok());
          ASSERT_TRUE(calls[lane].bind(wu,c::ConstTensorView::create(up_desc,up_weights.view()).value()).ok());
          ASSERT_TRUE(calls[lane].bind(x,c::ConstTensorView::create(input_desc,activation_buffer[lane].view()).value()).ok());
          ASSERT_TRUE(calls[lane].bind(indices,c::ConstTensorView::create(ids_desc,ids_buffer[lane].view()).value()).ok());
          for(std::size_t port=0;port<published.size();++port)
            ASSERT_TRUE(calls[lane].bind(published[port],c::TensorView::create(output_desc,outputs[lane][port].view()).value()).ok());
          ASSERT_TRUE(g::cuda::prepare_replay(calls[lane],queues[lane],
              {.mode=g::ReplayMode::kRequireCapture,.host_wait=g::HostWaitPolicy::kReject}).ok());
          auto recovered=calls[lane].submit(queues[lane]);ASSERT_TRUE(recovered)<<recovered.error().message();
          ASSERT_TRUE(recovered->wait().ok());*recovered={};
          for(std::size_t port=0;port<outputs[lane].size();++port)
            native_fusion_layout_test::same_bits(half_test::read<float>(outputs[lane][port]),previous[port]);
        }
      }
      for(auto&buffer:outputs[lane]){retained_values.push_back(half_test::read<float>(buffer));retained.push_back(std::move(buffer));}
      retained_inputs.push_back(std::move(activation_buffer[lane]));retained_inputs.push_back(std::move(ids_buffer[lane]));
    }
    // The next round rebinds both invocations to fresh input/output allocations.
    // Previous externally owned outputs remain valid after those rebindings.
    for(std::size_t i=0;i<retained.size();++i)
      native_fusion_layout_test::same_bits(half_test::read<float>(retained[i]),retained_values[i]);
  }
}

TEST(GgmlCudaGroupedFusion, Q8TopTenFinalOnlyUsesPublicGraphAcrossTwoTokenExtents){
  for(const auto tokens:{std::int64_t{1},std::int64_t{2}})
    exercise(c::GgufQuantizedType::kQ8_0,c::GgufQuantizedType::kQ8_0,tokens,false);
}
TEST(GgmlCudaGroupedFusion, Mxfp4TopTenFinalOnlyUsesPublicGraphAcrossTwoTokenExtents){
  for(const auto tokens:{std::int64_t{1},std::int64_t{2}})
    exercise(c::GgufQuantizedType::kMxfp4,c::GgufQuantizedType::kMxfp4,tokens,false);
}
TEST(GgmlCudaGroupedFusion, PublishedQuantizedProjectionsAndReadersRemainObservable){
  exercise(c::GgufQuantizedType::kQ8_0,c::GgufQuantizedType::kQ8_0,2,true);
  exercise(c::GgufQuantizedType::kMxfp4,c::GgufQuantizedType::kMxfp4,1,true);
  exercise(c::GgufQuantizedType::kQ8_0,c::GgufQuantizedType::kMxfp4,2,true);
}
TEST(GgmlCudaGroupedFusion, CheckedIdsSeparateExpertExtentsAndRecoverAcrossInvocations){
  exercise(c::GgufQuantizedType::kQ8_0,c::GgufQuantizedType::kQ8_0,1,true,true);
}
TEST(GgmlCudaGroupedFusion, NativeIndexedChannelsRespectActualDeviceGridY){
  native_fusion_layout_test::Owners owners;
  ASSERT_TRUE(owners.initialize());
  const int device=c::ggml_cuda::bridge::device_index(0);
  cudaDeviceProp properties{};ASSERT_EQ(cudaGetDeviceProperties(&properties,device),cudaSuccess);
  const auto maximum=static_cast<std::int64_t>(properties.maxGridSize[1]);
  ASSERT_GT(maximum,0);
  RecordProperty("actual_max_grid_y",std::to_string(maximum));
  const auto supported=[&](std::int64_t channels){
    auto *w=ggml_new_tensor_3d(owners.ctx,GGML_TYPE_Q8_0,reduction,columns,experts);
    auto *x=ggml_new_tensor_3d(owners.ctx,GGML_TYPE_F32,reduction,channels,1);
    auto *ids=ggml_new_tensor_2d(owners.ctx,GGML_TYPE_I32,channels,1);
    auto *mm=ggml_mul_mat_id(owners.ctx,w,x,ids);
    EXPECT_TRUE(ggml_prec_set_acc(mm,GGML_PREC_F32));
    EXPECT_TRUE(ggml_prec_set_src(mm,GGML_PREC_F32,1));
    return ggml_backend_supports_op(owners.backend,mm);
  };
  EXPECT_TRUE(supported(slots));
  // Metadata only: never allocate a payload, submit, or launch a deliberately
  // oversized grid. Native channel counters must remain representable too.
  if(maximum<INT32_MAX)EXPECT_FALSE(supported(maximum+1));
  else RecordProperty("oversized_control","not representable by native int32 channel extent");
}
} // namespace grouped_projection_fusion_test
} // namespace

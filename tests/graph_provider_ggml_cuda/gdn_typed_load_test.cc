#include "hyle/graph/providers/ggml_cuda.h"
#include "hyle/graph/executable.h"
#include "hyle/compute/model_ops.h"
#include "../compute_provider_ggml_cuda/tensor_test_support.h"
#include "../compute_models/gdn_snapshot_fixture.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
using BF16 = hyle::simd::bfloat16_t;
using hyle::mem::Buffer;
using ggml_cuda_tensor_test::output_buffer;
using ggml_cuda_tensor_test::upload;

hyle::Result<g::CompiledGraph> compile_graph(const g::Graph &graph,
    c::Context &context, hyle::mem::MemoryResourceRef resource) {
  const std::array storage{g::StorageDomain{.resource = resource}};
  const std::array targets{g::ExecutionTarget{
      .source = g::ExecutionSource::acquire(context).value(),
      .default_storage = g::StorageDomainId{0}}};
  const std::array lanes{g::ExecutionLane{.target = g::ExecutionTargetId{0}}};
  return g::compile(graph, {.storage_domains = storage, .targets = targets,
                            .lanes = lanes});
}

// The reference keeps the original all-F32 native GDN and its reduction tree.
// The independent recurrence oracle below also checks snapshots/head mapping.
std::array<std::vector<float>, 2> direct_reference(c::Context &context,
    hyle::mem::MemoryResourceRef resource,
    const gdn_snapshot_fixture::Fixture &fixture, std::uint32_t snapshots,
    bool recurrent) {
  std::vector<Buffer> storage;
  std::vector<c::ConstTensorView> inputs;
  for (std::size_t port = 0; port < fixture.inputs.size(); ++port) {
    storage.push_back(upload<float>(resource, fixture.data[port]));
    inputs.push_back(c::ConstTensorView::create(fixture.inputs[port],
        storage.back().view()).value());
  }
  if (recurrent) {
    const auto parameter = c::TensorDesc::dense(c::DataType::kFloat32,
        std::array<std::int64_t, 4>{1, 1, fixture.value_heads, 1}).value();
    for (const auto [position, value] :
         {std::pair{5, -0.125F}, std::pair{6, 0.F}}) {
      const std::vector<float> values(std::size_t(fixture.value_heads), value);
      storage.push_back(upload<float>(resource, values));
      inputs.insert(inputs.begin() + position, c::ConstTensorView::create(
          parameter, storage.back().view()).value());
    }
  }
  const auto descriptions = fixture.outputs(snapshots);
  std::array output_storage{output_buffer(resource, descriptions[0]),
                            output_buffer(resource, descriptions[1])};
  const std::array outputs{
      c::TensorView::create(descriptions[0], output_storage[0].view()).value(),
      c::TensorView::create(descriptions[1], output_storage[1].view()).value()};
  const auto operation = recurrent ? c::model_ops::recurrent_gated_delta_net(snapshots)
      : c::Operation(c::GatedDeltaNet{.snapshot_count = snapshots});
  auto prepared = c::prepare(context, operation, inputs, outputs);
  if (!prepared) {
    ADD_FAILURE() << prepared.error().message();
    return {};
  }
  auto workspace = ggml_cuda_test::workspace_buffer(*prepared, resource);
  auto invocation = prepared->create_invocation(workspace.view()).value();
  auto queue = context.create_queue().value();
  auto done = invocation.submit(queue, inputs, outputs);
  if (!done) {
    ADD_FAILURE() << done.error().message();
    return {};
  }
  const auto waited = done->wait();
  if (!waited.ok()) {
    ADD_FAILURE() << waited.message();
    return {};
  }
  return {half_test::read<float>(output_storage[0]),
          half_test::read<float>(output_storage[1])};
}

void exercise(c::Context &context, hyle::mem::MemoryResourceRef resource,
    unsigned typed, std::int64_t tokens, bool protect_intermediates,
    bool channel_decay = false, bool recurrent = false) {
  const std::uint32_t snapshots = tokens == 1 ? 1 : 2;
  gdn_snapshot_fixture::Fixture fixture(3, 2, tokens, 2, 6, 32, channel_decay);
  auto descriptions = fixture.inputs;
  std::array<std::vector<BF16>, 6> narrow;
  for (const auto port : {2U, 4U}) {
    if (!(typed & (port == 2 ? 1U : 2U))) continue;
    descriptions[port] = c::TensorDesc::dense(c::DataType::kBFloat16,
        fixture.inputs[port].dimensions()).value();
    narrow[port].resize(fixture.data[port].size());
    for (std::size_t i = 0; i < narrow[port].size(); ++i) {
      narrow[port][i] = half_test::encode<BF16>(fixture.data[port][i]);
      fixture.data[port][i] = float(half_test::decode(narrow[port][i]));
    }
  }
  // A real BF16 view with row padding: its native strides must survive
  // bypassing the dense F32 Cast, rather than being inferred from that Cast.
  const bool padded = typed == 3 && tokens == 3 && !protect_intermediates;
  if (padded) {
    descriptions[2] = c::TensorDesc::dense(c::DataType::kBFloat16,
        std::array<std::int64_t, 4>{2, tokens, 6, 40}).value();
  }
  const auto reference = direct_reference(context, resource, fixture, snapshots, recurrent);
  auto activated = fixture;
  if (recurrent) {
    for (std::size_t i = 0; i < activated.data[3].size(); ++i)
      activated.data[3][i] = float(-0.125 * std::log1p(std::exp(double(fixture.data[3][i]))));
    for (std::size_t i = 0; i < activated.data[4].size(); ++i)
      activated.data[4][i] = float(1.0 / (1.0 + std::exp(-double(fixture.data[4][i]))));
  }
  const auto oracle = activated.oracle(snapshots);
  auto builder = g::GraphBuilder::create().value();
  std::array<g::ValueId, 6> sources, arguments;
  for (std::size_t port = 0; port < sources.size(); ++port) {
    sources[port] = builder.input(descriptions[port]).value();
    // Real NodeOutput producers behind distinct metadata frontiers reproduce
    // ordinary discovery's prefix cuts. Raw Graph inputs alone cannot expose
    // the missing cross-frontier suffix proposal.
    auto value = builder.add(c::Copy{}, sources[port]).value();
    if (padded && port == 2)
      value = builder.add(c::View::slice(3, 0, 32).value(), value).value();
    value = builder.add(c::View::reshape(c::TensorShape::create(
        fixture.inputs[port].dimensions()).value()).value(), value).value();
    arguments[port] = narrow[port].empty() ? value
        : builder.add(c::Cast{c::DataType::kFloat32}, value).value();
  }
  std::vector<g::ValueId> operation_inputs(arguments.begin(), arguments.end());
  std::array<g::ValueId, 2> parameters;
  const auto parameter = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 4>{1, 1, fixture.value_heads, 1}).value();
  if (recurrent)
    for (std::size_t port = 0; port < parameters.size(); ++port) {
      parameters[port] = builder.input(parameter).value();
      operation_inputs.insert(operation_inputs.begin() + 5 + port, parameters[port]);
    }
  const auto operation = recurrent ? c::model_ops::recurrent_gated_delta_net(snapshots)
      : c::Operation(c::GatedDeltaNet{.snapshot_count = snapshots});
  const auto node = builder.add_node(operation, operation_inputs).value();
  const std::array results{builder.result(node, 0).value(),
                           builder.result(node, 1).value()};
  for (const auto id : results) ASSERT_TRUE(builder.publish(id).ok());
  g::ValueId beta_reader;
  if (protect_intermediates) {
    ASSERT_TRUE(builder.publish(arguments[2]).ok());
    beta_reader = builder.add(c::Copy{}, arguments[4]).value();
    ASSERT_TRUE(builder.publish(beta_reader).ok());
  }
  const auto graph = std::move(builder).finish().value();
  auto compiled = compile_graph(graph, context, resource);
  ASSERT_TRUE(compiled) << compiled.error().message();
  const auto step_for = [&](g::NodeId target) {
    for (std::size_t step = 0; step < compiled->report.step_count(); ++step)
      for (std::size_t n = 0; n < compiled->report.step(step)->node_count; ++n)
        if (compiled->report.step_node(step, n).value() == target) return step;
    return SIZE_MAX;
  };
  const auto gdn_step = step_for(node);
  ASSERT_NE(gdn_step, SIZE_MAX);
  for (const auto port : {2U, 4U}) {
    if (narrow[port].empty()) continue;
    const auto cast_step = step_for(graph.producer(arguments[port]).value());
    ASSERT_NE(cast_step, SIZE_MAX);
    const bool joined = !protect_intermediates && (port == 2 || !recurrent);
    EXPECT_EQ(cast_step == gdn_step, joined) << "port=" << port;
    if (joined) {
      EXPECT_EQ(compiled->report.step(gdn_step)->kind, g::CompileStepKind::kProviderRegion);
      EXPECT_EQ(compiled->report.step(gdn_step)->implementation, c::ProviderImplementationId{2});
    }
  }
  std::vector<Buffer> input_storage;
  const auto outputs = fixture.outputs(snapshots);
  std::array output_storage{output_buffer(resource, outputs[0]),
                            output_buffer(resource, outputs[1])};
  std::vector<Buffer> visible;
  std::vector<Buffer> parameter_storage;
  if (recurrent)
    for (const float value : {-0.125F, 0.F}) {
      const std::vector<float> values(std::size_t(fixture.value_heads), value);
      parameter_storage.push_back(upload<float>(resource, values));
    }
  if (protect_intermediates) {
    visible.push_back(output_buffer(resource, fixture.inputs[2]));
    visible.push_back(output_buffer(resource, fixture.inputs[4]));
  }
  auto queue = context.create_queue().value();
  auto invocation = compiled->executable.create_invocation().value();
  if (recurrent)
    for (std::size_t port = 0; port < parameters.size(); ++port)
      ASSERT_TRUE(invocation.bind(parameters[port], c::ConstTensorView::create(
          parameter, parameter_storage[port].view()).value()).ok());
  for (std::size_t port = 0; port < sources.size(); ++port) {
    if (narrow[port].empty())
      input_storage.push_back(upload<float>(resource, fixture.data[port]));
    else if (padded && port == 2) {
      std::vector<BF16> physical(descriptions[port].required_bytes().value() / sizeof(BF16),
                                 BF16::from_bits(0x7fc1));
      for (std::size_t i = 0; i < narrow[port].size(); ++i)
        physical[(i / 32) * 40 + i % 32] = narrow[port][i];
      input_storage.push_back(upload<BF16>(resource, physical));
    } else
      input_storage.push_back(upload<BF16>(resource, narrow[port]));
    ASSERT_TRUE(invocation.bind(sources[port], c::ConstTensorView::create(
        descriptions[port], input_storage.back().view()).value()).ok());
  }
  for (std::size_t port = 0; port < results.size(); ++port)
    ASSERT_TRUE(invocation.bind(results[port], c::TensorView::create(
        outputs[port], output_storage[port].view()).value()).ok());
  if (protect_intermediates) {
    ASSERT_TRUE(invocation.bind(arguments[2], c::TensorView::create(
        fixture.inputs[2], visible[0].view()).value()).ok());
    ASSERT_TRUE(invocation.bind(beta_reader, c::TensorView::create(
        fixture.inputs[4], visible[1].view()).value()).ok());
  }
  const std::array bindings{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  const auto status = invocation.prepare_replay(bindings,
      {.mode = g::ReplayMode::kRequireCapture});
  ASSERT_TRUE(status.ok()) << status.message();
  auto done = invocation.submit(queue);
  ASSERT_TRUE(done) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  for (std::size_t port = 0; port < results.size(); ++port) {
    const auto actual = half_test::read<float>(output_storage[port]);
    ASSERT_EQ(actual.size(), reference[port].size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
      ASSERT_TRUE(std::isfinite(actual[i])) << port << ':' << i;
      EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]),
                std::bit_cast<std::uint32_t>(reference[port][i])) << port << ':' << i;
      EXPECT_NEAR(actual[i], oracle[port][i], 3.e-6) << port << ':' << i;
    }
  }
  if (protect_intermediates) {
    EXPECT_EQ(half_test::read<float>(visible[0]), fixture.data[2]);
    EXPECT_EQ(half_test::read<float>(visible[1]), fixture.data[4]);
  }
  EXPECT_EQ(half_test::read<float>(input_storage[5]), fixture.data[5]);
}

TEST(GgmlCudaGdnTypedLoad, IndependentVAndBetaMatchOriginalAcrossTokensAndSnapshots) {
  auto context = c::Context::create(c::cuda::device(0).value(), g::ggml_cuda::provider()).value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  for (unsigned typed = 0; typed < 4; ++typed)
    for (const std::int64_t tokens : {1, 3}) {
      SCOPED_TRACE(::testing::Message() << "typed=" << typed << " tokens=" << tokens);
      exercise(context, memory.ref(), typed, tokens, false, typed == 3 && tokens == 3);
    }
}

TEST(GgmlCudaGdnTypedLoad, PublishedCastAndExtraReaderRetainF32Values) {
  auto context = c::Context::create(c::cuda::device(0).value(), g::ggml_cuda::provider()).value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  exercise(context, memory.ref(), 3, 3, true);
}

TEST(GgmlCudaGdnTypedLoad, RecurrentKeepsF32SigmoidAfterRawBetaWiden) {
  auto context = c::Context::create(c::cuda::device(0).value(), g::ggml_cuda::provider()).value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  exercise(context, memory.ref(), 3, 3, false, false, true);
}

TEST(GgmlCudaGdnTypedLoad, SingleSnapshotEndpointsKeepOldStateAcrossInvocations) {
  auto context = c::Context::create(c::cuda::device(0).value(), g::ggml_cuda::provider()).value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  for (const std::int64_t tokens : {1, 3}) {
    SCOPED_TRACE(tokens);
    const std::array fixtures{
        gdn_snapshot_fixture::Fixture(11, 2, tokens, 2, 6, 32, true),
        gdn_snapshot_fixture::Fixture(19, 2, tokens, 2, 6, 32, true)};
    const auto descriptions = fixtures[0].outputs(1);
    auto builder = g::GraphBuilder::create().value();
    std::array<g::ValueId, 6> inputs;
    for (std::size_t port = 0; port < inputs.size(); ++port)
      inputs[port] = builder.input(fixtures[0].inputs[port]).value();
    const auto node = builder.add_node(c::Operation(c::GatedDeltaNet{}), inputs).value();
    const std::array endpoints{builder.result(node, 0).value(), builder.result(node, 1).value()};
    // Both public endpoints escape and also have a real additional reader.
    // Publishing them first preserves the original two-output native split.
    for (const auto value : endpoints) ASSERT_TRUE(builder.publish(value).ok());
    const std::array readers{builder.add(c::Copy{}, endpoints[0]).value(),
                             builder.add(c::Copy{}, endpoints[1]).value()};
    for (const auto value : readers) ASSERT_TRUE(builder.publish(value).ok());
    const auto graph = std::move(builder).finish().value();
    auto compiled = compile_graph(graph, context, memory.ref());
    ASSERT_TRUE(compiled) << compiled.error().message();
    auto first = compiled->executable.create_invocation().value();
    auto second = compiled->executable.create_invocation().value();
    const std::array slots{&first, &second};
    std::array<std::vector<Buffer>, 2> input_storage, output_storage;
    for (std::size_t slot = 0; slot < slots.size(); ++slot) {
      for (std::size_t port = 0; port < inputs.size(); ++port) {
        input_storage[slot].push_back(upload<float>(memory.ref(), fixtures[slot].data[port]));
        ASSERT_TRUE(slots[slot]->bind(inputs[port], c::ConstTensorView::create(
            fixtures[slot].inputs[port], input_storage[slot].back().view()).value()).ok());
      }
      for (std::size_t port = 0; port < endpoints.size(); ++port) {
        output_storage[slot].push_back(output_buffer(memory.ref(), descriptions[port]));
        ASSERT_TRUE(slots[slot]->bind(endpoints[port], c::TensorView::create(
            descriptions[port], output_storage[slot].back().view()).value()).ok());
      }
      for (std::size_t port = 0; port < readers.size(); ++port) {
        output_storage[slot].push_back(output_buffer(memory.ref(), descriptions[port]));
        ASSERT_TRUE(slots[slot]->bind(readers[port], c::TensorView::create(
            descriptions[port], output_storage[slot].back().view()).value()).ok());
      }
    }
    auto queue = context.create_queue().value();
    const std::array bindings{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
    for (auto *slot : slots) {
      const auto prepared = slot->prepare_replay(bindings, {.mode = g::ReplayMode::kRequireCapture});
      ASSERT_TRUE(prepared.ok()) << prepared.message();
    }
    auto first_done = first.submit(queue);
    ASSERT_TRUE(first_done) << first_done.error().message();
    auto second_done = second.submit(queue);
    ASSERT_TRUE(second_done) << second_done.error().message();
    ASSERT_TRUE(second_done->wait().ok());
    ASSERT_TRUE(first_done->wait().ok());
    // Keep both Completions and both old-state buffers alive while observing
    // either result. The immutable program must not retain one slot's pointer.
    for (std::size_t slot = 0; slot < slots.size(); ++slot) {
      const auto reference = direct_reference(context, memory.ref(), fixtures[slot], 1, false);
      const auto oracle = fixtures[slot].oracle(1);
      for (std::size_t port = 0; port < endpoints.size(); ++port) {
        const auto actual = half_test::read<float>(output_storage[slot][port]);
        const auto copied = half_test::read<float>(output_storage[slot][port + 2]);
        ASSERT_EQ(actual.size(), reference[port].size());
        EXPECT_EQ(actual, copied);
        for (std::size_t i = 0; i < actual.size(); ++i) {
          EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]),
                    std::bit_cast<std::uint32_t>(reference[port][i])) << slot << ':' << port << ':' << i;
          EXPECT_NEAR(actual[i], oracle[port][i], 3.e-6) << slot << ':' << port << ':' << i;
        }
      }
      EXPECT_EQ(half_test::read<float>(input_storage[slot][5]), fixtures[slot].data[5]);
    }
  }
}
} // namespace

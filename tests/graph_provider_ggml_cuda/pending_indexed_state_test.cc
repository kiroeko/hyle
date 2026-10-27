#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "hyle/compute/model_ops.h"
#include "hyle/graph/cuda.h"
#include "hyle/graph/detail/output_source.h"
#include "hyle/graph/detail/store_source.h"
#include "hyle/graph/slots.h"
#include "../compute_models/gdn_snapshot_fixture.h"
#include "../compute_provider_ggml_cuda/tensor_test_support.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace gd = g::detail;
namespace s = g::slots;
namespace m = hyle::mem;
using ggml_cuda_tensor_test::dense;
using ggml_cuda_tensor_test::output_buffer;
using ggml_cuda_tensor_test::upload;

constexpr std::size_t kB = 2, kP = 8;
constexpr std::array<std::uint32_t, kB> kSources{1, 5}, kDestinations{3, 7};

template<class T> T take(hyle::Result<T> value, const char *what) {
  if (!value) throw std::runtime_error(std::string(what) + ": " + std::string(value.error().message()));
  return std::move(*value);
}
void check(hyle::Status value, const char *what) {
  if (!value.ok()) throw std::runtime_error(std::string(what) + ": " + std::string(value.message()));
}
void expect_bits(std::span<const float> actual, std::span<const float> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i)
    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]), std::bit_cast<std::uint32_t>(expected[i])) << i;
}

// The callback does no CUDA work. Every exit releases both gates before the
// original invocation/resource owners unwind; the watchdog bounds host waits.
struct Gate final {
  cudaStream_t stream = nullptr;
  std::atomic<bool> released{false}, timed_out{false};
  std::mutex mutex;
  std::condition_variable changed;
  std::thread watchdog;
  static void CUDART_CB wait(void *opaque) {
    static_cast<Gate *>(opaque)->released.wait(false, std::memory_order_acquire);
  }
  void arm(cudaStream_t value) {
    stream = value;
    watchdog = std::thread([this] {
      std::unique_lock lock(mutex);
      if (!changed.wait_for(lock, std::chrono::seconds(10),
              [this] { return released.load(std::memory_order_acquire); })) {
        timed_out.store(true, std::memory_order_release);
        release();
      }
    });
  }
  void release() noexcept {
    released.store(true, std::memory_order_release);
    released.notify_all(); changed.notify_all();
  }
  ~Gate() {
    release();
    if (watchdog.joinable()) watchdog.join();
    if (stream) EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  }
};

// Only mathematical fixture data is shared. Each reference is the original
// dense public CUDA operation, not the indexed path under test.
struct Math final {
  bool convolution = false;
  c::TensorDesc item, output;
  std::array<c::TensorDesc, 5> inputs;
  std::array<std::vector<float>, 5> data;
  std::vector<float> initial, expected_output, expected_state;
};
Math make_math(c::Context &context, m::MemoryResourceRef resource, bool convolution) {
  Math result;
  result.convolution = convolution;
  if (!convolution) {
    const gdn_snapshot_fixture::Fixture fixture(9, kB, 4, 2, 2, 16);
    result.item = dense(c::DataType::kFloat32, {2, 16, 16});
    result.output = fixture.inputs[2]; result.initial = fixture.data[5];
    std::array<m::Buffer, 6> input_storage;
    std::array<c::ConstTensorView, 6> inputs;
    for (std::size_t port = 0; port < inputs.size(); ++port) {
      input_storage[port] = upload<float>(resource, fixture.data[port]);
      inputs[port] = take(c::ConstTensorView::create(fixture.inputs[port],
          input_storage[port].view()), "dense GDN input");
      if (port < result.inputs.size()) {
        result.inputs[port] = fixture.inputs[port]; result.data[port] = fixture.data[port];
      }
    }
    const auto descriptions = fixture.outputs(1);
    std::array outputs_storage{output_buffer(resource, descriptions[0]), output_buffer(resource, descriptions[1])};
    const std::array outputs{
        take(c::TensorView::create(descriptions[0], outputs_storage[0].view()), "dense GDN Y"),
        take(c::TensorView::create(descriptions[1], outputs_storage[1].view()), "dense GDN state")};
    auto prepared = take(c::prepare(context, c::Operation(c::GatedDeltaNet{}), inputs, outputs), "dense GDN prepare");
    auto workspace = ggml_cuda_test::workspace_buffer(prepared, resource);
    auto invocation = take(prepared.create_invocation(workspace.view()), "dense GDN invocation");
    auto queue = take(context.create_queue(), "dense GDN Queue");
    auto done = take(invocation.submit(queue, inputs, outputs), "dense GDN submit");
    check(done.wait(), "dense GDN wait");
    result.expected_output = half_test::read<float>(outputs_storage[0]);
    result.expected_state = half_test::read<float>(outputs_storage[1]);
  } else {
    constexpr std::size_t channels = 128, history = 3;
    result.item = dense(c::DataType::kFloat32, {history, channels});
    result.inputs[0] = dense(c::DataType::kFloat32, {kB, 1, channels});
    result.inputs[1] = dense(c::DataType::kFloat32, {channels, history + 1});
    result.output = result.inputs[0];
    result.initial.resize(kB * history * channels);
    result.data[0].resize(kB * channels); result.data[1].resize(channels * (history + 1));
    for (std::size_t i = 0; i < result.initial.size(); ++i)
      result.initial[i] = static_cast<float>(static_cast<int>(i % 29) - 14) / 32.F;
    for (std::size_t i = 0; i < result.data[0].size(); ++i)
      result.data[0][i] = static_cast<float>(static_cast<int>(i % 19) - 9) / 16.F;
    for (std::size_t i = 0; i < result.data[1].size(); ++i)
      result.data[1][i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 32.F;
    std::vector<float> logical(kB * (history + 1) * channels);
    result.expected_state.resize(result.initial.size());
    for (std::size_t batch = 0; batch < kB; ++batch) {
      auto first = logical.begin() + batch * (history + 1) * channels;
      std::copy_n(result.initial.begin() + batch * history * channels, history * channels, first);
      std::copy_n(result.data[0].begin() + batch * channels, channels, first + history * channels);
      std::copy_n(first + channels, history * channels,
          result.expected_state.begin() + batch * history * channels);
    }
    auto samples = upload<float>(resource, logical), weights = upload<float>(resource, result.data[1]);
    auto output = output_buffer(resource, result.output);
    const std::array inputs{
        take(c::ConstTensorView::create(dense(c::DataType::kFloat32, {kB, history + 1, channels}), samples.view()), "dense conv samples"),
        take(c::ConstTensorView::create(result.inputs[1], weights.view()), "dense conv weights")};
    check(ggml_cuda_tensor_test::execute(context, c::model_ops::causal_conv1d(), inputs,
        take(c::TensorView::create(result.output, output.view()), "dense conv output")), "dense conv execute");
    result.expected_output = half_test::read<float>(output);
  }
  return result;
}

struct ConsumerGraph final {
  g::Graph graph;
  g::NodeId gather{}, store{};
  g::ValueId activation{};
  std::array<g::ValueId, 5> inputs;
};
ConsumerGraph make_consumer(const Math &math, const s::Schema &schema) {
  auto builder = take(g::GraphBuilder::create(), "B GraphBuilder");
  const auto bank = take(builder.input(take(schema.capacity_description(), "capacity")), "B pool");
  const auto ids = dense(c::DataType::kInt32, {kB});
  const auto src = take(builder.input(ids), "B source IDs"), dst = take(builder.input(ids), "B destination IDs");
  const auto gathered = take(builder.add(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked}, bank, src), "B Gather");
  const auto reshape = [&](g::ValueId value, const c::TensorDesc &description) {
    return take(builder.add(take(c::View::reshape(take(c::TensorShape::create(
        description.dimensions()), "view shape")), "view"), value), "reshape");
  };
  ConsumerGraph result;
  const auto count = math.convolution ? std::size_t{2} : math.inputs.size();
  for (std::size_t port = 0; port < count; ++port)
    result.inputs[port] = take(builder.input(math.inputs[port]), "B math input");
  g::ValueId rows;
  if (!math.convolution) {
    const auto initial = reshape(gathered, dense(c::DataType::kFloat32, {kB, 2, 16, 16}));
    const std::array arguments{result.inputs[0], result.inputs[1], result.inputs[2],
        result.inputs[3], result.inputs[4], initial};
    const auto node = take(builder.add_node(c::Operation(c::GatedDeltaNet{}), arguments), "B GDN");
    result.activation = take(builder.result(node, 0), "GDN Y");
    rows = reshape(take(builder.result(node, 1), "GDN state"),
        dense(c::DataType::kFloat32, {kB, schema.item_elements()}));
  } else {
    const auto history = reshape(gathered, dense(c::DataType::kFloat32, {kB, 3, 128}));
    const auto samples = take(builder.add(c::Concat{.axis = 1}, history, result.inputs[0]), "conv samples");
    result.activation = take(builder.add(c::model_ops::causal_conv1d(), samples, result.inputs[1]), "B conv");
    const auto window = take(builder.add(take(c::View::slice(1, 1, 3), "last history"), samples), "history window");
    const auto copied = take(builder.add(c::Copy{}, window), "materialize history gap");
    rows = reshape(copied, dense(c::DataType::kFloat32, {kB, schema.item_elements()}));
  }
  result.store = take(builder.add_node(c::Operation::custom(s::Store{schema}), std::array{rows, dst}), "B Store");
  check(builder.publish(result.activation), "publish B activation");
  result.graph = take(std::move(builder).finish(), "B Graph");
  result.gather = take(result.graph.producer(gathered), "original Gather");
  return result;
}

void relay(bool convolution, bool failed_producer = false) {
  SCOPED_TRACE(convolution ? "indexed F32 conv" : "indexed F32 GDN");
  SCOPED_TRACE(failed_producer ? "checked producer failure" : "successful producer");
  auto context = take(g::cuda::create_context(), "Context");
  auto memory = take(c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice), "device memory");
  auto storage = take(c::StorageOwner::adopt(context, std::move(memory)), "storage owner");
  auto staging = take(c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kPinnedHost), "metadata staging");
  auto queue = take(context.create_queue(), "Queue");
  check(queue.reserve_completions(12), "completion capacity");
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  const auto math = make_math(context, storage.ref(), convolution);
  const auto item_bytes = take(math.item.required_bytes(), "item bytes");
  const auto stride = convolution ? item_bytes + 256 : std::size_t{4096};
  const auto schema = take(s::Schema::create(math.item, kP, stride), "slot schema");
  std::vector<float> before(kP * stride / sizeof(float));
  for (std::size_t i = 0; i < before.size(); ++i) before[i] = 48.F + static_cast<float>(i % 31) / 32.F;
  auto backing = take(m::Buffer::create(storage.ref(), before.size() * sizeof(float), 256), "backing");
  check(backing.view().write(std::as_bytes(std::span(before))), "initialize backing");
  const m::ConstBufferView diagnostic = backing.view();
  auto pool = take(c::LeasePool::adopt(std::move(backing), {.max_blocks = 2 * kP}), "slot pool");
  const auto layout = take(c::LeaseSlotLayout::create(pool, math.item, kP, stride), "slot layout");
  std::array<c::WriteLease, kP> blocks;
  std::array<c::ReadLease, kP> padding, untouched;
  for (std::size_t slot = 0; slot < kP; ++slot) {
    blocks[slot] = take(pool.allocate(item_bytes, 256), "item block");
    ASSERT_EQ(take(layout.locate(blocks[slot]), "item coordinates").index, slot);
    padding[slot] = take(take(pool.allocate(stride - item_bytes, 256), "padding block").publish(), "padding guard");
  }
  std::array<c::WriteLease, kB> a_writers, b_writers;
  for (std::size_t row = 0; row < kB; ++row) {
    a_writers[row] = std::move(blocks[kSources[row]]);
    b_writers[row] = std::move(blocks[kDestinations[row]]);
  }
  for (std::size_t slot = 0; slot < kP; ++slot)
    if (blocks[slot].valid()) untouched[slot] = take(blocks[slot].publish(), "untouched old snapshot");
  const auto row_description = dense(c::DataType::kFloat32, {kB, schema.item_elements()});
  const auto ids_description = dense(c::DataType::kInt32, {kB});
  auto warm_values = math.initial;
  for (auto &value : warm_values) value += .25F;
  auto table_storage = upload<float>(storage.ref(), warm_values);
  std::array<std::int32_t, kB> checked_ids{0, 1};
  auto checked_storage = upload<std::int32_t>(storage.ref(), checked_ids);
  auto a_builder = take(g::GraphBuilder::create(), "A GraphBuilder");
  const auto table = take(a_builder.input(row_description), "A table");
  const auto checked = take(a_builder.input(ids_description), "A checked indices");
  const auto stored_rows = take(a_builder.add(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked}, table, checked), "A checked rows");
  const auto a_ids = take(a_builder.input(ids_description), "A private destination IDs");
  const auto a_store = take(a_builder.add_node(c::Operation::custom(s::Store{schema}), std::array{stored_rows, a_ids}), "A Store");
  const auto a_graph = take(std::move(a_builder).finish(), "A Graph");
  auto a_compiled = take(g::cuda::compile(a_graph, context, storage), "compile A");
  auto producer = take(a_compiled.executable.create_invocation(), "A invocation");
  check(producer.bind(table, take(c::ConstTensorView::create(row_description, table_storage.view()), "A table view")), "bind A table");
  check(producer.bind(checked, take(c::ConstTensorView::create(ids_description, checked_storage.view()), "A checked view")), "bind A checked IDs");
  check(s::prepare(producer, queues, {.host_workspace = staging.ref()}), "prepare A controls");
  auto a_transaction = take(s::SlotBindings::create(layout, {}, a_writers), "warm A transaction");
  check(s::bind(producer, a_graph, a_store, std::move(a_transaction)), "bind warm A");
  auto source = take(gd::StoreSource::prepare(producer, a_store, storage), "warm A Source metadata");
  check(producer.prepare_replay(queues, {.mode = g::ReplayMode::kCommands}), "prepare warm A");
  auto warm_submitted = take(producer.submit(queues), "submit warm A");
  auto common = take(gd::SubmissionSource::create(producer, std::move(warm_submitted), storage), "warm A terminal");
  check(source.capture(common), "capture warm A Store");
  check(common.wait(), "retire warm A"); check(common.detach(), "detach warm A");
  auto returned_a = take(s::take(producer, a_store), "take warm A");
  std::array<c::ReadLease, kB> warm_ready;
  std::array<std::uint64_t, kB> a_generations;
  for (std::size_t row = 0; row < kB; ++row) {
    warm_ready[row] = take(source.publish(row, returned_a.destinations()[row]), "publish actual warm A");
    a_generations[row] = warm_ready[row].generation();
    std::vector<float> actual(item_bytes / sizeof(float));
    check(take(warm_ready[row].view(), "warm A view").read(std::as_writable_bytes(std::span(actual))), "read warm A");
    expect_bits(actual, std::span(warm_values).subspan(row * actual.size(), actual.size()));
  }

  const auto graph = make_consumer(math, schema);
  auto b_compiled = take(g::cuda::compile(graph.graph, context, storage), "compile B");
  ASSERT_EQ(b_compiled.report.step_count(), 1u);
  const auto step = take(b_compiled.report.step(0), "B selected fusion");
  EXPECT_EQ(step.policy, (c::RoutePolicyId{0x48594c4547474355ULL, convolution ? 7u : 6u, 1}));
  EXPECT_EQ(step.implementation, c::ProviderImplementationId{convolution ? 8u : 7u});
  EXPECT_EQ(step.dispatch_count, 1u);
  auto consumer = take(b_compiled.executable.create_invocation(), "B invocation");
  std::array<m::Buffer, 5> constants;
  const auto input_count = convolution ? std::size_t{2} : constants.size();
  for (std::size_t port = 0; port < input_count; ++port) {
    constants[port] = upload<float>(storage.ref(), math.data[port]);
    check(consumer.bind(graph.inputs[port], take(c::ConstTensorView::create(math.inputs[port], constants[port].view()), "B input view")), "bind B input");
  }
  auto output = output_buffer(storage.ref(), math.output);
  check(consumer.bind(graph.activation, take(c::TensorView::create(math.output, output.view()), "B output view")), "bind B output");
  const std::array association{s::SourceBinding{graph.store, graph.gather, s::SourceMode::kImported}};
  check(s::prepare(consumer, graph.graph, queues, association, {.host_workspace = staging.ref()}), "prepare imported B");
  {
    auto wrong_ready = warm_ready;
    wrong_ready.front() = untouched.front(); // Same pool and exact row size, different real root/slot.
    auto rejected = source.reader_binding(consumer, graph.graph, graph.store, wrong_ready);
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code(), hyle::StatusCode::InvalidArgument);
  }
  auto warm_binding = take(source.reader_binding(consumer, graph.graph, graph.store, warm_ready), "true-ready B descriptor");
  const std::array warm_bindings{warm_binding};
  check(gd::bind_outputs(consumer, warm_bindings), "bind true-ready B");
  auto b_transaction = take(s::SlotBindings::create(layout, {}, b_writers), "warm B destination-only transaction");
  check(s::bind(consumer, graph.graph, graph.store, std::move(b_transaction)), "bind warm B destinations");
  check(gd::prepare_outputs(consumer, warm_bindings, queues), "record actual ready B");
  ASSERT_EQ(consumer.replay_route(), g::ReplayRoute::kCaptured);
  EXPECT_EQ(consumer.replay_report()->recorded_capture_range_count, 1u);
  auto warm_done = take(gd::submit_outputs(consumer, warm_bindings, queues), "run warm B");
  check(warm_done.wait(), "retire warm B"); warm_done = {};
  EXPECT_FALSE(gd::release_outputs(consumer, warm_bindings).ok()); // Original destination owners must be taken first.
  auto returned_b = take(s::take(consumer, graph.store), "take warm imported B");
  ASSERT_EQ(returned_b.source_count(), 0u);
  for (std::size_t row = 0; row < kB; ++row) b_writers[row] = std::move(returned_b.destinations()[row]);
  check(gd::release_outputs(consumer, warm_bindings), "release only retired warm bundle");
  source = {}; common = {}; returned_a = {}; returned_b = {};
  // Only this last owning ready handle remains. Reclaim the exact source
  // generations; both original physical ID carriers remain in their owners.
  for (std::size_t row = 0; row < kB; ++row) {
    a_writers[row] = take(warm_ready[row].try_write(), "reclaim original A slot");
    EXPECT_EQ(a_writers[row].generation(), a_generations[row]);
  }
  check(table_storage.view().write(std::as_bytes(std::span(math.initial))), "actual A rows differ from warm rows");
  if (failed_producer) {
    checked_ids.back() = static_cast<std::int32_t>(kB);
    check(checked_storage.view().write(std::as_bytes(std::span(checked_ids))), "actual checked failure input");
  }
  auto actual_a = take(s::SlotBindings::create(layout, {}, a_writers), "actual A transaction");
  check(s::bind(producer, a_graph, a_store, std::move(actual_a)), "rebind actual A same controls");
  source = take(gd::StoreSource::prepare(producer, a_store, storage), "actual A Source metadata");
  check(producer.prepare_replay(queues, {.mode = g::ReplayMode::kCommands}), "prepare actual A");

  Gate gate, reader_gate;
  gate.arm(take(c::cuda::stream(queue), "original Queue stream"));
  ASSERT_EQ(cudaLaunchHostFunc(gate.stream, &Gate::wait, &gate), cudaSuccess);
  auto submitted = take(producer.submit(queues), "submit actual A");
  common = take(gd::SubmissionSource::create(producer, std::move(submitted), storage), "actual A terminal");
  check(source.capture(common), "capture pending A Store");
  EXPECT_EQ(take(common.poll(), "pending A"), c::CompletionPoll::kPending);
  auto binding = take(source.reader_binding(consumer, graph.graph, graph.store), "pending B descriptor");
  const std::array bindings{binding};
  check(gd::bind_outputs(consumer, bindings), "bind original pending B source");
  auto actual_b = take(s::SlotBindings::create(layout, {}, b_writers), "actual B destination-only transaction");
  check(s::bind(consumer, graph.graph, graph.store, std::move(actual_b)), "bind actual B same destination controls");
  check(gd::prepare_outputs(consumer, bindings, queues), "reuse exact native B capture");
  EXPECT_EQ(consumer.replay_route(), g::ReplayRoute::kCaptured);
  EXPECT_EQ(consumer.replay_report()->recorded_capture_range_count, 0u);
  EXPECT_EQ(consumer.replay_report()->reused_capture_range_count, 1u);
  const auto dependencies = take(consumer.input_source_report(), "B dependency report");
  EXPECT_EQ(dependencies.source_count, 1u); EXPECT_EQ(dependencies.device_wait_count, 1u);
  EXPECT_EQ(dependencies.host_wait_count, 0u);
  EXPECT_FALSE(consumer.submit(queues));
  reader_gate.arm(gate.stream);
  ASSERT_EQ(cudaLaunchHostFunc(reader_gate.stream, &Gate::wait, &reader_gate), cudaSuccess);
  auto done = take(gd::submit_outputs(consumer, bindings, queues), "submit actual B");
  gate.release();
  const auto a_terminal = common.wait();
  EXPECT_EQ(take(done.poll(), "B still pending"), c::CompletionPoll::kPending);
  EXPECT_FALSE(gd::release_outputs(consumer, bindings).ok()); // Pending work still owns the complete source bundle.
  std::array<c::ReadLease, kB> published_a;
  if (!failed_producer) {
    ASSERT_TRUE(a_terminal.ok()) << a_terminal.message();
    check(common.detach(), "detach retired actual A");
    auto actual_returned = take(s::take(producer, a_store), "take actual A while B pending");
    for (std::size_t row = 0; row < kB; ++row) {
      EXPECT_FALSE(actual_returned.destinations()[row].publish());
      published_a[row] = take(source.publish(row, actual_returned.destinations()[row]), "publish actual A while B reads");
      EXPECT_EQ(published_a[row].generation(), a_generations[row]);
      EXPECT_FALSE(published_a[row].try_write());
    }
  } else {
    EXPECT_FALSE(a_terminal.ok());
    c::WriteLease absent;
    auto rejected = source.publish(0, absent);
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code(), a_terminal.code());
    EXPECT_EQ(rejected.error().message(), a_terminal.message());
  }
  reader_gate.release();
  const auto b_terminal = done.wait();
  done = {};
  if (failed_producer) {
    EXPECT_EQ(b_terminal.code(), a_terminal.code());
    EXPECT_EQ(b_terminal.message(), a_terminal.message());
    EXPECT_FALSE(consumer.output(graph.activation));
    auto failed = take(s::take(consumer, graph.store), "take failed imported B after actual retirement");
    ASSERT_EQ(failed.source_count(), 0u);
    ASSERT_EQ(failed.destination_count(), kB);
    check(gd::release_outputs(consumer, bindings), "release failed retired B bundle without publishing state");
  } else {
    ASSERT_TRUE(b_terminal.ok()) << b_terminal.message();
    expect_bits(half_test::read<float>(output), math.expected_output);
    auto final = take(s::take(consumer, graph.store), "take actual B");
    ASSERT_EQ(final.source_count(), 0u);
    ASSERT_EQ(final.destination_count(), kB);
    auto expected = before;
    std::array<c::ReadLease, kB> state;
    for (std::size_t row = 0; row < kB; ++row) {
      state[row] = take(final.destinations()[row].publish(), "publish successful B state");
      std::vector<float> actual(item_bytes / sizeof(float));
      check(take(state[row].view(), "B state view").read(std::as_writable_bytes(std::span(actual))), "read B state");
      const auto wanted = std::span(math.expected_state).subspan(row * actual.size(), actual.size());
      expect_bits(actual, wanted);
      std::copy(wanted.begin(), wanted.end(), expected.begin() + kDestinations[row] * stride / sizeof(float));
      const auto initial = std::span(math.initial).subspan(row * actual.size(), actual.size());
      std::copy(initial.begin(), initial.end(), expected.begin() + kSources[row] * stride / sizeof(float));
    }
    std::vector<float> actual(before.size());
    check(diagnostic.read(std::as_writable_bytes(std::span(actual))), "whole bank after actual retirement");
    expect_bits(actual, expected);
  }
  // A mathematical failure may have touched its nominated destinations, but
  // it grants neither publication nor permission to write an old snapshot/gap.
  for (std::size_t slot = 0; slot < kP; ++slot) {
    const auto count = (stride - item_bytes) / sizeof(float);
    std::vector<float> actual(count);
    check(take(padding[slot].view(), "padding view").read(std::as_writable_bytes(std::span(actual))), "read padding");
    expect_bits(actual, std::span(before).subspan((slot * stride + item_bytes) / sizeof(float), count));
    if (!untouched[slot].valid()) continue;
    actual.resize(item_bytes / sizeof(float));
    check(take(untouched[slot].view(), "old snapshot view").read(std::as_writable_bytes(std::span(actual))), "read old snapshot");
    expect_bits(actual, std::span(before).subspan(slot * stride / sizeof(float), actual.size()));
  }
  EXPECT_FALSE(gate.timed_out.load()); EXPECT_FALSE(reader_gate.timed_out.load());
}

TEST(GgmlCudaPendingIndexedState, GdnUsesPublishedWarmAndPendingOriginalSlots) { relay(false); }
TEST(GgmlCudaPendingIndexedState, ConvUsesPublishedWarmAndPendingOriginalHistory) { relay(true); }
TEST(GgmlCudaPendingIndexedState, CheckedProducerFailureNeverPublishesConvResult) { relay(true, true); }
} // namespace

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <source_location>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hyle/compute/sequence_ops.h"
#include "hyle/graph/cuda.h"
#include "hyle/graph/providers/ggml_cuda.h"
#include "hyle/graph/slots.h"
#include "../compute_models/gdn_snapshot_fixture.h"
#include "../compute_provider_ggml_cuda/tensor_test_support.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace s = g::slots;
namespace m = hyle::mem;
using ggml_cuda_tensor_test::dense;
using ggml_cuda_tensor_test::upload;
constexpr std::int64_t rows = 128, key_heads = 2, value_heads = 4;
constexpr std::uint32_t slots = 7, source_slot = 5, destination_slot = 2;
constexpr c::RoutePolicyId indexed_policy{0x48594c4547474355ULL, 6, 1};
enum class Escape { kNone, kGather, kState };
thread_local const char *current_stage = "environment setup";
struct Stage final {
  const char *previous;
  explicit Stage(const char *name) : previous(std::exchange(current_stage, name)) {}
  ~Stage() { current_stage = previous; }
};
std::string failure(const hyle::Status &status, const char *what, std::source_location caller) {
  return std::string(current_stage) + ": " + what + " at " + caller.file_name() + ":" +
      std::to_string(caller.line()) + ": " + std::string(status.message());
}
template<class T> T take(hyle::Result<T> value, const char *what = "take Result",
    std::source_location caller = std::source_location::current()) {
  if (!value) throw std::runtime_error(failure(value.error(), what, caller));
  return std::move(*value);
}
void check(hyle::Status status, const char *what = "check Status",
    std::source_location caller = std::source_location::current()) {
  if (!status.ok()) throw std::runtime_error(failure(status, what, caller));
}
void expect_bits(std::span<const float> actual, std::span<const float> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i)
    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]), std::bit_cast<std::uint32_t>(expected[i])) << i;
}
std::vector<std::byte> read(m::ConstBufferView view) {
  std::vector<std::byte> result(view.size()); check(view.read(result), "read original diagnostic byte view"); return result;
}

struct Environment final {
  // Graph discovery and ordinary references use this provider exclusively.
  c::Context context = take(c::Context::create(take(c::cuda::device(0)), g::ggml_cuda::provider()));
  c::cuda::MemoryResource memory = take(c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice));
  c::cuda::MemoryResource staging = take(c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kPinnedHost));
  c::Queue queue = take(context.create_queue());
};

struct Case final {
  Environment &environment;
  const Escape escape;
  gdn_snapshot_fixture::Fixture fixture;
  const std::size_t elements, item_bytes, stride;
  std::array<c::TensorDesc, 7> math_descriptions;
  std::array<std::vector<float>, 5> payload;
  std::vector<std::byte> expected_pool;
  m::ConstBufferView diagnostic;
  c::LeasePool pool;
  c::LeaseSlotLayout layout;
  std::array<c::ReadLease, slots> gaps, unused;
  std::array<c::ReadLease, 1> sources;
  std::array<c::WriteLease, 1> writers;
  std::array<m::Buffer, 6> input_storage;
  m::Buffer y_storage, escaped_storage;
  g::Graph graph;
  g::CompiledGraph compiled;
  std::array<g::ValueId, 6> inputs;
  g::ValueId y, escaped;
  g::NodeId gather, store;
  // Bindings and Invocation die before all original payload/lease owners.
  s::SlotBindings transaction;
  g::Invocation run;

  Case(Environment &selected, std::int64_t depth, Escape selected_escape = Escape::kNone, bool special_empty = false)
      : environment(selected), escape(selected_escape), fixture(9, 1, rows, key_heads, value_heads, depth),
        elements(std::size_t(value_heads * depth * depth)), item_bytes(elements * sizeof(float)), stride(item_bytes + 256) {
    Stage stage("Case constructor");
    if (special_empty) {
      fixture.data[5][0] = std::bit_cast<float>(std::uint32_t{0x80000000});
      fixture.data[5][1] = std::bit_cast<float>(std::uint32_t{0x7fc12345});
    }
    const auto f32 = c::DataType::kFloat32;
    math_descriptions = {dense(f32, {rows, key_heads, depth}), dense(f32, {rows, key_heads, depth}),
        dense(f32, {rows, value_heads, depth}), dense(f32, {rows, value_heads, 1}),
        dense(f32, {rows, value_heads, 1}), fixture.inputs[5], dense(c::DataType::kInt32, {2})};
    const auto item = dense(f32, {value_heads, depth, depth});
    const auto schema = take(s::Schema::create(item, slots, stride));
    expected_pool.assign(slots * stride, std::byte{0xa5});
    const auto initial = std::as_bytes(std::span(fixture.data[5]));
    std::copy(initial.begin(), initial.end(), expected_pool.begin() + source_slot * stride);
    auto backing = take(m::Buffer::create(environment.memory.ref(), expected_pool.size(), 256));
    check(backing.view().write(expected_pool));
    // Post-retirement diagnostic alias only. It is never bound to a Graph or
    // manufactured into a whole-pool ready/mutable permission.
    diagnostic = backing.view();
    pool = take(c::LeasePool::adopt(std::move(backing), {.max_blocks = 2 * slots}));
    layout = take(c::LeaseSlotLayout::create(pool, item, slots, stride));
    std::array<c::WriteLease, slots> blocks;
    for (std::size_t slot = 0; slot < slots; ++slot) {
      blocks[slot] = take(pool.allocate(item_bytes, 256));
      EXPECT_EQ(take(layout.locate(blocks[slot])).index, slot);
      gaps[slot] = take(take(pool.allocate(256, 256)).publish());
    }
    sources[0] = take(blocks[source_slot].publish());
    writers[0] = std::move(blocks[destination_slot]);
    for (std::size_t slot = 0; slot < slots; ++slot)
      if (blocks[slot]) unused[slot] = take(blocks[slot].publish());

    auto builder = take(g::GraphBuilder::create());
    const auto bank = take(builder.input(take(schema.capacity_description())));
    const auto source_ids = take(builder.input(dense(c::DataType::kInt32, {1})));
    const auto destination_ids = take(builder.input(dense(c::DataType::kInt32, {1})));
    gather = take(builder.add_node(c::Operation(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked}),
        std::array{bank, source_ids}));
    const auto gathered = take(builder.result(gather));
    const auto reshape = [&](g::ValueId value, std::initializer_list<std::int64_t> dimensions) {
      return take(builder.add(take(c::View::reshape(take(c::TensorShape::create(dimensions)))), value));
    };
    const auto initial_state = reshape(gathered, {1, value_heads, depth, depth});
    std::array<g::ValueId, 7> arguments;
    for (std::size_t port = 0; port < 5; ++port) arguments[port] = inputs[port] = take(builder.input(math_descriptions[port]));
    arguments[5] = initial_state;
    arguments[6] = inputs[5] = take(builder.input(math_descriptions[6]));
    const auto recurrence = take(builder.add_node(c::sequence_ops::packed_gated_delta_net(), arguments));
    y = take(builder.result(recurrence, 0));
    const auto state = take(builder.result(recurrence, 1));
    const auto updates = reshape(state, {1, static_cast<std::int64_t>(elements)});
    store = take(builder.add_node(c::Operation::custom(s::Store{schema}), std::array{updates, destination_ids}));
    check(builder.publish(y));
    if (escape != Escape::kNone) {
      escaped = escape == Escape::kGather ? gathered : state;
      check(builder.publish(escaped));
    }
    graph = take(std::move(builder).finish());
    compiled = take(g::cuda::compile(graph, environment.context, environment.memory.ref()));
    std::size_t indexed = 0;
    for (std::size_t step = 0; step < compiled.report.step_count(); ++step)
      indexed += take(compiled.report.step(step)).policy == indexed_policy;
    if (escape == Escape::kNone) {
      EXPECT_EQ(indexed, 1U);
      EXPECT_EQ(compiled.report.step_count(), 1U);
      const auto step = take(compiled.report.step(0));
      EXPECT_EQ(step.provider.id, c::ggml_cuda::kProviderId);
      EXPECT_EQ(step.kind, g::CompileStepKind::kProviderRegion);
      EXPECT_EQ(step.implementation, c::ProviderImplementationId{7});
      EXPECT_EQ(step.node_count, 5U);
    } else EXPECT_EQ(indexed, 0U) << "private state/gather escape must block indexed fusion";
    for (std::size_t port = 0; port < 5; ++port)
      input_storage[port] = take(m::Buffer::create(environment.memory.ref(), math_descriptions[port].required_bytes().value(), 256));
    input_storage[5] = take(m::Buffer::create(environment.memory.ref(), 2 * sizeof(std::int32_t), 256));
    y_storage = take(m::Buffer::create(environment.memory.ref(), math_descriptions[2].required_bytes().value() + 32, 256));
    check(y_storage.view().fill(std::byte{0xa5}));
    if (escaped) {
      escaped_storage = take(m::Buffer::create(environment.memory.ref(), item_bytes + 32, 256));
      check(escaped_storage.view().fill(std::byte{0xa5}));
    }
    transaction = take(s::SlotBindings::create(layout, sources, writers));
    fresh();
  }

  void fresh() {
    Stage stage("Case fresh Invocation");
    run = take(compiled.executable.create_invocation(), "create original Graph Invocation");
    for (std::size_t port = 0; port < 5; ++port)
      check(run.bind(inputs[port], take(c::ConstTensorView::create(math_descriptions[port], input_storage[port].view()))));
    check(run.bind(inputs[5], take(c::ConstTensorView::create(math_descriptions[6], input_storage[5].view()))));
    check(run.bind(y, take(c::TensorView::create(math_descriptions[2],
        take(y_storage.view().subview(0, math_descriptions[2].required_bytes().value()))))));
    if (escaped) check(run.bind(escaped, take(c::TensorView::create(take(graph.value_description(escaped)),
        take(escaped_storage.view().subview(0, item_bytes))))));
    const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &environment.queue}};
    const std::array associations{s::SourceBinding{store, gather}};
    check(s::prepare(run, graph, queues, associations, {.host_workspace = environment.staging.ref()}), "prepare original Store association");
  }
  void upload(std::size_t active) {
    Stage stage("Case upload inputs");
    for (std::size_t port = 0; port < 5; ++port) {
      payload[port] = fixture.data[port];
      std::fill(payload[port].begin() + active * payload[port].size() / std::size_t(rows), payload[port].end(),
          std::bit_cast<float>(std::uint32_t{0x7fc54321}));
      check(input_storage[port].view().write(std::as_bytes(std::span(payload[port]))));
    }
    offsets({0, static_cast<std::int32_t>(active)});
  }
  void offsets(std::array<std::int32_t, 2> values) {
    Stage stage("Case upload offsets");
    check(input_storage[5].view().write(std::as_bytes(std::span(values))), "write original offsets buffer");
  }
  void begin(g::ReplayMode mode = g::ReplayMode::kCommands) {
    Stage stage("Case begin Store episode");
    check(take(transaction.destinations()[0].view()).fill(std::byte{0x3d}));
    std::fill_n(expected_pool.begin() + destination_slot * stride, item_bytes, std::byte{0x3d});
    check(take(y_storage.view().subview(0, math_descriptions[2].required_bytes().value())).fill(std::byte{0x5a}));
    if (escaped) check(take(escaped_storage.view().subview(0, item_bytes)).fill(std::byte{0x5a}));
    EXPECT_EQ(transaction.source_addresses()[0].index, source_slot);
    EXPECT_EQ(transaction.destination_addresses()[0].index, destination_slot);
    check(s::bind(run, graph, store, std::move(transaction)), "bind original Store transaction");
    const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &environment.queue}};
    check(run.prepare_replay(queues, {.mode = mode, .host_wait = g::HostWaitPolicy::kReject}), "prepare original Graph replay");
  }
  hyle::Status submit() {
    Stage stage("Case submit and recover Store");
    const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &environment.queue}};
    auto done = take(run.submit(queues), "submit original Graph");
    const auto status = done.wait();
    std::cerr << "indexed_packed_terminal escape=" << static_cast<int>(escape)
        << " ok=" << status.ok() << " invalid_offsets=" << status.IsInvalidArgument()
        << " message=" << status.message() << '\n';
    done = {}; // Release the original observation only after actual wait.
    transaction = take(s::take(run, store), "take original Store after waited Completion and released observation");
    return status;
  }
  void refill() {
    Stage stage("Case refill retired Store");
    writers[0] = std::move(transaction.destinations()[0]); transaction.clear();
    check(transaction.refill(sources, writers), "refill original returned transaction");
  }
  void discard_failed_writer(const hyle::Status &terminal) {
    Stage stage("Case discard retired failed writer and acquire a fresh generation");
    const auto old_generation = transaction.destinations()[0].generation();
    const auto denied = transaction.destinations()[0].view();
    EXPECT_FALSE(denied) << "an admitted failed writer must remain unusable";
    if (!denied) {
      EXPECT_EQ(denied.error().code(), terminal.code());
      EXPECT_EQ(denied.error().message(), terminal.message());
    }
    // Completion wait and original Store take have already retired this use.
    // Failed payload generations cannot be refilled, published or relabelled.
    // Retain every other original slot/gap; only this released slot is free.
    transaction.clear();
    writers[0] = take(pool.allocate(item_bytes, 256), "allocate fresh original-pool destination after failed generation retirement");
    EXPECT_NE(writers[0].generation(), old_generation);
    EXPECT_EQ(take(layout.locate(writers[0]), "locate fresh original-pool writer").index, destination_slot);
    check(transaction.refill(sources, writers), "refill transaction with original readers and fresh destination generation");
  }
  void expect_poison() const {
    Stage stage("Case inspect unchanged failed outputs and pool");
    EXPECT_EQ(read(take(y_storage.view().subview(0, math_descriptions[2].required_bytes().value()))),
        std::vector<std::byte>(math_descriptions[2].required_bytes().value(), std::byte{0x5a}));
    if (escape == Escape::kState) EXPECT_EQ(read(take(escaped_storage.view().subview(0, item_bytes))),
        std::vector<std::byte>(item_bytes, std::byte{0x5a}));
    EXPECT_EQ(read(diagnostic), expected_pool);
    expect_guards();
  }
  void expect_guards() const {
    Stage stage("Case inspect output guards");
    EXPECT_EQ(read(take(y_storage.view().subview(math_descriptions[2].required_bytes().value(), 32))),
        std::vector<std::byte>(32, std::byte{0xa5}));
    if (escaped) EXPECT_EQ(read(take(escaped_storage.view().subview(item_bytes, 32))),
        std::vector<std::byte>(32, std::byte{0xa5}));
  }
};

// Original public PackedGDN with an independently owned dense initial state;
// no indexed Gather/Store pattern can enter this mathematical reference.
std::array<std::vector<float>, 2> reference(Case &test, std::size_t active) {
  Stage stage("standalone PackedGDN reference");
  auto &environment = test.environment;
  std::array<m::Buffer, 7> storage;
  std::array<c::ConstTensorView, 7> inputs;
  for (std::size_t port = 0; port < 7; ++port) {
    storage[port] = take(m::Buffer::create(environment.memory.ref(), test.math_descriptions[port].required_bytes().value(), 256));
    if (port < 5) check(storage[port].view().write(std::as_bytes(std::span(test.payload[port]))));
    else if (port == 5) check(storage[port].view().write(std::as_bytes(std::span(test.fixture.data[5]))));
    else {
      const std::array<std::int32_t, 2> offsets{0, static_cast<std::int32_t>(active)};
      check(storage[port].view().write(std::as_bytes(std::span(offsets))));
    }
    inputs[port] = take(c::ConstTensorView::create(test.math_descriptions[port], storage[port].view()));
  }
  const std::array descriptions{test.math_descriptions[2], test.math_descriptions[5]};
  std::array<m::Buffer, 2> output_storage;
  std::array<c::TensorView, 2> outputs;
  for (std::size_t port = 0; port < 2; ++port) {
    output_storage[port] = take(m::Buffer::create(environment.memory.ref(), descriptions[port].required_bytes().value(), 256));
    outputs[port] = take(c::TensorView::create(descriptions[port], output_storage[port].view()));
  }
  auto prepared = take(c::prepare(environment.context, c::sequence_ops::packed_gated_delta_net(), inputs, outputs));
  EXPECT_EQ(prepared.provider().id, c::ggml_cuda::kProviderId);
  auto workspace = ggml_cuda_test::workspace_buffer(prepared, environment.memory.ref());
  auto invocation = take(prepared.create_invocation(workspace.view()));
  auto done = take(invocation.submit(environment.queue, inputs, outputs), "submit standalone PackedGDN reference");
  check(done.wait(), "wait standalone PackedGDN reference");
  return {half_test::read<float>(output_storage[0]), half_test::read<float>(output_storage[1])};
}

void expect_result(Case &test, std::size_t active) {
  Stage stage("Case compare original mathematical outputs");
  const auto expected = reference(test, active);
  const auto y_bytes = take(test.y_storage.view().subview(0, test.math_descriptions[2].required_bytes().value()));
  std::vector<float> actual_y(expected[0].size()); check(y_bytes.read(std::as_writable_bytes(std::span(actual_y))));
  expect_bits(actual_y, expected[0]);
  for (std::size_t i = active * std::size_t(value_heads * test.fixture.depth); i < actual_y.size(); ++i)
    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual_y[i]), 0U) << "inactive row element " << i;
  std::vector<float> actual_state(test.elements);
  check(take(test.transaction.destinations()[0].view()).read(std::as_writable_bytes(std::span(actual_state))));
  expect_bits(actual_state, expected[1]);
  const auto state_bytes = std::as_bytes(std::span(expected[1]));
  std::copy(state_bytes.begin(), state_bytes.end(), test.expected_pool.begin() + destination_slot * test.stride);
  EXPECT_EQ(read(test.diagnostic), test.expected_pool);
  if (test.escaped) {
    std::vector<float> actual(test.elements);
    check(take(test.escaped_storage.view().subview(0, test.item_bytes)).read(std::as_writable_bytes(std::span(actual))));
    expect_bits(actual, test.escape == Escape::kGather ? test.fixture.data[5] : expected[1]);
  }
  test.expect_guards();
}

TEST(GgmlCudaIndexedPackedState, ActualLengthsAndPoolStridesMatchOriginalPackedBits) {
  Environment environment;
  for (const auto depth : {16, 128}) for (const auto active : {0u, 1u, 105u, 128u}) {
    SCOPED_TRACE(depth);
    SCOPED_TRACE(active);
    Case test(environment, depth, Escape::kNone, active == 0);
    test.upload(active); test.begin();
    const auto status = test.submit(); ASSERT_TRUE(status.ok()) << status.message();
    expect_result(test, active);
  }
}

TEST(GgmlCudaIndexedPackedState, EscapingGatherOrStatePreventsIndexedFusion) {
  Environment environment;
  for (const auto escape : {Escape::kGather, Escape::kState}) {
    SCOPED_TRACE(static_cast<int>(escape));
    Case test(environment, 16, escape); test.upload(105); test.begin();
    const auto status = test.submit(); ASSERT_TRUE(status.ok()) << status.message();
    expect_result(test, 105);
  }
}

TEST(GgmlCudaIndexedPackedState, InvalidOffsetsPreserveOutputsAndPoolUntilFreshRetry) {
  Environment environment;
  for (const auto escape : {Escape::kNone, Escape::kState}) {
    Case test(environment, 16, escape);
    for (const auto offsets : {std::array<std::int32_t, 2>{1, 105}, std::array<std::int32_t, 2>{-1, 105},
                               std::array<std::int32_t, 2>{0, -1}, std::array<std::int32_t, 2>{0, 129}}) {
      SCOPED_TRACE(::testing::Message() << "escape=" << static_cast<int>(escape)
          << " offsets=[" << offsets[0] << ',' << offsets[1] << "]");
      try {
        test.upload(105); test.offsets(offsets); test.begin();
        const auto status = test.submit(); EXPECT_TRUE(status.IsInvalidArgument()) << status.message();
        test.expect_poison();
        EXPECT_FALSE(test.run.submit(environment.queue));
        test.discard_failed_writer(status); test.fresh(); test.upload(105); test.begin();
        const auto recovered = test.submit(); ASSERT_TRUE(recovered.ok()) << recovered.message();
        expect_result(test, 105); test.refill();
      } catch (const std::exception &error) {
        FAIL() << "invalid-offset episode or fresh retry failed: " << error.what();
      }
    }
  }
}

TEST(GgmlCudaIndexedPackedState, CapturedLengthsReuseOriginalTupleAndWithdrawalKeepsPermissionsClosed) {
  Environment environment;
  Case test(environment, 128);
  std::size_t round = 0;
  for (const auto active : {128u, 105u, 1u, 0u, 105u, 128u}) {
    SCOPED_TRACE(active);
    test.upload(active); test.begin(g::ReplayMode::kRequireCapture);
    const auto replay = take(test.run.replay_report());
    EXPECT_EQ(replay.route, g::ReplayRoute::kCaptured);
    if (round++) {
      EXPECT_EQ(replay.recorded_capture_range_count, 0U);
      EXPECT_GE(replay.reused_capture_range_count, 1U);
    }
    const auto status = test.submit(); ASSERT_TRUE(status.ok()) << status.message();
    expect_result(test, active); test.refill();
  }
  test.upload(105); test.begin(g::ReplayMode::kRequireCapture);
  check(test.run.reset());
  test.transaction = take(s::take(test.run, test.store));
  EXPECT_FALSE(test.run.submit(environment.queue));
  test.expect_poison();
  EXPECT_EQ(test.transaction.source_count(), 1U);
  EXPECT_EQ(test.transaction.destination_count(), 1U);
}
} // namespace

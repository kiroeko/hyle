#include "hyle/compute/apple.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <vector>
#include <thread>
#include <optional>

#include <gtest/gtest.h>
#include <mlx/ops.h>

#include "../../src/compute_route_apple/apple_carrier.h"
#include "../../src/compute_route_apple/apple_copy_internal.h"
#include "../../src/compute_route_apple/apple_runtime_internal.h"
#include "hyle/mem/quota_allocator.h"
#include "hyle/compute/detail/apple_plan.h"
#include "hyle/compute/model_ops.h"
#include "recurrent_fixture.h"

namespace {
namespace a = hyle::compute::apple;
namespace ad = a::detail;
namespace c = hyle::compute;
namespace mx = mlx::core;

// A test-only gate in the original FIFO, with no extra thread or GPU work.
// Every assertion exit releases and joins it before its stack storage dies.
class FifoGate final : public ad::Job {
public:
  FifoGate() {
    execute = [](ad::Job *job) noexcept {
      auto &gate = *static_cast<FifoGate *>(job);
      std::unique_lock lock(gate.mutex_);
      gate.entered_ = true;
      gate.changed_.notify_all();
      gate.changed_.wait(lock, [&] { return gate.open_; });
      gate.finished_ = true;
      gate.changed_.notify_all();
    };
  }
  ~FifoGate() { open_and_join(); }
  hyle::Status enqueue(const std::shared_ptr<ad::Runtime> &runtime) noexcept {
    auto status = runtime->enqueue(*this);
    admitted_ = status.ok();
    return status;
  }
  void wait_entered() {
    std::unique_lock lock(mutex_);
    changed_.wait(lock, [&] { return entered_; });
  }
  void open_and_join() {
    if (!admitted_) return;
    std::unique_lock lock(mutex_);
    open_ = true;
    changed_.notify_all();
    changed_.wait(lock, [&] { return finished_; });
    admitted_ = false;
  }
private:
  std::mutex mutex_;
  std::condition_variable changed_;
  bool admitted_ = false, entered_ = false, open_ = false, finished_ = false;
};

void exercise_native_signal_source(bool close_pending) {
  auto context = a::Context::create();
  ASSERT_TRUE(context) << context.error().message();
  apple_test::Recurrent fixture(3, 2, 64);
  const auto expected = fixture.oracle();
  auto first = context->prepare(c::GatedDeltaNet{}, fixture.specs());
  ASSERT_TRUE(first) << first.error().message();
  const std::array specs{a::InputSpec{fixture.inputs.front().shape(), mx::float32}};
  auto second = ad::ContextAccess::compile(ad::ContextAccess::runtime(*context),
      [](const auto &args, mx::Stream stream) {
        return std::vector<mx::array>{mx::multiply(args[0], mx::array(2.f), stream)};
      }, specs);
  ASSERT_TRUE(second) << second.error().message();
  auto producer = first->create_invocation(), consumer = second->create_invocation();
  ASSERT_TRUE(producer); ASSERT_TRUE(consumer);
  auto source = ad::InvocationAccess::source(*producer, 0);
  ASSERT_TRUE(source);
  std::array<ad::InputBinding, 1> late;
  late[0].source = std::move(*source);
  ASSERT_TRUE(ad::InvocationAccess::prepare_inputs(*consumer, late).ok());
  late[0].source.reset();
  ASSERT_TRUE(producer->bind(fixture.inputs).ok());
  auto runtime = ad::ContextAccess::runtime(*context);
  for (int repeat = 0; repeat < (close_pending ? 1 : 3); ++repeat) {
    std::optional<FifoGate> gate;
    if (close_pending) {
      gate.emplace();
      ASSERT_TRUE(gate->enqueue(runtime).ok());
      gate->wait_entered();
    }
    auto produced = producer->submit(), consumed = consumer->submit();
    ASSERT_TRUE(produced) << produced.error().message();
    ASSERT_TRUE(consumed) << consumed.error().message();
    if (close_pending) {
      hyle::Status closed;
      std::thread closer([&] { closed = context->close(); });
      // The original owner is gated. Close must reject new public admission
      // and still accept the actual GPU tail of these already queued Jobs.
      for (;;) {
        bool closing;
        { std::lock_guard lock(runtime->queue_mutex); closing = !runtime->accepting; }
        if (closing) break;
        std::this_thread::yield();
      }
      gate->open_and_join();
      closer.join();
      ASSERT_TRUE(closed.ok()) << closed.message();
    }
    ASSERT_TRUE(consumed->completion().wait().ok());
    ASSERT_TRUE(produced->completion().wait().ok());
    auto output = ad::InvocationAccess::output(*consumer, 0);
    ASSERT_TRUE(output) << output.error().message();
    ASSERT_EQ(output->size(), expected[0].size());
    for (std::size_t index = 0; index < expected[0].size(); ++index)
      EXPECT_NEAR(output->data<float>()[index], 2 * expected[0][index], 1.e-5);
    {
      std::lock_guard lock(runtime->queue_mutex);
      EXPECT_GT(runtime->native_tail_peak, 0u);
      EXPECT_EQ(runtime->native_tails, 0u);
    }
  }
}

TEST(AppleLateInput, NativeSignalFeedsMlxAndReusesTheOriginalProducer) {
  exercise_native_signal_source(false);
}
TEST(AppleLateInput, CloseDrainsQueuedNativeSignalsBeforeStoppingTheOwner) {
  exercise_native_signal_source(true);
}

TEST(AppleLateInput, MixedMlxNativeMlxPlanPreservesValuesAndReusesItsSlot) {
  auto context = a::Context::create();
  ASSERT_TRUE(context) << context.error().message();
  apple_test::Recurrent fixture(3, 2, 64);
  const auto expected = fixture.oracle(2);
  auto specs = fixture.specs();
  ad::RegionPlan plan;
  plan.values = specs;
  plan.inputs = {0, 1, 2, 3, 4, 5};
  plan.values.push_back(specs[0]); // MLX query scaling.
  plan.values.push_back(specs[2]); // Original native scores.
  plan.values.push_back(specs[5]); // Complete native state.
  plan.values.push_back(specs[2]); // Final MLX scores.
  plan.outputs = {9, 8};
  ad::Region first;
  first.inputs = {0}; first.outputs = {6};
  first.mlx = [](const auto &args, mx::Stream stream) {
    return std::vector<mx::array>{mx::multiply(args[0], mx::array(2.f), stream)};
  };
  ad::Region native;
  native.kind = ad::RegionKind::kGgml;
  native.inputs = {6, 1, 2, 3, 4, 5}; native.outputs = {7, 8};
  native.operation = c::Operation(c::GatedDeltaNet{});
  for (const auto &input : fixture.inputs) {
    std::vector<std::int64_t> shape(input.shape().begin(), input.shape().end());
    native.native_inputs.push_back(c::TensorDesc::dense(c::DataType::kFloat32, shape).value());
  }
  ad::Region last;
  last.inputs = {7}; last.outputs = {9};
  last.mlx = [](const auto &args, mx::Stream stream) {
    return std::vector<mx::array>{mx::multiply(args[0], mx::array(3.f), stream)};
  };
  plan.regions = {std::move(first), std::move(native), std::move(last)};
  auto program = ad::ContextAccess::compile(*context, std::move(plan));
  ASSERT_TRUE(program) << program.error().message();
  auto invocation = program->create_invocation();
  ASSERT_TRUE(invocation) << invocation.error().message();
  for (int repeat = 0; repeat < 3; ++repeat) {
    ASSERT_TRUE(invocation->bind(fixture.inputs).ok());
    auto result = invocation->submit();
    ASSERT_TRUE(result) << result.error().message();
    ASSERT_TRUE(result->completion().wait().ok());
    auto scores = ad::InvocationAccess::output(*invocation, 0);
    auto state = ad::InvocationAccess::output(*invocation, 1);
    ASSERT_TRUE(scores); ASSERT_TRUE(state);
    for (std::size_t i = 0; i < expected[0].size(); ++i)
      EXPECT_NEAR(scores->data<float>()[i], 3 * expected[0][i], 1.e-5);
    for (std::size_t i = 0; i < expected[1].size(); ++i)
      EXPECT_NEAR(state->data<float>()[i], expected[1][i], 1.e-5);
  }
  ASSERT_TRUE(context->close().ok());
}

TEST(AppleLateInput, DeferredMlxInputErrorPreventsPublicationAndKeepsTheRuntimeUsable) {
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  const std::array<std::int64_t, 2> table_shape{4, 64}, dot_shape{1, 64};
  const std::array<std::int64_t, 2> weight_shape{2, 64};
  const std::array<std::int64_t, 1> index_shape{1};
  const std::array descriptions{
      c::TensorDesc::dense(c::DataType::kFloat32, table_shape).value(),
      c::TensorDesc::dense(c::DataType::kInt32, index_shape).value()};
  ad::RegionPlan plan;
  plan.values = {{{4, 64}, mx::float32}, {{1}, mx::int32}, {{2, 64}, mx::float32},
                 {{1, 64}, mx::float32}, {{1, 2}, mx::float32}};
  plan.inputs = {0, 1, 2}; plan.outputs = {4};
  ad::Region checked;
  checked.inputs = {0, 1}; checked.outputs = {3};
  checked.mlx = [descriptions](const auto &args, mx::Stream stream) {
    return ad::lower_mlx_float(c::Operation(c::Gather{}), descriptions, args, stream).value();
  };
  ad::Region native;
  native.kind = ad::RegionKind::kGgml;
  native.inputs = {3, 2}; native.outputs = {4};
  native.operation = c::model_ops::dense_linear_f32();
  native.native_inputs = {
      c::TensorDesc::dense(c::DataType::kFloat32, dot_shape).value(),
      c::TensorDesc::dense(c::DataType::kFloat32, weight_shape).value()};
  plan.regions = {std::move(checked), std::move(native)};
  auto program = ad::ContextAccess::compile(*context, std::move(plan));
  ASSERT_TRUE(program) << program.error().message();
  std::array<float, 256> table_values; table_values.fill(1.f);
  std::array<float, 128> weight_values; weight_values.fill(3.f / 64);
  const mx::array table(table_values.data(), {4, 64}), weight(weight_values.data(), {2, 64});
  for (const auto index : {8, 0}) {
    auto invocation = program->create_invocation();
    ASSERT_TRUE(invocation) << invocation.error().message();
    const std::array inputs{table, mx::array({index}, mx::int32), weight};
    ASSERT_TRUE(invocation->bind(inputs).ok());
    auto result = invocation->submit();
    ASSERT_TRUE(result);
    const auto status = result->completion().wait();
    if (index == 8) {
      EXPECT_TRUE(status.IsOutOfRange()) << status.message();
      EXPECT_FALSE(ad::InvocationAccess::output(*invocation, 0));
    } else {
      ASSERT_TRUE(status.ok()) << status.message();
      auto output = ad::InvocationAccess::output(*invocation, 0);
      ASSERT_TRUE(output);
      EXPECT_FLOAT_EQ(output->data<float>()[0], 3.f);
      EXPECT_FLOAT_EQ(output->data<float>()[1], 3.f);
    }
  }
  ASSERT_TRUE(context->close().ok());
}

TEST(AppleLateInput, PinsActualProducerUntilConsumerRetiresAndRejectsStaleEpoch) {
  auto context = a::Context::create();
  ASSERT_TRUE(context) << context.error().message();
  std::atomic<int> traces{0};
  const std::array specs{a::InputSpec{{4}, mx::float32}};
  auto first = ad::ContextAccess::compile(ad::ContextAccess::runtime(*context),
      [&](const auto &args, mx::Stream stream) {
        ++traces;
        return std::vector<mx::array>{mx::add(args[0], mx::array(1.f), stream)};
      }, specs);
  auto second = ad::ContextAccess::compile(ad::ContextAccess::runtime(*context),
      [&](const auto &args, mx::Stream stream) {
        ++traces;
        return std::vector<mx::array>{mx::multiply(args[0], mx::array(2.f), stream)};
      }, specs);
  ASSERT_TRUE(first); ASSERT_TRUE(second);
  auto producer = first->create_invocation();
  auto consumer = second->create_invocation();
  ASSERT_TRUE(producer); ASSERT_TRUE(consumer);
  auto source = ad::InvocationAccess::source(*producer, 0);
  ASSERT_TRUE(source);
  std::array<ad::InputBinding, 1> late;
  late[0].source = std::move(*source);
  ASSERT_TRUE(ad::InvocationAccess::prepare_inputs(*consumer, late).ok());
  late[0].source.reset(); // Only the consumer's cold binding retains this Job.
  const int cold_traces = traces.load();
  const std::array values{mx::array({1.f, 2.f, 3.f, 4.f})};
  ASSERT_TRUE(producer->bind(values).ok());
  {
    auto produced = producer->submit();
    ASSERT_TRUE(produced);
    ASSERT_TRUE(produced->completion().wait().ok());
  } // No ResultStorage owner is retained by the test.
  {
    FifoGate gate;
    ASSERT_TRUE(gate.enqueue(ad::InvocationAccess::runtime(*producer)).ok());
    gate.wait_entered();
    auto pending = consumer->submit();
    ASSERT_TRUE(pending);
    std::array<ad::InputBinding, 1> ready;
    ready[0].value = values[0];
    EXPECT_TRUE(producer->bind(values).IsUnavailable());
    auto replay = producer->submit();
    ASSERT_FALSE(replay);
    EXPECT_TRUE(replay.error().IsUnavailable());
    EXPECT_TRUE(ad::InvocationAccess::prepare_inputs(*producer, ready).IsUnavailable());
    EXPECT_TRUE(ad::InvocationAccess::release_inputs(*producer).IsUnavailable());
    EXPECT_TRUE(ad::InvocationAccess::reuse_gate(*producer).IsUnavailable());
    gate.open_and_join();
    ASSERT_TRUE(pending->completion().wait().ok());
    auto output = ad::InvocationAccess::output(*consumer, 0);
    ASSERT_TRUE(output);
    for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(output->data<float>()[i], 4.f + 2.f * i);
  }
  auto stale = consumer->submit();
  ASSERT_FALSE(stale);
  EXPECT_TRUE(stale.error().IsFailedPrecondition());
  ASSERT_TRUE(ad::InvocationAccess::reuse_gate(*producer).ok());
  const std::array changed{mx::array({5.f, 6.f, 7.f, 8.f})};
  ASSERT_TRUE(producer->bind(changed).ok());
  {
    auto produced = producer->submit();
    ASSERT_TRUE(produced);
    ASSERT_TRUE(produced->completion().wait().ok());
  }
  *producer = a::Invocation{}; // The consumer retains the actual source Job.
  auto repeated = consumer->submit();
  ASSERT_TRUE(repeated);
  ASSERT_TRUE(repeated->completion().wait().ok());
  EXPECT_EQ(traces.load(), cold_traces);
  ASSERT_TRUE(context->close().ok());
  auto output = ad::InvocationAccess::output(*consumer, 0);
  ASSERT_TRUE(output);
  for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(output->data<float>()[i], 12.f + 2.f * i);
}

TEST(AppleLateInput, ReplayCopyPinsNewEpochsAndKeepsFailedProducerBytesUntouched) {
  namespace m = hyle::mem;
  auto context = a::Context::create().value();
  const auto runtime = ad::ContextAccess::runtime(context);
  const std::array specs{a::InputSpec{{4}, mx::float32}};
  auto program = context.prepare(c::Unary{c::UnaryKind::kNeg}, specs).value();
  auto producer = program.create_invocation().value();
  const auto description = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{4}).value();
  auto target = m::Buffer::create(m::system_resource(), 16, alignof(float)).value();
  m::QuotaAllocator metadata(m::system_allocator());
  auto copy = ad::prepare_copy_recording_runtime(runtime, 1, metadata.get(), 16,
      ad::CopyRecordingMode::kReplayable).value();
  auto source = ad::InvocationAccess::source(producer, 0).value();
  ASSERT_TRUE(ad::bind_host_copy_recording(*copy, std::move(source), description, target.view()).ok());
  const auto cold_metadata = metadata.used();
  const std::array first_values{mx::array({1.f, 2.f, 3.f, 4.f})};
  ASSERT_TRUE(producer.bind(first_values).ok());
  {
    auto produced = producer.submit().value();
    ASSERT_TRUE(produced.completion().wait().ok());
  }
  c::Completion old_completion;
  {
    FifoGate gate;
    ASSERT_TRUE(gate.enqueue(runtime).ok());
    gate.wait_entered();
    old_completion = ad::submit_copy_recording(*copy).value();
    EXPECT_TRUE(ad::copy_recording_reuse_gate(*copy).IsUnavailable());
    EXPECT_TRUE(producer.bind(first_values).IsUnavailable());
    EXPECT_TRUE(ad::InvocationAccess::reuse_gate(producer).IsUnavailable());
    gate.open_and_join();
    ASSERT_TRUE(old_completion.wait().ok());
  }
  std::array<float, 4> observed{};
  ASSERT_TRUE(target.view().read(std::as_writable_bytes(std::span(observed))).ok());
  EXPECT_EQ(observed, (std::array<float, 4>{-1, -2, -3, -4}));
  ASSERT_TRUE(ad::copy_recording_reuse_gate(*copy).ok());
  auto stale = ad::submit_copy_recording(*copy);
  ASSERT_FALSE(stale);
  EXPECT_TRUE(stale.error().IsFailedPrecondition());
  ASSERT_TRUE(ad::InvocationAccess::reuse_gate(producer).ok());

  ASSERT_TRUE(producer.bind(std::array{mx::array({5.f, 6.f, 7.f, 8.f})}).ok());
  {
    auto produced = producer.submit().value();
    ASSERT_TRUE(produced.completion().wait().ok());
  }
  // No new bind/record/map: the settled first Completion remains alive and
  // keeps its own terminal state while the original recording consumes epoch 2.
  auto second = ad::submit_copy_recording(*copy).value();
  ASSERT_TRUE(old_completion.wait().ok());
  EXPECT_EQ(old_completion.poll().value(), c::CompletionPoll::kComplete);
  EXPECT_TRUE(old_completion.request_cancel().ok());
  ASSERT_TRUE(second.wait().ok());
  ASSERT_TRUE(target.view().read(std::as_writable_bytes(std::span(observed))).ok());
  EXPECT_EQ(observed, (std::array<float, 4>{-5, -6, -7, -8}));
  EXPECT_EQ(metadata.used(), cold_metadata);
  auto repeated = ad::submit_copy_recording(*copy);
  ASSERT_FALSE(repeated);
  EXPECT_TRUE(repeated.error().IsFailedPrecondition());

  auto gather = context.prepare(c::Gather{},
      std::array{a::InputSpec{{4}, mx::float32}, a::InputSpec{{1}, mx::int32}}).value();
  auto bad = gather.create_invocation().value();
  const auto one = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{1}).value();
  auto untouched = m::Buffer::create(m::system_resource(), sizeof(float), alignof(float)).value();
  const std::array<float, 1> sentinel{123.f};
  ASSERT_TRUE(untouched.view().write(std::as_bytes(std::span(sentinel))).ok());
  auto failed_copy = ad::prepare_copy_recording_runtime(runtime, 1, m::system_allocator(), sizeof(float),
      ad::CopyRecordingMode::kReplayable).value();
  ASSERT_TRUE(ad::bind_host_copy_recording(*failed_copy,
      ad::InvocationAccess::source(bad, 0).value(), one, untouched.view()).ok());
  ASSERT_TRUE(bad.bind(std::array{mx::array({1.f, 2.f, 3.f, 4.f}), mx::array({9}, mx::int32)}).ok());
  {
    FifoGate gate;
    ASSERT_TRUE(gate.enqueue(runtime).ok());
    gate.wait_entered();
    auto produced = bad.submit().value();
    auto consumed = ad::submit_copy_recording(*failed_copy).value();
    EXPECT_TRUE(ad::InvocationAccess::reuse_gate(bad).IsUnavailable());
    gate.open_and_join();
    EXPECT_TRUE(produced.completion().wait().IsOutOfRange());
    EXPECT_TRUE(consumed.wait().IsOutOfRange());
  }
  std::array<float, 1> unchanged{};
  ASSERT_TRUE(untouched.view().read(std::as_writable_bytes(std::span(unchanged))).ok());
  EXPECT_EQ(unchanged, sentinel);
  EXPECT_TRUE(ad::copy_recording_reuse_gate(*failed_copy).IsOutOfRange());
  failed_copy.reset();
  ASSERT_TRUE(context.close().ok()); // Checked failure still drains the original owner.
}

TEST(AppleLateInput, RejectsCyclesPortsAndForeignRuntimeAndPropagatesCheckedFailure) {
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  const std::array specs{a::InputSpec{{1}, mx::float32}};
  const auto identity = [](const auto &args, mx::Stream stream) {
    return std::vector<mx::array>{mx::negative(args[0], stream)};
  };
  auto program = context->compile(identity, specs);
  ASSERT_TRUE(program);
  auto first = program->create_invocation(), second = program->create_invocation();
  ASSERT_TRUE(first); ASSERT_TRUE(second);
  auto first_source = ad::InvocationAccess::source(*first, 0);
  auto second_source = ad::InvocationAccess::source(*second, 0);
  ASSERT_TRUE(first_source); ASSERT_TRUE(second_source);
  EXPECT_FALSE(ad::InvocationAccess::source(*first, 1));
  std::array<ad::InputBinding, 1> binding;
  binding[0].source = *first_source;
  EXPECT_TRUE(ad::InvocationAccess::prepare_inputs(*first, binding).IsInvalidArgument());
  binding[0].source = *second_source;
  ASSERT_TRUE(ad::InvocationAccess::prepare_inputs(*first, binding).ok());
  // This legitimate edge uses the later-created source slot.
  binding[0].source = *first_source;
  EXPECT_TRUE(ad::InvocationAccess::prepare_inputs(*second, binding).IsInvalidArgument());
  auto foreign = a::Context::create();
  ASSERT_TRUE(foreign);
  auto foreign_program = foreign->compile(identity, specs);
  ASSERT_TRUE(foreign_program);
  auto foreign_slot = foreign_program->create_invocation();
  ASSERT_TRUE(foreign_slot);
  auto foreign_source = ad::InvocationAccess::source(*foreign_slot, 0);
  ASSERT_TRUE(foreign_source);
  binding[0].source = *foreign_source;
  EXPECT_TRUE(ad::InvocationAccess::prepare_inputs(*second, binding).IsUnimplemented());

  auto gather = context->prepare(c::Gather{},
      std::array{a::InputSpec{{4}, mx::float32}, a::InputSpec{{1}, mx::int32}});
  ASSERT_TRUE(gather);
  auto bad = gather->create_invocation(), dependent = program->create_invocation();
  ASSERT_TRUE(bad); ASSERT_TRUE(dependent);
  auto bad_source = ad::InvocationAccess::source(*bad, 0);
  ASSERT_TRUE(bad_source);
  binding[0].source = *bad_source;
  ASSERT_TRUE(ad::InvocationAccess::prepare_inputs(*dependent, binding).ok());
  ASSERT_TRUE(bad->bind(std::array{mx::array({1.f, 2.f, 3.f, 4.f}),
                                 mx::array({9}, mx::int32)}).ok());
  {
    FifoGate gate;
    ASSERT_TRUE(gate.enqueue(ad::InvocationAccess::runtime(*bad)).ok());
    gate.wait_entered();
    auto produced = bad->submit();
    auto consumed = dependent->submit();
    ASSERT_TRUE(produced); ASSERT_TRUE(consumed);
    gate.open_and_join();
    EXPECT_TRUE(produced->completion().wait().IsOutOfRange());
    EXPECT_TRUE(consumed->completion().wait().IsOutOfRange());
    EXPECT_FALSE(ad::InvocationAccess::output(*dependent, 0));
  }
  // Checked input errors retire their slots, not the shared Runtime.
  ASSERT_TRUE(second->bind(std::array{mx::array({3.f})}).ok());
  auto independent = second->submit();
  ASSERT_TRUE(independent);
  ASSERT_TRUE(independent->completion().wait().ok());
  auto output = ad::InvocationAccess::output(*second, 0);
  ASSERT_TRUE(output);
  EXPECT_FLOAT_EQ(output->data<float>()[0], -3.f);
}
} // namespace

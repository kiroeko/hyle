#include "hyle/compute/apple.h"
#include "hyle/compute/apple_memory.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <mutex>
#include <vector>
#include <thread>
#include <optional>

#include <gtest/gtest.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>
#include <stdexcept>
#include <mlx/backend/metal/event.h>

#include "../../src/compute_route_apple/apple_carrier.h"
#include "../../src/compute_route_apple/apple_copy_internal.h"
#include "../../src/compute_route_apple/apple_runtime_internal.h"
#include "hyle/mem/quota_allocator.h"
#include "hyle/compute/detail/apple_plan.h"
#include "hyle/compute/detail/completion_access.h"
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
  bool wait_entered_for(std::chrono::seconds limit) {
    std::unique_lock lock(mutex_);
    return changed_.wait_for(lock, limit, [&] { return entered_; });
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

// Block real GPU work while the original FIFO remains free to encode Jobs.
// Destruction always releases the event, including assertion exits.
class GpuEventGate final : public ad::Job {
public:
  explicit GpuEventGate(std::shared_ptr<ad::Runtime> runtime) : runtime_(std::move(runtime)) {
    execute = [](ad::Job *job) noexcept {
      auto &gate = *static_cast<GpuEventGate *>(job);
      auto event = mx::Event(*gate.runtime_->stream);
      event.set_value(1);
      event.wait(*gate.runtime_->stream);
      auto &encoder = mx::metal::get_command_encoder(*gate.runtime_->stream);
      encoder.end_encoding();
      encoder.commit();
      { std::lock_guard lock(gate.mutex_); gate.event_ = std::move(event); }
      gate.changed_.notify_all();
    };
  }
  hyle::Status enqueue() noexcept {
    auto status = runtime_->enqueue(*this);
    if (!status.ok()) return status;
    std::unique_lock lock(mutex_);
    changed_.wait(lock, [&] { return event_.valid(); });
    return hyle::Status::Ok();
  }
  void open() {
    std::lock_guard lock(mutex_);
    if (event_.valid()) event_.cast<mx::metal::EventImpl>().signal(1);
  }
  ~GpuEventGate() { open(); }
private:
  std::shared_ptr<ad::Runtime> runtime_;
  std::mutex mutex_;
  std::condition_variable changed_;
  mx::Event event_;
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

void exercise_mixed_plan(bool pending_chain, bool close_pending = false, bool cancel_pending = false, std::size_t order_count = 0) {
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
  std::optional<a::Invocation> consumer;
  auto runtime = ad::ContextAccess::runtime(*context);
  apple_test::Recurrent following(3, 2, 64);
  for (std::size_t i = 0; i < expected[0].size(); ++i)
    following.data[0][i] = float(3 * expected[0][i]);
  const auto next_expected = following.oracle(2);
  if (pending_chain) {
    auto next = program->create_invocation();
    ASSERT_TRUE(next) << next.error().message();
    consumer.emplace(std::move(*next));
    if (order_count) {
      ASSERT_TRUE(consumer->bind(fixture.inputs).ok());
      ASSERT_TRUE(ad::InvocationAccess::reserve_dependencies(*consumer, order_count).ok());
    }
    std::vector<ad::InputBinding> bindings(fixture.inputs.size());
    auto source = ad::InvocationAccess::source(*invocation, 0);
    ASSERT_TRUE(source);
    bindings[0].source = std::move(*source);
    for (std::size_t i = 1; i < bindings.size(); ++i) bindings[i].value = fixture.inputs[i];
    if (!order_count) ASSERT_TRUE(ad::InvocationAccess::prepare_inputs(*consumer, bindings).ok());
  }
  std::vector<a::Invocation> preceding;
  for (std::size_t i = 1; i < order_count; ++i) {
    auto slot = program->create_invocation(); ASSERT_TRUE(slot);
    ASSERT_TRUE(slot->bind(fixture.inputs).ok()); preceding.push_back(std::move(*slot));
  }
  for (int repeat = 0; repeat < (close_pending || cancel_pending ? 1 : 3); ++repeat) {
    ASSERT_TRUE(invocation->bind(fixture.inputs).ok());
    std::optional<GpuEventGate> gpu_gate;
    if (pending_chain) {
      gpu_gate.emplace(runtime);
      ASSERT_TRUE(gpu_gate->enqueue().ok());
    }
    auto result = invocation->submit();
    ASSERT_TRUE(result) << result.error().message();
    if (pending_chain) {
      std::vector<c::OwnedResult<a::Results>> preceding_results;
      preceding_results.reserve(preceding.size());
      std::vector<c::detail::CompletionDependency> dependencies;
      if (order_count) {
        dependencies.push_back(c::detail::CompletionAccess::dependency(result->completion()));
        for (auto &slot : preceding) {
          auto work = slot.submit(); ASSERT_TRUE(work);
          preceding_results.push_back(std::move(*work));
          dependencies.push_back(c::detail::CompletionAccess::dependency(preceding_results.back().completion()));
        }
      }
      auto next = order_count
          ? ad::InvocationAccess::submit_ordered(*consumer, {}, nullptr, nullptr, dependencies)
          : consumer->submit();
      ASSERT_TRUE(next) << next.error().message();
      FifoGate owner_gate;
      ASSERT_TRUE(owner_gate.enqueue(runtime).ok());
      const bool owner_free = owner_gate.wait_entered_for(std::chrono::seconds(2));
      EXPECT_TRUE(owner_free) << "Mixed Plans waited on the host before encoding the next Job";
      EXPECT_EQ(result->completion().poll().value(), c::CompletionPoll::kPending);
      EXPECT_EQ(next->completion().poll().value(), c::CompletionPoll::kPending);
      if (cancel_pending) EXPECT_TRUE(result->request_cancel().ok());
      // Release the real GPU before any blocking assertion/close path.
      gpu_gate->open();
      if (close_pending && owner_free) {
        hyle::Status closed;
        std::thread closer([&] { closed = context->close(); });
        for (;;) {
          bool closing;
          { std::lock_guard lock(runtime->queue_mutex); closing = !runtime->accepting; }
          if (closing) break;
          std::this_thread::yield();
        }
        owner_gate.open_and_join();
        closer.join();
        ASSERT_TRUE(closed.ok()) << closed.message();
      } else owner_gate.open_and_join();
      const auto child_status = next->completion().wait();
      if (cancel_pending) {
        EXPECT_TRUE(child_status.IsCancelled()) << child_status.message();
        EXPECT_TRUE(result->completion().wait().IsCancelled());
        EXPECT_FALSE(ad::InvocationAccess::output(*invocation, 0));
        EXPECT_FALSE(ad::InvocationAccess::output(*consumer, 0));
        ASSERT_TRUE(context->close().ok());
        return;
      }
      ASSERT_TRUE(child_status.ok()) << child_status.message();
      auto next_scores = ad::InvocationAccess::output(*consumer, 0);
      auto next_state = ad::InvocationAccess::output(*consumer, 1);
      ASSERT_TRUE(next_scores); ASSERT_TRUE(next_state);
      const auto &expected_next = order_count ? expected : next_expected;
      for (std::size_t i = 0; i < expected_next[0].size(); ++i)
        EXPECT_NEAR(next_scores->data<float>()[i], 3 * expected_next[0][i], 1.e-5);
      for (std::size_t i = 0; i < expected_next[1].size(); ++i)
        EXPECT_NEAR(next_state->data<float>()[i], expected_next[1][i], 1.e-5);
      for (auto &work : preceding_results) ASSERT_TRUE(work.completion().wait().ok());
    }
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

TEST(AppleLateInput, MixedMlxNativeMlxPlanPreservesValuesAndReusesItsSlot) {
  exercise_mixed_plan(false);
}
TEST(AppleLateInput, MixedPlansEncodePendingSourcesWithoutBlockingTheirOwner) {
  exercise_mixed_plan(true);
}
TEST(AppleLateInput, CloseDrainsPendingMixedPlanSourcesOnTheirOriginalOwner) {
  exercise_mixed_plan(true, true);
}

TEST(AppleLateInput, CancelDrainsPendingMixedPlanAndRejectsDependentPublication) {
  exercise_mixed_plan(true, false, true);
}

TEST(AppleLateInput, NativeOrderEdgesKeepIndependentMathematicalInputs) {
  exercise_mixed_plan(true, false, false, 1);
}
TEST(AppleLateInput, NativeOrderFanInExceedsTheTensorInputPortCount) {
  exercise_mixed_plan(true, false, false, 9);
}
TEST(AppleLateInput, CloseDrainsOriginalOrderOnlyProducerAndConsumer) {
  exercise_mixed_plan(true, true, false, 1);
}
TEST(AppleLateInput, CancelledOrderOnlyProducerRejectsIndependentOutput) {
  exercise_mixed_plan(true, false, true, 1);
}

void exercise_multiple_native_prefixes(bool order_only) {
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  const auto x = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{1, 64}).value();
  const auto w = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{64, 64}).value();
  ad::RegionPlan plan;
  plan.values = {{{1, 64}, mx::float32}, {{64, 64}, mx::float32},
                 {{1, 64}, mx::float32}, {{1, 64}, mx::float32},
                 {{1, 64}, mx::float32}, {{1, 64}, mx::float32}};
  plan.inputs = {0, 1}; plan.outputs = {5, 2};
  ad::Region first, middle, last, output;
  first.kind = last.kind = ad::RegionKind::kGgml;
  first.operation = last.operation = c::model_ops::dense_linear_f32();
  first.native_inputs = last.native_inputs = {x, w};
  first.inputs = {0, 1}; first.outputs = {2};
  middle.inputs = {2}; middle.outputs = {3};
  middle.mlx = [](const auto &args, mx::Stream stream) {
    return std::vector<mx::array>{mx::multiply(args[0], mx::array(3.f), stream)};
  };
  last.inputs = {3, 1}; last.outputs = {4};
  output.inputs = {4}; output.outputs = {5};
  output.mlx = [](const auto &args, mx::Stream stream) {
    return std::vector<mx::array>{mx::multiply(args[0], mx::array(.5f), stream)};
  };
  plan.regions = {std::move(first), std::move(middle), std::move(last), std::move(output)};
  auto program = ad::ContextAccess::compile(*context, std::move(plan));
  ASSERT_TRUE(program) << program.error().message();
  auto invocation = program->create_invocation();
  ASSERT_TRUE(invocation);
  std::optional<a::Invocation> consumer;
  if (order_only) {
    auto next = program->create_invocation(); ASSERT_TRUE(next);
    consumer = std::move(*next);
    ASSERT_TRUE(ad::InvocationAccess::reserve_dependencies(*consumer, 1).ok());
  }
  std::array<float, 64> input;
  std::array<float, 64 * 64> weights{};
  for (std::size_t i = 0; i < input.size(); ++i) {
    input[i] = float(i + 1); weights[i * 64 + i] = 2.f;
  }
  const std::array inputs{mx::array(input.data(), {1, 64}), mx::array(weights.data(), {64, 64})};
  auto runtime = ad::ContextAccess::runtime(*context);
  // Exercise ready execution before the repeated pending-source episodes.
  ASSERT_TRUE(invocation->bind(inputs).ok());
  { auto warm = invocation->submit(); ASSERT_TRUE(warm); ASSERT_TRUE(warm->completion().wait().ok()); }
  for (int repeat = 0; repeat < 3; ++repeat) {
    ASSERT_TRUE(invocation->bind(inputs).ok());
    if (consumer) ASSERT_TRUE(consumer->bind(inputs).ok());
    GpuEventGate gpu_gate(runtime);
    ASSERT_TRUE(gpu_gate.enqueue().ok());
    auto result = invocation->submit();
    ASSERT_TRUE(result);
    std::optional<c::OwnedResult<a::Results>> dependent;
    if (consumer) {
      const std::array dependencies{c::detail::CompletionAccess::dependency(result->completion())};
      auto submitted = ad::InvocationAccess::submit_ordered(*consumer, {}, nullptr, nullptr, dependencies);
      ASSERT_TRUE(submitted); dependent = std::move(*submitted);
    }
    FifoGate owner_gate;
    ASSERT_TRUE(owner_gate.enqueue(runtime).ok());
    const bool owner_free = owner_gate.wait_entered_for(std::chrono::seconds(2));
    auto &completion = dependent ? dependent->completion() : result->completion();
    const auto pending = completion.poll();
    gpu_gate.open();
    owner_gate.open_and_join();
    EXPECT_TRUE(owner_free) << "The first prefix cannot retire the still-pending complete Plan";
    ASSERT_TRUE(pending) << pending.error().message(); EXPECT_EQ(*pending, c::CompletionPoll::kPending);
    ASSERT_TRUE(result->completion().wait().ok());
    ASSERT_TRUE(completion.wait().ok());
    auto &observed = consumer ? *consumer : *invocation;
    auto final = ad::InvocationAccess::output(observed, 0);
    auto early = ad::InvocationAccess::output(observed, 1);
    ASSERT_TRUE(final); ASSERT_TRUE(early);
    for (std::size_t i = 0; i < input.size(); ++i) {
      EXPECT_FLOAT_EQ(final->data<float>()[i], 6 * input[i]);
      EXPECT_FLOAT_EQ(early->data<float>()[i], 2 * input[i]);
    }
    std::lock_guard lock(runtime->queue_mutex);
    EXPECT_EQ(runtime->native_tails, 0u);
  }
  ASSERT_TRUE(context->close().ok());
}
TEST(AppleLateInput, MultipleNativePrefixesRetireOnTheLastOriginalCallback) {
  exercise_multiple_native_prefixes(false);
}
TEST(AppleLateInput, MultipleNativeConsumerPrefixesKeepBothOriginalQueueFrontiers) {
  exercise_multiple_native_prefixes(true);
}

void exercise_deferred_input_error(bool order_only) {
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
  const std::array source_specs{a::InputSpec{{1, 2}, mx::float32}};
  auto next_program = ad::ContextAccess::compile(ad::ContextAccess::runtime(*context),
      [](const auto &args, mx::Stream stream) {
        return std::vector<mx::array>{mx::multiply(args[0], mx::array(2.f), stream)};
      }, source_specs);
  ASSERT_TRUE(next_program);
  for (const auto index : {8, 0}) {
    auto invocation = program->create_invocation();
    ASSERT_TRUE(invocation) << invocation.error().message();
    const std::array inputs{table, mx::array({index}, mx::int32), weight};
    ASSERT_TRUE(invocation->bind(inputs).ok());
    auto consumer = next_program->create_invocation();
    ASSERT_TRUE(consumer);
    if (order_only) {
      ASSERT_TRUE(ad::InvocationAccess::reserve_dependencies(*consumer, 1).ok());
      ASSERT_TRUE(consumer->bind(std::array{mx::array({2.f, 3.f}, {1, 2})}).ok());
    } else {
      std::array<ad::InputBinding, 1> late;
      late[0].source = ad::InvocationAccess::source(*invocation, 0).value();
      ASSERT_TRUE(ad::InvocationAccess::prepare_inputs(*consumer, late).ok());
    }
    FifoGate gate;
    ASSERT_TRUE(gate.enqueue(ad::ContextAccess::runtime(*context)).ok());
    gate.wait_entered();
    auto result = invocation->submit();
    ASSERT_TRUE(result);
    const std::array dependencies{c::detail::CompletionAccess::dependency(result->completion())};
    auto consumed = order_only
        ? ad::InvocationAccess::submit_ordered(*consumer, {}, nullptr, nullptr, dependencies)
        : consumer->submit();
    ASSERT_TRUE(consumed);
    gate.open_and_join();
    const auto child_status = consumed->completion().wait();
    const auto status = result->completion().wait();
    if (index == 8) {
      EXPECT_TRUE(status.IsOutOfRange()) << status.message();
      EXPECT_TRUE(child_status.IsOutOfRange()) << child_status.message();
      EXPECT_FALSE(ad::InvocationAccess::output(*invocation, 0));
      EXPECT_FALSE(ad::InvocationAccess::output(*consumer, 0));
    } else {
      ASSERT_TRUE(status.ok()) << status.message();
      auto output = ad::InvocationAccess::output(*invocation, 0);
      ASSERT_TRUE(output);
      EXPECT_FLOAT_EQ(output->data<float>()[0], 3.f);
      EXPECT_FLOAT_EQ(output->data<float>()[1], 3.f);
      ASSERT_TRUE(child_status.ok()) << child_status.message();
      auto child = ad::InvocationAccess::output(*consumer, 0);
      ASSERT_TRUE(child);
      EXPECT_FLOAT_EQ(child->data<float>()[0], order_only ? 4.f : 6.f);
      EXPECT_FLOAT_EQ(child->data<float>()[1], 6.f);
    }
  }
  ASSERT_TRUE(context->close().ok());
}
TEST(AppleLateInput, DeferredMlxInputErrorPreventsPublicationAndKeepsTheRuntimeUsable) {
  exercise_deferred_input_error(false);
}
TEST(AppleLateInput, OrderOnlyErrorPreventsIndependentConsumerPublication) {
  exercise_deferred_input_error(true);
}

TEST(AppleLateInput, CopyAndMathPinOriginalOrderOnlyEpisodesAfterPublicHandleAbandonment) {
  namespace m = hyle::mem;
  auto context = a::Context::create().value();
  const auto runtime = ad::ContextAccess::runtime(context);
  const std::array specs{a::InputSpec{{4}, mx::float32}};
  auto program = context.prepare(c::Unary{c::UnaryKind::kNeg}, specs).value();
  auto producer = program.create_invocation().value(), consumer = program.create_invocation().value();
  ASSERT_TRUE(producer.bind(std::array{mx::array({1.f, 2.f, 3.f, 4.f})}).ok());
  ASSERT_TRUE(consumer.bind(std::array{mx::array({5.f, 6.f, 7.f, 8.f})}).ok());
  ASSERT_TRUE(ad::InvocationAccess::reserve_dependencies(consumer, 1).ok());
  const auto description = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{4}).value();
  auto memory = a::MemoryResource::create(context).value();
  auto source = m::Buffer::create(memory.ref(), 16, alignof(float)).value();
  auto target = m::Buffer::create(m::system_resource(), 16, alignof(float)).value();
  const std::array<float, 4> values{10.f, 20.f, 30.f, 40.f};
  ASSERT_TRUE(source.view().write(std::as_bytes(std::span(values))).ok());
  auto copy = ad::prepare_copy_recording_runtime(runtime, 1, m::system_allocator(), 16,
      ad::CopyRecordingMode::kReplayable).value();
  ASSERT_TRUE(ad::reserve_copy_dependencies(*copy, 1).ok());
  ASSERT_TRUE(ad::bind_host_copy_recording(*copy,
      c::ConstTensorView::create(description, source.view()).value(), target.view(), {}).ok());
  FifoGate gate;
  ASSERT_TRUE(gate.enqueue(runtime).ok()); gate.wait_entered();
  auto produced = producer.submit().value();
  const std::array before_copy{c::detail::CompletionAccess::dependency(produced.completion())};
  auto copied = ad::submit_copy_recording(*copy, before_copy).value();
  const std::array before_math{c::detail::CompletionAccess::dependency(copied)};
  auto consumed = ad::InvocationAccess::submit_ordered(consumer, {}, nullptr, nullptr, before_math);
  ASSERT_TRUE(consumed) << consumed.error().message();
  produced = {}; copied = {};
  EXPECT_TRUE(ad::copy_recording_reuse_gate(*copy).IsUnavailable());
  EXPECT_TRUE(ad::InvocationAccess::reuse_gate(producer).IsUnavailable());
  auto replay = ad::submit_copy_recording(*copy);
  ASSERT_FALSE(replay); EXPECT_TRUE(replay.error().IsFailedPrecondition());
  gate.open_and_join();
  ASSERT_TRUE(consumed->completion().wait().ok());
  std::array<float, 4> observed{};
  ASSERT_TRUE(target.view().read(std::as_writable_bytes(std::span(observed))).ok());
  EXPECT_EQ(observed, values);
  const auto output = ad::InvocationAccess::output(consumer, 0).value();
  EXPECT_FLOAT_EQ(output.data<float>()[0], -5.f);
  EXPECT_TRUE(ad::copy_recording_reuse_gate(*copy).ok());
  EXPECT_TRUE(ad::InvocationAccess::reuse_gate(producer).ok());
  ASSERT_TRUE(context.close().ok());
}

TEST(AppleLateInput, EmptyCopyStillPublishesItsOriginalOrderDependencyFailure) {
  auto context = a::Context::create().value();
  auto program = context.prepare(c::Gather{},
      std::array{a::InputSpec{{4}, mx::float32}, a::InputSpec{{1}, mx::int32}}).value();
  auto producer = program.create_invocation().value();
  ASSERT_TRUE(producer.bind(std::array{mx::array({1.f, 2.f, 3.f, 4.f}), mx::array({9}, mx::int32)}).ok());
  const auto runtime = ad::ContextAccess::runtime(context);
  auto copy = ad::prepare_copy_recording_runtime(runtime, 1, hyle::mem::system_allocator(), 16,
      ad::CopyRecordingMode::kReplayable).value();
  ASSERT_TRUE(ad::reserve_copy_dependencies(*copy, 1).ok());
  ASSERT_TRUE(ad::bind_copy_recording(*copy, {}).ok());
  FifoGate gate;
  ASSERT_TRUE(gate.enqueue(runtime).ok()); gate.wait_entered();
  auto produced = producer.submit().value();
  const std::array dependencies{c::detail::CompletionAccess::dependency(produced.completion())};
  auto copied = ad::submit_copy_recording(*copy, dependencies);
  ASSERT_TRUE(copied) << copied.error().message();
  EXPECT_EQ(copied->poll().value(), c::CompletionPoll::kPending);
  gate.open_and_join();
  EXPECT_TRUE(copied->wait().IsOutOfRange());
  EXPECT_TRUE(produced.completion().wait().IsOutOfRange());
  ASSERT_TRUE(context.close().ok());
}

TEST(AppleLateInput, RetiredCopyInputReleaseKeepsNativeReaderEpochAndVerdict) {
  for (const bool fail : {false,true}) {
    auto context=a::Context::create().value();
    auto runtime=ad::ContextAccess::runtime(context);
    auto program=context.prepare(c::Gather{},
        std::array{a::InputSpec{{4},mx::float32},a::InputSpec{{1},mx::int32}}).value();
    auto producer=program.create_invocation().value();
    ASSERT_TRUE(producer.bind(std::array{mx::array({1.f,2.f,3.f,4.f}),mx::array({fail?9:0},mx::int32)}).ok());
    auto memory=a::MemoryResource::create(context).value();
    const auto description=c::TensorDesc::dense(c::DataType::kFloat32,
        std::array<std::int64_t,1>{1}).value();
    auto source=hyle::mem::Buffer::create(memory.ref(),sizeof(float),256).value();
    auto target=hyle::mem::Buffer::create(memory.ref(),sizeof(float),256).value();
    const std::array<float,1> input{17.f},sentinel{-7.f};
    ASSERT_TRUE(source.view().write(std::as_bytes(std::span(input))).ok());
    ASSERT_TRUE(target.view().write(std::as_bytes(std::span(sentinel))).ok());
    auto copy=ad::prepare_copy_recording_runtime(runtime,1,hyle::mem::system_allocator(),sizeof(float),
        ad::CopyRecordingMode::kReplayable).value();
    ASSERT_TRUE(ad::reserve_copy_dependencies(*copy,1).ok());
    ASSERT_TRUE(ad::bind_host_copy_recording(*copy,
        c::ConstTensorView::create(description,source.view()).value(),target.view(),{}).ok());
    FifoGate gate;
    ASSERT_TRUE(gate.enqueue(runtime).ok());gate.wait_entered();
    auto produced=producer.submit().value();
    const std::array dependencies{c::detail::CompletionAccess::dependency(produced.completion())};
    auto copied=ad::submit_copy_recording(*copy,dependencies).value();
    std::uint64_t epoch=0;
    auto pinned=ad::pin_copy_dependency(runtime,c::detail::CompletionAccess::dependency(copied),&epoch).value();
    gate.open_and_join();
    const auto terminal=copied.wait();
    EXPECT_EQ(terminal.ok(),!fail);
    EXPECT_EQ(produced.completion().wait().code(),terminal.code());
    EXPECT_TRUE(ad::copy_recording_retired_gate(*copy).ok());
    EXPECT_TRUE(ad::copy_recording_reuse_gate(*copy).IsUnavailable());
    ASSERT_TRUE(ad::clear_copy_recording(*copy).ok());
    EXPECT_TRUE(ad::copy_recording_reuse_gate(*copy).IsUnavailable());
    struct Observation final:ad::Job {
      ad::CopyRecording *copy;
      std::uint64_t epoch;
      hyle::Status terminal;
      std::atomic<bool> done{false};
      Observation(ad::CopyRecording &selected,std::uint64_t generation):copy(&selected),epoch(generation) {
        execute=[](ad::Job *job) noexcept {
          auto &call=*static_cast<Observation *>(job);
          call.terminal=ad::observe_copy_dependency(*call.copy,call.epoch);
          call.done.store(true,std::memory_order_release);call.done.notify_all();
        };
      }
    } observation(*copy,epoch);
    ASSERT_TRUE(runtime->enqueue(observation).ok());
    observation.done.wait(false,std::memory_order_acquire);
    EXPECT_EQ(observation.terminal.code(),terminal.code());
    EXPECT_EQ(observation.terminal.message(),terminal.message());
    ad::release_copy_dependency(*pinned);pinned.reset();
    EXPECT_EQ(ad::copy_recording_reuse_gate(*copy).code(),terminal.code());
    std::array<float,1> actual{};
    ASSERT_TRUE(target.view().read(std::as_writable_bytes(std::span(actual))).ok());
    EXPECT_EQ(actual,fail?sentinel:input);
    ASSERT_TRUE(context.close().ok());
  }
}

// Inject only at actual eval, after cold compilation and the native prefix.
class LateEncodingFailure final : public mx::Primitive {
public:
  LateEncodingFailure(mx::Stream stream, std::shared_ptr<std::atomic<bool>> fail)
      : Primitive(stream), fail_(std::move(fail)) {}
  void eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) override {
    throw std::runtime_error("GPU encoding checkpoint cannot use CPU");
  }
  void eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) override {
    if (fail_->load()) throw std::runtime_error("injected late mixed-Plan encoding failure");
    outputs[0].copy_shared_buffer(inputs[0]);
  }
  const char *name() const override { return "HyleLatePlanEncodingFailure"; }
private:
  std::shared_ptr<std::atomic<bool>> fail_;
};

TEST(AppleLateInput, EarlierDeviceCheckPrecedesLaterEncodingFailure) {
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  const std::array<std::int64_t, 2> table_shape{4, 64}, dot_shape{1, 64};
  const std::array<std::int64_t, 2> weight_shape{2, 64};
  const std::array<std::int64_t, 1> index_shape{1};
  const std::array descriptions{
      c::TensorDesc::dense(c::DataType::kFloat32, table_shape).value(),
      c::TensorDesc::dense(c::DataType::kInt32, index_shape).value()};
  auto fail = std::make_shared<std::atomic<bool>>(false);
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
  plan.values.push_back({{1, 2}, mx::float32}); plan.outputs = {5};
  ad::Region finish;
  finish.inputs = {4}; finish.outputs = {5};
  finish.mlx = [fail](const auto &args, mx::Stream stream) {
    return std::vector<mx::array>{mx::array(args[0].shape(), args[0].dtype(),
        std::make_shared<LateEncodingFailure>(stream, fail), {args[0]})};
  };
  plan.regions = {std::move(checked), std::move(native), std::move(finish)};
  auto program = ad::ContextAccess::compile(*context, std::move(plan));
  ASSERT_TRUE(program) << program.error().message();
  std::array<float, 256> table_values; table_values.fill(1.f);
  std::array<float, 128> weight_values; weight_values.fill(3.f / 64);
  const mx::array table(table_values.data(), {4, 64}), weight(weight_values.data(), {2, 64});
  auto invocation = program->create_invocation();
  ASSERT_TRUE(invocation);
  const std::array inputs{table, mx::array({8}, mx::int32), weight};
  ASSERT_TRUE(invocation->bind(inputs).ok());
  fail->store(true);
  auto result = invocation->submit();
  ASSERT_TRUE(result);
  const auto status = result->completion().wait();
  EXPECT_TRUE(status.IsOutOfRange()) << status.message();
  EXPECT_FALSE(ad::InvocationAccess::output(*invocation, 0));
  { std::lock_guard lock(ad::ContextAccess::runtime(*context)->queue_mutex);
    EXPECT_EQ(ad::ContextAccess::runtime(*context)->native_tails, 0u); }
  context->close().IgnoreError(); // The actual encoding exception poisons this Runtime.
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

#include "hyle/compute/apple.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <vector>

#include <gtest/gtest.h>
#include <mlx/ops.h>

#include "../../src/compute_route_apple/apple_carrier.h"
#include "../../src/compute_route_apple/apple_copy_internal.h"
#include "../../src/compute_route_apple/apple_runtime_internal.h"
#include "hyle/mem/quota_allocator.h"

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

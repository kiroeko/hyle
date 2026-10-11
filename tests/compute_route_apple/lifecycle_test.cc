#include "hyle/compute/apple.h"
#include "hyle/compute/apple_memory.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>

#include "hyle/mem/quota_allocator.h"
#include "hyle/mem/stats.h"

namespace {
namespace a = hyle::compute::apple;
namespace c = hyle::compute;
namespace mx = mlx::core;

// Test-only native primitive boundary. Real MLX GPU work precedes it; no HYLE
// runtime hooks or substitute Provider are involved. Warmup passes normally.
// The test can hold native evaluation or fail after an encoded GPU prefix.
class NativeBoundary final {
public:
  void arm(bool fail = false) {
    std::lock_guard lock(mutex_);
    armed_ = true;
    fail_ = fail;
  }
  void intercept() {
    std::unique_lock lock(mutex_);
    if (!armed_)
      return;
    armed_ = false;
    entered_ = true;
    condition_.notify_all();
    condition_.wait(lock, [&] { return released_; });
    if (fail_)
      throw std::runtime_error("injected MLX native boundary failure");
  }
  bool entered() {
    std::unique_lock lock(mutex_);
    return condition_.wait_for(lock, std::chrono::seconds(10),
                               [&] { return entered_; });
  }
  void release() {
    std::lock_guard lock(mutex_);
    released_ = true;
    condition_.notify_all();
  }

private:
  std::mutex mutex_;
  std::condition_variable condition_;
  bool armed_ = false, entered_ = false, released_ = false, fail_ = false;
};

struct ReleaseBoundary final {
  std::shared_ptr<NativeBoundary> boundary;
  ~ReleaseBoundary() { boundary->release(); }
};

class Checkpoint final : public mx::Primitive {
public:
  Checkpoint(mx::Stream stream, std::shared_ptr<NativeBoundary> boundary)
      : Primitive(stream), boundary_(std::move(boundary)) {}
  void eval_cpu(const std::vector<mx::array> &,
                std::vector<mx::array> &) override {
    throw std::runtime_error("test checkpoint must execute on GPU");
  }
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override {
    boundary_->intercept();
    outputs[0].copy_shared_buffer(inputs[0]);
  }
  const char *name() const override { return "HyleTestNativeCheckpoint"; }

private:
  std::shared_ptr<NativeBoundary> boundary_;
};

hyle::Result<a::Program>
prepare(const a::Context &context,
        const std::shared_ptr<NativeBoundary> &boundary) {
  const std::array specs{a::InputSpec{{4}, mx::float32}};
  return context.compile(
      [boundary](const auto &inputs, mx::Stream stream) {
        auto prefix = mx::square(inputs[0], stream);
        return std::vector{mx::array(
            prefix.shape(), prefix.dtype(),
            std::make_shared<Checkpoint>(stream, boundary), {prefix})};
      },
      specs);
}

struct DrainLeaseBoundary final {
  a::Context &context;
  std::shared_ptr<NativeBoundary> boundary;
  ~DrainLeaseBoundary() {
    boundary->release();
    context.close().IgnoreError();
  }
};

TEST(AppleLeaseLifetime,
     QueuedCancellationAndAbandonmentRetireTwoNativeBlocks) {
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  {
    auto boundary = std::make_shared<NativeBoundary>();
    auto context = a::Context::create().value();
    auto memory = a::MemoryResource::create(context).value();
    auto pool =
        c::LeasePool::create(memory.ref(), 1024,
                             {.max_blocks = 2, .allocator = stats.get()})
            .value();
    DrainLeaseBoundary drain{context, boundary};
    const auto desc = c::TensorDesc::dense(c::DataType::kFloat32,
                                           std::array<std::int64_t, 1>{4})
                          .value();
    auto first_write = pool.allocate(16).value(),
         next_write = pool.allocate(16).value();
    const std::array<float, 4> a_values{1, 2, 3, 4}, b_values{5, 6, 7, 8};
    ASSERT_TRUE(
        first_write.view()->write(std::as_bytes(std::span(a_values))).ok());
    ASSERT_TRUE(
        next_write.view()->write(std::as_bytes(std::span(b_values))).ok());
    auto first_source = first_write.publish().value(),
         next_source = next_write.publish().value();
    auto program = prepare(context, boundary).value();
    auto first =
        a::LeasedInvocation::create(program, memory, {.allocator = stats.get()})
            .value();
    auto next =
        a::LeasedInvocation::create(program, memory, {.allocator = stats.get()})
            .value();
    ASSERT_TRUE(
        first.bind(std::array{a::LeasedInput{desc, first_source}}).ok());
    ASSERT_TRUE(next.bind(std::array{a::LeasedInput{desc, next_source}}).ok());
    boundary->arm();
    auto active = first.submit().value();
    ASSERT_TRUE(boundary->entered());
    auto queued = next.submit().value();
    ASSERT_TRUE(queued.request_cancel().ok());
    EXPECT_FALSE(queued.result());
    EXPECT_EQ(pool.stats().active_uses, 2U);
    active = {};
    first = {};
    first_source.reset();
    next_source.reset();
    EXPECT_EQ(pool.stats().live_blocks, 2U);
    EXPECT_FALSE(pool.allocate(16));
    boundary->release();
    auto cancelled = queued.wait();
    ASSERT_FALSE(cancelled);
    EXPECT_TRUE(cancelled.error().IsCancelled());
    EXPECT_TRUE(next.drain().IsCancelled());
    EXPECT_FALSE(next.in_flight());
    queued = {};
    next = {};
    program = {};
    EXPECT_TRUE(context.close().ok());
    EXPECT_EQ(pool.stats().active_uses, 0U);
    EXPECT_EQ(pool.stats().live_blocks, 0U);
    EXPECT_TRUE(pool.allocate(16));
    EXPECT_TRUE(pool.close().ok());
  }
  EXPECT_TRUE(stats.balanced());
}

TEST(AppleLeaseLifetime,
     LateNativeFailureCannotReleaseAbandonedLeaseBeforePrefixDrain) {
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  {
    auto boundary = std::make_shared<NativeBoundary>();
    auto context = a::Context::create().value();
    auto memory = a::MemoryResource::create(context).value();
    auto pool =
        c::LeasePool::create(memory.ref(), 512,
                             {.max_blocks = 1, .allocator = stats.get()})
            .value();
    DrainLeaseBoundary drain{context, boundary};
    const auto desc = c::TensorDesc::dense(c::DataType::kFloat32,
                                           std::array<std::int64_t, 1>{4})
                          .value();
    auto write = pool.allocate(16).value();
    const std::array<float, 4> values{1, 2, 3, 4};
    ASSERT_TRUE(write.view()->write(std::as_bytes(std::span(values))).ok());
    auto source = write.publish().value();
    auto program = prepare(context, boundary).value();
    auto run =
        a::LeasedInvocation::create(program, memory, {.allocator = stats.get()})
            .value();
    ASSERT_TRUE(run.bind(std::array{a::LeasedInput{desc, source}}).ok());
    boundary->arm(true);
    auto done = run.submit().value();
    ASSERT_TRUE(boundary->entered());
    EXPECT_FALSE(done.result());
    done = {};
    run = {};
    program = {};
    source.reset();
    EXPECT_EQ(pool.stats().active_uses, 1U);
    EXPECT_EQ(pool.stats().live_blocks, 1U);
    EXPECT_FALSE(pool.allocate(16));
    boundary->release();
    auto closed = context.close();
    EXPECT_TRUE(closed.IsInternal());
    EXPECT_NE(closed.message().find("injected MLX native boundary failure"),
              std::string_view::npos);
    EXPECT_EQ(pool.stats().active_uses, 0U);
    EXPECT_EQ(pool.stats().live_blocks, 0U);
  }
  EXPECT_TRUE(stats.balanced());
}

TEST(AppleLeaseLifetime, ZeroColdMetadataQuotaRejectsThenSameSourceComputes) {
  auto stats =
      hyle::mem::StatsAllocator::create(hyle::mem::system_allocator()).value();
  {
    auto boundary = std::make_shared<NativeBoundary>();
    auto context = a::Context::create().value();
    auto memory = a::MemoryResource::create(context).value();
    auto pool =
        c::LeasePool::create(memory.ref(), 512,
                             {.max_blocks = 1, .allocator = stats.get()})
            .value();
    DrainLeaseBoundary drain{context, boundary};
    const auto desc = c::TensorDesc::dense(c::DataType::kFloat32,
                                           std::array<std::int64_t, 1>{4})
                          .value();
    auto write = pool.allocate(16).value();
    const std::array<float, 4> values{1, 2, 3, 4};
    ASSERT_TRUE(write.view()->write(std::as_bytes(std::span(values))).ok());
    auto source = write.publish().value();
    auto program = prepare(context, boundary).value();
    hyle::mem::QuotaAllocator quota(stats.get(), 0);
    auto denied = a::LeasedInvocation::create(program, memory,
                                              {.allocator = quota.get()});
    ASSERT_FALSE(denied);
    EXPECT_TRUE(denied.error().IsResourceExhausted());
    EXPECT_TRUE(source.valid());
    auto run =
        a::LeasedInvocation::create(program, memory, {.allocator = stats.get()})
            .value();
    ASSERT_TRUE(run.bind(std::array{a::LeasedInput{desc, source}}).ok());
    auto done = run.submit().value();
    auto result = done.wait();
    ASSERT_TRUE(result) << result.error().message();
    EXPECT_FLOAT_EQ(result->get()->arrays()[0].data<float>()[3], 16.f);
    EXPECT_TRUE(run.drain().ok());
    EXPECT_EQ(pool.stats().active_uses, 0U);
  }
  EXPECT_TRUE(stats.balanced());
}

TEST(AppleLifetime, QueuedCancellationAndAbandonmentDrainBeforeClose) {
  auto boundary = std::make_shared<NativeBoundary>();
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  ReleaseBoundary release{boundary}; // release before Context destruction
  auto program = prepare(*context, boundary);
  ASSERT_TRUE(program) << program.error().message();
  auto first = program->create_invocation();
  auto queued = program->create_invocation();
  ASSERT_TRUE(first && queued);
  std::vector<mx::array> first_inputs{mx::array({1.f, 2.f, 3.f, 4.f})};
  std::weak_ptr<mx::array::Data> input_owner =
      first_inputs[0].data_shared_ptr();
  const std::array queued_inputs{mx::array({5.f, 6.f, 7.f, 8.f})};
  ASSERT_TRUE(first->bind(first_inputs).ok());
  ASSERT_TRUE(queued->bind(queued_inputs).ok());
  boundary->arm();
  auto active = first->submit();
  ASSERT_TRUE(active);
  ASSERT_TRUE(boundary->entered());
  auto cancelled = queued->submit();
  ASSERT_TRUE(cancelled);
  EXPECT_TRUE(cancelled->request_cancel().ok());
  EXPECT_FALSE(cancelled->result());
  EXPECT_FALSE(queued->bind(queued_inputs).ok());
  active = c::OwnedResult<a::Results>{};
  first = a::Invocation{};
  first_inputs.clear();
  EXPECT_FALSE(input_owner.expired());
  std::atomic<bool> close_started{false}, closed{false};
  hyle::Status close_status;
  std::thread closer([&] {
    close_started.store(true, std::memory_order_release);
    close_status = context->close();
    closed.store(true, std::memory_order_release);
  });
  while (!close_started.load(std::memory_order_acquire))
    std::this_thread::yield();
  EXPECT_FALSE(closed.load(std::memory_order_acquire));
  boundary->release();
  closer.join();
  EXPECT_TRUE(close_status.ok());
  auto result = cancelled->wait();
  ASSERT_FALSE(result);
  EXPECT_TRUE(result.error().IsCancelled());
  EXPECT_FALSE(queued->submit());
  EXPECT_TRUE(input_owner.expired());
}

TEST(AppleLifetime, InFlightCancellationWaitsForNativeDrainAndRetiresOnlySlot) {
  auto boundary = std::make_shared<NativeBoundary>();
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  ReleaseBoundary release{boundary};
  auto program = prepare(*context, boundary);
  ASSERT_TRUE(program) << program.error().message();
  auto invocation = program->create_invocation();
  ASSERT_TRUE(invocation);
  const std::array inputs{mx::array({1.f, 2.f, 3.f, 4.f})};
  ASSERT_TRUE(invocation->bind(inputs).ok());
  boundary->arm();
  auto work = invocation->submit();
  ASSERT_TRUE(work);
  ASSERT_TRUE(boundary->entered());
  EXPECT_TRUE(work->request_cancel().ok());
  auto pending = work->poll();
  ASSERT_TRUE(pending);
  EXPECT_EQ(*pending, c::CompletionPoll::kPending);
  EXPECT_FALSE(work->result());
  boundary->release();
  auto result = work->wait();
  ASSERT_FALSE(result);
  EXPECT_TRUE(result.error().IsCancelled());
  EXPECT_FALSE(invocation->bind(inputs).ok());
  auto another = program->create_invocation();
  ASSERT_TRUE(another);
  ASSERT_TRUE(another->bind(inputs).ok());
  auto healthy = another->submit();
  ASSERT_TRUE(healthy);
  auto output = healthy->wait();
  ASSERT_TRUE(output);
  EXPECT_FLOAT_EQ(output->get()->arrays()[0].data<float>()[3], 16.f);
  EXPECT_TRUE(context->close().ok());
}

TEST(AppleLifetime, LateNativeFailureDrainsAndFailsAlreadyQueuedWork) {
  auto boundary = std::make_shared<NativeBoundary>();
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  ReleaseBoundary release{boundary};
  auto program = prepare(*context, boundary);
  ASSERT_TRUE(program) << program.error().message();
  auto first = program->create_invocation();
  auto second = program->create_invocation();
  ASSERT_TRUE(first && second);
  const std::array inputs{mx::array({1.f, 2.f, 3.f, 4.f})};
  ASSERT_TRUE(first->bind(inputs).ok());
  ASSERT_TRUE(second->bind(inputs).ok());
  boundary->arm(true);
  auto active = first->submit();
  ASSERT_TRUE(active);
  ASSERT_TRUE(boundary->entered());
  auto queued = second->submit();
  ASSERT_TRUE(queued);
  EXPECT_FALSE(active->result());
  boundary->release();
  auto failed = active->wait();
  ASSERT_FALSE(failed);
  EXPECT_TRUE(failed.error().IsInternal());
  auto also_failed = queued->wait();
  ASSERT_FALSE(also_failed);
  EXPECT_EQ(also_failed.error().message(), failed.error().message());
  EXPECT_FALSE(first->submit());
  auto later = program->create_invocation();
  ASSERT_TRUE(later);
  ASSERT_TRUE(later->bind(inputs).ok());
  EXPECT_FALSE(later->submit());
  EXPECT_FALSE(context->close().ok());
}

TEST(AppleLifetime, IndependentContextsOwnIndependentNativeSubmissionThreads) {
  auto left_boundary = std::make_shared<NativeBoundary>();
  auto right_boundary = std::make_shared<NativeBoundary>();
  auto left = a::Context::create();
  auto right = a::Context::create();
  ASSERT_TRUE(left && right);
  ReleaseBoundary release_left{left_boundary}, release_right{right_boundary};
  auto left_program = prepare(*left, left_boundary);
  auto right_program = prepare(*right, right_boundary);
  ASSERT_TRUE(left_program && right_program);
  auto a_slot = left_program->create_invocation();
  auto b_slot = right_program->create_invocation();
  ASSERT_TRUE(a_slot && b_slot);
  const std::array inputs{mx::array({1.f, 2.f, 3.f, 4.f})};
  ASSERT_TRUE(a_slot->bind(inputs).ok());
  ASSERT_TRUE(b_slot->bind(inputs).ok());
  left_boundary->arm();
  right_boundary->arm();
  auto a_run = a_slot->submit();
  ASSERT_TRUE(a_run);
  ASSERT_TRUE(left_boundary->entered());
  auto b_run = b_slot->submit();
  ASSERT_TRUE(b_run);
  EXPECT_TRUE(right_boundary->entered()); // left is still held
  left_boundary->release();
  right_boundary->release();
  EXPECT_TRUE(a_run->wait());
  EXPECT_TRUE(b_run->wait());
  EXPECT_TRUE(left->close().ok());
  EXPECT_TRUE(right->close().ok());
}

TEST(AppleLifetime,
     CheckedFailureRetainsAnAcceptedNativePrefixAfterOwnersGoAway) {
  for (bool scatter : {false, true}) {
    auto boundary = std::make_shared<NativeBoundary>();
    auto context = a::Context::create().value();
    ReleaseBoundary release{boundary};
    auto program = context.compile(
        [boundary, scatter](const auto &inputs, mx::Stream stream)
            -> hyle::Result<std::vector<mx::array>> {
          auto prefix = mx::square(inputs[0], stream);
          auto held = mx::array(prefix.shape(), prefix.dtype(),
                                std::make_shared<Checkpoint>(stream, boundary),
                                {prefix});
          if (scatter)
            return a::lower(
                c::Operation(c::Scatter{.kind = c::ScatterKind::kAdd}),
                std::array{held, inputs[1],
                           mx::slice(inputs[0], {0}, {2}, stream)},
                stream);
          return a::lower(c::Operation(c::Gather{}),
                          std::array{held, inputs[1]}, stream);
        },
        std::array{a::InputSpec{{4}, mx::float32},
                   a::InputSpec{{2}, mx::int32}});
    ASSERT_TRUE(program) << program.error().message();
    auto slot = program->create_invocation().value();
    ASSERT_TRUE(slot.bind(std::array{mx::array({1.f, 2.f, 3.f, 4.f}),
                                     mx::array({0, 4})})
                    .ok());
    boundary->arm();
    auto work = slot.submit();
    ASSERT_TRUE(work) << work.error().message();
    ASSERT_TRUE(boundary->entered());
    EXPECT_FALSE(work->result());
    slot = {};
    *program = {};
    boundary->release();
    EXPECT_TRUE(context.close().ok());
    auto result = work->wait();
    ASSERT_FALSE(result);
    EXPECT_TRUE(result.error().IsOutOfRange());
  }
}

TEST(AppleLifetime, AllocationRefusalsPublishNoPartialProgramOrResult) {
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  auto stats = hyle::mem::StatsAllocator::create(hyle::mem::system_allocator());
  ASSERT_TRUE(stats);
  auto boundary = std::make_shared<NativeBoundary>();
  auto program = prepare(*context, boundary);
  ASSERT_TRUE(program);
  {
    hyle::mem::QuotaAllocator quota(stats->get(), 0);
    auto denied = program->create_invocation(quota.get());
    ASSERT_FALSE(denied);
    EXPECT_TRUE(denied.error().IsResourceExhausted());
  }
  EXPECT_TRUE(stats->balanced());
  const std::array small{a::InputSpec{{4}, mx::float32}};
  auto cold_oom = context->compile(
      [](const auto &, mx::Stream) -> hyle::Result<std::vector<mx::array>> {
        throw std::bad_alloc();
      },
      small);
  ASSERT_FALSE(cold_oom);
  EXPECT_TRUE(cold_oom.error().IsResourceExhausted());
  // The pinned Metal allocator checks maxBufferLength before allocation.
  // This exercises real native capacity refusal without exhausting the host.
  const std::array oversized{a::InputSpec{{1 << 30, 1 << 30}, mx::float32}};
  auto native_limit = context->compile(
      [](const auto &inputs, mx::Stream) { return inputs; }, oversized);
  ASSERT_FALSE(native_limit);
  EXPECT_NE(native_limit.error().message().find("maximum allowed buffer size"),
            std::string_view::npos);
  auto recovered = program->create_invocation();
  ASSERT_TRUE(recovered);
  const std::array inputs{mx::array({1.f, 2.f, 3.f, 4.f})};
  ASSERT_TRUE(recovered->bind(inputs).ok());
  auto work = recovered->submit();
  ASSERT_TRUE(work);
  auto result = work->wait();
  ASSERT_TRUE(result);
  EXPECT_FLOAT_EQ(result->get()->arrays()[0].data<float>()[3], 16.f);
}
} // namespace

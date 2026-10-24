#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

#include "../../src/compute_backend_cuda/native_context.h"
#include "hyle/compute/detail/completion_access.h"

namespace {
namespace c = hyle::compute;
namespace cuda = c::cuda;

struct NativeOwner final {
  std::atomic<bool> retired{false};
  std::atomic<int> &released;
  c::detail::CompletionLifetime lifetime{this, &terminal, &release};

  static hyle::Status observe(void *state, bool, bool *retired) noexcept {
    *retired = static_cast<NativeOwner *>(state)->retired.load(std::memory_order_acquire);
    return *retired ? hyle::Status::Ok()
                    : hyle::Status::Unavailable("native retirement observation failed");
  }
  static void terminal(void *, const hyle::Status &status) noexcept {
    EXPECT_TRUE(status.IsInternal()); // Preserve the original native enqueue failure.
  }
  static void release(void *state) noexcept {
    auto *owner = static_cast<NativeOwner *>(state);
    auto &counter = owner->released;
    delete owner;
    counter.store(1, std::memory_order_release);
  }
};

TEST(CudaNativeRetirement, RootFenceAndObservationErrorsDoNotReleaseNativeOwner) {
  auto context = c::Context::create(cuda::device(0).value()).value();
  auto queue = context.create_queue().value();
  for (bool wait_error : {false, true}) {
    SCOPED_TRACE(wait_error);
    std::atomic<int> released{0};
    auto *owner = new NativeOwner{{false}, released};
    auto sequence = c::detail::QueueSequence::begin(queue).value();
    cuda::detail::defer_native_retirement(sequence, {owner, &NativeOwner::observe});
    sequence.record(c::detail::EnqueueResult::accepted_command(
        hyle::Status::Internal("native graph failed before auxiliary join")));
    auto done = sequence.finish().value();
    ASSERT_TRUE(c::detail::CompletionAccess::attach_lifetime(done, owner->lifetime).ok());
    // The real root stream is empty and its actual event is complete. Only
    // the independently owned native prefix remains unproved.
    ASSERT_TRUE(queue.drain().ok());
    auto poll = done.poll();
    ASSERT_TRUE(poll);
    EXPECT_EQ(*poll, c::CompletionPoll::kPending);
    EXPECT_EQ(released.load(std::memory_order_acquire), 0);
    if (wait_error)
      EXPECT_TRUE(done.wait().IsInternal());
    done = {}; // Observation error or public abandonment uses the same reaper.
    EXPECT_EQ(released.load(std::memory_order_acquire), 0);
    owner->retired.store(true, std::memory_order_release);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (released.load(std::memory_order_acquire) == 0 &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    ASSERT_EQ(released.load(std::memory_order_acquire), 1);
  }
}
} // namespace

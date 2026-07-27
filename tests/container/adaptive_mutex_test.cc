#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>

#include "hyle/container/detail/adaptive_mutex.h"

namespace {

using hyle::container::detail::AdaptiveMutex;

TEST(AdaptiveMutex, SatisfiesUncontendedLockableContract) {
  AdaptiveMutex mutex;
  EXPECT_TRUE(mutex.try_lock());
  EXPECT_FALSE(mutex.try_lock());
  mutex.unlock();
  mutex.lock();
  mutex.unlock();
}

TEST(AdaptiveMutex, ContendedWritersConserveUpdates) {
  constexpr std::size_t kThreads = 8;
  constexpr std::size_t kUpdates = 20'000;
  AdaptiveMutex mutex;
  std::uint64_t value = 0;
  std::atomic<bool> start{false};
  std::array<std::thread, kThreads> threads;

  for (auto &thread : threads) {
    thread = std::thread([&] {
      while (!start.load(std::memory_order_acquire)) {}
      for (std::size_t i = 0; i < kUpdates; ++i) {
        std::lock_guard lock(mutex);
        ++value;
      }
    });
  }
  start.store(true, std::memory_order_release);
  for (auto &thread : threads)
    thread.join();

  EXPECT_EQ(value, kThreads * kUpdates);
}

} // namespace

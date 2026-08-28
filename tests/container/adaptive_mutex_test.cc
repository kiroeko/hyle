#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>

#include "hyle/container/detail/adaptive_mutex.h"
#include "hyle/container/detail/adaptive_rw_lock.h"

namespace {

using hyle::container::detail::AdaptiveMutex;
using hyle::container::detail::AdaptiveRwLock;

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

TEST(AdaptiveRwLock, PendingWriterClosesReaderGate) {
  AdaptiveRwLock lock;
  lock.lock_shared();

  std::atomic<bool> writer_started{false};
  std::atomic<bool> writer_acquired{false};
  std::atomic<bool> release_writer{false};
  std::thread writer([&] {
    writer_started.store(true, std::memory_order_release);
    lock.lock();
    writer_acquired.store(true, std::memory_order_release);
    while (!release_writer.load(std::memory_order_acquire))
      std::this_thread::yield();
    lock.unlock();
  });

  while (!writer_started.load(std::memory_order_acquire))
    std::this_thread::yield();
  for (;;) {
    if (!lock.try_lock_shared()) break;
    lock.unlock_shared();
    std::this_thread::yield();
  }

  // The original reader still owns the lock, so the writer cannot have taken
  // it.  Every failed admission from here is therefore the pending-writer gate.
  EXPECT_FALSE(writer_acquired.load(std::memory_order_acquire));
  for (std::size_t i = 0; i < 1024; ++i) {
    const bool acquired = lock.try_lock_shared();
    EXPECT_FALSE(acquired);
    if (acquired) lock.unlock_shared();
  }

  lock.unlock_shared();
  while (!writer_acquired.load(std::memory_order_acquire))
    std::this_thread::yield();
  release_writer.store(true, std::memory_order_release);
  writer.join();
}

} // namespace

#include "hyle/mem/reclamation.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <thread>
#include <utility>
#include <vector>

namespace hyle::mem {
namespace {

struct RetiredValue {
  int value = 0;
  RetireRecord record;
  std::atomic<std::size_t> *callbacks = nullptr;
};

void reclaim_value(void *state) noexcept {
  auto *value = static_cast<RetiredValue *>(state);
  value->callbacks->fetch_add(1, std::memory_order_relaxed);
}

TEST(Reclamation, GuardProtectsPublishedPointerAndMoves) {
  ReclamationDomain domain(0);
  RetiredValue value{};
  value.value = 42;
  std::atomic<RetiredValue *> published{&value};

  ReclamationGuard guard = domain.read();
  EXPECT_EQ(guard.protect(published)->value, 42);
  ReclamationGuard moved = std::move(guard);
  EXPECT_FALSE(guard.active());
  EXPECT_TRUE(moved.active());
  EXPECT_EQ(moved.protect(published), &value);
  moved.reset();
  EXPECT_FALSE(moved.active());
}

TEST(Reclamation, NonblockingCollectionWaitsForExistingReader) {
  ReclamationDomain domain(0);
  std::atomic<std::size_t> callbacks{0};
  RetiredValue value{};
  value.value = 7;
  value.callbacks = &callbacks;

  auto guard = domain.read();
  domain.retire(value.record, &value, &reclaim_value);
  EXPECT_TRUE(value.record.pending());
  EXPECT_EQ(domain.try_reclaim(), 0u);
  EXPECT_EQ(callbacks.load(std::memory_order_relaxed), 0u);

  guard.reset();
  EXPECT_EQ(domain.try_reclaim(), 1u);
  EXPECT_EQ(callbacks.load(std::memory_order_relaxed), 1u);
  EXPECT_FALSE(value.record.pending());
  EXPECT_EQ(domain.pending(), 0u);
  EXPECT_EQ(domain.reclaimed(), 1u);
}

TEST(Reclamation, NestedReadersKeepTheirOriginalEpoch) {
  ReclamationDomain domain(0);
  std::atomic<std::size_t> callbacks{0};
  RetiredValue value{};
  value.callbacks = &callbacks;
  auto outer = domain.read();
  auto inner = domain.read();
  domain.retire(value.record, &value, &reclaim_value);
  EXPECT_EQ(domain.try_reclaim(), 0u);
  inner.reset();
  EXPECT_EQ(domain.try_reclaim(), 0u);
  outer.reset();
  EXPECT_EQ(domain.try_reclaim(), 1u);
}

TEST(Reclamation, SynchronizeWaitsForExistingReader) {
  ReclamationDomain domain(0);
  std::atomic<std::size_t> callbacks{0};
  RetiredValue value{};
  value.callbacks = &callbacks;
  auto preceding = domain.read();
  domain.retire(value.record, &value, &reclaim_value);

  std::atomic<bool> started{false};
  std::atomic<bool> completed{false};
  std::thread collector([&] {
    started.store(true, std::memory_order_release);
    static_cast<void>(domain.synchronize());
    completed.store(true, std::memory_order_release);
  });
  while (!started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  for (int i = 0; i != 1000; ++i) {
    std::this_thread::yield();
  }
  EXPECT_FALSE(completed.load(std::memory_order_acquire));
  preceding.reset();
  for (int i = 0; i != 10000 &&
                  !completed.load(std::memory_order_acquire);
       ++i) {
    std::this_thread::yield();
  }
  EXPECT_TRUE(completed.load(std::memory_order_acquire));
  collector.join();
  EXPECT_EQ(callbacks.load(std::memory_order_relaxed), 1u);
}

TEST(Reclamation, ThresholdStartsNonblockingGracePeriod) {
  ReclamationDomain domain(2);
  std::atomic<std::size_t> callbacks{0};
  RetiredValue first{};
  first.callbacks = &callbacks;
  RetiredValue second{};
  second.callbacks = &callbacks;

  domain.retire(first.record, &first, &reclaim_value);
  EXPECT_EQ(callbacks.load(std::memory_order_relaxed), 0u);
  domain.retire(second.record, &second, &reclaim_value);
  EXPECT_EQ(callbacks.load(std::memory_order_relaxed), 2u);
  EXPECT_EQ(domain.pending(), 0u);
}

struct RecursiveState {
  ReclamationDomain *domain = nullptr;
  RetireRecord first;
  RetireRecord second;
  std::atomic<std::size_t> callbacks{0};
};

void reclaim_second(void *state) noexcept {
  static_cast<RecursiveState *>(state)->callbacks.fetch_add(
      1, std::memory_order_relaxed);
}

void reclaim_first(void *state) noexcept {
  auto *recursive = static_cast<RecursiveState *>(state);
  recursive->callbacks.fetch_add(1, std::memory_order_relaxed);
  recursive->domain->retire(recursive->second, recursive, &reclaim_second);
}

TEST(Reclamation, DrainHandlesCallbackThatRetiresMoreWork) {
  ReclamationDomain domain(0);
  RecursiveState state{};
  state.domain = &domain;
  domain.retire(state.first, &state, &reclaim_first);
  EXPECT_EQ(domain.drain(), 2u);
  EXPECT_EQ(state.callbacks.load(std::memory_order_relaxed), 2u);
  EXPECT_EQ(domain.pending(), 0u);
}

TEST(Reclamation, ThresholdCollectionAllowsCallbackToRetireMoreWork) {
  ReclamationDomain domain(1);
  RecursiveState state{};
  state.domain = &domain;
  domain.retire(state.first, &state, &reclaim_first);
  EXPECT_EQ(state.callbacks.load(std::memory_order_relaxed), 1u);
  EXPECT_EQ(domain.pending(), 1u);
  EXPECT_EQ(domain.drain(), 1u);
  EXPECT_EQ(state.callbacks.load(std::memory_order_relaxed), 2u);
}

TEST(Reclamation, ConcurrentReadersAndWritersReclaimExactlyOnce) {
  constexpr std::size_t kReaders = 8;
  constexpr std::size_t kValues = 512;
  ReclamationDomain domain(16);
  std::atomic<std::size_t> callbacks{0};
  std::vector<RetiredValue> values(kValues);
  for (std::size_t i = 0; i != kValues; ++i) {
    values[i].value = static_cast<int>(i);
    values[i].callbacks = &callbacks;
  }
  std::atomic<RetiredValue *> published{&values[0]};
  std::atomic<bool> stop{false};
  std::vector<std::thread> readers;
  readers.reserve(kReaders);
  for (std::size_t i = 0; i != kReaders; ++i) {
    readers.emplace_back([&] {
      std::size_t checksum = 0;
      while (!stop.load(std::memory_order_acquire)) {
        auto guard = domain.read();
        checksum += static_cast<std::size_t>(guard.protect(published)->value);
      }
      EXPECT_NE(checksum, static_cast<std::size_t>(-1));
    });
  }

  for (std::size_t i = 1; i != kValues; ++i) {
    RetiredValue *old =
        published.exchange(&values[i], std::memory_order_acq_rel);
    domain.retire(old->record, old, &reclaim_value);
  }
  stop.store(true, std::memory_order_release);
  for (std::thread &reader : readers) {
    reader.join();
  }
  static_cast<void>(domain.drain());
  EXPECT_EQ(callbacks.load(std::memory_order_relaxed), kValues - 1);
}

TEST(Reclamation, MoreReadersThanExclusiveSlotsUseSafeFallback) {
  constexpr std::size_t kReaders =
      detail::kReclamationReaderSlots + detail::kReclamationFallbackStripes;
  ReclamationDomain domain(0);
  std::atomic<std::size_t> callbacks{0};
  RetiredValue value{};
  value.callbacks = &callbacks;
  std::atomic<std::size_t> ready{0};
  std::atomic<bool> release{false};
  std::vector<std::thread> readers;
  readers.reserve(kReaders);
  for (std::size_t i = 0; i != kReaders; ++i) {
    readers.emplace_back([&] {
      auto guard = domain.read();
      ready.fetch_add(1, std::memory_order_release);
      while (!release.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
    });
  }
  while (ready.load(std::memory_order_acquire) != kReaders) {
    std::this_thread::yield();
  }
  domain.retire(value.record, &value, &reclaim_value);
  EXPECT_EQ(domain.try_reclaim(), 0u);
  release.store(true, std::memory_order_release);
  for (std::thread &reader : readers) {
    reader.join();
  }
  EXPECT_EQ(domain.synchronize(), 1u);
}

TEST(Reclamation, DefaultDomainHasStableProcessIdentity) {
  EXPECT_EQ(&default_reclamation_domain(), &default_reclamation_domain());
}

TEST(ReclamationDeathTest, RejectsDoubleRetire) {
  EXPECT_DEATH(
      {
        ReclamationDomain domain(0);
        RetireRecord record;
        domain.retire(record, nullptr, +[](void *) noexcept {});
        domain.retire(record, nullptr, +[](void *) noexcept {});
      },
      "");
}

TEST(ReclamationDeathTest, RejectsDestructionWithActiveReader) {
  EXPECT_DEATH(
      {
        auto *domain = new ReclamationDomain;
        auto guard = domain->read();
        delete domain;
        static_cast<void>(guard);
      },
      "");
}

} // namespace
} // namespace hyle::mem

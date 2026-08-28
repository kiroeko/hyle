#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "hyle/container/bounded_queue.h"
#include "hyle/mem/allocator.h"
#include "hyle/platform/thread.h"

using hyle::container::BoundedQueue;
using hyle::mem::Allocator;
using hyle::mem::system_allocator;

namespace {

struct alignas(128) TriviallyCopyableNonAssignable {
  std::uint64_t value;

  TriviallyCopyableNonAssignable() = delete;
  explicit TriviallyCopyableNonAssignable(std::uint64_t init) : value(init) {}

  TriviallyCopyableNonAssignable& operator=(const TriviallyCopyableNonAssignable&) =
      delete;
};

static_assert(std::is_trivially_copyable_v<TriviallyCopyableNonAssignable>);
static_assert(alignof(TriviallyCopyableNonAssignable) == 128);
static_assert(!std::is_default_constructible_v<TriviallyCopyableNonAssignable>);
static_assert(!std::is_copy_assignable_v<TriviallyCopyableNonAssignable>);
static_assert(alignof(BoundedQueue<std::uint64_t>) ==
              hyle::core::kMaxDestructiveInterferenceSize);
static_assert(sizeof(BoundedQueue<std::uint64_t>) %
                  hyle::core::kMaxDestructiveInterferenceSize ==
              0);

struct RecordingAllocatorState {
  void *allocated_pointer = nullptr;
  std::size_t allocated_size = 0;
  std::size_t allocated_alignment = 0;
  void *deallocated_pointer = nullptr;
  std::size_t deallocated_size = 0;
  std::size_t deallocated_alignment = 0;
};

struct RecordingAllocator {
  RecordingAllocatorState *state;

  void *allocate(std::size_t size, std::size_t alignment) const noexcept {
    state->allocated_size = size;
    state->allocated_alignment = alignment;
    state->allocated_pointer = system_allocator().allocate(size, alignment);
    return state->allocated_pointer;
  }
  void deallocate(void *pointer, std::size_t size,
                  std::size_t alignment) const noexcept {
    state->deallocated_pointer = pointer;
    state->deallocated_size = size;
    state->deallocated_alignment = alignment;
    system_allocator().deallocate(pointer, size, alignment);
  }
};

struct MoveOnlyNonAssignable {
  explicit MoveOnlyNonAssignable(int input) noexcept : value(input) {
    ++alive;
    ++constructed;
  }
  MoveOnlyNonAssignable(const MoveOnlyNonAssignable&) = delete;
  MoveOnlyNonAssignable& operator=(const MoveOnlyNonAssignable&) = delete;
  MoveOnlyNonAssignable(MoveOnlyNonAssignable&& other) noexcept
      : value(std::exchange(other.value, -1)) {
    ++alive;
    ++constructed;
  }
  MoveOnlyNonAssignable& operator=(MoveOnlyNonAssignable&&) = delete;
  ~MoveOnlyNonAssignable() noexcept { --alive; }

  static inline int alive = 0;
  static inline int constructed = 0;
  int value;
};

struct ConcurrentMoveOnly {
  explicit ConcurrentMoveOnly(std::uint64_t input) noexcept : value(input) {
    alive.fetch_add(1, std::memory_order_relaxed);
  }
  ConcurrentMoveOnly(const ConcurrentMoveOnly&) = delete;
  ConcurrentMoveOnly& operator=(const ConcurrentMoveOnly&) = delete;
  ConcurrentMoveOnly(ConcurrentMoveOnly&& other) noexcept
      : value(std::exchange(other.value, 0)) {
    alive.fetch_add(1, std::memory_order_relaxed);
  }
  ConcurrentMoveOnly& operator=(ConcurrentMoveOnly&&) = delete;
  ~ConcurrentMoveOnly() noexcept {
    alive.fetch_sub(1, std::memory_order_relaxed);
  }

  static inline std::atomic<int> alive{0};
  std::uint64_t value;
};

}  // namespace

TEST(BoundedQueue, SingleThreadFifo) {
  BoundedQueue<int> q(4);  // default = compile-time SystemAllocator
  EXPECT_EQ(q.capacity(), 4u);
  EXPECT_TRUE(q.try_push(10));
  EXPECT_TRUE(q.try_push(20));
  int v = 0;
  ASSERT_TRUE(q.try_pop(v));
  EXPECT_EQ(v, 10);
  ASSERT_TRUE(q.try_pop(v));
  EXPECT_EQ(v, 20);
}

TEST(BoundedQueue, RejectsWhenFull) {
  BoundedQueue<int> q(2);
  EXPECT_TRUE(q.try_push(1));
  EXPECT_TRUE(q.try_push(2));
  EXPECT_FALSE(q.try_push(3));  // full (capacity 2)
}

TEST(BoundedQueue, RejectsWhenEmpty) {
  BoundedQueue<int> q(2);
  int v = 99;
  EXPECT_FALSE(q.try_pop(v));
  EXPECT_EQ(v, 99);
}

TEST(BoundedQueue, WrapsAroundManyLaps) {
  BoundedQueue<int> q(4);
  // Push/pop far more than capacity to exercise sequence wrap-around.
  for (int i = 0; i < 1000; ++i) {
    ASSERT_TRUE(q.try_push(i));
    int v = 0;
    ASSERT_TRUE(q.try_pop(v));
    EXPECT_EQ(v, i);
  }
}

TEST(BoundedQueue, PointerPayload) {
  BoundedQueue<void*> q(8);
  int a = 1, b = 2;
  EXPECT_TRUE(q.try_push(&a));
  EXPECT_TRUE(q.try_push(&b));
  void* p = nullptr;
  ASSERT_TRUE(q.try_pop(p));
  EXPECT_EQ(p, &a);
}

TEST(BoundedQueue, DenseCellsUseTargetFamilySpacing) {
  constexpr std::size_t kCapacity = 8;
  RecordingAllocatorState state;
  {
    BoundedQueue<std::uint64_t, RecordingAllocator> queue(
        RecordingAllocator{&state}, kCapacity);
    ASSERT_TRUE(queue.valid());
    EXPECT_EQ(state.allocated_alignment,
              hyle::core::kDestructiveInterferenceSize);
    EXPECT_EQ(state.allocated_size,
              kCapacity * hyle::core::kDestructiveInterferenceSize);
  }
  EXPECT_EQ(state.deallocated_pointer, state.allocated_pointer);
  EXPECT_EQ(state.deallocated_alignment, state.allocated_alignment);
  EXPECT_EQ(state.deallocated_size, state.allocated_size);
}

TEST(BoundedQueue, TriviallyCopyablePayloadNeedNotBeAssignable) {
  BoundedQueue<TriviallyCopyableNonAssignable> q(2);
  ASSERT_TRUE(q.try_push(TriviallyCopyableNonAssignable{42}));
  TriviallyCopyableNonAssignable out{0};
  ASSERT_TRUE(q.try_pop(out));
  EXPECT_EQ(out.value, 42u);
}

TEST(BoundedQueue, OwnsMoveOnlyNonAssignableSlotLifetimes) {
  EXPECT_EQ(MoveOnlyNonAssignable::alive, 0);
  MoveOnlyNonAssignable::constructed = 0;
  {
    BoundedQueue<MoveOnlyNonAssignable> queue(2);
    ASSERT_TRUE(queue.try_emplace(10));
    ASSERT_TRUE(queue.try_emplace(20));
    EXPECT_EQ(MoveOnlyNonAssignable::alive, 2);

    const int constructed_before_full = MoveOnlyNonAssignable::constructed;
    EXPECT_FALSE(queue.try_emplace(30));
    EXPECT_EQ(MoveOnlyNonAssignable::constructed, constructed_before_full);

    int sum = 0;
    ASSERT_TRUE(queue.try_consume(
        [&sum](MoveOnlyNonAssignable& item) noexcept { sum += item.value; }));
    EXPECT_EQ(sum, 10);
    EXPECT_EQ(MoveOnlyNonAssignable::alive, 1);
    // The remaining live slot is destroyed by the quiescent queue destructor.
  }
  EXPECT_EQ(MoveOnlyNonAssignable::alive, 0);
}

TEST(BoundedQueue, MoveAssignablePayloadUsesNaturalPopShape) {
  BoundedQueue<std::unique_ptr<int>> queue(2);
  ASSERT_TRUE(queue.try_push(std::make_unique<int>(7)));
  std::unique_ptr<int> output;
  ASSERT_TRUE(queue.try_pop(output));
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(*output, 7);

  ASSERT_TRUE(queue.try_emplace(new int(11)));
  ASSERT_TRUE(queue.try_emplace(new int(13)));
  auto rejected = std::make_unique<int>(17);
  EXPECT_FALSE(queue.try_push(std::move(rejected)));
  EXPECT_NE(rejected, nullptr);
}

// Lock-free correctness gate (Windows stress; TSan deferred to Linux CI).
// N producers each push a disjoint id range; N consumers drain. The union of
// popped values must equal exactly the pushed set -- no loss, no duplication.
// Uses the OPT-IN runtime mem::Allocator variant (covers the scheduler's path).
TEST(BoundedQueue, MpmcStressConservation) {
  constexpr int kProducers = 4;
  constexpr int kConsumers = 4;
  constexpr int kPerProducer = 100000;
  constexpr int kTotal = kProducers * kPerProducer;

  BoundedQueue<std::uint64_t, Allocator> q(system_allocator(), 1024);
  std::atomic<int> produced_done{0};
  std::vector<std::vector<std::uint64_t>> got(kConsumers);
  std::atomic<int> popped_count{0};

  struct ProdArgs {
    BoundedQueue<std::uint64_t, Allocator>* q;
    int base;
    std::atomic<int>* done;
  };
  struct ConsArgs {
    BoundedQueue<std::uint64_t, Allocator>* q;
    std::vector<std::uint64_t>* out;
    std::atomic<int>* done_producers;
    std::atomic<int>* popped;
  };

  std::vector<ProdArgs> pa(kProducers);
  std::vector<ConsArgs> ca(kConsumers);
  std::vector<hyle::platform::Thread> threads;

  for (int i = 0; i < kProducers; ++i) {
    // ids in [1 + i*kPerProducer, ...] -- disjoint, nonzero.
    pa[i] = ProdArgs{&q, 1 + i * kPerProducer, &produced_done};
    auto r = hyle::platform::Thread::spawn(
        [](void* a) {
          auto* x = static_cast<ProdArgs*>(a);
          for (int k = 0; k < kPerProducer; ++k) {
            std::uint64_t id = static_cast<std::uint64_t>(x->base + k);
            while (!x->q->try_push(id)) hyle::platform::this_thread::yield();
          }
          x->done->fetch_add(1, std::memory_order_acq_rel);
        },
        &pa[i]);
    ASSERT_TRUE(r.has_value());
    threads.push_back(std::move(*r));
  }

  for (int i = 0; i < kConsumers; ++i) {
    got[i].reserve(kTotal / kConsumers + kPerProducer);
    ca[i] = ConsArgs{&q, &got[i], &produced_done, &popped_count};
    auto r = hyle::platform::Thread::spawn(
        [](void* a) {
          auto* x = static_cast<ConsArgs*>(a);
          for (;;) {
            std::uint64_t v = 0;
            if (x->q->try_pop(v)) {
              x->out->push_back(v);
              x->popped->fetch_add(1, std::memory_order_acq_rel);
            } else if (x->done_producers->load(std::memory_order_acquire) ==
                           kProducers &&
                       x->popped->load(std::memory_order_acquire) >= kTotal) {
              break;  // all produced and all drained
            } else {
              hyle::platform::this_thread::yield();
            }
          }
        },
        &ca[i]);
    ASSERT_TRUE(r.has_value());
    threads.push_back(std::move(*r));
  }

  for (auto& t : threads) t.join();

  // Conservation: every produced id appears exactly once across all consumers.
  std::unordered_set<std::uint64_t> seen;
  seen.reserve(kTotal * 2);
  std::size_t count = 0;
  for (auto& v : got) {
    for (std::uint64_t id : v) {
      EXPECT_TRUE(seen.insert(id).second) << "duplicate id " << id;
      ++count;
    }
  }
  EXPECT_EQ(count, static_cast<std::size_t>(kTotal));
  EXPECT_EQ(seen.size(), static_cast<std::size_t>(kTotal));
}

TEST(BoundedQueue, MoveOnlyMpmcStressBalancesLifetimeAndConservesValues) {
  constexpr int kProducers = 2;
  constexpr int kConsumers = 2;
  constexpr int kPerProducer = 25000;
  constexpr int kTotal = kProducers * kPerProducer;

  ConcurrentMoveOnly::alive.store(0, std::memory_order_relaxed);
  BoundedQueue<ConcurrentMoveOnly, Allocator> queue(system_allocator(), 1024);
  std::atomic<int> producers_done{0};
  std::atomic<int> popped_count{0};
  std::vector<std::vector<std::uint64_t>> received(kConsumers);
  for (auto& values : received) values.reserve(kTotal);

  std::vector<std::thread> threads;
  for (int producer = 0; producer < kProducers; ++producer) {
    threads.emplace_back([&, producer] {
      const std::uint64_t base =
          1 + static_cast<std::uint64_t>(producer) * kPerProducer;
      for (int index = 0; index < kPerProducer; ++index) {
        while (!queue.try_emplace(base + static_cast<std::uint64_t>(index)))
          std::this_thread::yield();
      }
      producers_done.fetch_add(1, std::memory_order_release);
    });
  }
  for (int consumer = 0; consumer < kConsumers; ++consumer) {
    threads.emplace_back([&, consumer] {
      for (;;) {
        if (queue.try_consume([&](ConcurrentMoveOnly& item) noexcept {
              received[consumer].push_back(item.value);
              popped_count.fetch_add(1, std::memory_order_relaxed);
            })) {
          continue;
        }
        if (producers_done.load(std::memory_order_acquire) == kProducers &&
            popped_count.load(std::memory_order_relaxed) == kTotal)
          break;
        std::this_thread::yield();
      }
    });
  }
  for (auto& thread : threads) thread.join();

  std::unordered_set<std::uint64_t> seen;
  seen.reserve(kTotal * 2);
  for (const auto& values : received)
    for (std::uint64_t value : values) EXPECT_TRUE(seen.insert(value).second);
  EXPECT_EQ(seen.size(), static_cast<std::size_t>(kTotal));
  EXPECT_EQ(ConcurrentMoveOnly::alive.load(std::memory_order_relaxed), 0);
}

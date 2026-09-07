#include "hyle/graph/data_parallel.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

namespace hyle::graph {
namespace {

using mem::AllocatorRef;
using mem::system_allocator;
using sched::Fiber;
using sched::kDefaultStackSize;
using sched::Scheduler;
using sched::Task;

// Fiber-driver harness: run `body(alloc)` (a coroutine returning int) on a Scheduler.
struct Drive {
  Task<int> task;
  std::atomic<bool> finished{false};
};
void drive_entry(void* p) noexcept {
  auto* d = static_cast<Drive*>(p);
  d->task.resume();
  d->finished.store(true, std::memory_order_release);
}
template <class Make>
int run(Make make, int workers = 4, std::size_t queue_capacity = 256) {
  Scheduler sched(static_cast<unsigned>(workers), system_allocator(), queue_capacity);
  auto d = std::make_unique<Drive>();
  d->task = make(&sched);
  Fiber f(kDefaultStackSize, &drive_entry, d.get());
  sched.spawn(&f);
  sched.drain();
  sched.shutdown();
  EXPECT_TRUE(d->finished.load());
  return d->task.done() ? d->task.result() : -999;
}

// ---- parallel_for ------------------------------------------------------------
// Body: out[i] = i*i for the block. ctx points at the output buffer.
Status square_body(void* ctx, std::size_t b, std::size_t e) noexcept {
  auto* out = static_cast<std::int64_t*>(ctx);
  for (std::size_t i = b; i < e; ++i) out[i] = static_cast<std::int64_t>(i) * static_cast<std::int64_t>(i);
  return Status();
}

Task<int> for_coro(mem::AllocatorRef alloc, Scheduler* sched, std::int64_t* out, std::size_t n,
                   std::size_t grain) {
  Status s = parallel_for(*sched, alloc, BlockedRange(0, n, grain), &square_body, out);
  co_return s.ok() ? 0 : -1;
}

TEST(DataParallel, ForFillsRangeMatchesSerial) {
  constexpr std::size_t kN = 10000;
  std::vector<std::int64_t> out(kN, -1);
  int rc = run([&](Scheduler* s) {
    return for_coro(system_allocator(), s, out.data(), kN, 64);
  });
  EXPECT_EQ(rc, 0);
  for (std::size_t i = 0; i < kN; ++i)
    ASSERT_EQ(out[i], static_cast<std::int64_t>(i) * static_cast<std::int64_t>(i)) << "at " << i;
}

TEST(DataParallel, ForAutoGrain) {
  constexpr std::size_t kN = 5000;
  std::vector<std::int64_t> out(kN, -1);
  int rc = run([&](Scheduler* s) {
    return for_coro(system_allocator(), s, out.data(), kN, kAutoGrain);
  });
  EXPECT_EQ(rc, 0);
  for (std::size_t i = 0; i < kN; ++i) ASSERT_EQ(out[i], static_cast<std::int64_t>(i * i));
}

TEST(DataParallel, ForGrainOneMaxSplit) {
  constexpr std::size_t kN = 512;
  std::vector<std::int64_t> out(kN, -1);
  int rc = run([&](Scheduler* s) {
    return for_coro(system_allocator(), s, out.data(), kN,
                    1);  // split to singletons
  });
  EXPECT_EQ(rc, 0);
  for (std::size_t i = 0; i < kN; ++i) ASSERT_EQ(out[i], static_cast<std::int64_t>(i * i));
}

TEST(DataParallel, ForEmptyRange) {
  std::int64_t dummy = 0;
  int rc = run([&](Scheduler* s) {
    return for_coro(system_allocator(), s, &dummy, 0, 8);
  });
  EXPECT_EQ(rc, 0);
}

// Each index written EXACTLY once (no block overlap / double-run) -- an atomic counter per idx.
Status count_body(void* ctx, std::size_t b, std::size_t e) noexcept {
  auto* hits = static_cast<std::atomic<int>*>(ctx);
  for (std::size_t i = b; i < e; ++i) hits[i].fetch_add(1, std::memory_order_relaxed);
  return Status();
}
Task<int> count_coro(mem::AllocatorRef alloc, Scheduler* sched, std::atomic<int>* hits, std::size_t n) {
  Status s = parallel_for(*sched, alloc, BlockedRange(0, n, 7), &count_body, hits);
  co_return s.ok() ? 0 : -1;
}
TEST(DataParallel, ForEachIndexExactlyOnce) {
  constexpr std::size_t kN = 3333;
  std::vector<std::atomic<int>> hits(kN);
  for (auto& h : hits) h.store(0);
  int rc = run([&](Scheduler* s) {
    return count_coro(system_allocator(), s, hits.data(), kN);
  });
  EXPECT_EQ(rc, 0);
  for (std::size_t i = 0; i < kN; ++i) ASSERT_EQ(hits[i].load(), 1) << "at " << i;
}

// Body error (on a specific index) surfaces as non-OK from parallel_for.
Status fail_at_body(void* ctx, std::size_t b, std::size_t e) noexcept {
  auto target = *static_cast<std::size_t*>(ctx);
  if (b <= target && target < e) return Status::Internal("boom");
  return Status();
}
Task<int> forfail_coro(mem::AllocatorRef alloc, Scheduler* sched, std::size_t* target, std::size_t n) {
  Status s = parallel_for(*sched, alloc, BlockedRange(0, n, 16), &fail_at_body, target);
  co_return s.ok() ? 0 : (s.IsInternal() ? 1 : 2);
}
TEST(DataParallel, ForBodyErrorPropagates) {
  std::size_t target = 777;
  int rc = run([&](Scheduler* s) {
    return forfail_coro(system_allocator(), s, &target, 2000);
  });
  EXPECT_EQ(rc, 1);  // Internal surfaced
}

// Load imbalance: one block sleeps; result still correct (work-stealing balances, no hang).
Status imbalanced_body(void* ctx, std::size_t b, std::size_t e) noexcept {
  auto* out = static_cast<std::int64_t*>(ctx);
  if (b == 0) { volatile long s = 0; for (long i = 0; i < 3000000; ++i) s = s + 1; (void)s; }  // slow block
  for (std::size_t i = b; i < e; ++i) out[i] = 1;
  return Status();
}
Task<int> imbal_coro(mem::AllocatorRef alloc, Scheduler* sched, std::int64_t* out, std::size_t n) {
  Status s = parallel_for(*sched, alloc, BlockedRange(0, n, 32), &imbalanced_body, out);
  co_return s.ok() ? 0 : -1;
}
TEST(DataParallel, ForLoadImbalanceStillCorrect) {
  constexpr std::size_t kN = 4096;
  std::vector<std::int64_t> out(kN, 0);
  int rc = run([&](Scheduler* s) {
    return imbal_coro(system_allocator(), s, out.data(), kN);
  });
  EXPECT_EQ(rc, 0);
  std::int64_t sum = 0; for (auto v : out) sum += v;
  EXPECT_EQ(sum, static_cast<std::int64_t>(kN));  // every index written
}

// ---- parallel_reduce (fast) --------------------------------------------------
// Reduce: sum of i over [b,e). elem = std::int64_t.
Status sum_body(void*, std::size_t b, std::size_t e, void* out) noexcept {
  std::int64_t acc = 0;
  for (std::size_t i = b; i < e; ++i) acc += static_cast<std::int64_t>(i);
  *static_cast<std::int64_t*>(out) = acc;
  return Status();
}
void sum_combine(void*, void* a, const void* b) noexcept {
  *static_cast<std::int64_t*>(a) += *static_cast<const std::int64_t*>(b);
}
void sum_identity(void*, void* out) noexcept { *static_cast<std::int64_t*>(out) = 0; }

Task<int> reduce_coro(mem::AllocatorRef alloc, Scheduler* sched, std::size_t n, std::size_t grain,
                      std::int64_t* result) {
  Status s = parallel_reduce(*sched, alloc, BlockedRange(0, n, grain), sizeof(std::int64_t),
                             &sum_body, &sum_combine, &sum_identity, nullptr, result);
  co_return s.ok() ? 0 : -1;
}

Task<int> saturated_for_coro(mem::AllocatorRef alloc, Scheduler* sched,
                             std::int64_t* out, std::size_t n) {
  Status s = parallel_for(*sched, alloc, BlockedRange(0, n, 1),
                          &square_body, out);
  co_return s.ok() ? 0 : (s.IsResourceExhausted() ? 1 : 2);
}

Task<int> saturated_reduce_coro(mem::AllocatorRef alloc, Scheduler* sched,
                                std::size_t n, std::int64_t* result) {
  Status s = parallel_reduce(*sched, alloc, BlockedRange(0, n, 1),
                             sizeof(std::int64_t), &sum_body, &sum_combine,
                             &sum_identity, nullptr, result);
  co_return s.ok() ? 0 : (s.IsResourceExhausted() ? 1 : 2);
}

TEST(DataParallel, BoundedQueueSaturationReturnsStructuredError) {
  // One worker cannot drain its own queues while recursively descending the
  // left branch.  Capacity two therefore deterministically rejects a deeper
  // right spawn.  Both algorithms must surface ResourceExhausted; reduce must
  // never combine the uninitialized partial belonging to the rejected task.
  constexpr std::size_t kN = 256;
  std::vector<std::int64_t> out(kN, -1);
  EXPECT_EQ(run([&](Scheduler* s) {
              return saturated_for_coro(system_allocator(), s,
                                        out.data(), kN);
            },
            1, 2),
            1);

  std::int64_t result = -1;
  EXPECT_EQ(run([&](Scheduler* s) {
              return saturated_reduce_coro(system_allocator(), s, kN,
                                           &result);
            },
            1, 2),
            1);
}

TEST(DataParallel, ReduceSumMatchesSerial) {
  constexpr std::size_t kN = 100000;
  std::int64_t result = -1;
  int rc = run([&](Scheduler* s) {
    return reduce_coro(system_allocator(), s, kN, 128, &result);
  });
  EXPECT_EQ(rc, 0);
  std::int64_t expect = static_cast<std::int64_t>(kN) * (kN - 1) / 2;  // sum 0..kN-1
  EXPECT_EQ(result, expect);
}

TEST(DataParallel, ReduceAutoGrainAndGrainOne) {
  constexpr std::size_t kN = 2048;
  std::int64_t expect = static_cast<std::int64_t>(kN) * (kN - 1) / 2;
  std::int64_t r_auto = -1, r_one = -1;
  EXPECT_EQ(run([&](Scheduler* s) {
              return reduce_coro(system_allocator(), s, kN, kAutoGrain,
                                 &r_auto);
            }),
            0);
  EXPECT_EQ(run([&](Scheduler* s) {
              return reduce_coro(system_allocator(), s, kN, 1, &r_one);
            }),
            0);
  EXPECT_EQ(r_auto, expect);
  EXPECT_EQ(r_one, expect);  // integer sum is associative -> exact regardless of tree shape
}

TEST(DataParallel, ReduceEmptyRangeIsIdentity) {
  std::int64_t result = 12345;
  int rc = run([&](Scheduler* s) {
    return reduce_coro(system_allocator(), s, 0, 8, &result);
  });
  EXPECT_EQ(rc, 0);
  EXPECT_EQ(result, 0);  // identity
}

Status rfail_body(void* ctx, std::size_t b, std::size_t e, void* out) noexcept {
  auto target = *static_cast<std::size_t*>(ctx);
  if (b <= target && target < e) return Status::Internal("boom");
  *static_cast<std::int64_t*>(out) = 0;
  return Status();
}
Task<int> reducefail_coro(mem::AllocatorRef alloc, Scheduler* sched, std::size_t* target, std::size_t n,
                          std::int64_t* result) {
  Status s = parallel_reduce(*sched, alloc, BlockedRange(0, n, 16), sizeof(std::int64_t),
                             &rfail_body, &sum_combine, &sum_identity, target, result);
  co_return s.ok() ? 0 : 1;
}
TEST(DataParallel, ReduceBodyErrorPropagates) {
  std::size_t target = 500;
  std::int64_t result = -1;
  int rc = run([&](Scheduler* s) {
    return reducefail_coro(system_allocator(), s, &target, 1500, &result);
  });
  EXPECT_EQ(rc, 1);
}

// NON-COMMUTATIVE combine: reduce indices into a fixed-width string "0,1,2,...". concat is
// associative but NOT commutative -> a left/right swap in the fold would scramble the order.
// Payload is a 64-char buffer + length (>8 bytes -> exercises the elem_size>8 blob path).
struct StrBuf { char s[200]; std::size_t len; };
Status concat_body(void*, std::size_t b, std::size_t e, void* out) noexcept {
  auto* sb = static_cast<StrBuf*>(out);
  sb->len = 0;
  for (std::size_t i = b; i < e; ++i) {
    int w = std::snprintf(sb->s + sb->len, sizeof(sb->s) - sb->len, "%zu,", i);
    if (w > 0) sb->len += static_cast<std::size_t>(w);
  }
  return Status();
}
void concat_combine(void*, void* a, const void* b) noexcept {
  auto* sa = static_cast<StrBuf*>(a);
  const auto* sb = static_cast<const StrBuf*>(b);
  std::size_t cp = sb->len;
  if (sa->len + cp >= sizeof(sa->s)) cp = sizeof(sa->s) - sa->len - 1;
  std::memcpy(sa->s + sa->len, sb->s, cp);  // append RIGHT after LEFT (order-preserving)
  sa->len += cp;
  sa->s[sa->len] = '\0';
}
void concat_identity(void*, void* out) noexcept {
  auto* sb = static_cast<StrBuf*>(out);
  sb->len = 0; sb->s[0] = '\0';
}
Task<int> concat_coro(mem::AllocatorRef alloc, Scheduler* sched, std::size_t n, std::size_t grain,
                      StrBuf* result) {
  Status s = parallel_reduce(*sched, alloc, BlockedRange(0, n, grain), sizeof(StrBuf),
                             &concat_body, &concat_combine, &concat_identity, nullptr, result);
  co_return s.ok() ? 0 : -1;
}
TEST(DataParallel, ReduceNonCommutativePreservesOrder) {
  constexpr std::size_t kN = 20;
  StrBuf result{};
  int rc = run([&](Scheduler* s) {
    return concat_coro(system_allocator(), s, kN, 3, &result);
  });
  EXPECT_EQ(rc, 0);
  // Serial reference: "0,1,2,...,19,". A left/right swap would reorder these.
  std::string expect;
  for (std::size_t i = 0; i < kN; ++i) { expect += std::to_string(i); expect += ','; }
  EXPECT_EQ(std::string(result.s), expect);
}

// Large range with a REASONABLE grain (deep-ish tree, many blocks, but not one fiber per
// element): correct + no OOM. NOTE: grain=1 on a huge range is the DELIBERATE anti-pattern the
// auto-grain floor guards against -- each fork is a full 64KB fiber stack, so 200k leaves would
// need ~13GB of fiber stacks and hang. Callers who want fine blocks on a big range pass a
// moderate grain (or kAutoGrain, which floors to keep forks coarse). This test uses grain=256.
Task<int> big_sum_coro(mem::AllocatorRef alloc, Scheduler* sched, std::size_t n, std::size_t grain,
                       std::int64_t* r) {
  Status s = parallel_reduce(*sched, alloc, BlockedRange(0, n, grain), sizeof(std::int64_t),
                             &sum_body, &sum_combine, &sum_identity, nullptr, r);
  co_return s.ok() ? 0 : -1;
}
TEST(DataParallel, ReduceLargeRange) {
  constexpr std::size_t kN = 1000000;  // 1M elements, grain 256 -> ~4k blocks, ~12-level tree
  std::int64_t result = -1;
  int rc = run([&](Scheduler* s) {
    return big_sum_coro(system_allocator(), s, kN, 256, &result);
  });
  EXPECT_EQ(rc, 0);
  EXPECT_EQ(result, static_cast<std::int64_t>(kN) * (kN - 1) / 2);
}
TEST(DataParallel, ReduceLargeRangeAutoGrain) {
  constexpr std::size_t kN = 1000000;
  std::int64_t result = -1;
  int rc = run([&](Scheduler* s) {
    return big_sum_coro(system_allocator(), s, kN, kAutoGrain, &result);
  });
  EXPECT_EQ(rc, 0);
  EXPECT_EQ(result, static_cast<std::int64_t>(kN) * (kN - 1) / 2);
}

}  // namespace
}  // namespace hyle::graph

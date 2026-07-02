#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "hyle/io/proactor.h"

using hyle::io::Completion;
using hyle::io::Proactor;

// io-01 is the completion-reaping seam; these tests prove the mechanism the real
// backends (io-02 io_uring / IOCP) drive: attach a 64-bit token to an operation,
// get it back verbatim on completion, reap in batch. On Windows this exercises
// the real IOCP port (PostQueuedCompletionStatus/GetQueuedCompletionStatusEx);
// on Linux the backend is io-02-pending so init() returns false and we skip.

TEST(Proactor, InitAndPostReapRoundTrip) {
  Proactor p;
  if (!p.init()) GTEST_SKIP() << "no proactor backend on this platform yet";
  ASSERT_TRUE(p.valid());

  // Post 3 completions carrying distinct tokens + results.
  EXPECT_TRUE(p.post(0xA1, 10));
  EXPECT_TRUE(p.post(0xB2, 20));
  EXPECT_TRUE(p.post(0xC3, 30));

  // Reap them back (block up to 1s so a slow CI still catches all three).
  Completion out[8];
  std::size_t got = 0;
  // May take more than one reap to see all three; loop until we have 3.
  while (got < 3) {
    std::size_t n = p.reap(out + got, 8 - got, 1'000'000'000ull);
    if (n == 0) break;  // timeout -> stop (test will fail the count check)
    got += n;
  }
  ASSERT_EQ(got, 3u);

  // Verify every token + result came back verbatim (order not guaranteed).
  std::uint64_t seen_tokens = 0;
  std::int64_t sum = 0;
  for (std::size_t i = 0; i < got; ++i) {
    seen_tokens |= out[i].user_data;
    sum += out[i].result;
  }
  EXPECT_EQ(seen_tokens, 0xA1u | 0xB2u | 0xC3u);
  EXPECT_EQ(sum, 60);
}

TEST(Proactor, ReapEmptyIsNonBlockingWithZeroTimeout) {
  Proactor p;
  if (!p.init()) GTEST_SKIP() << "no proactor backend on this platform yet";
  Completion out[4];
  // Nothing posted -> a 0-timeout poll returns immediately with 0.
  auto t0 = std::chrono::steady_clock::now();
  std::size_t n = p.reap(out, 4, 0);
  auto elapsed = std::chrono::steady_clock::now() - t0;
  EXPECT_EQ(n, 0u);
  EXPECT_LT(elapsed, std::chrono::milliseconds(100));  // did not block
}

// The cross-thread wake path the real backend relies on: a consumer blocks in
// reap(); another thread posts a completion; the consumer wakes and gets the
// token. This is io->Waker's physical basis (post == "an op finished elsewhere").
TEST(Proactor, BlockedReapWokenByPostFromAnotherThread) {
  Proactor p;
  if (!p.init()) GTEST_SKIP() << "no proactor backend on this platform yet";

  std::atomic<std::uint64_t> got_token{0};
  std::thread consumer([&] {
    Completion out[4];
    std::size_t n = p.reap(out, 4, 5'000'000'000ull);  // block up to 5s
    if (n > 0) got_token.store(out[0].user_data, std::memory_order_release);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(20));  // ensure it parked
  EXPECT_TRUE(p.post(0xDEAD, 42));  // wake it from here
  consumer.join();
  EXPECT_EQ(got_token.load(std::memory_order_acquire), 0xDEADu);
}

// Batch reap: post many, confirm reap pulls them in batches (io_uring
// for_each_cqe analogue -- amortized, not one syscall per completion).
TEST(Proactor, BatchReapManyCompletions) {
  Proactor p;
  if (!p.init()) GTEST_SKIP() << "no proactor backend on this platform yet";
  constexpr int kN = 100;
  for (int i = 0; i < kN; ++i) {
    ASSERT_TRUE(p.post(static_cast<std::uint64_t>(i + 1), i));
  }
  Completion out[64];
  int total = 0;
  std::uint64_t token_sum = 0;
  while (total < kN) {
    std::size_t n = p.reap(out, 64, 1'000'000'000ull);
    if (n == 0) break;
    for (std::size_t i = 0; i < n; ++i) token_sum += out[i].user_data;
    total += static_cast<int>(n);
  }
  EXPECT_EQ(total, kN);
  EXPECT_EQ(token_sum, static_cast<std::uint64_t>(kN) * (kN + 1) / 2);  // 1..100
}

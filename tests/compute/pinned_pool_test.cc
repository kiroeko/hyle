#include "hyle/compute/pinned_pool.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "hyle/compute/buffer.h"
#include "hyle/compute/device.h"
#include "hyle/mem/memory_arena.h"

// compute-02b: pinned staging built on mem/ (pinned_allocator + mem::MemoryArena +
// mem::Arena) + the batched-submit helper PinnedStaging. Runs on any platform (host
// fallback = plain RAM; a real GPU exercises cudaMallocHost + H2D). The point is to
// prove the "prepare contiguous in pinned host, submit in one copy" pattern (合批)
// and that pinned pooling reuses mem/ verbatim (no reinvented pool).

using namespace hyle;
using namespace hyle::compute;

// ---- pinned_allocator feeds mem/ directly (the whole point: reuse mem) -------

TEST(PinnedPool, AllocatorBacksMemArena) {
  Device d = default_device();
  mem::Allocator pinned = pinned_allocator(d);
  ASSERT_TRUE(pinned.valid());

  // A pinned FIXED-BLOCK pool is just mem::MemoryArena over the pinned allocator --
  // mem/ reused verbatim, no compute-side pool. 4 blocks of 64 KiB (scaled-down ring).
  constexpr std::size_t kBlock = 64 * 1024;
  mem::MemoryArena pool(pinned, kBlock, 4);

  // Acquire/release blocks from the pool (the reuse that amortizes cudaMallocHost).
  void* b0 = pool.allocate();
  void* b1 = pool.allocate();
  ASSERT_NE(b0, nullptr);
  ASSERT_NE(b1, nullptr);
  EXPECT_NE(b0, b1);
  // The pinned block is host RAM -- directly writable on the CPU.
  std::memset(b0, 0xAB, kBlock);
  pool.deallocate(b0);
  pool.deallocate(b1);
  // Reacquire -> pool reuses a freed block (no new cudaMallocHost).
  void* b2 = pool.allocate();
  EXPECT_TRUE(b2 == b0 || b2 == b1);
  pool.deallocate(b2);
}

// ---- mem::Arena bumps CONTIGUOUS data in a pinned block ----------------------

TEST(PinnedPool, ArenaContiguousInPinnedBlock) {
  Device d = default_device();
  mem::Allocator pinned = pinned_allocator(d);
  constexpr std::size_t kBlock = 1 << 20;  // 1 MiB
  mem::MemoryArena pool(pinned, kBlock, 2);

  mem::Arena staging(&pool);  // borrow one pinned block, bump from it
  // Prepare two arrays back-to-back -> they are CONTIGUOUS in pinned memory.
  auto* a = static_cast<std::uint32_t*>(staging.allocate(256 * sizeof(std::uint32_t), 16));
  auto* b = static_cast<std::uint32_t*>(staging.allocate(256 * sizeof(std::uint32_t), 16));
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  // Contiguity: b follows a within 16-byte alignment (256*4=1024 is 16-aligned).
  EXPECT_EQ(reinterpret_cast<char*>(b) - reinterpret_cast<char*>(a),
            static_cast<std::ptrdiff_t>(256 * sizeof(std::uint32_t)));
  EXPECT_EQ(staging.used(), 512 * sizeof(std::uint32_t));
}

// ---- PinnedStaging: prepare contiguous, submit in ONE copy (合批) ------------

TEST(PinnedPool, StagingBatchedSubmit) {
  Device d = default_device();

  // Device buffer to receive two logically-separate arrays in one H2D copy.
  constexpr std::size_t kElems = 512;
  constexpr std::size_t kArrBytes = kElems * sizeof(float);
  auto dev = Buffer::create(device_buffer_type(d), 2 * kArrBytes);
  ASSERT_TRUE(dev.has_value()) << dev.error().message();

  auto staging = PinnedStaging::create(d, 2 * kArrBytes);
  ASSERT_TRUE(staging.has_value()) << staging.error().message();

  // Prepare two arrays contiguously in the pinned staging region.
  auto* fa = static_cast<float*>(staging->allocate(kArrBytes, 16));
  auto* fb = static_cast<float*>(staging->allocate(kArrBytes, 16));
  ASSERT_NE(fa, nullptr);
  ASSERT_NE(fb, nullptr);
  for (std::size_t i = 0; i < kElems; ++i) {
    fa[i] = static_cast<float>(i);
    fb[i] = static_cast<float>(i) + 1000.0f;
  }
  EXPECT_EQ(staging->used(), 2 * kArrBytes);

  // ONE batched H2D copy of the whole contiguous run (合批 -- not two set() calls).
  ASSERT_TRUE(staging->commit_to(*dev, 0).ok());

  // Read the device buffer back and verify both arrays landed correctly.
  std::vector<float> out(2 * kElems);
  ASSERT_TRUE(dev->get(0, out.data(), 2 * kArrBytes).ok());
  for (std::size_t i = 0; i < kElems; ++i) {
    EXPECT_FLOAT_EQ(out[i], static_cast<float>(i)) << "a[" << i << "]";
    EXPECT_FLOAT_EQ(out[kElems + i], static_cast<float>(i) + 1000.0f) << "b[" << i << "]";
  }
}

// ---- PinnedStaging reset reuses the region for the next batch ----------------

TEST(PinnedPool, StagingResetReuse) {
  Device d = default_device();
  auto staging = PinnedStaging::create(d, 4096);
  ASSERT_TRUE(staging.has_value());

  void* p1 = staging->allocate(1024, 16);
  ASSERT_NE(p1, nullptr);
  EXPECT_EQ(staging->used(), 1024u);
  staging->reset();
  EXPECT_EQ(staging->used(), 0u);
  void* p2 = staging->allocate(1024, 16);
  EXPECT_EQ(p1, p2);  // reset rewinds to the same base
}

// ---- move semantics (no double-free of the pinned block) ---------------------

TEST(PinnedPool, StagingMove) {
  Device d = default_device();
  auto s1 = PinnedStaging::create(d, 2048);
  ASSERT_TRUE(s1.has_value());
  void* base = s1->base();
  PinnedStaging moved = std::move(*s1);
  EXPECT_TRUE(moved.valid());
  EXPECT_EQ(moved.base(), base);
  EXPECT_FALSE(s1->valid());
}

#include "hyle/compute/device_pool.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "hyle/compute/buffer.h"
#include "hyle/compute/device.h"

// compute-02c: device (VRAM) sub-allocator pool -- ported from ggml (VMM + leg).
// Runs on any platform: host fallback uses a leg-style malloc cache (reuse testable
// without a GPU); a real GPU exercises the VMM pool (RTX 4070 supports CUDA VMM) or
// the leg CUDA pool. The point is to prove "预分配大块换性能": a freed block is
// reused by the next alloc instead of hitting the driver.

using namespace hyle;
using namespace hyle::compute;

namespace {
Device gpu_or_host(bool* is_gpu) {
  Device devs[8];
  std::size_t n = enumerate(devs, 8);
  for (std::size_t i = 0; i < n && i < 8; ++i) {
    if (devs[i].is_gpu()) { *is_gpu = true; return devs[i]; }
  }
  *is_gpu = false;
  return default_device();
}
}  // namespace

TEST(DevicePool, CreateAndKind) {
  Device d = default_device();
  auto pool = DevicePool::create(d);
  ASSERT_TRUE(pool.has_value()) << pool.error().message();
  EXPECT_TRUE(pool->valid());
  // is_vmm just reports which backend was chosen; host is always leg (false).
  bool is_gpu = false;
  gpu_or_host(&is_gpu);
  if (!is_gpu) EXPECT_FALSE(pool->is_vmm());  // host = leg-style
}

TEST(DevicePool, AllocFreeRoundTrip) {
  Device d = default_device();
  auto pool = DevicePool::create(d);
  ASSERT_TRUE(pool.has_value());

  auto p = pool->alloc(4096);
  ASSERT_TRUE(p.has_value()) << p.error().message();
  EXPECT_NE(*p, nullptr);
  pool->free(*p, 4096);
}

// The core value: a freed block is REUSED by a same-size alloc (no fresh driver call).
// For BOTH VMM (bump reuses the just-freed tail) and leg (best-fit cache) the pointer
// returned after free-then-alloc is the same block.
TEST(DevicePool, FreedBlockIsReused) {
  Device d = default_device();
  auto pool = DevicePool::create(d);
  ASSERT_TRUE(pool.has_value());

  constexpr std::size_t kSize = 1 << 20;  // 1 MiB
  auto a = pool->alloc(kSize);
  ASSERT_TRUE(a.has_value());
  void* first = *a;
  pool->free(*a, kSize);  // return to pool (not to driver)

  auto b = pool->alloc(kSize);  // same size -> should reuse the cached/bumped block
  ASSERT_TRUE(b.has_value());
  EXPECT_EQ(*b, first) << "pool did not reuse the freed block";
  pool->free(*b, kSize);
}

// PoolAlloc RAII: scoped borrow, frees on destruction. Nested scopes free in reverse
// (LIFO) -- which is exactly what the VMM backend requires, given for free by scope.
TEST(DevicePool, PoolAllocRaiiLifo) {
  Device d = default_device();
  auto pool = DevicePool::create(d);
  ASSERT_TRUE(pool.has_value());

  {
    PoolAlloc<float> a(*pool, 256);
    ASSERT_TRUE(a.valid());
    EXPECT_NE(a.get(), nullptr);
    EXPECT_EQ(a.size_bytes(), 256 * sizeof(float));
    {
      PoolAlloc<float> b(*pool, 512);  // allocated after a
      ASSERT_TRUE(b.valid());
      EXPECT_NE(b.get(), a.get());
      // b destroyed here (freed first) -> LIFO order preserved for VMM
    }
    // a destroyed here (freed second)
  }
  // After both freed, a fresh alloc reuses reclaimed space.
  auto c = pool->alloc(256 * sizeof(float));
  ASSERT_TRUE(c.has_value());
  pool->free(*c, 256 * sizeof(float));
}

// Multiple live allocations get distinct blocks; LIFO free (reverse order).
TEST(DevicePool, MultipleLiveDistinctThenLifoFree) {
  Device d = default_device();
  auto pool = DevicePool::create(d);
  ASSERT_TRUE(pool.has_value());

  auto a = pool->alloc(1024);
  auto b = pool->alloc(2048);
  auto c = pool->alloc(4096);
  ASSERT_TRUE(a.has_value() && b.has_value() && c.has_value());
  EXPECT_NE(*a, *b);
  EXPECT_NE(*b, *c);
  EXPECT_NE(*a, *c);
  // Free in reverse order (LIFO -- required by VMM, fine for leg).
  pool->free(*c, 4096);
  pool->free(*b, 2048);
  pool->free(*a, 1024);
}

// A pool block is real device memory a kernel would use: verify it round-trips data
// through a Buffer-style copy path. (We use cudaMemcpy via Buffer to reach the pool
// pointer; on host it's plain memory.) Here we just confirm the pointer is usable as
// a device destination by copying into it through a Buffer wrapper is out of scope --
// instead confirm alloc alignment (VMM aligns to 128).
TEST(DevicePool, AllocationsAreAligned) {
  Device d = default_device();
  auto pool = DevicePool::create(d);
  ASSERT_TRUE(pool.has_value());
  auto p = pool->alloc(100);  // odd size
  ASSERT_TRUE(p.has_value());
  if (pool->is_vmm()) {
    // VMM rounds to 128B alignment.
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(*p) % 128, 0u);
  }
  pool->free(*p, 100);
}

TEST(DevicePool, MoveSemantics) {
  Device d = default_device();
  auto p1 = DevicePool::create(d);
  ASSERT_TRUE(p1.has_value());
  DevicePool moved = std::move(*p1);
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(p1->valid());  // moved-from won't double-destroy the backend pool
  auto a = moved.alloc(256);
  ASSERT_TRUE(a.has_value());
  moved.free(*a, 256);
}

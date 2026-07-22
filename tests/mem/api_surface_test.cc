#include <gtest/gtest.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#include "hyle/c/mem.h"
#include "hyle/mem/alignment.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/fiber_stack.h"
#include "hyle/mem/fixed_pool.h"
#include "hyle/mem/growable_pool.h"
#include "hyle/mem/hugepage.h"
#include "hyle/mem/object_pool.h"
#include "hyle/mem/pmr.h"
#include "hyle/mem/pool_cache.h"
#include "hyle/mem/pool_loan.h"
#include "hyle/mem/quota_allocator.h"
#include "hyle/mem/region.h"
#include "hyle/mem/shared_pool.h"
#include "hyle/mem/stack_allocator.h"
#include "hyle/mem/stats.h"
#include "hyle/mem/system_inline.h"
#include "hyle/platform/virtual_memory.h"

namespace {

using namespace hyle::mem;

// Compile-time inventory for the current public C++ surface. It catches
// accidental drift; before the first release, an intentional product-shape
// change updates this inventory and is not a compatibility promise.
static_assert(std::is_trivially_copyable_v<Allocator>);
static_assert(sizeof(Allocator) == sizeof(void*) * 4);
static_assert(std::is_empty_v<SystemAllocator>);
static_assert(std::is_standard_layout_v<RegionOptions>);
static_assert(RegionOptions{}.max_block_size == 4u * 1024u * 1024u);
static_assert(std::is_standard_layout_v<AllocStatsSnapshot>);
static_assert(std::derived_from<PmrResource, std::pmr::memory_resource>);

template <class A>
concept AllocatorSurface = requires(A a, void* p, std::size_t n) {
  { a.valid() } -> std::same_as<bool>;
  { a.allocate(n, n) } -> std::same_as<void*>;
  { a.deallocate(p, n, n) } -> std::same_as<void>;
  { a.reallocate(p, n, n, n) } -> std::same_as<void*>;
};
static_assert(AllocatorSurface<Allocator>);

template <class R>
concept RegionSurface = requires(R r, std::size_t n) {
  { r.allocate(n, n) } -> std::same_as<void*>;
  { r.clear() } -> std::same_as<void>;
  { r.as_allocator() } -> std::same_as<Allocator>;
};
static_assert(RegionSurface<FixedRegion>);
static_assert(RegionSurface<Region>);
static_assert(RegionSurface<PooledRegion>);

template <class P>
concept FixedBlockPoolSurface = requires(P p, void* block) {
  { p.valid() } -> std::same_as<bool>;
  { p.allocate() } -> std::same_as<void*>;
  { p.deallocate(block) } -> std::same_as<void>;
  { p.capacity() } -> std::same_as<std::uint32_t>;
  { p.as_allocator() } -> std::same_as<Allocator>;
};
static_assert(FixedBlockPoolSurface<SharedPool>);

static_assert(requires(SharedPoolCache cache, void* block) {
  { cache.valid() } -> std::same_as<bool>;
  { cache.allocate() } -> std::same_as<void*>;
  { cache.deallocate(block) } -> std::same_as<void>;
  { cache.publish() } -> std::same_as<void>;
  { cache.cached() } -> std::same_as<std::uint32_t>;
  { cache.as_allocator() } -> std::same_as<Allocator>;
});

template <class P>
concept ObjectPoolSurface = requires(P p, int* value) {
  { p.create(1) } -> std::same_as<int*>;
  { p.destroy(value) } -> std::same_as<void>;
  { p.try_loan(1) } -> std::same_as<PoolLoan<int, P>>;
};
static_assert(ObjectPoolSurface<ObjectPool<int>>);
static_assert(ObjectPoolSurface<ObjectPoolCache<int>>);
static_assert(ObjectPoolSurface<GrowablePool<int>>);

static_assert(requires(FixedPool p, void* ptr, std::size_t n) {
  { p.allocate(n, n) } -> std::same_as<void*>;
  { p.deallocate(ptr) } -> std::same_as<void>;
  { p.reallocate(ptr, n, n, n) } -> std::same_as<void*>;
  { p.reset() } -> std::same_as<void>;
  { p.as_allocator() } -> std::same_as<Allocator>;
});

static_assert(requires(QuotaAllocator q) {
  { q.get() } -> std::same_as<Allocator>;
  { q.used() } -> std::same_as<std::size_t>;
  { q.peak() } -> std::same_as<std::size_t>;
  { q.limit() } -> std::same_as<std::size_t>;
});

static_assert(requires(StatsAllocator s) {
  { s.get() } -> std::same_as<Allocator>;
  { s.snapshot() } -> std::same_as<AllocStatsSnapshot>;
  { s.balanced() } -> std::same_as<bool>;
  { s.leaks() } -> std::same_as<LeakTracker*>;
});

static_assert(std::is_move_constructible_v<Region>);
static_assert(!std::is_copy_constructible_v<Region>);
static_assert(!std::is_move_constructible_v<FixedRegion>);
static_assert(!std::is_move_constructible_v<FixedPool>);
static_assert(!std::is_move_constructible_v<SharedPool>);
static_assert(!std::is_move_constructible_v<SharedPoolCache>);
static_assert(std::is_move_constructible_v<FiberStack>);

TEST(MemApiSurface, PublicCapabilityFamiliesCompileTogether) {
  EXPECT_EQ(kFiberStackSmall, 16u * 1024u);
  EXPECT_EQ(kFiberStackDefault, 64u * 1024u);
  EXPECT_EQ(kFiberStackLarge, 256u * 1024u);
  EXPECT_GE(kDefaultAlignment, alignof(void*));
}

TEST(MemApiSurface, CErrorChannelDoesNotOverwritePlatformVmError) {
  hyle::platform::vm_set_last_error("platform sentinel");
  hyle_allocator invalid{};
  EXPECT_EQ(hyle_shared_pool_create(invalid, 64, 4, 0), nullptr);
  EXPECT_EQ(hyle_mem_last_error_code(), HYLE_MEM_ERROR_INVALID_ARGUMENT);
  EXPECT_NE(hyle_mem_last_error()[0], '\0');
  EXPECT_STREQ(hyle::platform::vm_last_error(), "platform sentinel");

  EXPECT_EQ(hyle_huge_region_open(0, HYLE_HUGE_NONE, HYLE_MEM_NUMA_ANY),
            nullptr);
  EXPECT_EQ(hyle_mem_last_error_code(), HYLE_MEM_ERROR_INVALID_ARGUMENT);
  EXPECT_NE(hyle_mem_last_error()[0], '\0');
  EXPECT_STREQ(hyle::platform::vm_last_error(), "platform sentinel");
  hyle::platform::vm_set_last_error(nullptr);
}

}  // namespace

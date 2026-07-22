#include <gtest/gtest.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#include "hyle/mem/alignment.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/fiber_stack.h"
#include "hyle/mem/fixed_pool.h"
#include "hyle/mem/growable_pool.h"
#include "hyle/mem/hugepage.h"
#include "hyle/mem/object_pool.h"
#include "hyle/mem/pmr.h"
#include "hyle/mem/pool_loan.h"
#include "hyle/mem/quota_allocator.h"
#include "hyle/mem/region.h"
#include "hyle/mem/shared_pool.h"
#include "hyle/mem/stack_allocator.h"
#include "hyle/mem/stats.h"
#include "hyle/mem/system_inline.h"

namespace {

using namespace hyle::mem;

// This is the C++ source-compatibility side of the freeze baseline. C++ object
// layout is intentionally not an ABI promise; the stable binary boundary is the
// C manifest checked by mem-c-abi-baseline.py.
static_assert(std::is_trivially_copyable_v<Allocator>);
static_assert(sizeof(Allocator) == sizeof(void*) * 4);
static_assert(std::is_empty_v<SystemAllocator>);
static_assert(std::is_standard_layout_v<RegionOptions>);
static_assert(std::is_standard_layout_v<AllocStatsSnapshot>);
static_assert(std::derived_from<PmrResource, std::pmr::memory_resource>);
static_assert(std::same_as<Arena, PooledRegion>);

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
static_assert(FixedBlockPoolSurface<SharedPoolCached>);

template <class P>
concept ObjectPoolSurface = requires(P p, int* value) {
  { p.create(1) } -> std::same_as<int*>;
  { p.destroy(value) } -> std::same_as<void>;
  { p.try_loan(1) } -> std::same_as<PoolLoan<int, P>>;
};
static_assert(ObjectPoolSurface<ObjectPool<int>>);
static_assert(ObjectPoolSurface<ObjectPoolCached<int>>);
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
static_assert(std::is_move_constructible_v<FiberStack>);

TEST(MemApiBaseline, PublicCapabilityFamiliesCompileTogether) {
  EXPECT_EQ(kFiberStackSmall, 16u * 1024u);
  EXPECT_EQ(kFiberStackDefault, 64u * 1024u);
  EXPECT_EQ(kFiberStackLarge, 256u * 1024u);
  EXPECT_GE(kDefaultAlignment, alignof(void*));
}

}  // namespace

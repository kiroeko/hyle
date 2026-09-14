#include <gtest/gtest.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#include "hyle/mem/alignment.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/arena.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/buffer_pool.h"
#include "hyle/mem/fiber_stack.h"
#include "hyle/mem/fixed_pool.h"
#include "hyle/mem/growable_pool.h"
#include "hyle/mem/heap.h"
#include "hyle/mem/lifetime_plan.h"
#include "hyle/mem/mapping.h"
#include "hyle/mem/large_mapping.h"
#include "hyle/mem/large_page_reservation.h"
#include "hyle/mem/object_pool.h"
#include "hyle/mem/pmr.h"
#include "hyle/mem/pool.h"
#include "hyle/mem/pool_cache.h"
#include "hyle/mem/pool_loan.h"
#include "hyle/mem/quota_allocator.h"
#include "hyle/mem/region.h"
#include "hyle/mem/resource.h"
#include "hyle/mem/shared_pool.h"
#include "hyle/mem/stack_allocator.h"
#include "hyle/mem/stats.h"
#include "hyle/mem/std_allocator.h"
#include "hyle/mem/system_inline.h"
#include "hyle/platform/virtual_memory.h"

namespace {

using namespace hyle::mem;

// Compile-time inventory for the current public C++ surface. It catches
// accidental drift; before the first release, an intentional product-shape
// change updates this inventory and is not a compatibility promise.
static_assert(std::is_trivially_copyable_v<AllocatorRef>);
static_assert(sizeof(AllocatorRef) == sizeof(void*) * 4);
static_assert(sizeof(MemoryResourceRef) == sizeof(void*));
static_assert(sizeof(Heap) == sizeof(void*));
static_assert(sizeof(Buffer) <= sizeof(void*) * 4);
static_assert(sizeof(BufferPool) == sizeof(void*));
static_assert(sizeof(CompositeBuffer) == sizeof(void*));
static_assert(sizeof(CompositeBufferView) == sizeof(void*) * 3);
static_assert(sizeof(ConstCompositeBufferView) == sizeof(void*) * 3);
static_assert(sizeof(LifetimePlan) == sizeof(void*));
static_assert(sizeof(PlannedBuffers) == sizeof(void*));
static_assert(std::is_standard_layout_v<MemoryPlacement>);
static_assert(std::is_standard_layout_v<LifetimeResourcePlan>);
static_assert(std::is_standard_layout_v<LifetimeBackingPlan>);
static_assert(sizeof(BufferView) == sizeof(void*) * 4);
static_assert(sizeof(ConstBufferView) == sizeof(void*) * 4);
static_assert(std::is_trivially_copyable_v<MemoryResourceRef>);
static_assert(std::is_trivially_copyable_v<BufferView>);
static_assert(std::is_trivially_copyable_v<ConstBufferView>);
static_assert(std::same_as<decltype(system_resource()), MemoryResourceRef>);
static_assert(std::is_empty_v<SystemAllocator>);
static_assert(std::is_empty_v<SystemStdAllocator<int>>);
static_assert(std::allocator_traits<SystemStdAllocator<int>>::is_always_equal::value);
static_assert(std::is_standard_layout_v<ArenaOptions>);
static_assert(ArenaOptions{}.max_block_size == 4u * 1024u * 1024u);
static_assert(std::is_standard_layout_v<AllocStatsSnapshot>);
static_assert(std::is_standard_layout_v<SystemMemorySnapshot>);
static_assert(std::derived_from<PmrResource, std::pmr::memory_resource>);
static_assert(
    std::derived_from<AlignedPmrResource, std::pmr::memory_resource>);
static_assert(std::same_as<std::allocator_traits<StdAllocator<int>>::value_type,
                           int>);
static_assert(!std::default_initializable<StdAllocator<int>>);
static_assert(
    std::same_as<std::allocator_traits<StdAllocator<int>>::rebind_alloc<double>,
                 StdAllocator<double>>);
static_assert(std::same_as<
              std::allocator_traits<SystemStdAllocator<int>>::rebind_alloc<double>,
              SystemStdAllocator<double>>);
static_assert(std::same_as<decltype(system_memory_resource()),
                           std::pmr::memory_resource*>);
static_assert(std::same_as<decltype(system_memory_snapshot()),
                           hyle::Result<SystemMemorySnapshot>>);
static_assert(std::same_as<VirtualMapping, hyle::platform::VmRegion>);
static_assert(std::same_as<MappingProtection, hyle::platform::Protect>);
static_assert(std::same_as<MappingAdvice, hyle::platform::VmAdvice>);
static_assert(std::is_standard_layout_v<MappingRange>);
static_assert(requires(FileMapping mapping, SharedMapping shared,
                       std::span<const MappingRange> ranges,
                       std::size_t offset, std::size_t size) {
  { mapping.address(offset, size) } -> std::same_as<hyle::Result<const void*>>;
  { mapping.bytes(offset, size) }
      -> std::same_as<hyle::Result<std::span<const std::byte>>>;
  { mapping.writable_bytes(offset, size) }
      -> std::same_as<hyle::Result<std::span<std::byte>>>;
  { mapping.release_ranges(ranges) }
      -> std::same_as<hyle::Result<std::size_t>>;
  { mapping.reset() } -> std::same_as<hyle::Status>;
  { shared.writable_data() } -> std::same_as<hyle::Result<void*>>;
  { shared.writable_bytes() }
      -> std::same_as<hyle::Result<std::span<std::byte>>>;
  { shared.reset() } -> std::same_as<hyle::Status>;
});

static_assert(requires(LifetimePlan plan, PlannedBuffers buffers,
                       std::span<const LifetimeRequest> requests,
                       std::span<const MemoryResourceRef> resources,
                       AllocatorRef allocator, std::size_t index) {
  { LifetimePlanner::build(requests, resources) }
      -> std::same_as<hyle::Result<LifetimePlan>>;
  { LifetimePlanner::build(requests, resources, allocator) }
      -> std::same_as<hyle::Result<LifetimePlan>>;
  { plan.placement(index) } -> std::same_as<hyle::Result<MemoryPlacement>>;
  { plan.resource_plan(index) }
      -> std::same_as<hyle::Result<LifetimeResourcePlan>>;
  { plan.backing_plan(index, index) }
      -> std::same_as<hyle::Result<LifetimeBackingPlan>>;
  { plan.instantiate(resources) }
      -> std::same_as<hyle::Result<PlannedBuffers>>;
  { plan.instantiate(resources, allocator) }
      -> std::same_as<hyle::Result<PlannedBuffers>>;
  { buffers.backing_count(index) }
      -> std::same_as<hyle::Result<std::size_t>>;
  { buffers.backing(index, index) }
      -> std::same_as<hyle::Result<BufferView>>;
});

template <class A>
concept AllocatorSurface = requires(A a, void* p, std::size_t n) {
  { a.valid() } -> std::same_as<bool>;
  { a.allocate(n, n) } -> std::same_as<void*>;
  { a.deallocate(p, n, n) } -> std::same_as<void>;
  { a.reallocate(p, n, n, n) } -> std::same_as<void*>;
};
static_assert(AllocatorSurface<AllocatorRef>);

template <class R>
concept RegionSurface = requires(R r, std::size_t n) {
  { r.allocate(n, n) } -> std::same_as<void*>;
  { r.clear() } -> std::same_as<void>;
  { r.as_allocator() } -> std::same_as<AllocatorRef>;
};
static_assert(RegionSurface<FixedArena>);
static_assert(RegionSurface<Arena>);
static_assert(RegionSurface<PooledArena>);

template <class P>
concept FixedBlockPoolSurface = requires(P p, void* block) {
  { p.valid() } -> std::same_as<bool>;
  { p.allocate() } -> std::same_as<void*>;
  { p.deallocate(block) } -> std::same_as<void>;
  { p.capacity() } -> std::same_as<std::uint32_t>;
  { p.as_allocator() } -> std::same_as<AllocatorRef>;
};
static_assert(FixedBlockPoolSurface<ConcurrentBlockPool>);

static_assert(requires(BlockPoolCache cache, void* block) {
  { cache.valid() } -> std::same_as<bool>;
  { cache.allocate() } -> std::same_as<void*>;
  { cache.deallocate(block) } -> std::same_as<void>;
  { cache.publish() } -> std::same_as<void>;
  { cache.cached() } -> std::same_as<std::uint32_t>;
  { cache.as_allocator() } -> std::same_as<AllocatorRef>;
});

template <class P>
concept ObjectPoolSurface = requires(P p, int* value) {
  { p.create(1) } -> std::same_as<int*>;
  { p.destroy(value) } -> std::same_as<void>;
  { p.try_loan(1) } -> std::same_as<PoolLoan<int, P>>;
};
static_assert(ObjectPoolSurface<ObjectPool<int>>);
static_assert(ObjectPoolSurface<ObjectPoolCache<int>>);
static_assert(ObjectPoolSurface<GrowableObjectPool<int>>);

static_assert(requires(FixedHeap p, void* ptr, std::size_t n) {
  { p.allocate(n, n) } -> std::same_as<void*>;
  { p.deallocate(ptr) } -> std::same_as<void>;
  { p.reallocate(ptr, n, n, n) } -> std::same_as<void*>;
  { p.reset() } -> std::same_as<void>;
  { p.peak() } -> std::same_as<std::size_t>;
  { p.as_allocator() } -> std::same_as<AllocatorRef>;
});

static_assert(requires(QuotaAllocator q) {
  { q.get() } -> std::same_as<AllocatorRef>;
  { q.used() } -> std::same_as<std::size_t>;
  { q.peak() } -> std::same_as<std::size_t>;
  { q.limit() } -> std::same_as<std::size_t>;
  { q.remaining() } -> std::same_as<std::size_t>;
});

static_assert(requires(FixedArena r) {
  { r.remaining() } -> std::same_as<std::size_t>;
});

static_assert(requires(StatsAllocator s) {
  { StatsAllocator::create(system_allocator()) }
      -> std::same_as<hyle::Result<StatsAllocator>>;
  { s.get() } -> std::same_as<AllocatorRef>;
  { s.snapshot() } -> std::same_as<AllocStatsSnapshot>;
  { s.balanced() } -> std::same_as<bool>;
  { s.leaks() } -> std::same_as<LeakTracker*>;
});

static_assert(requires(LeakTracker tracker) {
  { tracker.outstanding() } -> std::same_as<std::size_t>;
  { tracker.dropped_records() } -> std::same_as<std::uint64_t>;
  { tracker.complete() } -> std::same_as<bool>;
});

static_assert(std::is_move_constructible_v<Arena>);
static_assert(!std::is_copy_constructible_v<Arena>);
static_assert(!std::is_move_constructible_v<FixedArena>);
static_assert(!std::is_move_constructible_v<FixedHeap>);
static_assert(!std::is_move_constructible_v<ConcurrentBlockPool>);
static_assert(!std::is_move_constructible_v<BlockPoolCache>);
static_assert(std::is_move_constructible_v<FiberStack>);
static_assert(requires(FiberStack stack, FiberStackPool pool) {
  { FiberStack::allocate(kFiberStackDefault) }
      -> std::same_as<hyle::Result<FiberStack>>;
  { pool.acquire() } -> std::same_as<hyle::Result<FiberStack>>;
  { pool.capacity() } -> std::same_as<std::size_t>;
  { pool.trim() } -> std::same_as<std::size_t>;
  { pool.clear() } -> std::same_as<void>;
  { stack.reset() } -> std::same_as<hyle::Status>;
});
static_assert(requires {
  { reserve_large_page_arena(0) }
      -> std::same_as<hyle::Result<LargePageArenaReservationReport>>;
  { LargeMapping::allocate_interleaved(
        4096, std::span<const unsigned>{}, 0, LargeMappingFlags::kNone) }
      -> std::same_as<hyle::Result<LargeMapping>>;
});

TEST(MemApiSurface, PublicCapabilityFamiliesCompileTogether) {
  EXPECT_EQ(kFiberStackSmall, 16u * 1024u);
  EXPECT_EQ(kFiberStackDefault, 64u * 1024u);
  EXPECT_EQ(kFiberStackLarge, 256u * 1024u);
  EXPECT_GE(kDefaultAlignment, alignof(void*));
}

// Compile-checked user recipes. Keep these deliberately short and mirrored in
// docs/guide/mem.md; they test the public shape rather than internal details.
TEST(MemUsageRecipe, RequestArena) {
  Arena request(system_allocator());
  ASSERT_NE(request.allocate(128, 64), nullptr);
  request.clear();
  EXPECT_EQ(request.bytes_used(), 0u);
}

TEST(MemUsageRecipe, TypedCachedPool) {
  ObjectPool<int> core(system_allocator(), 32);
  ASSERT_TRUE(core.valid());
  ObjectPoolCache<int> local(core, 8, 2);
  ASSERT_TRUE(local.valid());
  {
    auto value = local.try_loan(7);
    ASSERT_TRUE(value);
    EXPECT_EQ(*value, 7);
  }
  local.publish();
}

TEST(MemUsageRecipe, QuotaAndStatsComposition) {
  StatsAllocator stats = StatsAllocator::create(system_allocator()).value();
  QuotaAllocator quota(stats.get(), 256);
  AllocatorRef bounded = quota.get();
  void* p = bounded.allocate(96, 16);
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(quota.remaining(), 160u);
  bounded.deallocate(p, 96, 16);
  EXPECT_TRUE(stats.balanced());
  EXPECT_EQ(quota.remaining(), 256u);
}

}  // namespace

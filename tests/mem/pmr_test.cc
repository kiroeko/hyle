#include "hyle/mem/pmr.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory_resource>
#include <vector>

#include "hyle/mem/allocator.h"

using hyle::mem::Allocator;
using hyle::mem::AlignedPmrResource;
using hyle::mem::PmrResource;
using hyle::mem::system_allocator;
using hyle::mem::system_memory_resource;

namespace {
struct EqualityState {
  int frees_a = 0;
  int frees_b = 0;
  int reallocs_b = 0;
};

void* equality_alloc(void*, std::size_t size, std::size_t align) noexcept {
  return system_allocator().allocate(size, align);
}
void equality_free_a(void* state, void* ptr, std::size_t size,
                     std::size_t align) noexcept {
  ++static_cast<EqualityState*>(state)->frees_a;
  system_allocator().deallocate(ptr, size, align);
}
void equality_free_b(void* state, void* ptr, std::size_t size,
                     std::size_t align) noexcept {
  ++static_cast<EqualityState*>(state)->frees_b;
  system_allocator().deallocate(ptr, size, align);
}
void* equality_realloc(void*, void* ptr, std::size_t old_size,
                       std::size_t new_size, std::size_t align) noexcept {
  return system_allocator().reallocate(ptr, old_size, new_size, align);
}
void* equality_realloc_b(void* state, void* ptr, std::size_t old_size,
                         std::size_t new_size, std::size_t align) noexcept {
  ++static_cast<EqualityState*>(state)->reallocs_b;
  return system_allocator().reallocate(ptr, old_size, new_size, align);
}
}  // namespace

// A pmr container built on a HYLE-backed resource allocates through HYLE.
TEST(SystemMemoryResource, IsProcessLifetimeAndServesPmrVector) {
  std::pmr::memory_resource* first = system_memory_resource();
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first, system_memory_resource());
  EXPECT_TRUE(first->is_equal(*system_memory_resource()));

  std::pmr::vector<int> values(first);
  for (int i = 0; i < 1000; ++i) values.push_back(i);
  EXPECT_EQ(values.front(), 0);
  EXPECT_EQ(values.back(), 999);
}

TEST(PmrResource, ServesPmrVector) {
  PmrResource res(system_allocator());
  std::pmr::vector<int> v(&res);
  for (int i = 0; i < 1000; ++i) v.push_back(i);
  ASSERT_EQ(v.size(), 1000u);
  for (int i = 0; i < 1000; ++i) EXPECT_EQ(v[i], i);
}

// One resource serves containers of different element types (pmr type-erases T,
// unlike clay's per-type stl_allocator<T>).
TEST(PmrResource, OneResourceManyTypes) {
  PmrResource res(system_allocator());
  std::pmr::vector<std::uint8_t> bytes(&res);
  std::pmr::vector<double> doubles(&res);
  bytes.resize(100);
  doubles.resize(100);
  EXPECT_EQ(bytes.size(), 100u);
  EXPECT_EQ(doubles.size(), 100u);
}

// do_is_equal: same backing -> equal; different state -> not equal.
TEST(PmrResource, EqualityFollowsBacking) {
  PmrResource a(system_allocator());
  PmrResource b(system_allocator());
  EXPECT_TRUE(a.is_equal(b));  // same {fn,state}

  int dummy = 0;
  // A distinct allocator (same fns, different state pointer) is not interchangeable.
  Allocator sys = system_allocator();
  Allocator other(sys.alloc_fn(), sys.free_fn(), sys.realloc_fn(), &dummy);
  PmrResource c(other);
  EXPECT_FALSE(a.is_equal(c));
}

TEST(PmrResource, DifferentFreeFunctionIsNotEqual) {
  EqualityState state;
  Allocator a(&equality_alloc, &equality_free_a, &equality_realloc, &state);
  Allocator b(&equality_alloc, &equality_free_b, &equality_realloc, &state);
  PmrResource ra(a);
  PmrResource rb(b);
  EXPECT_FALSE(ra.is_equal(rb));
}

TEST(PmrResource, DifferentReallocateFunctionIsConservativelyNotEqual) {
  EqualityState state;
  Allocator a(&equality_alloc, &equality_free_a, &equality_realloc, &state);
  Allocator b(&equality_alloc, &equality_free_a, &equality_realloc_b, &state);
  PmrResource ra(a);
  PmrResource rb(b);
  EXPECT_FALSE(ra.is_equal(rb));
}

TEST(PmrResourceDeathTest, InvalidAllocatorTrapsAtConstruction) {
  EXPECT_DEATH(
      {
        PmrResource resource(Allocator{});
        (void)resource;
      },
      ".*");
}

TEST(PmrResource, ExposesAllocator) {
  PmrResource res(system_allocator());
  EXPECT_EQ(res.allocator().alloc_fn(), system_allocator().alloc_fn());
}

TEST(AlignedPmrResource, AlignsContainerBaseWithoutChangingElementStride) {
  AlignedPmrResource resource(system_allocator(), 64);
  std::pmr::vector<std::uint8_t> bytes(&resource);
  bytes.resize(257);
  ASSERT_NE(bytes.data(), nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(bytes.data()) % 64, 0u);
  EXPECT_EQ(resource.minimum_alignment(), 64u);
}

TEST(AlignedPmrResource, EqualityIncludesMinimumAlignment) {
  AlignedPmrResource a(system_allocator(), 64);
  AlignedPmrResource b(system_allocator(), 64);
  AlignedPmrResource c(system_allocator(), 128);
  PmrResource plain(system_allocator());
  EXPECT_TRUE(a.is_equal(b));
  EXPECT_FALSE(a.is_equal(c));
  EXPECT_FALSE(a.is_equal(plain));
}

TEST(AlignedPmrResourceDeathTest, InvalidAlignmentTrapsAtConstruction) {
  EXPECT_DEATH(
      {
        AlignedPmrResource resource(system_allocator(), 3);
        (void)resource;
      },
      ".*");
}

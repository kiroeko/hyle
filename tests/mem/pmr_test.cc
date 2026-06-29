#include "hyle/mem/pmr.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory_resource>
#include <vector>

#include "hyle/mem/allocator.h"

using hyle::mem::Allocator;
using hyle::mem::PmrResource;
using hyle::mem::system_allocator;

// A pmr container built on a HYLE-backed resource allocates through HYLE.
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

TEST(PmrResource, ExposesAllocator) {
  PmrResource res(system_allocator());
  EXPECT_EQ(res.allocator().alloc_fn(), system_allocator().alloc_fn());
}

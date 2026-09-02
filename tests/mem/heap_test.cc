#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <thread>
#include <utility>

#include "hyle/mem/heap.h"

namespace {

using namespace hyle::mem;

TEST(Heap, IsolatedOwnerSupportsIndividualAndBulkRelease) {
  auto created = Heap::create();
  ASSERT_TRUE(created.has_value()) << created.error().message();
  Heap heap = std::move(*created);
  ASSERT_TRUE(heap.valid());
  static_assert(sizeof(Heap) == sizeof(void*));

  AllocatorRef allocator = heap.allocator();
  ASSERT_TRUE(allocator.valid());
  void* first = allocator.allocate(80, 64);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(first) & 63u, 0u);
  allocator.deallocate(first, 80, 64);

  void* invalidated_by_clear = allocator.allocate(96);
  ASSERT_NE(invalidated_by_clear, nullptr);
  ASSERT_TRUE(heap.clear().ok());
  void* after_clear = allocator.allocate(48);
  ASSERT_NE(after_clear, nullptr);
  allocator.deallocate(after_clear, 48);

  heap.release();
  EXPECT_FALSE(heap.valid());
  EXPECT_EQ(allocator.allocate(16), nullptr);
}

TEST(Heap, SupportsOwnerReallocationAndRemoteFree) {
  Heap heap = Heap::create().value();
  AllocatorRef allocator = heap.allocator();
  void* pointer = allocator.allocate(32);
  ASSERT_NE(pointer, nullptr);
  pointer = allocator.reallocate(pointer, 32, 128);
  ASSERT_NE(pointer, nullptr);

  std::thread remote([allocator, pointer] {
    allocator.deallocate(pointer, 128);
  });
  remote.join();
}

TEST(Heap, MoveTransfersOwnerAndInvalidatesOldBorrows) {
  Heap source = Heap::create().value();
  AllocatorRef old_borrow = source.allocator();
  Heap destination = std::move(source);

  EXPECT_FALSE(source.valid());
  EXPECT_TRUE(destination.valid());
  EXPECT_EQ(old_borrow.allocate(8), nullptr);

  AllocatorRef current = destination.allocator();
  void* pointer = current.allocate(8);
  ASSERT_NE(pointer, nullptr);
  current.deallocate(pointer, 8);
}

}  // namespace

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <random>
#include <span>

#include "hyle/container/byte_ring_buffer.h"
#include "hyle/mem/allocator.h"

namespace {
using hyle::container::ByteRingBuffer;

std::span<const std::byte> bytes(const char *text, std::size_t size) {
  return {reinterpret_cast<const std::byte *>(text), size};
}

void* fail_allocate(void*, std::size_t, std::size_t) noexcept {
  return nullptr;
}

void fail_deallocate(void*, void*, std::size_t, std::size_t) noexcept {}

void* fail_reallocate(void*, void*, std::size_t, std::size_t,
                      std::size_t) noexcept {
  return nullptr;
}
} // namespace

TEST(ByteRingBuffer, CopyWriteReadWrapsWithoutLosingOrder) {
  ByteRingBuffer ring(8);
  EXPECT_EQ(ring.write(bytes("abcdef", 6)), 6u);
  std::array<std::byte, 4> prefix{};
  EXPECT_EQ(ring.read(prefix), 4u);
  EXPECT_EQ(ring.write(bytes("ghijkl", 6)), 6u);
  EXPECT_TRUE(ring.full());

  std::array<std::byte, 8> output{};
  EXPECT_EQ(ring.read(output), 8u);
  const char expected[] = "efghijkl";
  EXPECT_EQ(std::memcmp(output.data(), expected, output.size()), 0);
  EXPECT_TRUE(ring.empty());
}

TEST(ByteRingBuffer, ZeroCopyWritableAndReadableSegments) {
  ByteRingBuffer ring(8);
  auto writable = ring.writable();
  ASSERT_EQ(writable.first.size(), 8u);
  std::memcpy(writable.first.data(), "123456", 6);
  ring.commit_write(6);
  ring.consume(5);

  writable = ring.writable();
  ASSERT_EQ(writable.first.size(), 2u);
  ASSERT_EQ(writable.second.size(), 5u);
  std::memcpy(writable.first.data(), "78", 2);
  std::memcpy(writable.second.data(), "90", 2);
  ring.commit_write(4);

  const auto readable = ring.readable();
  ASSERT_EQ(readable.first.size(), 3u);
  ASSERT_EQ(readable.second.size(), 2u);
  EXPECT_EQ(std::memcmp(readable.first.data(), "678", 3), 0);
  EXPECT_EQ(std::memcmp(readable.second.data(), "90", 2), 0);
}

TEST(ByteRingBuffer, ReserveLinearizesAndPreservesData) {
  ByteRingBuffer ring(4);
  ring.write(bytes("abcd", 4));
  ring.consume(3);
  ring.write(bytes("efg", 3));
  ASSERT_FALSE(ring.readable().second.empty());
  ASSERT_TRUE(ring.try_reserve(9));
  EXPECT_GE(ring.capacity(), 9u);
  EXPECT_TRUE(ring.readable().second.empty());
  std::array<std::byte, 4> output{};
  ring.read(output);
  EXPECT_EQ(std::memcmp(output.data(), "defg", 4), 0);
}

TEST(ByteRingBuffer, CopyAndMoveHaveIndependentStorage) {
  ByteRingBuffer source(8);
  source.write(bytes("payload", 7));
  ByteRingBuffer copy = source;
  source.consume(2);
  EXPECT_EQ(copy.size(), 7u);
  ByteRingBuffer moved = std::move(copy);
  EXPECT_TRUE(copy.empty());
  EXPECT_EQ(moved.size(), 7u);
}

TEST(ByteRingBuffer, AssignmentsAndFallibleReservePreserveState) {
  ByteRingBuffer source(8);
  source.write(bytes("payload", 7));

  ByteRingBuffer copy_target(4);
  copy_target.write(bytes("old", 3));
  copy_target = source;
  source.consume(2);
  EXPECT_EQ(copy_target.size(), 7u);

  ByteRingBuffer move_target(2);
  move_target = std::move(copy_target);
  EXPECT_TRUE(copy_target.empty());
  std::array<std::byte, 7> copied{};
  EXPECT_EQ(move_target.read(copied), copied.size());
  EXPECT_EQ(std::memcmp(copied.data(), "payload", copied.size()), 0);

  hyle::mem::AllocatorRef failing(&fail_allocate, &fail_deallocate,
                                  &fail_reallocate, nullptr);
  ByteRingBuffer unavailable(failing);
  EXPECT_FALSE(unavailable.try_reserve(32));
  EXPECT_EQ(unavailable.capacity(), 0u);
  EXPECT_TRUE(unavailable.empty());
}

TEST(ByteRingBuffer, RandomTransfersMatchByteQueueModel) {
  ByteRingBuffer ring(64);
  std::deque<std::byte> model;
  std::mt19937 random(0x42595445u);
  std::array<std::byte, 32> input{};
  std::array<std::byte, 32> output{};
  for (std::size_t step = 0; step < 10000; ++step) {
    if ((random() & 1u) != 0) {
      const std::size_t requested = random() % (input.size() + 1);
      for (std::size_t i = 0; i < requested; ++i)
        input[i] = static_cast<std::byte>((step + i) & 0xffu);
      const std::size_t written =
          ring.write(std::span<const std::byte>(input.data(), requested));
      for (std::size_t i = 0; i < written; ++i)
        model.push_back(input[i]);
    } else {
      const std::size_t requested = random() % (output.size() + 1);
      const std::size_t read =
          ring.read(std::span<std::byte>(output.data(), requested));
      ASSERT_EQ(read, std::min(requested, model.size()));
      for (std::size_t i = 0; i < read; ++i) {
        ASSERT_EQ(output[i], model.front());
        model.pop_front();
      }
    }
    ASSERT_EQ(ring.size(), model.size());
  }
}

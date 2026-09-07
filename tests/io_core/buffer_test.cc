#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <span>

#include "hyle/io/buffer.h"

namespace {

using hyle::io::BufferSequenceCursor;
using hyle::io::ConstBuffer;
using hyle::io::MutableBuffer;
using hyle::io::detail::BufferSequenceStorage;

TEST(IoBuffer, CursorAndProgressFollowLogicalSegments) {
  std::array<std::byte, 9> bytes{};
  const MutableBuffer sequence[] = {
      {bytes.data(), 2}, {bytes.data() + 2, 0}, {bytes.data() + 2, 3},
      {bytes.data() + 5, 4}};
  BufferSequenceStorage<MutableBuffer> storage(
      std::span<const MutableBuffer>{sequence}, BufferSequenceCursor{0, 1});
  ASSERT_TRUE(storage.ok()) << storage.status().message();
  EXPECT_EQ(storage.native_count(), 3u);
  EXPECT_EQ(storage.total_size(), 8u);

  auto first = storage.progress(1);
  EXPECT_EQ(first.next.segment, 2u);
  EXPECT_EQ(first.next.offset, 0u);
  EXPECT_FALSE(first.complete);

  auto middle = storage.progress(3);
  EXPECT_EQ(middle.next.segment, 2u);
  EXPECT_EQ(middle.next.offset, 2u);
  EXPECT_FALSE(middle.complete);

  auto complete = storage.progress(8);
  EXPECT_EQ(complete.next.segment, std::size(sequence));
  EXPECT_EQ(complete.next.offset, 0u);
  EXPECT_TRUE(complete.complete);
}

TEST(IoBuffer, RejectsRatherThanTruncatesNonPortableSequences) {
  std::byte byte{};
  std::array<ConstBuffer, hyle::io::kMaxBufferSequenceSegments + 1> too_many{};
  BufferSequenceStorage<ConstBuffer> segment_limit(too_many);
  EXPECT_TRUE(segment_limit.status().IsOutOfRange());

  const ConstBuffer too_large{
      &byte, hyle::io::kMaxBufferSequenceBytes + std::size_t{1}};
  BufferSequenceStorage<ConstBuffer> byte_limit(
      std::span<const ConstBuffer>{&too_large, 1});
  EXPECT_TRUE(byte_limit.status().IsOutOfRange());

  const MutableBuffer null_nonempty{nullptr, 1};
  BufferSequenceStorage<MutableBuffer> null_address(
      std::span<const MutableBuffer>{&null_nonempty, 1});
  EXPECT_TRUE(null_address.status().IsInvalidArgument());
}

} // namespace

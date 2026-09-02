#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <span>

#include "hyle/mem/buffer.h"

namespace {

using namespace hyle::mem;

TEST(CompositeBuffer, OwnsSegmentsAndOperatesAcrossBoundariesWithoutStaging) {
  std::array<Buffer, 2> segments{
      Buffer::create(system_resource(), 64, 64).value(),
      Buffer::create(system_resource(), 32, 32).value(),
  };
  void *first_handle = segments[0].native_handle();
  void *second_handle = segments[1].native_handle();
  CompositeBuffer composite = CompositeBuffer::create(segments).value();
  EXPECT_FALSE(segments[0].valid());
  EXPECT_FALSE(segments[1].valid());
  EXPECT_EQ(composite.size(), 96u);
  EXPECT_EQ(composite.segment_count(), 2u);
  EXPECT_EQ(composite.segment(0)->native_handle(), first_handle);
  EXPECT_EQ(composite.segment(1)->native_handle(), second_handle);

  std::array<std::byte, 40> source{};
  for (std::size_t index = 0; index < source.size(); ++index) {
    source[index] = static_cast<std::byte>(index + 1);
  }
  CompositeBufferView crossing = composite.view().subview(48, 40).value();
  ASSERT_EQ(crossing.piece_count(), 2u);
  EXPECT_EQ(crossing.piece(0)->size(), 16u);
  EXPECT_EQ(crossing.piece(1)->size(), 24u);
  EXPECT_TRUE(crossing.write(source).ok());

  std::array<std::byte, 40> observed{};
  EXPECT_TRUE(crossing.read(observed).ok());
  EXPECT_EQ(observed, source);
}

TEST(CompositeBuffer, SubviewsClipPiecesAndRemainBoundAcrossOwnerMove) {
  std::array<Buffer, 3> segments{
      Buffer::create(system_resource(), 16).value(),
      Buffer::create(system_resource(), 16).value(),
      Buffer::create(system_resource(), 16).value(),
  };
  CompositeBuffer owner = CompositeBuffer::create(segments).value();
  CompositeBufferView view = owner.view().subview(8, 32).value();
  CompositeBuffer moved = std::move(owner);
  ASSERT_FALSE(owner.valid());
  ASSERT_TRUE(moved.valid());
  ASSERT_EQ(view.piece_count(), 3u);
  EXPECT_EQ(view.piece(0)->size(), 8u);
  EXPECT_EQ(view.piece(1)->size(), 16u);
  EXPECT_EQ(view.piece(2)->size(), 8u);
  EXPECT_TRUE(view.fill(std::byte{0x6b}).ok());

  const CompositeBuffer &read_only_owner = moved;
  ConstCompositeBufferView read_only =
      read_only_owner.view().subview(8, 32).value();
  std::array<std::byte, 32> observed{};
  EXPECT_TRUE(read_only.read(observed).ok());
  for (std::byte value : observed)
    EXPECT_EQ(value, std::byte{0x6b});
}

TEST(CompositeBuffer, ValidatesBeforeMovingInputsAndSupportsEmptyOwner) {
  std::array<Buffer, 2> invalid{Buffer::create(system_resource(), 16).value(),
                                Buffer{}};
  auto rejected = CompositeBuffer::create(invalid);
  EXPECT_TRUE(rejected.error().IsInvalidArgument());
  EXPECT_TRUE(invalid[0].valid());

  std::span<Buffer> no_segments;
  CompositeBuffer empty = CompositeBuffer::create(no_segments).value();
  EXPECT_TRUE(empty.valid());
  EXPECT_EQ(empty.size(), 0u);
  EXPECT_EQ(empty.view().piece_count(), 0u);
  EXPECT_TRUE(empty.view().fill(std::byte{0}).ok());
  EXPECT_TRUE(empty.view().subview(1).error().IsOutOfRange());
}

TEST(CompositeBuffer, RejectsWrongRangeSizesAndPieceIndices) {
  std::array<Buffer, 1> segments{Buffer::create(system_resource(), 16).value()};
  CompositeBuffer composite = CompositeBuffer::create(segments).value();
  std::array<std::byte, 8> short_span{};
  EXPECT_TRUE(composite.view().write(short_span).IsInvalidArgument());
  EXPECT_TRUE(composite.view().read(short_span).IsInvalidArgument());
  EXPECT_TRUE(composite.view().piece(1).error().IsOutOfRange());
  EXPECT_TRUE(composite.segment(1).error().IsOutOfRange());
}

TEST(CompositeBuffer, CopiesBetweenDifferentSegmentationsWithoutFlattening) {
  std::array<Buffer, 2> source_segments{
      Buffer::create(system_resource(), 10).value(),
      Buffer::create(system_resource(), 22).value(),
  };
  std::array<Buffer, 2> destination_segments{
      Buffer::create(system_resource(), 16).value(),
      Buffer::create(system_resource(), 16).value(),
  };
  CompositeBuffer source = CompositeBuffer::create(source_segments).value();
  CompositeBuffer destination =
      CompositeBuffer::create(destination_segments).value();
  std::array<std::byte, 32> expected{};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    expected[index] = static_cast<std::byte>(index + 11);
  }
  EXPECT_TRUE(source.view().write(expected).ok());
  const CompositeBuffer &const_source = source;
  EXPECT_TRUE(destination.view().copy_from(const_source.view()).ok());
  std::array<std::byte, 32> observed{};
  EXPECT_TRUE(destination.view().read(observed).ok());
  EXPECT_EQ(observed, expected);

  EXPECT_TRUE(destination.view()
                  .subview(0, 16)
                  ->copy_from(destination.view().subview(8, 16).value())
                  .IsInvalidArgument());
}

} // namespace

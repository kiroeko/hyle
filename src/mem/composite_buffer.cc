#include "hyle/mem/buffer.h"

#include <algorithm>
#include <limits>
#include <new>
#include <utility>

#include "hyle/core/compiler.h"
#include "hyle/mem/allocator.h"

namespace hyle::mem {
namespace {

struct CompositeSegment {
  Buffer buffer;
  std::size_t logical_offset = 0;
};

struct CompositeImpl {
  CompositeSegment *segments = nullptr;
  std::size_t count = 0;
  std::size_t size = 0;
};

CompositeImpl *as_composite(void *pointer) noexcept {
  return static_cast<CompositeImpl *>(pointer);
}

const CompositeImpl *as_composite(const void *pointer) noexcept {
  return static_cast<const CompositeImpl *>(pointer);
}

void destroy_composite(CompositeImpl *impl) noexcept {
  if (impl == nullptr)
    return;
  if (impl->segments != nullptr) {
    for (std::size_t index = impl->count; index != 0; --index) {
      impl->segments[index - 1].~CompositeSegment();
    }
    system_allocator().deallocate(impl->segments,
                                  impl->count * sizeof(CompositeSegment),
                                  alignof(CompositeSegment));
  }
  impl->~CompositeImpl();
  system_allocator().deallocate(impl, sizeof(CompositeImpl),
                                alignof(CompositeImpl));
}

std::size_t first_segment(const CompositeImpl &impl,
                          std::size_t logical_offset) noexcept {
  HYLE_FATAL_ASSERT(logical_offset < impl.size);
  std::size_t begin = 0;
  std::size_t end = impl.count;
  while (begin < end) {
    const std::size_t middle = begin + (end - begin) / 2;
    const CompositeSegment &segment = impl.segments[middle];
    const std::size_t segment_end =
        segment.logical_offset + segment.buffer.size();
    if (segment_end <= logical_offset) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  HYLE_FATAL_ASSERT(begin < impl.count);
  return begin;
}

std::size_t piece_count(const CompositeImpl &impl, std::size_t offset,
                        std::size_t size) noexcept {
  if (size == 0)
    return 0;
  const std::size_t first = first_segment(impl, offset);
  const std::size_t last = first_segment(impl, offset + size - 1);
  return last - first + 1;
}

Result<BufferView> mutable_piece(CompositeImpl &impl, std::size_t view_offset,
                                 std::size_t view_size,
                                 std::size_t index) noexcept {
  const std::size_t count = piece_count(impl, view_offset, view_size);
  if (index >= count) {
    return std::unexpected(
        Status::OutOfRange("composite piece index is out of range"));
  }
  const std::size_t segment_index = first_segment(impl, view_offset) + index;
  CompositeSegment &segment = impl.segments[segment_index];
  const std::size_t view_end = view_offset + view_size;
  const std::size_t segment_end =
      segment.logical_offset + segment.buffer.size();
  const std::size_t begin = (std::max)(view_offset, segment.logical_offset);
  const std::size_t end = (std::min)(view_end, segment_end);
  return segment.buffer.view().subview(begin - segment.logical_offset,
                                       end - begin);
}

Result<ConstBufferView> const_piece(const CompositeImpl &impl,
                                    std::size_t view_offset,
                                    std::size_t view_size,
                                    std::size_t index) noexcept {
  const std::size_t count = piece_count(impl, view_offset, view_size);
  if (index >= count) {
    return std::unexpected(
        Status::OutOfRange("composite piece index is out of range"));
  }
  const std::size_t segment_index = first_segment(impl, view_offset) + index;
  const CompositeSegment &segment = impl.segments[segment_index];
  const std::size_t view_end = view_offset + view_size;
  const std::size_t segment_end =
      segment.logical_offset + segment.buffer.size();
  const std::size_t begin = (std::max)(view_offset, segment.logical_offset);
  const std::size_t end = (std::min)(view_end, segment_end);
  return segment.buffer.view().subview(begin - segment.logical_offset,
                                       end - begin);
}

} // namespace

Result<CompositeBuffer>
CompositeBuffer::create(std::span<Buffer> segments) noexcept {
  if (segments.size() >
      (std::numeric_limits<std::size_t>::max)() / sizeof(CompositeSegment)) {
    return std::unexpected(
        Status::OutOfRange("composite segment metadata overflows"));
  }
  std::size_t total = 0;
  for (const Buffer &segment : segments) {
    if (!segment.valid()) {
      return std::unexpected(
          Status::InvalidArgument("composite segment is invalid"));
    }
    if (segment.size() == 0) {
      return std::unexpected(
          Status::InvalidArgument("composite segments must be non-empty"));
    }
    if (segment.size() > (std::numeric_limits<std::size_t>::max)() - total) {
      return std::unexpected(
          Status::OutOfRange("composite logical size overflows"));
    }
    total += segment.size();
  }

  void *impl_storage = system_allocator().allocate(sizeof(CompositeImpl),
                                                   alignof(CompositeImpl));
  if (impl_storage == nullptr) {
    return std::unexpected(Status::ResourceExhausted(
        "composite owner metadata allocation failed"));
  }
  auto *impl =
      ::new (impl_storage) CompositeImpl{nullptr, segments.size(), total};
  if (!segments.empty()) {
    void *segment_storage = system_allocator().allocate(
        segments.size() * sizeof(CompositeSegment), alignof(CompositeSegment));
    if (segment_storage == nullptr) {
      destroy_composite(impl);
      return std::unexpected(Status::ResourceExhausted(
          "composite segment metadata allocation failed"));
    }
    impl->segments = static_cast<CompositeSegment *>(segment_storage);
    std::size_t offset = 0;
    for (std::size_t index = 0; index < segments.size(); ++index) {
      ::new (static_cast<void *>(impl->segments + index))
          CompositeSegment{std::move(segments[index]), offset};
      offset += impl->segments[index].buffer.size();
    }
  }
  return CompositeBuffer(impl);
}

CompositeBuffer::CompositeBuffer(CompositeBuffer &&other) noexcept
    : impl_(std::exchange(other.impl_, nullptr)) {}

CompositeBuffer &CompositeBuffer::operator=(CompositeBuffer &&other) noexcept {
  if (this != &other) {
    reset();
    impl_ = std::exchange(other.impl_, nullptr);
  }
  return *this;
}

CompositeBuffer::~CompositeBuffer() { reset(); }

std::size_t CompositeBuffer::size() const noexcept {
  return valid() ? as_composite(impl_)->size : 0;
}

std::size_t CompositeBuffer::segment_count() const noexcept {
  return valid() ? as_composite(impl_)->count : 0;
}

Result<BufferView> CompositeBuffer::segment(std::size_t index) noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("composite buffer is invalid"));
  }
  CompositeImpl &impl = *as_composite(impl_);
  if (index >= impl.count) {
    return std::unexpected(
        Status::OutOfRange("composite segment index is out of range"));
  }
  return impl.segments[index].buffer.view();
}

Result<ConstBufferView>
CompositeBuffer::segment(std::size_t index) const noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("composite buffer is invalid"));
  }
  const CompositeImpl &impl = *as_composite(impl_);
  if (index >= impl.count) {
    return std::unexpected(
        Status::OutOfRange("composite segment index is out of range"));
  }
  const Buffer &buffer = impl.segments[index].buffer;
  return buffer.view();
}

CompositeBufferView CompositeBuffer::view() noexcept {
  return valid() ? CompositeBufferView(impl_, 0, size())
                 : CompositeBufferView{};
}

ConstCompositeBufferView CompositeBuffer::view() const noexcept {
  return valid() ? ConstCompositeBufferView(impl_, 0, size())
                 : ConstCompositeBufferView{};
}

void CompositeBuffer::reset() noexcept {
  destroy_composite(as_composite(impl_));
  impl_ = nullptr;
}

Result<CompositeBufferView>
CompositeBufferView::subview(std::size_t offset,
                             std::size_t size) const noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("composite buffer view is invalid"));
  }
  if (offset > size_) {
    return std::unexpected(
        Status::OutOfRange("composite subview offset exceeds its range"));
  }
  const std::size_t available = size_ - offset;
  if (size == std::dynamic_extent)
    size = available;
  if (size > available ||
      offset_ > (std::numeric_limits<std::size_t>::max)() - offset) {
    return std::unexpected(
        Status::OutOfRange("composite subview size exceeds its range"));
  }
  return CompositeBufferView(impl_, offset_ + offset, size);
}

std::size_t CompositeBufferView::piece_count() const noexcept {
  return valid()
             ? ::hyle::mem::piece_count(*as_composite(impl_), offset_, size_)
             : 0;
}

Result<BufferView>
CompositeBufferView::piece(std::size_t index) const noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("composite buffer view is invalid"));
  }
  return mutable_piece(*as_composite(impl_), offset_, size_, index);
}

Status
CompositeBufferView::write(std::span<const std::byte> source) const noexcept {
  if (!valid())
    return Status::FailedPrecondition("composite buffer view is invalid");
  if (source.size() != size_) {
    return Status::InvalidArgument(
        "write size must equal composite buffer view size");
  }
  std::size_t consumed = 0;
  const std::size_t count = piece_count();
  for (std::size_t index = 0; index < count; ++index) {
    HYLE_ASSIGN_OR_RETURN(BufferView current, piece(index));
    HYLE_RETURN_IF_ERROR(
        current.write(source.subspan(consumed, current.size())));
    consumed += current.size();
  }
  return Status::Ok();
}

Status
CompositeBufferView::read(std::span<std::byte> destination) const noexcept {
  return ConstCompositeBufferView(*this).read(destination);
}

Status CompositeBufferView::fill(std::byte value) const noexcept {
  if (!valid())
    return Status::FailedPrecondition("composite buffer view is invalid");
  const std::size_t count = piece_count();
  for (std::size_t index = 0; index < count; ++index) {
    HYLE_ASSIGN_OR_RETURN(BufferView current, piece(index));
    HYLE_RETURN_IF_ERROR(current.fill(value));
  }
  return Status::Ok();
}

Status
CompositeBufferView::copy_from(ConstCompositeBufferView source) const noexcept {
  if (!valid() || !source.valid()) {
    return Status::FailedPrecondition("composite buffer view is invalid");
  }
  if (source.size() != size_) {
    return Status::InvalidArgument(
        "copy ranges must have equal composite buffer view size");
  }
  if (impl_ == source.impl_) {
    if (offset_ == source.offset_)
      return Status::Ok();
    const std::size_t destination_end = offset_ + size_;
    const std::size_t source_end = source.offset_ + source.size_;
    if (offset_ < source_end && source.offset_ < destination_end) {
      return Status::InvalidArgument(
          "overlapping composite copy requires explicit caller staging");
    }
  }

  std::size_t destination_piece = 0;
  std::size_t source_piece = 0;
  std::size_t destination_offset = 0;
  std::size_t source_offset = 0;
  std::size_t copied = 0;
  while (copied < size_) {
    HYLE_ASSIGN_OR_RETURN(BufferView destination, piece(destination_piece));
    HYLE_ASSIGN_OR_RETURN(ConstBufferView current_source,
                          source.piece(source_piece));
    const std::size_t destination_available =
        destination.size() - destination_offset;
    const std::size_t source_available = current_source.size() - source_offset;
    const std::size_t chunk =
        (std::min)(destination_available, source_available);
    HYLE_ASSIGN_OR_RETURN(BufferView destination_chunk,
                          destination.subview(destination_offset, chunk));
    HYLE_ASSIGN_OR_RETURN(ConstBufferView source_chunk,
                          current_source.subview(source_offset, chunk));
    HYLE_RETURN_IF_ERROR(destination_chunk.copy_from(source_chunk));
    copied += chunk;
    destination_offset += chunk;
    source_offset += chunk;
    if (destination_offset == destination.size()) {
      ++destination_piece;
      destination_offset = 0;
    }
    if (source_offset == current_source.size()) {
      ++source_piece;
      source_offset = 0;
    }
  }
  return Status::Ok();
}

Result<ConstCompositeBufferView>
ConstCompositeBufferView::subview(std::size_t offset,
                                  std::size_t size) const noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("composite buffer view is invalid"));
  }
  if (offset > size_) {
    return std::unexpected(
        Status::OutOfRange("composite subview offset exceeds its range"));
  }
  const std::size_t available = size_ - offset;
  if (size == std::dynamic_extent)
    size = available;
  if (size > available ||
      offset_ > (std::numeric_limits<std::size_t>::max)() - offset) {
    return std::unexpected(
        Status::OutOfRange("composite subview size exceeds its range"));
  }
  return ConstCompositeBufferView(impl_, offset_ + offset, size);
}

std::size_t ConstCompositeBufferView::piece_count() const noexcept {
  return valid()
             ? ::hyle::mem::piece_count(*as_composite(impl_), offset_, size_)
             : 0;
}

Result<ConstBufferView>
ConstCompositeBufferView::piece(std::size_t index) const noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("composite buffer view is invalid"));
  }
  return const_piece(*as_composite(impl_), offset_, size_, index);
}

Status ConstCompositeBufferView::read(
    std::span<std::byte> destination) const noexcept {
  if (!valid())
    return Status::FailedPrecondition("composite buffer view is invalid");
  if (destination.size() != size_) {
    return Status::InvalidArgument(
        "read size must equal composite buffer view size");
  }
  std::size_t consumed = 0;
  const std::size_t count = piece_count();
  for (std::size_t index = 0; index < count; ++index) {
    HYLE_ASSIGN_OR_RETURN(ConstBufferView current, piece(index));
    HYLE_RETURN_IF_ERROR(
        current.read(destination.subspan(consumed, current.size())));
    consumed += current.size();
  }
  return Status::Ok();
}

} // namespace hyle::mem

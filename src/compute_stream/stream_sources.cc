#include "hyle/compute/detail/file_stream_access.h"

#include <memory>
#include <utility>

#include "hyle/util/ref_counted.h"

namespace hyle::compute::detail {
struct StreamFileState final : util::RefCounted<StreamFileState> {
  StreamFileState(mem::AllocatorRef selected, std::uint64_t size) noexcept
      : allocator(selected), capacity(size) {}
  void destroy() noexcept {
    const auto selected = allocator;
    std::destroy_at(this); selected.deallocate(this, sizeof(StreamFileState), alignof(StreamFileState));
  }
  mem::AllocatorRef allocator;
  std::uint64_t capacity;
  io::File owned;
  io::File *file = nullptr;
};
struct StreamMappingState final : util::RefCounted<StreamMappingState> {
  explicit StreamMappingState(mem::AllocatorRef selected) noexcept : allocator(selected) {}
  void destroy() noexcept {
    const auto selected = allocator;
    std::destroy_at(this); selected.deallocate(this, sizeof(StreamMappingState), alignof(StreamMappingState));
  }
  mem::AllocatorRef allocator;
  mem::FileMapping owned;
  const mem::FileMapping *mapping = nullptr;
};
io::File *FileStreamAccess::file(const StreamFile &file) noexcept {
  return file.state_ ? file.state_->file : nullptr;
}
io::File *FileStreamAccess::file(const StreamSource &source) noexcept { return file(source.file_); }
const mem::FileMapping *FileStreamAccess::mapping(const StreamSource &source) noexcept {
  return source.mapping_ ? source.mapping_->mapping : nullptr;
}
} // namespace hyle::compute::detail

namespace hyle::compute {
StreamFile::StreamFile(const StreamFile &other) noexcept : state_(other.state_) { if (state_) state_->incref(); }
StreamFile &StreamFile::operator=(const StreamFile &other) noexcept {
  if (this != &other) { auto *next = other.state_; if (next) next->incref(); reset(); state_ = next; } return *this;
}
StreamFile::StreamFile(StreamFile &&other) noexcept : state_(std::exchange(other.state_, nullptr)) {}
StreamFile &StreamFile::operator=(StreamFile &&other) noexcept {
  if (this != &other) { reset(); state_ = std::exchange(other.state_, nullptr); } return *this;
}
StreamFile::~StreamFile() { reset(); }
void StreamFile::reset() noexcept { if (auto *old = std::exchange(state_, nullptr)) old->decref(); }
std::uint64_t StreamFile::capacity_bytes() const noexcept { return state_ ? state_->capacity : 0; }
Result<StreamFile> StreamFile::borrow(io::File &file, std::uint64_t size, mem::AllocatorRef allocator) noexcept {
  if (!file.valid() || !allocator.valid()) return std::unexpected(Status::InvalidArgument("invalid stream file/allocator"));
  auto *storage = allocator.allocate(sizeof(detail::StreamFileState), alignof(detail::StreamFileState));
  if (!storage) return std::unexpected(Status::ResourceExhausted("stream file metadata"));
  auto *state = std::construct_at(static_cast<detail::StreamFileState *>(storage), allocator, size);
  state->file = &file; return StreamFile(state);
}
Result<StreamFile> StreamFile::adopt(io::File &&file, std::uint64_t size, mem::AllocatorRef allocator) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto result, borrow(file, size, allocator));
  result.state_->owned = std::move(file); result.state_->file = &result.state_->owned; return result;
}
StreamSource::StreamSource(const StreamSource &other) noexcept : file_(other.file_), mapping_(other.mapping_) {
  if (mapping_) mapping_->incref();
}
StreamSource &StreamSource::operator=(const StreamSource &other) noexcept {
  if (this != &other) { StreamSource next(other); *this = std::move(next); } return *this;
}
StreamSource::StreamSource(StreamSource &&other) noexcept
    : file_(std::move(other.file_)), mapping_(std::exchange(other.mapping_, nullptr)) {}
StreamSource &StreamSource::operator=(StreamSource &&other) noexcept {
  if (this != &other) {
    if (mapping_) mapping_->decref();
    file_ = std::move(other.file_); mapping_ = std::exchange(other.mapping_, nullptr);
  } return *this;
}
StreamSource::~StreamSource() { if (mapping_) mapping_->decref(); }
std::uint64_t StreamSource::capacity_bytes() const noexcept {
  return mapping_ ? mapping_->mapping->size() : file_.capacity_bytes();
}
Result<StreamSource> StreamSource::borrow(const mem::FileMapping &mapping, mem::AllocatorRef allocator) noexcept {
  if (!mapping.valid() || mapping.access() != mem::MappingAccess::kReadOnly || !allocator.valid())
    return std::unexpected(Status::InvalidArgument("stream mapping must be valid and immutable"));
  auto *storage = allocator.allocate(sizeof(detail::StreamMappingState), alignof(detail::StreamMappingState));
  if (!storage) return std::unexpected(Status::ResourceExhausted("stream mapping metadata"));
  auto *state = std::construct_at(static_cast<detail::StreamMappingState *>(storage), allocator);
  state->mapping = &mapping; StreamSource result; result.mapping_ = state; return result;
}
Result<StreamSource> StreamSource::adopt(mem::FileMapping &&mapping, mem::AllocatorRef allocator) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto result, borrow(mapping, allocator));
  result.mapping_->owned = std::move(mapping); result.mapping_->mapping = &result.mapping_->owned; return result;
}
StreamWriteRange StreamWriteRange::ready(ReadLease source, std::uint64_t file_offset,
    std::size_t resource_offset, std::size_t bytes, std::size_t physical_bytes) noexcept {
  StreamWriteRange result; result.source = std::move(source); result.file_offset = file_offset;
  result.resource_offset = resource_offset; result.bytes = bytes; result.physical_bytes = physical_bytes; return result;
}
StreamWriteRange StreamWriteRange::produced(WriteLease &&source, Completion &&producer,
    std::uint64_t file_offset, std::size_t resource_offset, std::size_t bytes, std::size_t physical_bytes) noexcept {
  StreamWriteRange result; result.pending_source = std::move(source); result.producer = std::move(producer);
  result.file_offset = file_offset; result.resource_offset = resource_offset;
  result.bytes = bytes; result.physical_bytes = physical_bytes; return result;
}
} // namespace hyle::compute

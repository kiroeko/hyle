#include "metadata_arena.h"

#include <limits>
#include <utility>

#include "ggml.h"

namespace hyle::compute::ggml_vulkan::detail {

Result<MetadataArena>
MetadataArena::create(std::size_t bytes, mem::AllocatorRef allocator) noexcept {
  if (!allocator.valid() || bytes == 0)
    return std::unexpected(Status::InvalidArgument("ggml metadata arena"));
  // Keep ggml's aligned object offsets and pointer differences representable.
  constexpr auto maximum =
      static_cast<std::size_t>((std::numeric_limits<std::ptrdiff_t>::max)());
  if (bytes > maximum - (GGML_MEM_ALIGN - 1))
    return std::unexpected(Status::ResourceExhausted("ggml metadata size"));
  MetadataArena arena;
  arena.allocator_ = allocator;
  arena.bytes_ = bytes;
  arena.storage_ = allocator.allocate(bytes, GGML_MEM_ALIGN);
  if (arena.storage_ == nullptr)
    return std::unexpected(
        Status::ResourceExhausted("ggml metadata allocation"));
  arena.context_ = ggml_init({bytes, arena.storage_, true});
  if (arena.context_ == nullptr)
    return std::unexpected(
        Status::ResourceExhausted("ggml context allocation"));
  return arena;
}

MetadataArena::MetadataArena(MetadataArena &&other) noexcept
    : allocator_(other.allocator_),
      storage_(std::exchange(other.storage_, nullptr)),
      bytes_(std::exchange(other.bytes_, 0)),
      context_(std::exchange(other.context_, nullptr)) {}

MetadataArena &MetadataArena::operator=(MetadataArena &&other) noexcept {
  if (this != &other) {
    reset();
    allocator_ = other.allocator_;
    storage_ = std::exchange(other.storage_, nullptr);
    bytes_ = std::exchange(other.bytes_, 0);
    context_ = std::exchange(other.context_, nullptr);
  }
  return *this;
}

MetadataArena::~MetadataArena() { reset(); }

void MetadataArena::reset() noexcept {
  // ggml borrows storage_; its context must retire before the backing does.
  if (context_ != nullptr)
    ggml_free(std::exchange(context_, nullptr));
  if (storage_ != nullptr)
    allocator_.deallocate(std::exchange(storage_, nullptr), bytes_,
                          GGML_MEM_ALIGN);
  bytes_ = 0;
}

} // namespace hyle::compute::ggml_vulkan::detail

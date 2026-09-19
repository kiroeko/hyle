#include "metadata_arena.h"

#include <limits>
#include <utility>

#include "ggml.h"

namespace hyle::compute::ggml_runtime {

Result<MetadataArena>
MetadataArena::for_graph(std::size_t tensor_slots, std::size_t graph_capacity,
                         mem::AllocatorRef allocator) noexcept {
  if (tensor_slots == 0 || graph_capacity == 0)
    return std::unexpected(Status::InvalidArgument("ggml metadata counts"));
  // Upstream stores graph counts in int and doubles its hash capacity. Bound
  // that arithmetic before entering its infallible overhead calculation.
  constexpr auto maximum =
      static_cast<std::size_t>((std::numeric_limits<std::ptrdiff_t>::max)());
  if (graph_capacity >
          static_cast<std::size_t>((std::numeric_limits<int>::max)() / 4) ||
      graph_capacity > (maximum - 1024) / 128)
    return std::unexpected(Status::ResourceExhausted("ggml graph capacity"));
  const auto graph_bytes = ggml_graph_overhead_custom(graph_capacity, false);
  const auto tensor_bytes = ggml_tensor_overhead();
  if (tensor_slots > (maximum - graph_bytes) / tensor_bytes)
    return std::unexpected(Status::ResourceExhausted("ggml tensor capacity"));
  return create(graph_bytes + tensor_slots * tensor_bytes, allocator);
}

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

} // namespace hyle::compute::ggml_runtime

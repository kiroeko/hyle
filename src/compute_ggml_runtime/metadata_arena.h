#pragma once

#include <cstddef>

#include "hyle/error/result.h"
#include "hyle/mem/allocator.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_GGML_RUNTIME_INTERNAL_API
#elif defined(HYLE_COMPUTE_GGML_RUNTIME_BUILD)
#define HYLE_GGML_RUNTIME_INTERNAL_API HYLE_EXPORT
#else
#define HYLE_GGML_RUNTIME_INTERNAL_API HYLE_IMPORT
#endif

struct ggml_context;

namespace hyle::compute::ggml_runtime {

// Private cold metadata owner. Payload lives in native backend buffers, never
// in this arena. The allocator must outlive the arena; separate arenas may be
// used concurrently, but construction of one graph is thread-compatible.
class MetadataArena final {
public:
  MetadataArena() noexcept = default;
  MetadataArena(const MetadataArena &) = delete;
  MetadataArena &operator=(const MetadataArena &) = delete;
  HYLE_GGML_RUNTIME_INTERNAL_API MetadataArena(MetadataArena &&other) noexcept;
  HYLE_GGML_RUNTIME_INTERNAL_API MetadataArena &
  operator=(MetadataArena &&other) noexcept;
  HYLE_GGML_RUNTIME_INTERNAL_API ~MetadataArena();

  HYLE_GGML_RUNTIME_INTERNAL_API static Result<MetadataArena>
  create(std::size_t bytes, mem::AllocatorRef allocator) noexcept;
  // Includes one non-gradient graph and the declared tensor/view metadata.
  // Builders must count all intermediate metadata before constructing a graph.
  HYLE_GGML_RUNTIME_INTERNAL_API static Result<MetadataArena>
  for_graph(std::size_t tensor_slots, std::size_t graph_capacity,
            mem::AllocatorRef allocator) noexcept;
  ggml_context *get() const noexcept { return context_; }
  std::size_t capacity() const noexcept { return bytes_; }

private:
  void reset() noexcept;
  mem::AllocatorRef allocator_;
  void *storage_ = nullptr;
  std::size_t bytes_ = 0;
  ggml_context *context_ = nullptr;
};

} // namespace hyle::compute::ggml_runtime

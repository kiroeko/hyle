#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/compute/detail/backend_extension.h"
#include "hyle/mem/buffer.h"

namespace hyle::compute::detail {
inline constexpr BackendExtensionId kIndexedCopyExtensionId{
    0x48594c45494e4458ULL, 0x434f5059524f5731ULL, 1};

struct IndexedCopyNativeSpec final {
  std::uint32_t slot_count = 0, row_count = 0;
  std::size_t row_bytes = 0, slot_stride = 0;
};
struct IndexedCopyNativeBindings final {
  // Addressable capacity, not whole-pool read or mutable permission. The
  // original caller admits each distinct destination and initialized source.
  // Indices are exactly row_count immutable, initialized I32 coordinates.
  mem::ConstBufferView capacity, rows, indices;
};
// Optional raw row-copy primitive of this exact backend Context. All source
// rows and selected destination ranges must be disjoint; the common caller
// proves permissions, uniqueness and aliasing before this callback. Every I32
// index is checked before any payload write, using the original device status.
// Padding and previous destination contents are never read or written.
//
// prepare/destroy are cold metadata operations. enqueue records into the
// original admitted Queue sequence, whose Completion reports the full status
// and actual retirement. No private Queue, Completion, lease owner or executor
// is created. Captured arguments depend only on spec and all three complete
// carrier identities; changing their contents requires original retirement.
struct IndexedCopyBackendVTable final {
  Status (*prepare)(void *context, IndexedCopyNativeSpec, mem::AllocatorRef,
                    void **program) noexcept = nullptr;
  void (*destroy)(void *program, mem::AllocatorRef) noexcept = nullptr;
  EnqueueResult (*enqueue)(void *context, void *queue, void *program,
                          IndexedCopyNativeBindings) noexcept = nullptr;
  bool supports_command_capture = false;
};
} // namespace hyle::compute::detail

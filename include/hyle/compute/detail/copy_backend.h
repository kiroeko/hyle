#pragma once

#include <span>

#include "hyle/compute/detail/copy_ranges.h"
#include "hyle/compute/detail/queue_sequence.h"
#include "hyle/mem/buffer.h"

namespace hyle::compute::detail {

// Optional, versioned CPU/CUDA byte-copy capability. Apple/Vulkan keep their
// own native programs; this table does not create another execution runtime.
inline constexpr BackendExtensionId kCopyBatchExtensionId{
    0x48594c45434f4d50ULL, 0x434f505942415431ULL, 1};

struct CopyBackendRange final {
  mem::BufferView destination;
  mem::ConstBufferView source;
};

// All state and descriptor arrays are cold. bind_slot captures the complete
// replay before acceptance; enqueue only selects a contiguous phase slice.
// A slot is bound to the original queue used to qualify its native route.
// Program/slot destruction requires their original actual native retirement.
struct CopyBackendVTable final {
  Status (*prepare)(void *context, mem::AllocatorRef allocator,
                    void **program, std::size_t *metadata_bytes) noexcept;
  void (*destroy_program)(void *program, mem::AllocatorRef allocator) noexcept;
  Status (*create_slot)(void *context, void *queue, void *program,
                        std::uint32_t capacity, mem::AllocatorRef allocator,
                        void **slot, std::size_t *metadata_bytes) noexcept;
  void (*destroy_slot)(void *slot, mem::AllocatorRef allocator) noexcept;
  Result<PhysicalCopyResource> (*physical)(void *context,
                                          mem::ConstBufferView view) noexcept;
  Status (*bind_slot)(void *context, void *program, void *slot,
                      std::span<const CopyBackendRange> ranges) noexcept;
  EnqueueResult (*enqueue)(void *context, void *queue, void *program,
                           void *slot, std::size_t first, std::size_t count,
                           std::uint64_t *native_calls) noexcept;
};

} // namespace hyle::compute::detail

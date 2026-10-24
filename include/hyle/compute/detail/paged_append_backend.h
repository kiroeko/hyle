#pragma once

#include <cstddef>
#include <cstdint>
#include "hyle/compute/detail/paged_append_wire.h"
#include "hyle/compute/detail/backend_extension.h"
#include "hyle/mem/buffer.h"

namespace hyle::compute::detail {
inline constexpr BackendExtensionId kPagedAppendExtensionId{
    0x48594c4550414745ULL, 0x415050454e443031ULL, 1};

struct PagedAppendNativeBindings final {
  // Addressable capacity, NOT whole-bank mutable permission. The original
  // Graph admits precisely the typed instruction destinations and old reads.
  mem::ConstBufferView arena,rows,metadata;
};
// Optional byte-copy primitive of the exact backend Context. This records a
// single fixed-topology kernel into the original admitted Queue sequence.
// prepare/destroy are cold; no private Queue, Completion, page owner or plan.
// Captured arguments depend only on spec and all three complete carrier
// identities. Instruction contents/count may change only after retirement.
struct PagedAppendBackendVTable final {
  Status (*prepare)(void *context,PagedAppendNativeSpec,mem::AllocatorRef,
                    void **program) noexcept=nullptr;
  void (*destroy)(void *program,mem::AllocatorRef) noexcept=nullptr;
  EnqueueResult (*enqueue)(void *context,void *queue,void *program,
                          PagedAppendNativeBindings) noexcept=nullptr;
  bool supports_command_capture=false;
};
} // namespace hyle::compute::detail

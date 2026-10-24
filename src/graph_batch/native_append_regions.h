#pragma once

#include "hyle/graph/batch.h"
#include "hyle/graph/detail/region_access.h"
#include "hyle/graph/region_provider.h"

namespace hyle::graph::batch::detail {
// Private cold capability of the actual selected zero-output append region.
// Native slots own the original transaction, never a second Graph executor.
inline constexpr compute::ProviderExtensionId kNativeAppendRegionExtensionId{
    0x48594c454e415441ULL, 0x5050454e44524547ULL, 1};
struct NativeAppendRegionFacts final {
  NodeId node;
  PagedAppend descriptor;
  compute::TensorDesc rows;
  mem::MemoryResourceRef resource;
};
struct NativeAppendRange final {
  mem::BufferView destination;
  // An inherited prefix has its real ready source here. Otherwise source_offset
  // selects bytes from the one actual Graph rows input at native execution.
  mem::ConstBufferView prefix;
  std::size_t source_offset = 0, bytes = 0;
  bool from_rows = false;
};
struct NativeAppendConfiguration final {
  compute::BatchLayoutVersion layout;
  compute::ConstTensorView capacity;
  std::span<const NativeAppendRange> ranges;
};
struct NativeAppendRegionVTable final {
  Result<NativeAppendRegionFacts> (*inspect)(void *prepared,
                                           void *invocation) noexcept = nullptr;
  PagedAppendPages *(*pages)(void *invocation) noexcept = nullptr;
  // Cold-only: copy all needed descriptors into the original slot. This may
  // allocate/import/record but must not enqueue work or retain these spans.
  Status (*configure)(void *prepared, void *invocation,
                      const NativeAppendConfiguration &) noexcept = nullptr;
  Status (*qualify_reader)(void *prepared, void *invocation,
      compute::detail::CompletionEndpoint) noexcept = nullptr;
  // Check known write failures under the backend's original ordering contract:
  // in a FIFO worker preflight, or before enqueue on the same hardware queue.
  // A pending accepted write may proceed on that queue; publication still
  // requires the whole Graph's successful actual retirement. Never wait/enqueue.
  Status (*before_read)(void *prepared, void *invocation) noexcept = nullptr;
};
struct NativeAppendRegionRef final {
  const NativeAppendRegionVTable *table = nullptr;
  const ProviderRegionPreparedVTable *implementation = nullptr;
  void *prepared = nullptr, *invocation = nullptr;
  std::uint32_t step = 0;
  ExecutionLaneId lane;
  NodeId node;
};
} // namespace hyle::graph::batch::detail

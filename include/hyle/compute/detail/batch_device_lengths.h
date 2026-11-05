#pragma once
#include "hyle/compute/detail/batch_provider.h"

namespace hyle::compute::detail {
// Original prepared capacity and Queue own the immutable bounds, writable
// metadata and schedules. Device lengths cannot grant additional page access.
inline constexpr ProviderExtensionId kBatchDeviceLengthsExtensionId{
    0x48594c4542415443ULL,0x44455643484f4943ULL,2};
struct BatchDeviceLengthsVTable final {
  Status (*configure)(void *native, Queue &) noexcept=nullptr;
  Status (*qualify)(void *native, Queue &) noexcept=nullptr;
  EnqueueResult (*update)(void *native, QueueSequence &, ConstTensorView query_lengths,
      ConstTensorView kv_lengths, mem::ConstBufferView bounds,
      mem::BufferView metadata, BatchWorkspaceViews) noexcept=nullptr;
};
} // namespace hyle::compute::detail

#pragma once
#include "hyle/compute/detail/batch_provider.h"

namespace hyle::compute::detail {
// Separate, versioned opt-in. The existing BatchPlanning v7 ABI is unchanged.
inline constexpr ProviderExtensionId kBatchDeviceChoiceExtensionId{
    0x48594c4542415443ULL,0x44455643484f4943ULL,1};
enum class BatchDeviceChoiceStage { kPreflight, kPrepared };
struct BatchDeviceChoiceVTable final {
  // The native slot already owns the physical-two-row plan. This must prove
  // both finite layouts preserve its actual launch/scratch topology.
  Status (*qualify)(void *native, Queue &, const BatchLayoutVersion &,
                    const BatchLayoutVersion &, BatchDeviceChoiceStage) noexcept=nullptr;
  // Runs in the original admitted sequence before that same slot's math.
  // All buffers belong to its prepared owner. No public mutable metadata.
  EnqueueResult (*select)(void *native, QueueSequence &, ConstTensorView selector,
      mem::ConstBufferView zero, mem::ConstBufferView nonzero,
      mem::BufferView selected, BatchWorkspaceViews) noexcept=nullptr;
};
// Transitional capability while existing consumers migrate from V1. Capacity
// bounds are the prepared layout; each request supplies query/KV lengths.
// The original owner holds both the immutable page envelope and writable
// metadata/schedules. There is no table of accepted-layout combinations.
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

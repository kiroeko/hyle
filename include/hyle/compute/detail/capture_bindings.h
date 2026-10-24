#pragma once

#include "hyle/compute/provider.h"
#include "hyle/error/status.h"

namespace hyle::compute::detail {
struct ProviderPrepared;

// Selected-artifact proof for bounded Graph capture binding variants. The same
// original invocation must remain alive and all previous native uses must have
// actually retired. Ordinary TensorView rebinding and another outer recording
// preserve every private address referenced by prior captures. Before Commands,
// native/configuration mutation, or slot/workspace teardown, Graph must discard
// those recordings. This fact grants no permission, readiness or Queue identity.
inline constexpr ProviderExtensionId kCaptureBindingsExtensionId{
    0x48594c4543415054ULL, 0x42494e44494e4753ULL, 1};
struct CaptureBindingsVTable final {
  // Inspect the exact prepared implementation before interpreting its state.
  // Unknown implementations return Unimplemented, never a provider-wide proof.
  Status (*inspect)(const ProviderPrepared &,
                     bool *preserves_private_arguments) noexcept = nullptr;
};
} // namespace hyle::compute::detail

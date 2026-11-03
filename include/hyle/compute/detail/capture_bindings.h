#pragma once

#include "hyle/compute/provider.h"
#include "hyle/error/result.h"
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

// Optional, object-local cold configuration qualification. This is a separate
// extension so existing CaptureBindingsVTable/provider ABIs remain intact.
// True proves that applying this exact configuration cannot replace any
// address, algorithm or scalar encoded in an existing native recording.
// Readiness and permissions must still be revoked and qualified again.
inline constexpr ProviderExtensionId kCaptureConfigurationExtensionId{
    0x48594c4543415054ULL, 0x434f4e4649475552ULL, 1};
struct CaptureConfigurationVTable final {
  Result<bool> (*preserves)(const ProviderPrepared &, const void *invocation,
                           ProviderExtensionId configuration_kind,
                           const void *configuration) noexcept = nullptr;
};
} // namespace hyle::compute::detail

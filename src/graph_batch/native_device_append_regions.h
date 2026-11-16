#pragma once

#include "native_append_regions.h"
#include "hyle/compute/detail/device_lengths_paged_append_wire.h"

#include <array>

namespace hyle::graph::batch::detail {
// Additive cold interface of the selected device-controlled append. Ordinary
// native append facts and their existing ABI remain unchanged.
inline constexpr compute::ProviderExtensionId kNativeDeviceAppendRegionExtensionId{
    0x48594c454e415444ULL, 0x4556494345415050ULL, 1};
struct NativeDeviceAppendPage final {
  compute::detail::DeviceAppendPage page;
  mem::BufferView destination;
  mem::ConstBufferView prefix;
};
struct NativeDeviceAppendConfiguration final {
  compute::BatchLayoutVersion layout;
  compute::ConstTensorView capacity;
  std::span<const compute::detail::DeviceAppendRequest> requests;
  std::span<const NativeDeviceAppendPage> pages;
};
struct NativeDeviceAppendRegionVTable final {
  std::array<ValueId,2> (*length_inputs)(void *prepared) noexcept = nullptr;
  Status (*configure)(void *prepared,void *invocation,
      const NativeDeviceAppendConfiguration &) noexcept = nullptr;
  // Borrowed original device verdict, kept alive by the original Graph slot.
  // Readers import the original append dependency before accessing this word.
  Result<compute::ConstTensorView> (*status)(void *prepared,void *invocation) noexcept = nullptr;
};
inline const NativeDeviceAppendRegionVTable *native_device_append_table(
    const NativeAppendRegionRef &region) noexcept {
  return region.implementation && region.implementation->query_extension
      ? static_cast<const NativeDeviceAppendRegionVTable *>(region.implementation->query_extension(
          region.prepared,kNativeDeviceAppendRegionExtensionId)) : nullptr;
}
} // namespace hyle::graph::batch::detail

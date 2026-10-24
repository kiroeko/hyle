#pragma once

#include "hyle/graph/region_provider.h"
#include "hyle/compute/apple_memory.h"
#include "../compute_route_apple/apple_carrier.h"

namespace hyle::graph::apple::detail {
// Actual selected Apple implementations expose their original native Job, not
// a Region slot layout. The reference never authorizes a different Runtime.
inline constexpr compute::ProviderExtensionId kNativeOutputExtensionId{
    0x48594c454150504cULL, 0x4f55545055545352ULL, 1};
struct BorrowedRange final {
  mem::ConstBufferView storage;
};
struct NativeOutputOrigins final {
  std::span<const BorrowedRange> borrowed;
  std::span<const BorrowedRange> constants;
  std::span<const compute::ReadLease> owned;
  std::span<const RegionOutputReference> sources;
  // A mathematically fresh source supplies this only for an actual overlapping
  // ready output. Views instead expose their original input ranges/sources.
  compute::ReadLease owner;
  bool owned_overlap = false;
};
struct NativeOutputVTable final {
  Result<compute::apple::detail::InputSource> (*source)(void *, void *, std::size_t,
      const std::shared_ptr<compute::apple::detail::Runtime> &,
      const compute::TensorDesc &) noexcept = nullptr;
  Result<NativeOutputOrigins> (*origins)(void *, void *, std::size_t,
      const compute::apple::NativeStorage &) noexcept = nullptr;
};
} // namespace hyle::graph::apple::detail

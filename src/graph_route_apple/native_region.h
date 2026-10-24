#pragma once

#include "hyle/graph/apple.h"
#include "hyle/graph/region_provider.h"
#include "../compute_route_apple/apple_carrier.h"

namespace hyle::graph::apple::detail {
inline constexpr std::uint64_t kNativeCtPackedRoute = 4;
inline constexpr std::uint64_t kNativeStaticFloat8Route = 5;
// Source-owned immutable cold lowering configuration and its actual Runtime.
// Prepared native programs retain their compiled callable, not this object.
struct SourceContext final {
  std::shared_ptr<compute::apple::detail::Runtime> runtime;
  CustomLowering custom;
};
const std::shared_ptr<compute::apple::detail::Runtime> &source_runtime(void *) noexcept;
const RegionProviderVTable *native_region_provider() noexcept;
compute::detail::CompletionEndpoint region_ordered_endpoint(
    const std::shared_ptr<compute::apple::detail::Runtime> &) noexcept;
bool imports_region_ordered_endpoint(
    const std::shared_ptr<compute::apple::detail::Runtime> &,
    compute::detail::CompletionEndpoint) noexcept;
Result<compute::apple::detail::InputSource> native_output_source(
    const RegionOutputReference &, const std::shared_ptr<compute::apple::detail::Runtime> &,
    const compute::TensorDesc &) noexcept;
}

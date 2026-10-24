#pragma once

#include "hyle/compute/cpu.h"

namespace hyle::compute::cpu::internal {

enum class DefaultProvider : std::uint8_t { kOneDnn = 1, kGgml, kAten };
enum class DefaultScope : std::uint8_t { kDirect, kGraph };

// Private ordinary-facade factory seam. Original capability/resource admission
// and prepared execution remain in the mature provider. No timing takes place.
HYLE_COMPUTE_ROUTE_CPU_API Provider
with_defaults(Provider, DefaultProvider, DefaultScope,
              bool single_native_thread) noexcept;

// One original CPU provider assembly. Graph supplies the ggml provider with
// its additional region capability and an optional independent region provider;
// ordinary preparation and region discovery retain the same Context instances.
HYLE_COMPUTE_ROUTE_CPU_API Result<Context>
create_context(DefaultContextOptions, DefaultScope, Provider ggml,
               ProviderBinding region = {}, Provider native = {}) noexcept;

} // namespace hyle::compute::cpu::internal

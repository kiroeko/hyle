#pragma once

#include "hyle/compute/cpu.h"

namespace hyle::compute::cpu::internal {

enum class DefaultProvider : std::uint8_t { kOneDnn = 1, kGgml, kAten };
enum class DefaultScope : std::uint8_t { kDirect, kGraph };

// Private ordinary-facade factory seam. Original capability/resource admission
// and prepared execution remain in the mature provider. No timing takes place.
HYLE_COMPUTE_ROUTE_CPU_API Provider
with_defaults(Provider, DefaultProvider, DefaultScope) noexcept;

} // namespace hyle::compute::cpu::internal

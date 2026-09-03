#pragma once

#include <cstddef>
#include <span>

#include "hyle/error/status.h"
#include "hyle/platform/visibility.h"

namespace hyle::platform {

// Fill the complete caller-owned span from the operating system's
// cryptographically secure random source. Empty spans succeed. The function
// has no TLS PRNG state, heap allocation or silent statistical fallback.
HYLE_PLATFORM_API Status secure_random(
    std::span<std::byte> output) noexcept;

}  // namespace hyle::platform

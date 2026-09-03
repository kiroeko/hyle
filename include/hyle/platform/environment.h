#pragma once

#include <cstddef>

#include "hyle/error/result.h"
#include "hyle/platform/visibility.h"

namespace hyle::platform {

// Process environment mechanism. The buffer-fill result is the complete UTF-8
// byte length excluding NUL; cap>0 is always NUL-terminated on success. Unset
// is NotFound and remains distinct from a present empty value.
HYLE_PLATFORM_API Result<std::size_t> get_env_var(
    const char* name, char* dst, std::size_t cap) noexcept;

// Environment mutation is process-global and is not safe against concurrent
// C-library getenv/setenv access. Configure it during startup.
HYLE_PLATFORM_API Status set_env_var(
    const char* name, const char* value) noexcept;
HYLE_PLATFORM_API Status unset_env_var(const char* name) noexcept;

}  // namespace hyle::platform

#pragma once

#include <cstdint>

#include "hyle/core/version_defs.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_CORE_API
#elif defined(HYLE_CORE_BUILD)
#  define HYLE_CORE_API HYLE_EXPORT
#else
#  define HYLE_CORE_API HYLE_IMPORT
#endif

namespace hyle {

inline constexpr std::uint32_t kVersionMajor = HYLE_VERSION_MAJOR;
inline constexpr std::uint32_t kVersionMinor = HYLE_VERSION_MINOR;
inline constexpr std::uint32_t kVersionPatch = HYLE_VERSION_PATCH;
inline constexpr std::uint32_t kVersion = HYLE_VERSION_PACKED;
inline constexpr const char kVersionString[] = HYLE_VERSION_STRING;

// Runtime version of the linked hyle_core. Compare with kVersion when loading
// HYLE dynamically or crossing a plugin/FFI boundary.
HYLE_CORE_API std::uint32_t version() noexcept;
HYLE_CORE_API const char* version_string() noexcept;

}  // namespace hyle

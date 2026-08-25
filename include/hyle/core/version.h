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
inline constexpr std::uint32_t kFfiAbiVersionMajor =
    HYLE_FFI_ABI_VERSION_MAJOR;
inline constexpr std::uint32_t kFfiAbiVersionMinor =
    HYLE_FFI_ABI_VERSION_MINOR;
inline constexpr std::uint32_t kFfiAbiVersion = HYLE_FFI_ABI_VERSION;

// Product identity of the linked hyle_core. Compare with kVersion only to
// diagnose a header/runtime package mismatch; product equality is not an FFI
// compatibility rule. A foreign binding checks the artifact ABI bootstrap
// before using any product-identity query.
HYLE_CORE_API std::uint32_t version() noexcept;
// Human-readable runtime identity. The returned storage is borrowed from that
// hyle_core module; never free it and copy it before unloading.
HYLE_CORE_API const char* version_string() noexcept;

}  // namespace hyle

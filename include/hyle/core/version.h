#pragma once

#include <cstdint>
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_CORE_API
#elif defined(HYLE_CORE_BUILD)
#  define HYLE_CORE_API HYLE_EXPORT
#else
#  define HYLE_CORE_API HYLE_IMPORT
#endif

namespace hyle {

// Packed version: (major << 16) | (minor << 8) | patch.
HYLE_CORE_API std::uint32_t version() noexcept;

// Scaffolding self-check: returns true if core is linked and callable.
// Exists to validate the export macro across the DLL boundary in tests.
HYLE_CORE_API bool scaffolding_ok() noexcept;

}  // namespace hyle

#include "hyle/core/version.h"

namespace hyle {

static_assert(kVersionMajor == HYLE_CMAKE_PROJECT_VERSION_MAJOR);
static_assert(kVersionMinor == HYLE_CMAKE_PROJECT_VERSION_MINOR);
static_assert(kVersionPatch == HYLE_CMAKE_PROJECT_VERSION_PATCH);
static_assert(kVersionMajor <= 0xFFu && kVersionMinor <= 0xFFu && kVersionPatch <= 0xFFu);

std::uint32_t version() noexcept { return kVersion; }

const char* version_string() noexcept { return kVersionString; }

}  // namespace hyle

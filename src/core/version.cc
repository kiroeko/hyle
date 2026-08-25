#include "hyle/core/version.h"

namespace {

// Keep the linked runtime identity local to this translation unit. Referring
// to the public inline kVersionString here would let ELF symbol preemption bind
// the library implementation to a consumer/other DSO's header constant, which
// defeats header-vs-runtime mismatch diagnosis.
constexpr char kLinkedVersionString[] = HYLE_VERSION_STRING;

}  // namespace

namespace hyle {

static_assert(kVersionMajor == HYLE_CMAKE_PROJECT_VERSION_MAJOR);
static_assert(kVersionMinor == HYLE_CMAKE_PROJECT_VERSION_MINOR);
static_assert(kVersionPatch == HYLE_CMAKE_PROJECT_VERSION_PATCH);
static_assert(kVersionMajor <= 0xFFu && kVersionMinor <= 0xFFu && kVersionPatch <= 0xFFu);

std::uint32_t version() noexcept { return HYLE_VERSION_PACKED; }

const char* version_string() noexcept { return kLinkedVersionString; }

}  // namespace hyle

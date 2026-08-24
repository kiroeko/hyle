#pragma once

#include "hyle/platform/fs.h"

#if defined(HYLE_TESTING) && !defined(_WIN32)
namespace hyle::platform::detail {

// TEST-ONLY: replacement for POSIX readlink. It makes the n == capacity
// truncation boundary injectable even on filesystems with a small real
// symlink-target limit.
using ReadSymlinkFunctionForTesting = std::ptrdiff_t (*)(
    const char* path, char* buffer, std::size_t capacity,
    int* error_number) noexcept;
HYLE_PLATFORM_API void set_read_symlink_function_for_testing(
    ReadSymlinkFunctionForTesting function) noexcept;

}  // namespace hyle::platform::detail
#endif

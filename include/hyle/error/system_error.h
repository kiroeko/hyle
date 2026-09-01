#pragma once

#include <string_view>
#include <system_error>

#include "hyle/error/status.h"

namespace hyle {

// Classify standard generic/system errors without retaining an
// std::error_category pointer. Generic maps through errno; system maps through
// Win32 on Windows and errno on POSIX. A zero error_code is OK, and an unknown
// custom category fails closed to Unknown.
HYLE_ERROR_API StatusCode
status_code_from_error_code(const std::error_code& error) noexcept;

// Convert a standard-library or third-party error at the owning adapter
// boundary. Generic/system numeric detail is retained under a stable HYLE
// native domain. A custom category contributes human text only; its pointer is
// never stored because the defining DSO may unload.
HYLE_ERROR_API Status status_from_error_code(
    const std::error_code& error,
    std::string_view context = {}) noexcept;

// Operation-specific/custom-category overload. `canonical_code` is supplied
// by the category owner, while the adapter still captures a no-throw human
// diagnostic. A nonzero error paired with StatusCode::Ok fails closed to
// Unknown. A zero error remains OK.
HYLE_ERROR_API Status status_from_error_code_as(
    StatusCode canonical_code, const std::error_code& error,
    std::string_view context = {}) noexcept;

}  // namespace hyle

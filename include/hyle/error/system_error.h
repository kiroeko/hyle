#pragma once

#include <cstdint>
#include <string_view>
#include <system_error>

#include "hyle/error/status.h"

namespace hyle {

namespace detail {

// std::error_category singletons are a C++ runtime detail and may have a
// different address in a consumer executable and the HYLE DLL. Classify at the
// call site, before crossing that seam; the compiled helper receives only this
// closed route plus the still-live error object. Custom categories remain
// custom even if they choose a misleading name.
enum class StandardErrorCategoryRoute : std::uint8_t {
  Generic,
  System,
  Custom,
};

inline StandardErrorCategoryRoute standard_error_category_route(
    const std::error_code& error) noexcept {
  if (&error.category() == &std::generic_category()) {
    return StandardErrorCategoryRoute::Generic;
  }
  if (&error.category() == &std::system_category()) {
    return StandardErrorCategoryRoute::System;
  }
  return StandardErrorCategoryRoute::Custom;
}

HYLE_ERROR_API StatusCode status_code_from_error_code_parts(
    int value, StandardErrorCategoryRoute route) noexcept;

HYLE_ERROR_API Status status_from_error_code_parts(
    StatusCode canonical_code, const std::error_code& error,
    std::string_view context, StandardErrorCategoryRoute route) noexcept;

}  // namespace detail

// Classify standard generic/system errors without retaining an
// std::error_category pointer. Generic maps through errno; system maps through
// Win32 on Windows and errno on POSIX. A zero error_code is OK, and an unknown
// custom category fails closed to Unknown.
inline StatusCode status_code_from_error_code(
    const std::error_code& error) noexcept {
  if (!error) return StatusCode::Ok;
  const auto route = detail::standard_error_category_route(error);
  return detail::status_code_from_error_code_parts(error.value(), route);
}

// Convert a standard-library or third-party error at the owning adapter
// boundary. Generic/system numeric detail is retained under a stable HYLE
// native domain. A custom category contributes human text only; its pointer is
// never stored because the defining DSO may unload.
inline Status status_from_error_code(
    const std::error_code& error,
    std::string_view context = {}) noexcept {
  if (!error) return Status{};
  const auto route = detail::standard_error_category_route(error);
  return detail::status_from_error_code_parts(
      detail::status_code_from_error_code_parts(error.value(), route), error,
      context, route);
}

// Operation-specific/custom-category overload. `canonical_code` is supplied
// by the category owner, while the adapter still captures a no-throw human
// diagnostic. A nonzero error paired with StatusCode::Ok fails closed to
// Unknown. A zero error remains OK.
inline Status status_from_error_code_as(
    StatusCode canonical_code, const std::error_code& error,
    std::string_view context = {}) noexcept {
  if (!error) return Status{};
  return detail::status_from_error_code_parts(
      canonical_code, error, context,
      detail::standard_error_category_route(error));
}

}  // namespace hyle

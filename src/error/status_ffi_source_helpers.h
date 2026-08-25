#pragma once

// Temporary source-only conveniences for HYLE's still-unreviewed module FFI
// bridges. They keep those C++ translation units buildable while the public
// error artifact exposes only the task-driven managed-runtime query/release
// surface. Nothing in this file is installed or exported as a C symbol.

#include <utility>

#include "hyle/c/error.h"
#include "hyle/error/status.h"
#include "status_access.h"

inline hyle_status hyle_status_ok() noexcept { return nullptr; }

inline hyle_status hyle_status_invalid_argument(const char *message) noexcept {
  return hyle::detail::status_to_ffi(
      hyle::Status::InvalidArgument(message == nullptr ? "" : message));
}

inline hyle_status hyle_status_out_of_range(const char *message) noexcept {
  return hyle::detail::status_to_ffi(
      hyle::Status::OutOfRange(message == nullptr ? "" : message));
}

inline hyle_status hyle_status_resource_exhausted(
    const char *message) noexcept {
  return hyle::detail::status_to_ffi(
      hyle::Status::ResourceExhausted(message == nullptr ? "" : message));
}

inline hyle_status hyle_status_unimplemented(const char *message) noexcept {
  return hyle::detail::status_to_ffi(
      hyle::Status::Unimplemented(message == nullptr ? "" : message));
}

inline bool hyle_status_is_ok(hyle_status status) noexcept {
  return status == nullptr;
}

inline void hyle_status_free(hyle_status *status) noexcept {
  if (status == nullptr)
    return;
  hyle_status_destroy(*status);
  *status = nullptr;
}

#pragma once

#include <cstdint>

#include "hyle/c/error.h"
#include "hyle/error/status.h"

namespace hyle::detail {

// Source-internal ownership bridge for C ABI shims. This header lives under
// src/ so raw tagged words never become part of the installed C++ API.
class StatusAccess {
 public:
  static std::uintptr_t release(Status& status) noexcept {
    const std::uintptr_t rep = status.rep_;
    status.rep_ = Status::kMovedFromRep;
    return rep;
  }

  static Status adopt(std::uintptr_t rep) noexcept {
    Status status;
    status.rep_ = rep;
    return status;
  }
};

// Shared-library bridge used only by module FFI shims. It converts the C++
// tagged word into the opaque direct-FFI handle, including allocation-free
// process-lifetime sentinels for code-only errors.
HYLE_ERROR_API hyle_status status_to_ffi(Status&& status) noexcept;

}  // namespace hyle::detail

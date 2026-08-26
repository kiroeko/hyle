#pragma once

#include <cstdint>

#include "hyle/error/status.h"

namespace hyle::detail {

// Source-internal ownership bridge for Status algorithms. This header lives
// under src/ so raw tagged words never become part of the installed API.
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

}  // namespace hyle::detail

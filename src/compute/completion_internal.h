#pragma once

#include "hyle/compute/detail/completion_access.h"

namespace hyle::compute::detail {

struct CompletionVTable final {
  // If `lifetime` is valid, destroy must arrange for terminal() followed by
  // release() after the underlying work actually terminates.  It must not
  // release borrowed execution state merely because the public handle was
  // abandoned.  A terminal state may run both callbacks inline.
  void (*destroy)(void *state, CompletionLifetime lifetime) noexcept = nullptr;
  Result<CompletionPoll> (*poll)(void *state) noexcept = nullptr;
  Status (*wait)(void *state) noexcept = nullptr;
  Status (*request_cancel)(void *state) noexcept = nullptr;
};

} // namespace hyle::compute::detail

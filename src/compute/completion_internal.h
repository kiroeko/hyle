#pragma once

#include "hyle/compute/completion.h"

namespace hyle::compute::detail {

struct CompletionVTable final {
  void (*destroy)(void *state) noexcept = nullptr;
  Result<CompletionPoll> (*poll)(void *state) noexcept = nullptr;
  Status (*wait)(void *state) noexcept = nullptr;
  Status (*request_cancel)(void *state) noexcept = nullptr;
};

struct CompletionAccess final {
  static HYLE_COMPUTE_API Completion make(const CompletionVTable *vtable,
                                          void *state) noexcept;
};

} // namespace hyle::compute::detail

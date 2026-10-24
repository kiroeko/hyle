#pragma once

#include "hyle/error/status.h"
#include "hyle/compute/detail/operation_access.h"

namespace hyle::compute::detail {
struct ProviderPrepared;
// Object-local capability of the selected prepared implementation. Inspect
// checks its actual callback/recipe identity before interpreting native state.
// Configuration runs only on the original idle slot under Context admission.
struct OperationAccessVTable final {
  Status (*inspect)(const ProviderPrepared &, OperationPreparedAccess *) noexcept = nullptr;
  Status (*configure)(const ProviderPrepared &prepared, void *invocation,
                      const void *configuration) noexcept = nullptr;
};
} // namespace hyle::compute::detail

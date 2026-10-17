#pragma once

#include "hyle/compute/apple.h"
#include "hyle/compute/detail/completion_access.h"

#if defined(HYLE_COMPUTE_APPLE)
namespace hyle::compute::apple::detail {

// One readiness hint, independent of Completion's actual-retirement attachment.
// Success consumes one caller-owned reference: ready then release run exactly
// once, possibly inline. Failure leaves that reference with the caller. The
// callbacks must not wait, reenter the native owner, or observe the Completion.
// They may only publish caller-owned progress and return its retained reference.
struct CompletionNotification final {
  void *state = nullptr;
  void (*ready)(void *) noexcept = nullptr;
  void (*release)(void *) noexcept = nullptr;

  bool valid() const noexcept {
    return state != nullptr && ready != nullptr && release != nullptr;
  }
};

inline void deliver_notification(CompletionNotification notification) noexcept {
  if (!notification.valid()) return;
  notification.ready(notification.state);
  notification.release(notification.state);
  // release may destroy its owner. Never access callback storage afterwards.
}

// The borrowed dependency is used only during registration. Control of its
// Completion remains externally serialized. Only original native owners
// and inline terminal hints are recognized; arbitrary proxy vtables are errors.
// A pending owner has one subscription slot. Cancel/abandon does not retract it;
// the original FIFO owner delivers it after its normal completion tail. A hint
// never authorizes releasing an accepted prefix or bypassing checked poll/wait.
HYLE_COMPUTE_ROUTE_APPLE_API Status subscribe_completion(
    compute::detail::CompletionDependency source,
    CompletionNotification notification) noexcept;

} // namespace hyle::compute::apple::detail
#endif

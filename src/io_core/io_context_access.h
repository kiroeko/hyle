#pragma once

#include "hyle/io/io_context.h"

namespace hyle::io::detail {

// Source-only access for composed IO operations. The public IoContext surface
// deliberately does not expose Scheduler machinery merely so the implementation
// can fan out pinned connection attempts.
struct IoContextAccess {
  static sched::Scheduler& scheduler(IoContext& context) noexcept {
    return context.sched_;
  }
};

}  // namespace hyle::io::detail

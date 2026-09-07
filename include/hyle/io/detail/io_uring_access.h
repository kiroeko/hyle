#pragma once

#include "hyle/io/capabilities.h"
#include "hyle/io/detail/io_context_access.h"

#if HYLE_OS_LINUX

namespace hyle::io::detail {

struct IoUringAccess {
  static Proactor* proactor(const IoUringExtension& extension) noexcept {
    if (!extension.valid()) return nullptr;
    return &IoContextAccess::proactor(*extension.context_, extension.worker_);
  }

  static bool belongs_to(const IoUringExtension& extension,
                         const IoContext& context,
                         unsigned worker) noexcept {
    return extension.context_ == &context && extension.worker_ == worker;
  }
};

}  // namespace hyle::io::detail

#endif

#include "hyle/io/capabilities.h"

#if HYLE_OS_LINUX

#include <climits>

#include "hyle/io/detail/io_uring_access.h"

namespace hyle::io {

Status IoUringExtension::configure_receive_ring(
    IoUringBufferRingOptions options) const noexcept {
  if (!valid()) {
    return Status::FailedPrecondition("invalid io_uring extension");
  }
  if (options.group_id < 0 || options.group_id > UINT16_MAX ||
      options.entries == 0 || options.entries > 32768 ||
      options.buffer_size == 0) {
    return Status::InvalidArgument("invalid io_uring receive-ring options");
  }
  auto* proactor = detail::IoUringAccess::proactor(*this);
  if (proactor == nullptr ||
      !proactor->register_buf_ring(options.group_id, options.entries,
                                   options.buffer_size)) {
    return Status::Unavailable("io_uring receive-ring registration failed");
  }
  return Status{};
}

}  // namespace hyle::io

#endif

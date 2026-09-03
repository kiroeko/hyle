#include "hyle/platform/random.h"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "hyle/core/environment.h"

#if HYLE_OS_WINDOWS
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#  include <bcrypt.h>
#elif HYLE_OS_LINUX
#  include <sys/random.h>
#elif HYLE_OS_MACOS
#  include <sys/random.h>
#else
#  error "secure_random is not implemented for this target OS"
#endif

namespace hyle::platform {

Status secure_random(std::span<std::byte> output) noexcept {
#if HYLE_OS_WINDOWS
  while (!output.empty()) {
    const std::size_t chunk =
        (std::min)(output.size(),
                   static_cast<std::size_t>((std::numeric_limits<ULONG>::max)()));
    const NTSTATUS code = ::BCryptGenRandom(
        nullptr, reinterpret_cast<PUCHAR>(output.data()),
        static_cast<ULONG>(chunk), BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    if (code < 0) return Status::Unavailable("BCryptGenRandom failed");
    output = output.subspan(chunk);
  }
  return Status::Ok();
#elif HYLE_OS_LINUX
  while (!output.empty()) {
    const std::size_t chunk =
        (std::min)(output.size(),
                   static_cast<std::size_t>(
                       (std::numeric_limits<ssize_t>::max)()));
    const ssize_t count = ::getrandom(output.data(), chunk, 0);
    if (count > 0) {
      output = output.subspan(static_cast<std::size_t>(count));
      continue;
    }
    if (count < 0 && errno == EINTR) continue;
    return status_from_errno(count < 0 ? errno : EIO, "getrandom");
  }
  return Status::Ok();
#else
  // getentropy has a fixed 256-byte request ceiling but no descriptor,
  // process-global state or allocation. It is available on supported macOS.
  while (!output.empty()) {
    const std::size_t chunk = (std::min)(output.size(), std::size_t{256});
    if (::getentropy(output.data(), chunk) != 0) {
      return status_from_errno(errno, "getentropy");
    }
    output = output.subspan(chunk);
  }
  return Status::Ok();
#endif
}

}  // namespace hyle::platform

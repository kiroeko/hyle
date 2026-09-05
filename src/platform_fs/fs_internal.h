#pragma once

#include <cerrno>
#include <utility>

#include "hyle/platform/fs.h"

namespace hyle::platform::detail {

// Retry only a call whose owner has separately established that repeating it
// after EINTR cannot duplicate an externally visible effect. Keeping the
// failure predicate explicit prevents this helper from being casually applied
// to close/rename/unlink and other calls with ambiguous completion semantics.
template <class Call, class Failed>
auto retry_eintr(Call&& call, Failed&& failed) noexcept(noexcept(call()))
    -> decltype(call()) {
  for (;;) {
    auto result = call();
    if (!failed(result) || errno != EINTR) return result;
  }
}

template <class Call>
auto retry_eintr_negative(Call&& call) noexcept(noexcept(call()))
    -> decltype(call()) {
  return retry_eintr(std::forward<Call>(call),
                     [](const auto& result) noexcept { return result < 0; });
}

template <class Call>
auto retry_eintr_null(Call&& call) noexcept(noexcept(call()))
    -> decltype(call()) {
  return retry_eintr(
      [&call]() noexcept(noexcept(call())) {
        errno = 0;
        return call();
      },
      [](const auto* result) noexcept { return result == nullptr; });
}

// posix_fallocate reports an error number directly instead of setting errno.
template <class Call>
int retry_eintr_error_code(Call&& call) noexcept(noexcept(call())) {
  int result = 0;
  do { result = call(); } while (result == EINTR);
  return result;
}

}  // namespace hyle::platform::detail

#if defined(HYLE_TESTING) && !defined(_WIN32)
namespace hyle::platform::detail {

// TEST-ONLY: replacement for POSIX readlink. It makes the n == capacity
// truncation boundary injectable even on filesystems with a small real
// symlink-target limit.
using ReadSymlinkFunctionForTesting =
    std::ptrdiff_t (*)(const char* path, char* buffer, std::size_t capacity,
                       int* error_number) noexcept;
HYLE_PLATFORM_FS_API void set_read_symlink_function_for_testing(
    ReadSymlinkFunctionForTesting function) noexcept;

}  // namespace hyle::platform::detail
#endif

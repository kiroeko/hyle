#pragma once

#include <cstring>
#include <format>
#include <string>
#include <string_view>
#include <utility>

#include "hyle/error/status.h"

namespace hyle {

// C++ convenience formatting lives outside status.h so the hot carrier header
// does not pull in <string>. The result is created in the caller, not
// transferred across HYLE's shared-library ABI.
inline std::string to_string(const Status &status) {
  // Driver/OS diagnostics are normally short. Format those through one DLL
  // call into a stack buffer, then construct the returned string once. Only a
  // genuinely long message needs the size-query + second formatting call.
  char local[256];
  const std::size_t size = status.ToString(local, sizeof(local));
  if (size < sizeof(local)) return std::string(local, size);

  std::string text;
  text.resize_and_overwrite(size + 1, [&](char *data, std::size_t capacity) {
    status.ToString(data, capacity);
    return size;
  });
  return text;
}

// Build one diagnostic Status from runtime values without introducing a
// logging/source-location policy or allowing formatter exceptions to escape a
// HYLE failure boundary. The bounded caller-side scratch means formatting
// itself needs no HYLE allocation; Status then makes at most one exact-size
// backing allocation. Oversized output carries an explicit marker.
//
// Invalid format strings are rejected at compile time. A user-defined
// formatter can still throw when exceptions are enabled; that degrades to a
// deterministic message. In exception-disabled builds custom formatters must
// themselves be non-throwing, as there is no language mechanism to catch them.
template <class... Args>
[[nodiscard]] Status format_status(StatusCode code,
                                   std::format_string<Args...> format,
                                   Args&&... args) noexcept {
  if (code == StatusCode::Ok) return Status{};

  constexpr std::size_t kBufferSize = 1024;
  constexpr std::size_t kPayloadCapacity = kBufferSize - 1;
  constexpr std::string_view kTruncated = "...[truncated]";
  static_assert(kTruncated.size() < kPayloadCapacity);

  char buffer[kBufferSize];
  std::string_view message;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    const auto result = std::format_to_n(
        buffer, kPayloadCapacity, format, std::forward<Args>(args)...);
    std::size_t size = static_cast<std::size_t>(result.out - buffer);
    if (static_cast<std::size_t>(result.size) > kPayloadCapacity) {
      std::memcpy(buffer + kPayloadCapacity - kTruncated.size(),
                  kTruncated.data(), kTruncated.size());
      size = kPayloadCapacity;
    }
    buffer[size] = '\0';
    message = std::string_view(buffer, size);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    message = "[status formatting failed]";
  }
#endif
  return Status(code, message);
}

}  // namespace hyle

#pragma once

#include <cstdint>

namespace hyle::io::detail {

// Convert a finite nanosecond wait to the millisecond granularity expected by
// IOCP/epoll without the overflowing `(ns + 999999) / 1000000` idiom. The
// caller handles its backend's infinite sentinel separately; every finite
// duration rounds upward and clamps to a finite backend maximum rather than
// wrapping into a non-blocking poll.
constexpr std::uint64_t finite_timeout_milliseconds(
    std::uint64_t timeout_ns, std::uint64_t maximum_ms) noexcept {
  constexpr std::uint64_t kNsPerMs = 1'000'000;
  const std::uint64_t rounded =
      timeout_ns / kNsPerMs + (timeout_ns % kNsPerMs != 0 ? 1u : 0u);
  return rounded > maximum_ms ? maximum_ms : rounded;
}

}  // namespace hyle::io::detail

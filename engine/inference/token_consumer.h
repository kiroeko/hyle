#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>

#include "request_window.h"

namespace hyle::inference {

// Called by the serialized request owner, never by a device completion.
// A true write transfers one committed token to the consumer. A false write
// transfers nothing; the owner keeps the token and its original request key.
// Cancellation is an intent: the owner stops further work, discards output
// not accepted by the consumer, and calls finished exactly once after real
// retirement. Its key is an identity only and is no longer an active key.
// wait() is called only when no request can advance without output credit and
// no native work is awaiting observation. It must also wake for cancellation;
// its readiness/cancellation check and wait must not lose a concurrent wake.
// The consumer synchronizes with its own reader and must not reenter the owner.
// An optional bool start(RequestKey) noexcept is called after enqueue and before
// execution. False refuses attachment; it never submits or retires model work.
// An optional bool try_turn_end(RequestKey, size_t) noexcept accepts one
// zero-based published turn boundary in token order. False transfers nothing.
// A turn boundary does not imply that the request or device work retired.
template<class T>
concept TokenConsumer = requires(T &consumer, RequestKey key, std::int32_t token) {
  { consumer.max_pending() } noexcept -> std::same_as<std::size_t>;
  { consumer.try_write(key, token) } noexcept -> std::same_as<bool>;
  { consumer.cancel_requested(key) } noexcept -> std::same_as<bool>;
  { consumer.finished(key, false) } noexcept -> std::same_as<void>;
  { consumer.wait() } noexcept -> std::same_as<void>;
};

} // namespace hyle::inference

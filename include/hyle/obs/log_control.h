#pragma once

#include <atomic>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <limits>

namespace hyle::obs {
namespace detail {

template <std::integral Count>
constexpr std::uint64_t positive_log_count(Count value) noexcept {
  if constexpr (std::signed_integral<Count>) {
    if (value <= 0)
      return 0;
  } else if (value == 0) {
    return 0;
  }
  return static_cast<std::uint64_t>(value);
}

} // namespace detail

// Low-contention, best-effort call counter. Concurrent calls are race-free but
// deliberately use a relaxed load/store pair, so they may occasionally share
// a sequence number. This is the normal logging choice when exact cardinality
// is less important than avoiding a contended atomic RMW.
class LogEveryNState final {
public:
  LogEveryNState() noexcept = default;
  LogEveryNState(const LogEveryNState &) = delete;
  LogEveryNState &operator=(const LogEveryNState &) = delete;
  LogEveryNState(LogEveryNState &&other) noexcept
      : counter_(other.counter_.exchange(0, std::memory_order_relaxed)) {}
  LogEveryNState &operator=(LogEveryNState &&other) noexcept {
    if (this != &other) {
      counter_.store(other.counter_.exchange(0, std::memory_order_relaxed),
                     std::memory_order_relaxed);
    }
    return *this;
  }

  // Logs call 1 and then every n calls. n<=0 rejects without advancing.
  template <std::integral Count>
  [[nodiscard]] bool should_log(Count n) noexcept {
    const std::uint64_t period = detail::positive_log_count(n);
    if (period == 0)
      return false;
    const std::uint64_t value = counter_.load(std::memory_order_relaxed);
    counter_.store(value + 1, std::memory_order_relaxed);
    return value % period == 0;
  }

  [[nodiscard]] std::uint64_t count() const noexcept {
    return counter_.load(std::memory_order_relaxed);
  }
  void reset() noexcept { counter_.store(0, std::memory_order_relaxed); }

private:
  std::atomic<std::uint64_t> counter_{0};
};

// Exact global sequence for the cases where every-n cardinality matters.
// Compared with LogEveryNState this pays one relaxed atomic RMW per check.
class LogEveryNExactState final {
public:
  LogEveryNExactState() noexcept = default;
  LogEveryNExactState(const LogEveryNExactState &) = delete;
  LogEveryNExactState &operator=(const LogEveryNExactState &) = delete;
  LogEveryNExactState(LogEveryNExactState &&other) noexcept
      : counter_(other.counter_.exchange(0, std::memory_order_relaxed)) {}
  LogEveryNExactState &operator=(LogEveryNExactState &&other) noexcept {
    if (this != &other) {
      counter_.store(other.counter_.exchange(0, std::memory_order_relaxed),
                     std::memory_order_relaxed);
    }
    return *this;
  }

  template <std::integral Count>
  [[nodiscard]] bool should_log(Count n) noexcept {
    const std::uint64_t period = detail::positive_log_count(n);
    if (period == 0)
      return false;
    const std::uint64_t value =
        counter_.fetch_add(1, std::memory_order_relaxed);
    return value % period == 0;
  }

  [[nodiscard]] std::uint64_t count() const noexcept {
    return counter_.load(std::memory_order_relaxed);
  }
  void reset() noexcept { counter_.store(0, std::memory_order_relaxed); }

private:
  std::atomic<std::uint64_t> counter_{0};
};

// Exact cap: concurrent callers collectively receive at most n admissions.
// Once the cap is reached, checks are a relaxed load rather than an RMW.
class LogFirstNState final {
public:
  LogFirstNState() noexcept = default;
  LogFirstNState(const LogFirstNState &) = delete;
  LogFirstNState &operator=(const LogFirstNState &) = delete;
  LogFirstNState(LogFirstNState &&other) noexcept
      : counter_(other.counter_.exchange(0, std::memory_order_relaxed)) {}
  LogFirstNState &operator=(LogFirstNState &&other) noexcept {
    if (this != &other) {
      counter_.store(other.counter_.exchange(0, std::memory_order_relaxed),
                     std::memory_order_relaxed);
    }
    return *this;
  }

  template <std::integral Count>
  [[nodiscard]] bool should_log(Count n) noexcept {
    const std::uint64_t limit = detail::positive_log_count(n);
    if (limit == 0)
      return false;
    std::uint64_t observed = counter_.load(std::memory_order_relaxed);
    while (observed < limit) {
      if (counter_.compare_exchange_weak(observed, observed + 1,
                                         std::memory_order_relaxed,
                                         std::memory_order_relaxed)) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] std::uint64_t count() const noexcept {
    return counter_.load(std::memory_order_relaxed);
  }
  void reset() noexcept { counter_.store(0, std::memory_order_relaxed); }

private:
  std::atomic<std::uint64_t> counter_{0};
};

// Best-effort logarithmic sampling: calls 1, 2, 4, 8, ... are admitted in a
// single-threaded sequence. Like LogEveryNState, concurrent checks are
// intentionally approximate to avoid a contended RMW.
class LogEveryPow2State final {
public:
  LogEveryPow2State() noexcept = default;
  LogEveryPow2State(const LogEveryPow2State &) = delete;
  LogEveryPow2State &operator=(const LogEveryPow2State &) = delete;
  LogEveryPow2State(LogEveryPow2State &&other) noexcept
      : counter_(other.counter_.exchange(0, std::memory_order_relaxed)) {}
  LogEveryPow2State &operator=(LogEveryPow2State &&other) noexcept {
    if (this != &other) {
      counter_.store(other.counter_.exchange(0, std::memory_order_relaxed),
                     std::memory_order_relaxed);
    }
    return *this;
  }

  [[nodiscard]] bool should_log() noexcept {
    const std::uint64_t old_value = counter_.load(std::memory_order_relaxed);
    const std::uint64_t value = old_value + 1;
    counter_.store(value, std::memory_order_relaxed);
    return value != 0 && (value & (value - 1)) == 0;
  }

  [[nodiscard]] std::uint64_t count() const noexcept {
    return counter_.load(std::memory_order_relaxed);
  }
  void reset() noexcept { counter_.store(0, std::memory_order_relaxed); }

private:
  std::atomic<std::uint64_t> counter_{0};
};

// Exact at-most-once interval admission for a call site, request or inference
// slot. The explicit-time overload lets a loop read one coarse monotonic clock
// for many states. The convenience overload owns one steady_clock read.
class LogIntervalLimiter final {
public:
  LogIntervalLimiter() noexcept = default;
  LogIntervalLimiter(const LogIntervalLimiter &) = delete;
  LogIntervalLimiter &operator=(const LogIntervalLimiter &) = delete;
  LogIntervalLimiter(LogIntervalLimiter &&other) noexcept
      : last_plus_one_(
            other.last_plus_one_.exchange(0, std::memory_order_relaxed)) {}
  LogIntervalLimiter &operator=(LogIntervalLimiter &&other) noexcept {
    if (this != &other) {
      last_plus_one_.store(
          other.last_plus_one_.exchange(0, std::memory_order_relaxed),
          std::memory_order_relaxed);
    }
    return *this;
  }

  // The timestamp and interval must use the same monotonic nanosecond domain.
  // A zero interval disables limiting. If supplied time regresses, checks are
  // suppressed until it catches the last admitted timestamp.
  [[nodiscard]] bool should_log_at(std::uint64_t monotonic_ns,
                                   std::uint64_t interval_ns) noexcept {
    if (interval_ns == 0)
      return true;

    std::uint64_t observed = last_plus_one_.load(std::memory_order_relaxed);
    for (;;) {
      if (observed != 0) {
        const std::uint64_t last = observed - 1;
        if (monotonic_ns < last || monotonic_ns - last < interval_ns)
          return false;
      }

      const std::uint64_t desired =
          monotonic_ns == (std::numeric_limits<std::uint64_t>::max)()
              ? (std::numeric_limits<std::uint64_t>::max)()
              : monotonic_ns + 1;
      // At the representational maximum, refusing a duplicate is safer than
      // allowing multiple concurrent calls to CAS the same value successfully.
      if (desired == observed)
        return false;
      if (last_plus_one_.compare_exchange_weak(observed, desired,
                                               std::memory_order_relaxed,
                                               std::memory_order_relaxed)) {
        return true;
      }
    }
  }

  [[nodiscard]] bool should_log_at(std::chrono::steady_clock::time_point now,
                                   std::chrono::nanoseconds interval) noexcept {
    const auto raw_now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             now.time_since_epoch())
                             .count();
    const auto raw_interval = interval.count();
    return should_log_at(
        raw_now > 0 ? static_cast<std::uint64_t>(raw_now) : 0,
        raw_interval > 0 ? static_cast<std::uint64_t>(raw_interval) : 0);
  }

  [[nodiscard]] bool should_log(std::chrono::nanoseconds interval) noexcept {
    return should_log_at(std::chrono::steady_clock::now(), interval);
  }

  // Reset is intended for an externally synchronized lifecycle transition.
  // Racing reset with checks is memory-safe but may admit one extra record.
  void reset() noexcept { last_plus_one_.store(0, std::memory_order_relaxed); }

private:
  // Zero means never admitted; otherwise the encoded timestamp is last + 1.
  std::atomic<std::uint64_t> last_plus_one_{0};
};

} // namespace hyle::obs

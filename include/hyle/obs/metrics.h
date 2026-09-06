/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary */
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include "hyle/core/assert.h"
#include "hyle/core/compiler.h"
#include "hyle/error/result.h"
#include "hyle/error/status.h"

namespace hyle::obs {

namespace detail {

template <class T>
concept CounterValue =
    (std::unsigned_integral<T> && !std::same_as<std::remove_cv_t<T>, bool>) ||
    std::floating_point<T>;

template <class T>
concept GaugeValue =
    (std::integral<T> && !std::same_as<std::remove_cv_t<T>, bool>) ||
    std::floating_point<T>;

template <std::floating_point T> constexpr bool finite_value(T value) noexcept {
  // Range comparisons reject both infinities and NaN without relying on a
  // possibly out-of-line C library classification helper in a hot update.
  return value >= std::numeric_limits<T>::lowest() &&
         value <= std::numeric_limits<T>::max();
}

template <CounterValue T>
HYLE_FORCE_INLINE void validate_counter_delta(T delta) noexcept {
  if constexpr (std::floating_point<T>) {
    HYLE_FATAL_ASSERT_MSG(delta >= T{} &&
                              delta <= std::numeric_limits<T>::max(),
                          "counter delta must be finite and non-negative");
  }
}

template <std::integral T> constexpr bool addition_fits(T lhs, T rhs) noexcept {
  if constexpr (std::unsigned_integral<T>) {
    return rhs <= std::numeric_limits<T>::max() - lhs;
  } else {
    if (rhs > T{})
      return lhs <= std::numeric_limits<T>::max() - rhs;
    if (rhs < T{})
      return lhs >= std::numeric_limits<T>::min() - rhs;
    return true;
  }
}

template <std::integral T>
constexpr bool subtraction_fits(T lhs, T rhs) noexcept {
  if constexpr (std::unsigned_integral<T>) {
    return rhs <= lhs;
  } else {
    if (rhs > T{})
      return lhs >= std::numeric_limits<T>::min() + rhs;
    if (rhs < T{})
      return lhs <= std::numeric_limits<T>::max() + rhs;
    return true;
  }
}

template <GaugeValue T> HYLE_FORCE_INLINE T checked_add(T lhs, T rhs) noexcept {
  if constexpr (std::integral<T>) {
    HYLE_FATAL_ASSERT_MSG(addition_fits(lhs, rhs),
                          "integral gauge addition is not representable");
  }
  return static_cast<T>(lhs + rhs);
}

template <GaugeValue T>
HYLE_FORCE_INLINE T checked_subtract(T lhs, T rhs) noexcept {
  if constexpr (std::integral<T>) {
    HYLE_FATAL_ASSERT_MSG(subtraction_fits(lhs, rhs),
                          "integral gauge subtraction is not representable");
  }
  return static_cast<T>(lhs - rhs);
}

template <class T>
inline constexpr bool kAlwaysLockFree = std::atomic<T>::is_always_lock_free;

} // namespace detail

// Owner-thread production counter. The object owns only its numeric state: a
// host exporter owns names, labels and scrape policy.
template <detail::CounterValue T = std::uint64_t> class Counter final {
public:
  using value_type = T;

  constexpr Counter() noexcept = default;
  explicit Counter(T initial) noexcept : value_(initial) {
    detail::validate_counter_delta(initial);
  }

  void add(T delta = T{1}) noexcept {
    detail::validate_counter_delta(delta);
    if constexpr (std::unsigned_integral<T>) {
      HYLE_DEBUG_ASSERT(delta <= std::numeric_limits<T>::max() - value_);
    }
    value_ = static_cast<T>(value_ + delta);
  }

  [[nodiscard]] constexpr T value() const noexcept { return value_; }

  T reset() noexcept { return std::exchange(value_, T{}); }

private:
  T value_{};
};

// Owner-thread last-value metric. IEEE floating values, including NaN used as
// an unavailable sentinel, are preserved. Integral arithmetic never invokes
// signed overflow or silently wraps.
template <detail::GaugeValue T = double> class Gauge final {
public:
  using value_type = T;

  constexpr Gauge() noexcept = default;
  explicit constexpr Gauge(T initial) noexcept : value_(initial) {}

  void set(T value) noexcept { value_ = value; }
  void add(T delta) noexcept { value_ = detail::checked_add(value_, delta); }
  void subtract(T delta) noexcept {
    value_ = detail::checked_subtract(value_, delta);
  }

  [[nodiscard]] constexpr T value() const noexcept { return value_; }

  T reset() noexcept { return std::exchange(value_, T{}); }

private:
  T value_{};
};

// Explicit shared-state variants. Metrics do not synchronize application data;
// relaxed ordering is sufficient. Reject targets whose standard atomic would
// hide a library lock rather than silently changing the resource contract.
template <detail::CounterValue T = std::uint64_t>
class ConcurrentCounter final {
public:
  using value_type = T;
  static_assert(detail::kAlwaysLockFree<T>,
                "ConcurrentCounter requires an always-lock-free atomic type");

  constexpr ConcurrentCounter() noexcept = default;
  explicit ConcurrentCounter(T initial) noexcept : value_(initial) {
    detail::validate_counter_delta(initial);
  }

  ConcurrentCounter(const ConcurrentCounter &) = delete;
  ConcurrentCounter &operator=(const ConcurrentCounter &) = delete;
  ConcurrentCounter(ConcurrentCounter &&) = delete;
  ConcurrentCounter &operator=(ConcurrentCounter &&) = delete;

  void add(T delta = T{1}) noexcept {
    detail::validate_counter_delta(delta);
    value_.fetch_add(delta, std::memory_order_relaxed);
  }

  [[nodiscard]] T value() const noexcept {
    return value_.load(std::memory_order_relaxed);
  }

  T reset() noexcept { return value_.exchange(T{}, std::memory_order_relaxed); }

private:
  std::atomic<T> value_{};
};

template <detail::GaugeValue T = double> class ConcurrentGauge final {
public:
  using value_type = T;
  static_assert(detail::kAlwaysLockFree<T>,
                "ConcurrentGauge requires an always-lock-free atomic type");

  constexpr ConcurrentGauge() noexcept = default;
  explicit constexpr ConcurrentGauge(T initial) noexcept : value_(initial) {}

  ConcurrentGauge(const ConcurrentGauge &) = delete;
  ConcurrentGauge &operator=(const ConcurrentGauge &) = delete;
  ConcurrentGauge(ConcurrentGauge &&) = delete;
  ConcurrentGauge &operator=(ConcurrentGauge &&) = delete;

  void set(T value) noexcept { value_.store(value, std::memory_order_relaxed); }

  void add(T delta) noexcept {
    if constexpr (std::floating_point<T>) {
      value_.fetch_add(delta, std::memory_order_relaxed);
    } else {
      update_integral([delta](T current) noexcept {
        return detail::checked_add(current, delta);
      });
    }
  }

  void subtract(T delta) noexcept {
    if constexpr (std::floating_point<T>) {
      value_.fetch_add(-delta, std::memory_order_relaxed);
    } else {
      update_integral([delta](T current) noexcept {
        return detail::checked_subtract(current, delta);
      });
    }
  }

  [[nodiscard]] T value() const noexcept {
    return value_.load(std::memory_order_relaxed);
  }

  T reset() noexcept { return value_.exchange(T{}, std::memory_order_relaxed); }

private:
  template <class Operation>
  void update_integral(Operation operation) noexcept {
    T current = value_.load(std::memory_order_relaxed);
    while (!value_.compare_exchange_weak(current, operation(current),
                                         std::memory_order_relaxed,
                                         std::memory_order_relaxed)) {
    }
  }

  std::atomic<T> value_{};
};

template <std::size_t N> class FixedBuckets final {
public:
  static Result<FixedBuckets> create(std::array<double, N> bounds) noexcept {
    for (std::size_t index = 0; index < N; ++index) {
      if (!detail::finite_value(bounds[index])) {
        return make_unexpected(
            Status::InvalidArgument("histogram bounds must be finite"));
      }
      if (index != 0 && !(bounds[index - 1] < bounds[index])) {
        return make_unexpected(Status::InvalidArgument(
            "histogram bounds must be strictly increasing"));
      }
    }
    return FixedBuckets(PrivateTag{}, std::move(bounds));
  }

  [[nodiscard]] constexpr const std::array<double, N> &values() const noexcept {
    return values_;
  }

  friend constexpr bool operator==(const FixedBuckets &,
                                   const FixedBuckets &) noexcept = default;

private:
  struct PrivateTag {};

  constexpr FixedBuckets(PrivateTag, std::array<double, N> bounds) noexcept
      : values_(std::move(bounds)) {}

  std::array<double, N> values_{};
};

template <std::size_t N> class FixedHistogram;
template <std::size_t N> class ConcurrentFixedHistogram;

template <std::size_t N> class HistogramSnapshot final {
public:
  [[nodiscard]] constexpr const std::array<double, N> &bounds() const noexcept {
    return bounds_;
  }

  [[nodiscard]] constexpr const std::array<std::uint64_t, N + 1> &
  bucket_counts() const noexcept {
    return bucket_counts_;
  }

  [[nodiscard]] constexpr double sum() const noexcept { return sum_; }

  [[nodiscard]] std::uint64_t count() const noexcept {
    std::uint64_t result = 0;
    for (std::uint64_t bucket_count : bucket_counts_) {
      HYLE_DEBUG_ASSERT(bucket_count <=
                        std::numeric_limits<std::uint64_t>::max() - result);
      result += bucket_count;
    }
    return result;
  }

  [[nodiscard]] double bucket_upper_bound(std::size_t index) const noexcept {
    HYLE_FATAL_ASSERT_MSG(index <= N, "histogram bucket index is out of range");
    return index == N ? std::numeric_limits<double>::infinity()
                      : bounds_[index];
  }

  Status merge(const HistogramSnapshot &other) noexcept {
    if (bounds_ != other.bounds_) {
      return Status::InvalidArgument(
          "histogram snapshots use different bounds");
    }
    for (std::size_t index = 0; index <= N; ++index) {
      if (other.bucket_counts_[index] >
          std::numeric_limits<std::uint64_t>::max() - bucket_counts_[index]) {
        return Status::OutOfRange("histogram bucket count cannot be merged");
      }
    }
    for (std::size_t index = 0; index <= N; ++index)
      bucket_counts_[index] += other.bucket_counts_[index];
    sum_ += other.sum_;
    return Status::Ok();
  }

  Result<double> quantile_upper_bound(double quantile) const noexcept {
    if (!(quantile >= 0.0 && quantile <= 1.0)) {
      return make_unexpected(
          Status::InvalidArgument("quantile must be finite and in [0, 1]"));
    }
    const std::uint64_t total = count();
    if (total == 0) {
      return make_unexpected(
          Status::FailedPrecondition("histogram snapshot is empty"));
    }

    std::uint64_t rank = 1;
    if (quantile >= 1.0) {
      rank = total;
    } else if (quantile > 0.0) {
      const long double scaled = std::ceil(static_cast<long double>(quantile) *
                                           static_cast<long double>(total));
      rank = scaled >= static_cast<long double>(total)
                 ? total
                 : static_cast<std::uint64_t>(scaled);
      if (rank == 0)
        rank = 1;
    }

    std::uint64_t cumulative = 0;
    for (std::size_t index = 0; index <= N; ++index) {
      cumulative += bucket_counts_[index];
      if (cumulative >= rank)
        return bucket_upper_bound(index);
    }
    HYLE_FATAL_ASSERT_MSG(false, "histogram count invariant is broken");
  }

private:
  template <std::size_t> friend class FixedHistogram;
  template <std::size_t> friend class ConcurrentFixedHistogram;

  constexpr HistogramSnapshot(
      const std::array<double, N> &bounds,
      const std::array<std::uint64_t, N + 1> &bucket_counts,
      double sum) noexcept
      : bounds_(bounds), bucket_counts_(bucket_counts), sum_(sum) {}

  std::array<double, N> bounds_{};
  std::array<std::uint64_t, N + 1> bucket_counts_{};
  double sum_ = 0.0;
};

template <std::size_t N> class FixedHistogram final {
public:
  explicit constexpr FixedHistogram(FixedBuckets<N> buckets) noexcept
      : buckets_(std::move(buckets)) {}

  [[nodiscard]] bool observe(double value,
                             std::uint64_t multiplicity = 1) noexcept {
    if (multiplicity == 0)
      return true;
    if (!detail::finite_value(value))
      return false;
    const auto iterator = std::lower_bound(buckets_.values().begin(),
                                           buckets_.values().end(), value);
    const std::size_t index =
        static_cast<std::size_t>(iterator - buckets_.values().begin());
    HYLE_DEBUG_ASSERT(multiplicity <=
                      std::numeric_limits<std::uint64_t>::max() -
                          bucket_counts_[index]);
    bucket_counts_[index] += multiplicity;
    sum_ += value * static_cast<double>(multiplicity);
    return true;
  }

  [[nodiscard]] HistogramSnapshot<N> snapshot() const noexcept {
    return HistogramSnapshot<N>(buckets_.values(), bucket_counts_, sum_);
  }

  void reset() noexcept {
    bucket_counts_.fill(0);
    sum_ = 0.0;
  }

private:
  FixedBuckets<N> buckets_;
  std::array<std::uint64_t, N + 1> bucket_counts_{};
  double sum_ = 0.0;
};

template <std::size_t N> class ConcurrentFixedHistogram final {
public:
  static_assert(detail::kAlwaysLockFree<std::uint64_t>,
                "ConcurrentFixedHistogram requires lock-free uint64 atomics");
  static_assert(detail::kAlwaysLockFree<double>,
                "ConcurrentFixedHistogram requires lock-free double atomics");

  explicit ConcurrentFixedHistogram(FixedBuckets<N> buckets) noexcept
      : buckets_(std::move(buckets)) {
    for (auto &bucket_count : bucket_counts_)
      bucket_count.store(0, std::memory_order_relaxed);
  }

  ConcurrentFixedHistogram(const ConcurrentFixedHistogram &) = delete;
  ConcurrentFixedHistogram &
  operator=(const ConcurrentFixedHistogram &) = delete;
  ConcurrentFixedHistogram(ConcurrentFixedHistogram &&) = delete;
  ConcurrentFixedHistogram &operator=(ConcurrentFixedHistogram &&) = delete;

  [[nodiscard]] bool observe(double value,
                             std::uint64_t multiplicity = 1) noexcept {
    if (multiplicity == 0)
      return true;
    if (!detail::finite_value(value))
      return false;
    const auto iterator = std::lower_bound(buckets_.values().begin(),
                                           buckets_.values().end(), value);
    const std::size_t index =
        static_cast<std::size_t>(iterator - buckets_.values().begin());
    bucket_counts_[index].fetch_add(multiplicity, std::memory_order_relaxed);
    sum_.fetch_add(value * static_cast<double>(multiplicity),
                   std::memory_order_relaxed);
    return true;
  }

  [[nodiscard]] HistogramSnapshot<N> snapshot() const noexcept {
    std::array<std::uint64_t, N + 1> counts{};
    for (std::size_t index = 0; index <= N; ++index) {
      counts[index] = bucket_counts_[index].load(std::memory_order_relaxed);
    }
    return HistogramSnapshot<N>(buckets_.values(), counts,
                                sum_.load(std::memory_order_relaxed));
  }

  // Reset is a lifecycle operation: writers must be quiescent.
  void reset() noexcept {
    for (auto &bucket_count : bucket_counts_)
      bucket_count.store(0, std::memory_order_relaxed);
    sum_.store(0.0, std::memory_order_relaxed);
  }

private:
  FixedBuckets<N> buckets_;
  std::array<std::atomic<std::uint64_t>, N + 1> bucket_counts_{};
  std::atomic<double> sum_{};
};

} // namespace hyle::obs

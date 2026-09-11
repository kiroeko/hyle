/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary */

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "hyle/error/status.h"
#include "hyle/obs/metrics.h"

namespace hyle::obs {
namespace {

TEST(MetricsValue, OwnerThreadCounterAndGauge) {
  Counter<> requests;
  requests.add();
  requests.add(41);
  EXPECT_EQ(requests.value(), 42u);
  EXPECT_EQ(requests.reset(), 42u);
  EXPECT_EQ(requests.value(), 0u);

  Counter<double> seconds(0.25);
  seconds.add(0.5);
  EXPECT_DOUBLE_EQ(seconds.value(), 0.75);

  Gauge<std::int64_t> queue_depth(4);
  queue_depth.add(3);
  queue_depth.subtract(9);
  EXPECT_EQ(queue_depth.value(), -2);
  queue_depth.set(17);
  EXPECT_EQ(queue_depth.reset(), 17);

  Gauge<> utilization;
  utilization.set(std::numeric_limits<double>::quiet_NaN());
  EXPECT_TRUE(std::isnan(utilization.value()));
}

TEST(MetricsValue, InvalidCounterAndIntegralGaugeArithmeticFailFast) {
  EXPECT_DEATH(
      {
        Counter<double> counter;
        counter.add(-1.0);
      },
      "");
  EXPECT_DEATH(
      { Counter<double> counter(std::numeric_limits<double>::quiet_NaN()); },
      "");
  EXPECT_DEATH(
      {
        Gauge<std::uint8_t> gauge(255);
        gauge.add(1);
      },
      "");
  EXPECT_DEATH(
      {
        Gauge<std::int8_t> gauge(-128);
        gauge.subtract(1);
      },
      "");
  EXPECT_DEATH(
      {
        Gauge<std::int8_t> gauge(127);
        gauge.add(1);
      },
      "");
  EXPECT_DEATH(
      {
        Gauge<std::int8_t> gauge(127);
        gauge.subtract(-1);
      },
      "");
  EXPECT_DEATH(
      {
        Gauge<std::uint8_t> gauge(0);
        gauge.subtract(1);
      },
      "");
}

TEST(MetricsHistogram, ValidatesFixedBounds) {
  EXPECT_TRUE((FixedBuckets<0>::create({}).has_value()));
  EXPECT_TRUE((FixedBuckets<3>::create({1.0, 2.0, 3.0}).has_value()));

  EXPECT_EQ(FixedBuckets<2>::create({2.0, 1.0}).error().code(),
            StatusCode::InvalidArgument);
  EXPECT_EQ(FixedBuckets<2>::create({1.0, 1.0}).error().code(),
            StatusCode::InvalidArgument);
  EXPECT_EQ(FixedBuckets<1>::create({std::numeric_limits<double>::quiet_NaN()})
                .error()
                .code(),
            StatusCode::InvalidArgument);
  EXPECT_EQ(FixedBuckets<1>::create({std::numeric_limits<double>::infinity()})
                .error()
                .code(),
            StatusCode::InvalidArgument);
}

TEST(MetricsHistogram, ObservesBoundariesTailAndMultiplicity) {
  auto buckets = FixedBuckets<2>::create({1.0, 5.0});
  ASSERT_TRUE(buckets.has_value());
  FixedHistogram histogram(*buckets);

  EXPECT_TRUE(histogram.observe(-2.0));
  EXPECT_TRUE(histogram.observe(1.0, 2));
  EXPECT_TRUE(histogram.observe(1.5));
  EXPECT_TRUE(histogram.observe(5.0));
  EXPECT_TRUE(histogram.observe(6.0, 3));
  EXPECT_FALSE(histogram.observe(std::numeric_limits<double>::infinity()));
  EXPECT_FALSE(histogram.observe(std::numeric_limits<double>::quiet_NaN()));
  EXPECT_TRUE(histogram.observe(std::numeric_limits<double>::quiet_NaN(), 0));

  const HistogramSnapshot snapshot = histogram.snapshot();
  EXPECT_EQ(snapshot.bucket_counts(), (std::array<std::uint64_t, 3>{3, 2, 3}));
  EXPECT_EQ(snapshot.count(), 8u);
  EXPECT_DOUBLE_EQ(snapshot.sum(), 24.5);
  EXPECT_DOUBLE_EQ(snapshot.bucket_upper_bound(0), 1.0);
  EXPECT_DOUBLE_EQ(snapshot.bucket_upper_bound(1), 5.0);
  EXPECT_TRUE(std::isinf(snapshot.bucket_upper_bound(2)));

  histogram.reset();
  EXPECT_EQ(histogram.snapshot().count(), 0u);
}

TEST(MetricsHistogram, MergesWorkerSnapshotsAndQueriesQuantileBucket) {
  auto buckets = FixedBuckets<2>::create({1.0, 3.0});
  ASSERT_TRUE(buckets.has_value());
  FixedHistogram worker_a(*buckets);
  FixedHistogram worker_b(*buckets);
  EXPECT_TRUE(worker_a.observe(0.5));
  EXPECT_TRUE(worker_a.observe(2.0, 2));
  EXPECT_TRUE(worker_b.observe(2.0));
  EXPECT_TRUE(worker_b.observe(8.0));

  HistogramSnapshot merged = worker_a.snapshot();
  EXPECT_TRUE(merged.merge(worker_b.snapshot()).ok());
  EXPECT_EQ(merged.bucket_counts(), (std::array<std::uint64_t, 3>{1, 3, 1}));
  EXPECT_EQ(merged.count(), 5u);
  EXPECT_DOUBLE_EQ(*merged.quantile_upper_bound(0.0), 1.0);
  EXPECT_DOUBLE_EQ(*merged.quantile_upper_bound(0.5), 3.0);
  EXPECT_TRUE(std::isinf(*merged.quantile_upper_bound(1.0)));
  EXPECT_EQ(merged.quantile_upper_bound(-0.1).error().code(),
            StatusCode::InvalidArgument);
  EXPECT_EQ(merged.quantile_upper_bound(1.1).error().code(),
            StatusCode::InvalidArgument);
  EXPECT_EQ(merged
                .quantile_upper_bound(
                    std::numeric_limits<double>::quiet_NaN())
                .error()
                .code(),
            StatusCode::InvalidArgument);

  FixedHistogram empty(*buckets);
  EXPECT_EQ(empty.snapshot().quantile_upper_bound(0.5).error().code(),
            StatusCode::FailedPrecondition);

  auto other_buckets = FixedBuckets<2>::create({1.0, 4.0});
  ASSERT_TRUE(other_buckets.has_value());
  FixedHistogram other(*other_buckets);
  EXPECT_EQ(merged.merge(other.snapshot()).code(), StatusCode::InvalidArgument);
}

TEST(MetricsHistogram, MergeRejectsCountOverflowWithoutPartialMutation) {
  auto buckets = FixedBuckets<1>::create({1.0});
  ASSERT_TRUE(buckets.has_value());
  FixedHistogram large(*buckets);
  FixedHistogram one(*buckets);
  EXPECT_TRUE(large.observe(0.5, std::numeric_limits<std::uint64_t>::max()));
  EXPECT_TRUE(one.observe(0.5));

  HistogramSnapshot snapshot = large.snapshot();
  EXPECT_EQ(snapshot.merge(one.snapshot()).code(), StatusCode::OutOfRange);
  EXPECT_EQ(snapshot.bucket_counts()[0],
            std::numeric_limits<std::uint64_t>::max());
}

TEST(MetricsConcurrent, SharedValuesAndHistogramAreExactAfterQuiescence) {
  static_assert(!std::is_copy_constructible_v<ConcurrentCounter<>>);
  static_assert(!std::is_move_constructible_v<ConcurrentGauge<>>);
  static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
  static_assert(std::atomic<double>::is_always_lock_free);

  ConcurrentCounter<> tokens;
  ConcurrentGauge<std::int64_t> queue_depth;
  auto buckets = FixedBuckets<2>::create({1.0, 4.0});
  ASSERT_TRUE(buckets.has_value());
  ConcurrentFixedHistogram histogram(*buckets);

  EXPECT_TRUE(histogram.observe(std::numeric_limits<double>::quiet_NaN(), 0));
  EXPECT_FALSE(histogram.observe(std::numeric_limits<double>::infinity()));

  constexpr std::size_t kThreads = 4;
  constexpr std::size_t kIterations = 10'000;
  std::atomic<bool> observations_ok{true};
  std::vector<std::thread> threads;
  for (std::size_t thread = 0; thread < kThreads; ++thread) {
    threads.emplace_back([&] {
      for (std::size_t iteration = 0; iteration < kIterations; ++iteration) {
        tokens.add(2);
        queue_depth.add(1);
        if (!histogram.observe(2.0, 2))
          observations_ok.store(false, std::memory_order_relaxed);
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  EXPECT_TRUE(observations_ok.load(std::memory_order_relaxed));

  const std::uint64_t expected = kThreads * kIterations * 2;
  EXPECT_EQ(tokens.value(), expected);
  EXPECT_EQ(queue_depth.value(),
            static_cast<std::int64_t>(kThreads * kIterations));
  const HistogramSnapshot snapshot = histogram.snapshot();
  EXPECT_EQ(snapshot.count(), expected);
  EXPECT_EQ(snapshot.bucket_counts(),
            (std::array<std::uint64_t, 3>{0, expected, 0}));
  EXPECT_DOUBLE_EQ(snapshot.sum(), static_cast<double>(expected) * 2.0);

  EXPECT_EQ(tokens.reset(), expected);
  EXPECT_EQ(queue_depth.reset(),
            static_cast<std::int64_t>(kThreads * kIterations));
  histogram.reset();
  EXPECT_EQ(histogram.snapshot().count(), 0u);
}

} // namespace
} // namespace hyle::obs

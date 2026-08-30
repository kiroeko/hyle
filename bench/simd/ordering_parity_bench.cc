// SIMD-11 bounded ordering instrument.
//
// Mutating algorithms cannot reuse one input across nanobench epochs: after
// the first invocation it is already sorted. Every measured invocation here
// consumes a distinct, preallocated and prefilled buffer. epochIterations(1)
// makes that accounting exact, so input materialization, allocation and page
// faults stay outside the timed region. The ordinary HYLE API and the direct
// locked-Highway peer therefore see the same fresh-input and ISA contract.

#include <nanobench.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "hyle/core/compiler.h"
#include "hyle/simd/runtime.h"
#include "hyle/simd/sort.h"
#include "op_parity_engine.h"
#include "ordering_kernels.h"

namespace nb = ankerl::nanobench;
namespace hs = hyle::simd;

namespace {

constexpr std::size_t kResidentEpochs = 31;
constexpr std::size_t kStreamingEpochs = 11;
constexpr std::size_t kResidentCount = 4096;
constexpr std::size_t kStreamingCount = 1u << 20;
constexpr std::size_t kTopK = 10;
constexpr std::size_t kPartialRowsPerSample = 16;

[[noreturn]] void qualification_failure(const char* message) {
  std::fprintf(stderr, "SIMD-11 ordering qualification failed: %s\n", message);
  std::abort();
}

void require(bool condition, const char* message) {
  if (!condition) qualification_failure(message);
}

std::vector<float> make_f32_input(std::size_t count) {
  std::mt19937_64 rng(0x53494D443131ULL + count);
  std::uniform_real_distribution<float> distribution(-4096.0f, 4096.0f);
  std::vector<float> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    // A modest duplicate population represents logits/tables without turning
    // the row into an all-equal algorithm special case.
    values[i] = (i % 19 == 0) ? static_cast<float>(i % 31)
                              : distribution(rng);
  }
  return values;
}

std::vector<std::uint32_t> make_u32_input(std::size_t count) {
  std::mt19937_64 rng(0x52414449583131ULL + count);
  std::vector<std::uint32_t> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    values[i] = (i % 23 == 0) ? static_cast<std::uint32_t>(i % 257)
                              : static_cast<std::uint32_t>(rng());
  }
  return values;
}

template <class T>
class FreshBank {
 public:
  explicit FreshBank(const std::vector<T>& input, std::size_t calls)
      : count_(input.size()), storage_(count_ * calls) {
    for (std::size_t copy = 0; copy < calls; ++copy) {
      std::copy(input.begin(), input.end(),
                storage_.begin() + static_cast<std::ptrdiff_t>(copy * count_));
    }
  }

  T* data(std::size_t invocation) noexcept {
    return storage_.data() + invocation * count_;
  }

 private:
  std::size_t count_;
  std::vector<T> storage_;
};

template <class T>
class RadixBank {
 public:
  explicit RadixBank(const std::vector<T>& input, std::size_t calls)
      : inputs_(input, calls), scratch_(input.size() * calls) {}

  T* input(std::size_t invocation) noexcept { return inputs_.data(invocation); }
  T* scratch(std::size_t invocation, std::size_t count) noexcept {
    return scratch_.data() + invocation * count;
  }

 private:
  FreshBank<T> inputs_;
  std::vector<T> scratch_;
};

class ArgsortBank {
 public:
  explicit ArgsortBank(std::size_t count, std::size_t calls)
      : count_(count), indices_(count * calls), scratch_(count * calls) {}

  std::uint32_t* indices(std::size_t invocation) noexcept {
    return indices_.data() + invocation * count_;
  }
  std::uint32_t* scratch(std::size_t invocation) noexcept {
    return scratch_.data() + invocation * count_;
  }

 private:
  std::size_t count_;
  std::vector<std::uint32_t> indices_;
  std::vector<std::uint32_t> scratch_;
};

template <class Bank, class Factory>
std::pair<std::unique_ptr<Bank>, std::unique_ptr<Bank>> make_ordered_banks(
    Factory make) {
  std::unique_ptr<Bank> candidate;
  std::unique_ptr<Bank> peer;
  // Reverse allocation order together with temporal/code-layout order. This
  // prevents one side from permanently owning the first virtual-address run.
  if (hyle_bench::hwy_first()) {
    peer = make();
    candidate = make();
  } else {
    candidate = make();
    peer = make();
  }
  return {std::move(candidate), std::move(peer)};
}

nb::Bench fresh_input_bench(const std::string& title, std::size_t count,
                            std::size_t epochs) {
  nb::Bench bench;
  bench.title(title)
      .unit("value")
      .relative(true)
      .batch(count)
      .epochs(epochs)
      .epochIterations(1);
  return bench;
}

HYLE_NOINLINE void anchor_add(const float* x, const float* y, float* output,
                              std::size_t count) noexcept {
  for (std::size_t i = 0; i < count; ++i) output[i] = x[i] + y[i];
}

void run_anchor() {
  std::vector<float> x(kResidentCount), y(kResidentCount), output(kResidentCount);
  for (std::size_t i = 0; i < kResidentCount; ++i) {
    x[i] = static_cast<float>(i & 31u);
    y[i] = static_cast<float>((i * 3) & 63u);
  }
  const float* x_data = x.data();
  const float* y_data = y.data();
  float* output_data = output.data();
  nb::Bench bench;
  bench.title("Add f32 (4096)")
      .unit("value")
      .relative(true)
      .batch(kResidentCount)
      .minEpochIterations(4096);
  HYLE_RUN_PAIR(
      bench,
      [&](auto rep) {
        hyle_bench::stagger(rep);
        anchor_add(x_data, y_data, output_data, kResidentCount);
        hyle_bench::mark(rep);
      },
      [&](auto rep) {
        hyle_bench::stagger(rep);
        anchor_add(x_data, y_data, output_data, kResidentCount);
        hyle_bench::mark(rep);
      });
}

bool equivalent_float(float a, float b) noexcept {
  return (std::isnan(a) && std::isnan(b)) || a == b;
}

void qualify_peer() {
  require(hs::selected_target(hs::KernelFamily::kSort).isa == hs::Isa::kAvx2,
          "W1 benchmark requires the ordinary AVX2 product route");

  auto input = make_f32_input(kResidentCount);
  auto candidate = input;
  auto peer = input;
  hs::sort(candidate.data(), candidate.size());
  hyle_bench::ordering::highway_sort_f32(peer.data(), peer.size());
  require(candidate == peer, "finite full-sort peer mismatch");

  candidate = input;
  peer = input;
  hs::partial_sort(candidate.data(), candidate.size(), kTopK);
  hyle_bench::ordering::highway_partial_sort_f32(peer.data(), peer.size(),
                                                 kTopK);
  require(std::equal(candidate.begin(), candidate.begin() + kTopK,
                     peer.begin()),
          "finite partial-sort peer mismatch");

  const float nan = std::numeric_limits<float>::quiet_NaN();
  candidate = {nan, 4.0f, -0.0f, 2.0f, nan, -3.0f, 0.0f, 9.0f};
  peer = candidate;
  hs::sort(candidate.data(), candidate.size());
  hyle_bench::ordering::highway_sort_f32(peer.data(), peer.size());
  for (std::size_t i = 0; i < candidate.size(); ++i) {
    require(equivalent_float(candidate[i], peer[i]),
            "NaN/signed-zero peer contract mismatch");
  }
}

void run_comparison_sort(std::size_t count) {
  const std::size_t epochs =
      count == kResidentCount ? kResidentEpochs : kStreamingEpochs;
  const std::size_t calls = hyle_bench::kReplicas * epochs;
  const auto input = make_f32_input(count);
  auto [candidate_bank, peer_bank] =
      make_ordered_banks<FreshBank<float>>(
          [&] { return std::make_unique<FreshBank<float>>(input, calls); });
  std::size_t candidate_call = 0;
  std::size_t peer_call = 0;
  auto bench = fresh_input_bench(
      "full sort f32 (" + std::to_string(count) + ")", count, epochs);
  HYLE_RUN_PAIR(
      bench,
      [&](auto rep) {
        hyle_bench::stagger(rep);
        float* values = candidate_bank->data(candidate_call++);
        hs::sort(values, count);
        nb::doNotOptimizeAway(values);
        hyle_bench::mark(rep);
      },
      [&](auto rep) {
        hyle_bench::stagger(rep);
        float* values = peer_bank->data(peer_call++);
        hyle_bench::ordering::highway_sort_f32(values, count);
        nb::doNotOptimizeAway(values);
        hyle_bench::mark(rep);
      });
  require(candidate_call == calls && peer_call == calls,
          "fresh full-sort bank accounting changed");
}

void run_comparison_partial() {
  constexpr std::size_t kCalls = hyle_bench::kReplicas * kResidentEpochs *
                                 kPartialRowsPerSample;
  const auto input = make_f32_input(kResidentCount);
  auto [candidate_bank, peer_bank] =
      make_ordered_banks<FreshBank<float>>(
          [&] { return std::make_unique<FreshBank<float>>(input, kCalls); });
  std::size_t candidate_call = 0;
  std::size_t peer_call = 0;
  auto bench = fresh_input_bench("partial sort f32 (4096,k=10)",
                                 kResidentCount * kPartialRowsPerSample,
                                 kResidentEpochs);
  HYLE_RUN_PAIR(
      bench,
      [&](auto rep) {
        hyle_bench::stagger(rep);
        for (std::size_t row = 0; row < kPartialRowsPerSample; ++row) {
          float* values = candidate_bank->data(candidate_call++);
          hs::partial_sort(values, kResidentCount, kTopK);
          nb::doNotOptimizeAway(values);
        }
        hyle_bench::mark(rep);
      },
      [&](auto rep) {
        hyle_bench::stagger(rep);
        for (std::size_t row = 0; row < kPartialRowsPerSample; ++row) {
          float* values = peer_bank->data(peer_call++);
          hyle_bench::ordering::highway_partial_sort_f32(
              values, kResidentCount, kTopK);
          nb::doNotOptimizeAway(values);
        }
        hyle_bench::mark(rep);
      });
  require(candidate_call == kCalls && peer_call == kCalls,
          "fresh partial-sort bank accounting changed");
}

void run_radix_info() {
  constexpr std::size_t kCalls = hyle_bench::kReplicas * kStreamingEpochs;
  const auto input = make_u32_input(kStreamingCount);
  auto [first_bank, repeat_bank] = make_ordered_banks<RadixBank<std::uint32_t>>(
      [&] {
        return std::make_unique<RadixBank<std::uint32_t>>(input, kCalls);
      });
  std::size_t first_call = 0;
  std::size_t repeat_call = 0;
  auto bench = fresh_input_bench("scratch radix u32 (1048576)",
                                 kStreamingCount, kStreamingEpochs);
  HYLE_RUN_INFO_PAIR(
      bench,
      [&](auto rep) {
        hyle_bench::stagger(rep);
        const std::size_t invocation = first_call++;
        auto* values = first_bank->input(invocation);
        hs::radix_sort(values, kStreamingCount,
                       std::span<std::uint32_t>(
                           first_bank->scratch(invocation, kStreamingCount),
                           kStreamingCount));
        nb::doNotOptimizeAway(values);
        hyle_bench::mark(rep);
      },
      [&](auto rep) {
        hyle_bench::stagger(rep);
        const std::size_t invocation = repeat_call++;
        auto* values = repeat_bank->input(invocation);
        hs::radix_sort(values, kStreamingCount,
                       std::span<std::uint32_t>(
                           repeat_bank->scratch(invocation, kStreamingCount),
                           kStreamingCount));
        nb::doNotOptimizeAway(values);
        hyle_bench::mark(rep);
      });
  require(first_call == kCalls && repeat_call == kCalls,
          "fresh radix bank accounting changed");
}

void run_argsort_info() {
  constexpr std::size_t kCalls = hyle_bench::kReplicas * kStreamingEpochs;
  const auto keys = make_f32_input(kStreamingCount);
  auto [first_bank, repeat_bank] = make_ordered_banks<ArgsortBank>(
      [&] {
        return std::make_unique<ArgsortBank>(kStreamingCount, kCalls);
      });
  std::size_t first_call = 0;
  std::size_t repeat_call = 0;
  auto bench = fresh_input_bench("stable argsort f32 (1048576)",
                                 kStreamingCount, kStreamingEpochs);
  HYLE_RUN_INFO_PAIR(
      bench,
      [&](auto rep) {
        hyle_bench::stagger(rep);
        const std::size_t invocation = first_call++;
        auto* indices = first_bank->indices(invocation);
        hs::argsort(keys.data(), indices, kStreamingCount,
                    std::span<std::uint32_t>(first_bank->scratch(invocation),
                                             kStreamingCount));
        nb::doNotOptimizeAway(indices);
        hyle_bench::mark(rep);
      },
      [&](auto rep) {
        hyle_bench::stagger(rep);
        const std::size_t invocation = repeat_call++;
        auto* indices = repeat_bank->indices(invocation);
        hs::argsort(keys.data(), indices, kStreamingCount,
                    std::span<std::uint32_t>(repeat_bank->scratch(invocation),
                                             kStreamingCount));
        nb::doNotOptimizeAway(indices);
        hyle_bench::mark(rep);
      });
  require(first_call == kCalls && repeat_call == kCalls,
          "fresh argsort bank accounting changed");
}

}  // namespace

int main() {
  hyle_bench::pin_and_boost();
  std::printf(
      "# HYLE_ORDERING_PARITY candidate=ordinary-hyle peer=locked-highway "
      "isa=avx2 resident_epochs=%zu streaming_epochs=%zu "
      "resident_sort_fresh_buffers=%zu partial_fresh_buffers=%zu "
      "streaming_fresh_buffers=%zu "
      "partial_rows_per_sample=%zu\n",
      kResidentEpochs, kStreamingEpochs,
      hyle_bench::kReplicas * kResidentEpochs,
      hyle_bench::kReplicas * kResidentEpochs * kPartialRowsPerSample,
      hyle_bench::kReplicas * kStreamingEpochs, kPartialRowsPerSample);
  std::printf(
      "# HYLE_ORDERING_NOT_COMPARABLE rows=radix,argsort "
      "reason=no-qualified-peer-with-same-stability-nan-workspace-contract\n");
  qualify_peer();
  run_anchor();
  run_comparison_sort(kResidentCount);
  run_comparison_sort(kStreamingCount);
  run_comparison_partial();
  run_radix_info();
  run_argsort_info();
}

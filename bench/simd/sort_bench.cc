#include <nanobench.h>

#define VQSORT_ONLY_STATIC 1
#include "hwy/highway.h"
#include "hwy/contrib/sort/vqsort-inl.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "hyle/macros.h"
#include "hyle/simd/sort.h"

namespace {
namespace nb = ankerl::nanobench;
using hyle::simd::KeyValue32;
using hyle::simd::KeyValue64;
using hyle::simd::SortOrder;

template <typename T>
struct IsKeyValue : std::false_type {};
template <>
struct IsKeyValue<KeyValue32> : std::true_type {};
template <>
struct IsKeyValue<KeyValue64> : std::true_type {};

template <typename T>
auto key_of(const T& value) noexcept {
  if constexpr (IsKeyValue<T>::value) return value.key;
  else return value;
}

template <typename T>
std::vector<T> make_input(std::size_t n, bool few_unique) {
  std::mt19937_64 rng(0x5077ULL + n * 31 + sizeof(T));
  std::vector<T> values(n);
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint64_t raw = few_unique ? (rng() & 7u) : rng();
    if constexpr (std::is_same_v<T, KeyValue32>) {
      values[i] = {static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(raw)};
    } else if constexpr (std::is_same_v<T, KeyValue64>) {
      values[i] = {static_cast<std::uint64_t>(i), raw};
    } else {
      values[i] = static_cast<T>(raw);
    }
  }
  return values;
}

template <typename T>
HYLE_NOINLINE void hyle_sort(T* values, std::size_t n, bool ascending) noexcept {
  hyle::simd::sort(values, n,
                   ascending ? SortOrder::kAscending : SortOrder::kDescending);
}

template <typename T>
HYLE_NOINLINE void hyle_partial_sort(T* values, std::size_t n, std::size_t k,
                                     bool ascending) noexcept {
  hyle::simd::partial_sort(values, n, k,
                           ascending ? SortOrder::kAscending : SortOrder::kDescending);
}

// select (Quickselect / nth_element): partition so the k-th order statistic is in place and all
// smaller keys precede it (order within each side unspecified). Cheaper than partial_sort (no sort
// of the low-k). HYLE select takes plain numeric keys (not KeyValue) -> only the scalar-key types.
template <typename T>
HYLE_NOINLINE void hyle_select(T* keys, std::size_t n, std::size_t k, bool ascending) noexcept {
  hyle::simd::select(keys, n, k,
                     ascending ? SortOrder::kAscending : SortOrder::kDescending);
}
template <typename T>
HYLE_NOINLINE void std_nth(T* keys, std::size_t n, std::size_t k, bool ascending) noexcept {
  auto cmp = [ascending](T a, T b) { return ascending ? a < b : a > b; };
  std::nth_element(keys, keys + k, keys + n, cmp);
}
template <typename T>
HYLE_NOINLINE void highway_select(T* keys, std::size_t n, std::size_t k, bool ascending) noexcept {
  if (ascending) hwy::HWY_NAMESPACE::VQSelectStatic(keys, n, k, hwy::SortAscending{});
  else hwy::HWY_NAMESPACE::VQSelectStatic(keys, n, k, hwy::SortDescending{});
}

template <typename T>
HYLE_NOINLINE void candidate_partial_sort(T* values, std::size_t n, std::size_t k,
                                           bool ascending) noexcept {
  auto before = [ascending](const T& a, const T& b) {
    return ascending ? key_of(a) < key_of(b) : key_of(a) > key_of(b);
  };
  for (std::size_t i = 1; i < k; ++i) {
    T value = values[i];
    std::size_t pos = i;
    while (pos != 0 && before(value, values[pos - 1])) {
      values[pos] = values[pos - 1];
      --pos;
    }
    values[pos] = value;
  }
  for (std::size_t i = k; i < n; ++i) {
    if (!before(values[i], values[k - 1])) continue;
    T value = values[i];
    values[i] = values[k - 1];
    std::size_t pos = k - 1;
    while (pos != 0 && before(value, values[pos - 1])) {
      values[pos] = values[pos - 1];
      --pos;
    }
    values[pos] = value;
  }
}

template <typename T>
HYLE_NOINLINE void highway_sort(T* values, std::size_t n, bool ascending) noexcept {
  if constexpr (std::is_same_v<T, KeyValue32>) {
    auto* h = reinterpret_cast<hwy::K32V32*>(values);
    if (ascending) hwy::HWY_NAMESPACE::VQSortStatic(h, n, hwy::SortAscending{});
    else hwy::HWY_NAMESPACE::VQSortStatic(h, n, hwy::SortDescending{});
  } else if constexpr (std::is_same_v<T, KeyValue64>) {
    auto* h = reinterpret_cast<hwy::K64V64*>(values);
    if (ascending) hwy::HWY_NAMESPACE::VQSortStatic(h, n, hwy::SortAscending{});
    else hwy::HWY_NAMESPACE::VQSortStatic(h, n, hwy::SortDescending{});
  } else {
    if (ascending) hwy::HWY_NAMESPACE::VQSortStatic(values, n, hwy::SortAscending{});
    else hwy::HWY_NAMESPACE::VQSortStatic(values, n, hwy::SortDescending{});
  }
}

template <typename T>
HYLE_NOINLINE void highway_partial_sort(T* values, std::size_t n, std::size_t k,
                                         bool ascending) noexcept {
  if constexpr (std::is_same_v<T, KeyValue32>) {
    auto* h = reinterpret_cast<hwy::K32V32*>(values);
    if (ascending) hwy::HWY_NAMESPACE::VQPartialSortStatic(h, n, k, hwy::SortAscending{});
    else hwy::HWY_NAMESPACE::VQPartialSortStatic(h, n, k, hwy::SortDescending{});
  } else if constexpr (std::is_same_v<T, KeyValue64>) {
    auto* h = reinterpret_cast<hwy::K64V64*>(values);
    if (ascending) hwy::HWY_NAMESPACE::VQPartialSortStatic(h, n, k, hwy::SortAscending{});
    else hwy::HWY_NAMESPACE::VQPartialSortStatic(h, n, k, hwy::SortDescending{});
  } else {
    if (ascending) hwy::HWY_NAMESPACE::VQPartialSortStatic(values, n, k, hwy::SortAscending{});
    else hwy::HWY_NAMESPACE::VQPartialSortStatic(values, n, k, hwy::SortDescending{});
  }
}

template <typename T>
HYLE_NOINLINE void std_sort(T* values, std::size_t n, bool ascending) noexcept {
  auto lt = [](const T& a, const T& b) { return key_of(a) < key_of(b); };
  auto gt = [](const T& a, const T& b) { return key_of(a) > key_of(b); };
  if (ascending) std::sort(values, values + n, lt);
  else std::sort(values, values + n, gt);
}

template <typename T>
HYLE_NOINLINE void std_partial_sort(T* values, std::size_t n, std::size_t k,
                                    bool ascending) noexcept {
  auto lt = [](const T& a, const T& b) { return key_of(a) < key_of(b); };
  auto gt = [](const T& a, const T& b) { return key_of(a) > key_of(b); };
  if (ascending) std::partial_sort(values, values + k, values + n, lt);
  else std::partial_sort(values, values + k, values + n, gt);
}

template <typename T>
void run_full(const char* type, std::size_t n, bool few_unique) {
  const char* order = std::getenv("HYLE_BENCH_ORDER");
  const bool ascending = order == nullptr || std::strcmp(order, "desc") != 0;
  const auto source = make_input<T>(n, few_unique);
  std::vector<T> work(n);
  nb::Bench bench;
  const std::string title = std::string(type) + " sort n=" + std::to_string(n) +
                            (few_unique ? " few " : " random ") +
                            (ascending ? "asc" : "desc");
  bench.title(title).unit("key").relative(true).batch(n);
  bench.run("copy floor", [&] {
    std::copy(source.begin(), source.end(), work.begin());
    nb::doNotOptimizeAway(work.data());
  });
  bench.run("std direct", [&] {
    std::copy(source.begin(), source.end(), work.begin());
    std_sort(work.data(), n, ascending);
    nb::doNotOptimizeAway(work.data());
  });
  bench.run("Highway direct", [&] {
    std::copy(source.begin(), source.end(), work.begin());
    highway_sort(work.data(), n, ascending);
    nb::doNotOptimizeAway(work.data());
  });
  bench.run("HYLE public", [&] {
    std::copy(source.begin(), source.end(), work.begin());
    hyle_sort(work.data(), n, ascending);
    nb::doNotOptimizeAway(work.data());
  });
}

template <typename T>
void run_partial(const char* type, std::size_t n, std::size_t k, bool few_unique) {
  const char* order = std::getenv("HYLE_BENCH_ORDER");
  const bool ascending = order == nullptr || std::strcmp(order, "desc") != 0;
  const auto source = make_input<T>(n, few_unique);
  std::vector<T> work(n);
  std::vector<T> check = source;
  std::vector<T> reference = source;
  candidate_partial_sort(check.data(), n, k, ascending);
  std_partial_sort(reference.data(), n, k, ascending);
  for (std::size_t i = 0; i < k; ++i) {
    if (key_of(check[i]) != key_of(reference[i])) std::abort();
  }
  nb::Bench bench;
  const std::string title = std::string(type) + " partial n=" + std::to_string(n) +
                            " k=" + std::to_string(k) +
                            (few_unique ? " few " : " random ") +
                            (ascending ? "asc" : "desc");
  bench.title(title).unit("key").relative(true).batch(n);
  bench.run("copy floor", [&] {
    std::copy(source.begin(), source.end(), work.begin());
    nb::doNotOptimizeAway(work.data());
  });
  bench.run("std direct", [&] {
    std::copy(source.begin(), source.end(), work.begin());
    std_partial_sort(work.data(), n, k, ascending);
    nb::doNotOptimizeAway(work.data());
  });
  bench.run("HYLE candidate direct", [&] {
    std::copy(source.begin(), source.end(), work.begin());
    candidate_partial_sort(work.data(), n, k, ascending);
    nb::doNotOptimizeAway(work.data());
  });
  bench.run("Highway direct", [&] {
    std::copy(source.begin(), source.end(), work.begin());
    highway_partial_sort(work.data(), n, k, ascending);
    nb::doNotOptimizeAway(work.data());
  });
  bench.run("HYLE public", [&] {
    std::copy(source.begin(), source.end(), work.begin());
    hyle_partial_sort(work.data(), n, k, ascending);
    nb::doNotOptimizeAway(work.data());
  });
}

// select / nth_element: only for plain numeric-key types (HYLE select has no KeyValue overload).
// Correctness gate: the k-th key must match std::nth_element's, and all keys before k must not
// exceed it (partition invariant). Rivals: std::nth_element + Highway VQSelectStatic.
template <typename T>
void run_select(const char* type, std::size_t n, std::size_t k, bool few_unique) {
  const char* order = std::getenv("HYLE_BENCH_ORDER");
  const bool ascending = order == nullptr || std::strcmp(order, "desc") != 0;
  const auto source = make_input<T>(n, few_unique);
  std::vector<T> work(n), check = source, reference = source;
  hyle_select(check.data(), n, k, ascending);
  std_nth(reference.data(), n, k, ascending);
  if (key_of(check[k]) != key_of(reference[k])) std::abort();  // k-th order statistic matches
  nb::Bench bench;
  const std::string title = std::string(type) + " select n=" + std::to_string(n) +
                            " k=" + std::to_string(k) + (few_unique ? " few " : " random ") +
                            (ascending ? "asc" : "desc");
  bench.title(title).unit("key").relative(true).batch(n);
  bench.run("copy floor", [&] {
    std::copy(source.begin(), source.end(), work.begin());
    nb::doNotOptimizeAway(work.data());
  });
  bench.run("std::nth_element direct", [&] {
    std::copy(source.begin(), source.end(), work.begin());
    std_nth(work.data(), n, k, ascending);
    nb::doNotOptimizeAway(work.data());
  });
  bench.run("Highway VQSelect direct", [&] {
    std::copy(source.begin(), source.end(), work.begin());
    highway_select(work.data(), n, k, ascending);
    nb::doNotOptimizeAway(work.data());
  });
  bench.run("HYLE public", [&] {
    std::copy(source.begin(), source.end(), work.begin());
    hyle_select(work.data(), n, k, ascending);
    nb::doNotOptimizeAway(work.data());
  });
}

bool selected_text(const char* variable, const char* value) {
  const char* text = std::getenv(variable);
  return text == nullptr || std::strcmp(text, value) == 0;
}

bool selected_number(const char* variable, std::size_t value) {
  const char* text = std::getenv(variable);
  return text == nullptr || static_cast<std::size_t>(std::strtoull(text, nullptr, 10)) == value;
}

template <typename T>
void run_type(const char* name) {
  if (!selected_text("HYLE_BENCH_TYPE", name)) return;
  for (bool few_unique : {false, true}) {
    if (!selected_text("HYLE_BENCH_DIST", few_unique ? "few" : "random")) continue;
    for (std::size_t n : {std::size_t{16}, std::size_t{64}, std::size_t{256},
                          std::size_t{1024}, std::size_t{4096}, std::size_t{65536},
                          std::size_t{1048576}}) {
      if (!selected_number("HYLE_BENCH_N", n)) continue;
      if (std::getenv("HYLE_BENCH_K") == nullptr) run_full<T>(name, n, few_unique);
      std::vector<std::size_t> ks = {1, 2, 4, 8, 16, n / 10, n / 2};
      std::sort(ks.begin(), ks.end());
      ks.erase(std::unique(ks.begin(), ks.end()), ks.end());
      for (std::size_t k : ks) {
        if (k <= n && selected_number("HYLE_BENCH_K", k))
          run_partial<T>(name, n, k, few_unique);
        // select/nth_element: plain-key types only (no KeyValue overload); k is an index so k<n.
        if constexpr (!std::is_same_v<T, KeyValue32> && !std::is_same_v<T, KeyValue64>)
          if (k < n && selected_number("HYLE_BENCH_K", k))
            run_select<T>(name, n, k, few_unique);
      }
    }
  }
}

}  // namespace

int main() {
  static_assert(sizeof(KeyValue32) == sizeof(hwy::K32V32));
  static_assert(sizeof(KeyValue64) == sizeof(hwy::K64V64));
  std::printf("HYLE sort tier=%u\n", static_cast<unsigned>(hyle::simd::active_sort_variant()));
  run_type<std::uint32_t>("u32");
  run_type<std::uint64_t>("u64");
  run_type<KeyValue32>("kv32");
  run_type<KeyValue64>("kv64");
  return 0;
}

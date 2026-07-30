#include <nanobench.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include "hyle/c/util.h"
#include "hyle/container/string_ref.h"
#include "hyle/platform/cpu.h"
#include "hyle/platform/thread.h"
#include "hyle/util/random.h"
#include "hyle/util/registry.h"

#if defined(__APPLE__)
#  include <pthread.h>
#endif

#if defined(HYLE_BENCH_ABSL)
#  include <absl/container/flat_hash_map.h>
#  include <absl/hash/hash.h>
#  include <absl/strings/string_view.h>
#endif

namespace {

namespace nb = ankerl::nanobench;
using Creator = void* (*)();
using ResultCreator = hyle::Result<void*> (*)();

std::vector<unsigned> g_parallel_cpus;

[[maybe_unused]] void discover_benchmark_cpus() {
  auto count = hyle::platform::available_cpu_ids(
      hyle::platform::CpuOrder::kPerformanceFirst, nullptr, 0);
  if (!count || *count == 0) std::abort();
  g_parallel_cpus.resize(*count);
  auto filled = hyle::platform::available_cpu_ids(
      hyle::platform::CpuOrder::kPerformanceFirst, g_parallel_cpus.data(),
      g_parallel_cpus.size());
  if (!filled || *filled == 0) std::abort();
  if (*filled < g_parallel_cpus.size()) g_parallel_cpus.resize(*filled);
}

void configure_main_benchmark_thread() {
#if defined(__APPLE__)
  const char* requested = std::getenv("HYLE_BENCH_APPLE_QOS");
  qos_class_t qos = QOS_CLASS_USER_INTERACTIVE;
  const char* name = "user-interactive";
  if (requested != nullptr && std::strcmp(requested, "utility") == 0) {
    qos = QOS_CLASS_UTILITY;
    name = "utility";
  } else if (requested != nullptr &&
             std::strcmp(requested, "interactive") != 0) {
    std::fprintf(stderr,
                 "hyle_util_bench: HYLE_BENCH_APPLE_QOS must be "
                 "interactive or utility (got %s)\n",
                 requested);
    std::abort();
  }
  const int result = pthread_set_qos_class_self_np(qos, 0);
  std::fprintf(stderr,
               "# HYLE_UTIL_BENCH apple_main_qos=%s qos_ok=%d "
               "affinity=unsupported\n",
               name, result == 0 ? 1 : 0);
  if (result != 0) std::abort();
#else
  discover_benchmark_cpus();
  // Prefer the second strongest available CPU so the single-thread rows avoid
  // the boot/interrupt-heavy first logical CPU when the OS exposes a choice.
  unsigned cpu = g_parallel_cpus[g_parallel_cpus.size() > 1 ? 1 : 0];
#if defined(_WIN32)
  char* requested_storage = nullptr;
  std::size_t requested_size = 0;
  if (_dupenv_s(&requested_storage, &requested_size, "HYLE_BENCH_CPU") != 0) {
    std::abort();
  }
  const char* requested = requested_storage;
#else
  const char* requested = std::getenv("HYLE_BENCH_CPU");
#endif
  if (requested != nullptr) {
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(requested, &end, 10);
    if (end == requested || *end != '\0' ||
        parsed > (std::numeric_limits<unsigned>::max)()) {
      std::fprintf(stderr,
                   "hyle_util_bench: HYLE_BENCH_CPU is invalid (got %s)\n",
                   requested);
      std::abort();
    }
    cpu = static_cast<unsigned>(parsed);
  }
#if defined(_WIN32)
  std::free(requested_storage);
#endif
  const bool pin_ok =
      hyle::platform::this_thread::set_affinity(cpu).ok();
  const bool priority_ok = hyle::platform::this_thread::set_priority(
                               hyle::platform::ThreadPriority::kHigh)
                               .ok();
  std::fprintf(stderr,
               "# HYLE_UTIL_BENCH cpu=%u pin_ok=%d priority_ok=%d "
               "parallel_cpus=%zu\n",
               cpu, pin_ok ? 1 : 0, priority_ok ? 1 : 0,
               g_parallel_cpus.size());
  if (!pin_ok || !priority_ok) std::abort();
#endif
}

void configure_worker_benchmark_thread(std::size_t worker) {
#if defined(__APPLE__)
  (void)worker;
  const char* requested = std::getenv("HYLE_BENCH_APPLE_QOS");
  const qos_class_t qos =
      requested != nullptr && std::strcmp(requested, "utility") == 0
          ? QOS_CLASS_UTILITY
          : QOS_CLASS_USER_INTERACTIVE;
  if (pthread_set_qos_class_self_np(qos, 0) != 0) std::abort();
#else
  if (g_parallel_cpus.empty()) std::abort();
  const unsigned cpu = g_parallel_cpus[worker % g_parallel_cpus.size()];
  if (!hyle::platform::this_thread::set_affinity(cpu).ok()) std::abort();
  if (!hyle::platform::this_thread::set_priority(
           hyle::platform::ThreadPriority::kHigh)
           .ok()) {
    std::abort();
  }
#endif
}

void* create_token() {
  static std::uint64_t token = 7;
  return &token;
}

hyle::Result<void*> create_result_token() { return create_token(); }

hyle_status create_c_token(void** out_object, void*, const void*) {
  hyle_status status = HYLE_STATUS_INIT;
  *out_object = create_token();
  return status;
}

struct TransparentHash {
  using is_transparent = void;
  std::size_t operator()(std::string_view value) const noexcept {
    return std::hash<std::string_view>{}(value);
  }
  std::size_t operator()(const std::string& value) const noexcept {
    return (*this)(std::string_view(value));
  }
};

struct PeerEntry {
  Creator creator;
  int priority;
  std::string help;
};

struct ResultPeerEntry {
  ResultCreator creator;
  int priority;
  std::string help;
};

using StdMap = std::unordered_map<std::string, PeerEntry, TransparentHash,
                                  std::equal_to<>>;
using ResultStdMap =
    std::unordered_map<std::string, ResultPeerEntry, TransparentHash,
                       std::equal_to<>>;

struct RegistryFixture {
  static constexpr std::size_t kEntries = 256;
  std::array<std::string, kEntries> keys;
  std::array<hyle::container::string_ref, kEntries> hyle_keys;
  std::array<hyle_container_string_view, kEntries> c_keys;
  std::array<std::string_view, kEntries> std_keys;
  hyle::util::Registry<void*> mutable_hyle;
  hyle::util::Registry<void*> frozen_hyle;
  hyle::util::Registry<hyle::Result<void*>> fallible_hyle;
  StdMap mutable_std;
  StdMap frozen_std;
  ResultStdMap fallible_std;
#if defined(HYLE_BENCH_ABSL)
  absl::flat_hash_map<std::string, PeerEntry> frozen_absl;
  absl::flat_hash_map<std::string, ResultPeerEntry> fallible_absl;
#endif
  hyle_util_registry* frozen_c = nullptr;
  mutable std::mutex std_mutex;

  RegistryFixture() {
    check(mutable_hyle.reserve(kEntries));
    check(frozen_hyle.reserve(kEntries));
    check(fallible_hyle.reserve(kEntries));
    mutable_std.reserve(kEntries);
    frozen_std.reserve(kEntries);
    fallible_std.reserve(kEntries);
#if defined(HYLE_BENCH_ABSL)
    frozen_absl.reserve(kEntries);
    fallible_absl.reserve(kEntries);
#endif
    check(hyle_util_registry_create(&frozen_c, hyle_system_allocator()));
    check(hyle_util_registry_reserve(frozen_c, kEntries));
    for (std::size_t i = 0; i < kEntries; ++i) {
      char text[32];
      std::snprintf(text, sizeof(text), "backend-%03zu", i);
      keys[i] = text;
      hyle_keys[i] = hyle::container::string_ref(keys[i]);
      c_keys[i] = hyle_container_view(
          keys[i].data(), static_cast<std::uint32_t>(keys[i].size()));
      std_keys[i] = keys[i];
      check(mutable_hyle.register_creator(
          hyle_keys[i], &create_token, hyle::util::RegistryPriority::kDefault,
          "benchmark backend"));
      check(frozen_hyle.register_creator(
          hyle_keys[i], &create_token, hyle::util::RegistryPriority::kDefault,
          "benchmark backend"));
      check(fallible_hyle.register_creator(
          hyle_keys[i], &create_result_token,
          hyle::util::RegistryPriority::kDefault, "benchmark backend"));
      mutable_std.emplace(
          keys[i], PeerEntry{&create_token, 2, "benchmark backend"});
      frozen_std.emplace(
          keys[i], PeerEntry{&create_token, 2, "benchmark backend"});
      fallible_std.emplace(
          keys[i],
          ResultPeerEntry{&create_result_token, 2, "benchmark backend"});
#if defined(HYLE_BENCH_ABSL)
      frozen_absl.emplace(
          keys[i], PeerEntry{&create_token, 2, "benchmark backend"});
      fallible_absl.emplace(
          keys[i],
          ResultPeerEntry{&create_result_token, 2, "benchmark backend"});
#endif
      check(hyle_util_registry_register(
          frozen_c, c_keys[i], &create_c_token, nullptr,
          HYLE_UTIL_REGISTRY_DEFAULT,
          hyle_container_cstr("benchmark backend")));
    }
    (void)frozen_hyle.freeze();
    (void)fallible_hyle.freeze();
    check(hyle_util_registry_freeze(frozen_c));
  }

  ~RegistryFixture() { hyle_util_registry_destroy(frozen_c); }

  static void check(hyle::Status status) {
    if (!status.ok()) std::abort();
  }

  static void check(hyle_status status) {
    if (!hyle_status_is_ok(status)) {
      hyle_status_free(&status);
      std::abort();
    }
  }
};

template <class Lookup>
std::uint64_t parallel_reads(Lookup lookup, std::size_t threads,
                             std::size_t iterations) {
  std::atomic<std::uint64_t> total{0};
  std::vector<std::thread> workers;
  workers.reserve(threads);
  for (std::size_t worker = 0; worker < threads; ++worker) {
    workers.emplace_back([&, worker] {
      configure_worker_benchmark_thread(worker);
      std::uint64_t local = 0;
      for (std::size_t i = 0; i < iterations; ++i) {
        local += lookup((i + worker * 17) & 255) != nullptr;
      }
      total.fetch_add(local, std::memory_order_relaxed);
    });
  }
  for (auto& worker : workers) worker.join();
  return total.load(std::memory_order_relaxed);
}

}  // namespace

int main() {
  configure_main_benchmark_thread();
  RegistryFixture fixture;
  const auto frozen_hyle_view = fixture.frozen_hyle.freeze();
  const auto fallible_hyle_view = fixture.fallible_hyle.freeze();
  std::size_t index = 0;

  {
    constexpr std::size_t kBatch = 1024;
    std::array<std::uint64_t, kBatch> output{};
    hyle::util::Rng native_rng{12345};
    hyle_util_rng c_rng{};
    if (!hyle_util_rng_seed(&c_rng, 12345)) std::abort();
    nb::Bench bench;
    bench.title("util-runtime C RNG adapter, 1024-value batch")
        .relative(true)
        .batch(kBatch)
        .minEpochIterations(20'000);
    bench.run("C++ Rng fill loop", [&] {
      for (auto& value : output) value = native_rng.next_u64();
      nb::doNotOptimizeAway(output.data());
    });
    bench.run("C ABI rng_fill_u64", [&] {
      if (!hyle_util_rng_fill_u64(&c_rng, output.data(), output.size())) {
        std::abort();
      }
      nb::doNotOptimizeAway(output.data());
    });
  }

  {
    nb::Bench bench;
    bench.title("util-runtime registry-key hash, 11 bytes")
        .relative(true)
        .batch(1)
        .minEpochIterations(2'000'000);
    bench.run("std::hash<string_view>", [&] {
      nb::doNotOptimizeAway(
          std::hash<std::string_view>{}(fixture.std_keys[index++ & 255]));
    });
#if defined(HYLE_BENCH_ABSL)
    index = 0;
    bench.run("absl::Hash<string_view>", [&] {
      nb::doNotOptimizeAway(absl::Hash<absl::string_view>{}(
          absl::string_view(fixture.std_keys[index++ & 255])));
    });
#endif
    index = 0;
    bench.run("HYLE hash_bytes", [&] {
      const auto key = fixture.hyle_keys[index++ & 255];
      nb::doNotOptimizeAway(hyle::container::hash_bytes(key.data, key.size));
    });
  }

  {
    nb::Bench bench;
    bench.title("util-runtime mutable named factory lookup+call, N=256")
        .relative(true)
        .batch(1)
        .minEpochIterations(500'000);
    bench.run("std::mutex + std::unordered_map", [&] {
      const std::size_t at = index++ & 255;
      Creator creator;
      {
        std::lock_guard<std::mutex> lock(fixture.std_mutex);
        creator = fixture.mutable_std.find(fixture.std_keys[at])->second.creator;
      }
      nb::doNotOptimizeAway(creator());
    });
    index = 0;
    bench.run("HYLE Registry mutable", [&] {
      nb::doNotOptimizeAway(
          fixture.mutable_hyle.create(fixture.hyle_keys[index++ & 255]));
    });
  }

  {
    nb::Bench bench;
    bench.title("util-runtime frozen named factory lookup+call, N=256")
        .relative(true)
        .batch(1)
        .minEpochIterations(1'000'000);
    bench.run("std::unordered_map immutable", [&] {
      const auto found =
          fixture.frozen_std.find(fixture.std_keys[index++ & 255]);
      nb::doNotOptimizeAway(found->second.creator());
    });
#if defined(HYLE_BENCH_ABSL)
    index = 0;
    bench.run("absl::flat_hash_map immutable", [&] {
      const auto found = fixture.frozen_absl.find(
          absl::string_view(fixture.std_keys[index++ & 255]));
      nb::doNotOptimizeAway(found->second.creator());
    });
#endif
    index = 0;
    bench.run("HYLE Registry frozen", [&] {
      nb::doNotOptimizeAway(
          fixture.frozen_hyle.create(fixture.hyle_keys[index++ & 255]));
    });
    index = 0;
    bench.run("HYLE FrozenView", [&] {
      nb::doNotOptimizeAway(
          frozen_hyle_view.create(fixture.hyle_keys[index++ & 255]));
    });
    index = 0;
    bench.run("HYLE C Registry frozen", [&] {
      nb::doNotOptimizeAway(hyle_util_registry_create_or_null(
          fixture.frozen_c, fixture.c_keys[index++ & 255], nullptr));
    });
    index = 0;
    bench.run("HYLE C Registry frozen checked", [&] {
      void* result = nullptr;
      hyle_status status = hyle_util_registry_create_object(
          &result, fixture.frozen_c, fixture.c_keys[index++ & 255], nullptr);
      if (!hyle_status_is_ok(status)) std::abort();
      nb::doNotOptimizeAway(result);
    });
  }

  {
    nb::Bench bench;
    bench.title("util-runtime fallible named factory lookup+call, N=256")
        .relative(true)
        .batch(1)
        .minEpochIterations(1'000'000);
    bench.run("std::unordered_map + Result factory", [&] {
      const auto found =
          fixture.fallible_std.find(fixture.std_keys[index++ & 255]);
      auto result = found->second.creator();
      if (!result.has_value()) std::abort();
      nb::doNotOptimizeAway(*result);
    });
#if defined(HYLE_BENCH_ABSL)
    index = 0;
    bench.run("absl::flat_hash_map + Result factory", [&] {
      const auto found = fixture.fallible_absl.find(
          absl::string_view(fixture.std_keys[index++ & 255]));
      auto result = found->second.creator();
      if (!result.has_value()) std::abort();
      nb::doNotOptimizeAway(*result);
    });
#endif
    index = 0;
    bench.run("HYLE Result FrozenView", [&] {
      auto result =
          fallible_hyle_view.create(fixture.hyle_keys[index++ & 255]);
      if (!result.has_value()) std::abort();
      nb::doNotOptimizeAway(*result);
    });
  }

  constexpr std::size_t kThreads = 8;
  constexpr std::size_t kIterations = 1'000'000;
  {
    nb::Bench bench;
    bench.title("util-runtime frozen registry parallel reads, 8 threads")
        .relative(true)
        .batch(kThreads * kIterations)
        .epochs(7)
        .minEpochIterations(1);
    bench.run("std::unordered_map immutable", [&] {
      nb::doNotOptimizeAway(parallel_reads(
          [&](std::size_t at) {
            return fixture.frozen_std.find(fixture.std_keys[at])
                ->second.creator();
          },
          kThreads, kIterations));
    });
#if defined(HYLE_BENCH_ABSL)
    bench.run("absl::flat_hash_map immutable", [&] {
      nb::doNotOptimizeAway(parallel_reads(
          [&](std::size_t at) {
            return fixture.frozen_absl
                .find(absl::string_view(fixture.std_keys[at]))
                ->second.creator();
          },
          kThreads, kIterations));
    });
#endif
    bench.run("HYLE Registry frozen", [&] {
      nb::doNotOptimizeAway(parallel_reads(
          [&](std::size_t at) {
            return fixture.frozen_hyle.create(fixture.hyle_keys[at]);
          },
          kThreads, kIterations));
    });
    bench.run("HYLE FrozenView", [&] {
      nb::doNotOptimizeAway(parallel_reads(
          [&](std::size_t at) {
            return frozen_hyle_view.create(fixture.hyle_keys[at]);
          },
          kThreads, kIterations));
    });
    bench.run("HYLE C Registry frozen", [&] {
      nb::doNotOptimizeAway(parallel_reads(
          [&](std::size_t at) {
            return hyle_util_registry_create_or_null(
                fixture.frozen_c, fixture.c_keys[at], nullptr);
          },
          kThreads, kIterations));
    });
  }
}

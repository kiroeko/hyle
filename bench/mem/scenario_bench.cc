// HYLE mem user scenarios vs same-contract mature implementations.
// arena/bump allocators are "5-10x+ vs new/delete" -- never benchmarked until
// now. The honest SOTA references are in the STANDARD LIBRARY (no external
// clone):
//   * std::pmr::monotonic_buffer_resource   -- std's arena (bump, bulk release)
//   * std::pmr::unsynchronized_pool_resource -- std's pool (size-class
//   free-list)
// plus raw mi_malloc/mi_free (the general malloc = what `new/delete` resolve
// to, since HYLE's global backing is mimalloc) as the "5-10x" target baseline.
//
// Workload families. Rows with weaker/different allocation visibility are
// throughput references, and are not presented as same-contract SOTA:
//  A. BUMP: PooledRegion + StackAllocator vs pmr::monotonic vs raw mi.
//  B. SINGLE-THREAD pool churn: GrowablePool ONLY vs pmr::unsynchronized_pool
//  vs raw mi. SharedPool/SharedPoolCache belong to cross-thread sharing
//     scenarios.
//  C. 8-thread tight churn: hard pool vs cached pool vs pmr/tbb/mimalloc.
//  D. Shared-path burst and cross-thread handoff: the same full comparison.
//  E. Hard service admission and post-free visibility.
//  F. Closure shapes: request scratch, external workspace, mixed fixed-buffer
//     churn, quota decoration, and expensive-context preservation.
//  G. Request-level latency certification: p50/p95/p99 of whole requests, not
//     clock-instrumented individual allocations or nanobench epochs.
//  H. Request-scratch footprint certification: payload, backing reservation,
//     clear/release retention, and process RSS in isolated processes.
//  I. Fixed-buffer fragmentation certification: largest satisfiable object
//     after mixed fill/selective free, normalized to each pool's empty limit.

#include <mimalloc.h>
#include <nanobench.h>

#if defined(HYLE_BENCH_TBB)
#include <tbb/memory_pool.h>        // fixed-buffer, TLS-cached reference
#include <tbb/scalable_allocator.h> // growable MT throughput reference
#endif

#include <algorithm>
#include <array>
#include <barrier>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <memory_resource>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
// clang-format off: psapi.h requires the Windows base declarations first.
#include <windows.h>
#include <psapi.h>
// clang-format on
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <pthread.h>
#include <sys/resource.h>
#elif defined(__linux__)
#include <sys/resource.h>
#include <unistd.h>
#endif

#include "hyle/mem/allocator.h"
#include "hyle/mem/fixed_pool.h"
#include "hyle/mem/growable_pool.h"
#include "hyle/mem/object_pool.h"
#include "hyle/mem/pmr.h"
#include "hyle/mem/pool_cache.h"
#include "hyle/mem/quota_allocator.h"
#include "hyle/mem/region.h"
#include "hyle/mem/shared_pool.h"
#include "hyle/mem/stack_allocator.h"
#include "hyle/mem/stats.h"

using hyle::mem::Allocator;
using hyle::mem::FixedPool;
using hyle::mem::FixedRegion;
using hyle::mem::GrowablePool;
using hyle::mem::ObjectPool;
using hyle::mem::PmrResource;
using hyle::mem::PooledRegion;
using hyle::mem::PreservingObjectPool;
using hyle::mem::QuotaAllocator;
using hyle::mem::Region;
using hyle::mem::RegionOptions;
using hyle::mem::SharedPool;
using hyle::mem::SharedPoolCache;
using hyle::mem::StackAllocator;
using hyle::mem::StatsAllocator;
using hyle::mem::system_allocator;

namespace {
constexpr std::size_t kObj = 64;                  // one small same-size object
constexpr int kN = 8192;                          // allocations per round
constexpr std::size_t kRegion = kN * kObj + 4096; // bump region (+slack)
// oneTBB's own fixed-pool tests budget 1 MiB beyond payload for bootstrap
// metadata. Match HYLE payload capacity; this allowance is not extra
// user-visible capacity.
constexpr std::size_t kTbbFixedBootstrapBytes = 1u << 20;
namespace nb = ankerl::nanobench;

// Apple Silicon has heterogeneous performance/efficiency cores but no public
// CPU-affinity API. An unqualified SSH process may migrate between both core
// classes, preserving fixed-order bias in otherwise repeatable nanosecond rows.
// Default certification runs request the same USER_INTERACTIVE QoS as the SIMD
// harness. UTILITY is an explicit lower-priority scheduling audit; neither is
// a hard pin or proof of core class because Darwin exposes only a preference.
void configure_main_benchmark_thread() {
#if defined(__APPLE__)
  const char *requested = std::getenv("HYLE_BENCH_APPLE_QOS");
  qos_class_t qos = QOS_CLASS_USER_INTERACTIVE;
  const char *name = "user-interactive";
  if (requested != nullptr && std::strcmp(requested, "utility") == 0) {
    qos = QOS_CLASS_UTILITY;
    name = "utility";
  } else if (requested != nullptr &&
             std::strcmp(requested, "background") == 0) {
    qos = QOS_CLASS_BACKGROUND;
    name = "background";
  } else if (requested != nullptr &&
             std::strcmp(requested, "interactive") != 0) {
    std::fprintf(stderr,
                 "mem_scenarios_bench: HYLE_BENCH_APPLE_QOS must be "
                 "interactive, utility, or background (got %s)\n",
                 requested);
    std::abort();
  }
  const int rc = pthread_set_qos_class_self_np(qos, 0);
  std::printf(
      "# HYLE_MEM_BENCH apple_main_qos=%s qos_ok=%d affinity=unsupported\n",
      name, rc == 0 ? 1 : 0);
  if (rc != 0) {
    std::fprintf(stderr,
                 "mem_scenarios_bench: failed to set Apple QoS (%d)\n", rc);
    std::abort();
  }
#endif
}

// Section filter: HYLE_BENCH_SECTIONS="D" runs only section D, "AD" runs A+D,
// etc. Unset/empty runs everything. The full sweep is minutes-per-run (the
// bump/ST epochs dominate), so a bench-stat -n sweep over just the shared-path
// section D is orders of magnitude faster; this gate makes that possible
// without touching the rows.
bool want_section(char id) {
  const char *env = std::getenv("HYLE_BENCH_SECTIONS");
  if (env == nullptr || env[0] == '\0')
    return true; // no filter -> all
  for (const char *p = env; *p; ++p)
    if (*p == id)
      return true;
  return false;
}

int env_int(const char *name, int fallback) {
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0')
    return fallback;
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  return end != value && *end == '\0' && parsed > 0 ? static_cast<int>(parsed)
                                                    : fallback;
}

constexpr std::size_t scratch_size(int i) noexcept;
inline void *require_allocation(void *p) noexcept;

struct ProcessMemory {
  std::size_t current_rss = 0;
  std::size_t peak_rss = 0;
};

ProcessMemory process_memory() noexcept {
  ProcessMemory result;
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS counters{};
  counters.cb = sizeof(counters);
  if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)) !=
      0) {
    result.current_rss = static_cast<std::size_t>(counters.WorkingSetSize);
    result.peak_rss = static_cast<std::size_t>(counters.PeakWorkingSetSize);
  }
#elif defined(__APPLE__)
  mach_task_basic_info info{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS)
    result.current_rss = static_cast<std::size_t>(info.resident_size);
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) == 0)
    result.peak_rss = static_cast<std::size_t>(usage.ru_maxrss);
#elif defined(__linux__)
  long resident_pages = 0;
  if (FILE *statm = std::fopen("/proc/self/statm", "r")) {
    long total_pages = 0;
    const int fields =
        std::fscanf(statm, "%ld %ld", &total_pages, &resident_pages);
    std::fclose(statm);
    if (fields != 2)
      resident_pages = 0;
  }
  const long page_size = sysconf(_SC_PAGESIZE);
  rusage usage{};
  if (resident_pages > 0 && page_size > 0)
    result.current_rss = static_cast<std::size_t>(resident_pages) *
                         static_cast<std::size_t>(page_size);
  if (getrusage(RUSAGE_SELF, &usage) == 0)
    result.peak_rss = static_cast<std::size_t>(usage.ru_maxrss) * 1024;
#endif
  return result;
}

void *direct_mi_allocate(void *, std::size_t bytes,
                         std::size_t alignment) noexcept {
  return mi_malloc_aligned(bytes, alignment);
}

void direct_mi_deallocate(void *, void *p, std::size_t, std::size_t) noexcept {
  mi_free(p);
}

void *direct_mi_reallocate(void *, void *p, std::size_t, std::size_t new_bytes,
                           std::size_t alignment) noexcept {
  if (new_bytes == 0) {
    mi_free(p);
    return nullptr;
  }
  return mi_realloc_aligned(p, new_bytes, alignment);
}

Allocator direct_mi_allocator() noexcept {
  return Allocator(&direct_mi_allocate, &direct_mi_deallocate,
                   &direct_mi_reallocate, nullptr);
}

void touch_all_pages(void *allocation, std::size_t bytes) noexcept {
  auto *data = static_cast<volatile std::byte *>(allocation);
  constexpr std::size_t kPage = 4096;
  for (std::size_t offset = 0; offset < bytes; offset += kPage)
    data[offset] = std::byte{0x5a};
  data[bytes - 1] = std::byte{0xa5};
}

template <class Allocate>
std::size_t fill_scratch_request(std::size_t target_bytes,
                                 Allocate &&allocate) {
  constexpr std::size_t sizes[] = {24,  40,   72,   128,   192,
                                   320, 1024, 4096, 16384, 65536};
  std::size_t requested = 0;
  std::size_t allocation = 0;
  while (requested != target_bytes) {
    // A service request is dominated by ordinary parser/graph scratch, with
    // an occasional dedicated 2 MiB staging buffer (about 20% of payload).
    std::size_t bytes = allocation % 1024 == 1023
                            ? 2u * 1024u * 1024u
                            : sizes[allocation % std::size(sizes)];
    if (bytes > target_bytes - requested)
      bytes = target_bytes - requested;
    void *p = require_allocation(allocate(bytes));
    touch_all_pages(p, bytes);
    requested += bytes;
    ++allocation;
  }
  return requested;
}

void print_footprint_row(
    const char *variant, std::size_t requested, std::size_t reuse_requested,
    std::size_t region_max_block, std::size_t peak_reserved,
    std::size_t retained_after_clear, std::size_t reuse_reserved,
    std::size_t released, const ProcessMemory &baseline,
    const ProcessMemory &live, const ProcessMemory &cleared,
    const ProcessMemory &reuse, const ProcessMemory &released_memory) {
  std::printf(
      "HYLE_MEM_CERT footprint scenario=request_scratch variant=%s "
      "requested_bytes=%zu reuse_requested_bytes=%zu "
      "region_max_block_bytes=%zu "
      "peak_reserved_bytes=%zu retained_after_clear_bytes=%zu "
      "reuse_reserved_bytes=%zu released_bytes=%zu "
      "baseline_rss_bytes=%zu live_rss_bytes=%zu clear_rss_bytes=%zu "
      "reuse_rss_bytes=%zu release_rss_bytes=%zu process_peak_rss_bytes=%zu\n",
      variant, requested, reuse_requested, region_max_block, peak_reserved,
      retained_after_clear, reuse_reserved, released, baseline.current_rss,
      live.current_rss, cleared.current_rss, reuse.current_rss,
      released_memory.current_rss, released_memory.peak_rss);
}

template <class Allocate, class Free>
std::size_t largest_satisfiable(std::size_t upper, Allocate &&allocate,
                                Free &&free) {
  std::size_t low = 0;
  std::size_t high = upper + 1;
  while (low + 1 < high) {
    const std::size_t middle = low + (high - low) / 2;
    if (void *p = allocate(middle); p != nullptr) {
      free(p);
      low = middle;
    } else {
      high = middle;
    }
  }
  return low;
}

template <class Allocate, class Free>
void run_fragmentation_scenario(const char *variant,
                                std::size_t payload_budget,
                                std::size_t backing_storage,
                                Allocate &&allocate, Free &&free) {
  constexpr std::size_t sizes[] = {64,    256,   1024, 4096,  16384,
                                   65536, 128,   2048, 8192,  32768,
                                   512,   24576, 96,   12288, 49152};
  struct Live {
    void *pointer;
    std::size_t requested;
  };

  const std::size_t empty_largest =
      largest_satisfiable(backing_storage, allocate, free);
  if (empty_largest == 0) {
    std::fputs("mem_scenarios_bench: fixed pool has no usable capacity\n",
               stderr);
    std::abort();
  }
  const std::size_t fill_target = empty_largest * 3 / 4;
  std::vector<Live> allocations;
  allocations.reserve(fill_target / 4096);
  std::size_t fill_requested = 0;
  for (std::size_t index = 0; fill_requested < fill_target; ++index) {
    std::size_t bytes = sizes[index % std::size(sizes)];
    if (bytes > fill_target - fill_requested)
      bytes = fill_target - fill_requested;
    void *p = allocate(bytes);
    if (p == nullptr) {
      std::fputs("mem_scenarios_bench: mixed fill failed before 75% of empty "
                 "user capacity\n",
                 stderr);
      std::abort();
    }
    touch_all_pages(p, bytes);
    allocations.push_back({p, bytes});
    fill_requested += bytes;
  }

  std::size_t freed_requested = 0;
  std::size_t live_requested = fill_requested;
  for (std::size_t index = 1; index < allocations.size(); index += 3) {
    free(allocations[index].pointer);
    freed_requested += allocations[index].requested;
    live_requested -= allocations[index].requested;
    allocations[index].pointer = nullptr;
  }
  const std::size_t fragmented_largest =
      largest_satisfiable(backing_storage, allocate, free);
  for (const Live &allocation : allocations) {
    if (allocation.pointer != nullptr)
      free(allocation.pointer);
  }
  const std::size_t recovered_largest =
      largest_satisfiable(backing_storage, allocate, free);

  std::printf(
      "HYLE_MEM_CERT fragmentation scenario=fixed_pool_mixed_holes "
      "variant=%s payload_budget_bytes=%zu backing_storage_bytes=%zu "
      "allocations=%zu empty_largest_bytes=%zu fill_requested_bytes=%zu "
      "live_requested_bytes=%zu freed_requested_bytes=%zu "
      "fragmented_largest_bytes=%zu recovered_largest_bytes=%zu\n",
      variant, payload_budget, backing_storage, allocations.size(),
      empty_largest, fill_requested, live_requested, freed_requested,
      fragmented_largest, recovered_largest);
}

constexpr std::size_t scratch_size(int i) noexcept {
  constexpr std::size_t sizes[] = {24, 40, 72, 128, 192, 320};
  return sizes[static_cast<unsigned>(i) % std::size(sizes)];
}

struct RequestLatency {
  std::uint64_t p50_ns;
  std::uint64_t p95_ns;
  std::uint64_t p99_ns;
};

RequestLatency summarize_latency(std::vector<std::uint64_t> samples) {
  std::sort(samples.begin(), samples.end());
  auto nearest_rank = [&](std::size_t percentile) {
    const std::size_t rank =
        (percentile * samples.size() + 99) / 100; // ceil(p*N)
    return samples[rank == 0 ? 0 : rank - 1];
  };
  return {nearest_rank(50), nearest_rank(95), nearest_rank(99)};
}

void print_certificate_metadata() {
#if defined(__clang__) && defined(_MSC_VER)
  std::printf("HYLE_MEM_CERT meta format=1 compiler=clangcl-%d.%d.%d-msvc%d "
              "platform=windows\n",
              __clang_major__, __clang_minor__, __clang_patchlevel__,
              _MSC_VER);
#elif defined(_MSC_VER)
  std::printf("HYLE_MEM_CERT meta format=1 compiler=msvc-%d platform=windows\n",
              _MSC_VER);
#elif defined(__apple_build_version__)
  std::printf("HYLE_MEM_CERT meta format=1 "
              "compiler=appleclang-%d.%d.%d-%d platform=macos\n",
              __clang_major__, __clang_minor__, __clang_patchlevel__,
              __apple_build_version__);
#elif defined(__clang__)
  std::printf("HYLE_MEM_CERT meta format=1 compiler=clang-%d.%d.%d "
              "platform=linux\n",
              __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__GNUC__)
  std::printf("HYLE_MEM_CERT meta format=1 compiler=gcc-%d.%d.%d "
              "platform=linux\n",
              __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
  std::printf("HYLE_MEM_CERT meta format=1 compiler=unknown platform=unknown\n");
#endif
}

template <class First, class Second>
void report_request_latency_pair(const char *scenario, const char *first_name,
                                 const char *second_name, int operations,
                                 int samples, First &&first, Second &&second) {
  using Clock = std::chrono::steady_clock;
  auto measure = [](auto &fn) {
    const auto begin = Clock::now();
    fn();
    const auto end = Clock::now();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
            .count());
  };

  for (int i = 0; i != 64; ++i) {
    first();
    second();
  }
  std::vector<std::uint64_t> first_samples;
  std::vector<std::uint64_t> second_samples;
  first_samples.reserve(static_cast<std::size_t>(samples));
  second_samples.reserve(static_cast<std::size_t>(samples));
  for (int i = 0; i != samples; ++i) {
    if ((i & 1) == 0) {
      first_samples.push_back(measure(first));
      second_samples.push_back(measure(second));
    } else {
      second_samples.push_back(measure(second));
      first_samples.push_back(measure(first));
    }
  }
  const RequestLatency first_result = summarize_latency(std::move(first_samples));
  const RequestLatency second_result =
      summarize_latency(std::move(second_samples));
  auto print = [&](const char *variant, const RequestLatency &result) {
    std::printf("HYLE_MEM_CERT latency scenario=%s variant=%s samples=%d "
                "operations=%d p50_ns=%llu p95_ns=%llu p99_ns=%llu\n",
                scenario, variant, samples, operations,
                static_cast<unsigned long long>(result.p50_ns),
                static_cast<unsigned long long>(result.p95_ns),
                static_cast<unsigned long long>(result.p99_ns));
  };
  print(first_name, first_result);
  print(second_name, second_result);
}

inline void *require_allocation(void *p) noexcept {
  if (p == nullptr) {
    std::fputs("mem_scenarios_bench: unexpected allocation failure\n", stderr);
    std::abort();
  }
  return p;
}

struct QuotaResult {
  std::size_t attempted;
  std::size_t initial_success;
  std::size_t recovered_before_publish;
  std::size_t recovered_after_publish;
  double initial_ns_per_attempt;
};

template <class Pool>
QuotaResult run_quota_scenario(Pool &pool, int threads, std::size_t capacity,
                               int request_pct) {
  const std::size_t requested =
      (capacity * static_cast<std::size_t>(request_pct) + 99) / 100;
  const std::size_t per_thread = (requested + threads - 1) / threads;
  const std::size_t attempted = per_thread * threads;
  std::vector<std::vector<void *>> held(
      threads, std::vector<void *>(per_thread, nullptr));
  std::vector<std::size_t> successes(threads, 0);
  std::barrier start(threads + 1);
  std::barrier allocated(threads + 1);
  std::barrier release(threads + 1);
  std::barrier freed(threads + 1);
  std::barrier publish(threads + 1);
  std::barrier published(threads + 1);
  std::vector<std::thread> workers;
  workers.reserve(threads);
  for (int t = 0; t < threads; ++t) {
    workers.emplace_back([&, t] {
      start.arrive_and_wait();
      for (std::size_t i = 0; i < per_thread; ++i) {
        held[t][i] = pool.allocate();
        if (held[t][i] != nullptr)
          ++successes[t];
      }
      allocated.arrive_and_wait();
      release.arrive_and_wait();
      for (void *p : held[t])
        pool.deallocate(p);
      freed.arrive_and_wait();
      publish.arrive_and_wait();
      if constexpr (requires { pool.publish_thread_cache(); })
        pool.publish_thread_cache();
      published.arrive_and_wait();
    });
  }

  const auto begin = std::chrono::steady_clock::now();
  start.arrive_and_wait();
  allocated.arrive_and_wait();
  const auto end = std::chrono::steady_clock::now();
  std::size_t initial_success = 0;
  for (std::size_t n : successes)
    initial_success += n;

  release.arrive_and_wait();
  freed.arrive_and_wait();
  auto drain = [&] {
    std::vector<void *> blocks;
    blocks.reserve(capacity);
    while (void *p = pool.allocate())
      blocks.push_back(p);
    return blocks;
  };
  std::vector<void *> before = drain();
  for (void *p : before)
    pool.deallocate(p);
  if constexpr (requires { pool.publish_thread_cache(); })
    pool.publish_thread_cache();

  publish.arrive_and_wait();
  published.arrive_and_wait();
  std::vector<void *> after = drain();
  for (void *p : after)
    pool.deallocate(p);
  if constexpr (requires { pool.publish_thread_cache(); })
    pool.publish_thread_cache();
  for (auto &worker : workers)
    worker.join();

  const double ns =
      std::chrono::duration<double, std::nano>(end - begin).count();
  return QuotaResult{attempted, initial_success, before.size(), after.size(),
                     ns / static_cast<double>(attempted)};
}

QuotaResult run_cached_quota_scenario(SharedPool &pool, int threads,
                                      std::size_t capacity,
                                      int request_pct) {
  const std::size_t requested =
      (capacity * static_cast<std::size_t>(request_pct) + 99) / 100;
  const std::size_t per_thread = (requested + threads - 1) / threads;
  const std::size_t attempted = per_thread * threads;
  std::vector<std::vector<void *>> held(
      threads, std::vector<void *>(per_thread, nullptr));
  std::vector<std::size_t> successes(threads, 0);
  std::barrier start(threads + 1);
  std::barrier allocated(threads + 1);
  std::barrier release(threads + 1);
  std::barrier freed(threads + 1);
  std::barrier publish(threads + 1);
  std::barrier published(threads + 1);
  std::vector<std::thread> workers;
  workers.reserve(threads);
  for (int t = 0; t < threads; ++t) {
    workers.emplace_back([&, t] {
      std::array<std::uint32_t, 128> storage{};
      SharedPoolCache cache(pool, storage, 32);
      start.arrive_and_wait();
      for (std::size_t i = 0; i < per_thread; ++i) {
        held[t][i] = cache.allocate();
        if (held[t][i] != nullptr)
          ++successes[t];
      }
      allocated.arrive_and_wait();
      release.arrive_and_wait();
      for (void *p : held[t])
        cache.deallocate(p);
      freed.arrive_and_wait();
      publish.arrive_and_wait();
      cache.publish();
      published.arrive_and_wait();
    });
  }

  const auto begin = std::chrono::steady_clock::now();
  start.arrive_and_wait();
  allocated.arrive_and_wait();
  const auto end = std::chrono::steady_clock::now();
  std::size_t initial_success = 0;
  for (std::size_t n : successes)
    initial_success += n;

  release.arrive_and_wait();
  freed.arrive_and_wait();
  auto drain = [&] {
    std::vector<void *> blocks;
    blocks.reserve(capacity);
    while (void *p = pool.allocate())
      blocks.push_back(p);
    return blocks;
  };
  std::vector<void *> before = drain();
  for (void *p : before)
    pool.deallocate(p);

  publish.arrive_and_wait();
  published.arrive_and_wait();
  std::vector<void *> after = drain();
  for (void *p : after)
    pool.deallocate(p);
  for (auto &worker : workers)
    worker.join();

  const double ns =
      std::chrono::duration<double, std::nano>(end - begin).count();
  return QuotaResult{attempted, initial_success, before.size(), after.size(),
                     ns / static_cast<double>(attempted)};
}
} // namespace

int main() {
  configure_main_benchmark_thread();
  Allocator sys = system_allocator(); // mimalloc backing

  // ---- A. BUMP: alloc N small, no per-object free, bulk release ----
  if (want_section('A')) {
    nb::Bench b;
    b.title("bump: 8192x64B alloc + bulk release")
        .relative(true)
        .batch(kN)
        .minEpochIterations(50);

    // HYLE PooledRegion (borrows ONE pooled block; clear() reuses it per
    // round).
    SharedPool pool(sys, kRegion, 1); // one hard-pool block of kRegion bytes
    PooledRegion arena(&pool);
    b.run("HYLE PooledRegion (bump, pooled block)", [&] {
      arena.clear();
      for (int i = 0; i < kN; ++i) {
        void *p = require_allocation(arena.allocate(kObj));
        nb::doNotOptimizeAway(std::as_const(p));
      }
    });

    // HYLE StackAllocator (owns its slab; reset() per round).
    StackAllocator stack(sys, kRegion);
    b.run("HYLE StackAllocator (bump, own slab)", [&] {
      stack.reset();
      for (int i = 0; i < kN; ++i) {
        void *p = require_allocation(stack.allocate(kObj));
        nb::doNotOptimizeAway(std::as_const(p));
      }
    });

    // std::pmr has no portable API that rebinds an initial buffer after
    // release() (notably, MSVC's release does not rewind it). Reconstruct the
    // tiny resource shell for each request so every epoch really uses the same
    // external buffer.
    std::vector<std::byte> buf(kRegion);
    b.run("std::pmr::monotonic_buffer_resource (external, reconstruct)", [&] {
      std::pmr::monotonic_buffer_resource mbr(buf.data(), buf.size(),
                                              std::pmr::null_memory_resource());
      for (int i = 0; i < kN; ++i) {
        void *p = mbr.allocate(kObj, alignof(std::max_align_t));
        nb::doNotOptimizeAway(std::as_const(p));
      }
    });

    // raw mi_malloc: general malloc has NO bulk free -> alloc-all then
    // free-all.
    b.run("raw mi_malloc/free (general, no bulk)", [&] {
      std::vector<void *> ps(kN);
      for (int i = 0; i < kN; ++i)
        ps[i] = require_allocation(mi_malloc(kObj));
      for (int i = 0; i < kN; ++i)
        mi_free(ps[i]);
      nb::doNotOptimizeAway(ps.data());
    });
  }

  // ---- A2. StackAllocator mark/rewind: nested-scope LIFO release ---- (gated
  // 'A') mark()/rewind() is the StackAllocator capability reset() can't
  // express: save a checkpoint, allocate a temporary batch, roll back to the
  // checkpoint (freeing exactly that batch) while KEEPING everything before it.
  // The scratch/frame pattern (e.g. a taskgraph pass that allocates per-node
  // temporaries and rewinds between nodes). Each round: alloc a base batch
  // once, then inner loop {mark; alloc temps; rewind}. No std rival --
  // std::pmr::monotonic has no partial rewind (only whole-buffer release) -- so
  // reset()-per-round is the baseline the mark/rewind cost is read against
  // (both are pointer moves; this proves rewind is as cheap as a fresh bump,
  // not a free-list walk).
  if (want_section('A')) {
    constexpr int kBase = 256;   // persistent allocations kept across rewinds
    constexpr int kInner = 64;   // temporaries per rewind cycle
    constexpr int kCycles = 128; // rewind cycles per round
    StackAllocator stack(sys, kRegion);
    nb::Bench b;
    b.title("StackAllocator mark/rewind: 128 cycles x 64 temps")
        .relative(true)
        .batch(static_cast<std::uint64_t>(kCycles) * kInner)
        .minEpochIterations(200);
    b.run("HYLE StackAllocator mark/rewind (LIFO scratch)", [&] {
      stack.reset();
      for (int i = 0; i < kBase; ++i)
        nb::doNotOptimizeAway(require_allocation(stack.allocate(kObj)));
      for (int c = 0; c < kCycles; ++c) {
        auto m = stack.mark();
        for (int i = 0; i < kInner; ++i)
          nb::doNotOptimizeAway(require_allocation(stack.allocate(kObj)));
        stack.rewind(m); // free exactly this cycle's temps, keep the base batch
      }
    });
    // Baseline: a full reset() can't keep the base batch, but as a pure
    // pointer- move reference it shows rewind is the same O(1) cost (re-bump
    // the temps each cycle from a full reset of just the temp region).
    StackAllocator stack2(sys, kRegion);
    b.run("HYLE StackAllocator reset re-bump (baseline)", [&] {
      for (int c = 0; c < kCycles; ++c) {
        stack2.reset();
        for (int i = 0; i < kInner; ++i)
          nb::doNotOptimizeAway(require_allocation(stack2.allocate(kObj)));
      }
    });
  }

  // ---- B. SINGLE-THREAD pool churn (fair tools only) ----
  // GrowablePool is the ST same-size free-list tool. Rivals: pmr unsync pool
  // (std ST pool SOTA) + raw mi (general baseline). SharedPool/SharedPoolCache
  // are omitted: a single-thread-only owner should use the simpler
  // GrowablePool.
  if (want_section('B')) {
    nb::Bench b;
    b.title("single-thread pool churn: 8192x alloc/free 64B")
        .relative(true)
        .batch(kN)
        .minEpochIterations(50);

    struct Obj {
      std::byte b[kObj];
    };
    GrowablePool<Obj> gpool(sys, 64);
    b.run("HYLE GrowablePool (single-thread)", [&] {
      for (int i = 0; i < kN; ++i) {
        Obj *p = static_cast<Obj *>(require_allocation(gpool.create()));
        nb::doNotOptimizeAway(std::as_const(p));
        gpool.destroy(p);
      }
    });

    std::pmr::unsynchronized_pool_resource upr;
    b.run("std::pmr::unsynchronized_pool_resource (SOTA)", [&] {
      for (int i = 0; i < kN; ++i) {
        void *p = upr.allocate(kObj, alignof(std::max_align_t));
        nb::doNotOptimizeAway(std::as_const(p));
        upr.deallocate(p, kObj, alignof(std::max_align_t));
      }
    });

    b.run("raw mi_malloc/free (general)", [&] {
      for (int i = 0; i < kN; ++i) {
        void *p = require_allocation(mi_malloc(kObj));
        nb::doNotOptimizeAway(std::as_const(p));
        mi_free(p);
      }
    });
  }

  // ---- C. CONCURRENT pool churn: APPLES-TO-APPLES, not apples-to-oranges ----
  // This section separates three contracts instead of declaring a pooled fixed
  // slab faster/slower from one misleading denominator:
  //   (1) per-thread ownership -> GrowablePool per thread vs mimalloc's local
  //   heaps; (2) shared HARD capacity -> SharedPool, with immediate post-free
  //   visibility; (3) explicit owner-local SharedPoolCache over a shared core.
  // oneTBB fixed_pool shares one caller-owned fixed buffer, but its tbbmalloc
  // frontend keeps owner-thread private free lists and publishes only remote
  // frees. It is a mature fixed-buffer throughput reference, not a rival for
  // SharedPool's immediate post-free visibility. pmr, scalable_allocator,
  // mimalloc, and per-worker GrowablePool are growable references only.
  if (want_section('C')) {
    const int kThreads = env_int("HYLE_BENCH_THREADS", 8);
    constexpr int kPer = 50000;
    auto run_threads = [&](auto &&body) {
      std::vector<std::thread> ts;
      ts.reserve(kThreads);
      for (int t = 0; t < kThreads; ++t)
        ts.emplace_back(body);
      for (auto &th : ts)
        th.join();
    };
    struct Obj {
      std::byte b[kObj];
    };
    nb::Bench b;
    b.title(std::to_string(kThreads) + "-thread worker object reuse (batch=1)")
        .relative(true)
        .batch(kThreads * kPer)
        .minEpochIterations(10);

    // (1) APPLES-TO-APPLES: a GrowablePool PER THREAD (no sharing, no atomics)
    // -- HYLE's SOTA-parity answer for per-thread same-size churn, the SAME
    // shape as mimalloc's per-thread heaps.
    b.run("HYLE GrowablePool (one pool per worker)", [&] {
      run_threads([&] {
        GrowablePool<Obj> local(sys, 256);
        for (int i = 0; i < kPer; ++i) {
          Obj *p = static_cast<Obj *>(require_allocation(local.create()));
          nb::doNotOptimizeAway(std::as_const(p));
          local.destroy(p);
        }
      });
    });

    SharedPool shared_cache_core(sys, kObj, kThreads * 256);
    b.run("HYLE SharedPoolCache (explicit owner-local reuse)", [&] {
      run_threads([&] {
        std::array<std::uint32_t, 128> storage{};
        SharedPoolCache cache(shared_cache_core, storage, 32);
        for (int i = 0; i < kPer; ++i) {
          void *p = require_allocation(cache.allocate());
          nb::doNotOptimizeAway(std::as_const(p));
          cache.deallocate(p);
        }
      });
    });

    // Hard-capacity counterpart: every operation publishes through a shared
    // home list, isolating deterministic visibility from the cached
    // worker-local fast path.
    SharedPool shared_hard(sys, kObj, kThreads * 256);
    b.run("HYLE SharedPool (hard capacity)", [&] {
      run_threads([&] {
        for (int i = 0; i < kPer; ++i) {
          void *p = require_allocation(shared_hard.allocate());
          nb::doNotOptimizeAway(std::as_const(p));
          shared_hard.deallocate(p);
        }
      });
    });

#if defined(HYLE_BENCH_TBB)
    // Fixed-buffer reference: one caller-owned buffer shared by all workers,
    // with implicit owner-thread locality inside tbbmalloc.
    std::vector<std::byte> tbb_fixed_storage(
        static_cast<std::size_t>(kThreads) * 256 * kObj +
        kTbbFixedBootstrapBytes);
    tbb::fixed_pool tbb_fixed(tbb_fixed_storage.data(),
                              tbb_fixed_storage.size());
    b.run("oneTBB fixed_pool (fixed-buffer shared)", [&] {
      run_threads([&] {
        for (int i = 0; i < kPer; ++i) {
          void *p = require_allocation(tbb_fixed.malloc(kObj));
          nb::doNotOptimizeAway(std::as_const(p));
          tbb_fixed.free(p);
        }
      });
    });
#endif

    // Standard-library shared-pool reference. It is growable rather than
    // bounded, but exercises the same one-instance/many-thread allocation
    // topology. Construct outside the timed lambda, matching every other
    // persistent pool in this section.
    std::pmr::synchronized_pool_resource spr;
    b.run("std::pmr::synchronized_pool_resource (growable shared)", [&] {
      run_threads([&] {
        for (int i = 0; i < kPer; ++i) {
          void *p = spr.allocate(kObj, alignof(std::max_align_t));
          nb::doNotOptimizeAway(std::as_const(p));
          spr.deallocate(p, kObj, alignof(std::max_align_t));
        }
      });
    });

    // Strong production reference: oneTBB scalable_allocator uses per-thread
    // caches plus shared backing. It is growable, so compare throughput rather
    // than capacity semantics. Gated on HYLE_BENCH_TBB.
#if defined(HYLE_BENCH_TBB)
    b.run("oneTBB scalable_allocator (growable shared)", [&] {
      run_threads([&] {
        for (int i = 0; i < kPer; ++i) {
          void *p = require_allocation(scalable_malloc(kObj));
          nb::doNotOptimizeAway(std::as_const(p));
          scalable_free(p);
        }
      });
    });
#endif

    // raw mi_malloc/free, 8 threads -- mimalloc's per-thread heaps (non-atomic
    // same-thread free). The fair rival to row (1), the apples-to-oranges
    // baseline for row (2).
    b.run("raw mi_malloc/free (per-thread heaps)", [&] {
      run_threads([&] {
        for (int i = 0; i < kPer; ++i) {
          void *p = require_allocation(mi_malloc(kObj));
          nb::doNotOptimizeAway(std::as_const(p));
          mi_free(p);
        }
      });
    });
  }

  // ---- D. USER WORKLOADS: request batches + producer/consumer handoff
  // ---------- Derived from oneTBB pool tests and mimalloc's
  // threadtest/Larson/xmalloc shapes. Environment knobs sweep scaling instead
  // of certifying one hand-picked point.
  if (want_section('D')) {
    const int kThreads = env_int("HYLE_BENCH_THREADS", 8);
    const std::size_t block_size =
        static_cast<std::size_t>(env_int("HYLE_BENCH_BLOCK_SIZE", 64));
    auto run_threads = [&](auto &&body) {
      std::vector<std::thread> ts;
      ts.reserve(kThreads);
      for (int t = 0; t < kThreads; ++t)
        ts.emplace_back(body, t);
      for (auto &th : ts)
        th.join();
    };

    // ---- D1. Request batch lifecycle
    // ----------------------------------------- Each worker acquires a
    // request-local batch, keeps the objects live together, then releases the
    // batch. Sweep batch, threads, and working-set/capacity ratio from the
    // environment; the workload is user behavior, not a TLS-threshold probe.
    const int kHold = env_int("HYLE_BENCH_BATCH", 64);
    const int kRounds = env_int("HYLE_BENCH_ROUNDS", 500);
    const int capacity_pct = env_int("HYLE_BENCH_CAPACITY_PCT", 75);
    const std::uint32_t pool_capacity = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(kThreads) * kHold * 100 + capacity_pct -
         1) /
        capacity_pct);
    SharedPool burst_cache_core(sys, block_size, pool_capacity);
    SharedPool burst_hard(sys, block_size, pool_capacity);
    {
      nb::Bench b;
      b.title(std::to_string(kThreads) + " workers: request batch=" +
              std::to_string(kHold) + ", block=" + std::to_string(block_size) +
              "B, working-set=" + std::to_string(capacity_pct) + "% capacity")
          .relative(true)
          .batch(static_cast<std::uint64_t>(kThreads) * kRounds * kHold)
          .minEpochIterations(3);
      b.run("HYLE SharedPoolCache (explicit soft capacity)", [&] {
        run_threads([&](int) {
          std::array<std::uint32_t, 128> storage{};
          SharedPoolCache cache(burst_cache_core, storage, 32);
          std::vector<void *> held(kHold);
          for (int r = 0; r < kRounds; ++r) {
            for (int i = 0; i < kHold; ++i)
              held[i] = require_allocation(cache.allocate());
            for (int i = 0; i < kHold; ++i)
              cache.deallocate(held[i]);
          }
        });
      });
      b.run("HYLE SharedPool (hard capacity)", [&] {
        run_threads([&](int) {
          std::vector<void *> held(kHold);
          for (int r = 0; r < kRounds; ++r) {
            for (int i = 0; i < kHold; ++i)
              held[i] = require_allocation(burst_hard.allocate());
            for (int i = 0; i < kHold; ++i)
              burst_hard.deallocate(held[i]);
          }
        });
      });
#if defined(HYLE_BENCH_TBB)
      std::vector<std::byte> tbb_fixed_storage(
          static_cast<std::size_t>(pool_capacity) * block_size +
          kTbbFixedBootstrapBytes);
      tbb::fixed_pool tbb_fixed(tbb_fixed_storage.data(),
                                tbb_fixed_storage.size());
      b.run("oneTBB fixed_pool (fixed-buffer shared)", [&] {
        run_threads([&](int) {
          std::vector<void *> held(kHold);
          for (int r = 0; r < kRounds; ++r) {
            for (int i = 0; i < kHold; ++i)
              held[i] = require_allocation(tbb_fixed.malloc(block_size));
            for (int i = 0; i < kHold; ++i)
              tbb_fixed.free(held[i]);
          }
        });
      });
#endif
      std::pmr::synchronized_pool_resource spr;
      b.run("std::pmr::synchronized_pool_resource (growable shared)", [&] {
        run_threads([&](int) {
          std::vector<void *> held(kHold);
          for (int r = 0; r < kRounds; ++r) {
            for (int i = 0; i < kHold; ++i)
              held[i] = spr.allocate(block_size, alignof(std::max_align_t));
            for (int i = 0; i < kHold; ++i)
              spr.deallocate(held[i], block_size, alignof(std::max_align_t));
          }
        });
      });
#if defined(HYLE_BENCH_TBB)
      b.run("oneTBB scalable_allocator (burst)", [&] {
        run_threads([&](int) {
          std::vector<void *> held(kHold);
          for (int r = 0; r < kRounds; ++r) {
            for (int i = 0; i < kHold; ++i)
              held[i] = require_allocation(scalable_malloc(block_size));
            for (int i = 0; i < kHold; ++i)
              scalable_free(held[i]);
          }
        });
      });
#endif
      b.run("raw mi_malloc/free (burst, per-thread heaps)", [&] {
        run_threads([&](int) {
          std::vector<void *> held(kHold);
          for (int r = 0; r < kRounds; ++r) {
            for (int i = 0; i < kHold; ++i)
              held[i] = require_allocation(mi_malloc(block_size));
            for (int i = 0; i < kHold; ++i)
              mi_free(held[i]);
          }
        });
      });
    }

    // ---- D2. Producer/consumer handoff
    // --------------------------------------- Each worker produces one batch
    // and consumes its neighbor's batch, matching oneTBB's cross-thread-free
    // test and allocator Larson-style migration. Blocks may migrate between
    // HYLE shared home lists; Cached may park up to its bounded TLS window.
    // Sweep handoff batch independently to expose barrier amortization.
    const int kBatch = env_int("HYLE_BENCH_HANDOFF_BATCH", 256);
    const int kHandoffRounds = env_int("HYLE_BENCH_HANDOFF_ROUNDS", 250);
    const std::uint32_t handoff_capacity =
        static_cast<std::uint32_t>(kThreads * kBatch * 2);
    SharedPool handoff_cache_core(sys, block_size, handoff_capacity);
    SharedPool handoff_hard(sys, block_size, handoff_capacity);
    struct alignas(64) HandoffState {
      std::atomic<int> produced{0};
      std::atomic<int> consumed{0};
    };
    auto run_handoff = [&](auto &&allocate, auto &&deallocate) {
      std::vector<std::vector<void *>> slots(kThreads,
                                             std::vector<void *>(kBatch));
      auto states = std::make_unique<HandoffState[]>(kThreads);
      run_threads([&](int t) {
        const int nbr = (t + 1) % kThreads;
        for (int r = 0; r < kHandoffRounds; ++r) {
          while (states[t].consumed.load(std::memory_order_acquire) < r)
            std::this_thread::yield();
          for (int i = 0; i < kBatch; ++i)
            slots[t][i] = require_allocation(allocate());
          states[t].produced.store(r + 1, std::memory_order_release);
          while (states[nbr].produced.load(std::memory_order_acquire) < r + 1)
            std::this_thread::yield();
          for (int i = 0; i < kBatch; ++i)
            deallocate(slots[nbr][i]);
          states[nbr].consumed.store(r + 1, std::memory_order_release);
        }
      });
    };
    auto run_explicit_cache_handoff = [&] {
      std::vector<std::vector<void *>> slots(kThreads,
                                             std::vector<void *>(kBatch));
      auto states = std::make_unique<HandoffState[]>(kThreads);
      run_threads([&](int t) {
        std::array<std::uint32_t, 128> storage{};
        SharedPoolCache cache(handoff_cache_core, storage, 32);
        const int nbr = (t + 1) % kThreads;
        for (int r = 0; r < kHandoffRounds; ++r) {
          while (states[t].consumed.load(std::memory_order_acquire) < r)
            std::this_thread::yield();
          for (int i = 0; i < kBatch; ++i)
            slots[t][i] = require_allocation(cache.allocate());
          states[t].produced.store(r + 1, std::memory_order_release);
          while (states[nbr].produced.load(std::memory_order_acquire) < r + 1)
            std::this_thread::yield();
          for (int i = 0; i < kBatch; ++i)
            cache.deallocate(slots[nbr][i]);
          states[nbr].consumed.store(r + 1, std::memory_order_release);
        }
      });
    };
    {
      nb::Bench b;
      b.title(std::to_string(kThreads) +
              " workers: producer allocates, neighbor consumer frees; batch=" +
              std::to_string(kBatch) + ", block=" + std::to_string(block_size) +
              "B")
          .relative(true)
          .batch(static_cast<std::uint64_t>(kThreads) * kHandoffRounds * kBatch)
          .minEpochIterations(3);
      b.run("HYLE SharedPoolCache (explicit soft capacity)", [&] {
        run_explicit_cache_handoff();
      });
      b.run("HYLE SharedPool (hard capacity)", [&] {
        run_handoff([&] { return handoff_hard.allocate(); },
                    [&](void *p) { handoff_hard.deallocate(p); });
      });
#if defined(HYLE_BENCH_TBB)
      std::vector<std::byte> tbb_fixed_storage(
          static_cast<std::size_t>(handoff_capacity) * block_size +
          kTbbFixedBootstrapBytes);
      tbb::fixed_pool tbb_fixed(tbb_fixed_storage.data(),
                                tbb_fixed_storage.size());
      b.run("oneTBB fixed_pool (fixed-buffer shared)", [&] {
        run_handoff([&] { return tbb_fixed.malloc(block_size); },
                    [&](void *p) { tbb_fixed.free(p); });
      });
#endif
      std::pmr::synchronized_pool_resource spr;
      b.run("std::pmr::synchronized_pool_resource (growable shared)", [&] {
        run_handoff(
            [&] { return spr.allocate(block_size, alignof(std::max_align_t)); },
            [&](void *p) {
              spr.deallocate(p, block_size, alignof(std::max_align_t));
            });
      });
#if defined(HYLE_BENCH_TBB)
      b.run("oneTBB scalable_allocator (cross-thread)", [&] {
        run_handoff([&] { return scalable_malloc(block_size); },
                    [&](void *p) { scalable_free(p); });
      });
#endif
      b.run("raw mi_malloc/free (cross-thread, deferred atomic push)", [&] {
        run_handoff([&] { return mi_malloc(block_size); },
                    [&](void *p) { mi_free(p); });
      });
    }
  }

  // ---- E. SERVICE QUOTA: capacity pressure + recovery
  // ------------------------ Fixed-capacity-only contract test/measurement.
  // Growable allocators are excluded: they cannot answer whether a bounded
  // service admits exactly N concurrent objects.
  if (want_section('E')) {
    const int threads = env_int("HYLE_BENCH_THREADS", 8);
    const std::size_t block_size =
        static_cast<std::size_t>(env_int("HYLE_BENCH_BLOCK_SIZE", 64));
    const std::size_t capacity =
        static_cast<std::size_t>(env_int("HYLE_BENCH_QUOTA", 1024));
    const int request_pct = env_int("HYLE_BENCH_REQUEST_PCT", 110);
    SharedPool hard(sys, block_size, static_cast<std::uint32_t>(capacity));
    SharedPool cached_core(sys, block_size,
                           static_cast<std::uint32_t>(capacity));
    const QuotaResult hard_result =
        run_quota_scenario(hard, threads, capacity, request_pct);
    const QuotaResult cached_result = run_cached_quota_scenario(
        cached_core, threads, capacity, request_pct);

    std::printf(
        "\nservice quota: threads=%d, block=%zuB, capacity=%zu, request=%d%%\n",
        threads, block_size, capacity, request_pct);
    std::printf("\n| pool | attempted | admitted | failed | recovered before "
                "publish | recovered after publish | ns/attempt |\n");
    std::printf("|:--|--:|--:|--:|--:|--:|--:|\n");
    auto print = [](const char *name, const QuotaResult &r) {
      std::printf("| %s | %zu | %zu | %zu | %zu | %zu | %.2f |\n", name,
                  r.attempted, r.initial_success,
                  r.attempted - r.initial_success, r.recovered_before_publish,
                  r.recovered_after_publish, r.initial_ns_per_attempt);
    };
    print("HYLE SharedPool (hard)", hard_result);
    print("HYLE SharedPoolCache (explicit soft)", cached_result);
  }

  // ---- F. CLOSURE SHAPES: real lifetime/ownership policies ------------------
  if (want_section('F')) {
    constexpr int kScratchAllocations = 4096;
    constexpr std::size_t kScratchBytes = 1024 * 1024;

    // External parser/codec workspace: exactly the same caller-owned byte
    // buffer, no fallback, all temporaries die together at the end of one
    // request.
    {
      std::vector<std::byte> hyle_storage(kScratchBytes);
      std::vector<std::byte> pmr_storage(kScratchBytes);
      FixedRegion fixed(hyle_storage.data(), hyle_storage.size());
      nb::Bench b;
      b.title("external request workspace: 4096 mixed-size allocations + reset")
          .relative(true)
          .batch(kScratchAllocations)
          .minEpochIterations(50);
      b.run("HYLE FixedRegion (external, no fallback)", [&] {
        fixed.clear();
        for (int i = 0; i < kScratchAllocations; ++i) {
          void *p = require_allocation(fixed.allocate(scratch_size(i), 16));
          nb::doNotOptimizeAway(std::as_const(p));
        }
      });
      b.run("std::pmr::monotonic_buffer_resource (external, reconstruct)", [&] {
        std::pmr::monotonic_buffer_resource pmr(
            pmr_storage.data(), pmr_storage.size(),
            std::pmr::null_memory_resource());
        for (int i = 0; i < kScratchAllocations; ++i) {
          void *p = pmr.allocate(scratch_size(i), 16);
          nb::doNotOptimizeAway(std::as_const(p));
        }
      });
    }

    // Owned request/frame scratch: ordinary blocks are retained across clear(),
    // so this measures the warmed steady state expected by a server or graph
    // phase.
    {
      hyle::mem::RegionOptions options;
      options.initial_block_size = 64 * 1024;
      options.max_block_size = 256 * 1024;
      options.large_allocation_threshold = 128 * 1024;
      Region region(sys, options);
      nb::Bench b;
      b.title("owned request region: 4096 mixed-size allocations + clear")
          .relative(true)
          .batch(kScratchAllocations)
          .minEpochIterations(50);
      b.run("HYLE Region (retained ordinary blocks)", [&] {
        region.clear();
        for (int i = 0; i < kScratchAllocations; ++i) {
          void *p = require_allocation(region.allocate(scratch_size(i), 16));
          nb::doNotOptimizeAway(std::as_const(p));
        }
      });

      std::pmr::monotonic_buffer_resource pmr;
      b.run("std::pmr::monotonic_buffer_resource (release upstream blocks)",
            [&] {
              pmr.release();
              for (int i = 0; i < kScratchAllocations; ++i) {
                void *p = pmr.allocate(scratch_size(i), 16);
                nb::doNotOptimizeAway(std::as_const(p));
              }
            });
    }

    // Variable-size allocator inside a fixed shared-memory/packet workspace.
    // The live window forces splitting, reuse and adjacent coalescing.
    {
      constexpr int kOperations = 4096;
      constexpr int kWindow = 128;
      constexpr std::size_t kStorageBytes = 1024 * 1024;
      std::vector<std::byte> storage(kStorageBytes);
      FixedPool pool(storage.data(), storage.size());
      nb::Bench b;
      b.title("external fixed buffer: mixed-size sliding live window")
          .relative(true)
          .batch(kOperations)
          .minEpochIterations(30);
      b.run("HYLE FixedPool (single-thread, hard buffer)", [&] {
        std::array<void *, kWindow> live{};
        for (int i = 0; i < kOperations; ++i) {
          const int slot = i % kWindow;
          pool.deallocate(live[slot]);
          live[slot] = pool.allocate(scratch_size(i), 16);
          if (live[slot] == nullptr) {
            std::fprintf(stderr,
                         "FixedPool benchmark failure: operation=%d used=%zu "
                         "capacity=%zu\n",
                         i, pool.used(), pool.capacity());
            std::abort();
          }
          nb::doNotOptimizeAway(std::as_const(live[slot]));
        }
        for (void *p : live)
          pool.deallocate(p);
        if (pool.used() != 0) {
          std::fprintf(stderr,
                       "FixedPool benchmark leak: used=%zu capacity=%zu\n",
                       pool.used(), pool.capacity());
          std::abort();
        }
      });
#if defined(HYLE_BENCH_TBB)
      std::vector<std::byte> tbb_storage(kStorageBytes +
                                         kTbbFixedBootstrapBytes);
      tbb::fixed_pool tbb_pool(tbb_storage.data(), tbb_storage.size());
      b.run("oneTBB fixed_pool (thread-safe, hard buffer)", [&] {
        std::array<void *, kWindow> live{};
        for (int i = 0; i < kOperations; ++i) {
          const int slot = i % kWindow;
          if (live[slot] != nullptr)
            tbb_pool.free(live[slot]);
          live[slot] = require_allocation(tbb_pool.malloc(scratch_size(i)));
          nb::doNotOptimizeAway(std::as_const(live[slot]));
        }
        for (void *p : live)
          tbb_pool.free(p);
      });
#endif
    }

    // Strict admission is a policy adapter. Compare it only with the exact same
    // backing calls so the delta is the atomic reservation/accounting cost.
    {
      constexpr int kOperations = 8192;
      QuotaAllocator quota(sys, 1024 * 1024);
      Allocator bounded = quota.get();
      nb::Bench b;
      b.title("allocator decoration: strict requested-byte quota")
          .relative(true)
          .batch(kOperations)
          .minEpochIterations(30);
      b.run("HYLE system allocator (direct)", [&] {
        for (int i = 0; i < kOperations; ++i) {
          void *p = require_allocation(sys.allocate(64, 16));
          nb::doNotOptimizeAway(std::as_const(p));
          sys.deallocate(p, 64, 16);
        }
      });
      b.run("HYLE QuotaAllocator -> system allocator", [&] {
        for (int i = 0; i < kOperations; ++i) {
          void *p = require_allocation(bounded.allocate(64, 16));
          nb::doNotOptimizeAway(std::as_const(p));
          bounded.deallocate(p, 64, 16);
        }
      });
    }

    // Compression/codec contexts are expensive to initialize and intentionally
    // keep state between uses. This isolates reconstruct-on-return vs
    // preserve-return.
    {
      struct Context {
        std::array<std::uint32_t, 1024> table{};
        Context() {
          for (std::size_t i = 0; i < table.size(); ++i)
            table[i] = static_cast<std::uint32_t>(i * 2654435761u);
        }
      };
      constexpr int kOperations = 1024;
      ObjectPool<Context> reconstruct(sys, 8);
      PreservingObjectPool<Context> preserve(sys, 8);
      if (!reconstruct.valid() || !preserve.valid()) {
        std::fputs("mem_scenarios_bench: context pool construction failed\n",
                   stderr);
        return 2;
      }
      nb::Bench b;
      b.title("expensive reusable context: acquire/use/return")
          .relative(true)
          .batch(kOperations)
          .minEpochIterations(20);
      b.run("HYLE ObjectPool (reconstruct each loan)", [&] {
        for (int i = 0; i < kOperations; ++i) {
          auto loan = reconstruct.try_loan();
          if (!loan)
            std::abort();
          loan->table[0] ^= static_cast<std::uint32_t>(i);
          nb::doNotOptimizeAway(loan.get());
        }
      });
      b.run("HYLE PreservingObjectPool (construct once)", [&] {
        for (int i = 0; i < kOperations; ++i) {
          auto loan = preserve.try_loan();
          if (!loan)
            std::abort();
          loan->table[0] ^= static_cast<std::uint32_t>(i);
          nb::doNotOptimizeAway(loan.get());
        }
      });
    }
  }

  // ---- G. REQUEST-LEVEL LATENCY CERTIFICATE --------------------------------
  if (want_section('G')) {
    constexpr int kOperations = 4096;
    constexpr std::size_t kStorageBytes = 1024 * 1024;
    const int samples = env_int("HYLE_MEM_CERT_SAMPLES", 4096);
    print_certificate_metadata();

    {
      std::vector<std::byte> hyle_storage(kStorageBytes);
      std::vector<std::byte> pmr_storage(kStorageBytes);
      FixedRegion fixed(hyle_storage.data(), hyle_storage.size());
      report_request_latency_pair(
          "external_region", "hyle_fixed_region", "pmr_reconstruct",
          kOperations, samples,
          [&] {
            fixed.clear();
            for (int i = 0; i != kOperations; ++i) {
              void *p = require_allocation(fixed.allocate(scratch_size(i), 16));
              nb::doNotOptimizeAway(std::as_const(p));
            }
          },
          [&] {
            std::pmr::monotonic_buffer_resource pmr(
                pmr_storage.data(), pmr_storage.size(),
                std::pmr::null_memory_resource());
            for (int i = 0; i != kOperations; ++i) {
              void *p = pmr.allocate(scratch_size(i), 16);
              nb::doNotOptimizeAway(std::as_const(p));
            }
          });
    }

    {
      hyle::mem::RegionOptions options;
      options.initial_block_size = 64 * 1024;
      options.max_block_size = 256 * 1024;
      options.large_allocation_threshold = 128 * 1024;
      Region region(sys, options);
      std::pmr::monotonic_buffer_resource pmr;
      report_request_latency_pair(
          "owned_region", "hyle_region", "pmr_release", kOperations,
          samples,
          [&] {
            region.clear();
            for (int i = 0; i != kOperations; ++i) {
              void *p = require_allocation(region.allocate(scratch_size(i), 16));
              nb::doNotOptimizeAway(std::as_const(p));
            }
          },
          [&] {
            pmr.release();
            for (int i = 0; i != kOperations; ++i) {
              void *p = pmr.allocate(scratch_size(i), 16);
              nb::doNotOptimizeAway(std::as_const(p));
            }
          });
    }

#if defined(HYLE_BENCH_TBB)
    {
      constexpr int kWindow = 128;
      std::vector<std::byte> hyle_storage(kStorageBytes);
      std::vector<std::byte> tbb_storage(kStorageBytes +
                                         kTbbFixedBootstrapBytes);
      FixedPool pool(hyle_storage.data(), hyle_storage.size());
      tbb::fixed_pool tbb_pool(tbb_storage.data(), tbb_storage.size());
      report_request_latency_pair(
          "fixed_pool_window", "hyle_fixed_pool", "onetbb_fixed_pool",
          kOperations, samples,
          [&] {
            std::array<void *, kWindow> live{};
            for (int i = 0; i != kOperations; ++i) {
              const int slot = i % kWindow;
              pool.deallocate(live[slot]);
              live[slot] =
                  require_allocation(pool.allocate(scratch_size(i), 16));
              nb::doNotOptimizeAway(std::as_const(live[slot]));
            }
            for (void *p : live)
              pool.deallocate(p);
          },
          [&] {
            std::array<void *, kWindow> live{};
            for (int i = 0; i != kOperations; ++i) {
              const int slot = i % kWindow;
              if (live[slot] != nullptr)
                tbb_pool.free(live[slot]);
              live[slot] =
                  require_allocation(tbb_pool.malloc(scratch_size(i)));
              nb::doNotOptimizeAway(std::as_const(live[slot]));
            }
            for (void *p : live)
              tbb_pool.free(p);
          });
    }
#endif
  }

  // ---- H. ISOLATED REQUEST-SCRATCH FOOTPRINT CERTIFICATE ------------------
  // A single child runs exactly one allocator variant. Running both in one
  // process would let the first allocator's pages and caches contaminate the
  // second allocator's RSS. scripts/mem-footprint-stat.py owns the alternating
  // child-process order and commit-bound aggregation.
  if (want_section('H')) {
    const char *variant = std::getenv("HYLE_MEM_FOOTPRINT_VARIANT");
    if (variant == nullptr || variant[0] == '\0') {
      std::puts("# HYLE_MEM_CERT footprint skipped: set "
                "HYLE_MEM_FOOTPRINT_VARIANT=hyle_region or pmr_monotonic");
    } else {
      const int target_mib = env_int("HYLE_MEM_FOOTPRINT_MIB", 128);
      constexpr std::size_t kMiB = 1024u * 1024u;
      if (static_cast<std::size_t>(target_mib) >
          std::numeric_limits<std::size_t>::max() / kMiB) {
        std::fputs("mem_scenarios_bench: footprint target overflows size_t\n",
                   stderr);
        return 2;
      }
      const std::size_t target = static_cast<std::size_t>(target_mib) * kMiB;
      const std::size_t reuse_target = target / 4;
      const int block_mib = env_int("HYLE_MEM_FOOTPRINT_BLOCK_MIB", 4);
      if (static_cast<std::size_t>(block_mib) >
          std::numeric_limits<std::size_t>::max() / kMiB) {
        std::fputs(
            "mem_scenarios_bench: footprint block size overflows size_t\n",
            stderr);
        return 2;
      }
      const std::size_t region_max_block =
          static_cast<std::size_t>(block_mib) * kMiB;
      const Allocator footprint_backing = direct_mi_allocator();
      print_certificate_metadata();

      if (std::strcmp(variant, "hyle_region") == 0) {
        RegionOptions options;
        options.initial_block_size = 64 * 1024;
        options.max_block_size = region_max_block;
        options.large_allocation_threshold = 1024 * 1024;
        Region region(footprint_backing, options);
        mi_collect(true);
        const ProcessMemory baseline = process_memory();
        const std::size_t requested =
            fill_scratch_request(target, [&](std::size_t bytes) {
              return region.allocate(bytes, 16);
            });
        const std::size_t peak_reserved = region.reserved();
        const ProcessMemory live = process_memory();
        region.clear();
        mi_collect(true);
        const std::size_t retained_after_clear = region.reserved();
        const ProcessMemory cleared = process_memory();
        const std::size_t reuse_requested =
            fill_scratch_request(reuse_target, [&](std::size_t bytes) {
              return region.allocate(bytes, 16);
            });
        const std::size_t reuse_reserved = region.reserved();
        const ProcessMemory reuse = process_memory();
        region.clear();
        region.release();
        mi_collect(true);
        const std::size_t released = region.reserved();
        const ProcessMemory released_memory = process_memory();
        print_footprint_row(variant, requested, reuse_requested,
                            region_max_block, peak_reserved,
                            retained_after_clear, reuse_reserved, released,
                            baseline, live, cleared, reuse, released_memory);
      } else if (std::strcmp(variant, "pmr_monotonic") == 0) {
        StatsAllocator tracked(footprint_backing);
        PmrResource upstream(tracked.get());
        std::pmr::monotonic_buffer_resource region(&upstream);
        mi_collect(true);
        const ProcessMemory baseline = process_memory();
        const std::size_t requested =
            fill_scratch_request(target, [&](std::size_t bytes) {
              return region.allocate(bytes, 16);
            });
        const std::size_t peak_reserved =
            static_cast<std::size_t>(tracked.snapshot().live_bytes);
        const ProcessMemory live = process_memory();
        region.release();
        mi_collect(true);
        const std::size_t retained_after_clear =
            static_cast<std::size_t>(tracked.snapshot().live_bytes);
        const ProcessMemory cleared = process_memory();
        const std::size_t reuse_requested =
            fill_scratch_request(reuse_target, [&](std::size_t bytes) {
              return region.allocate(bytes, 16);
            });
        const std::size_t reuse_reserved =
            static_cast<std::size_t>(tracked.snapshot().live_bytes);
        const ProcessMemory reuse = process_memory();
        region.release();
        mi_collect(true);
        const std::size_t released =
            static_cast<std::size_t>(tracked.snapshot().live_bytes);
        const ProcessMemory released_memory = process_memory();
        print_footprint_row(variant, requested, reuse_requested,
                            region_max_block, peak_reserved,
                            retained_after_clear, reuse_reserved, released,
                            baseline, live, cleared, reuse, released_memory);
      } else {
        std::fprintf(stderr,
                     "mem_scenarios_bench: unknown footprint variant %s\n",
                     variant);
        return 2;
      }
    }
  }

  // ---- I. ISOLATED FIXED-POOL FRAGMENTATION CERTIFICATE -------------------
  if (want_section('I')) {
    const char *variant = std::getenv("HYLE_MEM_FRAGMENTATION_VARIANT");
    if (variant == nullptr || variant[0] == '\0') {
      std::puts("# HYLE_MEM_CERT fragmentation skipped: set "
                "HYLE_MEM_FRAGMENTATION_VARIANT=hyle_fixed_pool or "
                "onetbb_fixed_pool");
    } else {
      const int payload_mib = env_int("HYLE_MEM_FRAGMENTATION_MIB", 8);
      constexpr std::size_t kMiB = 1024u * 1024u;
      if (static_cast<std::size_t>(payload_mib) >
          (std::numeric_limits<std::size_t>::max() -
           kTbbFixedBootstrapBytes) /
              kMiB) {
        std::fputs(
            "mem_scenarios_bench: fragmentation buffer overflows size_t\n",
            stderr);
        return 2;
      }
      const std::size_t payload_budget =
          static_cast<std::size_t>(payload_mib) * kMiB;
      print_certificate_metadata();

      if (std::strcmp(variant, "hyle_fixed_pool") == 0) {
        std::vector<std::byte> storage(payload_budget);
        FixedPool pool(storage.data(), storage.size());
        if (!pool.valid())
          return 2;
        run_fragmentation_scenario(
            variant, payload_budget, storage.size(),
            [&](std::size_t bytes) { return pool.allocate(bytes, 16); },
            [&](void *p) { pool.deallocate(p); });
#if defined(HYLE_BENCH_TBB)
      } else if (std::strcmp(variant, "onetbb_fixed_pool") == 0) {
        std::vector<std::byte> storage(payload_budget +
                                       kTbbFixedBootstrapBytes);
        tbb::fixed_pool pool(storage.data(), storage.size());
        run_fragmentation_scenario(
            variant, payload_budget, storage.size(),
            [&](std::size_t bytes) { return pool.malloc(bytes); },
            [&](void *p) { pool.free(p); });
#endif
      } else {
        std::fprintf(stderr,
                     "mem_scenarios_bench: unknown fragmentation variant %s\n",
                     variant);
        return 2;
      }
    }
  }

  return 0;
}

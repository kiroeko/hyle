#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>

#include "hyle/core/environment.h"
#include "hyle/error/result.h"
#include "hyle/platform/visibility.h"

// Runtime CPU-feature detection (CPUID + XGETBV on x86). Lives in platform/ --
// the layer below both simd/ and container/ -- because "what can this CPU run"
// is a platform capability, not a SIMD-module concept. Any module can query it
// WITHOUT depending on the (heavy) simd module: simd/ and container/ each
// match these facts against the complete requirements of the implementation
// body they own. Platform deliberately does not choose either module's target.
//
// PROBE ISOLATION (llama.cpp's hard-won bug, conventions §10): the code that
// runs CPUID to answer "do we have AVX512?" must itself be compiled for the
// BASELINE ISA, in its own TU, with LTO OFF. Otherwise LTO inlines an
// advanced-ISA instruction INTO the detector, and the detector SIGILLs on an
// old CPU before it can report that the CPU is old. cpu.cc is that isolated,
// LTO-disabled TU.

namespace hyle::platform {

// Unordered CPU/OS facts. These are inputs to an implementation owner's target
// requirements, never a cross-architecture score. Ordinary vector facts mean
// the process may execute that extension (including the required x86 XCR0
// state). AMX is different: Linux tile permission is per-thread, so its
// explicitly named Hardware facts must be combined with platform-thread's
// permission API.
enum class CpuFeature : std::uint16_t {
  // x86-64
  kX86Sse2,
  kX86Sse3,
  kX86Ssse3,
  kX86Sse41,
  kX86Sse42,
  kX86Aes,
  kX86Pclmulqdq,
  kX86Avx,
  kX86Avx2,
  kX86Fma,
  kX86F16c,
  kX86Lzcnt,
  kX86Bmi1,
  kX86Bmi2,
  kX86Avx512F,
  kX86Avx512Bw,
  kX86Avx512Vl,
  kX86Avx512Dq,
  kX86Avx512Cd,
  kX86Avx512Vnni,
  kX86Avx512Vbmi,
  kX86Avx512Vbmi2,
  kX86Avx512Bitalg,
  kX86Avx512Vpopcntdq,
  kX86Gfni,
  kX86Vaes,
  kX86Vpclmulqdq,
  kX86AvxVnni,
  kX86Avx512Bf16,
  kX86Avx512Fp16,
  kX86Avx10_2_512,
  kX86AmxTileHardware,
  kX86AmxInt8Hardware,
  kX86AmxBf16Hardware,
  kX86AmxFp16Hardware,

  // AArch64
  kArmNeon,
  kArmAes,
  kArmPmull,
  kArmDotProd,
  kArmFp16,
  kArmSve,
  kArmSve2,
  kArmSveAes,
  kArmSvePmull,
  kArmSveI8mm,
  kArmSveF32mm,
  kArmSveF64mm,
  kArmSveBf16,
  kArmI8mm,
  kArmBf16,
  kArmSme,
  kArmSme2,

  // RISC-V
  kRiscvV,
  kRiscvZvbb,
  kRiscvZvbc,
  kRiscvZvkned,
  kRiscvZvfh,
  kRiscvZvfhmin,

  kCount,
};

// Fixed-size value set: cheap to copy into a target-requirements constant and
// free of allocation, provider state, architecture ordering or dynamic storage.
class CpuFeatureSet final {
 public:
  constexpr CpuFeatureSet() noexcept = default;

  constexpr explicit CpuFeatureSet(
      std::initializer_list<CpuFeature> features) noexcept {
    for (CpuFeature feature : features) set(feature);
  }

  [[nodiscard]] constexpr bool has(CpuFeature feature) const noexcept {
    const std::size_t index = static_cast<std::size_t>(feature);
    return index < kFeatureCount &&
           (words_[index / kBitsPerWord] &
            (std::uint64_t{1} << (index % kBitsPerWord))) != 0;
  }

  [[nodiscard]] constexpr bool has_all(
      const CpuFeatureSet& required) const noexcept {
    for (std::size_t i = 0; i < words_.size(); ++i) {
      if ((words_[i] & required.words_[i]) != required.words_[i]) return false;
    }
    return true;
  }

  [[nodiscard]] constexpr bool has_any(
      const CpuFeatureSet& candidates) const noexcept {
    for (std::size_t i = 0; i < words_.size(); ++i) {
      if ((words_[i] & candidates.words_[i]) != 0) return true;
    }
    return false;
  }

  constexpr CpuFeatureSet& set(CpuFeature feature,
                               bool enabled = true) noexcept {
    const std::size_t index = static_cast<std::size_t>(feature);
    if (index >= kFeatureCount) return *this;
    const std::uint64_t mask = std::uint64_t{1} << (index % kBitsPerWord);
    if (enabled) words_[index / kBitsPerWord] |= mask;
    else words_[index / kBitsPerWord] &= ~mask;
    return *this;
  }

  [[nodiscard]] friend constexpr bool operator==(
      const CpuFeatureSet&, const CpuFeatureSet&) noexcept = default;

 private:
  static constexpr std::size_t kBitsPerWord = 64;
  static constexpr std::size_t kFeatureCount =
      static_cast<std::size_t>(CpuFeature::kCount);
  static constexpr std::size_t kWordCount = 2;
  static_assert(kFeatureCount <= kWordCount * kBitsPerWord,
                "CpuFeatureSet exhausted its audited fixed capacity");

  std::array<std::uint64_t, kWordCount> words_{};
};

// The probed fact set for this process. Detection is cached and thread-safe;
// subsequent calls and set membership are O(1) and perform no system call.
HYLE_PLATFORM_API const CpuFeatureSet& cpu_features() noexcept;

// Cached identity of the processor that performs the first query. Identity is
// separate from executable ISA features, topology and machine-local CPU ranks.
// x86 CPUID supplies these facts; unsupported identities stay empty/zero.
// vendor/brand are NUL-terminated raw hardware spellings, not routing scores.
struct CpuIdentity final {
  std::array<char, 13> vendor{};
  std::array<char, 49> brand{};
  std::uint32_t family = 0;
  std::uint32_t model = 0;
  std::uint32_t stepping = 0;
};

// First query uses the isolated baseline probe; later calls are allocation-free
// and perform no CPUID or system call. Heterogeneous-package users must not
// reinterpret one observed identity as a guarantee for every processor.
HYLE_PLATFORM_API const CpuIdentity &cpu_identity() noexcept;

// Stable diagnostic spelling for every declared feature; invalid values return
// "unknown". This is not a target name or dispatch recommendation.
HYLE_PLATFORM_API const char* cpu_feature_name(CpuFeature feature) noexcept;

// Best-effort L1 data cache line observation in bytes (typically 64 on
// x86/ARM, 128 on Apple Silicon). Probed once from OS-published cache facts
// (Windows cache relations, Linux cache sysfs, or macOS sysctl) and cached;
// lower-level architecture queries are only a fallback when the OS view is
// unavailable. Falls back to 64 when no trustworthy query succeeds. Do not use
// this fallback-able result to define public ABI or safety-critical
// false-sharing padding; use core's compile-time layout contract for that.
HYLE_PLATFORM_API unsigned cache_line_size() noexcept;

// Cache capacity visible to a logical processor, in bytes. Zero means unknown.
struct CacheInfo {
  std::uint64_t l1d = 0;
  std::uint64_t l1i = 0;
  std::uint64_t l2 = 0;
  std::uint64_t l3 = 0;
};

HYLE_PLATFORM_API const CacheInfo& cache_info() noexcept;

// CPU ids on which the current process may run, in the same id space accepted
// by ThreadOpts::affinity / CpuAffinity. Buffer-fill: returns the full count
// and writes up to cap ids; dst may be null for a count-only query. This is
// intentionally NOT cached because affinity/cpuset restrictions can change
// while a process runs. Unlike hardware_concurrency(), this respects Linux
// sched affinity/cgroups and Windows process CPU Sets / single-group affinity
// masks. On Linux the affinity mask belongs to the calling thread and is
// inherited by threads it creates.
HYLE_PLATFORM_API Result<std::size_t> available_cpu_ids(
    unsigned* dst, std::size_t cap) noexcept;

// Per-CPU facts for the current process view. performance_rank is monotonic
// only within this machine: larger means a stronger OS-reported CPU class, 0
// means the OS did not provide a trustworthy CPU-id mapping. Kind is derived
// from the available set: the highest known rank is kPerformance, lower known
// ranks are kEfficiency, and rank 0 stays kUnknown.
enum class CpuKind : std::uint32_t {
  kUnknown = 0,
  kEfficiency = 1,
  kPerformance = 2,
};

struct CpuInfo {
  unsigned id = 0;
  unsigned performance_rank = 0;
  CpuKind kind = CpuKind::kUnknown;

  // Topology ids describe one live process-available snapshot. package/die/
  // core form a hierarchy; cache-domain ids are snapshot-local equivalence
  // classes. kUnknownCpuTopologyId means that the OS cannot map a logical id
  // honestly (not that every CPU belongs to domain zero).
  unsigned package_id = (std::numeric_limits<unsigned>::max)();
  unsigned die_id = (std::numeric_limits<unsigned>::max)();
  unsigned core_id = (std::numeric_limits<unsigned>::max)();
  unsigned smt_index = (std::numeric_limits<unsigned>::max)();
  unsigned smt_count = 0;
  unsigned numa_node = (std::numeric_limits<unsigned>::max)();
  unsigned l2_cache_domain = (std::numeric_limits<unsigned>::max)();
  unsigned l3_cache_domain = (std::numeric_limits<unsigned>::max)();
};

// One process-visible CPU performance/cache class. Records with different
// capacities or cache-sharing widths stay separate even when the OS assigns
// the same performance rank. Rank is machine-local (larger is stronger); all
// zero-valued facts are honestly unknown. logical_cpus counts CPUs currently
// available to this process, while logical_cpus_per_l2/l3 describe the native
// sharing domain and may include CPUs outside a restricted process view.
struct CpuCacheClassInfo {
  unsigned performance_rank = 0;
  CpuKind kind = CpuKind::kUnknown;
  unsigned logical_cpus = 0;
  unsigned physical_cores = 0;
  unsigned cache_line_size = 0;
  unsigned logical_cpus_per_l2 = 0;
  unsigned logical_cpus_per_l3 = 0;
  CacheInfo cache{};
};

inline constexpr unsigned kUnknownCpuTopologyId =
    (std::numeric_limits<unsigned>::max)();

enum class CpuOrder : std::uint8_t {
  kOsDefault = 0,
  kPerformanceFirst = 1,
  kEfficiencyFirst = 2,
};

// Same buffer-fill contract as available_cpu_ids(). The ordered overload keeps
// all process-available CPUs, merely ordering known stronger/weaker classes
// first. If ranks are unavailable it preserves the OS order instead of
// guessing.
HYLE_PLATFORM_API Result<std::size_t> available_cpu_info(
    CpuInfo* dst, std::size_t cap) noexcept;

// Dynamic buffer-fill snapshot of the available CPU cache classes. Returns
// the full record count and writes up to cap records, strongest known rank
// first. Unlike coarse cached cache_info(), this is the tuning input for
// heterogeneous hosts; it never guesses missing cache or CPU-id facts.
HYLE_PLATFORM_API Result<std::size_t> available_cpu_cache_classes(
    CpuCacheClassInfo* dst, std::size_t cap) noexcept;
HYLE_PLATFORM_API Result<std::size_t> available_cpu_ids(
    CpuOrder order, unsigned* dst, std::size_t cap) noexcept;

// Convenience count for the current placement set. Equivalent to a count-only
// ids query and likewise dynamic; it does not include a cgroup/job
// CPU-bandwidth quota. Use process_cpu_capacity() when selecting a default
// worker ceiling.
HYLE_PLATFORM_API Result<std::size_t> available_cpu_count() noexcept;

// One hard CPU-bandwidth limit represented as an exact ratio of CPU time to
// wall/scheduling period. Both fields are positive. For example, 150000/100000
// permits 1.5 logical CPUs worth of sustained execution; it does not restrict
// placement to any particular CPU id.
struct CpuQuota {
  std::uint64_t capacity = 0;
  std::uint64_t period = 0;

  friend constexpr bool operator==(CpuQuota, CpuQuota) noexcept = default;
};

// A live process capacity snapshot. placement_logical_cpus is the number of
// CPUs on which work may be placed. hard_quota is absent when the OS publishes
// no trustworthy hard bandwidth ceiling; weights/shares are deliberately not
// converted into a fictitious limit.
struct ProcessCpuCapacity {
  std::size_t placement_logical_cpus = 0;
  std::optional<CpuQuota> hard_quota;

  // Conservative upper bound for a default CPU-bound worker count. This is a
  // fact-derived convenience, not a scheduling policy: consumers may choose a
  // lower or workload-specific value. Always in [1, placement] on success.
  std::size_t parallelism_ceiling() const noexcept {
    if (!hard_quota || hard_quota->period == 0 || hard_quota->capacity == 0) {
      return placement_logical_cpus;
    }
    const std::uint64_t rounded =
        hard_quota->capacity / hard_quota->period +
        (hard_quota->capacity % hard_quota->period != 0 ? 1u : 0u);
    const std::size_t quota_ceiling =
        rounded > static_cast<std::uint64_t>(
                      (std::numeric_limits<std::size_t>::max)())
            ? (std::numeric_limits<std::size_t>::max)()
            : static_cast<std::size_t>(rounded);
    return (std::min)(placement_logical_cpus,
                      quota_ceiling == 0 ? std::size_t{1} : quota_ceiling);
  }
};

// Dynamic and uncached because affinity/cpuset and cgroup hard limits can
// change. Placement-query failure is an error; an unavailable hard limit is
// represented by an empty optional rather than by zero. In particular,
// documented Windows APIs cannot prove the effective rate of nested Jobs, so
// Windows currently leaves hard_quota absent.
HYLE_PLATFORM_API Result<ProcessCpuCapacity> process_cpu_capacity() noexcept;

// OS-published frequency facts for one process-available logical CPU. Values
// are hertz; zero means unavailable/unknown. current_hz is an observation or
// OS policy report, not a promise about the frequency during later work.
struct CpuFrequencyInfo {
  unsigned id = 0;
  std::uint64_t hardware_max_hz = 0;
  std::uint64_t policy_max_hz = 0;
  std::uint64_t current_hz = 0;
};

// Same dynamic caller-buffer contract as available_cpu_ids(): returns the full
// process-available count and writes up to cap records in OS-default id order.
// The query never estimates frequency from elapsed time or a CPU model table.
HYLE_PLATFORM_API Result<std::size_t> available_cpu_frequencies(
    CpuFrequencyInfo* dst, std::size_t cap) noexcept;

// Aggregate of the same live process-available view returned by
// available_cpu_info(). Counts never invent dense CPU ids and physical_cores
// may be unknown (zero) when the OS exposes a logical count but no trustworthy
// logical-to-core mapping. If a platform cannot report a P/E distinction, kind
// stays unknown rather than frequency-based guessing.
struct CpuTopology {
  unsigned logical_cpus = 0;
  unsigned physical_cores = 0;
  unsigned packages = 0;
  unsigned numa_nodes = 0;
  unsigned l2_cache_domains = 0;
  unsigned l3_cache_domains = 0;
  unsigned performance_cpus = 0;
  unsigned efficiency_cpus = 0;
  bool hybrid = false;
};

HYLE_PLATFORM_API Result<CpuTopology> available_cpu_topology() noexcept;

// OS-online logical processor count (>=1). Placement and worker sizing should
// normally use available_cpu_ids()/available_cpu_topology() for placement and
// process_cpu_capacity() for a hard CPU-bandwidth-aware planning ceiling.
HYLE_PLATFORM_API unsigned hardware_concurrency() noexcept;

}  // namespace hyle::platform

#include <nanobench.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <optional>
#include <string_view>
#include <utility>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "hyle/platform/cpu.h"
#include "hyle/platform/host_memory.h"

#include "../common/perf_process.h"

namespace {
namespace nb = ankerl::nanobench;

[[noreturn]] void fail(const char* operation) {
  std::fprintf(stderr, "hyle_platform_host_bench: %s failed\n", operation);
  std::abort();
}

void require(bool condition, const char* operation) {
  if (!condition) fail(operation);
}

bool environment_present(const char* name) {
  char* value = nullptr;
  std::size_t size = 0;
  const bool present =
      _dupenv_s(&value, &size, name) == 0 && value != nullptr;
  std::free(value);
  return present;
}

template <class Candidate, class Peer>
void run_pair(nb::Bench& bench, bool peer_first,
              std::string_view candidate_name, Candidate&& candidate,
              std::string_view peer_name, Peer&& peer) {
  if (peer_first) {
    bench.run(peer_name, std::forward<Peer>(peer));
    bench.run(candidate_name, std::forward<Candidate>(candidate));
  } else {
    bench.run(candidate_name, std::forward<Candidate>(candidate));
    bench.run(peer_name, std::forward<Peer>(peer));
  }
}

struct HostSnapshot {
  hyle::platform::CpuTopology topology;
  hyle::platform::HostMemoryInfo memory;
};

template <class T, std::size_t StackCount>
class StackFirstBuffer {
 public:
  StackFirstBuffer() = default;
  ~StackFirstBuffer() {
    if (data_ != stack_) delete[] data_;
  }
  StackFirstBuffer(const StackFirstBuffer&) = delete;
  StackFirstBuffer& operator=(const StackFirstBuffer&) = delete;

  [[nodiscard]] bool resize(std::size_t size) noexcept {
    require(size_ == 0, "stack-first buffer resized twice");
    size_ = size;
    if (size > StackCount) {
      data_ = new (std::nothrow) T[size];
      if (data_ == nullptr) {
        size_ = 0;
        data_ = stack_;
        return false;
      }
    }
    return true;
  }
  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] T* begin() noexcept { return data_; }
  [[nodiscard]] T* end() noexcept { return data_ + size_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] T& operator[](std::size_t index) noexcept {
    return data_[index];
  }
  [[nodiscard]] const T& operator[](std::size_t index) const noexcept {
    return data_[index];
  }

 private:
  alignas(std::max_align_t) T stack_[StackCount];
  T* data_ = stack_;
  std::size_t size_ = 0;
};

struct RawBuffers {
  StackFirstBuffer<ULONG, 256> default_cpu_sets;
  StackFirstBuffer<std::byte, 64 * 1024> cpu_sets;
  StackFirstBuffer<std::byte, 64 * 1024> relations;
  StackFirstBuffer<std::uint16_t, 4096> selected;

  [[nodiscard]] hyle::Status initialize() noexcept {
    const DWORD logical = ::GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (logical == 0) {
      return hyle::Status::OutOfRange(
          "available CPU count exceeds topology limits");
    }
    if (!default_cpu_sets.resize(logical) || !selected.resize(logical)) {
      return hyle::Status::ResourceExhausted(
          "Win32 topology scratch allocation failed");
    }

    DWORD bytes = 0;
    (void)::GetSystemCpuSetInformation(nullptr, 0, &bytes,
                                       ::GetCurrentProcess(), 0);
    if (bytes == 0) {
      return hyle::status_from_win32(
          ::GetLastError(), "GetSystemCpuSetInformation(size)");
    }
    if (!cpu_sets.resize(bytes)) {
      return hyle::Status::ResourceExhausted(
          "Win32 CPU-set scratch allocation failed");
    }

    bytes = 0;
    const BOOL sized =
        ::GetLogicalProcessorInformationEx(RelationAll, nullptr, &bytes);
    const DWORD code = ::GetLastError();
    if (sized) {
      return hyle::Status::DataLoss(
          "CPU topology sizing unexpectedly returned relation data");
    }
    if (code != ERROR_INSUFFICIENT_BUFFER || bytes == 0) {
      if (code == ERROR_INSUFFICIENT_BUFFER) {
        return hyle::Status::DataLoss(
            "CPU topology relation size is zero");
      }
      return hyle::status_from_win32(
          code, "GetLogicalProcessorInformationEx(size)");
    }
    if (!relations.resize(bytes)) {
      return hyle::Status::ResourceExhausted(
          "Win32 relation scratch allocation failed");
    }
    return hyle::Status::Ok();
  }
};

hyle::Result<hyle::platform::CpuTopology> raw_topology_snapshot(
    RawBuffers& buffers) noexcept {
  hyle::platform::CpuTopology out;
  HANDLE process = ::GetCurrentProcess();

  ULONG default_count = 0;
  ::SetLastError(ERROR_SUCCESS);
  const BOOL defaults_ok =
      ::GetProcessDefaultCpuSets(process, nullptr, 0, &default_count);
  DWORD code = defaults_ok ? ERROR_SUCCESS : ::GetLastError();
  if (!defaults_ok && code != ERROR_INSUFFICIENT_BUFFER) {
    return std::unexpected(
        hyle::status_from_win32(code, "GetProcessDefaultCpuSets(size)"));
  }
  if (default_count > buffers.default_cpu_sets.size()) {
    return std::unexpected(hyle::Status::Unavailable(
        "default CPU set grew beyond prepared storage"));
  }
  if (default_count != 0) {
    ULONG actual = default_count;
    if (!::GetProcessDefaultCpuSets(process, buffers.default_cpu_sets.data(),
                                    default_count, &actual)) {
      return std::unexpected(hyle::status_from_win32(
          ::GetLastError(), "GetProcessDefaultCpuSets"));
    }
    if (actual > buffers.default_cpu_sets.size()) {
      return std::unexpected(hyle::Status::Unavailable(
          "default CPU set expanded during snapshot"));
    }
    default_count = actual;
    std::sort(buffers.default_cpu_sets.begin(),
              buffers.default_cpu_sets.begin() + default_count);
  }

  DWORD cpu_set_bytes = static_cast<DWORD>(buffers.cpu_sets.size());
  if (!::GetSystemCpuSetInformation(
          reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(
              buffers.cpu_sets.data()),
          cpu_set_bytes, &cpu_set_bytes, process, 0)) {
    return std::unexpected(hyle::status_from_win32(
        ::GetLastError(), "GetSystemCpuSetInformation"));
  }
  if (cpu_set_bytes > buffers.cpu_sets.size()) {
    return std::unexpected(hyle::Status::Unavailable(
        "CPU set data grew beyond prepared storage"));
  }

  const WORD group_count = ::GetActiveProcessorGroupCount();
  if (group_count == 0 || group_count > 64) {
    return std::unexpected(hyle::Status::Unavailable(
        "active processor group count is invalid"));
  }
  std::array<unsigned, 65> group_begin{};
  for (WORD group = 0; group < group_count; ++group) {
    group_begin[group + 1] =
        group_begin[group] + ::GetActiveProcessorCount(group);
  }
  if (group_begin[group_count] > buffers.selected.size()) {
    return std::unexpected(hyle::Status::Unavailable(
        "logical CPU set grew beyond prepared storage"));
  }
  std::fill(buffers.selected.begin(), buffers.selected.end(), 0);

  DWORD_PTR process_mask = 0;
  DWORD_PTR system_mask = 0;
  const bool use_affinity_mask =
      group_count == 1 &&
      ::GetProcessAffinityMask(process, &process_mask, &system_mask) != 0;
  unsigned maximum_rank = 0;
  for (DWORD offset = 0; offset < cpu_set_bytes;) {
    const auto* info = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(
        buffers.cpu_sets.data() + offset);
    if (info->Size == 0 || info->Size > cpu_set_bytes - offset) {
      return std::unexpected(
          hyle::Status::DataLoss("CPU set record is malformed"));
    }
    if (info->Type == CpuSetInformation) {
      const auto& set = info->CpuSet;
      const bool valid_group = set.Group < group_count;
      const bool valid_index =
          valid_group &&
          set.LogicalProcessorIndex <
              group_begin[set.Group + 1u] - group_begin[set.Group];
      const bool chosen =
          default_count == 0 ||
          std::binary_search(buffers.default_cpu_sets.begin(),
                             buffers.default_cpu_sets.begin() + default_count,
                             set.Id);
      const bool mask_ok =
          !use_affinity_mask ||
          (set.LogicalProcessorIndex < sizeof(DWORD_PTR) * 8u &&
           (process_mask &
            (DWORD_PTR{1} << set.LogicalProcessorIndex)) != 0);
      if (valid_index && chosen && mask_ok &&
          (!set.Allocated || set.AllocatedToTargetProcess) &&
          set.Group < group_count) {
        const unsigned flat =
            group_begin[set.Group] + set.LogicalProcessorIndex;
        if (flat >= buffers.selected.size()) {
          return std::unexpected(hyle::Status::Unavailable(
              "available CPU id expanded during snapshot"));
        }
        buffers.selected[flat] =
            static_cast<std::uint16_t>(set.EfficiencyClass) + 1u;
        maximum_rank = (std::max)(
            maximum_rank,
            static_cast<unsigned>(set.EfficiencyClass) + 1u);
        ++out.logical_cpus;
      }
    }
    offset += info->Size;
  }
  if (out.logical_cpus == 0) {
    return std::unexpected(hyle::Status::FailedPrecondition(
        "process has no usable CPU set"));
  }

  for (std::uint16_t rank : buffers.selected) {
    if (rank == 0) continue;
    if (rank == maximum_rank) ++out.performance_cpus;
    else ++out.efficiency_cpus;
  }
  out.hybrid = out.performance_cpus != 0 && out.efficiency_cpus != 0;

  auto intersects = [&](const GROUP_AFFINITY& mask) noexcept {
    if (mask.Group >= group_count) return false;
    for (unsigned bit = 0; bit < sizeof(KAFFINITY) * 8u; ++bit) {
      if ((mask.Mask & (KAFFINITY{1} << bit)) == 0) continue;
      const unsigned flat = group_begin[mask.Group] + bit;
      if (flat < buffers.selected.size() && buffers.selected[flat] != 0) {
        return true;
      }
    }
    return false;
  };

  DWORD relation_bytes = static_cast<DWORD>(buffers.relations.size());
  if (!::GetLogicalProcessorInformationEx(
          RelationAll,
          reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
              buffers.relations.data()),
          &relation_bytes)) {
    return std::unexpected(hyle::status_from_win32(
        ::GetLastError(), "GetLogicalProcessorInformationEx"));
  }
  if (relation_bytes > buffers.relations.size()) {
    return std::unexpected(hyle::Status::Unavailable(
        "topology relations grew beyond prepared storage"));
  }
  constexpr DWORD kRelationHeader = static_cast<DWORD>(
      offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor));
  for (DWORD offset = 0; offset + kRelationHeader <= relation_bytes;) {
    const auto* relation =
        reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
            buffers.relations.data() + offset);
    if (relation->Size < kRelationHeader ||
        relation->Size > relation_bytes - offset) {
      return std::unexpected(
          hyle::Status::DataLoss("topology relation record is malformed"));
    }
    bool visible = false;
    if (relation->Relationship == RelationProcessorCore ||
        relation->Relationship == RelationProcessorPackage) {
      const std::size_t header =
          offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor) +
          offsetof(PROCESSOR_RELATIONSHIP, GroupMask);
      if (relation->Size < header ||
          relation->Processor.GroupCount >
              (relation->Size - header) / sizeof(GROUP_AFFINITY)) {
        return std::unexpected(hyle::Status::DataLoss(
            "processor relation record is malformed"));
      }
      for (WORD group = 0; group < relation->Processor.GroupCount; ++group) {
        visible = visible || intersects(relation->Processor.GroupMask[group]);
      }
      if (visible && relation->Relationship == RelationProcessorCore) {
        ++out.physical_cores;
      } else if (visible) {
        ++out.packages;
      }
    } else if (relation->Relationship == RelationNumaNode ||
               relation->Relationship == RelationNumaNodeEx) {
      const WORD masks = relation->NumaNode.GroupCount == 0
                             ? 1
                             : relation->NumaNode.GroupCount;
      const std::size_t header =
          offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, NumaNode) +
          offsetof(NUMA_NODE_RELATIONSHIP, GroupMasks);
      if (relation->Size < header ||
          masks > (relation->Size - header) / sizeof(GROUP_AFFINITY)) {
        return std::unexpected(
            hyle::Status::DataLoss("NUMA relation record is malformed"));
      }
      for (WORD group = 0; group < masks; ++group) {
        visible = visible || intersects(relation->NumaNode.GroupMasks[group]);
      }
      if (visible) ++out.numa_nodes;
    } else if (relation->Relationship == RelationCache &&
               (relation->Cache.Level == 2 || relation->Cache.Level == 3)) {
      const WORD masks = relation->Cache.GroupCount == 0
                             ? 1
                             : relation->Cache.GroupCount;
      const std::size_t header =
          offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Cache) +
          offsetof(CACHE_RELATIONSHIP, GroupMasks);
      if (relation->Size < header ||
          masks > (relation->Size - header) / sizeof(GROUP_AFFINITY)) {
        return std::unexpected(
            hyle::Status::DataLoss("cache relation record is malformed"));
      }
      for (WORD group = 0; group < masks; ++group) {
        visible = visible || intersects(relation->Cache.GroupMasks[group]);
      }
      if (visible && relation->Cache.Level == 2) {
        ++out.l2_cache_domains;
      } else if (visible) {
        ++out.l3_cache_domains;
      }
    }
    offset += relation->Size;
  }

  return out;
}

hyle::Result<hyle::platform::HostMemoryInfo> raw_memory_snapshot() noexcept {
  MEMORYSTATUSEX native{};
  native.dwLength = sizeof(native);
  if (!::GlobalMemoryStatusEx(&native)) {
    return std::unexpected(hyle::status_from_win32(
        ::GetLastError(), "GlobalMemoryStatusEx"));
  }
  hyle::platform::HostMemoryInfo out;
  out.physical_total_bytes = native.ullTotalPhys;
  out.physical_available_bytes = native.ullAvailPhys;
  out.commit_limit_bytes = native.ullTotalPageFile;
  out.commit_available_bytes = native.ullAvailPageFile;
  return out;
}

hyle::Result<HostSnapshot> raw_one_shot_snapshot() noexcept {
  // Match available_cpu_topology()'s public ownership contract: the caller
  // supplies no persistent OS scratch state, so discovery, workspace lifetime,
  // and teardown all belong to this snapshot. A pre-sized RawBuffers peer is a
  // different reusable-workspace API and must not be compared to the one-shot
  // HYLE entrypoint.
  RawBuffers buffers;
  hyle::Status initialized = buffers.initialize();
  if (!initialized.ok()) return std::unexpected(std::move(initialized));
  auto topology = raw_topology_snapshot(buffers);
  if (!topology) return std::unexpected(std::move(topology).error());
  auto memory = raw_memory_snapshot();
  if (!memory) return std::unexpected(std::move(memory).error());
  return HostSnapshot{*topology, *memory};
}

hyle::Result<HostSnapshot> hyle_snapshot() noexcept {
  auto topology = hyle::platform::available_cpu_topology();
  if (!topology) return std::unexpected(std::move(topology).error());
  auto memory = hyle::platform::host_memory_info();
  if (!memory) return std::unexpected(std::move(memory).error());
  return HostSnapshot{*topology, *memory};
}

std::uint64_t checksum_optional(
    const std::optional<std::uint64_t>& value,
    unsigned shift) noexcept {
  const std::uint64_t present = value.has_value() ? 1u : 0u;
  return (present << shift) ^
         (value.value_or(0) * (std::uint64_t{0x9e3779b97f4a7c15} + shift));
}

std::uint64_t checksum(const HostSnapshot& snapshot) noexcept {
  const auto& value = snapshot.topology;
  const auto& memory = snapshot.memory;
  return memory.physical_total_bytes ^
         (memory.physical_available_bytes * std::uint64_t{3}) ^
         checksum_optional(memory.commit_limit_bytes, 8u) ^
         checksum_optional(memory.commit_available_bytes, 9u) ^
         checksum_optional(memory.swap_total_bytes, 10u) ^
         checksum_optional(memory.swap_available_bytes, 11u) ^
         checksum_optional(memory.process_limit_bytes, 12u) ^
         checksum_optional(memory.process_available_bytes, 13u) ^
         value.logical_cpus ^
         (static_cast<std::uint64_t>(value.physical_cores) << 7u) ^
         (static_cast<std::uint64_t>(value.packages) << 14u) ^
         (static_cast<std::uint64_t>(value.numa_nodes) << 21u) ^
         (static_cast<std::uint64_t>(value.l2_cache_domains) << 28u) ^
         (static_cast<std::uint64_t>(value.l3_cache_domains) << 35u) ^
         (static_cast<std::uint64_t>(value.performance_cpus) << 42u) ^
         (static_cast<std::uint64_t>(value.efficiency_cpus) << 49u);
}

bool snapshots_compatible(const HostSnapshot& lhs,
                          const HostSnapshot& rhs) noexcept {
  const auto valid_memory = [](const hyle::platform::HostMemoryInfo& value) {
    return value.physical_total_bytes != 0 &&
           value.physical_available_bytes <= value.physical_total_bytes &&
           (!value.commit_limit_bytes ||
            !value.commit_available_bytes ||
            *value.commit_available_bytes <= *value.commit_limit_bytes);
  };
  return lhs.topology.logical_cpus == rhs.topology.logical_cpus &&
         lhs.topology.physical_cores == rhs.topology.physical_cores &&
         lhs.topology.packages == rhs.topology.packages &&
         lhs.topology.numa_nodes == rhs.topology.numa_nodes &&
         lhs.topology.l2_cache_domains == rhs.topology.l2_cache_domains &&
         lhs.topology.l3_cache_domains == rhs.topology.l3_cache_domains &&
         lhs.topology.performance_cpus == rhs.topology.performance_cpus &&
         lhs.topology.efficiency_cpus == rhs.topology.efficiency_cpus &&
         lhs.topology.hybrid == rhs.topology.hybrid &&
         lhs.memory.physical_total_bytes == rhs.memory.physical_total_bytes &&
         lhs.memory.commit_limit_bytes == rhs.memory.commit_limit_bytes &&
         lhs.memory.swap_total_bytes.has_value() ==
             rhs.memory.swap_total_bytes.has_value() &&
         lhs.memory.swap_available_bytes.has_value() ==
             rhs.memory.swap_available_bytes.has_value() &&
         lhs.memory.process_limit_bytes.has_value() ==
             rhs.memory.process_limit_bytes.has_value() &&
         lhs.memory.process_available_bytes.has_value() ==
             rhs.memory.process_available_bytes.has_value() &&
         valid_memory(lhs.memory) && valid_memory(rhs.memory);
}

}  // namespace

int main() {
  const auto affinity = hyle::bench::perf::configure_thread_affinity(
      "HYLE_PERF_CALLER_CPU");
  const bool peer_first = environment_present("HYLE_BENCH_RIVAL_FIRST");
  std::fprintf(stderr,
               "# HYLE_PLATFORM_HOST_RUN execution_order=%s "
               "affinity_topology=%s caller_cpu=%s affinity_ok=%s "
               "run_nonce=%s\n",
               peer_first ? "peer-first" : "candidate-first",
               affinity.topology, affinity.cpu, affinity.ok,
               hyle::bench::perf::run_nonce());

  const auto candidate = hyle_snapshot();
  const auto peer = raw_one_shot_snapshot();
  require(candidate.has_value(), "available_cpu_topology/host_memory_info");
  require(peer.has_value(), "Win32 topology/host-memory adapter");
  require(snapshots_compatible(*candidate, *peer),
          "raw/HYLE host snapshot mismatch");

  nb::Bench bench;
  bench.title("one-shot host planning snapshot")
      .relative(true)
      .batch(1)
      .minEpochTime(std::chrono::milliseconds(8));
  run_pair(bench, peer_first, "HYLE topology + memory snapshot",
           [&] {
             auto snapshot = hyle_snapshot();
             require(snapshot.has_value(),
                     "available_cpu_topology/host_memory_info");
             nb::doNotOptimizeAway(checksum(*snapshot));
           },
           "Win32 equivalent snapshot",
           [&] {
             auto snapshot = raw_one_shot_snapshot();
             require(snapshot.has_value(), "Win32 host snapshot");
             nb::doNotOptimizeAway(checksum(*snapshot));
           });
}

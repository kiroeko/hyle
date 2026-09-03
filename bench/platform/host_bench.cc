#include <nanobench.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <utility>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>

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
  std::uint64_t effective_available = 0;
};

struct RawBuffers {
  std::vector<ULONG> default_cpu_sets;
  std::vector<std::byte> cpu_sets;
  std::vector<std::byte> relations;
  std::vector<unsigned char> selected;

  RawBuffers() {
    const DWORD logical = ::GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    require(logical != 0, "GetActiveProcessorCount");
    default_cpu_sets.resize(logical);
    selected.resize(logical);

    DWORD bytes = 0;
    (void)::GetSystemCpuSetInformation(nullptr, 0, &bytes,
                                       ::GetCurrentProcess(), 0);
    require(bytes != 0, "GetSystemCpuSetInformation(size)");
    cpu_sets.resize(bytes);

    bytes = 0;
    const BOOL unexpected =
        ::GetLogicalProcessorInformationEx(RelationAll, nullptr, &bytes);
    require(!unexpected && ::GetLastError() == ERROR_INSUFFICIENT_BUFFER &&
                bytes != 0,
            "GetLogicalProcessorInformationEx(size)");
    relations.resize(bytes);
  }
};

std::uint64_t remaining(std::uint64_t limit, std::uint64_t used) noexcept {
  return used < limit ? limit - used : 0;
}

HostSnapshot raw_snapshot(RawBuffers& buffers) {
  HostSnapshot out;
  HANDLE process = ::GetCurrentProcess();

  ULONG default_count = 0;
  ::SetLastError(ERROR_SUCCESS);
  const BOOL defaults_ok =
      ::GetProcessDefaultCpuSets(process, nullptr, 0, &default_count);
  require(defaults_ok || ::GetLastError() == ERROR_INSUFFICIENT_BUFFER,
          "GetProcessDefaultCpuSets(size)");
  require(default_count <= buffers.default_cpu_sets.size(),
          "default CPU set grew beyond prepared storage");
  if (default_count != 0) {
    ULONG actual = default_count;
    require(::GetProcessDefaultCpuSets(process, buffers.default_cpu_sets.data(),
                                       default_count, &actual) != 0,
            "GetProcessDefaultCpuSets");
    default_count = actual;
    std::sort(buffers.default_cpu_sets.begin(),
              buffers.default_cpu_sets.begin() + default_count);
  }

  DWORD cpu_set_bytes = static_cast<DWORD>(buffers.cpu_sets.size());
  require(::GetSystemCpuSetInformation(
              reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(
                  buffers.cpu_sets.data()),
              cpu_set_bytes, &cpu_set_bytes, process, 0) != 0,
          "GetSystemCpuSetInformation");
  require(cpu_set_bytes <= buffers.cpu_sets.size(),
          "CPU set data grew beyond prepared storage");

  const WORD group_count = ::GetActiveProcessorGroupCount();
  require(group_count != 0 && group_count <= 64,
          "active processor group count");
  std::array<unsigned, 65> group_begin{};
  for (WORD group = 0; group < group_count; ++group) {
    group_begin[group + 1] =
        group_begin[group] + ::GetActiveProcessorCount(group);
  }
  require(group_begin[group_count] <= buffers.selected.size(),
          "logical CPU set grew beyond prepared storage");
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
    require(info->Size != 0 && info->Size <= cpu_set_bytes - offset,
            "CPU set record");
    if (info->Type == CpuSetInformation) {
      const auto& set = info->CpuSet;
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
      if (chosen && mask_ok &&
          (!set.Allocated || set.AllocatedToTargetProcess) &&
          set.Group < group_count) {
        const unsigned flat =
            group_begin[set.Group] + set.LogicalProcessorIndex;
        if (flat < buffers.selected.size()) {
          buffers.selected[flat] =
              static_cast<unsigned char>(set.EfficiencyClass + 1u);
          maximum_rank = (std::max)(
              maximum_rank,
              static_cast<unsigned>(set.EfficiencyClass) + 1u);
          ++out.topology.logical_cpus;
        }
      }
    }
    offset += info->Size;
  }
  require(out.topology.logical_cpus != 0, "usable CPU set");

  for (unsigned char rank : buffers.selected) {
    if (rank == 0) continue;
    if (rank == maximum_rank) ++out.topology.performance_cpus;
    else ++out.topology.efficiency_cpus;
  }
  out.topology.hybrid = out.topology.performance_cpus != 0 &&
                        out.topology.efficiency_cpus != 0;

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
  require(::GetLogicalProcessorInformationEx(
              RelationAll,
              reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
                  buffers.relations.data()),
              &relation_bytes) != 0,
          "GetLogicalProcessorInformationEx");
  require(relation_bytes <= buffers.relations.size(),
          "topology relations grew beyond prepared storage");
  constexpr DWORD kRelationHeader = static_cast<DWORD>(
      offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor));
  for (DWORD offset = 0; offset + kRelationHeader <= relation_bytes;) {
    const auto* relation =
        reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
            buffers.relations.data() + offset);
    if (relation->Size < kRelationHeader ||
        relation->Size > relation_bytes - offset) {
      std::fprintf(stderr,
                   "# invalid relation offset=%lu size=%lu struct=%zu "
                   "remaining=%lu kind=%u\n",
                   static_cast<unsigned long>(offset),
                   static_cast<unsigned long>(relation->Size),
                   static_cast<std::size_t>(kRelationHeader),
                   static_cast<unsigned long>(relation_bytes - offset),
                   static_cast<unsigned>(relation->Relationship));
      fail("topology relation record");
    }
    bool visible = false;
    if (relation->Relationship == RelationProcessorCore ||
        relation->Relationship == RelationProcessorPackage) {
      const std::size_t header =
          offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor) +
          offsetof(PROCESSOR_RELATIONSHIP, GroupMask);
      require(relation->Size >= header &&
                  relation->Processor.GroupCount <=
                      (relation->Size - header) / sizeof(GROUP_AFFINITY),
              "processor relation size");
      for (WORD group = 0; group < relation->Processor.GroupCount; ++group) {
        visible = visible || intersects(relation->Processor.GroupMask[group]);
      }
      if (visible && relation->Relationship == RelationProcessorCore) {
        ++out.topology.physical_cores;
      } else if (visible) {
        ++out.topology.packages;
      }
    } else if (relation->Relationship == RelationNumaNode ||
               relation->Relationship == RelationNumaNodeEx) {
      const WORD masks = relation->NumaNode.GroupCount == 0
                             ? 1
                             : relation->NumaNode.GroupCount;
      const std::size_t header =
          offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, NumaNode) +
          offsetof(NUMA_NODE_RELATIONSHIP, GroupMasks);
      require(relation->Size >= header &&
                  masks <=
                      (relation->Size - header) / sizeof(GROUP_AFFINITY),
              "NUMA relation size");
      for (WORD group = 0; group < masks; ++group) {
        visible = visible || intersects(relation->NumaNode.GroupMasks[group]);
      }
      if (visible) ++out.topology.numa_nodes;
    } else if (relation->Relationship == RelationCache &&
               (relation->Cache.Level == 2 || relation->Cache.Level == 3)) {
      const WORD masks = relation->Cache.GroupCount == 0
                             ? 1
                             : relation->Cache.GroupCount;
      const std::size_t header =
          offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Cache) +
          offsetof(CACHE_RELATIONSHIP, GroupMasks);
      require(relation->Size >= header &&
                  masks <=
                      (relation->Size - header) / sizeof(GROUP_AFFINITY),
              "cache relation size");
      for (WORD group = 0; group < masks; ++group) {
        visible = visible || intersects(relation->Cache.GroupMasks[group]);
      }
      if (visible && relation->Cache.Level == 2) {
        ++out.topology.l2_cache_domains;
      } else if (visible) {
        ++out.topology.l3_cache_domains;
      }
    }
    offset += relation->Size;
  }

  MEMORYSTATUSEX memory{};
  memory.dwLength = sizeof(memory);
  require(::GlobalMemoryStatusEx(&memory) != 0, "GlobalMemoryStatusEx");
  out.effective_available =
      (std::min)(memory.ullAvailPhys, memory.ullAvailPageFile);

  BOOL in_job = FALSE;
  if (::IsProcessInJob(process, nullptr, &in_job) && in_job) {
    std::uint64_t process_commit = 0;
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (::GetProcessMemoryInfo(
            process, reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
            sizeof(counters))) {
      process_commit = counters.PrivateUsage;
    }
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    if (::QueryInformationJobObject(nullptr, JobObjectExtendedLimitInformation,
                                    &limits, sizeof(limits), nullptr)) {
      const DWORD flags = limits.BasicLimitInformation.LimitFlags;
      if ((flags & JOB_OBJECT_LIMIT_PROCESS_MEMORY) != 0) {
        const auto used = process_commit != 0
                              ? process_commit
                              : limits.PeakProcessMemoryUsed;
        out.effective_available =
            (std::min)(out.effective_available,
                       remaining(limits.ProcessMemoryLimit, used));
      }
      if ((flags & JOB_OBJECT_LIMIT_JOB_MEMORY) != 0) {
        std::uint64_t used = limits.PeakJobMemoryUsed;
        JOBOBJECT_LIMIT_VIOLATION_INFORMATION_2 live{};
        if (::QueryInformationJobObject(
                nullptr, JobObjectLimitViolationInformation2, &live,
                sizeof(live), nullptr)) {
          used = live.JobMemory;
        }
        out.effective_available =
            (std::min)(out.effective_available,
                       remaining(limits.JobMemoryLimit, used));
      }
    }
  }
  return out;
}

HostSnapshot raw_one_shot_snapshot() {
  // Match available_cpu_topology()'s public ownership contract: the caller
  // supplies no persistent OS scratch state, so discovery, workspace lifetime,
  // and teardown all belong to this snapshot. A pre-sized RawBuffers peer is a
  // different reusable-workspace API and must not be compared to the one-shot
  // HYLE entrypoint.
  RawBuffers buffers;
  return raw_snapshot(buffers);
}

HostSnapshot hyle_snapshot() {
  auto topology = hyle::platform::available_cpu_topology();
  auto memory = hyle::platform::host_memory_info();
  require(topology.has_value(), "available_cpu_topology");
  require(memory.has_value(), "host_memory_info");
  return {*topology, memory->effective_available_bytes()};
}

std::uint64_t checksum(const HostSnapshot& snapshot) noexcept {
  const auto& value = snapshot.topology;
  return snapshot.effective_available ^ value.logical_cpus ^
         (static_cast<std::uint64_t>(value.physical_cores) << 7u) ^
         (static_cast<std::uint64_t>(value.packages) << 14u) ^
         (static_cast<std::uint64_t>(value.numa_nodes) << 21u) ^
         (static_cast<std::uint64_t>(value.l2_cache_domains) << 28u) ^
         (static_cast<std::uint64_t>(value.l3_cache_domains) << 35u) ^
         (static_cast<std::uint64_t>(value.performance_cpus) << 42u) ^
         (static_cast<std::uint64_t>(value.efficiency_cpus) << 49u);
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

  RawBuffers buffers;
  const HostSnapshot candidate = hyle_snapshot();
  const HostSnapshot peer = raw_snapshot(buffers);
  require(candidate.topology.logical_cpus == peer.topology.logical_cpus &&
              candidate.topology.physical_cores ==
                  peer.topology.physical_cores &&
              candidate.topology.packages == peer.topology.packages &&
              candidate.topology.numa_nodes == peer.topology.numa_nodes &&
              candidate.topology.l2_cache_domains ==
                  peer.topology.l2_cache_domains &&
              candidate.topology.l3_cache_domains ==
                  peer.topology.l3_cache_domains &&
              candidate.topology.performance_cpus ==
                  peer.topology.performance_cpus &&
              candidate.topology.efficiency_cpus ==
                  peer.topology.efficiency_cpus,
          "raw/HYLE topology mismatch");

  nb::Bench bench;
  bench.title("one-shot host planning snapshot")
      .relative(true)
      .batch(1)
      .minEpochTime(std::chrono::milliseconds(8));
  run_pair(bench, peer_first, "HYLE topology + memory snapshot",
           [&] { nb::doNotOptimizeAway(checksum(hyle_snapshot())); },
           "Win32 equivalent snapshot",
           [&] {
             nb::doNotOptimizeAway(checksum(raw_one_shot_snapshot()));
           });
}

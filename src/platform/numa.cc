#include "hyle/platform/numa.h"

#include "numa_internal.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>

#include "hyle/core/compiler.h"
#include "hyle/platform.h"
#include "hyle/platform/cpu.h"
#include "hyle/platform/virtual_memory.h"

#if defined(HYLE_OS_WINDOWS)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <errno.h>
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/syscall.h>
#  include <unistd.h>
#endif

namespace hyle::platform {
namespace {

#if defined(HYLE_OS_WINDOWS)
Status numa_native_status(const char* operation, DWORD code) noexcept {
  return status_from_win32(code, operation);
}
#elif defined(HYLE_OS_LINUX)
Status numa_native_status(const char* operation, int code) noexcept {
  return status_from_errno(code, operation);
}
#endif

#if defined(HYLE_OS_WINDOWS)

using PGetNumaHighestNodeNumber = BOOL(WINAPI*)(PULONG);
using PGetCurrentProcessorNumberEx = void(WINAPI*)(PROCESSOR_NUMBER*);
using PGetNumaProcessorNodeEx = BOOL(WINAPI*)(PPROCESSOR_NUMBER, PUSHORT);
using PGetNumaAvailableMemoryNodeEx = BOOL(WINAPI*)(USHORT, PULONGLONG);
using PVirtualAllocExNuma = LPVOID(WINAPI*)(HANDLE, LPVOID, SIZE_T, DWORD,
                                            DWORD, DWORD);

#  pragma warning(push)
#  pragma warning(disable : 4201)
struct HyleMemExtendedParameter {
  struct {
    DWORD64 Type : 8;
    DWORD64 Reserved : 56;
  };
  union {
    DWORD64 ULong64;
    PVOID Pointer;
    SIZE_T Size;
    HANDLE Handle;
    DWORD ULong;
  };
};
#  pragma warning(pop)

enum { kMemExtendedParameterNumaNode = 2 };

using PVirtualAlloc2 = PVOID(WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG,
                                      HyleMemExtendedParameter*, ULONG);

struct WinNumaFns {
  PGetNumaHighestNodeNumber get_highest = nullptr;
  PGetCurrentProcessorNumberEx get_cpu_ex = nullptr;
  PGetNumaProcessorNodeEx get_node_ex = nullptr;
  PGetNumaAvailableMemoryNodeEx get_memory_ex = nullptr;
  PVirtualAllocExNuma virtual_alloc_ex_numa = nullptr;
  PVirtualAlloc2 virtual_alloc2 = nullptr;
};

WinNumaFns& win_fns() {
  static WinNumaFns f = [] {
    WinNumaFns x;
    HMODULE k32 = ::GetModuleHandleW(L"kernel32.dll");
    if (k32) {
      x.get_highest =
          reinterpret_cast<PGetNumaHighestNodeNumber>(reinterpret_cast<void*>(
              ::GetProcAddress(k32, "GetNumaHighestNodeNumber")));
      x.get_cpu_ex = reinterpret_cast<PGetCurrentProcessorNumberEx>(
          reinterpret_cast<void*>(
              ::GetProcAddress(k32, "GetCurrentProcessorNumberEx")));
      x.get_node_ex =
          reinterpret_cast<PGetNumaProcessorNodeEx>(reinterpret_cast<void*>(
              ::GetProcAddress(k32, "GetNumaProcessorNodeEx")));
      x.get_memory_ex = reinterpret_cast<PGetNumaAvailableMemoryNodeEx>(
          reinterpret_cast<void*>(
              ::GetProcAddress(k32, "GetNumaAvailableMemoryNodeEx")));
      x.virtual_alloc_ex_numa = reinterpret_cast<PVirtualAllocExNuma>(
          reinterpret_cast<void*>(::GetProcAddress(k32, "VirtualAllocExNuma")));
      x.virtual_alloc2 = reinterpret_cast<PVirtualAlloc2>(
          reinterpret_cast<void*>(::GetProcAddress(k32, "VirtualAlloc2")));
      if (!x.virtual_alloc2) {
        x.virtual_alloc2 =
            reinterpret_cast<PVirtualAlloc2>(reinterpret_cast<void*>(
                ::GetProcAddress(k32, "VirtualAlloc2FromApp")));
      }
    }
    return x;
  }();
  return f;
}

// A selected-node allocation must retain the public one-owner contract:
// callers release the complete contiguous range with one vm_release call.
// VirtualAllocExNuma commits stripes inside one ordinary reservation and keeps
// that ownership shape. VirtualAlloc2 placeholder replacement does not: every
// replaced stripe becomes a separate allocation that must be released and
// coalesced individually. Advertising that path would make both rollback and
// the public raw/RAII/C owners leak all but one stripe.
constexpr bool win_selected_interleave_supported(
    bool has_virtual_alloc_ex_numa, bool has_virtual_alloc2) noexcept {
  (void)has_virtual_alloc2;
  return has_virtual_alloc_ex_numa;
}

#elif defined(HYLE_OS_LINUX)

// linux/mempolicy.h constants (avoid header dependency).
#  ifndef MPOL_DEFAULT
#    define MPOL_DEFAULT 0
#  endif
#  ifndef MPOL_PREFERRED
#    define MPOL_PREFERRED 1
#  endif
#  ifndef MPOL_BIND
#    define MPOL_BIND 2
#  endif
#  ifndef MPOL_INTERLEAVE
#    define MPOL_INTERLEAVE 3
#  endif
#  ifndef MPOL_F_MEMS_ALLOWED
#    define MPOL_F_MEMS_ALLOWED (1UL << 2)
#  endif

long sys_mbind(void* start, unsigned long len, unsigned long mode,
               const unsigned long* nmask, unsigned long maxnode,
               unsigned flags) {
#  if defined(SYS_mbind)
  return ::syscall(SYS_mbind, start, len, mode, nmask, maxnode, flags);
#  else
  (void)start;
  (void)len;
  (void)mode;
  (void)nmask;
  (void)maxnode;
  (void)flags;
  errno = ENOSYS;
  return -1;
#  endif
}

long sys_move_pages(unsigned long count, void** pages, const int* nodes,
                    int* status, int flags) {
#  if defined(SYS_move_pages)
  return ::syscall(SYS_move_pages, 0, count, pages, nodes, status, flags);
#  else
  (void)count;
  (void)pages;
  (void)nodes;
  (void)status;
  (void)flags;
  errno = ENOSYS;
  return -1;
#  endif
}

bool path_exists(const char* path) { return ::access(path, R_OK) == 0; }

#endif

// Linux supports up to 1024 NUMA nodes in common kernel configurations;
// reserve headroom without putting allocation or exception machinery in this
// platform-bottom topology path.
inline constexpr std::size_t kMaxNumaNodes = 4096;

inline constexpr unsigned kTopologyWordBits = sizeof(std::uint64_t) * 8u;
using NumaTopologyMask =
    std::array<std::uint64_t,
               (kMaxNumaNodes + kTopologyWordBits - 1u) / kTopologyWordBits>;

#if defined(HYLE_OS_LINUX)
inline constexpr unsigned kMaskBits = sizeof(unsigned long) * 8u;
using NumaMask =
    std::array<unsigned long, (kMaxNumaNodes + kMaskBits - 1u) / kMaskBits>;
#endif

struct NumaTopology {
  std::array<unsigned, kMaxNumaNodes> ids{};
  NumaTopologyMask present{};
  std::size_t count = 0;
  bool native = false;
};

void append_node(NumaTopology& topology, unsigned node) noexcept {
  if (node >= kMaxNumaNodes || topology.count >= topology.ids.size()) return;
  const std::uint64_t bit = std::uint64_t{1} << (node % kTopologyWordBits);
  std::uint64_t& word = topology.present[node / kTopologyWordBits];
  if ((word & bit) != 0) return;
  word |= bit;
  topology.ids[topology.count++] = node;
}

#if defined(HYLE_OS_LINUX)
bool read_linux_allowed_nodes(NumaTopology& topology) noexcept {
#  if defined(SYS_get_mempolicy)
  NumaMask allowed{};
  int mode = MPOL_DEFAULT;
  if (::syscall(SYS_get_mempolicy, &mode, allowed.data(), kMaxNumaNodes,
                nullptr, MPOL_F_MEMS_ALLOWED) != 0) {
    return false;
  }
  for (unsigned node = 0; node < kMaxNumaNodes; ++node) {
    if ((allowed[node / kMaskBits] & (1UL << (node % kMaskBits))) != 0) {
      append_node(topology, node);
    }
  }
  return topology.count != 0;
#  else
  (void)topology;
  return false;
#  endif
}

bool read_linux_online_nodes(NumaTopology& topology) noexcept {
  int fd = ::open("/sys/devices/system/node/online", O_RDONLY);
  if (fd < 0) return false;
  char text[4096]{};
  const ssize_t length = ::read(fd, text, sizeof(text) - 1);
  ::close(fd);
  if (length <= 0) return false;

  const char* cursor = text;
  const char* end = text + length;
  while (cursor < end) {
    while (cursor < end && (*cursor == ',' || *cursor == ' ' ||
                            *cursor == '\n' || *cursor == '\t')) {
      ++cursor;
    }
    if (cursor == end) break;
    if (*cursor < '0' || *cursor > '9') return false;
    unsigned first = 0;
    while (cursor < end && *cursor >= '0' && *cursor <= '9') {
      const unsigned digit = static_cast<unsigned>(*cursor++ - '0');
      if (first > (std::numeric_limits<unsigned>::max() - digit) / 10u) {
        return false;
      }
      first = first * 10u + digit;
    }
    unsigned last = first;
    if (cursor < end && *cursor == '-') {
      ++cursor;
      if (cursor == end || *cursor < '0' || *cursor > '9') return false;
      last = 0;
      while (cursor < end && *cursor >= '0' && *cursor <= '9') {
        const unsigned digit = static_cast<unsigned>(*cursor++ - '0');
        if (last > (std::numeric_limits<unsigned>::max() - digit) / 10u) {
          return false;
        }
        last = last * 10u + digit;
      }
      if (last < first) return false;
    }
    if (last >= kMaxNumaNodes ||
        last - first >= kMaxNumaNodes - topology.count) {
      return false;
    }
    for (unsigned node = first;; ++node) {
      append_node(topology, node);
      if (node == last) break;
    }
    if (cursor < end && *cursor != ',' && *cursor != ' ' && *cursor != '\n' &&
        *cursor != '\t') {
      return false;
    }
  }
  return topology.count != 0;
}
#endif

NumaTopology discover_topology() noexcept {
  NumaTopology topology;
#if defined(HYLE_OS_WINDOWS)
  ULONG highest = 0;
  auto& f = win_fns();
  if (f.get_highest && f.get_highest(&highest)) {
    const ULONG limit =
        highest < static_cast<ULONG>((std::numeric_limits<USHORT>::max)())
            ? highest
            : static_cast<ULONG>((std::numeric_limits<USHORT>::max)());
    for (ULONG node = 0; node <= limit && topology.count < kMaxNumaNodes;
         ++node) {
      ULONGLONG available = 0;
      if (!f.get_memory_ex ||
          f.get_memory_ex(static_cast<USHORT>(node), &available)) {
        append_node(topology, static_cast<unsigned>(node));
      }
    }
    topology.native = topology.count != 0;
  }
#elif defined(HYLE_OS_LINUX)
  // Prefer the process' current cpuset/mempolicy view. This avoids advertising
  // a system node that the process cannot legally target inside a container.
  topology.native = read_linux_allowed_nodes(topology);
  if (!topology.native) {
    topology = NumaTopology{};
    topology.native = read_linux_online_nodes(topology);
  }
  if (!topology.native) {
    // A malformed/truncated online list may have appended a valid prefix.
    // Discard it before the independent directory-scan fallback.
    topology = NumaTopology{};
    // Conservative fallback for restricted containers without the `online`
    // file. Do not stop at the first hole: NUMA IDs need not be dense.
    for (unsigned node = 0; node < kMaxNumaNodes; ++node) {
      char path[96];
      std::snprintf(path, sizeof(path), "/sys/devices/system/node/node%u",
                    node);
      if (path_exists(path)) append_node(topology, node);
    }
    topology.native = topology.count != 0;
  }
#endif
  if (topology.count == 0) append_node(topology, 0);
  return topology;
}

const NumaTopology& topology() noexcept {
  static const NumaTopology value = discover_topology();
  return value;
}

bool topology_has_node(unsigned node) noexcept {
  const auto& value = topology();
  if (node >= kMaxNumaNodes) return false;
  return (value.present[node / kTopologyWordBits] &
          (std::uint64_t{1} << (node % kTopologyWordBits))) != 0;
}

Result<bool> process_cpu_available(unsigned cpu) noexcept {
  unsigned stack[256];
  for (unsigned attempt = 0; attempt < 4; ++attempt) {
    auto count = available_cpu_ids(stack, std::size(stack));
    if (!count) return std::unexpected(std::move(count).error());
    const std::size_t stack_count =
        *count < std::size(stack) ? *count : std::size(stack);
    for (std::size_t i = 0; i < stack_count; ++i) {
      if (stack[i] == cpu) return true;
    }
    if (*count <= std::size(stack)) return false;

    auto* ids = new (std::nothrow) unsigned[*count];
    if (ids == nullptr) {
      return std::unexpected(Status::ResourceExhausted(
          "NUMA CPU lookup allocation failed"));
    }
    auto filled = available_cpu_ids(ids, *count);
    if (!filled) {
      delete[] ids;
      return std::unexpected(std::move(filled).error());
    }
    if (*filled > *count) {
      delete[] ids;
      continue;
    }
    bool found = false;
    for (std::size_t i = 0; i < *filled; ++i) found |= ids[i] == cpu;
    delete[] ids;
    return found;
  }
  return std::unexpected(Status::Unavailable(
      "NUMA CPU lookup changed repeatedly during snapshot"));
}

#if defined(HYLE_OS_LINUX)
Result<unsigned> make_numa_mask(std::span<const unsigned> nodes,
                                NumaMask& mask) noexcept {
  unsigned max_node = 0;
  for (unsigned node : nodes) {
    if (node >= kMaxNumaNodes) {
      return std::unexpected(
          Status::OutOfRange("NUMA node exceeds policy mask capacity"));
    }
    mask[node / kMaskBits] |= 1UL << (node % kMaskBits);
    if (node > max_node) max_node = node;
  }
  return max_node + 1u;
}
#endif

}  // namespace

#if defined(HYLE_OS_WINDOWS) && defined(HYLE_TESTING)
namespace detail {
bool windows_selected_interleave_supported_for_testing(
    bool has_virtual_alloc_ex_numa, bool has_virtual_alloc2) noexcept {
  return win_selected_interleave_supported(has_virtual_alloc_ex_numa,
                                            has_virtual_alloc2);
}
}  // namespace detail
#endif

NumaCapability numa_capabilities() noexcept {
  NumaCapability capabilities = NumaCapability::kNone;
  if (topology().native)
    capabilities = capabilities | NumaCapability::kTopology;
#if defined(HYLE_OS_WINDOWS)
  auto& f = win_fns();
  if (f.virtual_alloc2 || f.virtual_alloc_ex_numa)
    capabilities = capabilities | NumaCapability::kPreferredReservation;
  if (win_selected_interleave_supported(f.virtual_alloc_ex_numa != nullptr,
                                        f.virtual_alloc2 != nullptr))
    capabilities = capabilities | NumaCapability::kSelectedInterleave;
#elif defined(HYLE_OS_LINUX) && defined(SYS_mbind)
  capabilities = capabilities | NumaCapability::kPreferredReservation |
                 NumaCapability::kBindExisting |
                 NumaCapability::kInterleaveExisting |
                 NumaCapability::kSelectedInterleave;
#endif
  return capabilities;
}

std::span<const unsigned> numa_nodes() noexcept {
  const auto& value = topology();
  return std::span<const unsigned>(value.ids.data(), value.count);
}

bool numa_node_available(unsigned node) noexcept {
  return topology_has_node(node);
}

unsigned numa_node_count() noexcept {
  return static_cast<unsigned>(topology().count);
}

Result<unsigned> numa_current_node() noexcept {
  const auto nodes = numa_nodes();
  if (nodes.size() == 1u) return nodes.front();
#if defined(HYLE_OS_WINDOWS)
  auto& f = win_fns();
  if (!f.get_cpu_ex || !f.get_node_ex) {
    return std::unexpected(Status::Unimplemented(
        "current NUMA node query is unavailable on Windows"));
  }
  PROCESSOR_NUMBER pnum{};
  f.get_cpu_ex(&pnum);
  USHORT node = 0;
  if (!f.get_node_ex(&pnum, &node)) {
    return std::unexpected(numa_native_status(
        "GetNumaProcessorNodeEx(current CPU)", ::GetLastError()));
  }
  const unsigned result = static_cast<unsigned>(node);
  if (!numa_node_available(result)) {
    return std::unexpected(Status::FailedPrecondition(
        "current CPU belongs to a process-unavailable NUMA node"));
  }
  return result;
#elif defined(HYLE_OS_LINUX)
#  if defined(SYS_getcpu)
  unsigned cpu = 0;
  unsigned node = 0;
  if (::syscall(SYS_getcpu, &cpu, &node, nullptr) == 0) {
    if (numa_node_available(node)) return node;
    return std::unexpected(Status::FailedPrecondition(
        "current CPU belongs to a process-unavailable NUMA node"));
  }
  return std::unexpected(numa_native_status("getcpu", errno));
#  else
  return std::unexpected(
      Status::Unimplemented("current NUMA node query is unavailable"));
#  endif
#else
  return nodes.front();
#endif
}

Result<unsigned> numa_node_of_cpu(unsigned cpu) noexcept {
  const auto nodes = numa_nodes();
  auto available = process_cpu_available(cpu);
  if (!available) return std::unexpected(std::move(available).error());
  if (!*available) {
    return std::unexpected(
        Status::OutOfRange("NUMA CPU is unavailable to this process"));
  }
  if (nodes.size() <= 1u) return nodes.front();
#if defined(HYLE_OS_WINDOWS)
  auto& f = win_fns();
  if (!f.get_node_ex) {
    return std::unexpected(Status::Unimplemented(
        "CPU NUMA node query is unavailable on Windows"));
  }
  WORD target_group = 0;
  unsigned number_in_group = cpu;
  const WORD groups = ::GetActiveProcessorGroupCount();
  bool found = false;
  for (WORD group = 0; group < groups; ++group) {
    const DWORD active = ::GetActiveProcessorCount(group);
    if (number_in_group < active) {
      target_group = group;
      found = true;
      break;
    }
    number_in_group -= active;
  }
  if (!found || number_in_group > 0xFFu) {
    return std::unexpected(
        Status::OutOfRange("NUMA CPU is not online"));
  }
  PROCESSOR_NUMBER pnum{};
  pnum.Group = target_group;
  pnum.Number = static_cast<BYTE>(number_in_group);
  USHORT node = 0;
  if (!f.get_node_ex(&pnum, &node)) {
    return std::unexpected(numa_native_status(
        "GetNumaProcessorNodeEx", ::GetLastError()));
  }
  const unsigned result = static_cast<unsigned>(node);
  if (!numa_node_available(result)) {
    return std::unexpected(Status::FailedPrecondition(
        "CPU belongs to a process-unavailable NUMA node"));
  }
  return result;
#elif defined(HYLE_OS_LINUX)
  // sysfs exposes one symlink /sys/devices/system/node/nodeN/cpuM per
  // (node,cpu) membership; find the node whose directory contains this cpu.
  for (unsigned nd : nodes) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "/sys/devices/system/node/node%u/cpu%u", nd,
                  cpu);
    if (path_exists(buf)) return nd;
  }
  return std::unexpected(
      Status::Unavailable("NUMA node for CPU could not be discovered"));
#else
  return nodes.front();
#endif
}

Result<NumaPlacement> vm_bind_numa(void* ptr, std::size_t size, int node,
                                   NumaPolicy policy) noexcept {
  if (policy != NumaPolicy::kPreferred && policy != NumaPolicy::kBind &&
      policy != NumaPolicy::kInterleave) {
    return std::unexpected(
        Status::InvalidArgument("vm bind NUMA: invalid placement policy"));
  }
  if (size == 0) return NumaPlacement::kNotNeeded;
  if (ptr == nullptr) {
    return std::unexpected(
        Status::InvalidArgument("vm bind NUMA: null nonempty range"));
  }
  if ((reinterpret_cast<std::uintptr_t>(ptr) % page_size()) != 0) {
    return std::unexpected(
        Status::InvalidArgument("vm bind NUMA: pointer is not page-aligned"));
  }
  size = page_align_up(size);
  if (size == 0) {
    return std::unexpected(Status::OutOfRange("vm bind NUMA: size overflow"));
  }

  const auto available_nodes = numa_nodes();

  if (policy == NumaPolicy::kInterleave) {
    // Interleave is a genuine no-op on a single-node machine. Linux rejects
    // MPOL_INTERLEAVE with a one-bit nodemask on some kernels.
    if (available_nodes.size() <= 1u) return NumaPlacement::kNotNeeded;
#if defined(HYLE_OS_LINUX)
    NumaMask mask{};
    HYLE_ASSIGN_OR_RETURN(unsigned max_node,
                          make_numa_mask(available_nodes, mask));
    long rc = sys_mbind(ptr, size, MPOL_INTERLEAVE, mask.data(), max_node, 0);
    if (rc != 0) {
      return std::unexpected(
          numa_native_status("mbind(MPOL_INTERLEAVE)", errno));
    }
    return NumaPlacement::kApplied;
#else
    return std::unexpected(Status::Unimplemented(
        "NUMA interleave is unavailable on this platform"));
#endif
  }

  if (node == kNumaAny) return NumaPlacement::kNotNeeded;
  if (node < 0 || !numa_node_available(static_cast<unsigned>(node))) {
    return std::unexpected(
        Status::OutOfRange("vm bind NUMA: node ID is unavailable"));
  }

  // A valid per-node policy is a genuine no-op on a single-node machine.
  if (available_nodes.size() <= 1u) return NumaPlacement::kNotNeeded;

#if defined(HYLE_OS_LINUX)
  const unsigned nd = static_cast<unsigned>(node);
  NumaMask mask{};
  const std::array<unsigned, 1> selected{nd};
  HYLE_ASSIGN_OR_RETURN(unsigned max_node, make_numa_mask(selected, mask));
  const unsigned long mode =
      (policy == NumaPolicy::kBind) ? MPOL_BIND : MPOL_PREFERRED;
  long rc = sys_mbind(ptr, size, mode, mask.data(), max_node, 0);
  if (rc != 0) {
    return std::unexpected(numa_native_status(policy == NumaPolicy::kBind
                                                  ? "mbind(MPOL_BIND)"
                                                  : "mbind(MPOL_PREFERRED)",
                                              errno));
  }
  return NumaPlacement::kApplied;
#elif defined(HYLE_OS_WINDOWS)
  // No post-hoc bind for anonymous VirtualAlloc. Preferred is SOFT (real
  // placement is done at reserve time by vm_reserve_numa via VirtualAlloc2) ->
  // report success. Hard Bind cannot be honored post-hoc on a multi-node box ->
  // honest false (don't pretend).
  if (policy == NumaPolicy::kBind) {
    return std::unexpected(Status::Unimplemented(
        "post-allocation hard NUMA bind is unavailable on Windows; reserve "
        "on the target node instead"));
  }
  return NumaPlacement::kUnavailable;
#else
  // Non-Linux POSIX (e.g. macOS) never reports >1 node today, so this is only
  // reached defensively; hard bind has no API -> honest false.
  if (policy == NumaPolicy::kBind) {
    return std::unexpected(Status::Unimplemented(
        "hard NUMA bind is unavailable on this platform"));
  }
  return NumaPlacement::kUnavailable;
#endif
}

Result<NumaPlacement> vm_bind_numa_interleave(void* ptr,
                                              std::size_t size) noexcept {
  return vm_bind_numa(ptr, size, kNumaAny, NumaPolicy::kInterleave);
}

Result<NumaReservation> vm_reserve_numa(std::size_t size, int node) noexcept {
  if (node == kNumaAny) {
    HYLE_ASSIGN_OR_RETURN(void* data, vm_reserve(size));
    return NumaReservation{data, NumaPlacement::kNotNeeded};
  }
  if (node < 0 || !numa_node_available(static_cast<unsigned>(node))) {
    return std::unexpected(
        Status::OutOfRange("vm reserve NUMA: node ID is unavailable"));
  }
  size = page_align_up(size);
  if (size == 0) {
    return std::unexpected(Status::InvalidArgument(
        "vm reserve NUMA: size is zero or overflows page alignment"));
  }

  if (numa_node_count() <= 1u) {
    HYLE_ASSIGN_OR_RETURN(void* data, vm_reserve(size));
    return NumaReservation{data, NumaPlacement::kNotNeeded};
  }

#if defined(HYLE_OS_WINDOWS)
  auto& f = win_fns();
  if (f.virtual_alloc2) {
    HyleMemExtendedParameter param{};
    param.Type = kMemExtendedParameterNumaNode;
    param.ULong = static_cast<DWORD>(node);
    void* p = f.virtual_alloc2(GetCurrentProcess(), nullptr, size, MEM_RESERVE,
                               PAGE_NOACCESS, &param, 1);
    if (p != nullptr) return NumaReservation{p, NumaPlacement::kApplied};
    // Preferred placement is soft. Preserve a usable allocation and expose
    // that the requested placement was not applied.
  }
  if (f.virtual_alloc_ex_numa) {
    void* p =
        f.virtual_alloc_ex_numa(GetCurrentProcess(), nullptr, size, MEM_RESERVE,
                                PAGE_NOACCESS, static_cast<DWORD>(node));
    if (p != nullptr) return NumaReservation{p, NumaPlacement::kApplied};
  }
  HYLE_ASSIGN_OR_RETURN(void* data, vm_reserve(size));
  return NumaReservation{data, NumaPlacement::kUnavailable};
#elif defined(HYLE_OS_LINUX)
  HYLE_ASSIGN_OR_RETURN(void* data, vm_reserve(size));
  // Soft preferred placement: a bind failure must NOT drop the reservation.
  // The placement result records the fallback without inventing a side channel.
  auto placement = vm_bind_numa(data, size, node, NumaPolicy::kPreferred);
  return NumaReservation{data,
                         placement ? *placement : NumaPlacement::kUnavailable};
#else
  HYLE_ASSIGN_OR_RETURN(void* data, vm_reserve(size));
  return NumaReservation{data, NumaPlacement::kUnavailable};
#endif
}

Result<NumaAllocation>
vm_allocate_numa_interleaved(std::size_t size,
                             std::span<const unsigned> requested_nodes,
                             std::size_t bytes_per_chunk) noexcept {
  const std::size_t aligned = page_align_up(size);
  if (aligned == 0) {
    return std::unexpected(
        Status::InvalidArgument("vm allocate NUMA interleaved: size is zero or "
                                "overflows page alignment"));
  }

  const std::size_t page = page_size();
  const std::size_t chunk = bytes_per_chunk == 0 ? page : bytes_per_chunk;
  if (chunk % page != 0) {
    return std::unexpected(Status::InvalidArgument(
        "vm allocate NUMA interleaved: chunk must be a page-size multiple"));
  }

  const std::span<const unsigned> selected =
      requested_nodes.empty() ? numa_nodes() : requested_nodes;
  if (selected.empty() || selected.size() > kMaxNumaNodes) {
    return std::unexpected(Status::InvalidArgument(
        "vm allocate NUMA interleaved: node set is empty or too large"));
  }
  for (unsigned node : selected) {
    if (!numa_node_available(node)) {
      return std::unexpected(Status::OutOfRange(
          "vm allocate NUMA interleaved: node ID is unavailable"));
    }
  }

  bool all_same = true;
  bool strictly_increasing = true;
  for (std::size_t i = 1; i < selected.size(); ++i) {
    all_same = all_same && selected[i - 1] == selected[i];
    strictly_increasing =
        strictly_increasing && selected[i - 1] < selected[i];
  }

  // No placement machinery is needed when every usable page has the same
  // logical home. Preserve the same fully writable allocation contract.
  if (numa_nodes().size() <= 1u) {
    HYLE_ASSIGN_OR_RETURN(void* data, vm_allocate_rw(aligned));
    return NumaAllocation{data, aligned, NumaPlacement::kNotNeeded};
  }

  if (!has_numa_capability(numa_capabilities(),
                           NumaCapability::kSelectedInterleave)) {
    return std::unexpected(
        Status::Unimplemented("selected-node NUMA interleaved allocation is "
                              "unavailable on this platform"));
  }

#if defined(HYLE_OS_LINUX)
  HYLE_ASSIGN_OR_RETURN(void* data, vm_allocate_rw(aligned));

  // Linux can install one lazy whole-range policy for the common page stripe
  // over increasing node IDs. A repeated single node is the same one-range
  // bind; allowing repeats lets callers express weighted sequences without
  // penalizing this degenerate case.
  if (all_same) {
    NumaMask mask{};
    const std::array<unsigned, 1> one_node{selected.front()};
    auto max_node = make_numa_mask(one_node, mask);
    if (!max_node || sys_mbind(data, aligned, MPOL_BIND, mask.data(),
                               max_node ? *max_node : 0u, 0) != 0) {
      Status error =
          max_node
              ? numa_native_status("mbind(MPOL_BIND)", errno)
              : static_cast<Status>(std::move(max_node).error());
      (void)vm_release(data, aligned);
      return std::unexpected(std::move(error));
    }
  } else if (chunk == page && strictly_increasing) {
    NumaMask mask{};
    auto max_node = make_numa_mask(selected, mask);
    if (!max_node) {
      (void)vm_release(data, aligned);
      return std::unexpected(std::move(max_node).error());
    }
    if (sys_mbind(data, aligned, MPOL_INTERLEAVE, mask.data(), *max_node, 0) !=
        0) {
      Status error = numa_native_status("mbind(MPOL_INTERLEAVE)", errno);
      (void)vm_release(data, aligned);
      return std::unexpected(std::move(error));
    }
  } else {
    // Arbitrary order, repeated nodes (weighted placement), and custom chunks
    // cannot be represented by one Linux VMA policy. Keep a bounded number of
    // lazy policy ranges because this preserves demand paging for the normal
    // large-chunk case. Above the bound, first-touch and move pages in fixed
    // batches. This is the Pareto boundary for the kernel API: O(K) retained
    // policy segments or O(P) placement work. Batching keeps scratch O(1)
    // instead of oneTBB's page/node/status arrays that grow with the mapping.
    constexpr std::size_t kMaxLazyPolicyStripes = 1024;
    const std::size_t stripes =
        aligned / chunk + static_cast<std::size_t>(aligned % chunk != 0);
    const bool eager_bounded_policy = stripes > kMaxLazyPolicyStripes;

#  if defined(MADV_NOHUGEPAGE)
    if (eager_bounded_policy &&
        ::madvise(data, aligned, MADV_NOHUGEPAGE) != 0) {
      Status error = numa_native_status("madvise(MADV_NOHUGEPAGE)", errno);
      (void)vm_release(data, aligned);
      return std::unexpected(std::move(error));
    }
#  endif

    auto* base = static_cast<std::byte*>(data);
    if (!eager_bounded_policy) {
      NumaMask mask{};
      unsigned masked_node = kMaxNumaNodes;
      for (std::size_t offset = 0, stripe = 0; offset < aligned; ++stripe) {
        const std::size_t length =
            chunk < aligned - offset ? chunk : aligned - offset;
        const unsigned node = selected[stripe % selected.size()];
        if (masked_node != node) {
          if (masked_node < kMaxNumaNodes) {
            mask[masked_node / kMaskBits] &=
                ~(1UL << (masked_node % kMaskBits));
          }
          mask[node / kMaskBits] |= 1UL << (node % kMaskBits);
          masked_node = node;
        }
        if (sys_mbind(base + offset, length, MPOL_BIND, mask.data(), node + 1u,
                      0) != 0) {
          Status error = numa_native_status("mbind(MPOL_BIND stripe)", errno);
          (void)vm_release(data, aligned);
          return std::unexpected(std::move(error));
        }
        offset += length;
      }
    } else {
      constexpr std::size_t kMoveBatchPages = 256;
      std::array<void*, kMoveBatchPages> pages{};
      std::array<int, kMoveBatchPages> target_nodes{};
      std::array<int, kMoveBatchPages> statuses{};
      auto* bytes = reinterpret_cast<volatile unsigned char*>(base);
      for (std::size_t offset = 0; offset < aligned;) {
        const std::size_t remaining_pages = (aligned - offset) / page;
        const std::size_t count = remaining_pages < kMoveBatchPages
                                      ? remaining_pages
                                      : kMoveBatchPages;
        for (std::size_t i = 0; i < count; ++i) {
          const std::size_t page_offset = offset + i * page;
          bytes[page_offset] = 0;
          pages[i] = base + page_offset;
          target_nodes[i] = static_cast<int>(
              selected[(page_offset / chunk) % selected.size()]);
        }
        const long rc = sys_move_pages(static_cast<unsigned long>(count),
                                       pages.data(), target_nodes.data(),
                                       statuses.data(), 0);
        if (rc < 0) {
          Status error = numa_native_status("move_pages(NUMA batch)", errno);
          (void)vm_release(data, aligned);
          return std::unexpected(std::move(error));
        }
        for (std::size_t i = 0; i < count; ++i) {
          if (statuses[i] < 0) {
            Status error = numa_native_status("move_pages(NUMA page)",
                                              -statuses[i]);
            (void)vm_release(data, aligned);
            return std::unexpected(std::move(error));
          }
        }
        if (rc != 0) {
          (void)vm_release(data, aligned);
          return std::unexpected(Status::Unavailable(
              "move_pages left NUMA pages on unintended nodes"));
        }
        offset += count * page;
      }
    }
  }
  return NumaAllocation{data, aligned, NumaPlacement::kApplied};

#elif defined(HYLE_OS_WINDOWS)
  auto& f = win_fns();
  const PVirtualAllocExNuma commit_numa = f.virtual_alloc_ex_numa;
  // Only VirtualAllocExNuma preserves one ordinary reservation while
  // committing preferred-node stripes inside it. VirtualAlloc2 placeholder
  // replacement creates one independently owned allocation per stripe, which
  // cannot satisfy NumaAllocation/VmRegion's single vm_release contract.
  if (!win_selected_interleave_supported(commit_numa != nullptr,
                                         f.virtual_alloc2 != nullptr)) {
    return std::unexpected(Status::Unimplemented(
        "selected-node NUMA interleaved allocation requires "
        "VirtualAllocExNuma on Windows"));
  }
  void* data =
      ::VirtualAlloc(nullptr, aligned, MEM_RESERVE, PAGE_NOACCESS);
  if (data == nullptr) {
    return std::unexpected(numa_native_status(
        "VirtualAlloc(MEM_RESERVE NUMA interleave)", ::GetLastError()));
  }

  auto* base = static_cast<std::byte*>(data);
  const std::size_t placement_chunk = all_same ? aligned : chunk;
  for (std::size_t offset = 0, stripe = 0; offset < aligned; ++stripe) {
    const std::size_t length = placement_chunk < aligned - offset
                                   ? placement_chunk
                                   : aligned - offset;
    const DWORD node = static_cast<DWORD>(selected[stripe % selected.size()]);
    void* committed = commit_numa(GetCurrentProcess(), base + offset, length,
                                  MEM_COMMIT, PAGE_READWRITE, node);
    if (committed != base + offset) {
      const DWORD code = ::GetLastError();
      (void)vm_release(data, aligned);
      return std::unexpected(numa_native_status(
          "NUMA stripe commit",
          code != ERROR_SUCCESS ? code : ERROR_INVALID_ADDRESS));
    }
    offset += length;
  }
  return NumaAllocation{data, aligned, NumaPlacement::kApplied};
#else
  (void)selected;
  (void)chunk;
  return std::unexpected(
      Status::Unimplemented("selected-node NUMA interleaved allocation is "
                            "unavailable on this platform"));
#endif
}

}  // namespace hyle::platform

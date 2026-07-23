#include "hyle/platform/numa.h"

#include <cstdio>
#include <cstring>

#include "hyle/macros.h"
#include "hyle/platform.h"
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
  char description[160]{};
  DWORD length = ::FormatMessageA(
      FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr,
      code, 0, description, sizeof(description), nullptr);
  while (length > 0 && (description[length - 1] == '\r' ||
                        description[length - 1] == '\n' ||
                        description[length - 1] == ' ')) {
    description[--length] = '\0';
  }
  char message[288]{};
  std::snprintf(message, sizeof(message), "%s failed (win32=%lu): %s",
                operation, static_cast<unsigned long>(code),
                length != 0 ? description : "unknown Windows error");
  if (code == ERROR_NOT_ENOUGH_MEMORY || code == ERROR_OUTOFMEMORY ||
      code == ERROR_COMMITMENT_LIMIT) {
    return Status::ResourceExhausted(message);
  }
  if (code == ERROR_ACCESS_DENIED) return Status::PermissionDenied(message);
  if (code == ERROR_NOT_SUPPORTED || code == ERROR_CALL_NOT_IMPLEMENTED) {
    return Status::Unimplemented(message);
  }
  if (code == ERROR_INVALID_PARAMETER || code == ERROR_INVALID_ADDRESS) {
    return Status::InvalidArgument(message);
  }
  return Status::Unavailable(message);
}
#elif defined(HYLE_OS_LINUX)
Status numa_native_status(const char* operation, int code) noexcept {
  char description[160]{};
  const char* text = description;
#  if defined(__GLIBC__) && defined(_GNU_SOURCE)
  text = ::strerror_r(code, description, sizeof(description));
#  else
  if (::strerror_r(code, description, sizeof(description)) != 0) {
    text = "unknown POSIX error";
  }
#  endif
  char message[288]{};
  std::snprintf(message, sizeof(message), "%s failed (errno=%d): %s",
                operation, code, text);
  if (code == ENOMEM) return Status::ResourceExhausted(message);
  if (code == EACCES || code == EPERM) return Status::PermissionDenied(message);
  if (code == EINVAL) return Status::InvalidArgument(message);
  if (code == ENOSYS) return Status::Unimplemented(message);
  return Status::Unavailable(message);
}
#endif

#if defined(HYLE_OS_WINDOWS)

using PGetNumaHighestNodeNumber = BOOL(WINAPI*)(PULONG);
using PGetCurrentProcessorNumberEx = void(WINAPI*)(PROCESSOR_NUMBER*);
using PGetNumaProcessorNodeEx = BOOL(WINAPI*)(PPROCESSOR_NUMBER, PUSHORT);

#pragma warning(push)
#pragma warning(disable : 4201)
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
#pragma warning(pop)

enum { kMemExtendedParameterNumaNode = 2 };

using PVirtualAlloc2 = PVOID(WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG,
                                      HyleMemExtendedParameter*, ULONG);

struct WinNumaFns {
  PGetNumaHighestNodeNumber get_highest = nullptr;
  PGetCurrentProcessorNumberEx get_cpu_ex = nullptr;
  PGetNumaProcessorNodeEx get_node_ex = nullptr;
  PVirtualAlloc2 virtual_alloc2 = nullptr;
};

WinNumaFns& win_fns() {
  static WinNumaFns f = [] {
    WinNumaFns x;
    HMODULE k32 = ::GetModuleHandleW(L"kernel32.dll");
    if (k32) {
      x.get_highest = reinterpret_cast<PGetNumaHighestNodeNumber>(
          reinterpret_cast<void*>(::GetProcAddress(k32, "GetNumaHighestNodeNumber")));
      x.get_cpu_ex = reinterpret_cast<PGetCurrentProcessorNumberEx>(
          reinterpret_cast<void*>(::GetProcAddress(k32, "GetCurrentProcessorNumberEx")));
      x.get_node_ex = reinterpret_cast<PGetNumaProcessorNodeEx>(
          reinterpret_cast<void*>(::GetProcAddress(k32, "GetNumaProcessorNodeEx")));
      x.virtual_alloc2 = reinterpret_cast<PVirtualAlloc2>(
          reinterpret_cast<void*>(::GetProcAddress(k32, "VirtualAlloc2")));
      if (!x.virtual_alloc2) {
        x.virtual_alloc2 = reinterpret_cast<PVirtualAlloc2>(
            reinterpret_cast<void*>(::GetProcAddress(k32, "VirtualAlloc2FromApp")));
      }
    }
    return x;
  }();
  return f;
}

#elif defined(HYLE_OS_LINUX)

// linux/mempolicy.h constants (avoid header dependency).
#ifndef MPOL_DEFAULT
#  define MPOL_DEFAULT 0
#endif
#ifndef MPOL_PREFERRED
#  define MPOL_PREFERRED 1
#endif
#ifndef MPOL_BIND
#  define MPOL_BIND 2
#endif
#ifndef MPOL_INTERLEAVE
#  define MPOL_INTERLEAVE 3
#endif

long sys_mbind(void* start, unsigned long len, unsigned long mode,
               const unsigned long* nmask, unsigned long maxnode,
               unsigned flags) {
#if defined(SYS_mbind)
  return ::syscall(SYS_mbind, start, len, mode, nmask, maxnode, flags);
#else
  (void)start;
  (void)len;
  (void)mode;
  (void)nmask;
  (void)maxnode;
  (void)flags;
  errno = ENOSYS;
  return -1;
#endif
}

bool path_exists(const char* path) { return ::access(path, R_OK) == 0; }

#endif

}  // namespace

unsigned numa_node_count() noexcept {
#if defined(HYLE_OS_WINDOWS)
  ULONG highest = 0;
  auto& f = win_fns();
  if (f.get_highest && f.get_highest(&highest)) {
    return static_cast<unsigned>(highest) + 1u;
  }
  return 1u;
#elif defined(HYLE_OS_LINUX)
  unsigned n = 0;
  char buf[64];
  for (; n < 256; ++n) {
    static const char kPrefix[] = "/sys/devices/system/node/node";
    char* w = buf;
    for (const char* p = kPrefix; *p; ++p) *w++ = *p;
    char digs[8];
    unsigned v = n;
    int nd = 0;
    do {
      digs[nd++] = static_cast<char>('0' + (v % 10));
      v /= 10;
    } while (v && nd < 8);
    while (nd--) *w++ = digs[nd];
    *w = '\0';
    if (!path_exists(buf)) break;
  }
  return n == 0 ? 1u : n;
#else
  return 1u;
#endif
}

unsigned numa_current_node() noexcept {
  const unsigned count = numa_node_count();
#if defined(HYLE_OS_WINDOWS)
  auto& f = win_fns();
  if (f.get_cpu_ex && f.get_node_ex) {
    PROCESSOR_NUMBER pnum{};
    f.get_cpu_ex(&pnum);
    USHORT node = 0;
    if (f.get_node_ex(&pnum, &node)) {
      unsigned n = static_cast<unsigned>(node);
      return n < count ? n : 0u;
    }
  }
  return 0u;
#elif defined(HYLE_OS_LINUX)
#  if defined(SYS_getcpu)
  unsigned cpu = 0;
  unsigned node = 0;
  if (::syscall(SYS_getcpu, &cpu, &node, nullptr) == 0) {
    return node < count ? node : 0u;
  }
#  endif
  (void)count;
  return 0u;
#else
  (void)count;
  return 0u;
#endif
}

unsigned numa_node_of_cpu(unsigned cpu) noexcept {
  const unsigned count = numa_node_count();
  if (count <= 1u) return 0u;  // single-node / unsupported: everything is node 0
#if defined(HYLE_OS_WINDOWS)
  auto& f = win_fns();
  if (f.get_node_ex) {
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
    if (!found || number_in_group > 0xFFu) return 0u;
    PROCESSOR_NUMBER pnum{};
    pnum.Group = target_group;
    pnum.Number = static_cast<BYTE>(number_in_group);
    USHORT node = 0;
    if (f.get_node_ex(&pnum, &node)) {
      const unsigned n = static_cast<unsigned>(node);  // 0xFFFF (no node) -> 0 below
      return n < count ? n : 0u;
    }
  }
  return 0u;
#elif defined(HYLE_OS_LINUX)
  // sysfs exposes one symlink /sys/devices/system/node/nodeN/cpuM per (node,cpu)
  // membership; find the node whose directory contains this cpu.
  for (unsigned nd = 0; nd < count; ++nd) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "/sys/devices/system/node/node%u/cpu%u", nd,
                  cpu);
    if (path_exists(buf)) return nd;
  }
  return 0u;
#else
  (void)cpu;
  return 0u;
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

  const unsigned nodes = numa_node_count();

  if (policy == NumaPolicy::kInterleave) {
    // Interleave is a genuine no-op on a single-node machine. Linux rejects
    // MPOL_INTERLEAVE with a one-bit nodemask on some kernels.
    if (nodes <= 1u) return NumaPlacement::kNotNeeded;
#if defined(HYLE_OS_LINUX)
    constexpr unsigned kBits = sizeof(unsigned long) * 8u;
    unsigned long mask[4]{};
    if (nodes > kBits * 4u) {
      return std::unexpected(Status::OutOfRange(
          "vm bind NUMA interleave: too many nodes for policy mask"));
    }
    for (unsigned nd = 0; nd < nodes; ++nd) {
      mask[nd / kBits] |= 1UL << (nd % kBits);
    }
    long rc = sys_mbind(ptr, size, MPOL_INTERLEAVE, mask, kBits * 4u, 0);
    if (rc != 0) {
      return std::unexpected(numa_native_status("mbind(MPOL_INTERLEAVE)", errno));
    }
    return NumaPlacement::kApplied;
#else
    return std::unexpected(Status::Unimplemented(
        "NUMA interleave is unavailable on this platform"));
#endif
  }

  if (node == kNumaAny) return NumaPlacement::kNotNeeded;
  if (node < 0 || static_cast<unsigned>(node) >= nodes) {
    return std::unexpected(
        Status::OutOfRange("vm bind NUMA: node index out of range"));
  }

  // A valid per-node policy is a genuine no-op on a single-node machine.
  if (nodes <= 1u) return NumaPlacement::kNotNeeded;

#if defined(HYLE_OS_LINUX)
  constexpr unsigned kBits = sizeof(unsigned long) * 8u;
  unsigned long mask[4]{};
  const unsigned nd = static_cast<unsigned>(node);
  if (nd >= kBits * 4u) {
    return std::unexpected(
        Status::OutOfRange("vm bind NUMA: node exceeds policy mask"));
  }
  mask[nd / kBits] = 1UL << (nd % kBits);
  const unsigned long mode =
      (policy == NumaPolicy::kBind) ? MPOL_BIND : MPOL_PREFERRED;
  long rc = sys_mbind(ptr, size, mode, mask, kBits * 4u, 0);
  if (rc != 0) {
    return std::unexpected(numa_native_status(
        policy == NumaPolicy::kBind ? "mbind(MPOL_BIND)"
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
  if (node < 0 || static_cast<unsigned>(node) >= numa_node_count()) {
    return std::unexpected(
        Status::OutOfRange("vm reserve NUMA: node index out of range"));
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
  HYLE_ASSIGN_OR_RETURN(void* data, vm_reserve(size));
  return NumaReservation{data, NumaPlacement::kUnavailable};
#elif defined(HYLE_OS_LINUX)
  HYLE_ASSIGN_OR_RETURN(void* data, vm_reserve(size));
  // Soft preferred placement: a bind failure must NOT drop the reservation.
  // The placement result records the fallback without inventing a side channel.
  auto placement = vm_bind_numa(data, size, node, NumaPolicy::kPreferred);
  return NumaReservation{
      data, placement ? *placement : NumaPlacement::kUnavailable};
#else
  HYLE_ASSIGN_OR_RETURN(void* data, vm_reserve(size));
  return NumaReservation{data, NumaPlacement::kUnavailable};
#endif
}

}  // namespace hyle::platform

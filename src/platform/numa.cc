#include "hyle/platform/numa.h"

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

// Build a node mask for nodes [0, count). Caps at 64 bits (one ulong).
unsigned long all_nodes_mask(unsigned count) {
  if (count == 0) return 1UL;
  if (count >= sizeof(unsigned long) * 8u) {
    return ~0UL;
  }
  return (1UL << count) - 1UL;
}

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

bool vm_bind_numa(void* ptr, std::size_t size, int node,
                  NumaPolicy policy) noexcept {
  if (ptr == nullptr || size == 0) return true;
  size = page_align_up(size);
  if (size == 0) return true;

  if (policy == NumaPolicy::kInterleave) {
#if defined(HYLE_OS_LINUX)
    unsigned long mask = all_nodes_mask(numa_node_count());
    long rc =
        sys_mbind(ptr, size, MPOL_INTERLEAVE, &mask, sizeof(mask) * 8, 0);
    return rc == 0;
#else
    (void)ptr;
    return true;  // no portable interleave on Win/mac — soft success
#endif
  }

  if (node == kNumaAny) return true;
  if (node < 0 || static_cast<unsigned>(node) >= numa_node_count()) return false;

#if defined(HYLE_OS_LINUX)
  if (static_cast<unsigned>(node) >= sizeof(unsigned long) * 8u) return false;
  unsigned long mask = 1UL << static_cast<unsigned>(node);
  const unsigned long mode =
      (policy == NumaPolicy::kBind) ? MPOL_BIND : MPOL_PREFERRED;
  long rc = sys_mbind(ptr, size, mode, &mask, sizeof(mask) * 8, 0);
  return rc == 0;
#elif defined(HYLE_OS_WINDOWS)
  // Post-hoc hard bind is not generally available for anonymous VirtualAlloc.
  // Allocation-time placement is vm_reserve_numa. Preferred/Bind both soft-ok.
  (void)policy;
  return true;
#else
  (void)policy;
  return true;
#endif
}

bool vm_bind_numa_interleave(void* ptr, std::size_t size) noexcept {
  return vm_bind_numa(ptr, size, kNumaAny, NumaPolicy::kInterleave);
}

void* vm_reserve_numa(std::size_t size, int node) noexcept {
  if (node == kNumaAny || node < 0 ||
      static_cast<unsigned>(node) >= numa_node_count()) {
    return vm_reserve(size);
  }
  size = page_align_up(size);
  if (size == 0) return nullptr;

#if defined(HYLE_OS_WINDOWS)
  auto& f = win_fns();
  if (f.virtual_alloc2) {
    HyleMemExtendedParameter param{};
    param.Type = kMemExtendedParameterNumaNode;
    param.ULong = static_cast<DWORD>(node);
    void* p = f.virtual_alloc2(GetCurrentProcess(), nullptr, size, MEM_RESERVE,
                               PAGE_NOACCESS, &param, 1);
    if (p != nullptr) return p;
  }
  return vm_reserve(size);
#elif defined(HYLE_OS_LINUX)
  void* p = vm_reserve(size);
  if (p == nullptr) return nullptr;
  (void)vm_bind_numa(p, size, node, NumaPolicy::kPreferred);
  return p;
#else
  return vm_reserve(size);
#endif
}

}  // namespace hyle::platform

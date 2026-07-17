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

// Reuse virtual_memory's thread-local error channel via vm_last_error contract:
// we set errors by calling through failing paths or a local buffer. Simpler:
// numa failures call into the same style — use a thin local that mirrors vm.
// Actually vm_last_error is in virtual_memory.cc as static TLS. We cannot set it
// from here without exporting a setter. Keep numa soft-fail without shared error
// text for bind; reserve_numa falls back to vm_reserve which sets errors.

#if defined(HYLE_OS_WINDOWS)

using PGetNumaHighestNodeNumber = BOOL(WINAPI*)(PULONG);
using PGetCurrentProcessorNumberEx = void(WINAPI*)(PROCESSOR_NUMBER*);
using PGetNumaProcessorNodeEx = BOOL(WINAPI*)(PPROCESSOR_NUMBER, PUSHORT);

// VirtualAlloc2 + MEM_EXTENDED_PARAMETER (Win10+). Anonymous members so
// MSVC accepts .Type / .ULong (same layout as SDK MEM_EXTENDED_PARAMETER).
#pragma warning(push)
#pragma warning(disable : 4201)  // nameless struct/union
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
  bool inited = false;
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
        // Store apps variant.
        x.virtual_alloc2 = reinterpret_cast<PVirtualAlloc2>(
            reinterpret_cast<void*>(::GetProcAddress(k32, "VirtualAlloc2FromApp")));
      }
    }
    x.inited = true;
    return x;
  }();
  return f;
}

#elif defined(HYLE_OS_LINUX)

// mbind via syscall (no libnuma).
#ifndef MPOL_PREFERRED
#  define MPOL_PREFERRED 1
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

bool path_exists(const char* path) {
  return ::access(path, R_OK) == 0;
}

#endif

}  // namespace

unsigned numa_node_count() noexcept {
#if defined(HYLE_OS_WINDOWS)
  ULONG highest = 0;
  auto& f = win_fns();
  if (f.get_highest && f.get_highest(&highest)) {
    // highest is max index; count is highest+1 if that node exists. On most
    // systems nodes are dense 0..highest.
    return static_cast<unsigned>(highest) + 1u;
  }
  return 1u;
#elif defined(HYLE_OS_LINUX)
  // Enumerate /sys/devices/system/node/nodeN (same approach as mimalloc).
  unsigned n = 0;
  char buf[64];
  for (; n < 256; ++n) {
    // Check node{n} exists.
    // snprintf-free small format:
    // "/sys/devices/system/node/node" + decimal
    static const char kPrefix[] = "/sys/devices/system/node/node";
    char* w = buf;
    for (const char* p = kPrefix; *p; ++p) *w++ = *p;
    // write n in decimal
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

bool vm_bind_numa(void* ptr, std::size_t size, int node) noexcept {
  if (ptr == nullptr || size == 0 || node == kNumaAny) return true;
  if (node < 0 || static_cast<unsigned>(node) >= numa_node_count()) return false;
  size = page_align_up(size);
  if (size == 0) return true;

#if defined(HYLE_OS_LINUX)
  // Prefer node `node` without failing if memory is tight (MPOL_PREFERRED).
  unsigned long mask = 0;
  if (static_cast<unsigned>(node) >= sizeof(mask) * 8u) {
    // Node index beyond single-ulong mask — still try with truncated support.
    // For >64 nodes we'd need a multi-word mask; rare for HYLE targets.
    return false;
  }
  mask = 1UL << static_cast<unsigned>(node);
  // maxnode is bit count (Linux requires maxnode > highest bit index).
  long rc = sys_mbind(ptr, size, MPOL_PREFERRED, &mask, sizeof(mask) * 8, 0);
  return rc == 0;
#elif defined(HYLE_OS_WINDOWS)
  // Post-hoc bind is not generally available for anonymous VirtualAlloc.
  // Allocation-time placement is vm_reserve_numa. Treat as soft success so
  // portable callers can always call bind after reserve_numa/reserve.
  (void)ptr;
  return true;
#else
  (void)ptr;
  return true;
#endif
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
    // Fall through to plain reserve.
  }
  return vm_reserve(size);
#elif defined(HYLE_OS_LINUX)
  void* p = vm_reserve(size);
  if (p == nullptr) return nullptr;
  if (!vm_bind_numa(p, size, node)) {
    // Soft: keep the reservation unbound rather than failing the alloc.
    // (mbind can fail on some kernels/configs; memory is still usable.)
  }
  return p;
#else
  return vm_reserve(size);
#endif
}

}  // namespace hyle::platform

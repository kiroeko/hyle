#include "hyle/platform/virtual_memory.h"

#include <cstring>

#include "hyle/macros.h"
#include "hyle/platform.h"
#include "hyle/platform/numa.h"

// Linux THP advice flags live behind feature macros on some libcs.
#if defined(HYLE_OS_LINUX) && !defined(MADV_HUGEPAGE)
#  define MADV_HUGEPAGE 14
#endif
#if defined(HYLE_OS_LINUX) && !defined(MADV_NOHUGEPAGE)
#  define MADV_NOHUGEPAGE 15
#endif

#if defined(HYLE_OS_WINDOWS)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <errno.h>
#  include <sys/mman.h>
#  include <unistd.h>
#endif

namespace hyle::platform {
namespace {

// Thread-local error buffer (same contract as dynlib_last_error).
thread_local char g_error[256] = {0};

void set_error(const char* msg) noexcept {
  if (msg == nullptr) {
    g_error[0] = '\0';
    return;
  }
  std::size_t n = std::strlen(msg);
  if (n >= sizeof(g_error)) n = sizeof(g_error) - 1;
  std::memcpy(g_error, msg, n);
  g_error[n] = '\0';
}

void clear_error() noexcept { g_error[0] = '\0'; }

#if defined(HYLE_OS_WINDOWS)
void set_error_win() noexcept {
  DWORD code = GetLastError();
  char buf[256];
  DWORD len = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                             nullptr, code, 0, buf, sizeof(buf), nullptr);
  if (len == 0) {
    set_error("unknown VirtualAlloc/Protect error");
  } else {
    // Trim trailing CR/LF that FormatMessage often appends.
    while (len > 0 && (buf[len - 1] == '\r' || buf[len - 1] == '\n' || buf[len - 1] == ' ')) {
      buf[--len] = '\0';
    }
    set_error(buf);
  }
}
#else
void set_error_errno() noexcept {
  const char* msg = std::strerror(errno);
  set_error(msg != nullptr ? msg : "unknown mmap/mprotect error");
}
#endif

std::size_t page_size_cached() noexcept {
  static const std::size_t kPage = []() noexcept -> std::size_t {
#if defined(HYLE_OS_WINDOWS)
    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    return static_cast<std::size_t>(si.dwPageSize);
#else
    long p = ::sysconf(_SC_PAGESIZE);
    return p > 0 ? static_cast<std::size_t>(p) : 4096u;
#endif
  }();
  return kPage;
}

void require_page_aligned(const void* ptr) noexcept {
  if (ptr == nullptr) return;
  std::uintptr_t a = reinterpret_cast<std::uintptr_t>(ptr);
  HYLE_FATAL_ASSERT((a % page_size_cached()) == 0);
}

std::size_t align_size(std::size_t size) noexcept {
  if (size == 0) return 0;
  return page_align_up(size);
}

#if defined(HYLE_OS_WINDOWS)
DWORD protect_to_win(Protect prot) noexcept {
  if (prot == Protect::kRead) return PAGE_READONLY;
  if (prot == Protect::kReadWrite) return PAGE_READWRITE;
  return PAGE_NOACCESS;  // kNone (and any future unknown -> safest)
}
#else
int protect_to_posix(Protect prot) noexcept {
  if (prot == Protect::kRead) return PROT_READ;
  if (prot == Protect::kReadWrite) return PROT_READ | PROT_WRITE;
  return PROT_NONE;  // kNone
}
#endif

bool range_ok(void* base, std::size_t size, std::size_t offset,
              std::size_t len) noexcept {
  if (base == nullptr || size == 0) return false;
  if (len == 0) return true;
  if (offset > size || len > size - offset) return false;
  return true;
}

}  // namespace

std::size_t page_size() noexcept { return page_size_cached(); }

std::size_t large_page_size() noexcept {
  static const std::size_t kLarge = []() noexcept -> std::size_t {
#if defined(HYLE_OS_WINDOWS)
    // GetLargePageMinimum is the hard large-page size (often 2 MiB). Requires
    // SeLockMemoryPrivilege to actually allocate with MEM_LARGE_PAGES; the size
    // query itself is fine without it.
    using Fn = SIZE_T(WINAPI*)();
    HMODULE k32 = ::GetModuleHandleW(L"kernel32.dll");
    if (k32 == nullptr) return 0;
    auto fn = reinterpret_cast<Fn>(
        reinterpret_cast<void*>(::GetProcAddress(k32, "GetLargePageMinimum")));
    if (fn == nullptr) return 0;
    SIZE_T s = fn();
    return s > 0 ? static_cast<std::size_t>(s) : 0;
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
    defined(__aarch64__) || defined(_M_ARM64) || defined(__riscv)
    // Architectural default huge/THP size used by seastar/linux x86&arm.
    return static_cast<std::size_t>(2u) * 1024u * 1024u;
#elif defined(__PPC__)
    return static_cast<std::size_t>(16u) * 1024u * 1024u;
#else
    return 0;
#endif
  }();
  return kLarge;
}

const char* vm_last_error() noexcept { return g_error; }

void vm_set_last_error(const char* msg) noexcept {
  if (msg == nullptr) {
    clear_error();
    return;
  }
  set_error(msg);
}

bool vm_advise_hugepage(void* ptr, std::size_t size) noexcept {
  clear_error();
  if (ptr == nullptr || size == 0) return true;
  require_page_aligned(ptr);
  size = align_size(size);
#if defined(HYLE_OS_LINUX) && defined(MADV_HUGEPAGE)
  if (::madvise(ptr, size, MADV_HUGEPAGE) != 0) {
    set_error_errno();
    return false;
  }
  return true;
#else
  (void)ptr;
  (void)size;
  // No portable anonymous-THP madvise on Windows/macOS.
  set_error("vm_advise_hugepage: not supported on this platform");
  return false;
#endif
}

bool vm_advise_no_hugepage(void* ptr, std::size_t size) noexcept {
  clear_error();
  if (ptr == nullptr || size == 0) return true;
  require_page_aligned(ptr);
  size = align_size(size);
#if defined(HYLE_OS_LINUX) && defined(MADV_NOHUGEPAGE)
  if (::madvise(ptr, size, MADV_NOHUGEPAGE) != 0) {
    set_error_errno();
    return false;
  }
  return true;
#else
  (void)ptr;
  (void)size;
  set_error("vm_advise_no_hugepage: not supported on this platform");
  return false;
#endif
}

void* vm_reserve(std::size_t size) noexcept {
  clear_error();
  size = align_size(size);
  if (size == 0) {
    set_error("vm_reserve: size rounds to 0");
    return nullptr;
  }
#if defined(HYLE_OS_WINDOWS)
  // Reserve only -- no physical pages. PAGE_NOACCESS until commit.
  void* p = ::VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_NOACCESS);
  if (p == nullptr) {
    set_error_win();
    return nullptr;
  }
  return p;
#else
  // PROT_NONE + anonymous: address space reserved, no access until mprotect.
  void* p = ::mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) {
    set_error_errno();
    return nullptr;
  }
  return p;
#endif
}

bool vm_commit(void* ptr, std::size_t size) noexcept {
  clear_error();
  if (ptr == nullptr || size == 0) return true;
  require_page_aligned(ptr);
  size = align_size(size);
#if defined(HYLE_OS_WINDOWS)
  void* p = ::VirtualAlloc(ptr, size, MEM_COMMIT, PAGE_READWRITE);
  if (p == nullptr) {
    set_error_win();
    return false;
  }
  return true;
#else
  if (::mprotect(ptr, size, PROT_READ | PROT_WRITE) != 0) {
    set_error_errno();
    return false;
  }
  return true;
#endif
}

bool vm_decommit(void* ptr, std::size_t size) noexcept {
  clear_error();
  if (ptr == nullptr || size == 0) return true;
  require_page_aligned(ptr);
  size = align_size(size);
#if defined(HYLE_OS_WINDOWS)
  if (::VirtualFree(ptr, size, MEM_DECOMMIT) == 0) {
    set_error_win();
    return false;
  }
  return true;
#else
  // Drop physical pages, then restore no-access so accidental touch faults
  // instead of silently re-faulting in a RW mapping.
  if (::madvise(ptr, size, MADV_DONTNEED) != 0) {
    // Non-fatal on some platforms; still try mprotect.
  }
  if (::mprotect(ptr, size, PROT_NONE) != 0) {
    set_error_errno();
    return false;
  }
  return true;
#endif
}

bool vm_release(void* base, std::size_t size) noexcept {
  clear_error();
  if (base == nullptr) return true;
  require_page_aligned(base);
  size = align_size(size);
  if (size == 0) {
    set_error("vm_release: size rounds to 0");
    return false;
  }
#if defined(HYLE_OS_WINDOWS)
  // MEM_RELEASE requires dwSize == 0 and the ORIGINAL reserve base.
  (void)size;
  if (::VirtualFree(base, 0, MEM_RELEASE) == 0) {
    set_error_win();
    return false;
  }
  return true;
#else
  if (::munmap(base, size) != 0) {
    set_error_errno();
    return false;
  }
  return true;
#endif
}

bool vm_protect(void* ptr, std::size_t size, Protect prot) noexcept {
  clear_error();
  if (ptr == nullptr || size == 0) return true;
  require_page_aligned(ptr);
  size = align_size(size);
#if defined(HYLE_OS_WINDOWS)
  // Windows: VirtualProtect needs the pages to be committed for non-NOACCESS
  // changes in some cases; for PAGE_NOACCESS on committed pages it works.
  // Fiber guard use: commit usable pages, then protect the guard page as
  // NOACCESS (guard page is either never committed, or committed then locked).
  DWORD old = 0;
  if (::VirtualProtect(ptr, size, protect_to_win(prot), &old) == 0) {
    // If the range was only reserved (never committed), PAGE_NOACCESS is already
    // the reserve state -- treat kNone on reserved-only as success via re-reserve
    // semantics: try MEM_COMMIT first only for RW/Read.
    if (prot == Protect::kNone) {
      // Reserved-but-uncommitted pages are already no-access on Windows.
      // VirtualProtect on pure reserved memory fails; that is OK for kNone.
      MEMORY_BASIC_INFORMATION mbi{};
      if (::VirtualQuery(ptr, &mbi, sizeof(mbi)) != 0 &&
          (mbi.State & MEM_RESERVE) != 0 && (mbi.State & MEM_COMMIT) == 0) {
        return true;
      }
    }
    set_error_win();
    return false;
  }
  return true;
#else
  if (::mprotect(ptr, size, protect_to_posix(prot)) != 0) {
    set_error_errno();
    return false;
  }
  return true;
#endif
}

// ---- VmRegion ---------------------------------------------------------------

VmRegion VmRegion::reserve(std::size_t size) noexcept {
  VmRegion r;
  void* p = vm_reserve(size);
  if (p == nullptr) return r;
  r.base_ = p;
  r.size_ = page_align_up(size);
  r.guarded_ = false;
  return r;
}

VmRegion VmRegion::reserve_numa(std::size_t size, int node) noexcept {
  VmRegion r;
  void* p = vm_reserve_numa(size, node);
  if (p == nullptr) return r;
  r.base_ = p;
  r.size_ = page_align_up(size);
  r.guarded_ = false;
  return r;
}

VmRegion VmRegion::adopt(void* base, std::size_t size) noexcept {
  VmRegion r;
  if (base == nullptr || size == 0) return r;
  r.base_ = base;
  r.size_ = page_align_up(size);
  r.guarded_ = false;
  return r;
}

VmRegion VmRegion::reserve_guarded(std::size_t usable) noexcept {
  VmRegion r;
  std::size_t page = page_size();
  std::size_t usable_aligned = page_align_up(usable == 0 ? 1 : usable);
  std::size_t total = page + usable_aligned;
  void* p = vm_reserve(total);
  if (p == nullptr) return r;
  // Commit only the usable high range; leave the low page reserved/no-access.
  auto* bytes = static_cast<std::byte*>(p);
  if (!vm_commit(bytes + page, usable_aligned)) {
    vm_release(p, total);
    return VmRegion{};
  }
  // Explicit protect on the guard page (no-op-ish on POSIX reserve, documents
  // intent; on Windows reserved-uncommitted is already NOACCESS).
  if (!vm_protect(p, page, Protect::kNone)) {
    vm_release(p, total);
    return VmRegion{};
  }
  r.base_ = p;
  r.size_ = total;
  r.guarded_ = true;
  return r;
}

VmRegion::~VmRegion() { release(); }

VmRegion::VmRegion(VmRegion&& o) noexcept
    : base_(o.base_), size_(o.size_), guarded_(o.guarded_) {
  o.base_ = nullptr;
  o.size_ = 0;
  o.guarded_ = false;
}

VmRegion& VmRegion::operator=(VmRegion&& o) noexcept {
  if (this != &o) {
    release();
    base_ = o.base_;
    size_ = o.size_;
    guarded_ = o.guarded_;
    o.base_ = nullptr;
    o.size_ = 0;
    o.guarded_ = false;
  }
  return *this;
}

bool VmRegion::commit(std::size_t offset, std::size_t len) noexcept {
  if (!range_ok(base_, size_, offset, len)) {
    set_error("VmRegion::commit: out of range");
    return false;
  }
  if (len == 0) return true;
  return vm_commit(static_cast<std::byte*>(base_) + offset, len);
}

bool VmRegion::decommit(std::size_t offset, std::size_t len) noexcept {
  if (!range_ok(base_, size_, offset, len)) {
    set_error("VmRegion::decommit: out of range");
    return false;
  }
  if (len == 0) return true;
  return vm_decommit(static_cast<std::byte*>(base_) + offset, len);
}

bool VmRegion::protect(std::size_t offset, std::size_t len, Protect prot) noexcept {
  if (!range_ok(base_, size_, offset, len)) {
    set_error("VmRegion::protect: out of range");
    return false;
  }
  if (len == 0) return true;
  return vm_protect(static_cast<std::byte*>(base_) + offset, len, prot);
}

bool VmRegion::commit_usable() noexcept {
  if (!valid()) return false;
  if (guarded_) {
    return commit(page_size(), usable_size());
  }
  return commit(0, size_);
}

void VmRegion::release() noexcept {
  if (base_ == nullptr) return;
  // Best-effort; destructor must not throw / cannot report.
  (void)vm_release(base_, size_);
  base_ = nullptr;
  size_ = 0;
  guarded_ = false;
}

}  // namespace hyle::platform

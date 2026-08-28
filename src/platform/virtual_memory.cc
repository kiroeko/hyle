#include "hyle/platform/virtual_memory.h"

#include "virtual_memory_internal.h"

#include <cstdio>

#include "hyle/core/compiler.h"
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

#if defined(HYLE_OS_WINDOWS)
Status native_status(const char* operation, DWORD code) noexcept {
  return status_from_win32(code, operation);
}
#else
Status native_status(const char* operation, int code) noexcept {
  return status_from_errno(code, operation);
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

bool page_aligned(const void* ptr) noexcept {
  return ptr == nullptr ||
         (reinterpret_cast<std::uintptr_t>(ptr) % page_size_cached()) == 0;
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

bool range_is_reserved_uncommitted(void* ptr, std::size_t size) noexcept {
  auto cursor = reinterpret_cast<std::uintptr_t>(ptr);
  if (size > UINTPTR_MAX - cursor) return false;
  const std::uintptr_t end = cursor + size;
  while (cursor < end) {
    MEMORY_BASIC_INFORMATION mbi{};
    if (::VirtualQuery(reinterpret_cast<void*>(cursor), &mbi, sizeof(mbi)) ==
            0 ||
        mbi.State != MEM_RESERVE) {
      return false;
    }
    const auto region = reinterpret_cast<std::uintptr_t>(mbi.BaseAddress);
    if (mbi.RegionSize > UINTPTR_MAX - region) return false;
    const std::uintptr_t region_end = region + mbi.RegionSize;
    if (region_end <= cursor) return false;
    cursor = region_end < end ? region_end : end;
  }
  return true;
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
  if (offset > size) return false;
  if (len > size - offset) return false;
  return true;
}

bool align_nonempty_size(std::size_t requested, std::size_t& aligned) noexcept {
  aligned = align_size(requested);
  return requested == 0 || aligned != 0;
}

#if defined(HYLE_OS_LINUX)
std::size_t linux_large_page_size() noexcept {
  // The THP PMD size is the kernel's actual promotion granularity. Do not infer
  // it from ISA: Linux supports different base/huge-page configurations on the
  // same architecture.
  if (std::FILE* file =
          std::fopen("/sys/kernel/mm/transparent_hugepage/hpage_pmd_size", "rb")) {
    char text[64]{};
    const bool read = std::fgets(text, sizeof(text), file) != nullptr;
    std::fclose(file);
    if (read) {
      const std::size_t parsed =
          detail::parse_linux_thp_size(text, page_size_cached());
      if (parsed != 0) return parsed;
    }
  }

  // Hosts without THP sysfs may still expose hugetlb pages. `/proc/meminfo`
  // reports the default HugeTLB size in kB.
  if (std::FILE* file = std::fopen("/proc/meminfo", "rb")) {
    char line[256];
    while (std::fgets(line, sizeof(line), file) != nullptr) {
      const std::size_t parsed = detail::parse_linux_meminfo_huge_page_size(
          line, page_size_cached());
      if (parsed != 0) {
        std::fclose(file);
        return parsed;
      }
    }
    std::fclose(file);
  }
  return 0;
}
#endif

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
#elif defined(HYLE_OS_LINUX)
    return linux_large_page_size();
#else
    // macOS has no anonymous huge-page advice contract corresponding to Linux
    // THP or Windows large pages. Unknown is more truthful than an ISA guess.
    return 0;
#endif
  }();
  return kLarge;
}

Status vm_advise_hugepage(void* ptr, std::size_t size) noexcept {
  if (size == 0) return Status::Ok();
  if (ptr == nullptr || !page_aligned(ptr)) {
    return Status::InvalidArgument(
        "vm advise hugepage: pointer is null or not page-aligned");
  }
  if (!align_nonempty_size(size, size)) {
    return Status::OutOfRange("vm advise hugepage: size overflow");
  }
#if defined(HYLE_OS_LINUX) && defined(MADV_HUGEPAGE)
  if (::madvise(ptr, size, MADV_HUGEPAGE) != 0) {
    return native_status("madvise(MADV_HUGEPAGE)", errno);
  }
  return Status::Ok();
#else
  (void)ptr;
  (void)size;
  return Status::Unimplemented(
      "anonymous hugepage advice is unavailable on this platform");
#endif
}

Status vm_advise_no_hugepage(void* ptr, std::size_t size) noexcept {
  if (size == 0) return Status::Ok();
  if (ptr == nullptr || !page_aligned(ptr)) {
    return Status::InvalidArgument(
        "vm advise no-hugepage: pointer is null or not page-aligned");
  }
  if (!align_nonempty_size(size, size)) {
    return Status::OutOfRange("vm advise no-hugepage: size overflow");
  }
#if defined(HYLE_OS_LINUX) && defined(MADV_NOHUGEPAGE)
  if (::madvise(ptr, size, MADV_NOHUGEPAGE) != 0) {
    return native_status("madvise(MADV_NOHUGEPAGE)", errno);
  }
  return Status::Ok();
#else
  (void)ptr;
  (void)size;
  return Status::Unimplemented(
      "anonymous no-hugepage advice is unavailable on this platform");
#endif
}

Result<void*> vm_reserve(std::size_t size) noexcept {
  size = align_size(size);
  if (size == 0) {
    return std::unexpected(Status::InvalidArgument(
        "vm reserve: size is zero or overflows page alignment"));
  }
#if defined(HYLE_OS_WINDOWS)
  // Reserve only -- no physical pages. PAGE_NOACCESS until commit.
  void* p = ::VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_NOACCESS);
  if (p == nullptr) {
    return std::unexpected(native_status("VirtualAlloc(MEM_RESERVE)",
                                         ::GetLastError()));
  }
  return p;
#else
  // PROT_NONE + anonymous: address space reserved, no access until mprotect.
  void* p = ::mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) {
    return std::unexpected(native_status("mmap(PROT_NONE)", errno));
  }
  return p;
#endif
}

Result<void*> vm_allocate_rw(std::size_t size) noexcept {
  const std::size_t aligned = page_align_up(size);
  if (aligned == 0) {
    return std::unexpected(Status::InvalidArgument(
        "vm allocate RW: size is zero or overflows page alignment"));
  }
#if defined(HYLE_OS_WINDOWS)
  void* base = ::VirtualAlloc(nullptr, aligned, MEM_RESERVE | MEM_COMMIT,
                              PAGE_READWRITE);
  if (base == nullptr) {
    return std::unexpected(native_status("VirtualAlloc(MEM_RESERVE|MEM_COMMIT)",
                                         ::GetLastError()));
  }
#else
  void* base = ::mmap(nullptr, aligned, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (base == MAP_FAILED) {
    return std::unexpected(native_status("mmap(PROT_READ|PROT_WRITE)", errno));
  }
#endif
  return base;
}

Status vm_commit(void* ptr, std::size_t size) noexcept {
  if (size == 0) return Status::Ok();
  if (ptr == nullptr || !page_aligned(ptr)) {
    return Status::InvalidArgument(
        "vm commit: pointer is null or not page-aligned");
  }
  if (!align_nonempty_size(size, size)) {
    return Status::OutOfRange("vm commit: size overflow");
  }
#if defined(HYLE_OS_WINDOWS)
  void* p = ::VirtualAlloc(ptr, size, MEM_COMMIT, PAGE_READWRITE);
  if (p == nullptr) {
    return native_status("VirtualAlloc(MEM_COMMIT)", ::GetLastError());
  }
  return Status::Ok();
#else
  if (::mprotect(ptr, size, PROT_READ | PROT_WRITE) != 0) {
    return native_status("mprotect(PROT_READ|PROT_WRITE)", errno);
  }
  return Status::Ok();
#endif
}

Status vm_decommit(void* ptr, std::size_t size) noexcept {
  if (size == 0) return Status::Ok();
  if (ptr == nullptr || !page_aligned(ptr)) {
    return Status::InvalidArgument(
        "vm decommit: pointer is null or not page-aligned");
  }
  if (!align_nonempty_size(size, size)) {
    return Status::OutOfRange("vm decommit: size overflow");
  }
#if defined(HYLE_OS_WINDOWS)
  if (::VirtualFree(ptr, size, MEM_DECOMMIT) == 0) {
    return native_status("VirtualFree(MEM_DECOMMIT)", ::GetLastError());
  }
  return Status::Ok();
#else
  // Drop physical pages, then restore no-access so accidental touch faults
  // instead of silently re-faulting in a RW mapping.
  int advice_error = 0;
  if (::madvise(ptr, size, MADV_DONTNEED) != 0) {
    advice_error = errno;
  }
  if (::mprotect(ptr, size, PROT_NONE) != 0) {
    return native_status("mprotect(PROT_NONE)", errno);
  }
  if (advice_error != 0) {
    return native_status("madvise(MADV_DONTNEED)", advice_error);
  }
  return Status::Ok();
#endif
}

Status vm_release(void* base, std::size_t size) noexcept {
  if (base == nullptr) return Status::Ok();
  if (!page_aligned(base)) {
    return Status::InvalidArgument("vm release: base is not page-aligned");
  }
  size = align_size(size);
  if (size == 0) {
    return Status::InvalidArgument(
        "vm release: size is zero or overflows page alignment");
  }
#if defined(HYLE_OS_WINDOWS)
  // MEM_RELEASE requires dwSize == 0 and the ORIGINAL reserve base.
  (void)size;
  if (::VirtualFree(base, 0, MEM_RELEASE) == 0) {
    return native_status("VirtualFree(MEM_RELEASE)", ::GetLastError());
  }
  return Status::Ok();
#else
  if (::munmap(base, size) != 0) {
    return native_status("munmap", errno);
  }
  return Status::Ok();
#endif
}

Status vm_protect(void* ptr, std::size_t size, Protect prot) noexcept {
  if (prot != Protect::kNone && prot != Protect::kRead &&
      prot != Protect::kReadWrite) {
    return Status::InvalidArgument("vm protect: invalid protection mode");
  }
  if (size == 0) return Status::Ok();
  if (ptr == nullptr || !page_aligned(ptr)) {
    return Status::InvalidArgument(
        "vm protect: pointer is null or not page-aligned");
  }
  if (!align_nonempty_size(size, size)) {
    return Status::OutOfRange("vm protect: size overflow");
  }
#if defined(HYLE_OS_WINDOWS)
  // Windows: VirtualProtect needs the pages to be committed for non-NOACCESS
  // changes in some cases; for PAGE_NOACCESS on committed pages it works.
  // Fiber guard use: commit usable pages, then protect the guard page as
  // NOACCESS (guard page is either never committed, or committed then locked).
  DWORD old = 0;
  if (::VirtualProtect(ptr, size, protect_to_win(prot), &old) == 0) {
    const DWORD protect_error = ::GetLastError();
    // If the range was only reserved (never committed), PAGE_NOACCESS is already
    // the reserve state -- treat kNone on reserved-only as success via re-reserve
    // semantics: try MEM_COMMIT first only for RW/Read.
    if (prot == Protect::kNone) {
      // Reserved-but-uncommitted pages are already no-access on Windows.
      // VirtualProtect on pure reserved memory fails; that is OK for kNone.
      if (range_is_reserved_uncommitted(ptr, size)) {
        return Status::Ok();
      }
    }
    return native_status("VirtualProtect", protect_error);
  }
  return Status::Ok();
#else
  if (::mprotect(ptr, size, protect_to_posix(prot)) != 0) {
    return native_status("mprotect", errno);
  }
  return Status::Ok();
#endif
}

// ---- VmRegion ---------------------------------------------------------------

Result<VmRegion> VmRegion::reserve(std::size_t size) noexcept {
  VmRegion r;
  HYLE_ASSIGN_OR_RETURN(void* p, vm_reserve(size));
  r.base_ = p;
  r.size_ = page_align_up(size);
  r.guarded_ = false;
  return r;
}

Result<VmRegion> VmRegion::allocate_rw(std::size_t size) noexcept {
  HYLE_ASSIGN_OR_RETURN(void* base, vm_allocate_rw(size));
  VmRegion region;
  region.base_ = base;
  region.size_ = page_align_up(size);
  return region;
}

Result<VmRegion> VmRegion::reserve_numa(std::size_t size, int node) noexcept {
  VmRegion r;
  HYLE_ASSIGN_OR_RETURN(NumaReservation reservation,
                        vm_reserve_numa(size, node));
  r.base_ = reservation.data;
  r.size_ = page_align_up(size);
  r.guarded_ = false;
  r.numa_placement_ = reservation.placement;
  return r;
}

Result<VmRegion> VmRegion::allocate_numa_interleaved(
    std::size_t size, std::span<const unsigned> nodes,
    std::size_t bytes_per_chunk) noexcept {
  HYLE_ASSIGN_OR_RETURN(
      NumaAllocation allocation,
      vm_allocate_numa_interleaved(size, nodes, bytes_per_chunk));
  VmRegion region;
  region.base_ = allocation.data;
  region.size_ = allocation.size;
  region.guarded_ = false;
  region.numa_interleaved_ = true;
  region.numa_placement_ = allocation.placement;
  return region;
}

Result<VmRegion> VmRegion::adopt(void* base, std::size_t size) noexcept {
  VmRegion r;
  if (base == nullptr || size == 0) {
    return std::unexpected(
        Status::InvalidArgument("VmRegion adopt: null base or zero size"));
  }
  if ((reinterpret_cast<std::uintptr_t>(base) % page_size()) != 0) {
    return std::unexpected(
        Status::InvalidArgument("VmRegion adopt: base is not page-aligned"));
  }
  const std::size_t aligned = page_align_up(size);
  if (aligned == 0) {
    return std::unexpected(Status::OutOfRange("VmRegion adopt: size overflow"));
  }
  r.base_ = base;
  r.size_ = aligned;
  r.guarded_ = false;
  return r;
}

Result<VmRegion> VmRegion::reserve_guarded(std::size_t usable) noexcept {
  VmRegion r;
  std::size_t page = page_size();
  std::size_t usable_aligned = page_align_up(usable == 0 ? 1 : usable);
  if (usable_aligned == 0 || usable_aligned > SIZE_MAX - page) {
    return std::unexpected(
        Status::OutOfRange("VmRegion reserve guarded: size overflow"));
  }
  std::size_t total = page + usable_aligned;
  HYLE_ASSIGN_OR_RETURN(void* p, vm_reserve(total));
  // Commit only the usable high range; leave the low page reserved/no-access.
  auto* bytes = static_cast<std::byte*>(p);
  Status status = vm_commit(bytes + page, usable_aligned);
  if (!status.ok()) {
    (void)vm_release(p, total);
    return std::unexpected(std::move(status).WithContext(
        "VmRegion reserve guarded commit usable range"));
  }
  // Explicit protect on the guard page (no-op-ish on POSIX reserve, documents
  // intent; on Windows reserved-uncommitted is already NOACCESS).
  status = vm_protect(p, page, Protect::kNone);
  if (!status.ok()) {
    (void)vm_release(p, total);
    return std::unexpected(std::move(status).WithContext(
        "VmRegion reserve guarded protect guard page"));
  }
  r.base_ = p;
  r.size_ = total;
  r.guarded_ = true;
  return r;
}

VmRegion::~VmRegion() { (void)release(); }

VmRegion::VmRegion(VmRegion&& o) noexcept
    : base_(o.base_),
      size_(o.size_),
      guarded_(o.guarded_),
      numa_interleaved_(o.numa_interleaved_),
      numa_placement_(o.numa_placement_) {
  o.base_ = nullptr;
  o.size_ = 0;
  o.guarded_ = false;
  o.numa_interleaved_ = false;
  o.numa_placement_ = NumaPlacement::kNotNeeded;
}

VmRegion& VmRegion::operator=(VmRegion&& o) noexcept {
  if (this != &o) {
    (void)release();
    base_ = o.base_;
    size_ = o.size_;
    guarded_ = o.guarded_;
    numa_interleaved_ = o.numa_interleaved_;
    numa_placement_ = o.numa_placement_;
    o.base_ = nullptr;
    o.size_ = 0;
    o.guarded_ = false;
    o.numa_interleaved_ = false;
    o.numa_placement_ = NumaPlacement::kNotNeeded;
  }
  return *this;
}

Status VmRegion::commit(std::size_t offset, std::size_t len) noexcept {
  if (!range_ok(base_, size_, offset, len)) {
    return valid() ? Status::OutOfRange("VmRegion commit: range is out of bounds")
                   : Status::FailedPrecondition("VmRegion commit: region is empty");
  }
  if (len == 0) return Status::Ok();
  return vm_commit(static_cast<std::byte*>(base_) + offset, len);
}

Status VmRegion::decommit(std::size_t offset, std::size_t len) noexcept {
  if (!range_ok(base_, size_, offset, len)) {
    return valid()
               ? Status::OutOfRange("VmRegion decommit: range is out of bounds")
               : Status::FailedPrecondition("VmRegion decommit: region is empty");
  }
  if (len == 0) return Status::Ok();
  if (numa_interleaved_) {
    return Status::FailedPrecondition(
        "VmRegion decommit: selected NUMA placement would be lost; release the region instead");
  }
  return vm_decommit(static_cast<std::byte*>(base_) + offset, len);
}

Status VmRegion::protect(std::size_t offset, std::size_t len,
                         Protect prot) noexcept {
  if (!range_ok(base_, size_, offset, len)) {
    return valid()
               ? Status::OutOfRange("VmRegion protect: range is out of bounds")
               : Status::FailedPrecondition("VmRegion protect: region is empty");
  }
  if (len == 0) return Status::Ok();
  return vm_protect(static_cast<std::byte*>(base_) + offset, len, prot);
}

Status VmRegion::commit_usable() noexcept {
  if (!valid()) {
    return Status::FailedPrecondition("VmRegion commit usable: region is empty");
  }
  if (guarded_) {
    return commit(page_size(), usable_size());
  }
  return commit(0, size_);
}

Status VmRegion::release() noexcept {
  if (base_ == nullptr) return Status::Ok();
  Status status = vm_release(base_, size_);
  if (!status.ok()) return status;
  base_ = nullptr;
  size_ = 0;
  guarded_ = false;
  numa_interleaved_ = false;
  numa_placement_ = NumaPlacement::kNotNeeded;
  return Status::Ok();
}

}  // namespace hyle::platform

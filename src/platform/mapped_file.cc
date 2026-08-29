#include "hyle/platform/mapped_file.h"

#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

#include "hyle/platform.h"
#include "hyle/platform/fs.h"

#if defined(HYLE_OS_WINDOWS)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/resource.h>
#  include <sys/stat.h>
#  if defined(HYLE_OS_LINUX)
#    include <sys/syscall.h>
#  endif
#  include <unistd.h>
#endif

namespace hyle::platform {

namespace {

bool valid_lock_options(MappedFile::LockOptions options) noexcept {
  return (options.mode == MappedFile::LockMode::kEager ||
          options.mode == MappedFile::LockMode::kOnFault) &&
         (options.limit_policy == MappedFile::LockLimitPolicy::kPreserve ||
          options.limit_policy ==
              MappedFile::LockLimitPolicy::kAdjustIfNeeded);
}

bool checked_add(std::size_t a, std::size_t b,
                 std::size_t* result) noexcept {
  if (a > (std::numeric_limits<std::size_t>::max)() - b) return false;
  *result = a + b;
  return true;
}

bool round_up(std::size_t value, std::size_t alignment,
              std::size_t* result) noexcept {
  if (alignment == 0) return false;
  const std::size_t remainder = value % alignment;
  if (remainder == 0) {
    *result = value;
    return true;
  }
  return checked_add(value, alignment - remainder, result);
}

}  // namespace

#if defined(HYLE_OS_WINDOWS)

// ---- Windows ----------------------------------------------------------------
namespace {
constexpr std::size_t kLockLimitSlack = 1u << 20;

Status win32_mapping_error(const char* operation, DWORD code) noexcept {
  return status_from_win32(code, operation);
}

Status adjust_working_set_limit(std::size_t lock_bytes) noexcept {
  SIZE_T minimum = 0;
  SIZE_T maximum = 0;
  HANDLE process = ::GetCurrentProcess();
  if (!::GetProcessWorkingSetSize(process, &minimum, &maximum)) {
    return win32_mapping_error("GetProcessWorkingSetSize(mapped file lock)",
                               ::GetLastError());
  }

  std::size_t increment = 0;
  if (!checked_add(lock_bytes, kLockLimitSlack, &increment) ||
      minimum > (std::numeric_limits<SIZE_T>::max)() - increment ||
      maximum > (std::numeric_limits<SIZE_T>::max)() - increment) {
    return Status::OutOfRange(
        "mapped file lock: process working-set limit would overflow");
  }
  if (!::SetProcessWorkingSetSize(process, minimum + increment,
                                  maximum + increment)) {
    return win32_mapping_error("SetProcessWorkingSetSize(mapped file lock)",
                               ::GetLastError());
  }
  return Status{};
}

}  // namespace

Result<MappedFile> MappedFile::open(const char* path, Mode mode,
                                    std::size_t offset,
                                    std::size_t length) noexcept {
  if (mode != Mode::kReadOnly && mode != Mode::kReadWrite &&
      mode != Mode::kCopyOnWrite) {
    return std::unexpected(
        Status::InvalidArgument("mapped file open: invalid mode"));
  }
  const unsigned flags = mode == Mode::kReadWrite ? kOpenRead | kOpenWrite
                                                   : kOpenRead;
  auto file = open_file(path, flags);
  if (!file) {
    return std::unexpected(
        std::move(file).error().WithContext("mapped file open"));
  }
  return map(std::move(*file), mode, offset, length);
}

Result<MappedFile> MappedFile::map(FileHandle file, Mode mode,
                                   std::size_t offset,
                                   std::size_t length) noexcept {
  MappedFile m;
  if (!file.valid()) {
    return std::unexpected(
        Status::FailedPrecondition("mapped file map: handle is closed"));
  }
  if (mode != Mode::kReadOnly && mode != Mode::kReadWrite &&
      mode != Mode::kCopyOnWrite) {
    return std::unexpected(
        Status::InvalidArgument("mapped file map: invalid mode"));
  }
  HANDLE native_file = reinterpret_cast<HANDLE>(file.get());

  LARGE_INTEGER fsize;
  if (!::GetFileSizeEx(native_file, &fsize)) {
    return std::unexpected(
        win32_mapping_error("GetFileSizeEx", ::GetLastError()));
  }
  const std::uint64_t file_size = static_cast<std::uint64_t>(fsize.QuadPart);
  if (offset > file_size) {
    return std::unexpected(
        Status::OutOfRange("mapped file map: offset is past end of file"));
  }
  // An explicit length must stay within the file: mapping past EOF would let a
  // read-write mapping SILENTLY GROW the file on disk here (CreateFileMapping
  // extends to max_size), and access to the past-EOF region faults on POSIX.
  // Reject it so both platforms agree and there is no surprise file growth.
  if (length != 0 && length > file_size - offset) {
    return std::unexpected(Status::OutOfRange(
        "mapped file map: requested range extends past end of file"));
  }
  std::size_t map_len = length != 0 ? length
                                    : static_cast<std::size_t>(file_size - offset);
  if (map_len == 0) {
    // Empty file: a zero-length view is not creatable; report a valid-but-empty
    // mapping as failure (callers checking valid() get a clear signal).
    return std::unexpected(
        Status::FailedPrecondition("mapped file map: mapping is empty"));
  }

  const DWORD prot = (mode == Mode::kReadWrite)      ? PAGE_READWRITE
                     : (mode == Mode::kCopyOnWrite)  ? PAGE_WRITECOPY
                                                     : PAGE_READONLY;
  // CreateFileMapping size = offset+len (max size the mapping can address).
  const std::uint64_t max_size = offset + map_len;
  HANDLE mapping = ::CreateFileMappingW(
      native_file, nullptr, prot, static_cast<DWORD>(max_size >> 32),
      static_cast<DWORD>(max_size & 0xFFFFFFFFu), nullptr);
  if (mapping == nullptr) {
    return std::unexpected(
        win32_mapping_error("CreateFileMappingW", ::GetLastError()));
  }

  const DWORD view = (mode == Mode::kReadWrite)     ? FILE_MAP_WRITE
                     : (mode == Mode::kCopyOnWrite) ? FILE_MAP_COPY
                                                    : FILE_MAP_READ;
  // MapViewOfFile's file offset must be a multiple of the allocation granularity
  // (64 KiB), NOT the page size. Round the offset DOWN to it, map the extra
  // leading bytes, and hand the caller a pointer to the exact requested offset.
  SYSTEM_INFO si;
  ::GetSystemInfo(&si);
  const std::uint64_t gran = si.dwAllocationGranularity;
  const std::uint64_t aligned_off = (offset / gran) * gran;
  const std::size_t delta = static_cast<std::size_t>(offset - aligned_off);
  if (map_len > (std::numeric_limits<std::size_t>::max)() - delta) {
    ::CloseHandle(mapping);
    return std::unexpected(
        Status::OutOfRange("mapped file map: native view is too large"));
  }
  void* base = ::MapViewOfFile(mapping, view,
                               static_cast<DWORD>(aligned_off >> 32),
                               static_cast<DWORD>(aligned_off & 0xFFFFFFFFu),
                               map_len + delta);
  if (base == nullptr) {
    const DWORD code = ::GetLastError();
    ::CloseHandle(mapping);
    return std::unexpected(win32_mapping_error("MapViewOfFile", code));
  }

  m.addr_ = static_cast<char*>(base) + delta;
  m.size_ = map_len;
  m.map_base_ = base;
  m.map_len_ = map_len + delta;
  m.h0_ = file.release();
  m.h1_ = reinterpret_cast<std::intptr_t>(mapping);
  m.mode_ = mode;
  return m;
}

Status MappedFile::flush(std::size_t offset, std::size_t length) noexcept {
  if (!valid()) {
    return Status::FailedPrecondition("mapped file flush: mapping is not open");
  }
  if (offset > size_ || (length != 0 && length > size_ - offset)) {
    return Status::OutOfRange("mapped file flush: range is outside mapping");
  }
  const std::size_t span = length != 0 ? length : size_ - offset;
  if (span == 0) return Status{};
  if (mode_ != Mode::kReadWrite) return Status();
  if (::FlushViewOfFile(static_cast<char*>(addr_) + offset, span) == 0) {
    return win32_mapping_error("FlushViewOfFile", ::GetLastError());
  }
  if (::FlushFileBuffers(reinterpret_cast<HANDLE>(h0_)) == 0) {
    return win32_mapping_error("FlushFileBuffers", ::GetLastError());
  }
  return Status();
}

Status MappedFile::advise(Advice advice, std::size_t offset,
                          std::size_t length) noexcept {
  if (!valid()) {
    return Status::FailedPrecondition("mapped file advise: mapping is not open");
  }
  if (advice < Advice::kNormal || advice > Advice::kDontNeed) {
    return Status::InvalidArgument("mapped file advise: invalid advice");
  }
  if (offset > size_ || (length != 0 && length > size_ - offset)) {
    return Status::OutOfRange("mapped file advise: range is outside mapping");
  }
  const std::size_t span = length != 0 ? length : size_ - offset;
  if (span == 0 || advice == Advice::kNormal) return Status{};

  if (advice != Advice::kWillNeed) {
    return Status::Unimplemented(
        "mapped file advice is not available on Windows");
  }

  // Resolve dynamically: PrefetchVirtualMemory was added in Windows 8, while
  // HYLE's binary may still be loaded on an older host. Avoid a hard import
  // that would prevent the whole DLL from loading there.
  struct MemoryRangeEntry {
    void* virtual_address;
    std::size_t number_of_bytes;
  };
  using PrefetchFn = BOOL(WINAPI*)(HANDLE, ULONG_PTR, MemoryRangeEntry*, ULONG);
  static const PrefetchFn prefetch = []() noexcept -> PrefetchFn {
    HMODULE kernel = ::GetModuleHandleW(L"kernel32.dll");
    return kernel == nullptr
               ? nullptr
               : reinterpret_cast<PrefetchFn>(
                     ::GetProcAddress(kernel, "PrefetchVirtualMemory"));
  }();
  if (prefetch == nullptr) {
    return Status::Unimplemented(
        "PrefetchVirtualMemory is unavailable on this Windows version");
  }
  MemoryRangeEntry range{static_cast<char*>(addr_) + offset, span};
  if (!prefetch(::GetCurrentProcess(), 1, &range, 0)) {
    return win32_mapping_error("PrefetchVirtualMemory", ::GetLastError());
  }
  return Status{};
}

Status MappedFile::lock_prefix(std::size_t length,
                               LockOptions options) noexcept {
  if (!valid()) {
    return Status::FailedPrecondition("mapped file lock: mapping is not open");
  }
  if (!valid_lock_options(options)) {
    return Status::InvalidArgument("mapped file lock: invalid options");
  }
  if (length > size_) {
    return Status::OutOfRange("mapped file lock: prefix is outside mapping");
  }
  if (length == 0) return Status{};
  if (locked_native_size_ != 0 && options.mode != locked_mode_) {
    return Status::FailedPrecondition(
        "mapped file lock: cannot change mode while a prefix is locked");
  }
  if (options.mode == LockMode::kOnFault) {
    return Status::Unimplemented(
        "mapped file on-fault locking is unavailable on Windows");
  }

  SYSTEM_INFO info;
  ::GetSystemInfo(&info);
  const std::size_t page = info.dwPageSize;
  const std::size_t leading = static_cast<std::size_t>(
      reinterpret_cast<std::uintptr_t>(addr_) -
      reinterpret_cast<std::uintptr_t>(map_base_));
  std::size_t visible_end = 0;
  std::size_t native_target = 0;
  if (!checked_add(leading, length, &visible_end) ||
      !round_up(visible_end, page, &native_target)) {
    return Status::OutOfRange("mapped file lock: native prefix is too large");
  }
  if (native_target <= locked_native_size_) return Status{};

  void* start = static_cast<char*>(map_base_) + locked_native_size_;
  const std::size_t delta = native_target - locked_native_size_;
  if (!::VirtualLock(start, delta)) {
    DWORD code = ::GetLastError();
    if (options.limit_policy == LockLimitPolicy::kAdjustIfNeeded &&
        code == ERROR_WORKING_SET_QUOTA) {
      Status adjusted = adjust_working_set_limit(delta);
      if (!adjusted.ok()) return adjusted;
      if (!::VirtualLock(start, delta)) {
        code = ::GetLastError();
      } else {
        code = ERROR_SUCCESS;
      }
    }
    if (code != ERROR_SUCCESS) {
      return win32_mapping_error("VirtualLock", code);
    }
  }
  locked_native_size_ = native_target;
  locked_mode_ = options.mode;
  return Status{};
}

Status MappedFile::unlock() noexcept {
  if (!valid()) {
    return Status::FailedPrecondition("mapped file unlock: mapping is not open");
  }
  if (locked_native_size_ == 0) return Status{};
  if (!::VirtualUnlock(map_base_, locked_native_size_)) {
    return win32_mapping_error("VirtualUnlock", ::GetLastError());
  }
  locked_native_size_ = 0;
  locked_mode_ = LockMode::kEager;
  return Status{};
}

void MappedFile::reset() noexcept {
  if (map_base_ != nullptr) ::UnmapViewOfFile(map_base_);
  if (h1_ != -1) ::CloseHandle(reinterpret_cast<HANDLE>(h1_));
  if (h0_ != -1) ::CloseHandle(reinterpret_cast<HANDLE>(h0_));
  addr_ = nullptr;
  size_ = 0;
  map_base_ = nullptr;
  map_len_ = 0;
  h0_ = -1;
  h1_ = -1;
  locked_native_size_ = 0;
  locked_mode_ = LockMode::kEager;
}

#else

// ---- POSIX ------------------------------------------------------------------
namespace {

Status posix_mapping_error(const char* operation, int code) noexcept {
  Status status = status_from_errno(code, operation);
  if (code != ENOTDIR) return status;
  return Status::FromNative(StatusCode::NotFound, status.native_error(),
                            status.message());
}

Status adjust_memlock_limit(std::size_t lock_bytes, bool* changed) noexcept {
  *changed = false;
#if defined(RLIMIT_MEMLOCK)
  struct ::rlimit limit;
  if (::getrlimit(RLIMIT_MEMLOCK, &limit) != 0) {
    return posix_mapping_error("getrlimit(RLIMIT_MEMLOCK)", errno);
  }
  if (limit.rlim_cur == RLIM_INFINITY) return Status{};

  const std::size_t increment = lock_bytes;
  if (increment > (std::numeric_limits<rlim_t>::max)()) {
    return Status::OutOfRange(
        "mapped file lock: process memlock limit would overflow");
  }
  rlim_t desired = limit.rlim_cur;
  const rlim_t native_increment = static_cast<rlim_t>(increment);
  if (desired > (std::numeric_limits<rlim_t>::max)() - native_increment) {
    desired = RLIM_INFINITY;
  } else {
    desired += native_increment;
  }
  if (limit.rlim_max != RLIM_INFINITY && desired > limit.rlim_max) {
    desired = limit.rlim_max;
  }
  if (desired <= limit.rlim_cur) return Status{};

  struct ::rlimit updated{desired, limit.rlim_max};
  if (::setrlimit(RLIMIT_MEMLOCK, &updated) != 0) {
    return posix_mapping_error("setrlimit(RLIMIT_MEMLOCK)", errno);
  }
  *changed = true;
  return Status{};
#else
  (void)lock_bytes;
  return Status::Unimplemented(
      "mapped file lock: process memlock limit is unavailable");
#endif
}

int lock_pages(const void* address, std::size_t length,
               MappedFile::LockMode mode) noexcept {
  if (mode == MappedFile::LockMode::kEager) {
    return ::mlock(address, length);
  }
#if defined(HYLE_OS_LINUX) && defined(SYS_mlock2)
  // MLOCK_ONFAULT is 0x01 in the Linux UAPI. Keeping the literal here avoids
  // requiring non-portable GNU/Linux headers in the public interface.
  return static_cast<int>(::syscall(SYS_mlock2, address, length, 0x01));
#else
  (void)address;
  (void)length;
  errno = ENOSYS;
  return -1;
#endif
}

}  // namespace

Result<MappedFile> MappedFile::open(const char* path, Mode mode,
                                    std::size_t offset,
                                    std::size_t length) noexcept {
  if (mode != Mode::kReadOnly && mode != Mode::kReadWrite &&
      mode != Mode::kCopyOnWrite) {
    return std::unexpected(
        Status::InvalidArgument("mapped file open: invalid mode"));
  }
  const unsigned flags = mode == Mode::kReadWrite ? kOpenRead | kOpenWrite
                                                   : kOpenRead;
  auto file = open_file(path, flags);
  if (!file) {
    return std::unexpected(
        std::move(file).error().WithContext("mapped file open"));
  }
  return map(std::move(*file), mode, offset, length);
}

Result<MappedFile> MappedFile::map(FileHandle file, Mode mode,
                                   std::size_t offset,
                                   std::size_t length) noexcept {
  MappedFile m;
  if (!file.valid()) {
    return std::unexpected(
        Status::FailedPrecondition("mapped file map: handle is closed"));
  }
  if (mode != Mode::kReadOnly && mode != Mode::kReadWrite &&
      mode != Mode::kCopyOnWrite) {
    return std::unexpected(
        Status::InvalidArgument("mapped file map: invalid mode"));
  }
  const int fd = static_cast<int>(file.get());
  struct ::stat st;
  if (::fstat(fd, &st) != 0) {
    return std::unexpected(posix_mapping_error("fstat", errno));
  }
  if (st.st_size < 0) {
    return std::unexpected(
        Status::OutOfRange("mapped file map: file has negative size"));
  }
  const std::uint64_t file_size = static_cast<std::uint64_t>(st.st_size);
  if (offset > file_size) {
    return std::unexpected(
        Status::OutOfRange("mapped file map: offset is past end of file"));
  }
  // Reject an explicit length past EOF: accessing the past-EOF region SIGBUSes
  // on POSIX (and would silently grow the file on Windows) -- keep both agreeing.
  if (length != 0 && length > file_size - offset) {
    return std::unexpected(Status::OutOfRange(
        "mapped file map: requested range extends past end of file"));
  }
  std::size_t map_len = length != 0 ? length
                                    : static_cast<std::size_t>(file_size - offset);
  if (map_len == 0) {
    return std::unexpected(
        Status::FailedPrecondition("mapped file map: mapping is empty"));
  }

  const int prot = (mode == Mode::kReadOnly) ? PROT_READ
                                             : (PROT_READ | PROT_WRITE);
  const int flags = (mode == Mode::kReadWrite) ? MAP_SHARED : MAP_PRIVATE;
  // mmap requires a page-aligned offset. Round DOWN, map the extra leading bytes,
  // and hand back a pointer to the exact requested offset (uniform with Windows).
  const long native_page = ::sysconf(_SC_PAGESIZE);
  if (native_page <= 0) {
    return std::unexpected(
        Status::Unavailable("mapped file map: page size is unavailable"));
  }
  const std::uint64_t pg = static_cast<std::uint64_t>(native_page);
  const std::uint64_t aligned_off = (offset / pg) * pg;
  const std::size_t delta = static_cast<std::size_t>(offset - aligned_off);
  if (map_len > (std::numeric_limits<std::size_t>::max)() - delta) {
    return std::unexpected(
        Status::OutOfRange("mapped file map: native view is too large"));
  }
  void* base = ::mmap(nullptr, map_len + delta, prot, flags, fd,
                      static_cast<off_t>(aligned_off));
  const int mmap_error = base == MAP_FAILED ? errno : 0;
  // The fd can be closed right after mmap: the mapping keeps its own reference.
  if (base == MAP_FAILED) {
    return std::unexpected(posix_mapping_error("mmap", mmap_error));
  }

  m.addr_ = static_cast<char*>(base) + delta;
  m.size_ = map_len;
  m.map_base_ = base;
  m.map_len_ = map_len + delta;
  m.mode_ = mode;
  return m;
}

Status MappedFile::flush(std::size_t offset, std::size_t length) noexcept {
  if (!valid()) {
    return Status::FailedPrecondition("mapped file flush: mapping is not open");
  }
  if (offset > size_ || (length != 0 && length > size_ - offset)) {
    return Status::OutOfRange("mapped file flush: range is outside mapping");
  }
  const std::size_t span = length != 0 ? length : size_ - offset;
  if (span == 0) return Status{};
  if (mode_ != Mode::kReadWrite) return Status();
  // msync needs a page-aligned start. Include only the leading fragment of the
  // first page; the kernel rounds the tail as required.
  const long native_page = ::sysconf(_SC_PAGESIZE);
  if (native_page <= 0) {
    return Status::Unavailable("mapped file flush: page size is unavailable");
  }
  const std::size_t page = static_cast<std::size_t>(native_page);
  const std::uintptr_t requested =
      reinterpret_cast<std::uintptr_t>(addr_) + offset;
  const std::uintptr_t aligned = requested - requested % page;
  const std::size_t native_span = span + (requested - aligned);
  if (::msync(reinterpret_cast<void*>(aligned), native_span, MS_SYNC) != 0) {
    return posix_mapping_error("msync", errno);
  }
  return Status();
}

Status MappedFile::advise(Advice advice, std::size_t offset,
                          std::size_t length) noexcept {
  if (!valid()) {
    return Status::FailedPrecondition("mapped file advise: mapping is not open");
  }
  if (advice < Advice::kNormal || advice > Advice::kDontNeed) {
    return Status::InvalidArgument("mapped file advise: invalid advice");
  }
  if (offset > size_ || (length != 0 && length > size_ - offset)) {
    return Status::OutOfRange("mapped file advise: range is outside mapping");
  }
  const std::size_t span = length != 0 ? length : size_ - offset;
  if (span == 0) return Status{};

  int native = MADV_NORMAL;
  switch (advice) {
    case Advice::kNormal: native = MADV_NORMAL; break;
    case Advice::kRandom: native = MADV_RANDOM; break;
    case Advice::kSequential: native = MADV_SEQUENTIAL; break;
    case Advice::kWillNeed: native = MADV_WILLNEED; break;
    case Advice::kDontNeed: native = MADV_DONTNEED; break;
  }

  // madvise requires a page-aligned start. Mapping an unaligned file offset
  // created a leading prefix in map_base_; include only the unavoidable prefix
  // of the first page and keep the request inside the underlying view.
  const long native_page = ::sysconf(_SC_PAGESIZE);
  const std::size_t page =
      native_page > 0 ? static_cast<std::size_t>(native_page) : 4096u;
  const std::uintptr_t requested =
      reinterpret_cast<std::uintptr_t>(addr_) + offset;
  const std::uintptr_t aligned = requested - requested % page;
  const std::size_t native_span = span + (requested - aligned);
  if (::madvise(reinterpret_cast<void*>(aligned), native_span, native) != 0) {
    return posix_mapping_error("madvise", errno);
  }
  return Status{};
}

Status MappedFile::lock_prefix(std::size_t length,
                               LockOptions options) noexcept {
  if (!valid()) {
    return Status::FailedPrecondition("mapped file lock: mapping is not open");
  }
  if (!valid_lock_options(options)) {
    return Status::InvalidArgument("mapped file lock: invalid options");
  }
  if (length > size_) {
    return Status::OutOfRange("mapped file lock: prefix is outside mapping");
  }
  if (length == 0) return Status{};
  if (locked_native_size_ != 0 && options.mode != locked_mode_) {
    return Status::FailedPrecondition(
        "mapped file lock: cannot change mode while a prefix is locked");
  }

  const long native_page = ::sysconf(_SC_PAGESIZE);
  if (native_page <= 0) {
    return Status::Unavailable("mapped file lock: page size is unavailable");
  }
  const std::size_t page = static_cast<std::size_t>(native_page);
  const std::size_t leading = static_cast<std::size_t>(
      reinterpret_cast<std::uintptr_t>(addr_) -
      reinterpret_cast<std::uintptr_t>(map_base_));
  std::size_t visible_end = 0;
  std::size_t native_target = 0;
  if (!checked_add(leading, length, &visible_end) ||
      !round_up(visible_end, page, &native_target)) {
    return Status::OutOfRange("mapped file lock: native prefix is too large");
  }
  if (native_target <= locked_native_size_) return Status{};

  const void* start = static_cast<const char*>(map_base_) +
                      locked_native_size_;
  const std::size_t delta = native_target - locked_native_size_;
  if (lock_pages(start, delta, options.mode) != 0) {
    int code = errno;
    if (options.mode == LockMode::kOnFault &&
        (code == ENOSYS || code == EINVAL)) {
      Status native = posix_mapping_error("mlock2(MLOCK_ONFAULT)", code);
      return Status::FromNative(StatusCode::Unimplemented,
                                native.native_error(), native.message());
    }
    if (options.limit_policy == LockLimitPolicy::kAdjustIfNeeded &&
        (code == ENOMEM || code == EAGAIN)) {
      bool changed = false;
      Status adjusted = adjust_memlock_limit(delta, &changed);
      if (!adjusted.ok()) return adjusted;
      if (changed && lock_pages(start, delta, options.mode) == 0) {
        code = 0;
      } else if (changed) {
        code = errno;
      }
    }
    if (code != 0) {
      return posix_mapping_error(options.mode == LockMode::kEager
                                     ? "mlock"
                                     : "mlock2(MLOCK_ONFAULT)",
                                 code);
    }
  }
  locked_native_size_ = native_target;
  locked_mode_ = options.mode;
  return Status{};
}

Status MappedFile::unlock() noexcept {
  if (!valid()) {
    return Status::FailedPrecondition("mapped file unlock: mapping is not open");
  }
  if (locked_native_size_ == 0) return Status{};
  if (::munlock(map_base_, locked_native_size_) != 0) {
    return posix_mapping_error("munlock", errno);
  }
  locked_native_size_ = 0;
  locked_mode_ = LockMode::kEager;
  return Status{};
}

void MappedFile::reset() noexcept {
  if (map_base_ != nullptr) ::munmap(map_base_, map_len_);
  addr_ = nullptr;
  size_ = 0;
  map_base_ = nullptr;
  map_len_ = 0;
  locked_native_size_ = 0;
  locked_mode_ = LockMode::kEager;
}

#endif

// ---- portable ---------------------------------------------------------------

Status MappedFile::lock(LockOptions options) noexcept {
  return lock_prefix(size_, options);
}

Result<std::span<std::byte>> MappedFile::writable_bytes() noexcept {
  if (!valid()) {
    return std::unexpected(Status::FailedPrecondition(
        "mapped file writable bytes: mapping is not open"));
  }
  if (mode_ == Mode::kReadOnly) {
    return std::unexpected(Status::FailedPrecondition(
        "mapped file writable bytes: mapping is read-only"));
  }
  return std::span<std::byte>{static_cast<std::byte*>(addr_), size_};
}

MappedFile::~MappedFile() { reset(); }

MappedFile::MappedFile(MappedFile&& o) noexcept
    : addr_(o.addr_), size_(o.size_), map_base_(o.map_base_),
      map_len_(o.map_len_), h0_(o.h0_), h1_(o.h1_), mode_(o.mode_),
      locked_native_size_(o.locked_native_size_),
      locked_mode_(o.locked_mode_) {
  o.addr_ = nullptr;
  o.size_ = 0;
  o.map_base_ = nullptr;
  o.map_len_ = 0;
  o.h0_ = -1;
  o.h1_ = -1;
  o.locked_native_size_ = 0;
  o.locked_mode_ = LockMode::kEager;
}

MappedFile& MappedFile::operator=(MappedFile&& o) noexcept {
  if (this != &o) {
    reset();
    addr_ = o.addr_;
    size_ = o.size_;
    map_base_ = o.map_base_;
    map_len_ = o.map_len_;
    h0_ = o.h0_;
    h1_ = o.h1_;
    mode_ = o.mode_;
    locked_native_size_ = o.locked_native_size_;
    locked_mode_ = o.locked_mode_;
    o.addr_ = nullptr;
    o.size_ = 0;
    o.map_base_ = nullptr;
    o.map_len_ = 0;
    o.h0_ = -1;
    o.h1_ = -1;
    o.locked_native_size_ = 0;
    o.locked_mode_ = LockMode::kEager;
  }
  return *this;
}

}  // namespace hyle::platform

#include "hyle/platform/mapped_file.h"

#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <new>
#include <utility>

#include "hyle/platform.h"

#if defined(HYLE_OS_WINDOWS)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace hyle::platform {

#if defined(HYLE_OS_WINDOWS)

// ---- Windows ----------------------------------------------------------------
namespace {
const char* win32_mapping_error_name(DWORD code) noexcept {
  switch (code) {
    case ERROR_FILE_NOT_FOUND: return "ERROR_FILE_NOT_FOUND";
    case ERROR_PATH_NOT_FOUND: return "ERROR_PATH_NOT_FOUND";
    case ERROR_ACCESS_DENIED: return "ERROR_ACCESS_DENIED";
    case ERROR_NOT_ENOUGH_MEMORY: return "ERROR_NOT_ENOUGH_MEMORY";
    case ERROR_OUTOFMEMORY: return "ERROR_OUTOFMEMORY";
    case ERROR_INVALID_PARAMETER: return "ERROR_INVALID_PARAMETER";
    default: return "WIN32_MAPPING_ERROR";
  }
}

Status win32_mapping_error(const char* operation, DWORD code) noexcept {
  char description[192]{};
  DWORD length = ::FormatMessageA(
      FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, code,
      0, description, static_cast<DWORD>(sizeof(description)), nullptr);
  while (length != 0 &&
         (description[length - 1] == '\r' || description[length - 1] == '\n')) {
    description[--length] = '\0';
  }
  char message[320];
  std::snprintf(message, sizeof(message),
                "windows %s failed: %s (%lu)%s%s", operation,
                win32_mapping_error_name(code), static_cast<unsigned long>(code),
                length == 0 ? "" : ": ", length == 0 ? "" : description);
  if (code == ERROR_FILE_NOT_FOUND || code == ERROR_PATH_NOT_FOUND) {
    return Status::NotFound(message);
  }
  if (code == ERROR_ACCESS_DENIED) return Status::PermissionDenied(message);
  if (code == ERROR_NOT_ENOUGH_MEMORY || code == ERROR_OUTOFMEMORY) {
    return Status::ResourceExhausted(message);
  }
  if (code == ERROR_INVALID_PARAMETER) return Status::InvalidArgument(message);
  return Status::Unavailable(message);
}

// UTF-8 -> UTF-16 for CreateFileW. Dynamically sized so conversion itself does
// not impose an artificial path limit. nullptr on invalid UTF-8 / OOM.
struct WBuf {
  wchar_t stack[512];
  wchar_t* ptr = nullptr;
  wchar_t* heap = nullptr;
  bool out_of_memory = false;
  explicit WBuf(const char* utf8) noexcept {
    if (utf8 == nullptr) return;
    const int need = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8,
                                         -1, nullptr, 0);
    if (need <= 0) return;
    ptr = stack;
    if (need > static_cast<int>(sizeof(stack) / sizeof(stack[0]))) {
      heap = new (std::nothrow) wchar_t[static_cast<std::size_t>(need)];
      ptr = heap;
      out_of_memory = ptr == nullptr;
    }
    if (ptr == nullptr ||
        MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8, -1, ptr,
                            need) != need) {
      delete[] heap;
      heap = ptr = nullptr;
    }
  }
  ~WBuf() { delete[] heap; }
  WBuf(const WBuf&) = delete;
  WBuf& operator=(const WBuf&) = delete;
};
}  // namespace

Result<MappedFile> MappedFile::open(const char* path, Mode mode,
                                    std::size_t offset,
                                    std::size_t length) noexcept {
  MappedFile m;
  if (path == nullptr || path[0] == '\0') {
    return std::unexpected(
        Status::InvalidArgument("mapped file open: empty path"));
  }
  if (mode != Mode::kReadOnly && mode != Mode::kReadWrite &&
      mode != Mode::kCopyOnWrite) {
    return std::unexpected(
        Status::InvalidArgument("mapped file open: invalid mode"));
  }
  WBuf w(path);
  if (!w.ptr) {
    return std::unexpected(
        w.out_of_memory
            ? Status::ResourceExhausted(
                  "mapped file open: path conversion allocation failed")
            : Status::InvalidArgument(
                  "mapped file open: path is not valid UTF-8"));
  }

  const DWORD access = (mode == Mode::kReadWrite)
                           ? (GENERIC_READ | GENERIC_WRITE)
                           : GENERIC_READ;
  HANDLE file = ::CreateFileW(w.ptr, access, FILE_SHARE_READ | FILE_SHARE_WRITE,
                              nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                              nullptr);
  if (file == INVALID_HANDLE_VALUE) {
    return std::unexpected(
        win32_mapping_error("CreateFileW", ::GetLastError()));
  }

  LARGE_INTEGER fsize;
  if (!::GetFileSizeEx(file, &fsize)) {
    const DWORD code = ::GetLastError();
    ::CloseHandle(file);
    return std::unexpected(win32_mapping_error("GetFileSizeEx", code));
  }
  const std::uint64_t file_size = static_cast<std::uint64_t>(fsize.QuadPart);
  if (offset > file_size) {
    ::CloseHandle(file);
    return std::unexpected(
        Status::OutOfRange("mapped file open: offset is past end of file"));
  }
  // An explicit length must stay within the file: mapping past EOF would let a
  // read-write mapping SILENTLY GROW the file on disk here (CreateFileMapping
  // extends to max_size), and access to the past-EOF region faults on POSIX.
  // Reject it so both platforms agree and there is no surprise file growth.
  if (length != 0 && length > file_size - offset) {
    ::CloseHandle(file);
    return std::unexpected(Status::OutOfRange(
        "mapped file open: requested range extends past end of file"));
  }
  std::size_t map_len = length != 0 ? length
                                    : static_cast<std::size_t>(file_size - offset);
  if (map_len == 0) {
    // Empty file: a zero-length view is not creatable; report a valid-but-empty
    // mapping as failure (callers checking valid() get a clear signal).
    ::CloseHandle(file);
    return std::unexpected(
        Status::FailedPrecondition("mapped file open: mapping is empty"));
  }

  const DWORD prot = (mode == Mode::kReadWrite)      ? PAGE_READWRITE
                     : (mode == Mode::kCopyOnWrite)  ? PAGE_WRITECOPY
                                                     : PAGE_READONLY;
  // CreateFileMapping size = offset+len (max size the mapping can address).
  const std::uint64_t max_size = offset + map_len;
  HANDLE mapping = ::CreateFileMappingW(
      file, nullptr, prot, static_cast<DWORD>(max_size >> 32),
      static_cast<DWORD>(max_size & 0xFFFFFFFFu), nullptr);
  if (mapping == nullptr) {
    const DWORD code = ::GetLastError();
    ::CloseHandle(file);
    return std::unexpected(win32_mapping_error("CreateFileMappingW", code));
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
  void* base = ::MapViewOfFile(mapping, view,
                               static_cast<DWORD>(aligned_off >> 32),
                               static_cast<DWORD>(aligned_off & 0xFFFFFFFFu),
                               map_len + delta);
  if (base == nullptr) {
    const DWORD code = ::GetLastError();
    ::CloseHandle(mapping);
    ::CloseHandle(file);
    return std::unexpected(win32_mapping_error("MapViewOfFile", code));
  }

  m.addr_ = static_cast<char*>(base) + delta;
  m.size_ = map_len;
  m.map_base_ = base;
  m.map_len_ = map_len + delta;
  m.h0_ = reinterpret_cast<std::intptr_t>(file);
  m.h1_ = reinterpret_cast<std::intptr_t>(mapping);
  m.mode_ = mode;
  return m;
}

Status MappedFile::flush() noexcept {
  if (!valid()) {
    return Status::FailedPrecondition("mapped file flush: mapping is not open");
  }
  if (mode_ != Mode::kReadWrite) return Status();
  if (::FlushViewOfFile(addr_, size_) == 0) {
    return win32_mapping_error("FlushViewOfFile", ::GetLastError());
  }
  if (::FlushFileBuffers(reinterpret_cast<HANDLE>(h0_)) == 0) {
    return win32_mapping_error("FlushFileBuffers", ::GetLastError());
  }
  return Status();
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
}

#else

// ---- POSIX ------------------------------------------------------------------
namespace {

const char* posix_mapping_error_name(int code) noexcept {
  switch (code) {
    case ENOENT: return "ENOENT";
    case EACCES: return "EACCES";
    case EPERM: return "EPERM";
    case ENOMEM: return "ENOMEM";
    case EMFILE: return "EMFILE";
    case ENFILE: return "ENFILE";
    case EINVAL: return "EINVAL";
    case EOVERFLOW: return "EOVERFLOW";
    default: return "ERRNO_UNKNOWN";
  }
}

[[maybe_unused]] const char* strerror_text(int result, char* buffer) noexcept {
  return result == 0 ? buffer : nullptr;
}

[[maybe_unused]] const char* strerror_text(char* result, char*) noexcept {
  return result;
}

Status posix_mapping_error(const char* operation, int code) noexcept {
  char description[192]{};
  const char* detail = strerror_text(
      ::strerror_r(code, description, sizeof(description)), description);
  char message[320];
  std::snprintf(message, sizeof(message), "posix %s failed: %s (%d)%s%s",
                operation, posix_mapping_error_name(code), code,
                detail == nullptr || detail[0] == '\0' ? "" : ": ",
                detail == nullptr ? "" : detail);
  if (code == ENOENT) return Status::NotFound(message);
  if (code == EACCES || code == EPERM) return Status::PermissionDenied(message);
  if (code == ENOMEM || code == EMFILE || code == ENFILE) {
    return Status::ResourceExhausted(message);
  }
  if (code == EINVAL || code == EOVERFLOW) {
    return Status::InvalidArgument(message);
  }
  return Status::Unavailable(message);
}

}  // namespace

Result<MappedFile> MappedFile::open(const char* path, Mode mode,
                                    std::size_t offset,
                                    std::size_t length) noexcept {
  MappedFile m;
  if (path == nullptr || path[0] == '\0') {
    return std::unexpected(
        Status::InvalidArgument("mapped file open: empty path"));
  }
  if (mode != Mode::kReadOnly && mode != Mode::kReadWrite &&
      mode != Mode::kCopyOnWrite) {
    return std::unexpected(
        Status::InvalidArgument("mapped file open: invalid mode"));
  }
  // kReadWrite needs write access; RO/COW only read.
  const int oflags = (mode == Mode::kReadWrite) ? O_RDWR : O_RDONLY;
  int fd = ::open(path, oflags | O_CLOEXEC);
  if (fd < 0) {
    return std::unexpected(posix_mapping_error("open", errno));
  }
  struct ::stat st;
  if (::fstat(fd, &st) != 0) {
    const int code = errno;
    ::close(fd);
    return std::unexpected(posix_mapping_error("fstat", code));
  }
  const std::uint64_t file_size = static_cast<std::uint64_t>(st.st_size);
  if (offset > file_size) {
    ::close(fd);
    return std::unexpected(
        Status::OutOfRange("mapped file open: offset is past end of file"));
  }
  // Reject an explicit length past EOF: accessing the past-EOF region SIGBUSes
  // on POSIX (and would silently grow the file on Windows) -- keep both agreeing.
  if (length != 0 && length > file_size - offset) {
    ::close(fd);
    return std::unexpected(Status::OutOfRange(
        "mapped file open: requested range extends past end of file"));
  }
  std::size_t map_len = length != 0 ? length
                                    : static_cast<std::size_t>(file_size - offset);
  if (map_len == 0) {
    ::close(fd);
    return std::unexpected(
        Status::FailedPrecondition("mapped file open: mapping is empty"));
  }

  const int prot = (mode == Mode::kReadOnly) ? PROT_READ
                                             : (PROT_READ | PROT_WRITE);
  const int flags = (mode == Mode::kReadWrite) ? MAP_SHARED : MAP_PRIVATE;
  // mmap requires a page-aligned offset. Round DOWN, map the extra leading bytes,
  // and hand back a pointer to the exact requested offset (uniform with Windows).
  const std::uint64_t pg = static_cast<std::uint64_t>(::sysconf(_SC_PAGESIZE));
  const std::uint64_t aligned_off = (offset / pg) * pg;
  const std::size_t delta = static_cast<std::size_t>(offset - aligned_off);
  void* base = ::mmap(nullptr, map_len + delta, prot, flags, fd,
                      static_cast<off_t>(aligned_off));
  const int mmap_error = base == MAP_FAILED ? errno : 0;
  // The fd can be closed right after mmap: the mapping keeps its own reference.
  ::close(fd);
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

Status MappedFile::flush() noexcept {
  if (!valid()) {
    return Status::FailedPrecondition("mapped file flush: mapping is not open");
  }
  if (mode_ != Mode::kReadWrite) return Status();
  // msync needs a page-aligned address -> sync the whole underlying mapping.
  if (::msync(map_base_, map_len_, MS_SYNC) != 0) {
    return posix_mapping_error("msync", errno);
  }
  return Status();
}

void MappedFile::reset() noexcept {
  if (map_base_ != nullptr) ::munmap(map_base_, map_len_);
  addr_ = nullptr;
  size_ = 0;
  map_base_ = nullptr;
  map_len_ = 0;
}

#endif

// ---- portable ---------------------------------------------------------------

MappedFile::~MappedFile() { reset(); }

MappedFile::MappedFile(MappedFile&& o) noexcept
    : addr_(o.addr_), size_(o.size_), map_base_(o.map_base_),
      map_len_(o.map_len_), h0_(o.h0_), h1_(o.h1_), mode_(o.mode_) {
  o.addr_ = nullptr;
  o.size_ = 0;
  o.map_base_ = nullptr;
  o.map_len_ = 0;
  o.h0_ = -1;
  o.h1_ = -1;
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
    o.addr_ = nullptr;
    o.size_ = 0;
    o.map_base_ = nullptr;
    o.map_len_ = 0;
    o.h0_ = -1;
    o.h1_ = -1;
  }
  return *this;
}

}  // namespace hyle::platform

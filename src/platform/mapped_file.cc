#include "hyle/platform/mapped_file.h"

#include <cstdint>
#include <new>
#include <utility>

#include "hyle/platform.h"
#include "hyle/platform/virtual_memory.h"  // vm_set_last_error (shared error channel)

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
// UTF-8 -> UTF-16 for CreateFileW. Dynamically sized so conversion itself does
// not impose an artificial path limit. nullptr on invalid UTF-8 / OOM.
struct WBuf {
  wchar_t stack[512];
  wchar_t* ptr = nullptr;
  wchar_t* heap = nullptr;
  explicit WBuf(const char* utf8) noexcept {
    if (utf8 == nullptr) return;
    const int need = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8,
                                         -1, nullptr, 0);
    if (need <= 0) return;
    ptr = stack;
    if (need > static_cast<int>(sizeof(stack) / sizeof(stack[0]))) {
      heap = new (std::nothrow) wchar_t[static_cast<std::size_t>(need)];
      ptr = heap;
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

MappedFile MappedFile::open(const char* path, Mode mode, std::size_t offset,
                            std::size_t length) noexcept {
  vm_set_last_error(nullptr);
  MappedFile m;
  if (path == nullptr) {
    vm_set_last_error("MappedFile::open: null path");
    return m;
  }
  WBuf w(path);
  if (!w.ptr) {
    vm_set_last_error("MappedFile::open: path conversion failed");
    return m;
  }

  const DWORD access = (mode == Mode::kReadWrite)
                           ? (GENERIC_READ | GENERIC_WRITE)
                           : GENERIC_READ;
  HANDLE file = ::CreateFileW(w.ptr, access, FILE_SHARE_READ | FILE_SHARE_WRITE,
                              nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                              nullptr);
  if (file == INVALID_HANDLE_VALUE) {
    vm_set_last_error("MappedFile::open: CreateFile failed");
    return m;
  }

  LARGE_INTEGER fsize;
  if (!::GetFileSizeEx(file, &fsize)) {
    ::CloseHandle(file);
    vm_set_last_error("MappedFile::open: GetFileSizeEx failed");
    return m;
  }
  const std::uint64_t file_size = static_cast<std::uint64_t>(fsize.QuadPart);
  if (offset > file_size) {
    ::CloseHandle(file);
    vm_set_last_error("MappedFile::open: offset past end of file");
    return m;
  }
  // An explicit length must stay within the file: mapping past EOF would let a
  // read-write mapping SILENTLY GROW the file on disk here (CreateFileMapping
  // extends to max_size), and access to the past-EOF region faults on POSIX.
  // Reject it so both platforms agree and there is no surprise file growth.
  if (length != 0 && length > file_size - offset) {
    ::CloseHandle(file);
    vm_set_last_error("MappedFile::open: offset+length past end of file");
    return m;
  }
  std::size_t map_len = length != 0 ? length
                                    : static_cast<std::size_t>(file_size - offset);
  if (map_len == 0) {
    // Empty file: a zero-length view is not creatable; report a valid-but-empty
    // mapping as failure (callers checking valid() get a clear signal).
    ::CloseHandle(file);
    vm_set_last_error("MappedFile::open: zero-length mapping");
    return m;
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
    ::CloseHandle(file);
    vm_set_last_error("MappedFile::open: CreateFileMapping failed");
    return m;
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
    ::CloseHandle(mapping);
    ::CloseHandle(file);
    vm_set_last_error("MappedFile::open: MapViewOfFile failed");
    return m;
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

bool MappedFile::flush() noexcept {
  if (!valid() || mode_ != Mode::kReadWrite) return true;
  if (::FlushViewOfFile(addr_, size_) == 0) {
    vm_set_last_error("MappedFile::flush: FlushViewOfFile failed");
    return false;
  }
  if (::FlushFileBuffers(reinterpret_cast<HANDLE>(h0_)) == 0) {
    vm_set_last_error("MappedFile::flush: FlushFileBuffers failed");
    return false;
  }
  return true;
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

MappedFile MappedFile::open(const char* path, Mode mode, std::size_t offset,
                            std::size_t length) noexcept {
  vm_set_last_error(nullptr);
  MappedFile m;
  if (path == nullptr) {
    vm_set_last_error("MappedFile::open: null path");
    return m;
  }
  // kReadWrite needs write access; RO/COW only read.
  const int oflags = (mode == Mode::kReadWrite) ? O_RDWR : O_RDONLY;
  int fd = ::open(path, oflags | O_CLOEXEC);
  if (fd < 0) {
    vm_set_last_error("MappedFile::open: open failed");
    return m;
  }
  struct ::stat st;
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    vm_set_last_error("MappedFile::open: fstat failed");
    return m;
  }
  const std::uint64_t file_size = static_cast<std::uint64_t>(st.st_size);
  if (offset > file_size) {
    ::close(fd);
    vm_set_last_error("MappedFile::open: offset past end of file");
    return m;
  }
  // Reject an explicit length past EOF: accessing the past-EOF region SIGBUSes
  // on POSIX (and would silently grow the file on Windows) -- keep both agreeing.
  if (length != 0 && length > file_size - offset) {
    ::close(fd);
    vm_set_last_error("MappedFile::open: offset+length past end of file");
    return m;
  }
  std::size_t map_len = length != 0 ? length
                                    : static_cast<std::size_t>(file_size - offset);
  if (map_len == 0) {
    ::close(fd);
    vm_set_last_error("MappedFile::open: zero-length mapping");
    return m;
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
  // The fd can be closed right after mmap: the mapping keeps its own reference.
  ::close(fd);
  if (base == MAP_FAILED) {
    vm_set_last_error("MappedFile::open: mmap failed");
    return m;
  }

  m.addr_ = static_cast<char*>(base) + delta;
  m.size_ = map_len;
  m.map_base_ = base;
  m.map_len_ = map_len + delta;
  m.mode_ = mode;
  return m;
}

bool MappedFile::flush() noexcept {
  if (!valid() || mode_ != Mode::kReadWrite) return true;
  // msync needs a page-aligned address -> sync the whole underlying mapping.
  if (::msync(map_base_, map_len_, MS_SYNC) != 0) {
    vm_set_last_error("MappedFile::flush: msync failed");
    return false;
  }
  return true;
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

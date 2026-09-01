#include "hyle/io/shared_memory.h"

#include <limits>
#include <utility>

#include "hyle/core/environment.h"

#if HYLE_OS_WINDOWS
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <cerrno>
#  include <cstdio>
#  include <cstring>
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace hyle::io {
namespace {

#if HYLE_OS_WINDOWS
struct MappingHeader {
  std::uint64_t magic;
  std::uint64_t size;
};
constexpr std::uint64_t kMappingMagic = 0x48594c4553484d31ull;  // HYLESHM1

bool utf8_to_wide(const char* text, wchar_t* out, int cap) noexcept {
  if (text == nullptr || *text == '\0') return false;
  return ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text, -1, out,
                               cap) > 0;
}
#else
bool normalize_name(const char* name, char* out, std::size_t cap) noexcept {
  if (name == nullptr || *name == '\0' || cap < 3) return false;
  std::size_t n = std::strlen(name);
  const bool slash = name[0] == '/';
  if (n + (slash ? 1 : 2) > cap) return false;
  std::size_t at = 0;
  if (!slash) out[at++] = '/';
  std::memcpy(out + at, name, n + 1);
  return true;
}
#endif

}  // namespace

SharedMemory SharedMemory::create(const char* name, std::size_t size) noexcept {
  SharedMemory out;
  if (size == 0 || size > SIZE_MAX - 16) return out;
#if HYLE_OS_WINDOWS
  wchar_t wide[260];
  if (!utf8_to_wide(name, wide, 260)) return out;
  const DWORD high = static_cast<DWORD>(
      ((static_cast<std::uint64_t>(size) + sizeof(MappingHeader)) >> 32) &
      0xffffffffu);
  const DWORD low =
      static_cast<DWORD>((static_cast<std::uint64_t>(size) +
                          sizeof(MappingHeader)) &
                         0xffffffffu);
  HANDLE h = ::CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                                  high, low, wide);
  if (h == nullptr) return out;
  if (::GetLastError() == ERROR_ALREADY_EXISTS) {
    ::CloseHandle(h);
    return out;
  }
  void* p = ::MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0,
                            size + sizeof(MappingHeader));
  if (p == nullptr) {
    ::CloseHandle(h);
    return out;
  }
  auto* header = static_cast<MappingHeader*>(p);
  header->magic = kMappingMagic;
  header->size = size;
  out.mapping_ = p;
  out.data_ = header + 1;
  out.size_ = size;
  out.handle_ = reinterpret_cast<std::intptr_t>(h);
#else
  if (size > static_cast<std::size_t>((std::numeric_limits<off_t>::max)()))
    return out;
  char normalized[256];
  if (!normalize_name(name, normalized, sizeof(normalized))) return out;
  int fd = ::shm_open(normalized, O_RDWR | O_CREAT | O_EXCL, 0600);
  if (fd < 0) return out;
  if (::ftruncate(fd, static_cast<off_t>(size)) != 0) {
    ::close(fd);
    ::shm_unlink(normalized);
    return out;
  }
  void* p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ::close(fd);
  if (p == MAP_FAILED) {
    ::shm_unlink(normalized);
    return out;
  }
  out.mapping_ = p;
  out.data_ = p;
  out.size_ = size;
#endif
  return out;
}

SharedMemory SharedMemory::open(const char* name) noexcept {
  SharedMemory out;
#if HYLE_OS_WINDOWS
  wchar_t wide[260];
  if (!utf8_to_wide(name, wide, 260)) return out;
  HANDLE h = ::OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, wide);
  if (h == nullptr) return out;
  void* p =
      ::MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(MappingHeader));
  if (p == nullptr) {
    ::CloseHandle(h);
    return out;
  }
  auto* header = static_cast<const MappingHeader*>(p);
  if (header->magic != kMappingMagic || header->size == 0 ||
      header->size > SIZE_MAX - sizeof(MappingHeader)) {
    ::UnmapViewOfFile(p);
    ::CloseHandle(h);
    return out;
  }
  const std::size_t size = static_cast<std::size_t>(header->size);
  ::UnmapViewOfFile(p);
  p = ::MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0,
                      size + sizeof(MappingHeader));
  if (p == nullptr) {
    ::CloseHandle(h);
    return out;
  }
  header = static_cast<const MappingHeader*>(p);
  if (header->magic != kMappingMagic || header->size != size) {
    ::UnmapViewOfFile(p);
    ::CloseHandle(h);
    return out;
  }
  out.mapping_ = p;
  out.data_ = const_cast<MappingHeader*>(header) + 1;
  out.size_ = size;
  out.handle_ = reinterpret_cast<std::intptr_t>(h);
#else
  char normalized[256];
  if (!normalize_name(name, normalized, sizeof(normalized))) return out;
  int fd = ::shm_open(normalized, O_RDWR, 0600);
  if (fd < 0) return out;
  struct stat st {};
  if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
    ::close(fd);
    return out;
  }
  if (static_cast<std::uintmax_t>(st.st_size) >
      (std::numeric_limits<std::size_t>::max)()) {
    ::close(fd);
    return out;
  }
  const std::size_t size = static_cast<std::size_t>(st.st_size);
  void* p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ::close(fd);
  if (p == MAP_FAILED) return out;
  out.mapping_ = p;
  out.data_ = p;
  out.size_ = size;
#endif
  return out;
}

Status SharedMemory::unlink(const char* name) noexcept {
#if HYLE_OS_WINDOWS
  if (name == nullptr || *name == '\0')
    return Status::InvalidArgument("shared memory name is empty");
  return Status::Unimplemented(
      "Windows named mappings cannot be unlinked; the name disappears after "
      "the final handle closes");
#else
  char normalized[256];
  if (!normalize_name(name, normalized, sizeof(normalized)))
    return Status::InvalidArgument("shared memory name is invalid");
  if (::shm_unlink(normalized) == 0 || errno == ENOENT) return Status();
  return Status::Unavailable("shm_unlink failed");
#endif
}

bool SharedMemory::flush() noexcept {
  if (!valid()) return false;
#if HYLE_OS_WINDOWS
  return ::FlushViewOfFile(mapping_, size_ + sizeof(MappingHeader)) != 0;
#else
  return ::msync(data_, size_, MS_SYNC) == 0;
#endif
}

void SharedMemory::reset() noexcept {
  if (data_ == nullptr) return;
#if HYLE_OS_WINDOWS
  ::UnmapViewOfFile(mapping_);
  if (handle_ != -1) ::CloseHandle(reinterpret_cast<HANDLE>(handle_));
#else
  ::munmap(data_, size_);
#endif
  mapping_ = nullptr;
  data_ = nullptr;
  size_ = 0;
  handle_ = -1;
}

SharedMemory::~SharedMemory() { reset(); }

SharedMemory::SharedMemory(SharedMemory&& other) noexcept
    : mapping_(std::exchange(other.mapping_, nullptr)),
      data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      handle_(std::exchange(other.handle_, -1)) {}

SharedMemory& SharedMemory::operator=(SharedMemory&& other) noexcept {
  if (this != &other) {
    reset();
    mapping_ = std::exchange(other.mapping_, nullptr);
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
    handle_ = std::exchange(other.handle_, -1);
  }
  return *this;
}

}  // namespace hyle::io

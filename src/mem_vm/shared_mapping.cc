#include "hyle/mem/mapping.h"

#include <limits>
#include <string_view>
#include <utility>

#include "hyle/core/environment.h"

#if HYLE_OS_WINDOWS
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <cerrno>
#  include <cstring>
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace hyle::mem {
namespace {

struct MappingHeader {
  std::uint64_t magic;
  std::uint64_t size;
};
constexpr std::uint64_t kMappingMagic = 0x48594c4553484d32ull;  // HYLESHM2

#if HYLE_OS_WINDOWS

Result<std::wstring_view> wide_name(const char* text, wchar_t* output,
                                    int capacity) noexcept {
  if (text == nullptr || *text == '\0') {
    return std::unexpected(
        Status::InvalidArgument("shared mapping name is empty"));
  }
  const int length = ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text,
                                           -1, output, capacity);
  if (length <= 0) {
    return std::unexpected(status_from_win32(
        ::GetLastError(), "shared mapping UTF-8 name conversion"));
  }
  return std::wstring_view(output, static_cast<std::size_t>(length - 1));
}
#else
Result<std::size_t> normalize_name(const char* name, char* output,
                                   std::size_t capacity) noexcept {
  if (name == nullptr || *name == '\0' || capacity < 3) {
    return std::unexpected(
        Status::InvalidArgument("shared mapping name is empty"));
  }
  const std::size_t length = std::strlen(name);
  const bool has_slash = name[0] == '/';
  if (length + (has_slash ? 1 : 2) > capacity) {
    return std::unexpected(
        Status::OutOfRange("shared mapping name is too long"));
  }
  std::size_t cursor = 0;
  if (!has_slash) output[cursor++] = '/';
  std::memcpy(output + cursor, name, length + 1);
  return cursor + length;
}
#endif

}  // namespace

Result<SharedMapping> SharedMapping::create(
    const char* name, std::size_t size) noexcept {
  if (size == 0) {
    return std::unexpected(
        Status::InvalidArgument("shared mapping size is zero"));
  }
  if (size > SIZE_MAX - sizeof(MappingHeader)) {
    return std::unexpected(
        Status::OutOfRange("shared mapping size overflows"));
  }
  const std::size_t native_size = size + sizeof(MappingHeader);
  SharedMapping output;
#if HYLE_OS_WINDOWS
  wchar_t wide[260];
  HYLE_ASSIGN_OR_RETURN(auto parsed_name, wide_name(name, wide, 260));
  (void)parsed_name;
  const std::uint64_t windows_size = static_cast<std::uint64_t>(native_size);
  HANDLE handle = ::CreateFileMappingW(
      INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
      static_cast<DWORD>(windows_size >> 32),
      static_cast<DWORD>(windows_size & 0xffffffffu), wide);
  if (handle == nullptr) {
    return std::unexpected(status_from_win32(::GetLastError(),
                                             "CreateFileMappingW(shared)"));
  }
  if (::GetLastError() == ERROR_ALREADY_EXISTS) {
    ::CloseHandle(handle);
    return std::unexpected(
        Status::AlreadyExists("shared mapping name already exists"));
  }
  void* mapping =
      ::MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0,
                      static_cast<SIZE_T>(native_size));
  if (mapping == nullptr) {
    const DWORD code = ::GetLastError();
    ::CloseHandle(handle);
    return std::unexpected(
        status_from_win32(code, "MapViewOfFile(shared create)"));
  }
  auto* header = static_cast<MappingHeader*>(mapping);
  header->magic = kMappingMagic;
  header->size = size;
  output.mapping_ = mapping;
  output.data_ = header + 1;
  output.size_ = size;
  output.mapping_size_ = native_size;
  output.handle_ = reinterpret_cast<std::intptr_t>(handle);
#else
  if (native_size >
      static_cast<std::size_t>((std::numeric_limits<off_t>::max)())) {
    return std::unexpected(
        Status::OutOfRange("shared mapping size exceeds off_t"));
  }
  char normalized[256];
  HYLE_ASSIGN_OR_RETURN(
      auto normalized_size,
      normalize_name(name, normalized, sizeof(normalized)));
  (void)normalized_size;
  const int descriptor =
      ::shm_open(normalized, O_RDWR | O_CREAT | O_EXCL, 0600);
  if (descriptor < 0) {
    return std::unexpected(status_from_errno(errno, "shm_open(create)"));
  }
  if (::ftruncate(descriptor, static_cast<off_t>(native_size)) != 0) {
    const int code = errno;
    ::close(descriptor);
    ::shm_unlink(normalized);
    return std::unexpected(status_from_errno(code, "ftruncate(shared)"));
  }
  void* mapping =
      ::mmap(nullptr, native_size, PROT_READ | PROT_WRITE, MAP_SHARED,
             descriptor, 0);
  const int map_code = mapping == MAP_FAILED ? errno : 0;
  ::close(descriptor);
  if (mapping == MAP_FAILED) {
    ::shm_unlink(normalized);
    return std::unexpected(status_from_errno(map_code, "mmap(shared create)"));
  }
  auto* header = static_cast<MappingHeader*>(mapping);
  header->magic = kMappingMagic;
  header->size = size;
  output.mapping_ = mapping;
  output.data_ = header + 1;
  output.size_ = size;
  output.mapping_size_ = native_size;
#endif
  return output;
}

Result<SharedMapping> SharedMapping::open(const char* name) noexcept {
  SharedMapping output;
#if HYLE_OS_WINDOWS
  wchar_t wide[260];
  HYLE_ASSIGN_OR_RETURN(auto parsed_name, wide_name(name, wide, 260));
  (void)parsed_name;
  HANDLE handle = ::OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, wide);
  if (handle == nullptr) {
    return std::unexpected(status_from_win32(::GetLastError(),
                                             "OpenFileMappingW(shared)"));
  }
  void* header_mapping =
      ::MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(MappingHeader));
  if (header_mapping == nullptr) {
    const DWORD code = ::GetLastError();
    ::CloseHandle(handle);
    return std::unexpected(
        status_from_win32(code, "MapViewOfFile(shared header)"));
  }
  const auto* header = static_cast<const MappingHeader*>(header_mapping);
  if (header->magic != kMappingMagic || header->size == 0 ||
      header->size > SIZE_MAX - sizeof(MappingHeader)) {
    ::UnmapViewOfFile(header_mapping);
    ::CloseHandle(handle);
    return std::unexpected(
        Status::DataLoss("shared mapping header is invalid"));
  }
  const std::size_t size = static_cast<std::size_t>(header->size);
  ::UnmapViewOfFile(header_mapping);
  const std::size_t native_size = size + sizeof(MappingHeader);
  void* mapping =
      ::MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, native_size);
  if (mapping == nullptr) {
    const DWORD code = ::GetLastError();
    ::CloseHandle(handle);
    return std::unexpected(
        status_from_win32(code, "MapViewOfFile(shared open)"));
  }
  header = static_cast<const MappingHeader*>(mapping);
  if (header->magic != kMappingMagic || header->size != size) {
    ::UnmapViewOfFile(mapping);
    ::CloseHandle(handle);
    return std::unexpected(
        Status::DataLoss("shared mapping header changed while opening"));
  }
  output.mapping_ = mapping;
  output.data_ = const_cast<MappingHeader*>(header) + 1;
  output.size_ = size;
  output.mapping_size_ = native_size;
  output.handle_ = reinterpret_cast<std::intptr_t>(handle);
#else
  char normalized[256];
  HYLE_ASSIGN_OR_RETURN(
      auto normalized_size,
      normalize_name(name, normalized, sizeof(normalized)));
  (void)normalized_size;
  const int descriptor = ::shm_open(normalized, O_RDWR, 0600);
  if (descriptor < 0) {
    return std::unexpected(status_from_errno(errno, "shm_open(open)"));
  }
  struct ::stat info {};
  if (::fstat(descriptor, &info) != 0) {
    const int code = errno;
    ::close(descriptor);
    return std::unexpected(status_from_errno(code, "fstat(shared)"));
  }
  if (info.st_size < static_cast<off_t>(sizeof(MappingHeader)) ||
      static_cast<std::uintmax_t>(info.st_size) >
          (std::numeric_limits<std::size_t>::max)()) {
    ::close(descriptor);
    return std::unexpected(
        Status::DataLoss("shared mapping has an invalid size"));
  }
  const std::size_t mapping_size = static_cast<std::size_t>(info.st_size);
  void* mapping =
      ::mmap(nullptr, mapping_size, PROT_READ | PROT_WRITE, MAP_SHARED,
             descriptor, 0);
  const int map_code = mapping == MAP_FAILED ? errno : 0;
  ::close(descriptor);
  if (mapping == MAP_FAILED) {
    return std::unexpected(status_from_errno(map_code, "mmap(shared open)"));
  }
  const auto* header = static_cast<const MappingHeader*>(mapping);
  if (header->magic != kMappingMagic || header->size == 0 ||
      header->size > mapping_size - sizeof(MappingHeader)) {
    ::munmap(mapping, mapping_size);
    return std::unexpected(
        Status::DataLoss("shared mapping header is invalid"));
  }
  const std::size_t size = static_cast<std::size_t>(header->size);
  output.mapping_ = mapping;
  output.data_ = const_cast<MappingHeader*>(header) + 1;
  output.size_ = size;
  output.mapping_size_ = mapping_size;
#endif
  return output;
}

Status SharedMapping::unlink(const char* name) noexcept {
#if HYLE_OS_WINDOWS
  if (name == nullptr || *name == '\0') {
    return Status::InvalidArgument("shared mapping name is empty");
  }
  return Status::Unimplemented(
      "Windows named mappings disappear after the final handle closes");
#else
  char normalized[256];
  HYLE_ASSIGN_OR_RETURN(
      std::size_t ignored,
      normalize_name(name, normalized, sizeof(normalized)));
  (void)ignored;
  if (::shm_unlink(normalized) == 0 || errno == ENOENT) return Status::Ok();
  return status_from_errno(errno, "shm_unlink");
#endif
}

SharedMapping::SharedMapping(SharedMapping&& other) noexcept
    : mapping_(std::exchange(other.mapping_, nullptr)),
      data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      mapping_size_(std::exchange(other.mapping_size_, 0)),
      handle_(std::exchange(other.handle_, -1)) {}

SharedMapping& SharedMapping::operator=(
    SharedMapping&& other) noexcept {
  if (this != &other) {
    reset().IgnoreError();
    mapping_ = std::exchange(other.mapping_, nullptr);
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
    mapping_size_ = std::exchange(other.mapping_size_, 0);
    handle_ = std::exchange(other.handle_, -1);
  }
  return *this;
}

SharedMapping::~SharedMapping() { reset().IgnoreError(); }

Status SharedMapping::flush() noexcept {
  if (!valid()) {
    return Status::FailedPrecondition("shared mapping is invalid");
  }
#if HYLE_OS_WINDOWS
  if (!::FlushViewOfFile(mapping_, mapping_size_)) {
    return status_from_win32(::GetLastError(), "FlushViewOfFile(shared)");
  }
#else
  if (::msync(mapping_, mapping_size_, MS_SYNC) != 0) {
    return status_from_errno(errno, "msync(shared)");
  }
#endif
  return Status::Ok();
}

Status SharedMapping::reset() noexcept {
  if (!valid()) return Status::Ok();
  Status result = Status::Ok();
#if HYLE_OS_WINDOWS
  if (!::UnmapViewOfFile(mapping_)) {
    result = status_from_win32(::GetLastError(), "UnmapViewOfFile(shared)");
  }
  if (handle_ != -1 && !::CloseHandle(reinterpret_cast<HANDLE>(handle_))) {
    result.Update(
        status_from_win32(::GetLastError(), "CloseHandle(shared mapping)"));
  }
#else
  if (::munmap(mapping_, mapping_size_) != 0) {
    result = status_from_errno(errno, "munmap(shared)");
  }
#endif
  mapping_ = nullptr;
  data_ = nullptr;
  size_ = 0;
  mapping_size_ = 0;
  handle_ = -1;
  return result;
}

}  // namespace hyle::mem

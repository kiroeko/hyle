#include "hyle/mem/mapping.h"

#include <limits>
#include <utility>

#include "hyle/core/environment.h"
#include "hyle/mem/allocator.h"

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

bool valid_shared_access(MappingAccess access) noexcept {
  return access == MappingAccess::kReadOnly ||
         access == MappingAccess::kReadWrite;
}

#if HYLE_OS_WINDOWS

class WideName {
 public:
  WideName() noexcept = default;
  ~WideName() { reset(); }
  WideName(const WideName&) = delete;
  WideName& operator=(const WideName&) = delete;
  WideName(WideName&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        count_(std::exchange(other.count_, 0)) {}
  WideName& operator=(WideName&& other) noexcept {
    if (this != &other) {
      reset();
      data_ = std::exchange(other.data_, nullptr);
      count_ = std::exchange(other.count_, 0);
    }
    return *this;
  }

  static Result<WideName> from_utf8(const char* text) noexcept {
    if (text == nullptr || *text == '\0') {
      return std::unexpected(
          Status::InvalidArgument("shared mapping name is empty"));
    }
    const int required = ::MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, text, -1, nullptr, 0);
    if (required <= 0) {
      return std::unexpected(status_from_win32(
          ::GetLastError(), "shared mapping UTF-8 name conversion"));
    }
    const std::size_t count = static_cast<std::size_t>(required);
    if (count > (std::numeric_limits<std::size_t>::max)() /
                    sizeof(wchar_t)) {
      return std::unexpected(
          Status::OutOfRange("shared mapping name size overflows"));
    }
    auto* data = static_cast<wchar_t*>(system_allocator().allocate(
        count * sizeof(wchar_t), alignof(wchar_t)));
    if (data == nullptr) {
      return std::unexpected(
          Status::ResourceExhausted("shared mapping name allocation failed"));
    }
    if (::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text, -1, data,
                              required) != required) {
      const DWORD code = ::GetLastError();
      system_allocator().deallocate(data, count * sizeof(wchar_t),
                                    alignof(wchar_t));
      return std::unexpected(status_from_win32(
          code, "shared mapping UTF-8 name conversion"));
    }
    WideName result;
    result.data_ = data;
    result.count_ = count;
    return result;
  }

  const wchar_t* data() const noexcept { return data_; }

 private:
  void reset() noexcept {
    if (data_ != nullptr) {
      system_allocator().deallocate(data_, count_ * sizeof(wchar_t),
                                    alignof(wchar_t));
      data_ = nullptr;
      count_ = 0;
    }
  }

  wchar_t* data_ = nullptr;
  std::size_t count_ = 0;
};

#else

class PosixName {
 public:
  PosixName() noexcept = default;
  ~PosixName() { reset(); }
  PosixName(const PosixName&) = delete;
  PosixName& operator=(const PosixName&) = delete;
  PosixName(PosixName&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        count_(std::exchange(other.count_, 0)) {}
  PosixName& operator=(PosixName&& other) noexcept {
    if (this != &other) {
      reset();
      data_ = std::exchange(other.data_, nullptr);
      count_ = std::exchange(other.count_, 0);
    }
    return *this;
  }

  static Result<PosixName> normalize(const char* name) noexcept {
    if (name == nullptr || *name == '\0') {
      return std::unexpected(
          Status::InvalidArgument("shared mapping name is empty"));
    }
    const char* payload = name[0] == '/' ? name + 1 : name;
    if (*payload == '\0') {
      return std::unexpected(Status::InvalidArgument(
          "shared mapping name needs a non-empty basename"));
    }
    const std::size_t payload_size = std::strlen(payload);
    if (std::memchr(payload, '/', payload_size) != nullptr) {
      return std::unexpected(Status::InvalidArgument(
          "POSIX shared mapping name cannot contain an interior slash"));
    }
    if (payload_size > (std::numeric_limits<std::size_t>::max)() - 2) {
      return std::unexpected(
          Status::OutOfRange("shared mapping name size overflows"));
    }
    const std::size_t count = payload_size + 2;
    auto* data = static_cast<char*>(
        system_allocator().allocate(count, alignof(char)));
    if (data == nullptr) {
      return std::unexpected(
          Status::ResourceExhausted("shared mapping name allocation failed"));
    }
    data[0] = '/';
    std::memcpy(data + 1, payload, payload_size + 1);
    PosixName result;
    result.data_ = data;
    result.count_ = count;
    return result;
  }

  const char* data() const noexcept { return data_; }

 private:
  void reset() noexcept {
    if (data_ != nullptr) {
      system_allocator().deallocate(data_, count_, alignof(char));
      data_ = nullptr;
      count_ = 0;
    }
  }

  char* data_ = nullptr;
  std::size_t count_ = 0;
};

#endif

Status validate_mapping_size(std::size_t size) noexcept {
  if (size == 0) {
    return Status::InvalidArgument("shared mapping size is zero");
  }
#if !HYLE_OS_WINDOWS
  if (size > static_cast<std::size_t>(
                 (std::numeric_limits<off_t>::max)())) {
    return Status::OutOfRange("shared mapping size exceeds off_t");
  }
#endif
  return Status::Ok();
}

}  // namespace

Result<SharedMapping> SharedMapping::create(const char* name,
                                             std::size_t size) noexcept {
  HYLE_RETURN_IF_ERROR(validate_mapping_size(size));
  SharedMapping output;
#if HYLE_OS_WINDOWS
  HYLE_ASSIGN_OR_RETURN(WideName parsed_name, WideName::from_utf8(name));
  const std::uint64_t windows_size = static_cast<std::uint64_t>(size);
  HANDLE handle = ::CreateFileMappingW(
      INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
      static_cast<DWORD>(windows_size >> 32),
      static_cast<DWORD>(windows_size & 0xffffffffu), parsed_name.data());
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
      ::MapViewOfFile(handle, FILE_MAP_WRITE, 0, 0, static_cast<SIZE_T>(size));
  if (mapping == nullptr) {
    const DWORD code = ::GetLastError();
    ::CloseHandle(handle);
    return std::unexpected(
        status_from_win32(code, "MapViewOfFile(shared create)"));
  }
  output.mapping_ = mapping;
  output.size_ = size;
  output.handle_ = reinterpret_cast<std::intptr_t>(handle);
#else
  HYLE_ASSIGN_OR_RETURN(PosixName parsed_name, PosixName::normalize(name));
  const int descriptor =
      ::shm_open(parsed_name.data(), O_RDWR | O_CREAT | O_EXCL, 0600);
  if (descriptor < 0) {
    return std::unexpected(status_from_errno(errno, "shm_open(create)"));
  }
  if (::ftruncate(descriptor, static_cast<off_t>(size)) != 0) {
    const int code = errno;
    ::close(descriptor);
    ::shm_unlink(parsed_name.data());
    return std::unexpected(status_from_errno(code, "ftruncate(shared)"));
  }
  void* mapping =
      ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, descriptor, 0);
  const int map_code = mapping == MAP_FAILED ? errno : 0;
  ::close(descriptor);
  if (mapping == MAP_FAILED) {
    ::shm_unlink(parsed_name.data());
    return std::unexpected(status_from_errno(map_code, "mmap(shared create)"));
  }
  output.mapping_ = mapping;
  output.size_ = size;
#endif
  output.access_ = MappingAccess::kReadWrite;
  return output;
}

Result<SharedMapping> SharedMapping::open(const char* name, std::size_t size,
                                           MappingAccess access) noexcept {
  HYLE_RETURN_IF_ERROR(validate_mapping_size(size));
  if (!valid_shared_access(access)) {
    return std::unexpected(Status::InvalidArgument(
        "shared mapping open supports read-only or read-write access"));
  }
  SharedMapping output;
#if HYLE_OS_WINDOWS
  HYLE_ASSIGN_OR_RETURN(WideName parsed_name, WideName::from_utf8(name));
  const DWORD desired_access = access == MappingAccess::kReadOnly
                                   ? FILE_MAP_READ
                                   : FILE_MAP_WRITE;
  HANDLE handle =
      ::OpenFileMappingW(desired_access, FALSE, parsed_name.data());
  if (handle == nullptr) {
    return std::unexpected(status_from_win32(::GetLastError(),
                                             "OpenFileMappingW(shared)"));
  }
  void* mapping = ::MapViewOfFile(handle, desired_access, 0, 0,
                                  static_cast<SIZE_T>(size));
  if (mapping == nullptr) {
    const DWORD code = ::GetLastError();
    ::CloseHandle(handle);
    return std::unexpected(
        status_from_win32(code, "MapViewOfFile(shared open)"));
  }
  output.mapping_ = mapping;
  output.size_ = size;
  output.handle_ = reinterpret_cast<std::intptr_t>(handle);
#else
  HYLE_ASSIGN_OR_RETURN(PosixName parsed_name, PosixName::normalize(name));
  const int descriptor =
      ::shm_open(parsed_name.data(),
                 access == MappingAccess::kReadOnly ? O_RDONLY : O_RDWR, 0600);
  if (descriptor < 0) {
    return std::unexpected(status_from_errno(errno, "shm_open(open)"));
  }
  struct ::stat info {};
  if (::fstat(descriptor, &info) != 0) {
    const int code = errno;
    ::close(descriptor);
    return std::unexpected(status_from_errno(code, "fstat(shared)"));
  }
  if (info.st_size < 0) {
    ::close(descriptor);
    return std::unexpected(
        Status::DataLoss("shared mapping has a negative native size"));
  }
  const auto native_size = static_cast<std::uintmax_t>(info.st_size);
  if (native_size < size) {
    ::close(descriptor);
    return std::unexpected(Status::Unavailable(
        "shared mapping has not published the requested size"));
  }
  const int protection =
      access == MappingAccess::kReadOnly ? PROT_READ : PROT_READ | PROT_WRITE;
  void* mapping =
      ::mmap(nullptr, size, protection, MAP_SHARED, descriptor, 0);
  const int map_code = mapping == MAP_FAILED ? errno : 0;
  ::close(descriptor);
  if (mapping == MAP_FAILED) {
    return std::unexpected(status_from_errno(map_code, "mmap(shared open)"));
  }
  output.mapping_ = mapping;
  output.size_ = size;
#endif
  output.access_ = access;
  return output;
}

Status SharedMapping::unlink(const char* name) noexcept {
#if HYLE_OS_WINDOWS
  HYLE_ASSIGN_OR_RETURN(WideName ignored, WideName::from_utf8(name));
  (void)ignored;
  return Status::Unimplemented(
      "Windows named mappings disappear after the final handle closes");
#else
  HYLE_ASSIGN_OR_RETURN(PosixName parsed_name, PosixName::normalize(name));
  if (::shm_unlink(parsed_name.data()) == 0 || errno == ENOENT) {
    return Status::Ok();
  }
  return status_from_errno(errno, "shm_unlink");
#endif
}

Result<void*> SharedMapping::writable_data() noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("shared mapping is invalid"));
  }
  if (access_ != MappingAccess::kReadWrite) {
    return std::unexpected(
        Status::PermissionDenied("shared mapping is read-only"));
  }
  return mapping_;
}

Result<std::span<std::byte>> SharedMapping::writable_bytes() noexcept {
  if (!valid()) {
    return std::unexpected(
        Status::FailedPrecondition("shared mapping is invalid"));
  }
  if (access_ != MappingAccess::kReadWrite) {
    return std::unexpected(
        Status::PermissionDenied("shared mapping is read-only"));
  }
  return std::span<std::byte>(static_cast<std::byte*>(mapping_), size_);
}

SharedMapping::SharedMapping(SharedMapping&& other) noexcept
    : mapping_(std::exchange(other.mapping_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      handle_(std::exchange(other.handle_, -1)),
      access_(std::exchange(other.access_, MappingAccess::kReadOnly)) {}

SharedMapping& SharedMapping::operator=(SharedMapping&& other) noexcept {
  if (this != &other) {
    reset().IgnoreError();
    mapping_ = std::exchange(other.mapping_, nullptr);
    size_ = std::exchange(other.size_, 0);
    handle_ = std::exchange(other.handle_, -1);
    access_ = std::exchange(other.access_, MappingAccess::kReadOnly);
  }
  return *this;
}

SharedMapping::~SharedMapping() { reset().IgnoreError(); }

Status SharedMapping::flush() noexcept {
  if (!valid()) {
    return Status::FailedPrecondition("shared mapping is invalid");
  }
  if (access_ != MappingAccess::kReadWrite) {
    return Status::FailedPrecondition(
        "shared mapping flush requires read-write access");
  }
#if HYLE_OS_WINDOWS
  if (!::FlushViewOfFile(mapping_, size_)) {
    return status_from_win32(::GetLastError(), "FlushViewOfFile(shared)");
  }
#else
  if (::msync(mapping_, size_, MS_SYNC) != 0) {
    return status_from_errno(errno, "msync(shared)");
  }
#endif
  return Status::Ok();
}

Status SharedMapping::reset() noexcept {
  if (!valid()) return Status::Ok();
  Status result;
#if HYLE_OS_WINDOWS
  if (!::UnmapViewOfFile(mapping_)) {
    result = status_from_win32(::GetLastError(), "UnmapViewOfFile(shared)");
  }
  if (handle_ != -1 &&
      !::CloseHandle(reinterpret_cast<HANDLE>(handle_)) && result.ok()) {
    result =
        status_from_win32(::GetLastError(), "CloseHandle(shared mapping)");
  }
#else
  if (::munmap(mapping_, size_) != 0) {
    result = status_from_errno(errno, "munmap(shared)");
  }
#endif
  mapping_ = nullptr;
  size_ = 0;
  handle_ = -1;
  access_ = MappingAccess::kReadOnly;
  return result;
}

}  // namespace hyle::mem

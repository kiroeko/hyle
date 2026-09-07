#include "hyle/io/file.h"

#include <algorithm>
#include <cerrno>
#include <limits>
#include <utility>

#include "hyle/core/environment.h"
#include "hyle/sched/blocking.h"
#include "hyle/sched/cancel.h"

#if HYLE_OS_WINDOWS
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace hyle::io {
namespace {

constexpr std::uint64_t kLargestFileOffset =
    static_cast<std::uint64_t>((std::numeric_limits<std::int64_t>::max)());

std::int64_t native_file_error() noexcept {
#if HYLE_OS_WINDOWS
  return -static_cast<std::int64_t>(::GetLastError());
#else
  return -static_cast<std::int64_t>(errno);
#endif
}

template <class Fn> std::int64_t offload(IoContext &io, Fn &&fn) noexcept {
  auto handle = sched::spawn_blocking(
      io.blocking_pool(), io.operation_allocator(), std::forward<Fn>(fn));
  auto result = handle.join();
  return result.has_value() ? *result : kSubmitFailed;
}

template <class Submit>
bool submit_and_suspend(IoOperation &operation, bool &submitted,
                        Submit &&submit) noexcept {
  operation.waker = sched::current_waker();
  submitted = std::forward<Submit>(submit)(to_token(&operation));
  if (!submitted) {
    operation.result = kSubmitFailed;
    return false;
  }
  sched::suspend();
  return false;
}

Status validate_open(std::string_view path,
                     const FileOpenOptions &options) noexcept {
  if (path.empty() || path.find('\0') != std::string_view::npos)
    return Status::InvalidArgument(
        "open_file requires one non-empty UTF-8 path");
  const bool writable = options.access == FileAccess::kWrite ||
                        options.access == FileAccess::kReadWrite;
  if ((options.create || options.truncate) && !writable)
    return Status::InvalidArgument("create/truncate requires write access");
  if ((options.create_permissions & ~0777u) != 0)
    return Status::InvalidArgument(
        "create_permissions contains non-portable bits");
#if HYLE_OS_MACOS
  if (options.cache == FileCacheMode::kDirect)
    return Status::Unimplemented(
        "macOS has no portable direct-IO contract for this API");
#endif
  return Status{};
}

#if HYLE_OS_WINDOWS

std::int64_t query_direct_alignment(HANDLE handle,
                                    DirectIoAlignment *alignment) noexcept {
  FILE_STORAGE_INFO storage{};
  if (!::GetFileInformationByHandleEx(handle, FileStorageInfo, &storage,
                                      sizeof(storage)))
    return native_file_error();
  FILE_ALIGNMENT_INFO native_alignment{};
  if (!::GetFileInformationByHandleEx(handle, FileAlignmentInfo,
                                      &native_alignment,
                                      sizeof(native_alignment)))
    return native_file_error();

  const std::size_t logical =
      std::max<std::size_t>(1, storage.LogicalBytesPerSector);
  const std::size_t physical = std::max<std::size_t>(
      logical, storage.PhysicalBytesPerSectorForPerformance);
  const std::size_t address = std::max<std::size_t>(
      physical, native_alignment.AlignmentRequirement + 1u);
  *alignment = DirectIoAlignment{address, logical, logical};
  return 0;
}

std::int64_t file_open_sync(mem::AllocatorRef allocator, const char *path,
                            const FileOpenOptions &options,
                            std::intptr_t *output,
                            DirectIoAlignment *alignment) noexcept {
  const int required = ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                             path, -1, nullptr, 0);
  if (required <= 0)
    return native_file_error();
  container::SmallVector<wchar_t, 512, mem::AllocatorRef> wide(allocator);
  if (!wide.try_resize(static_cast<std::size_t>(required)))
    return kSubmitFailed;
  if (::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1,
                            wide.data(), required) != required)
    return native_file_error();

  DWORD access = 0;
  if (options.access == FileAccess::kRead ||
      options.access == FileAccess::kReadWrite)
    access |= GENERIC_READ;
  if (options.access == FileAccess::kWrite ||
      options.access == FileAccess::kReadWrite)
    access |= GENERIC_WRITE;
  const DWORD disposition =
      options.create ? (options.truncate ? CREATE_ALWAYS : OPEN_ALWAYS)
                     : (options.truncate ? TRUNCATE_EXISTING : OPEN_EXISTING);
  DWORD attributes = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED;
  if (options.cache == FileCacheMode::kDirect)
    attributes |= FILE_FLAG_NO_BUFFERING;

  HANDLE handle =
      ::CreateFileW(wide.data(), access,
                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                    nullptr, disposition, attributes, nullptr);
  if (handle == INVALID_HANDLE_VALUE)
    return native_file_error();
  if (options.cache == FileCacheMode::kDirect) {
    const std::int64_t queried = query_direct_alignment(handle, alignment);
    if (queried < 0) {
      ::CloseHandle(handle);
      return queried;
    }
  }
  *output = reinterpret_cast<std::intptr_t>(handle);
  return 0;
}

std::int64_t file_read_sync(std::intptr_t handle, void *buffer,
                            std::uint32_t length,
                            std::uint64_t offset) noexcept {
  HANDLE native = reinterpret_cast<HANDLE>(handle);
  LARGE_INTEGER position{};
  position.QuadPart = static_cast<LONGLONG>(offset);
  if (!::SetFilePointerEx(native, position, nullptr, FILE_BEGIN))
    return native_file_error();
  DWORD transferred = 0;
  if (!::ReadFile(native, buffer, length, &transferred, nullptr))
    return native_file_error();
  return static_cast<std::int64_t>(transferred);
}

std::int64_t file_write_sync(std::intptr_t handle, const void *buffer,
                             std::uint32_t length,
                             std::uint64_t offset) noexcept {
  HANDLE native = reinterpret_cast<HANDLE>(handle);
  LARGE_INTEGER position{};
  position.QuadPart = static_cast<LONGLONG>(offset);
  if (!::SetFilePointerEx(native, position, nullptr, FILE_BEGIN))
    return native_file_error();
  DWORD transferred = 0;
  if (!::WriteFile(native, buffer, length, &transferred, nullptr))
    return native_file_error();
  return static_cast<std::int64_t>(transferred);
}

std::int64_t file_stat_sync(std::intptr_t handle,
                            Proactor::FileStat *output) noexcept {
  BY_HANDLE_FILE_INFORMATION info{};
  if (!::GetFileInformationByHandle(reinterpret_cast<HANDLE>(handle), &info))
    return native_file_error();
  *output = Proactor::FileStat{};
  output->size = (static_cast<std::uint64_t>(info.nFileSizeHigh) << 32) |
                 info.nFileSizeLow;
  const std::uint64_t ticks =
      (static_cast<std::uint64_t>(info.ftLastWriteTime.dwHighDateTime) << 32) |
      info.ftLastWriteTime.dwLowDateTime;
  output->modified_time =
      time::unix_timestamp_from_windows_filetime_ticks(ticks);
  output->is_directory =
      (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
  return 0;
}

std::int64_t file_sync_sync(std::intptr_t handle, bool) noexcept {
  return ::FlushFileBuffers(reinterpret_cast<HANDLE>(handle))
             ? 0
             : native_file_error();
}

std::int64_t file_preallocate_sync(std::intptr_t handle, std::uint64_t offset,
                                   std::uint64_t length) noexcept {
  if (length > kLargestFileOffset || offset > kLargestFileOffset - length)
    return -static_cast<std::int64_t>(ERROR_ARITHMETIC_OVERFLOW);
  HANDLE native = reinterpret_cast<HANDLE>(handle);
  LARGE_INTEGER current_size{};
  if (!::GetFileSizeEx(native, &current_size))
    return native_file_error();
  const auto requested_end = static_cast<LONGLONG>(offset + length);
  if (current_size.QuadPart >= requested_end)
    return 0;
  LARGE_INTEGER old_position{};
  LARGE_INTEGER zero{};
  if (!::SetFilePointerEx(native, zero, &old_position, FILE_CURRENT))
    return native_file_error();
  LARGE_INTEGER end{};
  end.QuadPart = requested_end;
  if (!::SetFilePointerEx(native, end, nullptr, FILE_BEGIN) ||
      !::SetEndOfFile(native)) {
    const std::int64_t error = native_file_error();
    (void)::SetFilePointerEx(native, old_position, nullptr, FILE_BEGIN);
    return error;
  }
  (void)::SetFilePointerEx(native, old_position, nullptr, FILE_BEGIN);
  return 0;
}

std::int64_t file_close_sync(std::intptr_t handle) noexcept {
  return ::CloseHandle(reinterpret_cast<HANDLE>(handle)) ? 0
                                                         : native_file_error();
}

#else

std::int64_t query_direct_alignment(int fd,
                                    DirectIoAlignment *alignment) noexcept {
  struct stat info{};
  if (::fstat(fd, &info) != 0)
    return native_file_error();
  std::size_t transfer =
      static_cast<std::size_t>(info.st_blksize > 0 ? info.st_blksize : 4096);
  long page_size = ::sysconf(_SC_PAGESIZE);
  std::size_t memory = std::max<std::size_t>(
      transfer, page_size > 0 ? static_cast<std::size_t>(page_size) : 4096);
#if defined(STATX_DIOALIGN) && defined(AT_EMPTY_PATH)
  struct statx extended{};
  if (::statx(fd, "", AT_EMPTY_PATH | AT_STATX_SYNC_AS_STAT, STATX_DIOALIGN,
              &extended) == 0 &&
      (extended.stx_mask & STATX_DIOALIGN) != 0) {
    if (extended.stx_dio_mem_align != 0)
      memory = extended.stx_dio_mem_align;
    if (extended.stx_dio_offset_align != 0)
      transfer = extended.stx_dio_offset_align;
  }
#endif
  *alignment = DirectIoAlignment{memory, transfer, transfer};
  return 0;
}

int file_open_flags(const FileOpenOptions &options) noexcept {
  int flags = O_CLOEXEC;
  if (options.access == FileAccess::kReadWrite)
    flags |= O_RDWR;
  else if (options.access == FileAccess::kWrite)
    flags |= O_WRONLY;
  else
    flags |= O_RDONLY;
  if (options.create)
    flags |= O_CREAT;
  if (options.truncate)
    flags |= O_TRUNC;
#if defined(O_DIRECT)
  if (options.cache == FileCacheMode::kDirect)
    flags |= O_DIRECT;
#endif
  return flags;
}

std::int64_t file_open_sync(mem::AllocatorRef, const char *path,
                            const FileOpenOptions &options,
                            std::intptr_t *output,
                            DirectIoAlignment *alignment) noexcept {
#if !defined(O_DIRECT)
  if (options.cache == FileCacheMode::kDirect)
    return -ENOTSUP;
#endif
  const int flags = file_open_flags(options);
  int fd;
  do {
    fd = options.create
             ? ::open(path, flags,
                      static_cast<mode_t>(options.create_permissions))
             : ::open(path, flags);
  } while (fd < 0 && errno == EINTR);
  if (fd < 0)
    return native_file_error();
  if (options.cache == FileCacheMode::kDirect) {
    const std::int64_t queried = query_direct_alignment(fd, alignment);
    if (queried < 0) {
      ::close(fd);
      return queried;
    }
  }
  *output = static_cast<std::intptr_t>(fd);
  return 0;
}

std::int64_t file_read_sync(std::intptr_t handle, void *buffer,
                            std::uint32_t length,
                            std::uint64_t offset) noexcept {
  ssize_t result;
  do {
    result = ::pread(static_cast<int>(handle), buffer, length,
                     static_cast<off_t>(offset));
  } while (result < 0 && errno == EINTR);
  return result < 0 ? native_file_error() : static_cast<std::int64_t>(result);
}

std::int64_t file_write_sync(std::intptr_t handle, const void *buffer,
                             std::uint32_t length,
                             std::uint64_t offset) noexcept {
  ssize_t result;
  do {
    result = ::pwrite(static_cast<int>(handle), buffer, length,
                      static_cast<off_t>(offset));
  } while (result < 0 && errno == EINTR);
  return result < 0 ? native_file_error() : static_cast<std::int64_t>(result);
}

std::int64_t file_stat_sync(std::intptr_t handle,
                            Proactor::FileStat *output) noexcept {
  struct stat info{};
  if (::fstat(static_cast<int>(handle), &info) != 0)
    return native_file_error();
  *output = Proactor::FileStat{};
  output->size = static_cast<std::uint64_t>(info.st_size);
#if HYLE_OS_MACOS
  output->modified_time = time::unix_timestamp(
      static_cast<std::int64_t>(info.st_mtimespec.tv_sec),
      static_cast<std::uint32_t>(info.st_mtimespec.tv_nsec));
#else
  output->modified_time =
      time::unix_timestamp(static_cast<std::int64_t>(info.st_mtim.tv_sec),
                           static_cast<std::uint32_t>(info.st_mtim.tv_nsec));
#endif
  output->mode = static_cast<std::uint32_t>(info.st_mode);
  output->is_directory = S_ISDIR(info.st_mode);
  return 0;
}

std::int64_t file_sync_sync(std::intptr_t handle, bool data_only) noexcept {
#if HYLE_OS_MACOS
  (void)data_only;
  const int result = ::fsync(static_cast<int>(handle));
#else
  const int result = data_only ? ::fdatasync(static_cast<int>(handle))
                               : ::fsync(static_cast<int>(handle));
#endif
  return result == 0 ? 0 : native_file_error();
}

std::int64_t file_preallocate_sync(std::intptr_t handle, std::uint64_t offset,
                                   std::uint64_t length) noexcept {
  if (length > kLargestFileOffset || offset > kLargestFileOffset - length)
    return -EOVERFLOW;
#if HYLE_OS_MACOS
  const off_t end = static_cast<off_t>(offset + length);
  struct stat info{};
  if (::fstat(static_cast<int>(handle), &info) != 0)
    return native_file_error();
  if (info.st_size >= end)
    return 0;
  fstore_t store{};
  store.fst_flags = F_ALLOCATECONTIG;
  store.fst_posmode = F_PEOFPOSMODE;
  store.fst_length = end;
  int result = ::fcntl(static_cast<int>(handle), F_PREALLOCATE, &store);
  if (result < 0) {
    store.fst_flags = F_ALLOCATEALL;
    result = ::fcntl(static_cast<int>(handle), F_PREALLOCATE, &store);
  }
  if (result < 0)
    return native_file_error();
  return ::ftruncate(static_cast<int>(handle), end) == 0 ? 0
                                                         : native_file_error();
#else
  const int result =
      ::posix_fallocate(static_cast<int>(handle), static_cast<off_t>(offset),
                        static_cast<off_t>(length));
  return result == 0 ? 0 : -static_cast<std::int64_t>(result);
#endif
}

std::int64_t file_close_sync(std::intptr_t handle) noexcept {
  const int result = ::close(static_cast<int>(handle));
  return result == 0 ? 0 : native_file_error();
}

#endif

std::uint32_t transfer_length(const File &file,
                              std::size_t requested) noexcept {
  std::size_t maximum =
      static_cast<std::size_t>((std::numeric_limits<std::uint32_t>::max)());
  if (file.cache_mode() == FileCacheMode::kDirect) {
    const std::size_t alignment = file.direct_alignment().length_alignment;
    maximum -= maximum % alignment;
  }
  return static_cast<std::uint32_t>(std::min(requested, maximum));
}

Status validate_transfer(const File &file, const void *buffer,
                         std::size_t requested, std::uint64_t offset,
                         std::uint32_t length) noexcept {
  if (!file.valid())
    return Status::FailedPrecondition("file is not open");
  if (offset > kLargestFileOffset ||
      static_cast<std::uint64_t>(length) > kLargestFileOffset - offset)
    return Status::OutOfRange("file transfer offset overflows");
  if (requested != 0 && length == 0)
    return Status::OutOfRange("file transfer exceeds native chunk limits");
  if (file.cache_mode() == FileCacheMode::kDirect &&
      !file.direct_alignment().accepts(buffer, offset, length))
    return Status::InvalidArgument(
        "direct file transfer violates address/offset/length alignment");
  return Status{};
}

platform::FileInfo
file_info_from_native(const Proactor::FileStat &native) noexcept {
  platform::FileInfo info{};
  info.size = native.size;
  info.modified_time = native.modified_time;
#if HYLE_OS_WINDOWS
  info.type = native.is_directory ? platform::FileType::kDirectory
                                  : platform::FileType::kRegular;
#else
  const mode_t mode = static_cast<mode_t>(native.mode);
  if (S_ISREG(mode))
    info.type = platform::FileType::kRegular;
  else if (S_ISDIR(mode))
    info.type = platform::FileType::kDirectory;
  else if (S_ISLNK(mode))
    info.type = platform::FileType::kSymlink;
  else if (S_ISBLK(mode))
    info.type = platform::FileType::kBlockDevice;
  else if (S_ISCHR(mode))
    info.type = platform::FileType::kCharacterDevice;
  else if (S_ISFIFO(mode))
    info.type = platform::FileType::kFifo;
  else if (S_ISSOCK(mode))
    info.type = platform::FileType::kSocket;
#endif
  return info;
}

} // namespace

File::File(IoContext &io, platform::FileHandle &&handle, unsigned owner_worker,
           bool native_async, FileCacheMode cache,
           DirectIoAlignment alignment) noexcept
    : io_(&io), handle_(std::move(handle)), owner_(owner_worker),
      native_async_(native_async), cache_(cache), alignment_(alignment) {}

File::~File() = default;

File::File(File &&other) noexcept
    : io_(std::exchange(other.io_, nullptr)), handle_(std::move(other.handle_)),
      owner_(other.owner_),
      native_async_(std::exchange(other.native_async_, false)),
      cache_(std::exchange(other.cache_, FileCacheMode::kBuffered)),
      alignment_(std::exchange(other.alignment_, DirectIoAlignment{})) {}

File &File::operator=(File &&other) noexcept {
  if (this != &other) {
    io_ = std::exchange(other.io_, nullptr);
    handle_ = std::move(other.handle_);
    owner_ = other.owner_;
    native_async_ = std::exchange(other.native_async_, false);
    cache_ = std::exchange(other.cache_, FileCacheMode::kBuffered);
    alignment_ = std::exchange(other.alignment_, DirectIoAlignment{});
  }
  return *this;
}

Result<File> File::adopt(IoContext &io, unsigned owner_worker,
                         platform::FileHandle &&handle) noexcept {
  if (!io.valid())
    return make_unexpected(
        Status::FailedPrecondition("adopt requires a valid IoContext"));
  if (owner_worker >= io.num_workers())
    return make_unexpected(
        Status::OutOfRange("file owner worker is outside IoContext"));
  if (!handle.valid())
    return make_unexpected(
        Status::InvalidArgument("cannot adopt an empty FileHandle"));
#if HYLE_OS_LINUX
  const bool native_async = io.proactor(owner_worker).has_async_file_data();
#else
  const bool native_async = false;
#endif
  return File(io, std::move(handle), owner_worker, native_async,
              FileCacheMode::kBuffered, DirectIoAlignment{});
}

platform::FileHandle File::release() noexcept {
  io_ = nullptr;
  native_async_ = false;
  cache_ = FileCacheMode::kBuffered;
  alignment_ = {};
  return platform::FileHandle(handle_.release());
}

bool File::ReadSomeAtAwaitable::await_ready() noexcept {
  if (self == nullptr || !self->valid()) {
    immediate = TransferResult{
        0, Status::FailedPrecondition("file is not open"), false};
    immediate_ready = true;
    return true;
  }
  HYLE_FATAL_ASSERT(sched::current_worker_index() ==
                    static_cast<int>(self->owner_));
  if (buffer.empty()) {
    immediate_ready = true;
    return true;
  }
  const std::uint32_t length = transfer_length(*self, buffer.size());
  Status valid =
      validate_transfer(*self, buffer.data(), buffer.size(), offset, length);
  if (!valid.ok()) {
    immediate = TransferResult{0, std::move(valid), false};
    immediate_ready = true;
    return true;
  }
  Proactor *proactor = &self->io_->proactor_for_current();
  state =
      detail::BudgetState{proactor, self->io_->operation_allocator(), options};
  if (state.await_ready())
    return true;
  if (!self->native_async_ &&
      (options.deadline.has_value() || options.cancel != nullptr)) {
    immediate = TransferResult{
        0,
        Status::Unimplemented(
            "bounded file IO requires a natively cancellable backend"),
        false};
    immediate_ready = true;
    return true;
  }
  return false;
}

bool File::ReadSomeAtAwaitable::await_suspend(
    std::coroutine_handle<>) noexcept {
  const std::uint32_t length = transfer_length(*self, buffer.size());
  if (!self->native_async_) {
    const std::int64_t result = offload(
        *self->io_, [handle = self->native_handle(), data = buffer.data(),
                     length, offset = offset]() noexcept {
          return file_read_sync(handle, data, length, offset);
        });
    immediate = detail::transfer_from_native(result, detail::kFileNativeDomain,
                                             "file read", true);
    immediate_ready = true;
    return false;
  }
  return state.await_suspend([&](std::uint64_t token) noexcept {
    return state.proactor->submit_read(self->native_handle(), buffer.data(),
                                       length, offset, token);
  });
}

TransferResult File::ReadSomeAtAwaitable::await_resume() noexcept {
  if (immediate_ready)
    return std::move(immediate);
  return detail::transfer_from_native(
      state.await_resume(), detail::kFileNativeDomain, "file read", true);
}

bool File::WriteSomeAtAwaitable::await_ready() noexcept {
  if (self == nullptr || !self->valid()) {
    immediate = TransferResult{
        0, Status::FailedPrecondition("file is not open"), false};
    immediate_ready = true;
    return true;
  }
  HYLE_FATAL_ASSERT(sched::current_worker_index() ==
                    static_cast<int>(self->owner_));
  if (buffer.empty()) {
    immediate_ready = true;
    return true;
  }
  const std::uint32_t length = transfer_length(*self, buffer.size());
  Status valid =
      validate_transfer(*self, buffer.data(), buffer.size(), offset, length);
  if (!valid.ok()) {
    immediate = TransferResult{0, std::move(valid), false};
    immediate_ready = true;
    return true;
  }
  Proactor *proactor = &self->io_->proactor_for_current();
  state =
      detail::BudgetState{proactor, self->io_->operation_allocator(), options};
  if (state.await_ready())
    return true;
  if (!self->native_async_ &&
      (options.deadline.has_value() || options.cancel != nullptr)) {
    immediate = TransferResult{
        0,
        Status::Unimplemented(
            "bounded file IO requires a natively cancellable backend"),
        false};
    immediate_ready = true;
    return true;
  }
  return false;
}

bool File::WriteSomeAtAwaitable::await_suspend(
    std::coroutine_handle<>) noexcept {
  const std::uint32_t length = transfer_length(*self, buffer.size());
  if (!self->native_async_) {
    const std::int64_t result = offload(
        *self->io_, [handle = self->native_handle(), data = buffer.data(),
                     length, offset = offset]() noexcept {
          return file_write_sync(handle, data, length, offset);
        });
    immediate = detail::transfer_from_native(result, detail::kFileNativeDomain,
                                             "file write", false);
    immediate_ready = true;
    return false;
  }
  return state.await_suspend([&](std::uint64_t token) noexcept {
    return state.proactor->submit_write(self->native_handle(), buffer.data(),
                                        length, offset, token);
  });
}

TransferResult File::WriteSomeAtAwaitable::await_resume() noexcept {
  if (immediate_ready)
    return std::move(immediate);
  return detail::transfer_from_native(
      state.await_resume(), detail::kFileNativeDomain, "file write", false);
}

bool File::InfoAwaitable::await_suspend(std::coroutine_handle<>) noexcept {
  if (self == nullptr || !self->valid()) {
    preparation = Status::FailedPrecondition("file is not open");
    return false;
  }
  HYLE_FATAL_ASSERT(sched::current_worker_index() ==
                    static_cast<int>(self->owner_));
  (void)self->io_->proactor_for_current();
  result = offload(*self->io_, [handle = self->native_handle(),
                                output = &native]() noexcept {
    return file_stat_sync(handle, output);
  });
  return false;
}

Result<platform::FileInfo> File::InfoAwaitable::await_resume() noexcept {
  if (!preparation.ok())
    return make_unexpected(std::move(preparation));
  Status status = detail::status_from_native_result(
      result, detail::kFileNativeDomain, "file info");
  if (!status.ok())
    return make_unexpected(std::move(status));
  return file_info_from_native(native);
}

bool File::SyncAwaitable::await_suspend(std::coroutine_handle<>) noexcept {
  if (self == nullptr || !self->valid()) {
    preparation = Status::FailedPrecondition("file is not open");
    return false;
  }
  HYLE_FATAL_ASSERT(sched::current_worker_index() ==
                    static_cast<int>(self->owner_));
  Proactor &proactor = self->io_->proactor_for_current();
  const bool data_only = mode == FileSyncMode::kDataOnly;
  if (self->native_async_ && proactor.backend_kind() == BackendKind::kIoUring) {
    return submit_and_suspend(
        operation, submitted, [&](std::uint64_t token) noexcept {
          return proactor.submit_fsync(self->native_handle(), data_only, token);
        });
  }
  operation.result = offload(
      *self->io_, [handle = self->native_handle(), data_only]() noexcept {
        return file_sync_sync(handle, data_only);
      });
  return false;
}

Status File::SyncAwaitable::await_resume() noexcept {
  if (!preparation.ok())
    return std::move(preparation);
  return detail::status_from_native_result(
      operation.result, detail::kFileNativeDomain, "file sync");
}

bool File::PreallocateAwaitable::await_suspend(
    std::coroutine_handle<>) noexcept {
  if (self == nullptr || !self->valid() || length > kLargestFileOffset ||
      offset > kLargestFileOffset - length) {
    preparation = self == nullptr || !self->valid()
                      ? Status::FailedPrecondition("file is not open")
                      : Status::OutOfRange("file preallocate range overflows");
    return false;
  }
  HYLE_FATAL_ASSERT(sched::current_worker_index() ==
                    static_cast<int>(self->owner_));
  Proactor &proactor = self->io_->proactor_for_current();
  if (length == 0) {
    operation.result = 0;
    return false;
  }
  if (self->native_async_ && proactor.backend_kind() == BackendKind::kIoUring) {
    return submit_and_suspend(
        operation, submitted, [&](std::uint64_t token) noexcept {
          return proactor.submit_fallocate(self->native_handle(), 0, offset,
                                           length, token);
        });
  }
  operation.result =
      offload(*self->io_, [handle = self->native_handle(), offset = offset,
                           length = length]() noexcept {
        return file_preallocate_sync(handle, offset, length);
      });
  return false;
}

Status File::PreallocateAwaitable::await_resume() noexcept {
  if (!preparation.ok())
    return std::move(preparation);
  return detail::status_from_native_result(
      operation.result, detail::kFileNativeDomain, "file preallocate");
}

bool File::CloseAwaitable::await_suspend(std::coroutine_handle<>) noexcept {
  HYLE_FATAL_ASSERT(self != nullptr && self->valid() &&
                    sched::current_worker_index() ==
                        static_cast<int>(self->owner_));
  Proactor &proactor = self->io_->proactor_for_current();
  const std::intptr_t handle = self->handle_.release();
  self->native_async_ = false;
  self->cache_ = FileCacheMode::kBuffered;
  self->alignment_ = {};
  if (proactor.backend_kind() == BackendKind::kIoUring) {
    return submit_and_suspend(operation, submitted,
                              [&](std::uint64_t token) noexcept {
                                return proactor.submit_close(handle, token);
                              });
  }
  operation.result = offload(
      *self->io_, [handle]() noexcept { return file_close_sync(handle); });
  return false;
}

Status File::CloseAwaitable::await_resume() noexcept {
  return detail::status_from_native_result(
      operation.result, detail::kFileNativeDomain, "file close");
}

OpenFileAwaitable::OpenFileAwaitable(IoContext &context, unsigned owner_worker,
                                     std::string_view input,
                                     FileOpenOptions file_options) noexcept
    : io(&context), owner(owner_worker), options(file_options),
      path(context.operation_allocator()) {
  if (!context.valid()) {
    preparation =
        Status::FailedPrecondition("open_file requires a valid IoContext");
    return;
  }
  if (owner_worker >= context.num_workers()) {
    preparation = Status::OutOfRange("file owner worker is outside IoContext");
    return;
  }
  preparation = validate_open(input, options);
  if (!preparation.ok())
    return;
  if (!path.try_reserve(input.size() + 1)) {
    preparation = Status::ResourceExhausted("file path allocation failed");
    return;
  }
  for (char byte : input) {
    if (!path.try_push_back(byte)) {
      preparation = Status::ResourceExhausted("file path allocation failed");
      return;
    }
  }
  if (!path.try_push_back('\0'))
    preparation = Status::ResourceExhausted("file path allocation failed");
}

OpenFileAwaitable::~OpenFileAwaitable() {
  if (native_handle != platform::FileHandle::kInvalid)
    (void)file_close_sync(native_handle);
}

OpenFileAwaitable::OpenFileAwaitable(OpenFileAwaitable &&other) noexcept
    : io(std::exchange(other.io, nullptr)), owner(other.owner),
      options(other.options), path(std::move(other.path)),
      preparation(std::move(other.preparation)),
      native_result(other.native_result),
      native_handle(
          std::exchange(other.native_handle, platform::FileHandle::kInvalid)),
      alignment(other.alignment), native_async(other.native_async) {}

OpenFileAwaitable &
OpenFileAwaitable::operator=(OpenFileAwaitable &&other) noexcept {
  if (this != &other) {
    if (native_handle != platform::FileHandle::kInvalid)
      (void)file_close_sync(native_handle);
    io = std::exchange(other.io, nullptr);
    owner = other.owner;
    options = other.options;
    path = std::move(other.path);
    preparation = std::move(other.preparation);
    native_result = other.native_result;
    native_handle =
        std::exchange(other.native_handle, platform::FileHandle::kInvalid);
    alignment = other.alignment;
    native_async = other.native_async;
  }
  return *this;
}

bool OpenFileAwaitable::await_ready() noexcept { return !preparation.ok(); }

bool OpenFileAwaitable::await_suspend(std::coroutine_handle<>) noexcept {
  HYLE_FATAL_ASSERT(io != nullptr &&
                    sched::current_worker_index() == static_cast<int>(owner));
  native_result =
      offload(*io, [allocator = io->operation_allocator(), input = path.data(),
                    options = options, output = &native_handle,
                    alignment = &alignment]() noexcept {
        return file_open_sync(allocator, input, options, output, alignment);
      });
  if (native_result < 0)
    return false;

  Proactor &proactor = io->proactor_for_current();
#if HYLE_OS_WINDOWS || HYLE_OS_LINUX
  native_async = proactor.has_async_file_data();
#else
  native_async = false;
#endif
  if (native_async && !proactor.register_handle(native_handle)) {
    (void)file_close_sync(native_handle);
    native_handle = platform::FileHandle::kInvalid;
    native_result = kSubmitFailed;
  }
  return false;
}

Result<File> OpenFileAwaitable::await_resume() noexcept {
  if (!preparation.ok())
    return make_unexpected(std::move(preparation));
  Status status = detail::status_from_native_result(
      native_result, detail::kFileNativeDomain, "open file");
  if (!status.ok())
    return make_unexpected(std::move(status));
  platform::FileHandle handle(native_handle);
  native_handle = platform::FileHandle::kInvalid;
  return File(*io, std::move(handle), owner, native_async, options.cache,
              alignment);
}

sched::Task<TransferResult> read_exact_at(mem::AllocatorRef, File &file,
                                          std::uint64_t offset,
                                          std::span<std::byte> buffer,
                                          OperationOptions options) noexcept {
  std::size_t done = 0;
  while (done < buffer.size()) {
    if (offset > (std::numeric_limits<std::uint64_t>::max)() - done) {
      co_return TransferResult{
          done, Status::OutOfRange("file read offset overflows"), false};
    }
    TransferResult part = co_await file.read_some_at(
        offset + done, buffer.subspan(done), options);
    done += part.bytes;
    if (!part.ok()) {
      part.bytes = done;
      co_return part;
    }
    if (part.eof) {
      co_return TransferResult{
          done, Status::DataLoss("EOF before exact file read completed"), true};
    }
    if (part.bytes == 0) {
      co_return TransferResult{
          done, Status::Unavailable("file read made no forward progress"),
          false};
    }
  }
  co_return TransferResult{done, Status{}, false};
}

sched::Task<TransferResult> write_all_at(mem::AllocatorRef, File &file,
                                         std::uint64_t offset,
                                         std::span<const std::byte> buffer,
                                         OperationOptions options) noexcept {
  std::size_t done = 0;
  while (done < buffer.size()) {
    if (offset > (std::numeric_limits<std::uint64_t>::max)() - done) {
      co_return TransferResult{
          done, Status::OutOfRange("file write offset overflows"), false};
    }
    TransferResult part = co_await file.write_some_at(
        offset + done, buffer.subspan(done), options);
    done += part.bytes;
    if (!part.ok()) {
      part.bytes = done;
      co_return part;
    }
    if (part.bytes == 0) {
      co_return TransferResult{
          done, Status::Unavailable("file write made no forward progress"),
          false};
    }
  }
  co_return TransferResult{done, Status{}, false};
}

} // namespace hyle::io

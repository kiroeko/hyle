#include "hyle/io/local.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <utility>

#include "hyle/core/environment.h"
#include "hyle/error/system_error.h"
#include "hyle/io/detail/io_context_access.h"
#include "hyle/sched/blocking.h"
#include "hyle/sched/cancel.h"

#if HYLE_OS_WINDOWS
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <cerrno>
#  include <cstddef>
#  include <fcntl.h>
#  include <sys/socket.h>
#  include <sys/stat.h>
#  include <sys/un.h>
#  include <unistd.h>
#endif

namespace hyle::io {
namespace {

constexpr std::uint32_t kLargestNativeTransfer = 0x7fff'ffffu;

Status validate_local_endpoint_kind(const LocalEndpoint& endpoint) noexcept {
#if HYLE_OS_WINDOWS
  if (endpoint.kind() != LocalEndpointKind::kWindowsNamedPipe)
    return Status::Unimplemented(
        "Windows local IPC requires a named-pipe endpoint");
#elif HYLE_OS_LINUX
  if (endpoint.kind() != LocalEndpointKind::kUnixPath &&
      endpoint.kind() != LocalEndpointKind::kLinuxAbstract)
    return Status::Unimplemented(
        "Linux local IPC requires a Unix or abstract endpoint");
#else
  if (endpoint.kind() != LocalEndpointKind::kUnixPath)
    return Status::Unimplemented(
        "this platform supports filesystem Unix endpoints only");
#endif
  return Status{};
}

void close_local_handle(std::intptr_t handle) noexcept {
  if (handle < 0) return;
#if HYLE_OS_WINDOWS
  (void)::CloseHandle(reinterpret_cast<HANDLE>(handle));
#else
  (void)::close(static_cast<int>(handle));
#endif
}

Status local_native_error(const char* operation) noexcept {
#if HYLE_OS_WINDOWS
  return status_from_win32(::GetLastError(), operation);
#else
  return status_from_errno(errno, operation);
#endif
}

#if HYLE_OS_WINDOWS

Status pipe_path(const LocalEndpoint& endpoint, wchar_t* output,
                 std::size_t capacity) noexcept {
  if (endpoint.kind() != LocalEndpointKind::kWindowsNamedPipe)
    return Status::Unimplemented(
        "Windows local IPC requires a named-pipe endpoint");
  constexpr wchar_t kPrefix[] = L"\\\\.\\pipe\\";
  constexpr std::size_t kPrefixLength = std::size(kPrefix) - 1;
  if (capacity <= kPrefixLength)
    return Status::ResourceExhausted("named-pipe path buffer is too small");
  std::memcpy(output, kPrefix, kPrefixLength * sizeof(wchar_t));
  const std::string_view name = endpoint.name();
  const int converted = ::MultiByteToWideChar(
      CP_UTF8, MB_ERR_INVALID_CHARS, name.data(), static_cast<int>(name.size()),
      output + kPrefixLength,
      static_cast<int>(capacity - kPrefixLength - 1));
  if (converted <= 0)
    return status_from_win32(::GetLastError(), "decode named-pipe UTF-8 name");
  const std::size_t complete = kPrefixLength + static_cast<std::size_t>(converted);
  if (complete >= 256)
    return Status::OutOfRange(
        "Windows named-pipe path exceeds 255 UTF-16 code units");
  output[complete] = L'\0';
  return Status{};
}

Result<std::intptr_t> create_pipe_instance(
    IoContext& io, unsigned owner, const LocalEndpoint& endpoint,
    const LocalListenOptions& options, bool first) noexcept {
  wchar_t path[1024]{};
  Status encoded = pipe_path(endpoint, path, std::size(path));
  if (!encoded.ok()) return make_unexpected(std::move(encoded));
  if (options.max_pipe_instances == 0 || options.max_pipe_instances > 255)
    return make_unexpected(Status::OutOfRange(
        "max_pipe_instances must be between 1 and 255"));
  DWORD open_mode = PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED;
  if (first) open_mode |= FILE_FLAG_FIRST_PIPE_INSTANCE;
  DWORD pipe_mode = PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT;
#  if defined(PIPE_REJECT_REMOTE_CLIENTS)
  pipe_mode |= PIPE_REJECT_REMOTE_CLIENTS;
#  endif
  HANDLE pipe = ::CreateNamedPipeW(
      path, open_mode, pipe_mode, options.max_pipe_instances,
      options.pipe_output_buffer, options.pipe_input_buffer, 0, nullptr);
  if (pipe == INVALID_HANDLE_VALUE)
    return make_unexpected(local_native_error("create local named pipe"));
  const std::intptr_t handle = reinterpret_cast<std::intptr_t>(pipe);
  if (!detail::IoContextAccess::bind_handle(io, owner, handle)) {
    Status status = local_native_error("associate named pipe with IO shard");
    ::CloseHandle(pipe);
    return make_unexpected(std::move(status));
  }
  return handle;
}

Result<std::intptr_t> connect_pipe_blocking(
    const LocalEndpoint& endpoint, OperationOptions options) noexcept {
  wchar_t path[1024]{};
  Status encoded = pipe_path(endpoint, path, std::size(path));
  if (!encoded.ok()) return make_unexpected(std::move(encoded));

  for (;;) {
    if (options.cancel != nullptr && options.cancel->is_cancelled())
      return make_unexpected(Status::Cancelled("local connect cancelled"));
    if (options.deadline.has_value() &&
        *options.deadline <= time::monotonic_now())
      return make_unexpected(
          Status::DeadlineExceeded("local connect deadline exceeded"));

    HANDLE pipe = ::CreateFileW(
        path, GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING,
        FILE_FLAG_OVERLAPPED, nullptr);
    if (pipe != INVALID_HANDLE_VALUE)
      return reinterpret_cast<std::intptr_t>(pipe);

    const DWORD error = ::GetLastError();
    if (error != ERROR_PIPE_BUSY)
      return make_unexpected(
          status_from_win32(error, "connect local named pipe"));

    DWORD wait_ms = 20;
    if (options.deadline.has_value()) {
      const auto remaining = *options.deadline - time::monotonic_now();
      if (remaining <= time::Duration::zero())
        return make_unexpected(
            Status::DeadlineExceeded("local connect deadline exceeded"));
      const auto rounded = std::chrono::duration_cast<std::chrono::milliseconds>(
          remaining + std::chrono::microseconds(999));
      wait_ms = static_cast<DWORD>(std::clamp<std::int64_t>(
          rounded.count(), 1, static_cast<std::int64_t>(wait_ms)));
    }
    if (::WaitNamedPipeW(path, wait_ms) == 0) {
      const DWORD wait_error = ::GetLastError();
      if (wait_error != ERROR_SEM_TIMEOUT && wait_error != ERROR_PIPE_BUSY)
        return make_unexpected(
            status_from_win32(wait_error, "wait for local named pipe"));
    }
  }
}

#else

struct NativeLocalAddress {
  sockaddr_un address{};
  socklen_t length = 0;
};

static_assert(sizeof(sockaddr_un) <= 128,
              "LocalStream native-address storage is too small");

Result<NativeLocalAddress>
native_local_address(const LocalEndpoint& endpoint) noexcept {
  NativeLocalAddress native{};
  native.address.sun_family = AF_UNIX;
  const std::string_view name = endpoint.name();
  if (endpoint.kind() == LocalEndpointKind::kUnixPath) {
    if (name.size() >= sizeof(native.address.sun_path))
      return make_unexpected(
          Status::OutOfRange("Unix-domain socket path is too long"));
    std::memcpy(native.address.sun_path, name.data(), name.size());
    native.address.sun_path[name.size()] = '\0';
    native.length = static_cast<socklen_t>(
        offsetof(sockaddr_un, sun_path) + name.size() + 1);
    return native;
  }
  if (endpoint.kind() == LocalEndpointKind::kLinuxAbstract) {
#  if HYLE_OS_LINUX
    if (name.size() + 1 > sizeof(native.address.sun_path))
      return make_unexpected(
          Status::OutOfRange("Linux abstract socket name is too long"));
    native.address.sun_path[0] = '\0';
    std::memcpy(native.address.sun_path + 1, name.data(), name.size());
    native.length = static_cast<socklen_t>(
        offsetof(sockaddr_un, sun_path) + name.size() + 1);
    return native;
#  else
    return make_unexpected(Status::Unimplemented(
        "abstract local endpoints are supported only on Linux"));
#  endif
  }
  return make_unexpected(Status::Unimplemented(
      "POSIX local IPC requires a Unix endpoint"));
}

Result<std::intptr_t> create_local_socket() noexcept {
#  if HYLE_OS_LINUX
  int socket = ::socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
#  else
  int socket = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (socket >= 0) {
    const int status = ::fcntl(socket, F_GETFL, 0);
    if (status < 0 || ::fcntl(socket, F_SETFL, status | O_NONBLOCK) != 0 ||
        ::fcntl(socket, F_SETFD, FD_CLOEXEC) != 0) {
      const int saved = errno;
      ::close(socket);
      errno = saved;
      socket = -1;
    }
  }
#  endif
  if (socket < 0)
    return make_unexpected(local_native_error("create Unix-domain socket"));
  return static_cast<std::intptr_t>(socket);
}

Status remove_owned_socket(const LocalEndpoint& endpoint,
                           std::uint64_t device,
                           std::uint64_t inode) noexcept {
  if (endpoint.kind() != LocalEndpointKind::kUnixPath || inode == 0)
    return Status{};
  struct stat current{};
  const std::string_view path = endpoint.name();
  if (::lstat(path.data(), &current) != 0) {
    if (errno == ENOENT) return Status{};
    return local_native_error("inspect owned Unix socket path");
  }
  if (static_cast<std::uint64_t>(current.st_dev) != device ||
      static_cast<std::uint64_t>(current.st_ino) != inode)
    return Status::FailedPrecondition(
        "Unix socket path now belongs to a different filesystem node");
  if (::unlink(path.data()) != 0)
    return local_native_error("remove owned Unix socket path");
  return Status{};
}

#endif

template <class Buffer>
Status validate_local_buffers(
    const LocalStream* stream,
    const detail::BufferSequenceStorage<Buffer>& buffers) noexcept {
  if (stream == nullptr || !stream->valid())
    return Status::FailedPrecondition("local stream is not open");
  if (!buffers.ok()) return buffers.status();
  return Status{};
}

TransferResult local_read_result(std::int64_t result,
                                 const char* operation) noexcept {
#if HYLE_OS_WINDOWS
  // A byte-mode named/anonymous pipe reports peer closure as
  // ERROR_BROKEN_PIPE rather than a successful zero-byte ReadFile. It is the
  // stream EOF condition, not a FailedPrecondition on the local owner.
  if (result == -static_cast<std::int64_t>(ERROR_BROKEN_PIPE))
    return TransferResult{0, Status{}, true};
#endif
  return detail::transfer_from_native(result, detail::kFileNativeDomain,
                                      operation, true);
}

} // namespace

Result<LocalEndpoint> LocalEndpoint::make(LocalEndpointKind kind,
                                          std::string_view name) noexcept {
  if (name.empty() || name.find('\0') != std::string_view::npos)
    return make_unexpected(
        Status::InvalidArgument("local endpoint name must be non-empty UTF-8"));
  if (name.size() >= LocalEndpoint{}.bytes_.size())
    return make_unexpected(Status::OutOfRange("local endpoint name is too long"));
  LocalEndpoint endpoint;
  std::memcpy(endpoint.bytes_.data(), name.data(), name.size());
  endpoint.bytes_[name.size()] = '\0';
  endpoint.size_ = static_cast<std::uint16_t>(name.size());
  endpoint.kind_ = kind;
  endpoint.valid_ = true;
  return endpoint;
}

Result<LocalEndpoint>
LocalEndpoint::unix_path(std::string_view path) noexcept {
  return make(LocalEndpointKind::kUnixPath, path);
}

Result<LocalEndpoint>
LocalEndpoint::linux_abstract(std::string_view name) noexcept {
  return make(LocalEndpointKind::kLinuxAbstract, name);
}

Result<LocalEndpoint>
LocalEndpoint::windows_named_pipe(std::string_view name) noexcept {
  return make(LocalEndpointKind::kWindowsNamedPipe, name);
}

LocalStream::LocalStream(IoContext& io, std::intptr_t handle,
                         unsigned owner_worker,
                         const LocalEndpoint& endpoint) noexcept
    : io_(&io), handle_(handle), owner_(owner_worker), endpoint_(endpoint) {}

LocalStream::~LocalStream() { (void)close(); }

LocalStream::LocalStream(LocalStream&& other) noexcept
    : io_(std::exchange(other.io_, nullptr)),
      handle_(std::exchange(other.handle_, -1)), owner_(other.owner_),
      endpoint_(other.endpoint_) {}

LocalStream& LocalStream::operator=(LocalStream&& other) noexcept {
  if (this != &other) {
    (void)close();
    io_ = std::exchange(other.io_, nullptr);
    handle_ = std::exchange(other.handle_, -1);
    owner_ = other.owner_;
    endpoint_ = other.endpoint_;
  }
  return *this;
}

Result<LocalStream> LocalStream::adopt(IoContext& io, unsigned owner_worker,
                                       std::intptr_t native_handle,
                                       const LocalEndpoint& endpoint) noexcept {
  if (!io.valid()) {
    close_local_handle(native_handle);
    return make_unexpected(
        Status::FailedPrecondition("local adopt requires a valid IoContext"));
  }
  if (owner_worker >= io.num_workers()) {
    close_local_handle(native_handle);
    return make_unexpected(
        Status::OutOfRange("local stream owner is outside IoContext"));
  }
  if (native_handle < 0 ||
#if HYLE_OS_WINDOWS
      native_handle == 0 ||
#endif
      !endpoint.valid()) {
    close_local_handle(native_handle);
    return make_unexpected(
        Status::InvalidArgument("local adopt requires a handle and endpoint"));
  }
  Status endpoint_status = validate_local_endpoint_kind(endpoint);
  if (!endpoint_status.ok()) {
    close_local_handle(native_handle);
    return make_unexpected(std::move(endpoint_status));
  }
  if (!detail::IoContextAccess::bind_handle(io, owner_worker,
                                            native_handle)) {
    Status status = local_native_error("associate local stream with IO shard");
    close_local_handle(native_handle);
    return make_unexpected(std::move(status));
  }
  return LocalStream(io, native_handle, owner_worker, endpoint);
}

Result<LocalStream>
LocalStream::adopt_anonymous(IoContext& io, unsigned owner_worker,
                             std::intptr_t native_handle) noexcept {
  if (!io.valid()) {
    close_local_handle(native_handle);
    return make_unexpected(Status::FailedPrecondition(
        "anonymous local adopt requires a valid IoContext"));
  }
  if (owner_worker >= io.num_workers()) {
    close_local_handle(native_handle);
    return make_unexpected(Status::OutOfRange(
        "anonymous local stream owner is outside IoContext"));
  }
  bool invalid_handle = native_handle < 0;
#if HYLE_OS_WINDOWS
  invalid_handle = invalid_handle || native_handle == 0;
#endif
  if (invalid_handle) {
    close_local_handle(native_handle);
    return make_unexpected(Status::InvalidArgument(
        "anonymous local adopt requires a valid native handle"));
  }
  if (!detail::IoContextAccess::bind_handle(io, owner_worker,
                                            native_handle)) {
    Status status =
        local_native_error("associate anonymous local stream with IO shard");
    close_local_handle(native_handle);
    return make_unexpected(std::move(status));
  }
  return LocalStream(io, native_handle, owner_worker, LocalEndpoint{});
}

std::intptr_t LocalStream::release_native_handle() noexcept {
  io_ = nullptr;
  return std::exchange(handle_, -1);
}

Status LocalStream::shutdown_write() noexcept {
  if (!valid())
    return Status::FailedPrecondition("local stream is not open");
#if HYLE_OS_WINDOWS
  return Status::Unimplemented(
      "Windows byte-mode named pipes have no portable half-close");
#else
  if (::shutdown(static_cast<int>(handle_), SHUT_WR) != 0)
    return local_native_error("shut down local stream write side");
  return Status{};
#endif
}

Status LocalStream::close() noexcept {
  if (!valid()) return Status{};
  const std::intptr_t handle = std::exchange(handle_, -1);
  io_ = nullptr;
#if HYLE_OS_WINDOWS
  if (::CloseHandle(reinterpret_cast<HANDLE>(handle)) == 0)
    return local_native_error("close local stream");
#else
  if (::close(static_cast<int>(handle)) != 0)
    return local_native_error("close local stream");
#endif
  return Status{};
}

bool LocalStream::ConnectAwaitable::await_ready() noexcept {
  if (io == nullptr || !io->valid()) {
    preparation =
        Status::FailedPrecondition("local connect requires a valid IoContext");
    return true;
  }
  if (owner >= io->num_workers()) {
    preparation = Status::OutOfRange("local stream owner is outside IoContext");
    return true;
  }
  if (!endpoint.valid()) {
    preparation = Status::InvalidArgument("local endpoint is invalid");
    return true;
  }
  HYLE_FATAL_ASSERT(sched::current_worker_index() == static_cast<int>(owner));
#if HYLE_OS_WINDOWS
  if (endpoint.kind() != LocalEndpointKind::kWindowsNamedPipe) {
    preparation = Status::Unimplemented(
        "Windows local connect requires a named-pipe endpoint");
    return true;
  }
  if (options.cancel != nullptr && options.cancel->is_cancelled()) {
    preparation = Status::Cancelled("local connect cancelled");
    return true;
  }
  if (options.deadline.has_value() &&
      *options.deadline <= time::monotonic_now()) {
    preparation = Status::DeadlineExceeded("local connect deadline exceeded");
    return true;
  }
  return false;
#else
  auto address = native_local_address(endpoint);
  if (!address.has_value()) {
    preparation = std::move(address).error();
    return true;
  }
  auto socket = create_local_socket();
  if (!socket.has_value()) {
    preparation = std::move(socket).error();
    return true;
  }
  handle = *socket;
  std::memcpy(native_address.data(), &address->address, address->length);
  native_length = address->length;
  if (!detail::IoContextAccess::bind_handle(*io, owner, handle)) {
    preparation = local_native_error("associate local socket with IO shard");
    close_local_handle(std::exchange(handle, -1));
    return true;
  }
  inner = detail::BudgetConnectOp{
      &detail::IoContextAccess::proactor_for_current(*io), handle,
      native_address.data(), native_length,
      detail::IoContextAccess::operation_allocator(*io), options};
  return inner.await_ready();
#endif
}

bool LocalStream::ConnectAwaitable::await_suspend(
    std::coroutine_handle<> coroutine) noexcept {
#if HYLE_OS_WINDOWS
  (void)coroutine;
  auto job = sched::spawn_blocking(
      detail::IoContextAccess::blocking_pool(*io),
      detail::IoContextAccess::operation_allocator(*io),
      [endpoint = endpoint, options = options]() noexcept {
        return connect_pipe_blocking(endpoint, options);
      });
  auto joined = job.join();
  if (!joined.has_value()) {
    preparation = std::move(joined).error();
    return false;
  }
  if (!joined->has_value()) {
    preparation = std::move(*joined).error();
    return false;
  }
  handle = **joined;
  if (!detail::IoContextAccess::bind_handle(*io, owner, handle)) {
    preparation = local_native_error("associate local pipe with IO shard");
    close_local_handle(std::exchange(handle, -1));
    return false;
  }
  platform_ready = true;
  return false;
#else
  return inner.await_suspend(coroutine);
#endif
}

Result<LocalStream> LocalStream::ConnectAwaitable::await_resume() noexcept {
  if (!preparation.ok()) {
    close_local_handle(std::exchange(handle, -1));
    return make_unexpected(std::move(preparation));
  }
#if !HYLE_OS_WINDOWS
  TransferResult connect_result = detail::transfer_from_native(
      inner.await_resume(), detail::kNetworkNativeDomain, "connect local stream",
      false);
  if (!connect_result.ok()) {
    close_local_handle(std::exchange(handle, -1));
    return make_unexpected(std::move(connect_result.status));
  }
#else
  if (!platform_ready) {
    close_local_handle(std::exchange(handle, -1));
    return make_unexpected(Status::Internal("local pipe connect did not run"));
  }
#endif
  const std::intptr_t connected_handle = std::exchange(handle, -1);
  return LocalStream(*io, connected_handle, owner, endpoint);
}

bool LocalStream::ReadSomeAwaitable::await_ready() noexcept {
  if (self == nullptr || !self->valid()) {
    immediate.status = Status::FailedPrecondition("local stream is not open");
    immediate_ready = true;
    return true;
  }
  HYLE_FATAL_ASSERT(sched::current_worker_index() ==
                    static_cast<int>(self->owner_));
  if (buffer.empty()) {
    immediate_ready = true;
    return true;
  }
  if (buffer.size() > kLargestNativeTransfer) {
    immediate.status = Status::OutOfRange("local read exceeds native limit");
    immediate_ready = true;
    return true;
  }
  state = detail::BudgetState{
      &detail::IoContextAccess::proactor_for_current(*self->io_),
      detail::IoContextAccess::operation_allocator(*self->io_), options};
  return state.await_ready();
}

bool LocalStream::ReadSomeAwaitable::await_suspend(
    std::coroutine_handle<>) noexcept {
  return state.await_suspend([&](std::uint64_t token) noexcept {
    return state.proactor->submit_read(
        self->handle_, buffer.data(), static_cast<std::uint32_t>(buffer.size()),
        0, token);
  });
}

TransferResult LocalStream::ReadSomeAwaitable::await_resume() noexcept {
  if (immediate_ready) return std::move(immediate);
  return local_read_result(state.await_resume(), "local stream read");
}

bool LocalStream::WriteSomeAwaitable::await_ready() noexcept {
  if (self == nullptr || !self->valid()) {
    immediate.status = Status::FailedPrecondition("local stream is not open");
    immediate_ready = true;
    return true;
  }
  HYLE_FATAL_ASSERT(sched::current_worker_index() ==
                    static_cast<int>(self->owner_));
  if (buffer.empty()) {
    immediate_ready = true;
    return true;
  }
  if (buffer.size() > kLargestNativeTransfer) {
    immediate.status = Status::OutOfRange("local write exceeds native limit");
    immediate_ready = true;
    return true;
  }
  state = detail::BudgetState{
      &detail::IoContextAccess::proactor_for_current(*self->io_),
      detail::IoContextAccess::operation_allocator(*self->io_), options};
  return state.await_ready();
}

bool LocalStream::WriteSomeAwaitable::await_suspend(
    std::coroutine_handle<>) noexcept {
  return state.await_suspend([&](std::uint64_t token) noexcept {
    return state.proactor->submit_write(
        self->handle_, buffer.data(),
        static_cast<std::uint32_t>(buffer.size()), 0, token);
  });
}

TransferResult LocalStream::WriteSomeAwaitable::await_resume() noexcept {
  if (immediate_ready) return std::move(immediate);
  return detail::transfer_from_native(state.await_resume(),
                                      detail::kFileNativeDomain,
                                      "local stream write", false);
}

bool LocalStream::ReadBuffersAwaitable::await_ready() noexcept {
  Status valid = validate_local_buffers(self, buffers);
  if (!valid.ok()) {
    immediate.status = std::move(valid);
    immediate_ready = true;
    return true;
  }
  HYLE_FATAL_ASSERT(sched::current_worker_index() ==
                    static_cast<int>(self->owner_));
  if (buffers.empty()) {
    immediate.progress = buffers.progress(0);
    immediate_ready = true;
    return true;
  }
  state = detail::BudgetState{
      &detail::IoContextAccess::proactor_for_current(*self->io_),
      detail::IoContextAccess::operation_allocator(*self->io_), options};
  if (state.await_ready()) return true;
#if HYLE_OS_WINDOWS
  if (!staging.allocate(
          detail::IoContextAccess::operation_allocator(*self->io_),
          buffers.total_size(), alignof(std::max_align_t))) {
    immediate.status =
        Status::ResourceExhausted("local read staging allocation failed");
    immediate_ready = true;
    return true;
  }
#endif
  return false;
}

bool LocalStream::ReadBuffersAwaitable::await_suspend(
    std::coroutine_handle<>) noexcept {
  return state.await_suspend([&](std::uint64_t token) noexcept {
#if HYLE_OS_WINDOWS
    return state.proactor->submit_read(
        self->handle_, staging.data(),
        static_cast<std::uint32_t>(buffers.total_size()), 0, token);
#else
    return state.proactor->submit_read_buffers(
        self->handle_, buffers.native_data(),
        static_cast<std::uint32_t>(buffers.native_count()), 0, token);
#endif
  });
}

BufferSequenceResult LocalStream::ReadBuffersAwaitable::await_resume() noexcept {
  if (!immediate_ready) {
    TransferResult transfer =
        local_read_result(state.await_resume(), "local vectored read");
    if (transfer.bytes > buffers.total_size()) {
      return BufferSequenceResult{
          {}, Status::Internal("local read exceeded requested bytes"), false};
    }
    immediate = BufferSequenceResult{buffers.progress(transfer.bytes),
                                     std::move(transfer.status), transfer.eof};
  }
#if HYLE_OS_WINDOWS
  if (staging.data() != nullptr && immediate.progress.bytes != 0)
    buffers.copy_from_contiguous(staging.data(), immediate.progress.bytes);
#endif
  return std::move(immediate);
}

bool LocalStream::WriteBuffersAwaitable::await_ready() noexcept {
  Status valid = validate_local_buffers(self, buffers);
  if (!valid.ok()) {
    immediate.status = std::move(valid);
    immediate_ready = true;
    return true;
  }
  HYLE_FATAL_ASSERT(sched::current_worker_index() ==
                    static_cast<int>(self->owner_));
  if (buffers.empty()) {
    immediate.progress = buffers.progress(0);
    immediate_ready = true;
    return true;
  }
  state = detail::BudgetState{
      &detail::IoContextAccess::proactor_for_current(*self->io_),
      detail::IoContextAccess::operation_allocator(*self->io_), options};
  if (state.await_ready()) return true;
#if HYLE_OS_WINDOWS
  if (!staging.allocate(
          detail::IoContextAccess::operation_allocator(*self->io_),
          buffers.total_size(), alignof(std::max_align_t))) {
    immediate.status =
        Status::ResourceExhausted("local write staging allocation failed");
    immediate_ready = true;
    return true;
  }
  buffers.copy_to_contiguous(staging.data());
#endif
  return false;
}

bool LocalStream::WriteBuffersAwaitable::await_suspend(
    std::coroutine_handle<>) noexcept {
  return state.await_suspend([&](std::uint64_t token) noexcept {
#if HYLE_OS_WINDOWS
    return state.proactor->submit_write(
        self->handle_, staging.data(),
        static_cast<std::uint32_t>(buffers.total_size()), 0, token);
#else
    return state.proactor->submit_write_buffers(
        self->handle_, buffers.native_data(),
        static_cast<std::uint32_t>(buffers.native_count()), 0, token);
#endif
  });
}

BufferSequenceResult
LocalStream::WriteBuffersAwaitable::await_resume() noexcept {
  if (immediate_ready) return std::move(immediate);
  TransferResult transfer = detail::transfer_from_native(
      state.await_resume(), detail::kFileNativeDomain, "local vectored write",
      false);
  if (transfer.bytes > buffers.total_size()) {
    return BufferSequenceResult{
        {}, Status::Internal("local write exceeded requested bytes"), false};
  }
  return BufferSequenceResult{buffers.progress(transfer.bytes),
                              std::move(transfer.status), false};
}

LocalListener::~LocalListener() { (void)close(); }

LocalListener::LocalListener(LocalListener&& other) noexcept
    : io_(std::exchange(other.io_, nullptr)), endpoint_(other.endpoint_),
      options_(other.options_), handle_(std::exchange(other.handle_, -1)),
      owner_(other.owner_),
      accept_active_(std::exchange(other.accept_active_, false)),
      path_device_(std::exchange(other.path_device_, 0)),
      path_inode_(std::exchange(other.path_inode_, 0)) {}

LocalListener& LocalListener::operator=(LocalListener&& other) noexcept {
  if (this != &other) {
    (void)close();
    io_ = std::exchange(other.io_, nullptr);
    endpoint_ = other.endpoint_;
    options_ = other.options_;
    handle_ = std::exchange(other.handle_, -1);
    owner_ = other.owner_;
    accept_active_ = std::exchange(other.accept_active_, false);
    path_device_ = std::exchange(other.path_device_, 0);
    path_inode_ = std::exchange(other.path_inode_, 0);
  }
  return *this;
}

Result<LocalListener> LocalListener::bind(
    IoContext& io, unsigned owner_worker, const LocalEndpoint& endpoint,
    const LocalListenOptions& options) noexcept {
  if (!io.valid())
    return make_unexpected(
        Status::FailedPrecondition("local bind requires a valid IoContext"));
  if (owner_worker >= io.num_workers())
    return make_unexpected(
        Status::OutOfRange("local listener owner is outside IoContext"));
  if (!endpoint.valid())
    return make_unexpected(Status::InvalidArgument("local endpoint is invalid"));
  Status endpoint_status = validate_local_endpoint_kind(endpoint);
  if (!endpoint_status.ok())
    return make_unexpected(std::move(endpoint_status));
  LocalListener listener;
  listener.io_ = &io;
  listener.endpoint_ = endpoint;
  listener.options_ = options;
  listener.owner_ = owner_worker;

#if HYLE_OS_WINDOWS
  auto pipe = create_pipe_instance(io, owner_worker, endpoint, options, true);
  if (!pipe.has_value()) return make_unexpected(std::move(pipe).error());
  listener.handle_ = *pipe;
#else
  if (options.backlog <= 0)
    return make_unexpected(
        Status::InvalidArgument("local listener backlog must be positive"));
  if ((options.unix_permissions & ~0777u) != 0)
    return make_unexpected(
        Status::InvalidArgument("Unix socket permissions contain invalid bits"));
  auto native = native_local_address(endpoint);
  if (!native.has_value()) return make_unexpected(std::move(native).error());
  if (endpoint.kind() == LocalEndpointKind::kUnixPath &&
      options.replace_existing_socket) {
    struct stat existing{};
    if (::lstat(endpoint.name().data(), &existing) == 0) {
      if (!S_ISSOCK(existing.st_mode))
        return make_unexpected(Status::AlreadyExists(
            "local endpoint path is not a socket and was not removed"));
      if (::unlink(endpoint.name().data()) != 0)
        return make_unexpected(local_native_error("remove stale Unix socket"));
    } else if (errno != ENOENT) {
      return make_unexpected(local_native_error("inspect Unix socket path"));
    }
  }
  auto socket = create_local_socket();
  if (!socket.has_value()) return make_unexpected(std::move(socket).error());
  listener.handle_ = *socket;
  if (::bind(static_cast<int>(listener.handle_),
             reinterpret_cast<const sockaddr*>(&native->address),
             native->length) != 0 ||
      ::listen(static_cast<int>(listener.handle_), options.backlog) != 0) {
    Status status = local_native_error("bind/listen Unix-domain socket");
    close_local_handle(std::exchange(listener.handle_, -1));
    return make_unexpected(std::move(status));
  }
  if (endpoint.kind() == LocalEndpointKind::kUnixPath) {
    if (::chmod(endpoint.name().data(),
                static_cast<mode_t>(options.unix_permissions)) != 0) {
      Status status = local_native_error("set Unix socket permissions");
      close_local_handle(std::exchange(listener.handle_, -1));
      (void)::unlink(endpoint.name().data());
      return make_unexpected(std::move(status));
    }
    struct stat owned{};
    if (::lstat(endpoint.name().data(), &owned) != 0) {
      Status status = local_native_error("inspect bound Unix socket");
      close_local_handle(std::exchange(listener.handle_, -1));
      (void)::unlink(endpoint.name().data());
      return make_unexpected(std::move(status));
    }
    listener.path_device_ = static_cast<std::uint64_t>(owned.st_dev);
    listener.path_inode_ = static_cast<std::uint64_t>(owned.st_ino);
  }
  if (!detail::IoContextAccess::bind_handle(io, owner_worker,
                                            listener.handle_)) {
    Status status = local_native_error("associate local listener with IO shard");
    close_local_handle(std::exchange(listener.handle_, -1));
    (void)remove_owned_socket(listener.endpoint_, listener.path_device_,
                              listener.path_inode_);
    return make_unexpected(std::move(status));
  }
#endif
  return listener;
}

Status LocalListener::close() noexcept {
  if (!valid()) return Status{};
  if (accept_active_)
    return Status::FailedPrecondition(
        "cannot close local listener while accept is in flight");
  Status result{};
  if (handle_ >= 0) {
    const std::intptr_t handle = std::exchange(handle_, -1);
#if HYLE_OS_WINDOWS
    if (::CloseHandle(reinterpret_cast<HANDLE>(handle)) == 0)
      result = local_native_error("close local named-pipe listener");
#else
    if (::close(static_cast<int>(handle)) != 0)
      result = local_native_error("close local listener");
#endif
  }
#if !HYLE_OS_WINDOWS
  if (options_.cleanup_path_on_close) {
    Status cleanup =
        remove_owned_socket(endpoint_, path_device_, path_inode_);
    if (result.ok() && !cleanup.ok()) result = std::move(cleanup);
  }
#endif
  io_ = nullptr;
  path_device_ = 0;
  path_inode_ = 0;
  return result;
}

bool LocalListener::AcceptAwaitable::await_ready() noexcept {
  if (self == nullptr || !self->valid()) {
    preparation = Status::FailedPrecondition("local listener is not open");
    return true;
  }
  HYLE_FATAL_ASSERT(sched::current_worker_index() ==
                    static_cast<int>(self->owner_));
  if (self->accept_active_) {
    preparation = Status::FailedPrecondition(
        "local listener already has an accept in flight");
    return true;
  }
#if HYLE_OS_WINDOWS
  if (self->handle_ < 0) {
    auto instance = create_pipe_instance(*self->io_, self->owner_,
                                         self->endpoint_, self->options_, false);
    if (!instance.has_value()) {
      preparation = std::move(instance).error();
      return true;
    }
    self->handle_ = *instance;
  }
#endif
  self->accept_active_ = true;
  armed = true;
  state = detail::BudgetState{
      &detail::IoContextAccess::proactor_for_current(*self->io_),
      detail::IoContextAccess::operation_allocator(*self->io_), options};
  return state.await_ready();
}

bool LocalListener::AcceptAwaitable::await_suspend(
    std::coroutine_handle<>) noexcept {
  return state.await_suspend([&](std::uint64_t token) noexcept {
#if HYLE_OS_WINDOWS
    return state.proactor->submit_pipe_accept(self->handle_, token);
#else
    return state.proactor->submit_accept(self->handle_, nullptr, nullptr, token);
#endif
  });
}

Result<LocalStream> LocalListener::AcceptAwaitable::await_resume() noexcept {
  if (!preparation.ok()) return make_unexpected(std::move(preparation));
  if (!armed)
    return make_unexpected(Status::Internal("local accept was not armed"));
  const std::int64_t result = state.await_resume();
  self->accept_active_ = false;
  if (result < 0) {
#if HYLE_OS_WINDOWS
    close_local_handle(std::exchange(self->handle_, -1));
#endif
    return make_unexpected(
        detail::transfer_from_native(result, detail::kFileNativeDomain,
                                     "accept local stream", false)
            .status);
  }
#if HYLE_OS_WINDOWS
  const std::intptr_t accepted = std::exchange(self->handle_, -1);
#else
  const std::intptr_t accepted = static_cast<std::intptr_t>(result);
  if (::fcntl(static_cast<int>(accepted), F_SETFD, FD_CLOEXEC) != 0) {
    Status status = local_native_error("mark accepted local stream close-on-exec");
    close_local_handle(accepted);
    return make_unexpected(std::move(status));
  }
  if (!detail::IoContextAccess::bind_handle(*self->io_, self->owner_,
                                            accepted)) {
    Status status = local_native_error("associate accepted local stream");
    close_local_handle(accepted);
    return make_unexpected(std::move(status));
  }
#endif
  return LocalStream(*self->io_, accepted, self->owner_, self->endpoint_);
}

sched::Task<TransferResult> read_exact(mem::AllocatorRef, LocalStream& stream,
                                      std::span<std::byte> buffer,
                                      OperationOptions options) noexcept {
  std::size_t done = 0;
  while (done < buffer.size()) {
    TransferResult part =
        co_await stream.read_some(buffer.subspan(done), options);
    done += part.bytes;
    if (!part.ok() || part.eof) {
      part.bytes = done;
      co_return part;
    }
    if (part.bytes == 0)
      co_return TransferResult{
          done, Status::Unavailable("local read made no forward progress"),
          false};
  }
  co_return TransferResult{done, Status{}, false};
}

sched::Task<TransferResult> write_all(mem::AllocatorRef, LocalStream& stream,
                                     std::span<const std::byte> buffer,
                                     OperationOptions options) noexcept {
  std::size_t done = 0;
  while (done < buffer.size()) {
    TransferResult part =
        co_await stream.write_some(buffer.subspan(done), options);
    done += part.bytes;
    if (!part.ok()) {
      part.bytes = done;
      co_return part;
    }
    if (part.bytes == 0)
      co_return TransferResult{
          done, Status::Unavailable("local write made no forward progress"),
          false};
  }
  co_return TransferResult{done, Status{}, false};
}

namespace {

sched::Task<BufferSequenceResult> read_exact_local_buffers(
    mem::AllocatorRef, LocalStream& stream,
    detail::BufferSequenceStorage<MutableBuffer> buffers,
    OperationOptions options) noexcept {
  if (!buffers.ok())
    co_return BufferSequenceResult{{}, buffers.status(), false};
  BufferSequenceProgress progress = buffers.progress(0);
  while (!progress.complete) {
    BufferSequenceResult part =
        co_await stream.read_some(buffers.sequence(), options, progress.next);
    progress.bytes += part.progress.bytes;
    progress.next = part.progress.next;
    progress.complete = part.progress.complete;
    if (!part.ok()) {
      part.progress = progress;
      co_return part;
    }
    if (part.eof)
      co_return BufferSequenceResult{
          progress,
          Status::DataLoss("EOF before exact local vectored read completed"),
          true};
    if (part.progress.bytes == 0)
      co_return BufferSequenceResult{
          progress,
          Status::Unavailable("local vectored read made no forward progress"),
          false};
  }
  co_return BufferSequenceResult{progress, Status{}, false};
}

sched::Task<BufferSequenceResult> write_all_local_buffers(
    mem::AllocatorRef, LocalStream& stream,
    detail::BufferSequenceStorage<ConstBuffer> buffers,
    OperationOptions options) noexcept {
  if (!buffers.ok())
    co_return BufferSequenceResult{{}, buffers.status(), false};
  BufferSequenceProgress progress = buffers.progress(0);
  while (!progress.complete) {
    BufferSequenceResult part =
        co_await stream.write_some(buffers.sequence(), options, progress.next);
    progress.bytes += part.progress.bytes;
    progress.next = part.progress.next;
    progress.complete = part.progress.complete;
    if (!part.ok()) {
      part.progress = progress;
      co_return part;
    }
    if (part.progress.bytes == 0)
      co_return BufferSequenceResult{
          progress,
          Status::Unavailable("local vectored write made no forward progress"),
          false};
  }
  co_return BufferSequenceResult{progress, Status{}, false};
}

} // namespace

sched::Task<BufferSequenceResult> read_exact(
    mem::AllocatorRef coroutine_allocator, LocalStream& stream,
    std::span<const MutableBuffer> buffers, OperationOptions options) noexcept {
  return read_exact_local_buffers(
      coroutine_allocator, stream,
      detail::BufferSequenceStorage<MutableBuffer>(buffers), options);
}

sched::Task<BufferSequenceResult> write_all(
    mem::AllocatorRef coroutine_allocator, LocalStream& stream,
    std::span<const ConstBuffer> buffers, OperationOptions options) noexcept {
  return write_all_local_buffers(
      coroutine_allocator, stream,
      detail::BufferSequenceStorage<ConstBuffer>(buffers), options);
}

} // namespace hyle::io

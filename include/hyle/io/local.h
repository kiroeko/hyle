#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "hyle/core/assert.h"
#include "hyle/error/result.h"
#include "hyle/io/buffer.h"
#include "hyle/io/io_context.h"
#include "hyle/io/net_budget.h"
#include "hyle/io/operation.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_IO_API
#elif defined(HYLE_IO_CORE_BUILD)
#  define HYLE_IO_API HYLE_EXPORT
#else
#  define HYLE_IO_API HYLE_IMPORT
#endif

namespace hyle::io {

enum class LocalEndpointKind : std::uint8_t {
  kUnixPath,
  kLinuxAbstract,
  kWindowsNamedPipe,
};

// Owning local-transport address. The factories are intentionally explicit:
// Unix filesystem lifetime, Linux abstract namespace, and Windows named-pipe
// lifetime are different contracts and are not disguised as one family flag.
class LocalEndpoint {
 public:
  LocalEndpoint() noexcept = default;

  HYLE_IO_API static Result<LocalEndpoint>
  unix_path(std::string_view path) noexcept;
  HYLE_IO_API static Result<LocalEndpoint>
  linux_abstract(std::string_view name) noexcept;
  HYLE_IO_API static Result<LocalEndpoint>
  windows_named_pipe(std::string_view name) noexcept;

  bool valid() const noexcept { return valid_; }
  explicit operator bool() const noexcept { return valid(); }
  LocalEndpointKind kind() const noexcept { return kind_; }
  std::string_view name() const noexcept { return {bytes_.data(), size_}; }

  friend bool operator==(const LocalEndpoint&, const LocalEndpoint&) noexcept =
      default;

 private:
  HYLE_IO_API static Result<LocalEndpoint>
  make(LocalEndpointKind kind, std::string_view name) noexcept;

  // Windows names are capped at 256 UTF-16 code units including the pipe
  // prefix. 1024 UTF-8 bytes can represent that whole legal range; Unix has a
  // smaller native limit which bind/connect validate without truncation.
  std::array<char, 1024> bytes_{};
  std::uint16_t size_ = 0;
  LocalEndpointKind kind_ = LocalEndpointKind::kUnixPath;
  bool valid_ = false;
};

struct LocalListenOptions {
  int backlog = 128;
  // Unix pathname only. Replacement refuses to unlink anything except an
  // existing socket node; ordinary files and directories are never removed.
  bool replace_existing_socket = false;
  bool cleanup_path_on_close = true;
  std::uint32_t unix_permissions = 0600;
  // Named-pipe instance count and kernel buffer hints. Zero buffer sizes ask
  // Windows to choose its defaults.
  unsigned max_pipe_instances = 64;
  std::uint32_t pipe_input_buffer = 64 * 1024;
  std::uint32_t pipe_output_buffer = 64 * 1024;
};

class LocalStream {
 public:
  LocalStream() noexcept = default;
  HYLE_IO_API ~LocalStream();
  HYLE_IO_API LocalStream(LocalStream&& other) noexcept;
  HYLE_IO_API LocalStream& operator=(LocalStream&& other) noexcept;
  LocalStream(const LocalStream&) = delete;
  LocalStream& operator=(const LocalStream&) = delete;

  HYLE_IO_API static Result<LocalStream>
  adopt(IoContext& io, unsigned owner_worker, std::intptr_t native_handle,
        const LocalEndpoint& endpoint) noexcept;

  bool valid() const noexcept { return handle_ >= 0; }
  explicit operator bool() const noexcept { return valid(); }
  std::intptr_t native_handle() const noexcept { return handle_; }
  unsigned owner_worker() const noexcept { return owner_; }
  const LocalEndpoint& endpoint() const noexcept { return endpoint_; }
  HYLE_IO_API std::intptr_t release_native_handle() noexcept;
  HYLE_IO_API Status shutdown_write() noexcept;
  HYLE_IO_API Status close() noexcept;

  struct ConnectAwaitable {
    IoContext* io = nullptr;
    unsigned owner = 0;
    LocalEndpoint endpoint;
    OperationOptions options{};
    alignas(16) std::array<std::byte, 128> native_address{};
    std::uint32_t native_length = 0;
    std::intptr_t handle = -1;
    BudgetConnectOp inner{};
    Status preparation{};
    bool platform_ready = false;

    HYLE_IO_API bool await_ready() noexcept;
    HYLE_IO_API bool await_suspend(std::coroutine_handle<>) noexcept;
    HYLE_IO_API Result<LocalStream> await_resume() noexcept;
  };

  static ConnectAwaitable connect(
      IoContext& io, unsigned owner_worker, const LocalEndpoint& endpoint,
      OperationOptions options = {}) noexcept {
    return ConnectAwaitable{&io, owner_worker, endpoint, options};
  }

  struct ReadSomeAwaitable {
    LocalStream* self = nullptr;
    std::span<std::byte> buffer;
    OperationOptions options{};
    detail::BudgetState state{};
    TransferResult immediate{};
    bool immediate_ready = false;

    HYLE_IO_API bool await_ready() noexcept;
    HYLE_IO_API bool await_suspend(std::coroutine_handle<>) noexcept;
    HYLE_IO_API TransferResult await_resume() noexcept;
  };

  struct WriteSomeAwaitable {
    LocalStream* self = nullptr;
    std::span<const std::byte> buffer;
    OperationOptions options{};
    detail::BudgetState state{};
    TransferResult immediate{};
    bool immediate_ready = false;

    HYLE_IO_API bool await_ready() noexcept;
    HYLE_IO_API bool await_suspend(std::coroutine_handle<>) noexcept;
    HYLE_IO_API TransferResult await_resume() noexcept;
  };

  struct ReadBuffersAwaitable {
    LocalStream* self = nullptr;
    detail::BufferSequenceStorage<MutableBuffer> buffers;
    OperationOptions options{};
    detail::BudgetState state{};
    detail::StagingBuffer staging{};
    BufferSequenceResult immediate{};
    bool immediate_ready = false;

    HYLE_IO_API bool await_ready() noexcept;
    HYLE_IO_API bool await_suspend(std::coroutine_handle<>) noexcept;
    HYLE_IO_API BufferSequenceResult await_resume() noexcept;
  };

  struct WriteBuffersAwaitable {
    LocalStream* self = nullptr;
    detail::BufferSequenceStorage<ConstBuffer> buffers;
    OperationOptions options{};
    detail::BudgetState state{};
    detail::StagingBuffer staging{};
    BufferSequenceResult immediate{};
    bool immediate_ready = false;

    HYLE_IO_API bool await_ready() noexcept;
    HYLE_IO_API bool await_suspend(std::coroutine_handle<>) noexcept;
    HYLE_IO_API BufferSequenceResult await_resume() noexcept;
  };

  ReadSomeAwaitable read_some(
      std::span<std::byte> buffer,
      OperationOptions options = {}) noexcept {
    return ReadSomeAwaitable{this, buffer, options};
  }
  WriteSomeAwaitable write_some(
      std::span<const std::byte> buffer,
      OperationOptions options = {}) noexcept {
    return WriteSomeAwaitable{this, buffer, options};
  }
  ReadBuffersAwaitable read_some(
      std::span<const MutableBuffer> buffers,
      OperationOptions options = {},
      BufferSequenceCursor cursor = {}) noexcept {
    return ReadBuffersAwaitable{
        this, detail::BufferSequenceStorage<MutableBuffer>(buffers, cursor),
        options};
  }
  WriteBuffersAwaitable write_some(
      std::span<const ConstBuffer> buffers,
      OperationOptions options = {},
      BufferSequenceCursor cursor = {}) noexcept {
    return WriteBuffersAwaitable{
        this, detail::BufferSequenceStorage<ConstBuffer>(buffers, cursor),
        options};
  }

 private:
  HYLE_IO_API LocalStream(IoContext& io, std::intptr_t handle,
                          unsigned owner_worker,
                          const LocalEndpoint& endpoint) noexcept;

  IoContext* io_ = nullptr;
  std::intptr_t handle_ = -1;
  unsigned owner_ = 0;
  LocalEndpoint endpoint_{};

  friend class LocalListener;
};

class LocalListener {
 public:
  LocalListener() noexcept = default;
  HYLE_IO_API ~LocalListener();
  HYLE_IO_API LocalListener(LocalListener&& other) noexcept;
  HYLE_IO_API LocalListener& operator=(LocalListener&& other) noexcept;
  LocalListener(const LocalListener&) = delete;
  LocalListener& operator=(const LocalListener&) = delete;

  HYLE_IO_API static Result<LocalListener>
  bind(IoContext& io, unsigned owner_worker, const LocalEndpoint& endpoint,
       const LocalListenOptions& options = {}) noexcept;

  bool valid() const noexcept { return io_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  unsigned owner_worker() const noexcept { return owner_; }
  const LocalEndpoint& endpoint() const noexcept { return endpoint_; }
  HYLE_IO_API Status close() noexcept;

  struct AcceptAwaitable {
    LocalListener* self = nullptr;
    OperationOptions options{};
    detail::BudgetState state{};
    Status preparation{};
    bool armed = false;

    HYLE_IO_API bool await_ready() noexcept;
    HYLE_IO_API bool await_suspend(std::coroutine_handle<>) noexcept;
    HYLE_IO_API Result<LocalStream> await_resume() noexcept;
  };

  AcceptAwaitable accept(OperationOptions options = {}) noexcept {
    return AcceptAwaitable{this, options};
  }

 private:
  IoContext* io_ = nullptr;
  LocalEndpoint endpoint_{};
  LocalListenOptions options_{};
  std::intptr_t handle_ = -1;
  unsigned owner_ = 0;
  bool accept_active_ = false;
  std::uint64_t path_device_ = 0;
  std::uint64_t path_inode_ = 0;
};

HYLE_IO_API sched::Task<TransferResult> read_exact(
    mem::AllocatorRef coroutine_allocator, LocalStream& stream,
    std::span<std::byte> buffer, OperationOptions options = {}) noexcept;
HYLE_IO_API sched::Task<TransferResult> write_all(
    mem::AllocatorRef coroutine_allocator, LocalStream& stream,
    std::span<const std::byte> buffer,
    OperationOptions options = {}) noexcept;
HYLE_IO_API sched::Task<BufferSequenceResult> read_exact(
    mem::AllocatorRef coroutine_allocator, LocalStream& stream,
    std::span<const MutableBuffer> buffers,
    OperationOptions options = {}) noexcept;
HYLE_IO_API sched::Task<BufferSequenceResult> write_all(
    mem::AllocatorRef coroutine_allocator, LocalStream& stream,
    std::span<const ConstBuffer> buffers,
    OperationOptions options = {}) noexcept;

} // namespace hyle::io

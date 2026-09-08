#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "hyle/container/string_view.h"
#include "hyle/error/result.h"
#include "hyle/io/operation.h"
#include "hyle/io/tcp.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_IO_TLS_API
#elif defined(HYLE_IO_TLS_BUILD)
#define HYLE_IO_TLS_API HYLE_EXPORT
#else
#define HYLE_IO_TLS_API HYLE_IMPORT
#endif

namespace hyle::io {

// Package/runtime capability query. A build without the TLS factor exposes no
// operation-shaped failure stubs; callers can test this before selecting TLS.
HYLE_IO_TLS_API bool tls_available() noexcept;

#if defined(HYLE_IO_TLS)

enum class TlsVersion : std::uint8_t {
  kProviderDefault,
  kTls12,
  kTls13,
};

enum class TlsVerifyMode : std::uint8_t {
  kNone,
  kPeer,
};

// All strings are borrowed only for make_client()/make_server(). The resulting
// context owns provider state and encoded ALPN data. Peer verification is the
// secure client default; tests and explicitly opportunistic transports must opt
// out rather than reaching through a provider-native handle.
struct TlsClientOptions {
  TlsVerifyMode verify = TlsVerifyMode::kPeer;
  TlsVersion min_version = TlsVersion::kTls12;
  TlsVersion max_version = TlsVersion::kProviderDefault;
  container::StringView certificate_authority_file{};
  // Optional client identity for mutual TLS. Supply both fields together;
  // make_client() loads them before returning and does not retain the paths.
  container::StringView certificate_chain_file{};
  container::StringView private_key_file{};
  std::span<const container::StringView> alpn_protocols{};
  // Backs the one reusable 16 KiB pump buffer allocated for each stream.
  // Custom allocator state must outlive every stream made from this context.
  mem::AllocatorRef connection_allocator = mem::system_allocator();
  bool require_alpn = false;
  // Session caching/tickets retain provider state and are explicit opt-in.
  bool enable_session_resumption = false;
  bool allow_truncated_eof = false;
};

struct TlsServerOptions {
  container::StringView certificate_chain_file{};
  container::StringView private_key_file{};
  TlsVerifyMode client_verification = TlsVerifyMode::kNone;
  TlsVersion min_version = TlsVersion::kTls12;
  TlsVersion max_version = TlsVersion::kProviderDefault;
  container::StringView client_certificate_authority_file{};
  std::span<const container::StringView> alpn_protocols{};
  // Same lifetime contract as TlsClientOptions::connection_allocator.
  mem::AllocatorRef connection_allocator = mem::system_allocator();
  bool require_alpn = false;
  bool enable_session_resumption = false;
  bool allow_truncated_eof = false;
};

// Stable convenience lists for HTTP negotiation, ordered by server preference.
// true -> {"h2", "http/1.1"}; false -> {"h2"}.
HYLE_IO_TLS_API std::span<const container::StringView>
tls_http2_protocols(bool allow_http1 = true) noexcept;

// An owning provider session ticket. A ticket is tied to the context that
// created it; connect() rejects a ticket from another context instead of
// silently weakening or misconfiguring resumption.
class TlsSession {
public:
  TlsSession() noexcept = default;
  HYLE_IO_TLS_API ~TlsSession();
  HYLE_IO_TLS_API TlsSession(TlsSession &&other) noexcept;
  HYLE_IO_TLS_API TlsSession &operator=(TlsSession &&other) noexcept;
  TlsSession(const TlsSession &) = delete;
  TlsSession &operator=(const TlsSession &) = delete;

  bool valid() const noexcept { return session_ != nullptr; }

private:
  friend class TlsStream;
  explicit TlsSession(void *session, std::uint64_t context_id) noexcept
      : session_(session), context_id_(context_id) {}

  void *session_ = nullptr;
  std::uint64_t context_id_ = 0;
};

// Shared immutable TLS configuration. The opaque provider state is retained by
// every stream, so destroying this handle while connections are active is safe.
class TlsContext {
public:
  TlsContext() noexcept = default;
  HYLE_IO_TLS_API ~TlsContext();
  HYLE_IO_TLS_API TlsContext(TlsContext &&other) noexcept;
  HYLE_IO_TLS_API TlsContext &operator=(TlsContext &&other) noexcept;
  TlsContext(const TlsContext &) = delete;
  TlsContext &operator=(const TlsContext &) = delete;

  bool valid() const noexcept { return impl_ != nullptr; }

  HYLE_IO_TLS_API static Result<TlsContext>
  make_client(const TlsClientOptions &options = {}) noexcept;
  HYLE_IO_TLS_API static Result<TlsContext>
  make_server(const TlsServerOptions &options) noexcept;

private:
  friend class TlsStream;
  explicit TlsContext(void *impl) noexcept : impl_(impl) {}
  void *impl_ = nullptr;
};

struct TlsConnectOptions {
  // Required when peer verification is enabled; used for both SNI (DNS names)
  // and certificate hostname/IP matching. Borrowed only during connect().
  container::StringView server_name{};
  const TlsSession *resume_session = nullptr;
};

// One-fiber encrypted transport over TcpStream. Construction is typed and
// recoverable; handshake/read/write carry one absolute OperationOptions budget
// through every underlying socket operation.
class TlsStream {
public:
  TlsStream() noexcept = default;
  HYLE_IO_TLS_API ~TlsStream();
  HYLE_IO_TLS_API TlsStream(TlsStream &&other) noexcept;
  HYLE_IO_TLS_API TlsStream &operator=(TlsStream &&other) noexcept;
  TlsStream(const TlsStream &) = delete;
  TlsStream &operator=(const TlsStream &) = delete;

  HYLE_IO_TLS_API static Result<TlsStream>
  connect(TcpStream socket, const TlsContext &context,
          const TlsConnectOptions &options = {}) noexcept;
  HYLE_IO_TLS_API static Result<TlsStream>
  accept(TcpStream socket, const TlsContext &context) noexcept;

  bool valid() const noexcept { return ssl_ != nullptr; }
  unsigned owner_worker() const noexcept { return socket_.owner_worker(); }
  HYLE_IO_TLS_API container::StringView negotiated_protocol() const noexcept;
  HYLE_IO_TLS_API bool session_reused() const noexcept;
  HYLE_IO_TLS_API Result<TlsSession> take_session() const noexcept;

private:
  friend HYLE_IO_TLS_API sched::Task<Status>
  tls_handshake(mem::AllocatorRef, TlsStream &, OperationOptions) noexcept;
  friend HYLE_IO_TLS_API sched::Task<TransferResult>
  tls_read(mem::AllocatorRef, TlsStream &, std::span<std::byte>,
           OperationOptions) noexcept;
  friend HYLE_IO_TLS_API sched::Task<TransferResult>
  tls_write(mem::AllocatorRef, TlsStream &, std::span<const std::byte>,
            OperationOptions) noexcept;
  friend HYLE_IO_TLS_API sched::Task<Status>
  tls_shutdown(mem::AllocatorRef, TlsStream &,
               struct TlsShutdownOptions) noexcept;

  HYLE_IO_TLS_API void reset_() noexcept;
  HYLE_IO_TLS_API static Result<TlsStream>
  make_(TcpStream socket, const TlsContext &context, bool server,
        const TlsConnectOptions *options) noexcept;

  TcpStream socket_{};
  void *ssl_ = nullptr;
  void *read_bio_ = nullptr;
  void *write_bio_ = nullptr;
  void *context_impl_ = nullptr;
  std::byte *scratch_ = nullptr;
  mem::AllocatorRef scratch_allocator_{};
  bool server_ = false;
  bool handshake_complete_ = false;
};

enum class TlsShutdownMode : std::uint8_t {
  // Send close_notify and return after the ciphertext has reached the socket.
  kSendOnly,
  // Also wait for the peer's close_notify under the supplied budget.
  kBidirectional,
};

struct TlsShutdownOptions {
  TlsShutdownMode mode = TlsShutdownMode::kSendOnly;
  OperationOptions operation{};
};

HYLE_IO_TLS_API sched::Task<Status>
tls_handshake(mem::AllocatorRef coroutine_allocator, TlsStream &stream,
              OperationOptions options = {}) noexcept;
HYLE_IO_TLS_API sched::Task<TransferResult>
tls_read(mem::AllocatorRef coroutine_allocator, TlsStream &stream,
         std::span<std::byte> buffer, OperationOptions options = {}) noexcept;
HYLE_IO_TLS_API sched::Task<TransferResult>
tls_write(mem::AllocatorRef coroutine_allocator, TlsStream &stream,
          std::span<const std::byte> buffer,
          OperationOptions options = {}) noexcept;
HYLE_IO_TLS_API sched::Task<Status>
tls_shutdown(mem::AllocatorRef coroutine_allocator, TlsStream &stream,
             TlsShutdownOptions options = {}) noexcept;

#endif // HYLE_IO_TLS

} // namespace hyle::io

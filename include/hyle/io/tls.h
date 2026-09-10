#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <type_traits>

#include "hyle/container/string_view.h"
#include "hyle/error/result.h"
#include "hyle/io/operation.h"
#include "hyle/io/tcp.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/util/function_ref.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_IO_TLS_API
#elif defined(HYLE_IO_TLS_BUILD)
#define HYLE_IO_TLS_API HYLE_EXPORT
#else
#define HYLE_IO_TLS_API HYLE_IMPORT
#endif

namespace hyle::io {

namespace detail {
struct TlsCredentialAccess;
}

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

enum class TlsCredentialSourceKind : std::uint8_t {
  kNone,
  kFile,
  kPem,
};

// A provider-neutral, non-owning credential input. The source is borrowed only
// during make_client()/make_server(); the resulting context owns parsed
// provider state, never the path or PEM bytes.
class TlsCredentialSource {
public:
  constexpr TlsCredentialSource() noexcept = default;

  static constexpr TlsCredentialSource
  file(container::StringView path) noexcept {
    return TlsCredentialSource(TlsCredentialSourceKind::kFile, path);
  }

  static constexpr TlsCredentialSource
  pem(container::StringView bytes) noexcept {
    return TlsCredentialSource(TlsCredentialSourceKind::kPem, bytes);
  }

  constexpr TlsCredentialSourceKind kind() const noexcept { return kind_; }
  constexpr container::StringView value() const noexcept { return value_; }
  constexpr bool supplied() const noexcept {
    return kind_ != TlsCredentialSourceKind::kNone;
  }

private:
  constexpr TlsCredentialSource(TlsCredentialSourceKind kind,
                                container::StringView value) noexcept
      : value_(value), kind_(kind) {}

  container::StringView value_{};
  TlsCredentialSourceKind kind_ = TlsCredentialSourceKind::kNone;
};

// Called synchronously, on the make_client()/make_server() thread, only when
// the provider needs to decrypt a private key. The callback and the successful
// returned view must remain valid until make_* returns; neither is retained.
using TlsPasswordCallback =
    util::FunctionRef<Result<container::StringView>() noexcept>;

enum class TlsPrivateKeyPasswordKind : std::uint8_t {
  kNone,
  kBorrowed,
  kCallback,
};

class TlsPrivateKeyPassword {
public:
  constexpr TlsPrivateKeyPassword() noexcept = default;

  static constexpr TlsPrivateKeyPassword
  borrowed(container::StringView password) noexcept {
    TlsPrivateKeyPassword result;
    result.kind_ = TlsPrivateKeyPasswordKind::kBorrowed;
    result.borrowed_ = password;
    return result;
  }

  static TlsPrivateKeyPassword callback(TlsPasswordCallback callback) noexcept {
    TlsPrivateKeyPassword result;
    result.kind_ = TlsPrivateKeyPasswordKind::kCallback;
    result.callback_.emplace(callback);
    return result;
  }

  // A temporary capturing callable would dangle if the options outlived the
  // full expression. Function pointers and explicit FunctionRef values remain
  // safe rvalues; stateful callables must be named lvalues.
  template <class Callback>
    requires(!std::is_lvalue_reference_v<Callback &&> &&
             !std::is_pointer_v<std::remove_cvref_t<Callback>> &&
             !std::is_same_v<std::remove_cvref_t<Callback>,
                             TlsPasswordCallback>)
  static TlsPrivateKeyPassword callback(Callback &&) = delete;

  constexpr TlsPrivateKeyPasswordKind kind() const noexcept { return kind_; }
  constexpr bool supplied() const noexcept {
    return kind_ != TlsPrivateKeyPasswordKind::kNone;
  }

private:
  friend struct detail::TlsCredentialAccess;

  constexpr container::StringView borrowed_value() const noexcept {
    return borrowed_;
  }
  const TlsPasswordCallback *callback_ref() const noexcept {
    return callback_ ? &*callback_ : nullptr;
  }

  container::StringView borrowed_{};
  std::optional<TlsPasswordCallback> callback_{};
  TlsPrivateKeyPasswordKind kind_ = TlsPrivateKeyPasswordKind::kNone;
};

// All option views are borrowed only for make_client()/make_server(). The
// resulting context owns provider state and encoded ALPN data. Peer
// verification is the secure client default; tests and explicitly
// opportunistic transports must opt out rather than reaching through a
// provider-native handle.
struct TlsClientOptions {
  TlsVerifyMode verify = TlsVerifyMode::kPeer;
  TlsVersion min_version = TlsVersion::kTls12;
  TlsVersion max_version = TlsVersion::kProviderDefault;
  // kNone uses the provider's system trust store when peer verification is on.
  TlsCredentialSource certificate_authority{};
  // Optional client identity for mutual TLS. Supply both fields together;
  // certificate and key sources may independently be file or borrowed PEM.
  TlsCredentialSource certificate_chain{};
  TlsCredentialSource private_key{};
  TlsPrivateKeyPassword private_key_password{};
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
  TlsCredentialSource certificate_chain{};
  TlsCredentialSource private_key{};
  TlsPrivateKeyPassword private_key_password{};
  TlsVerifyMode client_verification = TlsVerifyMode::kNone;
  TlsVersion min_version = TlsVersion::kTls12;
  TlsVersion max_version = TlsVersion::kProviderDefault;
  TlsCredentialSource client_certificate_authority{};
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
// Write-some operation. A successful call submits at most one provider record;
// advance by TransferResult::bytes until the complete plaintext is sent.
HYLE_IO_TLS_API sched::Task<TransferResult>
tls_write(mem::AllocatorRef coroutine_allocator, TlsStream &stream,
          std::span<const std::byte> buffer,
          OperationOptions options = {}) noexcept;
HYLE_IO_TLS_API sched::Task<Status>
tls_shutdown(mem::AllocatorRef coroutine_allocator, TlsStream &stream,
             TlsShutdownOptions options = {}) noexcept;

#endif // HYLE_IO_TLS

} // namespace hyle::io

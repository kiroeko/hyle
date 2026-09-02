#pragma once

// io-08 TLS (capability C9): an encrypted socket layered over the io-04 co_await
// TcpStream, borrowing OpenSSL as a memory-BIO shim -- the SSL engine only does the
// handshake + encrypt/decrypt (a pure in-memory, synchronous state machine); the
// ciphertext is carried by our own co_await TcpStream read/write. This is the
// asio ssl::stream / seastar tls_openssl / libevent bufferevent_ssl model:
// SSL_set_bio(ssl, rbio, wbio) with two BIO_s_mem, and an outer PUMP loop that,
// after each SSL_* call, flushes wbio's ciphertext to the peer and/or feeds the
// peer's ciphertext into rbio, driven by SSL_get_error's WANT_READ/WANT_WRITE.
//
// Only compiled when OpenSSL is available (CMake sets HYLE_IO_TLS). Without it this
// header is inert, so a TLS-free build carries no OpenSSL dependency.
//
// LIFETIME / THREADING: a TlsStream is owned by ONE fiber (the connection's), which
// is pinned to its owner worker (shard-per-core). SSL is not thread-safe, but a
// single fiber accessing it serially IS safe -- while a co_await suspends the
// fiber, no other fiber touches this SSL. The one subtlety (OpenSSL's per-thread
// error queue) is handled inside tls.cc: the SSL_call -> SSL_get_error ->
// ERR_get_error triple is never split by a co_await.

#if defined(HYLE_IO_TLS)

#include <cstddef>
#include <cstdint>

#include "hyle/io/tcp.h"
#include "hyle/core/compiler.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/container/string_view.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_IO_API
#elif defined(HYLE_IO_BUILD)
#  define HYLE_IO_API HYLE_EXPORT
#else
#  define HYLE_IO_API HYLE_IMPORT
#endif

// Opaque OpenSSL types kept out of this header (no <openssl/*> leak into consumers).
struct ssl_ctx_st;
struct ssl_st;
struct bio_st;

namespace hyle::io {

// A shared TLS configuration (one SSL_CTX: certificate + key + CA store + protocol
// options). Create once per listener/client; each connection makes its own SSL from
// it (cheap, shares the parsed cert). Move-only; frees the SSL_CTX on destruction.
class TlsContext {
 public:
  TlsContext() noexcept = default;
  HYLE_IO_API ~TlsContext();
  HYLE_IO_API TlsContext(TlsContext&& o) noexcept;
  HYLE_IO_API TlsContext& operator=(TlsContext&& o) noexcept;
  TlsContext(const TlsContext&) = delete;
  TlsContext& operator=(const TlsContext&) = delete;

  bool valid() const noexcept { return ctx_ != nullptr; }
  ssl_ctx_st* native() const noexcept { return ctx_; }

  // Build a CLIENT context (verifies the peer against the system trust store by
  // default). Returns an invalid TlsContext on failure.
  static HYLE_IO_API TlsContext client() noexcept;
  // Build a SERVER context from a PEM certificate chain + private key file.
  // Returns an invalid TlsContext on failure (bad/missing files, key mismatch).
  static HYLE_IO_API TlsContext server(const char* cert_pem_path,
                                       const char* key_pem_path) noexcept;

  // Advertise/select HTTP/2 through TLS ALPN. With allow_http1=true the list is
  // "h2, http/1.1" (h2 preferred); false makes the context HTTP/2-only.
  HYLE_IO_API bool enable_http2(bool allow_http1 = true) noexcept;

 private:
  explicit TlsContext(ssl_ctx_st* ctx) noexcept : ctx_(ctx) {}
  ssl_ctx_st* ctx_ = nullptr;
};

// An encrypted stream over a TcpStream. Move-only; frees the SSL (which frees its
// BIOs) and closes the underlying socket on destruction. The co_await methods MUST
// be awaited from the fiber pinned to the TcpStream's owner_worker(). Each returns
// a Task (allocator-backed coroutine, HYLE's first-arg-allocator idiom) whose value
// is bytes/0/negative-errno-ish (see each method). The connection fiber does e.g.
//   TlsStream tls = ...; if ((co_await tls.handshake(alloc)) != 0) fail;
//   auto n = co_await tls.read(alloc, buf, len);
class TlsStream {
 public:
  TlsStream() noexcept = default;
  // Wraps `sock` (ownership moved in) with a new SSL from `ctx`. `is_server`
  // selects accept vs connect state. Invalid (valid()==false) on allocation
  // failure. Does NOT handshake yet -- co_await handshake() does that.
  // A verifying client MUST provide `server_name`: it is used for both SNI and
  // certificate hostname/IP matching. It may be null only for a server stream or
  // a client context explicitly configured with SSL_VERIFY_NONE.
  HYLE_IO_API TlsStream(TcpStream sock, const TlsContext& ctx, bool is_server,
                        const char* server_name = nullptr) noexcept;
  HYLE_IO_API ~TlsStream();
  HYLE_IO_API TlsStream(TlsStream&& o) noexcept;
  HYLE_IO_API TlsStream& operator=(TlsStream&& o) noexcept;
  TlsStream(const TlsStream&) = delete;
  TlsStream& operator=(const TlsStream&) = delete;

  bool valid() const noexcept { return ssl_ != nullptr; }
  unsigned owner_worker() const noexcept { return sock_.owner_worker(); }
  // Borrowed view into SSL state, valid for this stream's lifetime. Empty before
  // a successful handshake or when the peer did not negotiate ALPN.
  HYLE_IO_API container::StringView negotiated_protocol() const noexcept;

  // The co_await operations are FREE FUNCTIONS taking (mem::AllocatorRef, TlsStream&),
  // not member coroutines. A coroutine's frame is allocated by its promise's
  // operator new, which requires the AllocatorRef to be the FIRST parameter -- a
  // member coroutine's first parameter is the implicit `this`, so the AllocatorRef
  // lands second and no longer matches PromiseAllocBase::operator new. Free
  // functions keep the AllocatorRef first (HYLE's allocator-first idiom, same as
  // dns::resolve / spawn_task). See tls_handshake/read/write/shutdown below.

 private:
  friend HYLE_IO_API sched::Task<int> tls_handshake(mem::AllocatorRef,
                                                    TlsStream&) noexcept;
  friend HYLE_IO_API sched::Task<long> tls_read(mem::AllocatorRef, TlsStream&, void*,
                                               std::uint32_t) noexcept;
  friend HYLE_IO_API sched::Task<long> tls_write(mem::AllocatorRef, TlsStream&,
                                                const void*,
                                                std::uint32_t) noexcept;
  friend HYLE_IO_API sched::Task<int> tls_shutdown(mem::AllocatorRef,
                                                   TlsStream&) noexcept;

  // Lazily allocate the 16KB pump scratch on first use (reused for the stream's
  // life, freed in the dtor). Returns the block, or nullptr on OOM. Called by the
  // pump entry points before they touch flush_wbio/feed_rbio.
  HYLE_IO_API char* ensure_scratch(mem::AllocatorRef alloc) noexcept;

  TcpStream sock_{};
  ssl_st* ssl_ = nullptr;   // per-connection SSL (owns rbio_/wbio_ after SSL_set_bio)
  bio_st* rbio_ = nullptr;  // memory BIO: SSL reads peer ciphertext from here
  bio_st* wbio_ = nullptr;  // memory BIO: SSL writes ciphertext to be sent here
  // Ciphertext pump scratch (one 16KB block), lazily allocated on first pump and
  // reused for the stream's whole life. Previously each flush_wbio/feed_rbio had a
  // 16KB LOCAL array, which -- because those are coroutines -- lived in the
  // per-call coroutine FRAME (AllocatorRef-backed operator new), so every tls_read/
  // tls_write paid a 16KB frame alloc+free. Moving it here shrinks those frames to
  // tens of bytes. Safe as ONE shared block: a TlsStream is owned by one fiber and
  // flush/feed run serially (each co_await completes before the next), never nested.
  char* scratch_ = nullptr;
  mem::AllocatorRef scratch_alloc_{};  // set when scratch_ is allocated, for the free

};

// ---- co_await TLS operations (free functions, allocator-first) --------------
// Await from the fiber pinned to the stream's owner_worker(). Each runs a memory-
// BIO pump: an SSL_* call, then flush ciphertext to / feed ciphertext from the
// underlying TcpStream per SSL_get_error's WANT_READ/WANT_WRITE, until it resolves.

// co_await tls_handshake(alloc, s) -> 0 on success, negative on error.
HYLE_IO_API sched::Task<int> tls_handshake(mem::AllocatorRef alloc,
                                           TlsStream& s) noexcept;
// co_await tls_read(alloc, s, buf, len) -> plaintext bytes (>0), 0 on clean close,
// negative on error.
HYLE_IO_API sched::Task<long> tls_read(mem::AllocatorRef alloc, TlsStream& s,
                                       void* buf, std::uint32_t len) noexcept;
// co_await tls_write(alloc, s, buf, len) -> bytes written (== len) or negative.
HYLE_IO_API sched::Task<long> tls_write(mem::AllocatorRef alloc, TlsStream& s,
                                        const void* buf,
                                        std::uint32_t len) noexcept;
// co_await tls_shutdown(alloc, s) -> 0 once the close_notify exchange is done
// (best-effort on error / unresponsive peer).
HYLE_IO_API sched::Task<int> tls_shutdown(mem::AllocatorRef alloc,
                                          TlsStream& s) noexcept;

}  // namespace hyle::io

#endif  // HYLE_IO_TLS

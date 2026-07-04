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
#include "hyle/macros.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"

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
  HYLE_DISABLE_COPY(TlsContext);

  bool valid() const noexcept { return ctx_ != nullptr; }
  ssl_ctx_st* native() const noexcept { return ctx_; }

  // Build a CLIENT context (verifies the peer against the system trust store by
  // default). Returns an invalid TlsContext on failure.
  static HYLE_IO_API TlsContext client() noexcept;
  // Build a SERVER context from a PEM certificate chain + private key file.
  // Returns an invalid TlsContext on failure (bad/missing files, key mismatch).
  static HYLE_IO_API TlsContext server(const char* cert_pem_path,
                                       const char* key_pem_path) noexcept;

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
  HYLE_IO_API TlsStream(TcpStream sock, const TlsContext& ctx,
                        bool is_server) noexcept;
  HYLE_IO_API ~TlsStream();
  HYLE_IO_API TlsStream(TlsStream&& o) noexcept;
  HYLE_IO_API TlsStream& operator=(TlsStream&& o) noexcept;
  HYLE_DISABLE_COPY(TlsStream);

  bool valid() const noexcept { return ssl_ != nullptr; }
  unsigned owner_worker() const noexcept { return sock_.owner_worker(); }

  // co_await tls.handshake(alloc) -> 0 on success, negative on error (bad handshake
  // / peer closed / verify failure). Runs the TLS handshake pump: SSL_do_handshake,
  // flushing/feeding ciphertext via the underlying TcpStream until established.
  HYLE_IO_API sched::Task<int> handshake(mem::Allocator alloc) noexcept;

  // co_await tls.read(alloc, buf, len) -> plaintext bytes read (>0), 0 on clean
  // peer close (close_notify / EOF), or negative on error. Decrypts via SSL_read,
  // pulling ciphertext from the socket as the pump needs it.
  HYLE_IO_API sched::Task<long> read(mem::Allocator alloc, void* buf,
                                     std::uint32_t len) noexcept;

  // co_await tls.write(alloc, buf, len) -> plaintext bytes written (all of `len` on
  // success) or negative on error. Encrypts via SSL_write, pushing ciphertext to
  // the socket. Handles partial writes internally.
  HYLE_IO_API sched::Task<long> write(mem::Allocator alloc, const void* buf,
                                      std::uint32_t len) noexcept;

  // co_await tls.shutdown(alloc) -> 0 once the TLS close_notify handshake is done
  // (or best-effort on error). Sends our close_notify; a full duplex close also
  // waits for the peer's. The socket is then closed by the destructor.
  HYLE_IO_API sched::Task<int> shutdown(mem::Allocator alloc) noexcept;

 private:
  TcpStream sock_{};
  ssl_st* ssl_ = nullptr;   // per-connection SSL (owns rbio_/wbio_ after SSL_set_bio)
  bio_st* rbio_ = nullptr;  // memory BIO: SSL reads peer ciphertext from here
  bio_st* wbio_ = nullptr;  // memory BIO: SSL writes ciphertext to be sent here

  // Internal pump helpers (defined in tls.cc). flush_wbio: drain wbio -> co_await
  // sock_.write. feed_rbio: co_await sock_.read -> BIO_write rbio_ (false = EOF).
  sched::Task<int> flush_wbio(mem::Allocator alloc) noexcept;
  sched::Task<int> feed_rbio(mem::Allocator alloc) noexcept;
};

}  // namespace hyle::io

#endif  // HYLE_IO_TLS

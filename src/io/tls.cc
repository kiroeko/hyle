#include "hyle/io/tls.h"

#if defined(HYLE_IO_TLS)

#include <openssl/bio.h>
#include <openssl/err.h>
#include <openssl/ssl.h>

#include <utility>

// io-08 TLS implementation. memory-BIO pump over the co_await TcpStream. Reference:
// seastar src/net/tls_openssl.cc (handshake/read/write/shutdown state machines) +
// libevent bufferevent_openssl.c (SSL_MODE_ACCEPT_MOVING_WRITE_BUFFER, BIO
// ownership, dirty-shutdown). See tls.h for the model + threading contract.

namespace hyle::io {
namespace {

constexpr int kBufSize = 16 * 1024;  // ciphertext pump chunk

// Map the current SSL op's non-positive return to what the pump must do next. We
// call SSL_get_error IMMEDIATELY after the SSL_* call (nothing between), per
// OpenSSL's per-thread-error-queue contract -- a co_await must never split this.
enum class Want { kDone, kReadMore, kWriteMore, kError, kEof };

Want classify(SSL* ssl, int ret) noexcept {
  int e = SSL_get_error(ssl, ret);
  switch (e) {
    case SSL_ERROR_NONE:
      return Want::kDone;
    case SSL_ERROR_WANT_READ:
      return Want::kReadMore;
    case SSL_ERROR_WANT_WRITE:
      return Want::kWriteMore;
    case SSL_ERROR_ZERO_RETURN:
      return Want::kEof;  // peer sent close_notify: clean TLS EOF
    case SSL_ERROR_SYSCALL:
      // OpenSSL 3.0.x: a clean transport EOF with no error on the queue shows up
      // here with errno 0 -- treat as EOF, not error (seastar tls_openssl notes
      // this explicitly). A real syscall error would have queued something.
      return (ERR_peek_error() == 0) ? Want::kEof : Want::kError;
    default:
      return Want::kError;  // SSL_ERROR_SSL etc.
  }
}

}  // namespace

// ---- TlsContext ------------------------------------------------------------

TlsContext::~TlsContext() {
  if (ctx_ != nullptr) SSL_CTX_free(ctx_);
}
TlsContext::TlsContext(TlsContext&& o) noexcept
    : ctx_(std::exchange(o.ctx_, nullptr)) {}
TlsContext& TlsContext::operator=(TlsContext&& o) noexcept {
  if (this != &o) {
    if (ctx_ != nullptr) SSL_CTX_free(ctx_);
    ctx_ = std::exchange(o.ctx_, nullptr);
  }
  return *this;
}

TlsContext TlsContext::client() noexcept {
  SSL_CTX* ctx = SSL_CTX_new(TLS_client_method());
  if (ctx == nullptr) return TlsContext{};
  SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION);
  // Verify the peer against the default trust store. A caller that wants to skip
  // verification (self-signed test certs) can adjust the returned ctx's native().
  SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, nullptr);
  SSL_CTX_set_default_verify_paths(ctx);
  return TlsContext{ctx};
}

TlsContext TlsContext::server(const char* cert_pem_path,
                              const char* key_pem_path) noexcept {
  SSL_CTX* ctx = SSL_CTX_new(TLS_server_method());
  if (ctx == nullptr) return TlsContext{};
  SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION);
  if (SSL_CTX_use_certificate_chain_file(ctx, cert_pem_path) != 1 ||
      SSL_CTX_use_PrivateKey_file(ctx, key_pem_path, SSL_FILETYPE_PEM) != 1 ||
      SSL_CTX_check_private_key(ctx) != 1) {
    SSL_CTX_free(ctx);
    return TlsContext{};
  }
  return TlsContext{ctx};
}

// ---- TlsStream lifetime ----------------------------------------------------

TlsStream::TlsStream(TcpStream sock, const TlsContext& ctx,
                     bool is_server) noexcept
    : sock_(std::move(sock)) {
  if (!ctx.valid()) return;
  ssl_ = SSL_new(ctx.native());
  if (ssl_ == nullptr) return;
  rbio_ = BIO_new(BIO_s_mem());
  wbio_ = BIO_new(BIO_s_mem());
  if (rbio_ == nullptr || wbio_ == nullptr) {
    if (rbio_ != nullptr) BIO_free(rbio_);
    if (wbio_ != nullptr) BIO_free(wbio_);
    SSL_free(ssl_);
    ssl_ = nullptr;
    rbio_ = wbio_ = nullptr;
    return;
  }
  // SSL_set_bio TAKES OWNERSHIP of both BIOs -- SSL_free frees them, so we must NOT
  // BIO_free them again (double free). After this, rbio_/wbio_ are observed-only.
  SSL_set_bio(ssl_, rbio_, wbio_);
  // Partial SSL_write must be retryable with a possibly-moved buffer pointer, or it
  // asserts. Mandatory for the write pump (libevent sets this too).
  SSL_set_mode(ssl_, SSL_MODE_ACCEPT_MOVING_WRITE_BUFFER);
  if (is_server) {
    SSL_set_accept_state(ssl_);
  } else {
    SSL_set_connect_state(ssl_);
  }
}

TlsStream::~TlsStream() {
  if (ssl_ != nullptr) SSL_free(ssl_);  // also frees rbio_ + wbio_ (SSL_set_bio owns)
}

TlsStream::TlsStream(TlsStream&& o) noexcept
    : sock_(std::move(o.sock_)),
      ssl_(std::exchange(o.ssl_, nullptr)),
      rbio_(std::exchange(o.rbio_, nullptr)),
      wbio_(std::exchange(o.wbio_, nullptr)) {}

TlsStream& TlsStream::operator=(TlsStream&& o) noexcept {
  if (this != &o) {
    if (ssl_ != nullptr) SSL_free(ssl_);
    sock_ = std::move(o.sock_);
    ssl_ = std::exchange(o.ssl_, nullptr);
    rbio_ = std::exchange(o.rbio_, nullptr);
    wbio_ = std::exchange(o.wbio_, nullptr);
  }
  return *this;
}

// ---- pump helpers (allocator-first free coroutines; take the raw socket + BIO so
// they need no access to TlsStream privates) --------------------------------

// Drain all pending ciphertext from `wbio` and co_await-write it to the peer via
// `sock`. Returns 0 on success, negative on a socket write error.
static sched::Task<int> flush_wbio(mem::Allocator, TcpStream& sock,
                                   bio_st* wbio) noexcept {
  char buf[kBufSize];
  for (;;) {
    int n = BIO_read(wbio, buf, kBufSize);  // pull ciphertext SSL queued for us
    if (n <= 0) co_return 0;                // nothing (more) to send
    std::uint32_t off = 0;
    while (off < static_cast<std::uint32_t>(n)) {
      std::int64_t w = co_await sock.write(buf + off,
                                           static_cast<std::uint32_t>(n) - off);
      if (w <= 0) co_return -1;  // socket write failed
      off += static_cast<std::uint32_t>(w);
    }
  }
}

// co_await-read a chunk of ciphertext from the peer and feed it into `rbio` so SSL
// can consume it. Returns 1 on data fed, 0 on peer EOF, negative on error.
static sched::Task<int> feed_rbio(mem::Allocator, TcpStream& sock,
                                  bio_st* rbio) noexcept {
  char buf[kBufSize];
  std::int64_t n = co_await sock.read(buf, kBufSize);
  if (n == 0) co_return 0;   // peer closed the transport
  if (n < 0) co_return -1;   // socket read error
  BIO_write(rbio, buf, static_cast<int>(n));  // memory BIO write never blocks
  co_return 1;
}

// ---- handshake -------------------------------------------------------------

sched::Task<int> tls_handshake(mem::Allocator alloc, TlsStream& s) noexcept {
  if (s.ssl_ == nullptr) co_return -1;
  for (;;) {
    ERR_clear_error();
    int r = SSL_do_handshake(s.ssl_);
    Want w = classify(s.ssl_, r);     // MUST be right after SSL_do_handshake
    // Always flush any ciphertext the handshake produced (our ClientHello/Finished
    // etc.) before waiting to read the peer's -- the handshake is a ping-pong.
    if ((co_await flush_wbio(alloc, s.sock_, s.wbio_)) != 0) co_return -1;
    if (w == Want::kDone) co_return 0;         // SSL_is_init_finished
    if (w == Want::kError || w == Want::kEof) co_return -1;
    if (w == Want::kReadMore) {
      int f = co_await feed_rbio(alloc, s.sock_, s.rbio_);
      if (f <= 0) co_return -1;  // EOF/err mid-handshake
    }
    // kWriteMore: the flush above already drained wbio_; loop and retry.
  }
}

// ---- read ------------------------------------------------------------------

sched::Task<long> tls_read(mem::Allocator alloc, TlsStream& s, void* buf,
                           std::uint32_t len) noexcept {
  if (s.ssl_ == nullptr) co_return -1;
  for (;;) {
    ERR_clear_error();
    int r = SSL_read(s.ssl_, buf, static_cast<int>(len));
    if (r > 0) co_return static_cast<long>(r);  // plaintext bytes
    Want w = classify(s.ssl_, r);               // right after SSL_read
    // A read may still need to WRITE (renegotiation) -- flush first either way.
    if ((co_await flush_wbio(alloc, s.sock_, s.wbio_)) != 0) co_return -1;
    if (w == Want::kEof) co_return 0;           // clean close_notify / transport EOF
    if (w == Want::kError) co_return -1;
    if (w == Want::kReadMore) {
      int f = co_await feed_rbio(alloc, s.sock_, s.rbio_);
      if (f == 0) co_return 0;   // peer closed while we needed more ciphertext
      if (f < 0) co_return -1;
    }
    // kWriteMore: flushed above; loop and retry SSL_read.
  }
}

// ---- write -----------------------------------------------------------------

sched::Task<long> tls_write(mem::Allocator alloc, TlsStream& s, const void* buf,
                            std::uint32_t len) noexcept {
  if (s.ssl_ == nullptr) co_return -1;
  const char* p = static_cast<const char*>(buf);
  std::uint32_t remaining = len;
  while (remaining > 0) {
    ERR_clear_error();
    int r = SSL_write(s.ssl_, p, static_cast<int>(remaining));
    if (r > 0) {
      // Ciphertext for these bytes is now in wbio_; send it before continuing.
      if ((co_await flush_wbio(alloc, s.sock_, s.wbio_)) != 0) co_return -1;
      p += r;
      remaining -= static_cast<std::uint32_t>(r);
      continue;
    }
    Want w = classify(s.ssl_, r);  // right after SSL_write
    if ((co_await flush_wbio(alloc, s.sock_, s.wbio_)) != 0) co_return -1;
    if (w == Want::kError || w == Want::kEof) co_return -1;
    if (w == Want::kReadMore) {  // renegotiation: write needs to read
      int f = co_await feed_rbio(alloc, s.sock_, s.rbio_);
      if (f <= 0) co_return -1;
    }
    // kWriteMore: flushed above; loop and retry SSL_write.
  }
  co_return static_cast<long>(len);
}

// ---- shutdown --------------------------------------------------------------

sched::Task<int> tls_shutdown(mem::Allocator alloc, TlsStream& s) noexcept {
  if (s.ssl_ == nullptr) co_return 0;
  // Best-effort duplex close: send our close_notify, then flush. SSL_shutdown()==1
  // means the peer's close_notify was also received (full duplex done); ==0 means
  // ours is sent and we could wait for the peer's -- we do a bounded number of read
  // attempts but don't hang forever on a peer that never replies (dirty shutdown is
  // acceptable at teardown). WANT_* pumps the BIOs.
  for (int attempt = 0; attempt < 4; ++attempt) {
    ERR_clear_error();
    int r = SSL_shutdown(s.ssl_);
    if ((co_await flush_wbio(alloc, s.sock_, s.wbio_)) != 0) co_return -1;
    if (r == 1) co_return 0;  // full duplex close_notify exchange complete
    Want w = classify(s.ssl_, r);
    if (w == Want::kError || w == Want::kEof) co_return 0;  // give up cleanly
    if (w == Want::kReadMore) {
      int f = co_await feed_rbio(alloc, s.sock_, s.rbio_);
      if (f <= 0) co_return 0;  // peer gone -> dirty shutdown, acceptable
    }
  }
  co_return 0;
}

}  // namespace hyle::io

#endif  // HYLE_IO_TLS

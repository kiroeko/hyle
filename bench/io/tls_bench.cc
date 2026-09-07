// io-08 TLS benchmark: the honest TLS perf comparison the io-08 milestone deferred.
// TLS throughput/handshake over HYLE's memory-BIO pump (co_await TcpStream) has TWO
// cost components: (1) OpenSSL's crypto (handshake asymmetric + record symmetric) --
// UNAVOIDABLE, identical for any OpenSSL user; (2) HYLE's pump + fiber scheduling
// overhead ON TOP. A single throughput number can't separate them, so the SOTA
// baseline here is RAW OpenSSL over a blocking socket (same cert, same cipher, same
// payload, NO HYLE) -- the ratio is exactly what HYLE's co_await abstraction costs
// over bare OpenSSL. Both pay the identical crypto floor, so the delta is pure HYLE.
//
// Three measurements:
//   1. HYLE TLS handshake/s -- full connect + handshake + close round-trip (the
//      connection-setup cost, dominated by the asymmetric key exchange).
//   2. HYLE TLS echo throughput -- post-handshake N encrypted round-trips (steady-
//      state record read/write pump).
//   3. raw OpenSSL (blocking socket, no HYLE) -- same handshake + echo, the crypto
//      floor + a plain blocking-socket loop. HYLE/this ratio = the pump overhead.
//
// Run under scripts/bench-stat.py -n 15 for the statistical verdict.

#include <nanobench.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

#include <openssl/ssl.h>
#include <openssl/bio.h>
#include <openssl/err.h>

#include "hyle/io/io_context.h"
#include "hyle/io/tcp.h"
#include "hyle/io/tls.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

#if !defined(_WIN32)
#  include <sys/socket.h>
#  include <netinet/in.h>
#  include <arpa/inet.h>
#  include <unistd.h>
#  include <csignal>  // ignore SIGPIPE: a write to a peer-closed socket must not kill us
#endif
#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <winsock2.h>
#  include <ws2tcpip.h>
#endif

using hyle::io::IoContext;
using hyle::io::resolve_addr;
using hyle::io::TcpListener;
using hyle::io::TcpStream;
using hyle::io::tcp_listen;
using hyle::io::tcp_socket;
using hyle::io::TlsContext;
using hyle::io::TlsStream;
using hyle::io::tls_handshake;
using hyle::io::tls_read;
using hyle::io::tls_write;
using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::Task;

namespace {

constexpr unsigned kOwner = 0;
constexpr int kMsgLen = 256;       // TLS record payload
constexpr int kEchoRoundTrips = 64;  // encrypted round-trips per connection per epoch

// Self-signed cert + key (CN=localhost, RSA 2048), same as tls_test.cc.
const char* kCertPem =
    "-----BEGIN CERTIFICATE-----\n"
    "MIIDCTCCAfGgAwIBAgIUN8o8u2IFjiNjqkZO2bW2LU5tXXowDQYJKoZIhvcNAQEL\n"
    "BQAwFDESMBAGA1UEAwwJbG9jYWxob3N0MB4XDTI2MDcwNDA3MTA1M1oXDTM2MDcw\n"
    "MTA3MTA1M1owFDESMBAGA1UEAwwJbG9jYWxob3N0MIIBIjANBgkqhkiG9w0BAQEF\n"
    "AAOCAQ8AMIIBCgKCAQEAqdKC1htqD9qXhLF2wDBn565oY4rs6rP2iCyTDfO5ePP/\n"
    "sLOCzDkiVza1MjhwSbuK6zDRbrY2joX3Ir6KLbzoV/5VaVckBocPTcNxQa42DmIH\n"
    "7xZoDRu9pstp8zdpy3UfAruPRyNWqHqVnw09DQETLbwh3blhT1PaMAOMIBJGCOnd\n"
    "/NsBqiVOsYAkNVgcbuqV8h9b3b6FMMoPPRCVnfn5XROLk5Bh47zT+Q1WAsqyARDe\n"
    "WZ5q8TKQzHuFSursB8rB9kYZLHXmGnnTe2dsiypqqWyCOWryxchEL9mKEk2Q2UUc\n"
    "7VDtvOxgZKu4GJ1r9S4UYm4nB4OPqdIJ96EE+QcIYwIDAQABo1MwUTAdBgNVHQ4E\n"
    "FgQUZIsMaR+5CUbVAhXUhUBDh1+ubNIwHwYDVR0jBBgwFoAUZIsMaR+5CUbVAhXU\n"
    "hUBDh1+ubNIwDwYDVR0TAQH/BAUwAwEB/zANBgkqhkiG9w0BAQsFAAOCAQEAFvOf\n"
    "AcST8Bmpf26HQCnLYCjZfrvw7XIsoJ08KT9JVx8PXXnXsfKB/n8ZlfU7GMviMIYI\n"
    "c27FLOzb8hA2Ifw8wcRW3sC9rEReblI8+bwPchcexB3jyAdf13zcN4GFBrUy3j91\n"
    "1fdlNBllsgizWy7fF5ARfClcDNM/q4t/KcYfSpwcWebXJc6B7IdfDGLcMJ5NzR4C\n"
    "DzAn6Ks4CLxOXJVq2A/h14uUlyS9lPMps2zQ2ggeEg4tOKgln0BRBkiUBKwgBHAo\n"
    "i4W6kCNLXKk2OG5WP/A+F3FukS9r5sbGgdQC5ANxndDwfSUuYcEYDRn+P7mdEbBZ\n"
    "+9hlP09t1qimUNJa/g==\n"
    "-----END CERTIFICATE-----\n";
const char* kKeyPem =
    "-----BEGIN PRIVATE KEY-----\n"
    "MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCp0oLWG2oP2peE\n"
    "sXbAMGfnrmhjiuzqs/aILJMN87l48/+ws4LMOSJXNrUyOHBJu4rrMNFutjaOhfci\n"
    "vootvOhX/lVpVyQGhw9Nw3FBrjYOYgfvFmgNG72my2nzN2nLdR8Cu49HI1aoepWf\n"
    "DT0NARMtvCHduWFPU9owA4wgEkYI6d382wGqJU6xgCQ1WBxu6pXyH1vdvoUwyg89\n"
    "EJWd+fldE4uTkGHjvNP5DVYCyrIBEN5ZnmrxMpDMe4VK6uwHysH2RhksdeYaedN7\n"
    "Z2yLKmqpbII5avLFyEQv2YoSTZDZRRztUO287GBkq7gYnWv1LhRibicHg4+p0gn3\n"
    "oQT5BwhjAgMBAAECggEAAIkdvFLwaTKW/4ykI4pwV9s4qhCCNPuyhdugiE7VLjjH\n"
    "lpH7+IZSzoEC9+Phesh0JJzBTfH8k90ZTYtAGh46T18jSGzfCLdvu9H0SMqlxTLX\n"
    "wPaXOmwn4EXlxIyCsWkVQXSmlYj0TFC8QxJ1t4iu1YCX5lnTOqUfKL9ohDtg7CtE\n"
    "8dXYQ6tC6vyc8YdFpt9NoEcBxmy3voIZw37z1f1bVQAVIo47vq1K/a0rHhqXtJPc\n"
    "bjbpd1EWbCkR/1Azlw4AIIScJLHHsIvpUlP3xg30qNKkxp5LIg8dfA3vihmEArQ/\n"
    "dJPv1xn6ySkoKFRooPOwQRtqcrPeVUVSI3Pd8sCR/QKBgQDWXfiOe29tRkydBvL4\n"
    "Ktlh5u9PPxwTrmTmGwoh50H5XBkL/3+Oz5cNGVSGdzES/3+oZwFkZgshM/2Fvo83\n"
    "uYui/76G8HhK8B7duEH4kGieyki5kVInq7gBUbpCsZrhbv3b1NndXcnJUUkkM+5f\n"
    "3jvzCHTaaxy3Fgb1RfqNES/DfwKBgQDKzdZQHZZQk8k5+IAqZ0vpI0w4606iha8C\n"
    "vk4fUapZejXnMcRobiP1etVtfVg3YTjFZb3+dSp7zCWe9v7P2tRCm/LfwnbAiRmb\n"
    "LavUTtDD2RE1Mnnz6FbtE5nGZ7BxdxYiuF3Gof1zW0ooCSNFW4m3VymIaHzY5nV9\n"
    "AmTzPMedHQKBgFk80UXvGJmy1B3iUK8Cl2/sBN9GdFe1F4e65eLrNEmsDzkxaVZ4\n"
    "EHeBffBkGefAuNHP34J+AFAYJofQpioacYivUg82d70qbI78L2pcI1FHADNPzZV6\n"
    "JKVX8mPlL4Z8/+zNlMxUSLYnwnzxqV/7UZWAhbSPNmr5HnhoYz7qUqndAoGBAL/q\n"
    "Hb7e4NoLoEcNlCuMaNeMoZrTgixmGYGdGUyl+XCHJ+9KtzSj1b99p+7eyNZPWGvk\n"
    "pfDga32PaKKP1f5YdLk73pBE9B+rHeItTAezNfe/uHsA80sPOQNMox6QDPay3O2q\n"
    "ToOj07aQUxKj4w7FWwN2pkUBq/ON0uyjkWhkwfk5AoGBALKE/2Nc8VAIlhcnqZDJ\n"
    "SRROsPLt1I56SykkMB9H3OhH3jW2xaM043IcqyS8GAXAXBS4ltZ3vYzyy+dkoKHb\n"
    "1/HH0oyQh5LdX6vLMtXxoZI8GPnmtbeJ4fmXckbRXev925vXgdcknQ2RY6OU9HzI\n"
    "A48heHgTiFQ6QIIrkn9z7lQJ\n"
    "-----END PRIVATE KEY-----\n";

std::string write_temp(const char* name, const char* pem) {
  std::string path = std::string("hyle_tls_bench_") + name;
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (f) { std::fwrite(pem, 1, std::strlen(pem), f); std::fclose(f); }
  return path;
}

// ---- HYLE side --------------------------------------------------------------
struct HyleTls {
  std::uint16_t port;
  bool do_echo;          // false = handshake-only (then close); true = handshake + N echoes
  TlsContext* sctx;
  TlsContext* cctx;
  std::atomic<int> client_units{0};
};

Task<int> hyle_server(AllocatorRef alloc, TcpListener* ls, HyleTls* h) {
  TcpStream conn = co_await ls->accept(kOwner);
  if (!conn.valid()) co_return 0;
  TlsStream tls(std::move(conn), *h->sctx, /*is_server=*/true);
  if (co_await tls_handshake(alloc, tls) != 0) co_return 0;
  if (!h->do_echo) h->client_units.store(1, std::memory_order_relaxed);
  if (h->do_echo) {
    char buf[kMsgLen];
    for (int i = 0; i < kEchoRoundTrips; ++i) {
      long n = co_await tls_read(alloc, tls, buf, sizeof(buf));
      if (n <= 0) break;
      if (co_await tls_write(alloc, tls, buf, static_cast<std::uint32_t>(n)) != n) break;
    }
  }
  co_return 0;
}

Task<int> hyle_client(AllocatorRef alloc, IoContext* io, HyleTls* h) {
  TcpStream s = tcp_socket(*io, kOwner);
  alignas(16) unsigned char ss[128];
  std::uint32_t len = 0;
  resolve_addr("127.0.0.1", h->port, &ss, &len);
  if (co_await s.connect(&ss, len) != 0) co_return 0;
  TlsStream tls(std::move(s), *h->cctx, /*is_server=*/false);
  if (co_await tls_handshake(alloc, tls) != 0) co_return 0;
  if (h->do_echo) {
    static const char kMsg[kMsgLen] = "hyle-tls-bench-payload";
    char buf[kMsgLen];
    for (int i = 0; i < kEchoRoundTrips; ++i) {
      if (co_await tls_write(alloc, tls, kMsg, kMsgLen) != kMsgLen) break;
      if (co_await tls_read(alloc, tls, buf, kMsgLen) <= 0) break;
      h->client_units.fetch_add(1, std::memory_order_relaxed);
    }
  }
  co_return 0;
}

struct Drive { Task<int> task; };
void drive_entry(void* p) noexcept { static_cast<Drive*>(p)->task.resume(); }

// One HYLE TLS connection: handshake (+ optional N echoes), both fibers pinned to
// one worker. `port` is bumped per call to dodge TIME_WAIT.
int run_hyle_tls(std::uint16_t port, bool do_echo, TlsContext* sctx,
                 TlsContext* cctx) {
  Scheduler sched(1, system_allocator(), 1024);
  IoContext io(sched, 1);
  TcpListener ls = tcp_listen(io, kOwner, "127.0.0.1", port);
  if (!ls.valid()) return -1;
  HyleTls h{port, do_echo, sctx, cctx};
  auto ds = std::make_unique<Drive>();
  auto dc = std::make_unique<Drive>();
  ds->task = hyle_server(system_allocator(), &ls, &h);
  dc->task = hyle_client(system_allocator(), &io, &h);
  Fiber fs(kDefaultStackSize, &drive_entry, ds.get());
  Fiber fc(kDefaultStackSize, &drive_entry, dc.get());
  sched.spawn_on(kOwner, &fs);
  sched.spawn_on(kOwner, &fc);
  sched.drain();
  sched.shutdown();
  return h.client_units.load(std::memory_order_relaxed);
}

// ---- raw OpenSSL over a blocking socket (SOTA baseline, no HYLE) ------------
// Same self-signed cert, same cipher defaults, same payload. Server runs on a
// std::thread; client on the main thread. Plain blocking SSL_read/SSL_write over a
// connected loopback TCP socket -- the crypto floor + a bare blocking loop. HYLE's
// number vs this = the co_await pump + fiber scheduling overhead over bare OpenSSL.
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
int make_listener(std::uint16_t port) {
  int ls = static_cast<int>(::socket(AF_INET, SOCK_STREAM, 0));
  int one = 1;
  ::setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&one), sizeof(one));
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  sa.sin_port = htons(port);
  if (::bind(ls, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) != 0 ||
      ::listen(ls, 8) != 0) {
#if defined(_WIN32)
    ::closesocket(ls);
#else
    ::close(ls);
#endif
    return -1;
  }
  return ls;
}

void raw_openssl_server(int ls, SSL_CTX* sctx, bool do_echo) {
  int c = static_cast<int>(::accept(ls, nullptr, nullptr));
  if (c < 0) return;
  SSL* ssl = SSL_new(sctx);
  SSL_set_fd(ssl, c);
  if (SSL_accept(ssl) == 1 && do_echo) {
    char buf[kMsgLen];
    for (int i = 0; i < kEchoRoundTrips; ++i) {
      int n = SSL_read(ssl, buf, kMsgLen);
      if (n <= 0) break;
      if (SSL_write(ssl, buf, n) != n) break;
    }
  }
  SSL_shutdown(ssl);
  SSL_free(ssl);
#if defined(_WIN32)
  ::closesocket(c);
#else
  ::close(c);
#endif
}

int run_raw_openssl_tls(std::uint16_t port, bool do_echo, SSL_CTX* sctx,
                        SSL_CTX* cctx) {
  int ls = make_listener(port);
  if (ls < 0) return -1;
  std::thread srv(raw_openssl_server, ls, sctx, do_echo);

  int c = static_cast<int>(::socket(AF_INET, SOCK_STREAM, 0));
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  sa.sin_port = htons(port);
  int units = 0;
  if (::connect(c, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) == 0) {
    SSL* ssl = SSL_new(cctx);
    SSL_set_fd(ssl, c);
    if (SSL_connect(ssl) == 1) {
      if (!do_echo) units = 1;
    }
    if (SSL_is_init_finished(ssl) && do_echo) {
      static const char kMsg[kMsgLen] = "hyle-tls-bench-payload";
      char buf[kMsgLen];
      for (int i = 0; i < kEchoRoundTrips; ++i) {
        if (SSL_write(ssl, kMsg, kMsgLen) != kMsgLen) break;
        if (SSL_read(ssl, buf, kMsgLen) <= 0) break;
        ++units;
      }
    }
    SSL_shutdown(ssl);
    SSL_free(ssl);
  }
#if defined(_WIN32)
  ::closesocket(c);
#else
  ::close(c);
#endif
  srv.join();
#if defined(_WIN32)
  ::closesocket(ls);
#else
  ::close(ls);
#endif
  return units;
}
#endif  // POSIX || Windows

}  // namespace

int main() {
#if !defined(_WIN32)
  // A blocking SSL_write to a socket the peer already closed raises SIGPIPE, which
  // is fatal by default -- ignore it so the raw-OpenSSL baseline's teardown races
  // don't kill the bench (the write just returns EPIPE, which we don't act on here).
  std::signal(SIGPIPE, SIG_IGN);
#endif
#if defined(_WIN32)
  WSADATA wsa;
  ::WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
  std::string cert = write_temp("cert.pem", kCertPem);
  std::string key = write_temp("key.pem", kKeyPem);

  // HYLE contexts (reused across iterations -- SSL_CTX is shareable; per-conn SSL is
  // made from it, matching how a real server holds one CTX).
  TlsContext hsctx = TlsContext::server(cert.c_str(), key.c_str());
  TlsContext hcctx = TlsContext::client();
  if (hsctx.valid() && hcctx.valid())
    SSL_CTX_set_verify(hcctx.native(), SSL_VERIFY_NONE, nullptr);

  // Raw OpenSSL contexts (same cert/key/verify).
  SSL_CTX* rsctx = SSL_CTX_new(TLS_server_method());
  SSL_CTX_use_certificate_chain_file(rsctx, cert.c_str());
  SSL_CTX_use_PrivateKey_file(rsctx, key.c_str(), SSL_FILETYPE_PEM);
  SSL_CTX* rcctx = SSL_CTX_new(TLS_client_method());
  SSL_CTX_set_verify(rcctx, SSL_VERIFY_NONE, nullptr);

  // Stay below the OS ephemeral-client ranges (Linux commonly starts at 32768,
  // Windows at 49152). The old 54460 base could be selected as the previous
  // connection's client port, so the next sequential server bind intermittently
  // failed with EADDRINUSE and fabricated a benchmark abort.
  std::uint16_t port = 18460;

  // ---- 1. handshake/s: full connect + TLS handshake + close --------------------
  {
    ankerl::nanobench::Bench b;
    b.title("TLS handshake (connect + handshake + close)")
        .relative(true)
        .unit("handshake")
        .minEpochIterations(1)
        .epochs(5);
    b.run("HYLE TLS (memory-BIO pump over co_await)", [&] {
      if (run_hyle_tls(port++, /*do_echo=*/false, &hsctx, &hcctx) != 1)
        std::abort();
    });
    b.run("raw OpenSSL (blocking socket, no HYLE)", [&] {
      if (run_raw_openssl_tls(port++, /*do_echo=*/false, rsctx, rcctx) != 1)
        std::abort();
    });
  }

  // ---- 2. encrypted echo throughput: handshake + N record round-trips ----------
  {
    ankerl::nanobench::Bench b;
    b.title("TLS encrypted echo (64 round-trips x 256B, post-handshake)")
        .relative(true)
        .batch(kEchoRoundTrips)
        .unit("round-trip")
        .minEpochIterations(1)
        .epochs(5);
    b.run("HYLE TLS (memory-BIO pump over co_await)", [&] {
      if (run_hyle_tls(port++, /*do_echo=*/true, &hsctx, &hcctx) !=
          kEchoRoundTrips)
        std::abort();
    });
    b.run("raw OpenSSL (blocking socket, no HYLE)", [&] {
      if (run_raw_openssl_tls(port++, /*do_echo=*/true, rsctx, rcctx) !=
          kEchoRoundTrips)
        std::abort();
    });
  }

  SSL_CTX_free(rsctx);
  SSL_CTX_free(rcctx);
#if defined(_WIN32)
  ::WSACleanup();
#endif
  return 0;
}

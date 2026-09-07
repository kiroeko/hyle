#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

#include <openssl/ssl.h>  // SSL_CTX_set_verify (client skips verify for self-signed)

#include "hyle/io/http.h"
#if defined(HYLE_IO_HTTP2)
#  include "hyle/io/http2.h"
#endif
#include "hyle/io/io_context.h"
#include "hyle/io/tcp.h"
#include "hyle/io/tls.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

// io-08 TLS test: a real TLS 1.2+ handshake + encrypted echo over loopback TCP,
// using the memory-BIO pump (TlsContext + TlsStream + co_await tls_handshake/read/
// write). A self-signed cert (written to temp files) backs the server; the client
// skips verification (self-signed). Proves the whole pump: handshake ping-pong,
// SSL_write ciphertext -> socket, socket -> rbio -> SSL_read plaintext.

using hyle::io::IoContext;
using hyle::io::HttpClientConn;
#if defined(HYLE_IO_HTTP2)
using hyle::io::Http2ClientConn;
using hyle::io::Http2ServerConn;
#endif
using hyle::io::HttpResponse;
using hyle::io::HttpServerConn;
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
constexpr std::uint16_t kPort = 54395;

// A self-signed cert + key (CN=localhost, RSA 2048, 10y) for the test server.
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
  std::string path = std::string("hyle_tls_test_") + name;
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (f) { std::fwrite(pem, 1, std::strlen(pem), f); std::fclose(f); }
  return path;
}

struct TlsResult {
  std::atomic<bool> server_done{false};
  std::atomic<bool> client_done{false};
  std::atomic<int> server_hs{-999};
  std::atomic<int> client_hs{-999};
  std::atomic<long> server_echoed{0};
  std::atomic<long> client_recv{0};
  std::atomic<bool> server_h2{false};
  std::atomic<bool> client_h2{false};
  char client_got[64] = {};
};

// Server: accept, wrap in server TlsStream, handshake, read plaintext, echo it back.
Task<int> tls_server(AllocatorRef alloc, TcpListener* ls, TlsContext* sctx,
                     TlsResult* r) {
  TcpStream conn = co_await ls->accept(kOwner);
  if (!conn.valid()) { r->server_done.store(true); co_return 0; }
  TlsStream tls(std::move(conn), *sctx, /*is_server=*/true);
  int hs = co_await tls_handshake(alloc, tls);
  r->server_hs.store(hs, std::memory_order_release);
  r->server_h2.store(tls.negotiated_protocol() == "h2",
                     std::memory_order_release);
  if (hs == 0) {
    char buf[64] = {};
    long n = co_await tls_read(alloc, tls, buf, sizeof(buf));
    if (n > 0) {
      long w = co_await tls_write(alloc, tls, buf, static_cast<std::uint32_t>(n));
      r->server_echoed.store(w, std::memory_order_release);
    }
  }
  r->server_done.store(true, std::memory_order_release);
  co_return 0;
}

// Client: connect, wrap in client TlsStream, handshake, send, read echo, verify.
Task<int> tls_client(AllocatorRef alloc, IoContext* io, TlsContext* cctx,
                     TlsResult* r) {
  TcpStream s = tcp_socket(*io, kOwner);
  alignas(16) unsigned char ss[128];
  std::uint32_t len = 0;
  resolve_addr("127.0.0.1", kPort, &ss, &len);
  if (co_await s.connect(&ss, len) != 0) { r->client_done.store(true); co_return 0; }
  TlsStream tls(std::move(s), *cctx, /*is_server=*/false);
  int hs = co_await tls_handshake(alloc, tls);
  r->client_hs.store(hs, std::memory_order_release);
  r->client_h2.store(tls.negotiated_protocol() == "h2",
                     std::memory_order_release);
  if (hs == 0) {
    static const char kMsg[] = "tls-encrypted-echo";
    co_await tls_write(alloc, tls, kMsg, sizeof(kMsg));
    r->client_recv.store(
        co_await tls_read(alloc, tls, r->client_got, sizeof(kMsg)),
        std::memory_order_release);
  }
  r->client_done.store(true, std::memory_order_release);
  co_return 0;
}

struct Drive { Task<int> task; };
void drive_entry(void* p) noexcept { static_cast<Drive*>(p)->task.resume(); }

}  // namespace

TEST(IoTls, VerifyingClientRequiresServerIdentity) {
  TlsContext ctx = TlsContext::client();
  if (!ctx.valid()) GTEST_SKIP() << "system trust store unavailable";
  TlsStream missing_name(TcpStream{}, ctx, /*is_server=*/false);
  EXPECT_FALSE(missing_name.valid());
  TlsStream named(TcpStream{}, ctx, /*is_server=*/false, "example.com");
  EXPECT_TRUE(named.valid());
}

TEST(IoTls, HandshakeAndEncryptedEcho) {
  std::string cert = write_temp("cert.pem", kCertPem);
  std::string key = write_temp("key.pem", kKeyPem);

  TlsContext sctx = TlsContext::server(cert.c_str(), key.c_str());
  ASSERT_TRUE(sctx.valid()) << "server ctx (cert/key load) failed";
  TlsContext cctx = TlsContext::client();
  ASSERT_TRUE(cctx.valid());
  ASSERT_TRUE(sctx.enable_http2());
  ASSERT_TRUE(cctx.enable_http2());
  // Self-signed cert: client must NOT verify against a CA store (test-only).
  SSL_CTX_set_verify(cctx.native(), SSL_VERIFY_NONE, nullptr);

  constexpr unsigned kWorkers = 2;
  Scheduler sched(kWorkers, system_allocator(), 256);
  IoContext io(sched, kWorkers);
  ASSERT_TRUE(io.valid());
  TcpListener ls = tcp_listen(io, kOwner, "127.0.0.1", kPort);
  ASSERT_TRUE(ls.valid()) << "listen failed (port in use?)";

  auto r = std::make_unique<TlsResult>();
  auto ds = std::make_unique<Drive>();
  auto dc = std::make_unique<Drive>();
  ds->task = tls_server(system_allocator(), &ls, &sctx, r.get());
  dc->task = tls_client(system_allocator(), &io, &cctx, r.get());
  Fiber fs(kDefaultStackSize, &drive_entry, ds.get());
  Fiber fc(kDefaultStackSize, &drive_entry, dc.get());
  sched.spawn_on(kOwner, &fs);
  sched.spawn_on(kOwner, &fc);
  sched.drain();
  sched.shutdown();

  std::remove(cert.c_str());
  std::remove(key.c_str());

  EXPECT_TRUE(r->server_done.load());
  EXPECT_TRUE(r->client_done.load());
  EXPECT_EQ(r->server_hs.load(), 0) << "server handshake failed";
  EXPECT_EQ(r->client_hs.load(), 0) << "client handshake failed";
  EXPECT_TRUE(r->server_h2.load());
  EXPECT_TRUE(r->client_h2.load());
  static const char kMsg[] = "tls-encrypted-echo";
  EXPECT_EQ(r->server_echoed.load(), static_cast<long>(sizeof(kMsg)));
  EXPECT_EQ(r->client_recv.load(), static_cast<long>(sizeof(kMsg)));
  EXPECT_STREQ(r->client_got, "tls-encrypted-echo");
}

namespace {
struct HttpsResult {
  std::atomic<bool> server_done{false};
  std::atomic<bool> client_done{false};
  std::atomic<int> status{0};
  std::atomic<int> server_stage{0};
  std::atomic<int> client_stage{0};
  std::atomic<bool> handler_used_region{false};
  char body[32] = {};
};

Task<HttpResponse> https_handler(AllocatorRef alloc, void* context,
                                 const hyle::io::HttpRequest&) noexcept {
  auto* result = static_cast<HttpsResult*>(context);
  result->handler_used_region.store(
      alloc.alloc_fn() != system_allocator().alloc_fn(),
      std::memory_order_relaxed);
  char* body = alloc.allocate_n<char>(8);
  HttpResponse response;
  if (body == nullptr) {
    response.set_status(500);
    response.set_body({"request-region-oom", 18});
    co_return response;
  }
  std::memcpy(body, "https-ok", 8);
  response.set_body({body, 8});
  co_return response;
}

Task<int> https_server(AllocatorRef alloc, IoContext*, TcpListener* listener,
                       TlsContext* context, HttpsResult* result) {
  TcpStream socket = co_await listener->accept(kOwner);
  TlsStream tls(std::move(socket), *context, true);
  if (co_await tls_handshake(alloc, tls) == 0) {
    co_await hyle::io::http_serve(alloc, std::move(tls), &https_handler,
                                  result);
  }
  result->server_done.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> https_client(AllocatorRef alloc, IoContext* io, TlsContext* context,
                       HttpsResult* result, std::uint16_t port) {
  TcpStream socket = tcp_socket(*io, kOwner);
  alignas(16) unsigned char address[128]{};
  std::uint32_t address_length = 0;
  resolve_addr("127.0.0.1", port, address, &address_length);
  if (co_await socket.connect(address, address_length) == 0) {
    TlsStream tls(std::move(socket), *context, false);
    if (co_await tls_handshake(alloc, tls) == 0) {
      HttpClientConn http(alloc, std::move(tls));
      auto* response =
          co_await http.request({"GET", 3}, {"/", 1}, {"localhost", 9});
      if (response != nullptr) {
        result->status.store(response->status(), std::memory_order_release);
        const auto body = response->body();
        const auto size =
            body.size() < sizeof(result->body) - 1 ? body.size()
                                                  : sizeof(result->body) - 1;
        std::memcpy(result->body, body.data(), size);
      }
    }
  }
  result->client_done.store(true, std::memory_order_release);
  co_return 0;
}
}  // namespace

TEST(IoTls, HttpRunsOverTlsTransport) {
  constexpr std::uint16_t kHttpsPort = 54397;
  std::string cert = write_temp("https-cert.pem", kCertPem);
  std::string key = write_temp("https-key.pem", kKeyPem);
  TlsContext server_context =
      TlsContext::server(cert.c_str(), key.c_str());
  ASSERT_TRUE(server_context.valid());
  TlsContext client_context = TlsContext::client();
  ASSERT_TRUE(client_context.valid());
  SSL_CTX_set_verify(client_context.native(), SSL_VERIFY_NONE, nullptr);

  Scheduler sched(2, system_allocator(), 256);
  IoContext io(sched, 2);
  ASSERT_TRUE(io.valid());
  TcpListener listener =
      tcp_listen(io, kOwner, "127.0.0.1", kHttpsPort);
  ASSERT_TRUE(listener.valid());

  auto result = std::make_unique<HttpsResult>();
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task = https_server(system_allocator(), &io, &listener,
                              &server_context, result.get());
  client->task = https_client(system_allocator(), &io, &client_context,
                              result.get(), kHttpsPort);
  Fiber server_fiber(kDefaultStackSize, &drive_entry, server.get());
  Fiber client_fiber(kDefaultStackSize, &drive_entry, client.get());
  sched.spawn_on(kOwner, &server_fiber);
  sched.spawn_on(kOwner, &client_fiber);
  sched.drain();
  sched.shutdown();

  std::remove(cert.c_str());
  std::remove(key.c_str());
  EXPECT_TRUE(result->server_done.load());
  EXPECT_TRUE(result->client_done.load());
  EXPECT_EQ(result->status.load(), 200);
  EXPECT_STREQ(result->body, "https-ok");
  EXPECT_TRUE(result->handler_used_region.load());
}

#if defined(HYLE_IO_HTTP2)
namespace {
Task<int> h2_tls_server(AllocatorRef alloc, TcpListener* listener,
                        TlsContext* context, HttpsResult* result) {
  TcpStream socket = co_await listener->accept(kOwner);
  result->server_stage.store(1);
  TlsStream tls(std::move(socket), *context, true);
  if (co_await tls_handshake(alloc, tls) == 0) {
    result->server_stage.store(2);
    Http2ServerConn h2(alloc, std::move(tls));
    if (h2.valid()) result->server_stage.store(3);
    auto* request = co_await h2.next_request();
    if (request != nullptr) {
      result->server_stage.store(4);
      if (co_await h2.respond(request->stream_id(), 207, nullptr, 0,
                              {"h2-tls-ok", 9})) {
        result->server_stage.store(5);
        // Best-effort GOAWAY + TLS close_notify. A client is allowed to close
        // immediately after consuming the response, so EPIPE here is not a
        // response failure.
        (void)co_await h2.shutdown();
      }
    }
  }
  result->server_done.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> h2_tls_client(AllocatorRef alloc, IoContext* io, TlsContext* context,
                        HttpsResult* result, std::uint16_t port) {
  TcpStream socket = tcp_socket(*io, kOwner);
  alignas(16) unsigned char address[128]{};
  std::uint32_t address_length = 0;
  resolve_addr("127.0.0.1", port, address, &address_length);
  if (co_await socket.connect(address, address_length) == 0) {
    result->client_stage.store(1);
    TlsStream tls(std::move(socket), *context, false);
    if (co_await tls_handshake(alloc, tls) == 0) {
      result->client_stage.store(2);
      Http2ClientConn h2(alloc, std::move(tls));
      if (h2.valid()) result->client_stage.store(3);
      if (co_await h2.submit_request("GET", "https", "localhost", "/") > 0) {
        result->client_stage.store(4);
        auto* response = co_await h2.next_response();
        if (response != nullptr) {
          result->client_stage.store(5);
          result->status.store(response->status(), std::memory_order_release);
          const auto body = response->body();
          std::memcpy(result->body, body.data(), body.size());
        }
      }
    }
  }
  result->client_done.store(true, std::memory_order_release);
  co_return 0;
}
}  // namespace

TEST(IoTls, Http2RunsOverNegotiatedTlsTransport) {
  constexpr std::uint16_t kH2TlsPort = 54398;
  std::string cert = write_temp("h2-cert.pem", kCertPem);
  std::string key = write_temp("h2-key.pem", kKeyPem);
  TlsContext server_context = TlsContext::server(cert.c_str(), key.c_str());
  TlsContext client_context = TlsContext::client();
  ASSERT_TRUE(server_context.valid());
  ASSERT_TRUE(client_context.valid());
  ASSERT_TRUE(server_context.enable_http2());
  ASSERT_TRUE(client_context.enable_http2());
  SSL_CTX_set_verify(client_context.native(), SSL_VERIFY_NONE, nullptr);

  Scheduler sched(2, system_allocator(), 256);
  IoContext io(sched, 2);
  TcpListener listener =
      tcp_listen(io, kOwner, "127.0.0.1", kH2TlsPort);
  ASSERT_TRUE(listener.valid());
  auto result = std::make_unique<HttpsResult>();
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task = h2_tls_server(system_allocator(), &listener, &server_context,
                               result.get());
  client->task = h2_tls_client(system_allocator(), &io, &client_context,
                               result.get(), kH2TlsPort);
  Fiber server_fiber(kDefaultStackSize, &drive_entry, server.get());
  Fiber client_fiber(kDefaultStackSize, &drive_entry, client.get());
  sched.spawn_on(kOwner, &server_fiber);
  sched.spawn_on(kOwner, &client_fiber);
  sched.drain();
  sched.shutdown();
  std::remove(cert.c_str());
  std::remove(key.c_str());

  EXPECT_TRUE(result->server_done.load());
  EXPECT_TRUE(result->client_done.load());
  EXPECT_EQ(result->server_stage.load(), 5);
  EXPECT_EQ(result->client_stage.load(), 5);
  EXPECT_EQ(result->status.load(), 207);
  EXPECT_STREQ(result->body, "h2-tls-ok");
}
#endif

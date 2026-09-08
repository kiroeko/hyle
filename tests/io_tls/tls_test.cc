#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>

#if defined(HYLE_IO_TLS_HTTP_INTEGRATION)
#include "hyle/io/http.h"
#endif
#if defined(HYLE_IO_TLS_HTTP_INTEGRATION) && defined(HYLE_IO_HTTP2)
#include "hyle/io/http2.h"
#endif
#include "hyle/io/io_context.h"
#include "hyle/io/tcp.h"
#include "hyle/io/tls.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

// io-08 TLS test: a real TLS 1.2+ handshake + encrypted echo over loopback TCP,
// using the memory-BIO pump (TlsContext + TlsStream + co_await
// tls_handshake/read/ write). A self-signed cert (written to temp files) backs
// the server; the client skips verification (self-signed). Proves the whole
// pump: handshake ping-pong, SSL_write ciphertext -> socket, socket -> rbio ->
// SSL_read plaintext.

using hyle::io::IoContext;
#if defined(HYLE_IO_TLS_HTTP_INTEGRATION)
using hyle::io::HttpClientConnection;
using hyle::io::HttpRequestHead;
using hyle::io::HttpResponseHead;
using hyle::io::HttpServerConnection;
#endif
#if defined(HYLE_IO_TLS_HTTP_INTEGRATION) && defined(HYLE_IO_HTTP2)
using hyle::io::Http2ClientConn;
using hyle::io::Http2ServerConn;
#endif
using hyle::io::AddressFamily;
using hyle::io::Endpoint;
using hyle::io::TcpListener;
using hyle::io::TcpStream;
using hyle::io::tls_handshake;
using hyle::io::tls_read;
using hyle::io::tls_write;
using hyle::io::TlsClientOptions;
using hyle::io::TlsConnectOptions;
using hyle::io::TlsContext;
using hyle::io::TlsServerOptions;
using hyle::io::TlsStream;
using hyle::io::TransferResult;
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
const char *kCertPem =
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
const char *kKeyPem =
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

std::string write_temp(const char *name, const char *pem) {
  std::string path = std::string("hyle_tls_test_") + name;
  std::FILE *f = nullptr;
#if defined(_WIN32)
  (void)::fopen_s(&f, path.c_str(), "wb");
#else
  f = std::fopen(path.c_str(), "wb");
#endif
  if (f) {
    std::fwrite(pem, 1, std::strlen(pem), f);
    std::fclose(f);
  }
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

// Server: accept, wrap in server TlsStream, handshake, read plaintext, echo it
// back.
Task<int> tls_server(AllocatorRef alloc, TcpListener *ls, TlsContext *sctx,
                     TlsResult *r) {
  auto accepted = co_await ls->accept(kOwner);
  if (!accepted.has_value()) {
    r->server_done.store(true);
    co_return 0;
  }
  TcpStream conn = std::move(*accepted);
  auto wrapped = TlsStream::accept(std::move(conn), *sctx);
  if (!wrapped) {
    r->server_done.store(true);
    co_return 0;
  }
  TlsStream tls = std::move(*wrapped);
  hyle::Status hs = co_await tls_handshake(alloc, tls);
  r->server_hs.store(hs.ok() ? 0 : -1, std::memory_order_release);
  r->server_h2.store(tls.negotiated_protocol() == "h2",
                     std::memory_order_release);
  if (hs.ok()) {
    char buf[64] = {};
    auto read =
        co_await tls_read(alloc, tls, std::as_writable_bytes(std::span(buf)));
    if (read.ok() && read.bytes > 0) {
      auto written = co_await tls_write(
          alloc, tls, std::as_bytes(std::span(buf, read.bytes)));
      r->server_echoed.store(static_cast<long>(written.bytes),
                             std::memory_order_release);
    }
  }
  r->server_done.store(true, std::memory_order_release);
  co_return 0;
}

// Client: connect, wrap in client TlsStream, handshake, send, read echo,
// verify.
Task<int> tls_client(AllocatorRef alloc, IoContext *io, TlsContext *cctx,
                     TlsResult *r, std::uint16_t port = kPort) {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened.has_value()) {
    r->client_done.store(true);
    co_return 0;
  }
  TcpStream s = std::move(*opened);
  if (!(co_await s.connect(Endpoint::loopback(AddressFamily::kIpv4, port)))
           .ok()) {
    r->client_done.store(true);
    co_return 0;
  }
  auto wrapped = TlsStream::connect(std::move(s), *cctx);
  if (!wrapped) {
    r->client_done.store(true);
    co_return 0;
  }
  TlsStream tls = std::move(*wrapped);
  hyle::Status hs = co_await tls_handshake(alloc, tls);
  r->client_hs.store(hs.ok() ? 0 : -1, std::memory_order_release);
  r->client_h2.store(tls.negotiated_protocol() == "h2",
                     std::memory_order_release);
  if (hs.ok()) {
    static const char kMsg[] = "tls-encrypted-echo";
    (void)co_await tls_write(alloc, tls, std::as_bytes(std::span(kMsg)));
    auto received = co_await tls_read(
        alloc, tls, std::as_writable_bytes(std::span(r->client_got)));
    r->client_recv.store(static_cast<long>(received.bytes),
                         std::memory_order_release);
  }
  r->client_done.store(true, std::memory_order_release);
  co_return 0;
}

struct Drive {
  Task<int> task;
};
void drive_entry(void *p) noexcept { static_cast<Drive *>(p)->task.resume(); }

struct TlsDeadlineResult {
  std::atomic<bool> connected{false};
  std::atomic<hyle::StatusCode> status{hyle::StatusCode::Unknown};
  TcpStream held;
};

Task<int> tls_silent_peer(AllocatorRef, TcpListener *listener,
                          TlsDeadlineResult *result) {
  auto accepted = co_await listener->accept(kOwner);
  if (accepted)
    result->held = std::move(*accepted);
  co_return 0;
}

Task<int> tls_deadline_client(AllocatorRef allocator, IoContext *io,
                              TlsContext *context, TlsDeadlineResult *result,
                              std::uint16_t port) {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened)
    co_return 0;
  TcpStream socket = std::move(*opened);
  if (!(co_await socket.connect(Endpoint::loopback(AddressFamily::kIpv4, port)))
           .ok()) {
    co_return 0;
  }
  result->connected.store(true, std::memory_order_release);
  auto wrapped = TlsStream::connect(std::move(socket), *context);
  if (!wrapped)
    co_return 0;
  TlsStream stream = std::move(*wrapped);
  hyle::Status status = co_await tls_handshake(
      allocator, stream,
      hyle::io::OperationOptions{.deadline = hyle::time::deadline_after(
                                     std::chrono::milliseconds(25))});
  result->status.store(status.code(), std::memory_order_release);
  co_return 0;
}

Task<int> tls_cancelled_client(AllocatorRef allocator, IoContext *io,
                               TlsContext *context,
                               const hyle::sched::CancelToken *cancel,
                               TlsDeadlineResult *result, std::uint16_t port) {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened)
    co_return 0;
  TcpStream socket = std::move(*opened);
  if (!(co_await socket.connect(Endpoint::loopback(AddressFamily::kIpv4, port)))
           .ok()) {
    co_return 0;
  }
  result->connected.store(true, std::memory_order_release);
  auto wrapped = TlsStream::connect(std::move(socket), *context);
  if (!wrapped)
    co_return 0;
  TlsStream stream = std::move(*wrapped);
  hyle::Status status = co_await tls_handshake(
      allocator, stream,
      hyle::io::OperationOptions{.deadline = {}, .cancel = cancel});
  result->status.store(status.code(), std::memory_order_release);
  co_return 0;
}

struct TlsResumeResult {
  std::atomic<int> server_handshakes{0};
  std::atomic<int> client_handshakes{0};
  std::atomic<bool> reused{false};
  std::optional<hyle::io::TlsSession> session;
};

Task<int> tls_resume_server(AllocatorRef allocator, TcpListener *listener,
                            TlsContext *context, TlsResumeResult *result) {
  for (int attempt = 0; attempt < 2; ++attempt) {
    auto accepted = co_await listener->accept(kOwner);
    if (!accepted)
      co_return 0;
    auto wrapped = TlsStream::accept(std::move(*accepted), *context);
    if (!wrapped)
      co_return 0;
    TlsStream stream = std::move(*wrapped);
    if (!(co_await tls_handshake(allocator, stream)).ok())
      co_return 0;
    result->server_handshakes.fetch_add(1, std::memory_order_relaxed);
    (void)co_await hyle::io::tls_shutdown(allocator, stream);
  }
  co_return 0;
}

Task<int> tls_resume_client(AllocatorRef allocator, IoContext *io,
                            TlsContext *context, TlsResumeResult *result,
                            std::uint16_t port) {
  for (int attempt = 0; attempt < 2; ++attempt) {
    auto opened = TcpStream::open(*io, kOwner);
    if (!opened)
      co_return 0;
    TcpStream socket = std::move(*opened);
    if (!(co_await socket.connect(
              Endpoint::loopback(AddressFamily::kIpv4, port)))
             .ok()) {
      co_return 0;
    }
    TlsConnectOptions options;
    if (attempt != 0)
      options.resume_session = &*result->session;
    auto wrapped = TlsStream::connect(std::move(socket), *context, options);
    if (!wrapped)
      co_return 0;
    TlsStream stream = std::move(*wrapped);
    if (!(co_await tls_handshake(allocator, stream)).ok())
      co_return 0;
    result->client_handshakes.fetch_add(1, std::memory_order_relaxed);
    if (attempt == 0) {
      auto session = stream.take_session();
      if (!session)
        co_return 0;
      result->session.emplace(std::move(*session));
    } else {
      result->reused.store(stream.session_reused(), std::memory_order_relaxed);
    }
    (void)co_await hyle::io::tls_shutdown(allocator, stream);
  }
  co_return 0;
}

struct TlsTruncationResult {
  std::atomic<hyle::StatusCode> status{hyle::StatusCode::Unknown};
};

Task<int> tls_truncating_server(AllocatorRef allocator, TcpListener *listener,
                                TlsContext *context) {
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted)
    co_return 0;
  auto wrapped = TlsStream::accept(std::move(*accepted), *context);
  if (!wrapped)
    co_return 0;
  TlsStream stream = std::move(*wrapped);
  (void)co_await tls_handshake(allocator, stream);
  // Deliberately omit tls_shutdown: destruction closes the TCP stream without
  // close_notify, which must not be reported as a clean TLS EOF.
  co_return 0;
}

Task<int> tls_truncation_client(AllocatorRef allocator, IoContext *io,
                                TlsContext *context,
                                TlsTruncationResult *result,
                                std::uint16_t port) {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened)
    co_return 0;
  TcpStream socket = std::move(*opened);
  if (!(co_await socket.connect(Endpoint::loopback(AddressFamily::kIpv4, port)))
           .ok()) {
    co_return 0;
  }
  auto wrapped = TlsStream::connect(std::move(socket), *context);
  if (!wrapped)
    co_return 0;
  TlsStream stream = std::move(*wrapped);
  if (!(co_await tls_handshake(allocator, stream)).ok())
    co_return 0;
  std::byte byte{};
  TransferResult received = co_await tls_read(
      allocator, stream, std::span(&byte, 1),
      hyle::io::OperationOptions{
          .deadline = hyle::time::deadline_after(std::chrono::seconds(1))});
  result->status.store(received.status.code(), std::memory_order_release);
  co_return 0;
}

} // namespace

TEST(IoTls, VerifyingClientRequiresServerIdentity) {
  auto made = TlsContext::make_client();
  if (!made)
    GTEST_SKIP() << "system trust store unavailable";
  TlsContext ctx = std::move(*made);
  Scheduler scheduler(1, system_allocator(), 32);
  IoContext io(scheduler);
  auto first = TcpStream::open(io, kOwner);
  auto second = TcpStream::open(io, kOwner);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  auto missing_name = TlsStream::connect(std::move(*first), ctx);
  EXPECT_FALSE(missing_name);
  auto named = TlsStream::connect(
      std::move(*second), ctx, TlsConnectOptions{.server_name = "example.com"});
  EXPECT_TRUE(named);
  scheduler.shutdown();
}

TEST(IoTls, TypedFactoryRejectsInvalidConfiguration) {
  TlsClientOptions reversed;
  reversed.min_version = hyle::io::TlsVersion::kTls13;
  reversed.max_version = hyle::io::TlsVersion::kTls12;
  auto bad_range = TlsContext::make_client(reversed);
  ASSERT_FALSE(bad_range);
  EXPECT_TRUE(bad_range.error().IsInvalidArgument());

  std::array<char, 256> oversized_name{};
  const hyle::container::StringView protocol{
      oversized_name.data(), static_cast<std::uint32_t>(oversized_name.size())};
  TlsClientOptions oversized;
  oversized.verify = hyle::io::TlsVerifyMode::kNone;
  oversized.alpn_protocols = std::span(&protocol, 1);
  auto bad_alpn = TlsContext::make_client(oversized);
  ASSERT_FALSE(bad_alpn);
  EXPECT_TRUE(bad_alpn.error().IsInvalidArgument());

  auto missing_server_identity = TlsContext::make_server(TlsServerOptions{});
  ASSERT_FALSE(missing_server_identity);
  EXPECT_TRUE(missing_server_identity.error().IsInvalidArgument());

  TlsClientOptions incomplete_client_identity;
  incomplete_client_identity.verify = hyle::io::TlsVerifyMode::kNone;
  incomplete_client_identity.certificate_chain_file = "client.pem";
  auto missing_client_key = TlsContext::make_client(incomplete_client_identity);
  ASSERT_FALSE(missing_client_key);
  EXPECT_TRUE(missing_client_key.error().IsInvalidArgument());

  TlsClientOptions invalid_allocator;
  invalid_allocator.verify = hyle::io::TlsVerifyMode::kNone;
  invalid_allocator.connection_allocator = {};
  auto bad_allocator = TlsContext::make_client(invalid_allocator);
  ASSERT_FALSE(bad_allocator);
  EXPECT_TRUE(bad_allocator.error().IsInvalidArgument());
}

TEST(IoTls, StreamRetainsOpaqueContextState) {
  TlsClientOptions options;
  options.verify = hyle::io::TlsVerifyMode::kNone;
  auto made = TlsContext::make_client(options);
  ASSERT_TRUE(made);
  TlsContext context = std::move(*made);

  Scheduler scheduler(1, system_allocator(), 32);
  IoContext io(scheduler);
  {
    auto opened = TcpStream::open(io, kOwner);
    ASSERT_TRUE(opened);
    auto wrapped = TlsStream::connect(std::move(*opened), context);
    ASSERT_TRUE(wrapped);
    context = TlsContext{};
    EXPECT_TRUE(wrapped->valid());
  }
  scheduler.shutdown();
}

TEST(IoTls, HandshakeUsesOneAbsoluteOperationBudget) {
  constexpr std::uint16_t kDeadlinePort = 54396;
  TlsClientOptions options;
  options.verify = hyle::io::TlsVerifyMode::kNone;
  auto made = TlsContext::make_client(options);
  ASSERT_TRUE(made);
  TlsContext context = std::move(*made);

  Scheduler scheduler(1, system_allocator(), 64);
  IoContext io(scheduler);
  auto made_listener = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kDeadlinePort));
  ASSERT_TRUE(made_listener);
  TcpListener listener = std::move(*made_listener);
  auto result = std::make_unique<TlsDeadlineResult>();
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task = tls_silent_peer(system_allocator(), &listener, result.get());
  client->task = tls_deadline_client(system_allocator(), &io, &context,
                                     result.get(), kDeadlinePort);
  Fiber server_fiber(kDefaultStackSize, &drive_entry, server.get());
  Fiber client_fiber(kDefaultStackSize, &drive_entry, client.get());
  scheduler.spawn_on(kOwner, &server_fiber);
  scheduler.spawn_on(kOwner, &client_fiber);
  scheduler.drain();
  scheduler.shutdown();

  EXPECT_TRUE(result->connected.load());
  EXPECT_EQ(result->status.load(), hyle::StatusCode::DeadlineExceeded);
}

TEST(IoTls, HandshakePropagatesCancellationBudget) {
  constexpr std::uint16_t kCancelPort = 54401;
  TlsClientOptions options;
  options.verify = hyle::io::TlsVerifyMode::kNone;
  auto made = TlsContext::make_client(options);
  ASSERT_TRUE(made);
  TlsContext context = std::move(*made);
  auto cancel = hyle::sched::CancelToken::try_create_root(system_allocator());
  ASSERT_TRUE(cancel);
  cancel->cancel();

  Scheduler scheduler(1, system_allocator(), 64);
  IoContext io(scheduler);
  auto made_listener = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kCancelPort));
  ASSERT_TRUE(made_listener);
  TcpListener listener = std::move(*made_listener);
  auto result = std::make_unique<TlsDeadlineResult>();
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task = tls_silent_peer(system_allocator(), &listener, result.get());
  client->task = tls_cancelled_client(system_allocator(), &io, &context,
                                      &*cancel, result.get(), kCancelPort);
  Fiber server_fiber(kDefaultStackSize, &drive_entry, server.get());
  Fiber client_fiber(kDefaultStackSize, &drive_entry, client.get());
  scheduler.spawn_on(kOwner, &server_fiber);
  scheduler.spawn_on(kOwner, &client_fiber);
  scheduler.drain();
  scheduler.shutdown();

  EXPECT_TRUE(result->connected.load());
  EXPECT_EQ(result->status.load(), hyle::StatusCode::Cancelled);
}

TEST(IoTls, HandshakeAndEncryptedEcho) {
  std::string cert = write_temp("cert.pem", kCertPem);
  std::string key = write_temp("key.pem", kKeyPem);

  TlsServerOptions server_options;
  server_options.certificate_chain_file = cert;
  server_options.private_key_file = key;
  server_options.alpn_protocols = hyle::io::tls_http2_protocols();
  auto made_server = TlsContext::make_server(server_options);
  ASSERT_TRUE(made_server) << "server ctx (cert/key load) failed";
  TlsContext sctx = std::move(*made_server);
  TlsClientOptions client_options;
  client_options.verify = hyle::io::TlsVerifyMode::kNone;
  client_options.alpn_protocols = hyle::io::tls_http2_protocols();
  auto made_client = TlsContext::make_client(client_options);
  ASSERT_TRUE(made_client);
  TlsContext cctx = std::move(*made_client);

  constexpr unsigned kWorkers = 2;
  Scheduler sched(kWorkers, system_allocator(), 256);
  IoContext io(sched);
  ASSERT_TRUE(io.valid());
  auto opened_listener = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kPort));
  ASSERT_TRUE(opened_listener.has_value()) << "listen failed (port in use?)";
  TcpListener ls = std::move(*opened_listener);

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

TEST(IoTls, MutualTlsAuthenticatesClientIdentity) {
  constexpr std::uint16_t kMutualTlsPort = 54402;
  std::string cert = write_temp("mtls-cert.pem", kCertPem);
  std::string key = write_temp("mtls-key.pem", kKeyPem);

  TlsServerOptions server_options;
  server_options.certificate_chain_file = cert;
  server_options.private_key_file = key;
  server_options.client_verification = hyle::io::TlsVerifyMode::kPeer;
  server_options.client_certificate_authority_file = cert;
  auto made_server = TlsContext::make_server(server_options);
  ASSERT_TRUE(made_server);
  TlsContext server_context = std::move(*made_server);

  TlsClientOptions client_options;
  client_options.verify = hyle::io::TlsVerifyMode::kNone;
  client_options.certificate_chain_file = cert;
  client_options.private_key_file = key;
  auto made_client = TlsContext::make_client(client_options);
  ASSERT_TRUE(made_client);
  TlsContext client_context = std::move(*made_client);

  Scheduler scheduler(1, system_allocator(), 128);
  IoContext io(scheduler);
  auto made_listener = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kMutualTlsPort));
  ASSERT_TRUE(made_listener);
  TcpListener listener = std::move(*made_listener);
  auto result = std::make_unique<TlsResult>();
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task =
      tls_server(system_allocator(), &listener, &server_context, result.get());
  client->task = tls_client(system_allocator(), &io, &client_context,
                            result.get(), kMutualTlsPort);
  Fiber server_fiber(kDefaultStackSize, &drive_entry, server.get());
  Fiber client_fiber(kDefaultStackSize, &drive_entry, client.get());
  scheduler.spawn_on(kOwner, &server_fiber);
  scheduler.spawn_on(kOwner, &client_fiber);
  scheduler.drain();
  scheduler.shutdown();

  std::remove(cert.c_str());
  std::remove(key.c_str());
  EXPECT_EQ(result->server_hs.load(), 0);
  EXPECT_EQ(result->client_hs.load(), 0);
  EXPECT_GT(result->client_recv.load(), 0);
}

TEST(IoTls, OpaqueSessionCanResumeOnItsOwningContext) {
  constexpr std::uint16_t kResumePort = 54399;
  std::string cert = write_temp("resume-cert.pem", kCertPem);
  std::string key = write_temp("resume-key.pem", kKeyPem);
  TlsServerOptions server_options;
  server_options.certificate_chain_file = cert;
  server_options.private_key_file = key;
  server_options.max_version = hyle::io::TlsVersion::kTls12;
  server_options.enable_session_resumption = true;
  auto made_server = TlsContext::make_server(server_options);
  ASSERT_TRUE(made_server);
  TlsContext server_context = std::move(*made_server);
  TlsClientOptions client_options;
  client_options.verify = hyle::io::TlsVerifyMode::kNone;
  client_options.max_version = hyle::io::TlsVersion::kTls12;
  client_options.enable_session_resumption = true;
  auto made_client = TlsContext::make_client(client_options);
  ASSERT_TRUE(made_client);
  TlsContext client_context = std::move(*made_client);

  Scheduler scheduler(1, system_allocator(), 128);
  IoContext io(scheduler);
  auto made_listener = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kResumePort));
  ASSERT_TRUE(made_listener);
  TcpListener listener = std::move(*made_listener);
  auto result = std::make_unique<TlsResumeResult>();
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task = tls_resume_server(system_allocator(), &listener,
                                   &server_context, result.get());
  client->task = tls_resume_client(system_allocator(), &io, &client_context,
                                   result.get(), kResumePort);
  Fiber server_fiber(kDefaultStackSize, &drive_entry, server.get());
  Fiber client_fiber(kDefaultStackSize, &drive_entry, client.get());
  scheduler.spawn_on(kOwner, &server_fiber);
  scheduler.spawn_on(kOwner, &client_fiber);
  scheduler.drain();

  ASSERT_TRUE(result->session.has_value());
  auto other_context_result = TlsContext::make_client(client_options);
  ASSERT_TRUE(other_context_result);
  TlsContext other_context = std::move(*other_context_result);
  auto mismatch_socket = TcpStream::open(io, kOwner);
  ASSERT_TRUE(mismatch_socket);
  auto mismatch = TlsStream::connect(
      std::move(*mismatch_socket), other_context,
      TlsConnectOptions{.resume_session = &*result->session});
  ASSERT_FALSE(mismatch);
  EXPECT_TRUE(mismatch.error().IsInvalidArgument());
  scheduler.shutdown();
  std::remove(cert.c_str());
  std::remove(key.c_str());

  EXPECT_EQ(result->server_handshakes.load(), 2);
  EXPECT_EQ(result->client_handshakes.load(), 2);
  EXPECT_TRUE(result->reused.load());
}

TEST(IoTls, MissingCloseNotifyIsReportedAsDataLoss) {
  constexpr std::uint16_t kTruncationPort = 54400;
  std::string cert = write_temp("truncation-cert.pem", kCertPem);
  std::string key = write_temp("truncation-key.pem", kKeyPem);
  TlsServerOptions server_options;
  server_options.certificate_chain_file = cert;
  server_options.private_key_file = key;
  auto made_server = TlsContext::make_server(server_options);
  ASSERT_TRUE(made_server);
  TlsContext server_context = std::move(*made_server);
  TlsClientOptions client_options;
  client_options.verify = hyle::io::TlsVerifyMode::kNone;
  auto made_client = TlsContext::make_client(client_options);
  ASSERT_TRUE(made_client);
  TlsContext client_context = std::move(*made_client);

  Scheduler scheduler(1, system_allocator(), 64);
  IoContext io(scheduler);
  auto made_listener = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kTruncationPort));
  ASSERT_TRUE(made_listener);
  TcpListener listener = std::move(*made_listener);
  auto result = std::make_unique<TlsTruncationResult>();
  auto server = std::make_unique<Drive>();
  auto client = std::make_unique<Drive>();
  server->task =
      tls_truncating_server(system_allocator(), &listener, &server_context);
  client->task = tls_truncation_client(system_allocator(), &io, &client_context,
                                       result.get(), kTruncationPort);
  Fiber server_fiber(kDefaultStackSize, &drive_entry, server.get());
  Fiber client_fiber(kDefaultStackSize, &drive_entry, client.get());
  scheduler.spawn_on(kOwner, &server_fiber);
  scheduler.spawn_on(kOwner, &client_fiber);
  scheduler.drain();
  scheduler.shutdown();
  std::remove(cert.c_str());
  std::remove(key.c_str());

  EXPECT_EQ(result->status.load(), hyle::StatusCode::DataLoss);
}

#if defined(HYLE_IO_TLS_HTTP_INTEGRATION)

namespace {
struct HttpsResult {
  std::atomic<bool> server_done{false};
  std::atomic<bool> client_done{false};
  std::atomic<int> status{0};
  char body[32] = {};
};

Task<int> https_server(AllocatorRef alloc, IoContext *, TcpListener *listener,
                       TlsContext *context, HttpsResult *result) {
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted.has_value())
    co_return 0;
  TcpStream socket = std::move(*accepted);
  auto wrapped = TlsStream::accept(std::move(socket), *context);
  if (!wrapped)
    co_return 0;
  TlsStream tls = std::move(*wrapped);
  if ((co_await tls_handshake(alloc, tls)).ok()) {
    hyle::io::HttpLimits limits;
    // Exercise both encrypted chunk routes: the 5-byte chunk exceeds this
    // staging budget once framing is included, while the 3-byte chunk fits.
    limits.max_retained_body_bytes = 8;
    auto made = HttpServerConnection::create(alloc, std::move(tls), limits);
    if (made) {
      HttpServerConnection connection = std::move(*made);
      auto incoming = co_await connection.read_request();
      if (incoming && incoming->has_value() &&
          (co_await incoming->value().body.discard()).ok()) {
        static constexpr char kBody[] = "https-ok";
        HttpResponseHead response;
        auto started = co_await connection.start_response(response);
        if (started) {
          auto writer = std::move(*started);
          auto first = co_await writer.write(
              std::as_bytes(std::span(kBody, std::size_t{5})));
          auto second = co_await writer.write(std::as_bytes(
              std::span(kBody + 5, sizeof(kBody) - 1 - std::size_t{5})));
          if (first.ok() && second.ok())
            (void)co_await writer.finish();
        }
      }
    }
  }
  result->server_done.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> https_client(AllocatorRef alloc, IoContext *io, TlsContext *context,
                       HttpsResult *result, std::uint16_t port) {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened.has_value())
    co_return 0;
  TcpStream socket = std::move(*opened);
  if ((co_await socket.connect(Endpoint::loopback(AddressFamily::kIpv4, port)))
          .ok()) {
    auto wrapped = TlsStream::connect(std::move(socket), *context);
    if (!wrapped)
      co_return 0;
    TlsStream tls = std::move(*wrapped);
    if ((co_await tls_handshake(alloc, tls)).ok()) {
      auto made = HttpClientConnection::create(alloc, std::move(tls));
      if (made) {
        HttpClientConnection http = std::move(*made);
        HttpRequestHead request;
        request.authority = "localhost";
        auto response =
            co_await http.request(request, std::span<const std::byte>{});
        if (response) {
          result->status.store(response->head.status(),
                               std::memory_order_release);
          auto body =
              co_await response->body.read_all(sizeof(result->body) - 1);
          if (body) {
            std::memcpy(result->body, body->data(), body->size());
          }
        }
      }
    }
  }
  result->client_done.store(true, std::memory_order_release);
  co_return 0;
}
} // namespace

TEST(IoTls, HttpRunsOverTlsTransport) {
  constexpr std::uint16_t kHttpsPort = 54397;
  std::string cert = write_temp("https-cert.pem", kCertPem);
  std::string key = write_temp("https-key.pem", kKeyPem);
  TlsServerOptions server_options;
  server_options.certificate_chain_file = cert;
  server_options.private_key_file = key;
  auto made_server = TlsContext::make_server(server_options);
  ASSERT_TRUE(made_server);
  TlsContext server_context = std::move(*made_server);
  TlsClientOptions client_options;
  client_options.verify = hyle::io::TlsVerifyMode::kNone;
  auto made_client = TlsContext::make_client(client_options);
  ASSERT_TRUE(made_client);
  TlsContext client_context = std::move(*made_client);

  Scheduler sched(2, system_allocator(), 256);
  IoContext io(sched);
  ASSERT_TRUE(io.valid());
  auto opened_listener = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kHttpsPort));
  ASSERT_TRUE(opened_listener.has_value());
  TcpListener listener = std::move(*opened_listener);

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
}

#if defined(HYLE_IO_HTTP2)
namespace {
Task<int> h2_tls_server(AllocatorRef alloc, TcpListener *listener,
                        TlsContext *context, HttpsResult *result) {
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted.has_value())
    co_return 0;
  TcpStream socket = std::move(*accepted);
  result->server_stage.store(1);
  auto wrapped = TlsStream::accept(std::move(socket), *context);
  if (!wrapped)
    co_return 0;
  TlsStream tls = std::move(*wrapped);
  if ((co_await tls_handshake(alloc, tls)).ok()) {
    result->server_stage.store(2);
    Http2ServerConn h2(alloc, std::move(tls));
    if (h2.valid())
      result->server_stage.store(3);
    auto *request = co_await h2.next_request();
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

Task<int> h2_tls_client(AllocatorRef alloc, IoContext *io, TlsContext *context,
                        HttpsResult *result, std::uint16_t port) {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened.has_value())
    co_return 0;
  TcpStream socket = std::move(*opened);
  if ((co_await socket.connect(Endpoint::loopback(AddressFamily::kIpv4, port)))
          .ok()) {
    result->client_stage.store(1);
    auto wrapped = TlsStream::connect(std::move(socket), *context);
    if (!wrapped)
      co_return 0;
    TlsStream tls = std::move(*wrapped);
    if ((co_await tls_handshake(alloc, tls)).ok()) {
      result->client_stage.store(2);
      Http2ClientConn h2(alloc, std::move(tls));
      if (h2.valid())
        result->client_stage.store(3);
      if (co_await h2.submit_request("GET", "https", "localhost", "/") > 0) {
        result->client_stage.store(4);
        auto *response = co_await h2.next_response();
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
} // namespace

TEST(IoTls, Http2RunsOverNegotiatedTlsTransport) {
  constexpr std::uint16_t kH2TlsPort = 54398;
  std::string cert = write_temp("h2-cert.pem", kCertPem);
  std::string key = write_temp("h2-key.pem", kKeyPem);
  TlsServerOptions server_options;
  server_options.certificate_chain_file = cert;
  server_options.private_key_file = key;
  server_options.alpn_protocols = hyle::io::tls_http2_protocols();
  server_options.require_alpn = true;
  auto made_server = TlsContext::make_server(server_options);
  TlsClientOptions client_options;
  client_options.verify = hyle::io::TlsVerifyMode::kNone;
  client_options.alpn_protocols = hyle::io::tls_http2_protocols();
  client_options.require_alpn = true;
  auto made_client = TlsContext::make_client(client_options);
  ASSERT_TRUE(made_server);
  ASSERT_TRUE(made_client);
  TlsContext server_context = std::move(*made_server);
  TlsContext client_context = std::move(*made_client);

  Scheduler sched(2, system_allocator(), 256);
  IoContext io(sched);
  auto opened_listener = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, kH2TlsPort));
  ASSERT_TRUE(opened_listener.has_value());
  TcpListener listener = std::move(*opened_listener);
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

#endif // HYLE_IO_TLS_HTTP_INTEGRATION

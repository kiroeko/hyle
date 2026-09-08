// io_tls P1 consumer benchmark. The product pair keeps one runtime, TLS
// contexts and listener alive, then charges one complete connection episode:
// HYLE's public spawn_task/TLS route versus standalone Asio's public async SSL
// route over the same OpenSSL provider. Both sides connect, handshake, move the
// same bytes and exchange close_notify. Runtime/context/listener construction
// is outside the charged episode, matching a long-lived inference server.
//
// A raw blocking OpenSSL route remains available only when
// HYLE_TLS_RAW_DIAGNOSTIC is set. Blocking versus scheduler-bound async IO has
// a different progress/lifecycle contract, so that row is a mechanism
// diagnostic and MUST NOT be used for product non-inferiority or SOTA claims.

#include <nanobench.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <string>
#include <thread>

#include <openssl/bio.h>
#include <openssl/err.h>
#include <openssl/ssl.h>

#include "hyle/io/io_context.h"
#include "hyle/io/tcp.h"
#include "hyle/io/tls.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/join.h"
#include "hyle/sched/scheduler.h"

#if defined(HYLE_BENCH_ASIO_TLS)
#include <asio.hpp>
#include <asio/ssl.hpp>
#endif

#if !defined(_WIN32)
#include <arpa/inet.h>
#include <csignal> // ignore SIGPIPE: a write to a peer-closed socket must not kill us
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif
#if defined(_WIN32)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

using hyle::io::AddressFamily;
using hyle::io::Endpoint;
using hyle::io::IoContext;
using hyle::io::TcpListener;
using hyle::io::TcpStream;
using hyle::io::tls_handshake;
using hyle::io::tls_read;
using hyle::io::tls_shutdown;
using hyle::io::tls_write;
using hyle::io::TlsConnectOptions;
using hyle::io::TlsContext;
using hyle::io::TlsShutdownMode;
using hyle::io::TlsShutdownOptions;
using hyle::io::TlsStream;
using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::SchedulerIdlePolicy;
using hyle::sched::SchedulerOptions;
using hyle::sched::Task;

namespace {

constexpr unsigned kOwner = 0;
constexpr int kMsgLen = 256; // TLS record payload
constexpr int kEchoRoundTrips =
    512; // amortizes one honest connection lifecycle without hiding it

// Diagnostic-only scheduler selector. The qualified product route leaves this
// unset and therefore exercises HYLE's public default. Selecting power-saving
// removes the scheduler's idle search phase without changing the TLS or socket
// contract, which isolates completion-poll latency from provider work.
SchedulerOptions hyle_scheduler_options() noexcept {
  SchedulerOptions options;
  if (const char *value = std::getenv("HYLE_TLS_DIAGNOSTIC_IDLE_POLICY")) {
    if (std::strcmp(value, "power-saving") == 0)
      options.idle_policy = SchedulerIdlePolicy::kPowerSaving;
    else if (std::strcmp(value, "low-latency") == 0)
      options.idle_policy = SchedulerIdlePolicy::kLowLatency;
  }
  return options;
}

// Self-signed cert + key (CN=localhost, RSA 2048), same as tls_test.cc.
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
  std::string path = std::string("hyle_tls_bench_") + name;
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

// ---- HYLE side --------------------------------------------------------------
struct HyleTls {
  std::uint16_t port;
  bool do_echo; // false = handshake-only (then close); true = handshake + N
                // echoes
  TlsContext *sctx;
  TlsContext *cctx;
};

Task<int> hyle_server(AllocatorRef alloc, TcpListener *ls, HyleTls *h) {
  auto accepted = co_await ls->accept(kOwner);
  if (!accepted.has_value())
    co_return 0;
  TcpStream conn = std::move(*accepted);
  auto wrapped = TlsStream::accept(std::move(conn), *h->sctx);
  if (!wrapped)
    co_return 0;
  TlsStream tls = std::move(*wrapped);
  if (!(co_await tls_handshake(alloc, tls)).ok())
    co_return 0;
  int units = 1;
  if (h->do_echo) {
    units = 0;
    char buf[kMsgLen];
    for (int i = 0; i < kEchoRoundTrips; ++i) {
      std::size_t received_bytes = 0;
      while (received_bytes != sizeof(buf)) {
        auto received = co_await tls_read(
            alloc, tls,
            std::as_writable_bytes(std::span(buf)).subspan(received_bytes));
        if (!received.ok() || received.bytes == 0)
          co_return 0;
        received_bytes += received.bytes;
      }
      std::size_t sent_bytes = 0;
      while (sent_bytes != sizeof(buf)) {
        auto sent = co_await tls_write(
            alloc, tls, std::as_bytes(std::span(buf)).subspan(sent_bytes));
        if (!sent.ok() || sent.bytes == 0)
          co_return 0;
        sent_bytes += sent.bytes;
      }
      ++units;
    }
  }
  TlsShutdownOptions shutdown;
  shutdown.mode = TlsShutdownMode::kBidirectional;
  if (!(co_await tls_shutdown(alloc, tls, shutdown)).ok())
    co_return 0;
  co_return units;
}

Task<int> hyle_client(AllocatorRef alloc, IoContext *io, HyleTls *h) {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened.has_value())
    co_return 0;
  TcpStream s = std::move(*opened);
  if (!(co_await s.connect(Endpoint::loopback(AddressFamily::kIpv4, h->port)))
           .ok())
    co_return 0;
  TlsConnectOptions connect_options;
  connect_options.server_name = "localhost";
  auto wrapped = TlsStream::connect(std::move(s), *h->cctx, connect_options);
  if (!wrapped)
    co_return 0;
  TlsStream tls = std::move(*wrapped);
  if (!(co_await tls_handshake(alloc, tls)).ok())
    co_return 0;
  int units = 1;
  if (h->do_echo) {
    units = 0;
    static const char kMsg[kMsgLen] = "hyle-tls-bench-payload";
    char buf[kMsgLen];
    for (int i = 0; i < kEchoRoundTrips; ++i) {
      std::size_t sent_bytes = 0;
      while (sent_bytes != sizeof(kMsg)) {
        auto sent = co_await tls_write(
            alloc, tls, std::as_bytes(std::span(kMsg)).subspan(sent_bytes));
        if (!sent.ok() || sent.bytes == 0)
          co_return 0;
        sent_bytes += sent.bytes;
      }
      std::size_t received_bytes = 0;
      while (received_bytes != sizeof(buf)) {
        auto received = co_await tls_read(
            alloc, tls,
            std::as_writable_bytes(std::span(buf)).subspan(received_bytes));
        if (!received.ok() || received.bytes == 0)
          co_return 0;
        received_bytes += received.bytes;
      }
      ++units;
    }
  }
  TlsShutdownOptions shutdown;
  shutdown.mode = TlsShutdownMode::kBidirectional;
  if (!(co_await tls_shutdown(alloc, tls, shutdown)).ok())
    co_return 0;
  co_return units;
}

// Persistent public HYLE consumer. Scheduler/IoContext/TLS contexts/listener
// are server-level state; one run charges only a connection task tree.
class HyleTlsHarness {
public:
  HyleTlsHarness(std::uint16_t port, TlsContext *server, TlsContext *client)
      : scheduler_(1, system_allocator(), 1024, hyle_scheduler_options()),
        io_(scheduler_) {
    auto opened = TcpListener::bind(
        io_, kOwner, Endpoint::loopback(AddressFamily::kIpv4, port));
    if (!opened)
      std::abort();
    listener_ = std::move(*opened);
    state_.port = port;
    state_.sctx = server;
    state_.cctx = client;
  }

  ~HyleTlsHarness() { scheduler_.shutdown(); }

  int run(bool do_echo) {
    state_.do_echo = do_echo;
    auto server = hyle::sched::spawn_task(
        scheduler_, system_allocator(),
        hyle_server(system_allocator(), &listener_, &state_));
    auto client =
        hyle::sched::spawn_task(scheduler_, system_allocator(),
                                hyle_client(system_allocator(), &io_, &state_));
    auto client_result = client.join();
    auto server_result = server.join();
    if (!client_result || !server_result || *client_result != *server_result)
      return -1;
    return *client_result;
  }

private:
  Scheduler scheduler_;
  IoContext io_;
  TcpListener listener_;
  HyleTls state_{0, false, nullptr, nullptr};
};

#if defined(HYLE_BENCH_ASIO_TLS)

class AsioTlsHarness {
  using Tcp = asio::ip::tcp;

  struct Episode : std::enable_shared_from_this<Episode> {
    explicit Episode(AsioTlsHarness *owner, bool echo)
        : owner(owner), server(owner->io_, owner->server_context_),
          client(owner->io_, owner->client_context_), do_echo(echo) {
      static constexpr char kPayload[] = "hyle-tls-bench-payload";
      std::memcpy(client_message.data(), kPayload, sizeof(kPayload));
    }

    void start() {
      auto self = shared_from_this();
      if (SSL_set_tlsext_host_name(client.native_handle(), "localhost") != 1)
        return fail();
      client.set_verify_callback(
          asio::ssl::host_name_verification("localhost"));
      owner->listener_.async_accept(
          server.next_layer(), [self](const asio::error_code &error) {
            if (error)
              return self->fail();
            self->server.async_handshake(
                asio::ssl::stream_base::server,
                [self](const asio::error_code &handshake_error) {
                  if (handshake_error)
                    return self->fail();
                  if (self->do_echo)
                    self->server_read();
                  else
                    self->shutdown_server();
                });
          });
      client.next_layer().async_connect(
          owner->listener_.local_endpoint(),
          [self](const asio::error_code &error) {
            if (error)
              return self->fail();
            self->client.async_handshake(
                asio::ssl::stream_base::client,
                [self](const asio::error_code &handshake_error) {
                  if (handshake_error)
                    return self->fail();
                  if (self->do_echo)
                    self->client_write();
                  else
                    self->shutdown_client();
                });
          });
    }

    void server_read() {
      auto self = shared_from_this();
      asio::async_read(
          server, asio::buffer(server_buffer),
          [self](const asio::error_code &error, std::size_t bytes) {
            if (error || bytes != self->server_buffer.size())
              return self->fail();
            asio::async_write(self->server, asio::buffer(self->server_buffer),
                              [self](const asio::error_code &write_error,
                                     std::size_t written) {
                                if (write_error ||
                                    written != self->server_buffer.size())
                                  return self->fail();
                                if (++self->server_units == kEchoRoundTrips)
                                  self->shutdown_server();
                                else
                                  self->server_read();
                              });
          });
    }

    void client_write() {
      auto self = shared_from_this();
      asio::async_write(
          client, asio::buffer(client_message),
          [self](const asio::error_code &error, std::size_t written) {
            if (error || written != self->client_message.size())
              return self->fail();
            asio::async_read(
                self->client, asio::buffer(self->client_buffer),
                [self](const asio::error_code &read_error, std::size_t bytes) {
                  if (read_error || bytes != self->client_buffer.size())
                    return self->fail();
                  ++self->client_units;
                  if (self->client_units == kEchoRoundTrips)
                    self->shutdown_client();
                  else
                    self->client_write();
                });
          });
    }

    void shutdown_server() {
      auto self = shared_from_this();
      server.async_shutdown([self](const asio::error_code &error) {
        if (error)
          return self->fail();
        asio::error_code ignored;
        self->server.next_layer().close(ignored);
        ++self->shutdowns;
        self->finish_if_ready();
      });
    }

    void shutdown_client() {
      auto self = shared_from_this();
      client.async_shutdown([self](const asio::error_code &error) {
        if (error)
          return self->fail();
        asio::error_code ignored;
        self->client.next_layer().close(ignored);
        ++self->shutdowns;
        self->finish_if_ready();
      });
    }

    void finish_if_ready() {
      if (shutdowns == 2)
        complete(do_echo ? client_units : 1);
    }

    void fail() {
      if (finished)
        return;
      asio::error_code ignored;
      server.next_layer().cancel(ignored);
      server.next_layer().close(ignored);
      client.next_layer().cancel(ignored);
      client.next_layer().close(ignored);
      complete(-1);
    }

    void complete(int value) {
      if (finished)
        return;
      finished = true;
      result.set_value(value);
    }

    AsioTlsHarness *owner;
    asio::ssl::stream<Tcp::socket> server;
    asio::ssl::stream<Tcp::socket> client;
    std::array<char, kMsgLen> server_buffer{};
    std::array<char, kMsgLen> client_buffer{};
    std::array<char, kMsgLen> client_message{};
    std::promise<int> result;
    bool do_echo = false;
    bool finished = false;
    int server_units = 0;
    int client_units = 0;
    int shutdowns = 0;
  };

public:
  AsioTlsHarness(std::uint16_t port, const std::string &certificate,
                 const std::string &private_key)
      : io_(1), server_context_(asio::ssl::context::tls_server),
        client_context_(asio::ssl::context::tls_client), listener_(io_),
        work_(asio::make_work_guard(io_)) {
    server_context_.use_certificate_chain_file(certificate);
    server_context_.use_private_key_file(private_key, asio::ssl::context::pem);
    client_context_.load_verify_file(certificate);
    client_context_.set_verify_mode(asio::ssl::verify_peer);
    for (SSL_CTX *context :
         {server_context_.native_handle(), client_context_.native_handle()}) {
      SSL_CTX_set_min_proto_version(context, TLS1_2_VERSION);
      SSL_CTX_set_session_cache_mode(context, SSL_SESS_CACHE_OFF);
      SSL_CTX_set_options(context, SSL_OP_NO_TICKET);
    }
    asio::error_code error;
    listener_.open(Tcp::v4(), error);
    if (error)
      std::abort();
    listener_.set_option(Tcp::acceptor::reuse_address(true), error);
    if (error)
      std::abort();
    listener_.bind(Tcp::endpoint(asio::ip::address_v4::loopback(), port),
                   error);
    if (error)
      std::abort();
    listener_.listen(asio::socket_base::max_listen_connections, error);
    if (error)
      std::abort();
    worker_ = std::thread([this] { io_.run(); });
  }

  ~AsioTlsHarness() {
    work_.reset();
    io_.stop();
    if (worker_.joinable())
      worker_.join();
  }

  int run(bool do_echo) {
    auto episode = std::make_shared<Episode>(this, do_echo);
    auto result = episode->result.get_future();
    asio::post(io_, [episode] { episode->start(); });
    return result.get();
  }

private:
  asio::io_context io_;
  asio::ssl::context server_context_;
  asio::ssl::context client_context_;
  Tcp::acceptor listener_;
  asio::executor_work_guard<asio::io_context::executor_type> work_;
  std::thread worker_;
};

#endif // HYLE_BENCH_ASIO_TLS

// ---- raw OpenSSL over a blocking socket (non-gating diagnostic) -------------
// Same provider/certificate/payload, but a blocking thread and an asynchronous
// scheduler are intentionally different progress and lifecycle contracts. This
// route can help localize provider/pump cost; it is not a product peer.
#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
int make_listener(std::uint16_t port) {
  int ls = static_cast<int>(::socket(AF_INET, SOCK_STREAM, 0));
  int one = 1;
  ::setsockopt(ls, SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char *>(&one), sizeof(one));
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  sa.sin_port = htons(port);
  if (::bind(ls, reinterpret_cast<sockaddr *>(&sa), sizeof(sa)) != 0 ||
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

void raw_openssl_server(int ls, SSL_CTX *sctx, bool do_echo) {
  int c = static_cast<int>(::accept(ls, nullptr, nullptr));
  if (c < 0)
    return;
  SSL *ssl = SSL_new(sctx);
  SSL_set_fd(ssl, c);
  if (SSL_accept(ssl) == 1 && do_echo) {
    char buf[kMsgLen];
    for (int i = 0; i < kEchoRoundTrips; ++i) {
      int n = SSL_read(ssl, buf, kMsgLen);
      if (n <= 0)
        break;
      if (SSL_write(ssl, buf, n) != n)
        break;
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

int run_raw_openssl_tls(std::uint16_t port, bool do_echo, SSL_CTX *sctx,
                        SSL_CTX *cctx) {
  int ls = make_listener(port);
  if (ls < 0)
    return -1;
  std::thread srv(raw_openssl_server, ls, sctx, do_echo);

  int c = static_cast<int>(::socket(AF_INET, SOCK_STREAM, 0));
  sockaddr_in sa{};
  sa.sin_family = AF_INET;
  sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  sa.sin_port = htons(port);
  int units = 0;
  if (::connect(c, reinterpret_cast<sockaddr *>(&sa), sizeof(sa)) == 0) {
    SSL *ssl = SSL_new(cctx);
    SSL_set_fd(ssl, c);
    if (SSL_connect(ssl) == 1) {
      if (!do_echo)
        units = 1;
    }
    if (SSL_is_init_finished(ssl) && do_echo) {
      static const char kMsg[kMsgLen] = "hyle-tls-bench-payload";
      char buf[kMsgLen];
      for (int i = 0; i < kEchoRoundTrips; ++i) {
        if (SSL_write(ssl, kMsg, kMsgLen) != kMsgLen)
          break;
        if (SSL_read(ssl, buf, kMsgLen) <= 0)
          break;
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
#endif // POSIX || Windows

} // namespace

int main() {
#if !defined(_WIN32)
  // A blocking SSL_write to a socket the peer already closed raises SIGPIPE,
  // which is fatal by default -- ignore it so the raw-OpenSSL baseline's
  // teardown races don't kill the bench (the write just returns EPIPE, which we
  // don't act on here).
  std::signal(SIGPIPE, SIG_IGN);
#endif
#if defined(_WIN32)
  WSADATA wsa;
  ::WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
  std::string cert = write_temp("cert.pem", kCertPem);
  std::string key = write_temp("key.pem", kKeyPem);

  // HYLE contexts (reused across iterations -- SSL_CTX is shareable; per-conn
  // SSL is made from it, matching how a real server holds one CTX).
  hyle::io::TlsServerOptions server_options;
  server_options.certificate_chain_file = cert;
  server_options.private_key_file = key;
  auto made_server = TlsContext::make_server(server_options);
  hyle::io::TlsClientOptions client_options;
  client_options.verify = hyle::io::TlsVerifyMode::kPeer;
  client_options.certificate_authority_file = cert;
  auto made_client = TlsContext::make_client(client_options);
  if (!made_server || !made_client)
    std::abort();
  TlsContext hsctx = std::move(*made_server);
  TlsContext hcctx = std::move(*made_client);

  HyleTlsHarness hyle_harness(18460, &hsctx, &hcctx);
#if defined(HYLE_BENCH_ASIO_TLS)
  AsioTlsHarness asio_harness(18461, cert, key);
#endif

  const bool raw_diagnostic = std::getenv("HYLE_TLS_RAW_DIAGNOSTIC") != nullptr;
  SSL_CTX *rsctx = nullptr;
  SSL_CTX *rcctx = nullptr;
  if (raw_diagnostic) {
    rsctx = SSL_CTX_new(TLS_server_method());
    SSL_CTX_use_certificate_chain_file(rsctx, cert.c_str());
    SSL_CTX_use_PrivateKey_file(rsctx, key.c_str(), SSL_FILETYPE_PEM);
    rcctx = SSL_CTX_new(TLS_client_method());
    SSL_CTX_set_verify(rcctx, SSL_VERIFY_NONE, nullptr);
    SSL_CTX_set_min_proto_version(rsctx, TLS1_2_VERSION);
    SSL_CTX_set_min_proto_version(rcctx, TLS1_2_VERSION);
    SSL_CTX_set_session_cache_mode(rsctx, SSL_SESS_CACHE_OFF);
    SSL_CTX_set_session_cache_mode(rcctx, SSL_SESS_CACHE_OFF);
    SSL_CTX_set_options(rsctx, SSL_OP_NO_TICKET);
    SSL_CTX_set_options(rcctx, SSL_OP_NO_TICKET);
  }

  std::uint16_t diagnostic_port = 18462;
  const bool peer_first = std::getenv("HYLE_BENCH_RIVAL_FIRST") != nullptr;
  const char *run_nonce = std::getenv("HYLE_PERF_RUN_NONCE");
  std::fprintf(stderr, "# HYLE_IO_TLS_RUN execution_order=%s run_nonce=%s\n",
               peer_first ? "peer-first" : "candidate-first",
               run_nonce != nullptr ? run_nonce : "unmanaged");

  auto warm_hyle = [&] {
    if (hyle_harness.run(/*do_echo=*/false) != 1 ||
        hyle_harness.run(/*do_echo=*/true) != kEchoRoundTrips)
      std::abort();
  };
#if defined(HYLE_BENCH_ASIO_TLS)
  auto warm_peer = [&] {
    if (asio_harness.run(/*do_echo=*/false) != 1 ||
        asio_harness.run(/*do_echo=*/true) != kEchoRoundTrips)
      std::abort();
  };
  if (peer_first) {
    warm_peer();
    warm_hyle();
  } else {
    warm_hyle();
    warm_peer();
  }
#else
  warm_hyle();
#endif

  // ---- 1. handshake/s: full connect + TLS handshake + close
  // --------------------
  {
    ankerl::nanobench::Bench b;
    b.title("TLS handshake (connect + handshake + close)")
        .relative(true)
        .unit("handshake")
        .minEpochIterations(10)
        .epochs(5);
    auto hyle = [&] {
      if (hyle_harness.run(/*do_echo=*/false) != 1)
        std::abort();
    };
#if defined(HYLE_BENCH_ASIO_TLS)
    auto peer = [&] {
      if (asio_harness.run(/*do_echo=*/false) != 1)
        std::abort();
    };
    if (peer_first) {
      b.run("Asio SSL persistent runtime handshake", peer);
      b.run("HYLE TLS persistent runtime handshake", hyle);
    } else {
      b.run("HYLE TLS persistent runtime handshake", hyle);
      b.run("Asio SSL persistent runtime handshake", peer);
    }
#else
    (void)peer_first;
    b.run("HYLE TLS persistent runtime handshake", hyle);
#endif
    if (raw_diagnostic) {
      b.run("raw OpenSSL blocking handshake [diagnostic]", [&] {
        if (run_raw_openssl_tls(diagnostic_port++, /*do_echo=*/false, rsctx,
                                rcctx) != 1)
          std::abort();
      });
    }
  }

  // ---- 2. encrypted record stream: one handshake + 512 round-trips + close
  {
    ankerl::nanobench::Bench b;
    b.title("TLS encrypted stream (512 round-trips x 256B)")
        .relative(true)
        .unit("episode")
        .minEpochIterations(1)
        .epochs(5);
    auto hyle = [&] {
      if (hyle_harness.run(/*do_echo=*/true) != kEchoRoundTrips)
        std::abort();
    };
#if defined(HYLE_BENCH_ASIO_TLS)
    auto peer = [&] {
      if (asio_harness.run(/*do_echo=*/true) != kEchoRoundTrips)
        std::abort();
    };
    if (peer_first) {
      b.run("Asio SSL persistent runtime encrypted stream", peer);
      b.run("HYLE TLS persistent runtime encrypted stream", hyle);
    } else {
      b.run("HYLE TLS persistent runtime encrypted stream", hyle);
      b.run("Asio SSL persistent runtime encrypted stream", peer);
    }
#else
    b.run("HYLE TLS persistent runtime encrypted stream", hyle);
#endif
    if (raw_diagnostic) {
      b.run("raw OpenSSL blocking encrypted stream [diagnostic]", [&] {
        if (run_raw_openssl_tls(diagnostic_port++, /*do_echo=*/true, rsctx,
                                rcctx) != kEchoRoundTrips)
          std::abort();
      });
    }
  }

  if (rsctx != nullptr)
    SSL_CTX_free(rsctx);
  if (rcctx != nullptr)
    SSL_CTX_free(rcctx);
#if defined(_WIN32)
  ::WSACleanup();
#endif
  return 0;
}

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "hyle/error/status.h"
#include "hyle/io/file.h"
#include "hyle/io/http.h"
#include "hyle/io/io_context.h"
#include "hyle/io/tcp.h"
#if defined(HYLE_IO_TLS)
#include "../support/tls_test_material.h"
#include "hyle/io/tls.h"
#endif
#include "hyle/mem/allocator.h"
#include "hyle/platform/fs.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

namespace {

using hyle::Result;
using hyle::Status;
using hyle::StatusCode;
using hyle::container::StringView;
using hyle::io::AddressFamily;
using hyle::io::Endpoint;
using hyle::io::File;
using hyle::io::FileRegion;
using hyle::io::HttpClientConnection;
using hyle::io::HttpRequestHead;
using hyle::io::HttpResponseHead;
using hyle::io::HttpServerConnection;
using hyle::io::IoContext;
using hyle::io::TcpListener;
using hyle::io::TcpStream;
#if defined(HYLE_IO_TLS)
using hyle::io::tls_handshake;
using hyle::io::TlsClientOptions;
using hyle::io::TlsContext;
using hyle::io::TlsCredentialSource;
using hyle::io::TlsServerOptions;
using hyle::io::TlsStream;
#endif
using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;
using hyle::platform::FileHandle;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::Task;

constexpr unsigned kOwner = 0;

std::span<const std::byte> no_body() noexcept { return {}; }

std::string temp_path() {
  static std::atomic<unsigned> counter{0};
  return "hyle_http_file_body_" +
         std::to_string(counter.fetch_add(1, std::memory_order_relaxed)) +
         ".tmp";
}

Result<FileHandle> make_file(const char *path,
                             std::string_view bytes) noexcept {
  auto opened = hyle::platform::open_file(
      path, hyle::platform::kOpenRead | hyle::platform::kOpenWrite |
                hyle::platform::kOpenCreate | hyle::platform::kOpenTruncate);
  if (!opened)
    return hyle::make_unexpected(std::move(opened).error());
  if (Status status = opened->write_all(bytes.data(), bytes.size());
      !status.ok())
    return hyle::make_unexpected(std::move(status));
  return std::move(*opened);
}

struct FileBodyResult {
  std::atomic<bool> server_done{false};
  std::atomic<bool> client_done{false};
  std::atomic<int> served{0};
  std::atomic<StatusCode> server_status{StatusCode::Unknown};
  std::atomic<StatusCode> client_status{StatusCode::Unknown};
  std::string body;
  std::size_t head_body_size = 1;
};

Task<int> file_server(AllocatorRef allocator, IoContext *file_io,
                      TcpListener *listener, FileHandle *file_owner,
                      FileRegion region, int requests, int response_status,
                      FileBodyResult *result) {
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted)
    co_return 0;
  auto made = HttpServerConnection::create(allocator, std::move(*accepted));
  if (!made)
    co_return 0;
  HttpServerConnection connection = std::move(*made);
  auto adopted = File::adopt(*file_io, kOwner, std::move(*file_owner));
  if (!adopted)
    co_return 0;
  File file = std::move(*adopted);

  for (int index = 0; index < requests; ++index) {
    auto request = co_await connection.read_request();
    if (!request || !request->has_value())
      break;
    if (!(co_await request->value().body.discard()).ok())
      break;
    HttpResponseHead head;
    head.status = response_status;
    Status sent = co_await connection.send_response(head, file, region);
    result->server_status.store(sent.code(), std::memory_order_release);
    if (!sent.ok())
      break;
    result->served.fetch_add(1, std::memory_order_release);
  }
  result->server_done.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> file_client(AllocatorRef allocator, IoContext *io, std::uint16_t port,
                      int requests, FileBodyResult *result) {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened)
    co_return 0;
  TcpStream stream = std::move(*opened);
  if (!(co_await stream.connect(Endpoint::loopback(AddressFamily::kIpv4, port)))
           .ok())
    co_return 0;
  auto made = HttpClientConnection::create(allocator, std::move(stream));
  if (!made)
    co_return 0;
  HttpClientConnection connection = std::move(*made);

  for (int index = 0; index < requests; ++index) {
    HttpRequestHead head;
    head.method = index == 0 ? StringView("GET") : StringView("HEAD");
    head.target = "/model-sidecar";
    head.authority = "127.0.0.1";
    auto response = co_await connection.request(head, no_body());
    if (!response) {
      result->client_status.store(response.error().code(),
                                  std::memory_order_release);
      break;
    }
    auto body = co_await response->body.read_all(1024);
    if (!body) {
      result->client_status.store(body.error().code(),
                                  std::memory_order_release);
      break;
    }
    if (index == 0)
      result->body.assign(body->data(), body->size());
    else
      result->head_body_size = body->size();
    result->client_status.store(StatusCode::Ok, std::memory_order_release);
  }
  result->client_done.store(true, std::memory_order_release);
  co_return 0;
}

struct Drive {
  Task<int> task;
};

void drive_entry(void *context) noexcept {
  static_cast<Drive *>(context)->task.resume();
}

#if defined(HYLE_IO_TLS)

constexpr std::size_t kTlsScratchBytes = 64u * 1024u;

struct FailTlsScratchAllocator {
  AllocatorRef backing = system_allocator();
  std::atomic<bool> armed{false};
};

void *selective_allocate(void *opaque, std::size_t size,
                         std::size_t alignment) noexcept {
  auto &state = *static_cast<FailTlsScratchAllocator *>(opaque);
  if (state.armed.load(std::memory_order_acquire) && size == kTlsScratchBytes)
    return nullptr;
  return state.backing.allocate(size, alignment);
}

void selective_deallocate(void *opaque, void *address, std::size_t size,
                          std::size_t alignment) noexcept {
  static_cast<FailTlsScratchAllocator *>(opaque)->backing.deallocate(
      address, size, alignment);
}

void *selective_reallocate(void *opaque, void *address, std::size_t old_size,
                           std::size_t new_size,
                           std::size_t alignment) noexcept {
  auto &state = *static_cast<FailTlsScratchAllocator *>(opaque);
  if (state.armed.load(std::memory_order_acquire) &&
      new_size == kTlsScratchBytes)
    return nullptr;
  return state.backing.reallocate(address, old_size, new_size, alignment);
}

AllocatorRef selective_allocator(FailTlsScratchAllocator &state) noexcept {
  return AllocatorRef(&selective_allocate, &selective_deallocate,
                      &selective_reallocate, &state);
}

Task<int> tls_file_server(AllocatorRef allocator, IoContext *io,
                          TcpListener *listener, TlsContext *context,
                          FileHandle *file_owner, FileRegion region,
                          FailTlsScratchAllocator *failure,
                          FileBodyResult *result) {
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted)
    co_return 0;
  auto wrapped = TlsStream::accept(std::move(*accepted), *context);
  if (!wrapped)
    co_return 0;
  TlsStream tls = std::move(*wrapped);
  if (!(co_await tls_handshake(allocator, tls)).ok())
    co_return 0;
  auto made = HttpServerConnection::create(allocator, std::move(tls));
  if (!made)
    co_return 0;
  HttpServerConnection connection = std::move(*made);
  auto adopted = File::adopt(*io, kOwner, std::move(*file_owner));
  if (!adopted)
    co_return 0;
  File file = std::move(*adopted);
  auto request = co_await connection.read_request();
  if (!request || !request->has_value() ||
      !(co_await request->value().body.discard()).ok())
    co_return 0;

  if (failure != nullptr)
    failure->armed.store(true, std::memory_order_release);
  Status sent = co_await connection.send_response({}, file, region);
  result->server_status.store(sent.code(), std::memory_order_release);
  if (sent.ok())
    result->served.store(1, std::memory_order_release);
  result->server_done.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> tls_file_client(AllocatorRef allocator, IoContext *io,
                          std::uint16_t port, TlsContext *context,
                          FileBodyResult *result) {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened)
    co_return 0;
  TcpStream stream = std::move(*opened);
  if (!(co_await stream.connect(Endpoint::loopback(AddressFamily::kIpv4, port)))
           .ok())
    co_return 0;
  auto wrapped = TlsStream::connect(std::move(stream), *context);
  if (!wrapped)
    co_return 0;
  TlsStream tls = std::move(*wrapped);
  if (!(co_await tls_handshake(allocator, tls)).ok())
    co_return 0;
  auto made = HttpClientConnection::create(allocator, std::move(tls));
  if (!made)
    co_return 0;
  HttpClientConnection connection = std::move(*made);
  HttpRequestHead head;
  head.authority = "localhost";
  auto response = co_await connection.request(head, no_body());
  if (!response) {
    result->client_status.store(response.error().code(),
                                std::memory_order_release);
    result->client_done.store(true, std::memory_order_release);
    co_return 0;
  }
  auto body = co_await response->body.read_all(1024);
  if (!body) {
    result->client_status.store(body.error().code(), std::memory_order_release);
  } else {
    result->body.assign(body->data(), body->size());
    result->client_status.store(StatusCode::Ok, std::memory_order_release);
  }
  result->client_done.store(true, std::memory_order_release);
  co_return 0;
}

void run_tls_file_body_case(std::uint16_t port, FileHandle &file,
                            FileRegion region, bool fail_scratch,
                            FileBodyResult &result,
                            bool foreign_file_context = false) {
  TlsServerOptions server_options;
  server_options.certificate_chain =
      TlsCredentialSource::pem(hyle::test_support::kTlsCertificatePem);
  server_options.private_key =
      TlsCredentialSource::pem(hyle::test_support::kTlsPrivateKeyPem);
  auto made_server = TlsContext::make_server(server_options);
  ASSERT_TRUE(made_server) << made_server.error().message();
  TlsContext server_context = std::move(*made_server);
  TlsClientOptions client_options;
  client_options.verify = hyle::io::TlsVerifyMode::kNone;
  auto made_client = TlsContext::make_client(client_options);
  ASSERT_TRUE(made_client) << made_client.error().message();
  TlsContext client_context = std::move(*made_client);

  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid()) << io.status().message();
  std::unique_ptr<Scheduler> foreign_scheduler;
  std::unique_ptr<IoContext> foreign_io;
  if (foreign_file_context) {
    foreign_scheduler = std::make_unique<Scheduler>(1, system_allocator(), 64);
    foreign_io = std::make_unique<IoContext>(*foreign_scheduler);
    ASSERT_TRUE(foreign_io->valid()) << foreign_io->status().message();
  }
  auto bound = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, port));
  ASSERT_TRUE(bound) << bound.error().message();
  TcpListener listener = std::move(*bound);
  FailTlsScratchAllocator failure;
  AllocatorRef server_allocator = selective_allocator(failure);
  Drive server{tls_file_server(server_allocator,
                               foreign_io ? foreign_io.get() : &io, &listener,
                               &server_context, &file, region,
                               fail_scratch ? &failure : nullptr, &result)};
  Drive client{
      tls_file_client(system_allocator(), &io, port, &client_context, &result)};
  Fiber server_fiber(kDefaultStackSize, &drive_entry, &server);
  Fiber client_fiber(kDefaultStackSize, &drive_entry, &client);
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();
  if (foreign_scheduler)
    foreign_scheduler->shutdown();
}

#endif

void run_file_body_case(std::uint16_t port, FileHandle &file, FileRegion region,
                        int requests, FileBodyResult &result,
                        bool foreign_file_context = false,
                        int response_status = 200) {
  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid()) << io.status().message();
  std::unique_ptr<Scheduler> foreign_scheduler;
  std::unique_ptr<IoContext> foreign_io;
  if (foreign_file_context) {
    foreign_scheduler = std::make_unique<Scheduler>(1, system_allocator(), 64);
    foreign_io = std::make_unique<IoContext>(*foreign_scheduler);
    ASSERT_TRUE(foreign_io->valid()) << foreign_io->status().message();
  }
  auto bound = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, port));
  ASSERT_TRUE(bound) << bound.error().message();
  TcpListener listener = std::move(*bound);

  Drive server{file_server(system_allocator(),
                           foreign_io ? foreign_io.get() : &io, &listener,
                           &file, region, requests, response_status, &result)};
  Drive client{file_client(system_allocator(), &io, port, requests, &result)};
  Fiber server_fiber(kDefaultStackSize, &drive_entry, &server);
  Fiber client_fiber(kDefaultStackSize, &drive_entry, &client);
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();
  if (foreign_scheduler)
    foreign_scheduler->shutdown();
}

TEST(IoHttpFileBody, SendsRegionAndReusesConnectionForHead) {
  const std::string path = temp_path();
  auto created = make_file(path.c_str(), "prefix-model-bytes-suffix");
  ASSERT_TRUE(created) << created.error().message();
  FileHandle file = std::move(*created);
  FileBodyResult result;
  run_file_body_case(54582, file, FileRegion{7, 11}, 2, result);
  std::remove(path.c_str());

  EXPECT_TRUE(result.server_done.load(std::memory_order_acquire));
  EXPECT_TRUE(result.client_done.load(std::memory_order_acquire));
  EXPECT_EQ(result.served.load(std::memory_order_acquire), 2);
  EXPECT_EQ(result.server_status.load(std::memory_order_acquire),
            StatusCode::Ok);
  EXPECT_EQ(result.client_status.load(std::memory_order_acquire),
            StatusCode::Ok);
  EXPECT_EQ(result.body, "model-bytes");
  EXPECT_EQ(result.head_body_size, 0u);
}

TEST(IoHttpFileBody, ShortFileAbandonsConnection) {
  const std::string path = temp_path();
  auto created = make_file(path.c_str(), "short");
  ASSERT_TRUE(created) << created.error().message();
  FileHandle file = std::move(*created);
  FileBodyResult result;
  run_file_body_case(54583, file, FileRegion{0, 9}, 1, result);
  std::remove(path.c_str());

  EXPECT_TRUE(result.server_done.load(std::memory_order_acquire));
  EXPECT_TRUE(result.client_done.load(std::memory_order_acquire));
  EXPECT_EQ(result.served.load(std::memory_order_acquire), 0);
  EXPECT_EQ(result.server_status.load(std::memory_order_acquire),
            StatusCode::DataLoss);
  EXPECT_EQ(result.client_status.load(std::memory_order_acquire),
            StatusCode::DataLoss);
}

TEST(IoHttpFileBody, RejectsDifferentIoContextBeforeResponse) {
  const std::string path = temp_path();
  auto created = make_file(path.c_str(), "context");
  ASSERT_TRUE(created) << created.error().message();
  FileHandle file = std::move(*created);
  FileBodyResult result;
  run_file_body_case(54585, file, FileRegion{0, 7}, 1, result, true);
  std::remove(path.c_str());

  EXPECT_TRUE(result.server_done.load(std::memory_order_acquire));
  EXPECT_TRUE(result.client_done.load(std::memory_order_acquire));
  EXPECT_EQ(result.served.load(std::memory_order_acquire), 0);
  EXPECT_EQ(result.server_status.load(std::memory_order_acquire),
            StatusCode::InvalidArgument);
  EXPECT_NE(result.client_status.load(std::memory_order_acquire),
            StatusCode::Ok);
}

TEST(IoHttpFileBody, BodylessStatusesNeverReadTheFile) {
  static constexpr int kStatuses[] = {204, 205, 304};
  for (std::size_t index = 0; index < std::size(kStatuses); ++index) {
    const std::string path = temp_path();
    auto created = make_file(path.c_str(), "x");
    ASSERT_TRUE(created) << created.error().message();
    FileHandle file = std::move(*created);
    FileBodyResult result;
    const FileRegion region =
        kStatuses[index] == 304 ? FileRegion{1024, 7} : FileRegion{0, 0};
    run_file_body_case(static_cast<std::uint16_t>(54590 + index), file, region,
                       1, result, false, kStatuses[index]);
    std::remove(path.c_str());

    EXPECT_TRUE(result.server_done.load(std::memory_order_acquire));
    EXPECT_TRUE(result.client_done.load(std::memory_order_acquire));
    EXPECT_EQ(result.served.load(std::memory_order_acquire), 1);
    EXPECT_EQ(result.server_status.load(std::memory_order_acquire),
              StatusCode::Ok);
    EXPECT_EQ(result.client_status.load(std::memory_order_acquire),
              StatusCode::Ok);
    EXPECT_TRUE(result.body.empty());
  }
}

#if defined(HYLE_IO_TLS)

TEST(IoHttpFileBody, SendsRegionOverTlsWithOneBoundedScratch) {
  const std::string path = temp_path();
  auto created = make_file(path.c_str(), "prefix-tls-model-bytes-suffix");
  ASSERT_TRUE(created) << created.error().message();
  FileHandle file = std::move(*created);
  FileBodyResult result;
  run_tls_file_body_case(54588, file, FileRegion{7, 15}, false, result);
  std::remove(path.c_str());

  EXPECT_TRUE(result.server_done.load(std::memory_order_acquire));
  EXPECT_TRUE(result.client_done.load(std::memory_order_acquire));
  EXPECT_EQ(result.served.load(std::memory_order_acquire), 1);
  EXPECT_EQ(result.server_status.load(std::memory_order_acquire),
            StatusCode::Ok);
  EXPECT_EQ(result.client_status.load(std::memory_order_acquire),
            StatusCode::Ok);
  EXPECT_EQ(result.body, "tls-model-bytes");
}

TEST(IoHttpFileBody, TlsScratchOomIsTypedAndAbandonsResponse) {
  const std::string path = temp_path();
  auto created = make_file(path.c_str(), "tls-oom");
  ASSERT_TRUE(created) << created.error().message();
  FileHandle file = std::move(*created);
  FileBodyResult result;
  run_tls_file_body_case(54589, file, FileRegion{0, 7}, true, result);
  std::remove(path.c_str());

  EXPECT_TRUE(result.server_done.load(std::memory_order_acquire));
  EXPECT_TRUE(result.client_done.load(std::memory_order_acquire));
  EXPECT_EQ(result.served.load(std::memory_order_acquire), 0);
  EXPECT_EQ(result.server_status.load(std::memory_order_acquire),
            StatusCode::ResourceExhausted);
  EXPECT_EQ(result.client_status.load(std::memory_order_acquire),
            StatusCode::DataLoss);
}

TEST(IoHttpFileBody, TlsRejectsDifferentIoContextBeforeResponse) {
  const std::string path = temp_path();
  auto created = make_file(path.c_str(), "tls-context");
  ASSERT_TRUE(created) << created.error().message();
  FileHandle file = std::move(*created);
  FileBodyResult result;
  run_tls_file_body_case(54594, file, FileRegion{0, 11}, false, result, true);
  std::remove(path.c_str());

  EXPECT_TRUE(result.server_done.load(std::memory_order_acquire));
  EXPECT_TRUE(result.client_done.load(std::memory_order_acquire));
  EXPECT_EQ(result.served.load(std::memory_order_acquire), 0);
  EXPECT_EQ(result.server_status.load(std::memory_order_acquire),
            StatusCode::InvalidArgument);
  EXPECT_NE(result.client_status.load(std::memory_order_acquire),
            StatusCode::Ok);
}

#endif

} // namespace

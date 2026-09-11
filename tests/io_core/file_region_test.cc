#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#else
#include <sys/socket.h>
#endif
#if defined(__linux__)
#include <csignal>
#endif

#include "hyle/error/status.h"
#include "hyle/io/file.h"
#include "hyle/io/io_context.h"
#include "hyle/io/tcp.h"
#include "hyle/mem/allocator.h"
#include "hyle/platform/fs.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

namespace {

using hyle::Result;
using hyle::Status;
using hyle::StatusCode;
using hyle::io::AddressFamily;
using hyle::io::Endpoint;
using hyle::io::File;
using hyle::io::FileRegion;
using hyle::io::IoContext;
using hyle::io::OperationOptions;
using hyle::io::read_exact;
using hyle::io::send_file_region;
using hyle::io::TcpListener;
using hyle::io::TcpStream;
using hyle::io::TransferResult;
using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;
using hyle::platform::FileHandle;
using hyle::sched::CancelToken;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::Task;

constexpr unsigned kOwner = 0;
constexpr std::uint64_t kPeerCloseBytes = 8u * 1024u * 1024u;

#if defined(__linux__)
volatile std::sig_atomic_t g_file_region_sigpipe = 0;

extern "C" void count_file_region_sigpipe(int) {
  g_file_region_sigpipe =
      static_cast<std::sig_atomic_t>(g_file_region_sigpipe + 1);
}
#endif

std::string temp_path() {
  static std::atomic<unsigned> counter{0};
  return "hyle_file_region_" +
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
  if (hyle::Status status = opened->write_all(bytes.data(), bytes.size());
      !status.ok())
    return hyle::make_unexpected(std::move(status));
  return std::move(*opened);
}

struct RegionResult {
  std::atomic<bool> server_done{false};
  std::atomic<bool> client_done{false};
  std::atomic<bool> zero_ok{false};
  std::atomic<bool> overflow_rejected{false};
  std::atomic<bool> cancellation_rejected{false};
  std::atomic<bool> deadline_rejected{false};
  std::atomic<StatusCode> send_status{StatusCode::Unknown};
  std::atomic<std::int64_t> native_error{0};
  std::atomic<std::size_t> sent{0};
  std::atomic<std::size_t> received{0};
  std::array<std::byte, 64> bytes{};
};

Task<int> region_server(AllocatorRef allocator, IoContext *file_io,
                        TcpListener *listener, FileHandle *file_owner,
                        FileRegion region, RegionResult *result,
                        bool preflight) {
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted)
    co_return 0;
  TcpStream stream = std::move(*accepted);
  auto adopted = File::adopt(*file_io, kOwner, std::move(*file_owner));
  if (!adopted)
    co_return 0;
  File file = std::move(*adopted);

  if (preflight) {
    TransferResult zero = co_await send_file_region(
        allocator, stream, file, FileRegion{region.offset, 0});
    result->zero_ok.store(zero.ok() && zero.bytes == 0,
                          std::memory_order_release);

    TransferResult overflow = co_await send_file_region(
        allocator, stream, file,
        FileRegion{(std::numeric_limits<std::uint64_t>::max)(), 1});
    result->overflow_rejected.store(overflow.status.IsOutOfRange(),
                                    std::memory_order_release);

    CancelToken cancelled = CancelToken::root(allocator);
    cancelled.cancel();
    OperationOptions cancel_options;
    cancel_options.cancel = &cancelled;
    TransferResult stopped = co_await send_file_region(allocator, stream, file,
                                                       region, cancel_options);
    result->cancellation_rejected.store(stopped.status.IsCancelled() &&
                                            stopped.bytes == 0,
                                        std::memory_order_release);

    TransferResult expired = co_await send_file_region(
        allocator, stream, file, region,
        OperationOptions{.deadline = hyle::time::monotonic_now()});
    result->deadline_rejected.store(expired.status.IsDeadlineExceeded() &&
                                        expired.bytes == 0,
                                    std::memory_order_release);
  }

  TransferResult sent =
      co_await send_file_region(allocator, stream, file, region);
  result->send_status.store(sent.status.code(), std::memory_order_release);
  result->native_error.store(sent.status.native_error().code,
                             std::memory_order_release);
  result->sent.store(sent.bytes, std::memory_order_release);
  result->server_done.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> region_client(AllocatorRef allocator, IoContext *io,
                        std::uint16_t port, std::size_t expected,
                        RegionResult *result) {
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened)
    co_return 0;
  TcpStream stream = std::move(*opened);
  if (!(co_await stream.connect(Endpoint::loopback(AddressFamily::kIpv4, port)))
           .ok())
    co_return 0;
  TransferResult received = co_await read_exact(
      allocator, stream, std::span<std::byte>(result->bytes.data(), expected));
  result->received.store(received.bytes, std::memory_order_release);
  result->client_done.store(true, std::memory_order_release);
  co_return 0;
}

struct Drive {
  Task<int> task;
};

void drive_entry(void *context) noexcept {
  static_cast<Drive *>(context)->task.resume();
}

void run_region_case(std::uint16_t port, FileHandle &file, FileRegion region,
                     std::size_t expected_receive, RegionResult &result,
                     bool preflight = false,
                     bool foreign_file_context = false) {
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

  Drive server{region_server(system_allocator(),
                             foreign_io ? foreign_io.get() : &io, &listener,
                             &file, region, &result, preflight)};
  Drive client{
      region_client(system_allocator(), &io, port, expected_receive, &result)};
  Fiber server_fiber(kDefaultStackSize, &drive_entry, &server);
  Fiber client_fiber(kDefaultStackSize, &drive_entry, &client);
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();
  if (foreign_scheduler)
    foreign_scheduler->shutdown();
}

struct PeerCloseResult {
  std::atomic<bool> peer_closed{false};
  std::atomic<bool> server_done{false};
  std::atomic<StatusCode> status{StatusCode::Unknown};
  std::atomic<std::size_t> sent{0};
};

struct CancelInFlightResult {
  std::atomic<bool> connected{false};
  std::atomic<bool> send_started{false};
  std::atomic<bool> server_done{false};
  std::atomic<StatusCode> status{StatusCode::Unknown};
  std::atomic<std::size_t> sent{0};
  TcpStream held;
};

Task<int> cancel_in_flight_server(AllocatorRef allocator, IoContext *io,
                                  TcpListener *listener, FileHandle *file_owner,
                                  CancelToken *cancel,
                                  CancelInFlightResult *result) {
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted)
    co_return 0;
  auto adopted = File::adopt(*io, kOwner, std::move(*file_owner));
  if (!adopted)
    co_return 0;
  File file = std::move(*adopted);
  result->send_started.store(true, std::memory_order_release);
  OperationOptions options;
  options.cancel = cancel;
  TransferResult sent = co_await send_file_region(
      allocator, *accepted, file, FileRegion{0, kPeerCloseBytes}, options);
  result->status.store(sent.status.code(), std::memory_order_release);
  result->sent.store(sent.bytes, std::memory_order_release);
  result->server_done.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> cancel_in_flight_client(AllocatorRef allocator, IoContext *io,
                                  std::uint16_t port,
                                  CancelInFlightResult *result) {
  (void)allocator;
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened)
    co_return 0;
  TcpStream stream = std::move(*opened);
  if (!(co_await stream.connect(Endpoint::loopback(AddressFamily::kIpv4, port)))
           .ok())
    co_return 0;
  result->held = std::move(stream);
  result->connected.store(true, std::memory_order_release);
  co_return 0;
}

void reset_connection_on_close(TcpStream &stream) noexcept {
  linger reset{1, 0};
#if defined(_WIN32)
  (void)::setsockopt(static_cast<SOCKET>(stream.native_handle()), SOL_SOCKET,
                     SO_LINGER, reinterpret_cast<const char *>(&reset),
                     sizeof(reset));
#else
  (void)::setsockopt(static_cast<int>(stream.native_handle()), SOL_SOCKET,
                     SO_LINGER, &reset, sizeof(reset));
#endif
}

Task<int> peer_close_server(AllocatorRef allocator, IoContext *io,
                            TcpListener *listener, FileHandle *file_owner,
                            PeerCloseResult *result) {
  auto accepted = co_await listener->accept(kOwner);
  if (!accepted)
    co_return 0;
  auto adopted = File::adopt(*io, kOwner, std::move(*file_owner));
  if (!adopted)
    co_return 0;
  File file = std::move(*adopted);
  while (!result->peer_closed.load(std::memory_order_acquire))
    Fiber::yield();

  TransferResult sent = co_await send_file_region(
      allocator, *accepted, file, FileRegion{0, kPeerCloseBytes});
  result->status.store(sent.status.code(), std::memory_order_release);
  result->sent.store(sent.bytes, std::memory_order_release);
  result->server_done.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> peer_close_client(AllocatorRef allocator, IoContext *io,
                            std::uint16_t port, PeerCloseResult *result) {
  (void)allocator;
  auto opened = TcpStream::open(*io, kOwner);
  if (!opened) {
    result->peer_closed.store(true, std::memory_order_release);
    co_return 0;
  }
  {
    TcpStream stream = std::move(*opened);
    Status connected =
        co_await stream.connect(Endpoint::loopback(AddressFamily::kIpv4, port));
    if (connected.ok())
      reset_connection_on_close(stream);
  }
  result->peer_closed.store(true, std::memory_order_release);
  co_return 0;
}

TEST(IoFileRegion, SendsExplicitRegionAndKeepsPreflightFailuresOffWire) {
  const std::string path = temp_path();
  auto created = make_file(path.c_str(), "prefix-model-bytes-suffix");
  ASSERT_TRUE(created) << created.error().message();
  FileHandle file = std::move(*created);
  RegionResult result;
  run_region_case(54580, file, FileRegion{7, 11}, 11, result, true);
  std::remove(path.c_str());

  EXPECT_TRUE(result.server_done.load(std::memory_order_acquire));
  EXPECT_TRUE(result.client_done.load(std::memory_order_acquire));
  EXPECT_TRUE(result.zero_ok.load(std::memory_order_acquire));
  EXPECT_TRUE(result.overflow_rejected.load(std::memory_order_acquire));
  EXPECT_TRUE(result.cancellation_rejected.load(std::memory_order_acquire));
  EXPECT_TRUE(result.deadline_rejected.load(std::memory_order_acquire));
  EXPECT_EQ(result.send_status.load(std::memory_order_acquire), StatusCode::Ok);
  EXPECT_EQ(result.sent.load(std::memory_order_acquire), 11u);
  EXPECT_EQ(result.received.load(std::memory_order_acquire), 11u);
  EXPECT_EQ(
      std::string_view(reinterpret_cast<const char *>(result.bytes.data()), 11),
      "model-bytes");
}

TEST(IoFileRegion, RejectsDifferentIoContextBeforeWriting) {
  const std::string path = temp_path();
  auto created = make_file(path.c_str(), "context");
  ASSERT_TRUE(created) << created.error().message();
  FileHandle file = std::move(*created);
  RegionResult result;
  run_region_case(54584, file, FileRegion{0, 7}, 0, result, false, true);
  std::remove(path.c_str());

  EXPECT_TRUE(result.server_done.load(std::memory_order_acquire));
  EXPECT_TRUE(result.client_done.load(std::memory_order_acquire));
  EXPECT_EQ(result.send_status.load(std::memory_order_acquire),
            StatusCode::InvalidArgument);
  EXPECT_EQ(result.sent.load(std::memory_order_acquire), 0u);
  EXPECT_EQ(result.received.load(std::memory_order_acquire), 0u);
}

TEST(IoFileRegion, ReportsPartialProgressWhenRegionCrossesEof) {
  const std::string path = temp_path();
  auto created = make_file(path.c_str(), "short");
  ASSERT_TRUE(created) << created.error().message();
  FileHandle file = std::move(*created);
  RegionResult result;
  run_region_case(54581, file, FileRegion{0, 9}, 9, result);
  std::remove(path.c_str());

  EXPECT_TRUE(result.server_done.load(std::memory_order_acquire));
  EXPECT_TRUE(result.client_done.load(std::memory_order_acquire));
  EXPECT_EQ(result.send_status.load(std::memory_order_acquire),
            StatusCode::DataLoss)
      << "native error=" << result.native_error.load(std::memory_order_acquire);
  EXPECT_EQ(result.sent.load(std::memory_order_acquire), 5u);
  EXPECT_EQ(result.received.load(std::memory_order_acquire), 5u);
  EXPECT_EQ(
      std::string_view(reinterpret_cast<const char *>(result.bytes.data()), 5),
      "short");
}

TEST(IoFileRegion, PeerResetReturnsTypedFailureWithoutProcessSignal) {
  const std::string path = temp_path();
  auto created = make_file(path.c_str(), "x");
  ASSERT_TRUE(created) << created.error().message();
  ASSERT_TRUE(created->resize(kPeerCloseBytes).ok());
  FileHandle file = std::move(*created);

#if defined(__linux__)
  struct sigaction replacement{};
  struct sigaction previous{};
  replacement.sa_handler = &count_file_region_sigpipe;
  ::sigemptyset(&replacement.sa_mask);
  ASSERT_EQ(::sigaction(SIGPIPE, &replacement, &previous), 0);
  g_file_region_sigpipe = 0;
#endif

  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid()) << io.status().message();
  auto bound = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, 54586));
  ASSERT_TRUE(bound) << bound.error().message();
  TcpListener listener = std::move(*bound);
  PeerCloseResult result;
  Drive server{
      peer_close_server(system_allocator(), &io, &listener, &file, &result)};
  Drive client{peer_close_client(system_allocator(), &io, 54586, &result)};
  Fiber server_fiber(kDefaultStackSize, &drive_entry, &server);
  Fiber client_fiber(kDefaultStackSize, &drive_entry, &client);
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();
  std::remove(path.c_str());

#if defined(__linux__)
  EXPECT_EQ(::sigaction(SIGPIPE, &previous, nullptr), 0);
  EXPECT_EQ(g_file_region_sigpipe, 0);
#endif
  EXPECT_TRUE(result.server_done.load(std::memory_order_acquire));
  EXPECT_NE(result.status.load(std::memory_order_acquire), StatusCode::Ok);
  EXPECT_LT(result.sent.load(std::memory_order_acquire), kPeerCloseBytes);
}

TEST(IoFileRegion, CancellationRetiresAnInFlightNativeSend) {
  const std::string path = temp_path();
  auto created = make_file(path.c_str(), "x");
  ASSERT_TRUE(created) << created.error().message();
  ASSERT_TRUE(created->resize(kPeerCloseBytes).ok());
  FileHandle file = std::move(*created);
  auto made_cancel = CancelToken::try_create_root(system_allocator());
  ASSERT_TRUE(made_cancel) << made_cancel.error().message();
  CancelToken cancel = std::move(*made_cancel);

  Scheduler scheduler(2, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid()) << io.status().message();
  auto bound = TcpListener::bind(
      io, kOwner, Endpoint::loopback(AddressFamily::kIpv4, 54587));
  ASSERT_TRUE(bound) << bound.error().message();
  TcpListener listener = std::move(*bound);
  CancelInFlightResult result;
  Drive server{cancel_in_flight_server(system_allocator(), &io, &listener,
                                       &file, &cancel, &result)};
  Drive client{
      cancel_in_flight_client(system_allocator(), &io, 54587, &result)};
  Fiber server_fiber(kDefaultStackSize, &drive_entry, &server);
  Fiber client_fiber(kDefaultStackSize, &drive_entry, &client);
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  std::thread canceller([&] {
    while (!result.send_started.load(std::memory_order_acquire))
      std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    cancel.cancel();
  });
  scheduler.drain();
  canceller.join();
  scheduler.shutdown();
  std::remove(path.c_str());

  EXPECT_TRUE(result.connected.load(std::memory_order_acquire));
  EXPECT_TRUE(result.server_done.load(std::memory_order_acquire));
  EXPECT_EQ(result.status.load(std::memory_order_acquire),
            StatusCode::Cancelled);
  EXPECT_LT(result.sent.load(std::memory_order_acquire), kPeerCloseBytes);
}

} // namespace

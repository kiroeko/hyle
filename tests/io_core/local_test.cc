#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <string>

#include "hyle/io/local.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"
#include "hyle/time/clock.h"

#if !defined(_WIN32)
#  include <sys/stat.h>
#endif

using hyle::io::ConstBuffer;
using hyle::io::IoContext;
using hyle::io::LocalEndpoint;
using hyle::io::LocalListenOptions;
using hyle::io::LocalListener;
using hyle::io::LocalStream;
using hyle::io::MutableBuffer;
using hyle::io::read_exact;
using hyle::io::write_all;
using hyle::mem::AllocatorRef;
using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::Scheduler;
using hyle::sched::Task;
using hyle::sched::kDefaultStackSize;

namespace {

constexpr unsigned kOwner = 0;

std::string unique_local_name() {
  static std::atomic<unsigned> sequence{0};
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
#if defined(_WIN32)
  return "hyle-local-test-" + std::to_string(stamp) + "-" +
         std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
#else
  return "hyle-local-test-" + std::to_string(stamp) + "-" +
         std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)) +
         ".sock";
#endif
}

hyle::Result<LocalEndpoint> native_endpoint(std::string_view name) {
#if defined(_WIN32)
  return LocalEndpoint::windows_named_pipe(name);
#else
  return LocalEndpoint::unix_path(name);
#endif
}

struct RoundTripState {
  std::atomic<bool> server_done{false};
  std::atomic<bool> client_done{false};
  std::atomic<bool> server_ok{false};
  std::atomic<bool> client_ok{false};
  char received[64]{};
};

Task<int> local_server(AllocatorRef allocator, LocalListener* listener,
                       RoundTripState* state) {
  auto accepted = co_await listener->accept();
  if (!accepted.has_value()) {
    state->server_done.store(true, std::memory_order_release);
    co_return 0;
  }
  LocalStream stream = std::move(*accepted);
  char input[64]{};
  MutableBuffer incoming[] = {{input, 5}, {input + 5, 12}};
  auto read = co_await read_exact(
      allocator, stream, std::span<const MutableBuffer>{incoming});
  if (read.ok()) {
    ConstBuffer outgoing[] = {{input, 7}, {input + 7, 10}};
    auto written = co_await write_all(
        allocator, stream, std::span<const ConstBuffer>{outgoing});
    state->server_ok.store(written.ok() && written.progress.bytes == 17,
                           std::memory_order_release);
  }
  state->server_done.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> local_client(AllocatorRef allocator, IoContext* io,
                       const LocalEndpoint* endpoint, RoundTripState* state) {
  auto connected = co_await LocalStream::connect(*io, kOwner, *endpoint);
  if (!connected.has_value()) {
    state->client_done.store(true, std::memory_order_release);
    co_return 0;
  }
  LocalStream stream = std::move(*connected);
  static constexpr char kMessage[] = "local-vector-echo";
  ConstBuffer outgoing[] = {{kMessage, 4}, {kMessage + 4, 13}};
  auto written = co_await write_all(
      allocator, stream, std::span<const ConstBuffer>{outgoing});
  MutableBuffer incoming[] = {{state->received, 3},
                              {state->received + 3, 14}};
  auto read = co_await read_exact(
      allocator, stream, std::span<const MutableBuffer>{incoming});
  state->client_ok.store(
      written.ok() && written.progress.bytes == 17 && read.ok() &&
          read.progress.bytes == 17 &&
          std::memcmp(state->received, kMessage, 17) == 0,
      std::memory_order_release);
  state->client_done.store(true, std::memory_order_release);
  co_return 0;
}

struct Drive {
  Task<int> task;
};

void drive_entry(void* raw) noexcept { static_cast<Drive*>(raw)->task.resume(); }

struct FailureState {
  std::atomic<bool> done{false};
  std::atomic<bool> expected{false};
  std::atomic<bool> retired{false};
};

Task<int> deadline_accept(AllocatorRef, IoContext* io,
                          LocalListener* listener, FailureState* state) {
  hyle::io::OperationOptions options;
  options.deadline =
      hyle::time::deadline_after(std::chrono::milliseconds(20));
  auto accepted = co_await listener->accept(options);
  state->expected.store(!accepted.has_value() &&
                            accepted.error().IsDeadlineExceeded(),
                        std::memory_order_release);
  state->retired.store(!io->proactor_for_current().in_flight(),
                       std::memory_order_release);
  state->done.store(true, std::memory_order_release);
  co_return 0;
}

Task<int> cancelled_connect(AllocatorRef, IoContext* io,
                            const LocalEndpoint* endpoint,
                            const hyle::sched::CancelToken* token,
                            FailureState* state) {
  hyle::io::OperationOptions options;
  options.cancel = token;
  auto connected =
      co_await LocalStream::connect(*io, kOwner, *endpoint, options);
  state->expected.store(!connected.has_value() &&
                            connected.error().IsCancelled(),
                        std::memory_order_release);
  state->retired.store(!io->proactor_for_current().in_flight(),
                       std::memory_order_release);
  state->done.store(true, std::memory_order_release);
  co_return 0;
}

} // namespace

TEST(IoLocal, EndpointFactoriesOwnAndRejectInvalidNames) {
  std::string source = "owned-endpoint";
  auto endpoint = LocalEndpoint::unix_path(source);
  ASSERT_TRUE(endpoint.has_value());
  source.assign("changed");
  EXPECT_EQ(endpoint->name(), "owned-endpoint");
  EXPECT_FALSE(LocalEndpoint::unix_path("").has_value());
  EXPECT_FALSE(LocalEndpoint::unix_path(std::string_view{"bad\0name", 8})
                   .has_value());
}

TEST(IoLocal, VectorRoundTripUsesNativeLocalTransport) {
  Scheduler scheduler(1, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid()) << io.status().message();

  const std::string name = unique_local_name();
  auto endpoint = native_endpoint(name);
  ASSERT_TRUE(endpoint.has_value());
  LocalListenOptions options;
#if !defined(_WIN32)
  options.replace_existing_socket = true;
#endif
  auto bound = LocalListener::bind(io, kOwner, *endpoint, options);
  ASSERT_TRUE(bound.has_value())
      << (bound.has_value() ? "" : bound.error().message());
  LocalListener listener = std::move(*bound);
#if !defined(_WIN32)
  struct stat socket_info {};
  ASSERT_EQ(::lstat(name.c_str(), &socket_info), 0);
  EXPECT_EQ(socket_info.st_mode & 0777, options.unix_permissions);
#endif

  RoundTripState state;
  Drive server{local_server(system_allocator(), &listener, &state)};
  Drive client{local_client(system_allocator(), &io, &*endpoint, &state)};
  Fiber server_fiber(kDefaultStackSize, &drive_entry, &server);
  Fiber client_fiber(kDefaultStackSize, &drive_entry, &client);
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();

  EXPECT_TRUE(state.server_done.load(std::memory_order_acquire));
  EXPECT_TRUE(state.client_done.load(std::memory_order_acquire));
  EXPECT_TRUE(state.server_ok.load(std::memory_order_acquire));
  EXPECT_TRUE(state.client_ok.load(std::memory_order_acquire));
  EXPECT_TRUE(listener.close().ok());
#if !defined(_WIN32)
  EXPECT_EQ(std::fopen(name.c_str(), "rb"), nullptr);
#endif
}

#if defined(__linux__)
TEST(IoLocal, LinuxAbstractEndpointHasNoFilesystemLifetime) {
  Scheduler scheduler(1, system_allocator(), 256);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  const std::string name = unique_local_name();
  auto endpoint = LocalEndpoint::linux_abstract(name);
  ASSERT_TRUE(endpoint.has_value());
  auto bound = LocalListener::bind(io, kOwner, *endpoint);
  ASSERT_TRUE(bound.has_value())
      << (bound.has_value() ? "" : bound.error().message());
  LocalListener listener = std::move(*bound);
  RoundTripState state;
  Drive server{local_server(system_allocator(), &listener, &state)};
  Drive client{local_client(system_allocator(), &io, &*endpoint, &state)};
  Fiber server_fiber(kDefaultStackSize, &drive_entry, &server);
  Fiber client_fiber(kDefaultStackSize, &drive_entry, &client);
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &server_fiber));
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &client_fiber));
  scheduler.drain();
  scheduler.shutdown();
  EXPECT_TRUE(state.server_ok.load(std::memory_order_acquire));
  EXPECT_TRUE(state.client_ok.load(std::memory_order_acquire));
  EXPECT_TRUE(listener.close().ok());
  EXPECT_EQ(std::fopen(name.c_str(), "rb"), nullptr);
}
#endif

TEST(IoLocal, AcceptDeadlineRetiresBeforeReturning) {
  Scheduler scheduler(1, system_allocator(), 128);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  const std::string name = unique_local_name();
  auto endpoint = native_endpoint(name);
  ASSERT_TRUE(endpoint.has_value());
  LocalListenOptions options;
#if !defined(_WIN32)
  options.replace_existing_socket = true;
#endif
  auto bound = LocalListener::bind(io, kOwner, *endpoint, options);
  ASSERT_TRUE(bound.has_value());
  LocalListener listener = std::move(*bound);
  FailureState state;
  Drive driver{deadline_accept(system_allocator(), &io, &listener, &state)};
  Fiber fiber(kDefaultStackSize, &drive_entry, &driver);
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &fiber));
  scheduler.drain();
  scheduler.shutdown();
  EXPECT_TRUE(state.done.load(std::memory_order_acquire));
  EXPECT_TRUE(state.expected.load(std::memory_order_acquire));
  EXPECT_TRUE(state.retired.load(std::memory_order_acquire));
  EXPECT_TRUE(listener.close().ok());
}

TEST(IoLocal, CancelledConnectDoesNotTouchTheNativeTransport) {
  Scheduler scheduler(1, system_allocator(), 128);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  const std::string name = unique_local_name();
  auto endpoint = native_endpoint(name);
  ASSERT_TRUE(endpoint.has_value());
  auto token = hyle::sched::CancelToken::try_create_root(system_allocator());
  ASSERT_TRUE(token.has_value());
  token->cancel();
  FailureState state;
  Drive driver{cancelled_connect(system_allocator(), &io, &*endpoint,
                                 &*token, &state)};
  Fiber fiber(kDefaultStackSize, &drive_entry, &driver);
  ASSERT_TRUE(scheduler.spawn_on(kOwner, &fiber));
  scheduler.drain();
  scheduler.shutdown();
  EXPECT_TRUE(state.done.load(std::memory_order_acquire));
  EXPECT_TRUE(state.expected.load(std::memory_order_acquire));
  EXPECT_TRUE(state.retired.load(std::memory_order_acquire));
}

#if !defined(_WIN32)
TEST(IoLocal, StaleReplacementNeverDeletesAnOrdinaryFile) {
  Scheduler scheduler(1, system_allocator(), 64);
  IoContext io(scheduler);
  ASSERT_TRUE(io.valid());
  const std::string path = unique_local_name();
  std::FILE* file = std::fopen(path.c_str(), "wb");
  ASSERT_NE(file, nullptr);
  std::fclose(file);
  auto endpoint = LocalEndpoint::unix_path(path);
  ASSERT_TRUE(endpoint.has_value());
  LocalListenOptions options;
  options.replace_existing_socket = true;
  auto listener = LocalListener::bind(io, kOwner, *endpoint, options);
  EXPECT_FALSE(listener.has_value());
  EXPECT_TRUE(listener.error().IsAlreadyExists());
  std::FILE* preserved = std::fopen(path.c_str(), "rb");
  EXPECT_NE(preserved, nullptr);
  if (preserved != nullptr) std::fclose(preserved);
  std::remove(path.c_str());
  scheduler.shutdown();
}
#endif

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <string>
#include <thread>

#include "hyle/io/dns.h"
#include "hyle/io/io_context.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/blocking.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/sync.h"
#include "hyle/time/clock.h"

using hyle::io::AddressFamily;
using hyle::io::IoContext;
using hyle::io::IoContextOptions;
using hyle::io::OperationOptions;
using hyle::io::resolve;
using hyle::mem::system_allocator;
using hyle::sched::CancelToken;
using hyle::sched::Event;
using hyle::sched::EventResetMode;
using hyle::sched::Scheduler;

namespace {

struct DnsRuntime {
  explicit DnsRuntime(unsigned max_blocking_threads = 4)
      : scheduler(1, system_allocator(), 64),
        io(scheduler, options(max_blocking_threads)) {}

  ~DnsRuntime() { scheduler.shutdown(); }

  static IoContextOptions options(unsigned max_blocking_threads) {
    IoContextOptions value;
    value.max_blocking_threads = max_blocking_threads;
    return value;
  }

  Scheduler scheduler;
  IoContext io;
};

struct BlockingGate {
  Event started{EventResetMode::kManual};
  Event release{EventResetMode::kManual};
};

int occupy_blocking_thread(BlockingGate* gate) noexcept {
  gate->started.signal();
  gate->release.wait();
  return 0;
}

}  // namespace

TEST(IoDns, ResolvesIpv4Literal) {
  DnsRuntime runtime;
  auto result =
      resolve(runtime.io, system_allocator(), "127.0.0.1", 8080).join();
  ASSERT_TRUE(result.has_value());
  ASSERT_FALSE(result->empty());
  bool found_v4_loopback = false;
  for (const auto& endpoint : *result) {
    if (endpoint.family() == AddressFamily::kIpv4 &&
        endpoint.port() == 8080 && endpoint.is_loopback())
      found_v4_loopback = true;
  }
  EXPECT_TRUE(found_v4_loopback);
}

TEST(IoDns, ResolvesLocalhostAndOwnsHostInput) {
  DnsRuntime runtime;
  hyle::io::ResolveHandle handle;
  {
    std::string temporary_host = "localhost";
    handle = resolve(runtime.io, system_allocator(), temporary_host, 443);
  }
  auto result = handle.join();
  if (!result.has_value())
    GTEST_SKIP() << "localhost did not resolve: " << result.error().message();
  ASSERT_FALSE(result->empty());
  for (const auto& endpoint : *result) {
    EXPECT_TRUE(endpoint.family() == AddressFamily::kIpv4 ||
                endpoint.family() == AddressFamily::kIpv6);
    EXPECT_EQ(endpoint.port(), 443);
  }
}

TEST(IoDns, RejectsEmptyAndEmbeddedNullHost) {
  DnsRuntime runtime;
  auto empty =
      resolve(runtime.io, system_allocator(), "", 80).join();
  ASSERT_FALSE(empty.has_value());
  EXPECT_TRUE(empty.error().IsInvalidArgument());

  const char embedded[] = {'h', 'o', 's', 't', '\0', 'x'};
  auto invalid =
      resolve(runtime.io, system_allocator(),
              std::string_view(embedded, sizeof(embedded)), 80)
          .join();
  ASSERT_FALSE(invalid.has_value());
  EXPECT_TRUE(invalid.error().IsInvalidArgument());
}

TEST(IoDns, UnresolvableNameReturnsTypedError) {
  DnsRuntime runtime;
  auto result = resolve(runtime.io, system_allocator(),
                        "no.such.host.hyle.invalid.", 80)
                    .join();
  if (result.has_value())
    GTEST_SKIP() << "resolver answered a reserved .invalid name; count="
                 << result->size();
  EXPECT_FALSE(result.error().ok());
}

TEST(IoDns, DeadlineCoversQueuedResolverAndDetachedStateRetiresSafely) {
  DnsRuntime runtime(1);
  BlockingGate gate;
  auto blocker = hyle::sched::spawn_blocking(
      runtime.io.blocking_pool(), system_allocator(),
      [&gate]() noexcept { return occupy_blocking_thread(&gate); });
  gate.started.wait();

  auto handle = resolve(runtime.io, system_allocator(), "localhost", 443);
  OperationOptions options;
  options.deadline = hyle::time::deadline_after(std::chrono::milliseconds(10));
  const auto started = std::chrono::steady_clock::now();
  auto result = handle.join(options);
  const auto elapsed = std::chrono::steady_clock::now() - started;

  EXPECT_FALSE(result.has_value());
  EXPECT_TRUE(result.error().IsDeadlineExceeded());
  EXPECT_LT(elapsed, std::chrono::seconds(1));
  gate.release.signal();
  EXPECT_TRUE(blocker.join().has_value());
}

TEST(IoDns, CancellationWakesQueuedResolverPromptly) {
  DnsRuntime runtime(1);
  BlockingGate gate;
  auto blocker = hyle::sched::spawn_blocking(
      runtime.io.blocking_pool(), system_allocator(),
      [&gate]() noexcept { return occupy_blocking_thread(&gate); });
  gate.started.wait();

  auto token = CancelToken::try_create_root(system_allocator());
  ASSERT_TRUE(token.has_value());
  auto handle = resolve(runtime.io, system_allocator(), "localhost", 443);
  std::thread canceler([&token] {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    token->cancel();
  });
  OperationOptions options;
  options.cancel = &*token;
  const auto started = std::chrono::steady_clock::now();
  auto result = handle.join(options);
  const auto elapsed = std::chrono::steady_clock::now() - started;
  canceler.join();

  EXPECT_FALSE(result.has_value());
  EXPECT_TRUE(result.error().IsCancelled());
  EXPECT_LT(elapsed, std::chrono::seconds(1));
  gate.release.signal();
  EXPECT_TRUE(blocker.join().has_value());
}

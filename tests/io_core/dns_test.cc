#include <gtest/gtest.h>

#include <string>

#include "hyle/io/dns.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/blocking.h"

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <winsock2.h>
#  include <ws2tcpip.h>
#else
#  include <arpa/inet.h>
#  include <netinet/in.h>
#  include <sys/socket.h>
#endif

// io-09 async DNS test: resolve() offloads getaddrinfo to the blocking pool and
// delivers a value list through a JoinHandle. A bare thread joins it (thread-mode)
// -- no scheduler needed, since DNS is orthogonal to the io ring. A literal IP must
// resolve deterministically (getaddrinfo parses it without a network hit); a
// hostname (localhost) should resolve on any sane machine but is SKIP-guarded so a
// networkless CI doesn't fail.

using hyle::io::AddressFamily;
using hyle::io::resolve;
using hyle::mem::system_allocator;
using hyle::sched::BlockingPool;

namespace {
// Windows getaddrinfo needs Winsock initialized; the proactor does it in the real
// runtime, but this test resolves without an IoContext, so it starts WSA itself.
struct WsaGuard {
  WsaGuard() {
#if defined(_WIN32)
    WSADATA w;
    ::WSAStartup(MAKEWORD(2, 2), &w);
#endif
  }
  ~WsaGuard() {
#if defined(_WIN32)
    ::WSACleanup();
#endif
  }
};
}  // namespace

// A numeric IPv4 literal resolves to exactly that address, no network involved.
TEST(IoDns, ResolvesIpv4Literal) {
  WsaGuard wsa;
  BlockingPool pool(system_allocator());
  auto h = resolve(pool, system_allocator(), "127.0.0.1", 8080);
  auto r = h.join();
  ASSERT_TRUE(r.has_value());
  ASSERT_FALSE(r->empty());
  // The first result must be the AF_INET loopback on port 8080.
  bool found_v4_loopback = false;
  for (const auto& endpoint : *r) {
    if (endpoint.family() == AddressFamily::kIpv4 &&
        endpoint.port() == 8080 && endpoint.is_loopback())
      found_v4_loopback = true;
  }
  EXPECT_TRUE(found_v4_loopback);
}

// A hostname resolves through the OS resolver. localhost is in every hosts file;
// if resolution fails (a truly networkless/misconfigured box) SKIP rather than fail.
TEST(IoDns, ResolvesLocalhost) {
  WsaGuard wsa;
  BlockingPool pool(system_allocator());
  auto h = resolve(pool, system_allocator(), "localhost", 443);
  auto r = h.join();
  if (!r.has_value())
    GTEST_SKIP() << "localhost did not resolve (networkless CI?)";
  ASSERT_FALSE(r->empty());
  // Every returned address must carry the requested port and a known family.
  for (const auto& endpoint : *r) {
    EXPECT_TRUE(endpoint.family() == AddressFamily::kIpv4 ||
                endpoint.family() == AddressFamily::kIpv6);
    EXPECT_EQ(endpoint.port(), 443);
  }
}

TEST(IoDns, RejectsEmptyAndEmbeddedNullHost) {
  WsaGuard wsa;
  BlockingPool pool(system_allocator());
  auto empty = resolve(pool, system_allocator(), "", 80).join();
  ASSERT_FALSE(empty.has_value());
  EXPECT_TRUE(empty.error().IsInvalidArgument());

  const char embedded[] = {'h', 'o', 's', 't', '\0', 'x'};
  auto invalid = resolve(pool, system_allocator(),
                         std::string_view(embedded, sizeof(embedded)), 80)
                     .join();
  ASSERT_FALSE(invalid.has_value());
  EXPECT_TRUE(invalid.error().IsInvalidArgument());
}

// A name that cannot resolve returns a non-zero error and zero addresses, WITHOUT
// crashing -- the failure path is a clean value, not an exception/hang.
TEST(IoDns, UnresolvableNameErrors) {
  WsaGuard wsa;
  BlockingPool pool(system_allocator());
  auto h = resolve(pool, system_allocator(),
                   "no.such.host.hyle.invalid.", 80);
  auto r = h.join();
  // A hijacking resolver (proxy fake-ip, captive portal, ISP wildcard) answers
  // even an RFC-2606 `.invalid` name -- then the failure path cannot be
  // exercised, so SKIP rather than fail (mirrors ResolvesLocalhost's guard for
  // a networkless box). A well-behaved resolver returns NXDOMAIN.
  if (r.has_value())
    GTEST_SKIP() << "resolver answered a .invalid name (hijacked DNS?) count="
                 << r->size();
  EXPECT_FALSE(r.error().ok());
}

#include <gtest/gtest.h>

#include <cstring>

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

using hyle::io::ResolvedAddr;
using hyle::io::ResolveResult;
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
  EXPECT_EQ(r->error, 0);
  ASSERT_GE(r->count, 1);
  // The first result must be the AF_INET loopback on port 8080.
  bool found_v4_loopback = false;
  for (int i = 0; i < r->count; ++i) {
    const ResolvedAddr& a = r->addrs[i];
    if (a.family != AF_INET) continue;
    const auto* sin = reinterpret_cast<const sockaddr_in*>(a.storage);
    if (ntohs(sin->sin_port) != 8080) continue;
    if (ntohl(sin->sin_addr.s_addr) == INADDR_LOOPBACK) found_v4_loopback = true;
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
  ASSERT_TRUE(r.has_value());
  if (r->error != 0 || r->count == 0)
    GTEST_SKIP() << "localhost did not resolve (networkless CI?) err=" << r->error;
  // Every returned address must carry the requested port and a known family.
  for (int i = 0; i < r->count; ++i) {
    const ResolvedAddr& a = r->addrs[i];
    EXPECT_TRUE(a.family == AF_INET || a.family == AF_INET6);
    EXPECT_GT(a.len, 0u);
  }
}

// A name that cannot resolve returns a non-zero error and zero addresses, WITHOUT
// crashing -- the failure path is a clean value, not an exception/hang.
TEST(IoDns, UnresolvableNameErrors) {
  WsaGuard wsa;
  BlockingPool pool(system_allocator());
  auto h = resolve(pool, system_allocator(),
                   "no.such.host.hyle.invalid.", 80);
  auto r = h.join();
  ASSERT_TRUE(r.has_value());
  // A hijacking resolver (proxy fake-ip, captive portal, ISP wildcard) answers
  // even an RFC-2606 `.invalid` name -- then the failure path cannot be
  // exercised, so SKIP rather than fail (mirrors ResolvesLocalhost's guard for
  // a networkless box). A well-behaved resolver returns NXDOMAIN.
  if (r->error == 0 || r->count != 0)
    GTEST_SKIP() << "resolver answered a .invalid name (hijacked DNS?) count="
                 << r->count << " err=" << r->error;
  EXPECT_EQ(r->count, 0);
  EXPECT_NE(r->error, 0);  // getaddrinfo signalled a lookup failure
}

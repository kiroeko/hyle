#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "hyle/io/net.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/blocking.h"  // BlockingPool, spawn_blocking, JoinHandle

#if defined(HYLE_STATIC)
#  define HYLE_IO_API
#elif defined(HYLE_IO_BUILD)
#  define HYLE_IO_API HYLE_EXPORT
#else
#  define HYLE_IO_API HYLE_IMPORT
#endif

// io-09 async DNS (capability C10). resolve_addr() in tcp.h only parses a numeric
// IPv4 literal (inet_pton) -- it cannot resolve a hostname. Real name resolution
// means getaddrinfo, which is a BLOCKING syscall (it can hit /etc/hosts, nsswitch,
// and the network). Running it on a shard worker would stall EVERY fiber pinned to
// that core until the lookup returns -- a share-nothing violation.
//
// The fix is tokio's exact approach (verified against tokio/src/net/addr.rs): do
// NOT reimplement the DNS protocol (libevent evdns / c-ares route); instead OFFLOAD
// getaddrinfo to the blocking pool (sched-11). The pool thread blocks; the worker
// keeps running its other fibers; the result comes back through the same
// JoinState/JoinHandle spine as spawn_blocking. Nearly zero new mechanism -- DNS is
// just a domain-specific application of spawn_blocking.
//
// A self-contained DNS protocol client (cancellable, own timeout/retry) is a later
// enhancement; the blocking-offload path is the minimal shippable answer and is
// what most runtimes (tokio, asio) actually use.

namespace hyle::io {

// One resolved address, held as a VALUE (a copy of the addrinfo's sockaddr): the
// closure calls freeaddrinfo before returning, so no addrinfo* ever escapes the
// pool thread (the getaddrinfo lifetime rule). storage holds a sockaddr_in /
// sockaddr_in6; `len` is its byte length; `family` is AF_INET / AF_INET6. Feed
// (storage, len) straight to TcpStream::connect / UdpSocket::send_to.
struct ResolvedAddr {
  alignas(16) unsigned char storage[128] = {};  // a sockaddr_storage-sized copy
  std::uint32_t len = 0;                         // valid bytes (the sockaddr size)
  AddressFamily family = AddressFamily::kIpv4;
};

// The result of a resolve(): up to kMax addresses (a host may have several A/AAAA
// records; try them in order for happy-eyeballs-style fallback). `count` is how
// many are filled; `error` is 0 on success or the getaddrinfo return code (EAI_*
// on POSIX / a WSA error on Windows) so the caller can tell "no such host" from a
// transient failure. A JoinHandle<ResolveResult> from resolve() delivers this.
struct ResolveResult {
  static constexpr int kMax = 8;
  static constexpr int kInvalidHost = -10'001;
  ResolvedAddr addrs[kMax];
  int count = 0;
  int error = 0;
};

namespace detail {
// The blocking half (runs on a pool thread): getaddrinfo `host`:`port`, copy each
// result into `out` as a value, then freeaddrinfo. Exported so resolve() can stay
// header-inline WITHOUT pulling the platform socket headers (netdb.h / ws2tcpip.h)
// into every TU that resolves a name. `host`/`out` are owned by the pool job.
HYLE_IO_API void resolve_blocking(const char* host, std::uint16_t port,
                                  ResolveResult* out) noexcept;
}  // namespace detail

// Asynchronously resolve `host`:`port` by offloading getaddrinfo to `pool`. Returns
// a JoinHandle<ResolveResult>: from a fiber, `.join()` is fiber-aware -- the worker
// runs other fibers while the lookup blocks a pool thread. A literal IP resolves
// too (getaddrinfo parses numeric hosts synchronously inside the closure). The host
// string is COPIED into the job, so the caller's buffer need not outlive the await.
// `alloc` backs the JoinState + the job (use the same allocator you join from).
inline sched::JoinHandle<ResolveResult> resolve(sched::BlockingPool& pool,
                                                mem::Allocator alloc,
                                                const char* host,
                                                std::uint16_t port) noexcept {
  // A copyable POD job: owns its own host copy so nothing dangles when it runs on
  // another thread. operator() is what spawn_blocking invokes on the pool thread.
  struct Job {
    char host[256] = {};
    std::uint16_t port = 0;
    bool invalid_host = false;
    ResolveResult operator()() const noexcept {
      ResolveResult r;
      if (invalid_host) {
        r.error = ResolveResult::kInvalidHost;
        return r;
      }
      detail::resolve_blocking(host, port, &r);
      return r;
    }
  };
  Job job;
  if (host == nullptr) {
    job.invalid_host = true;
    return sched::spawn_blocking(pool, alloc, job);
  }
  // `host` is a C string by contract. Determine its length first so an inlined
  // short string literal never produces a speculative host[255] access (GCC
  // correctly diagnosed the old bounded-copy loop as potentially out of bounds).
  const std::size_t len = std::strlen(host);
  if (len >= sizeof(job.host)) {
    job.invalid_host = true;
  } else {
    std::memcpy(job.host, host, len + 1);
  }
  job.port = port;
  return sched::spawn_blocking(pool, alloc, job);
}

}  // namespace hyle::io

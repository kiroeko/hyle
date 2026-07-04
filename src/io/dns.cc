#include "hyle/io/dns.h"

#include <cstring>

#include "hyle/platform.h"  // HYLE_OS_* (must precede the per-OS socket headers)

#if defined(HYLE_OS_WINDOWS)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <winsock2.h>
#  include <ws2tcpip.h>  // getaddrinfo / addrinfo
#  include <windows.h>
#else
#  include <netdb.h>      // getaddrinfo / freeaddrinfo / addrinfo
#  include <netinet/in.h>
#  include <sys/socket.h>

#  include <cstdio>  // snprintf (port -> service string)
#endif

// io-09 async DNS, blocking half. This runs on a sched-11 BlockingPool thread (see
// dns.h): a plain synchronous getaddrinfo, with every result COPIED out as a value
// before freeaddrinfo, so no addrinfo* escapes. The header's resolve() offloads it.

namespace hyle::io::detail {

void resolve_blocking(const char* host, std::uint16_t port,
                      ResolveResult* out) noexcept {
  *out = ResolveResult{};

  // hints: AF_UNSPEC (want both A and AAAA -- IPv4 and IPv6), SOCK_STREAM so a
  // dual-proto host doesn't return each address twice (tokio uses the same). No
  // AI_PASSIVE: these are connect targets, not bind addresses.
  addrinfo hints;
  std::memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  // getaddrinfo takes the port as a decimal service string.
  char service[8];
#if defined(HYLE_OS_WINDOWS)
  ::_itoa_s(static_cast<int>(port), service, sizeof(service), 10);
#else
  std::snprintf(service, sizeof(service), "%u", static_cast<unsigned>(port));
#endif

  addrinfo* res = nullptr;
  int rc = ::getaddrinfo(host, service, &hints, &res);
  if (rc != 0) {
    out->error = rc;  // EAI_* (POSIX) / WSA error (Windows) -- caller distinguishes
    return;
  }

  // Copy each address as a VALUE, then free the list. Only AF_INET / AF_INET6 are
  // representable in a sockaddr_storage-sized slot; skip anything larger.
  int n = 0;
  for (addrinfo* ai = res; ai != nullptr && n < ResolveResult::kMax;
       ai = ai->ai_next) {
    if (ai->ai_family != AF_INET && ai->ai_family != AF_INET6) continue;
    if (ai->ai_addrlen == 0 ||
        ai->ai_addrlen > sizeof(out->addrs[n].storage))
      continue;
    std::memcpy(out->addrs[n].storage, ai->ai_addr,
                static_cast<std::size_t>(ai->ai_addrlen));
    out->addrs[n].len = static_cast<std::uint32_t>(ai->ai_addrlen);
    out->addrs[n].family = ai->ai_family;
    ++n;
  }
  out->count = n;
  ::freeaddrinfo(res);  // the addrinfo list dies HERE -- no pointer escapes.
}

}  // namespace hyle::io::detail

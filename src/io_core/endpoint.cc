#include "hyle/io/endpoint.h"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstring>

#include "hyle/core/environment.h"
#include "endpoint_native.h"

#if HYLE_OS_WINDOWS
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  include <windows.h>
#else
#  include <arpa/inet.h>
#  include <netinet/in.h>
#endif

namespace hyle::io {
namespace {

Status invalid_endpoint(std::string_view message) noexcept {
  return Status::InvalidArgument(message);
}

bool all_zero(const std::byte* data, std::size_t size) noexcept {
  for (std::size_t i = 0; i < size; ++i) {
    if (data[i] != std::byte{0}) return false;
  }
  return true;
}

}  // namespace

Result<Endpoint> Endpoint::parse(std::string_view input,
                                 std::uint16_t port) noexcept {
  if (input.empty()) return make_unexpected(invalid_endpoint("empty IP address"));

  if (input.size() >= 2 && input.front() == '[' && input.back() == ']') {
    input.remove_prefix(1);
    input.remove_suffix(1);
  }

  Endpoint endpoint;
  endpoint.port_ = port;
  std::string_view address = input;
  if (const std::size_t percent = input.rfind('%');
      percent != std::string_view::npos) {
    address = input.substr(0, percent);
    const std::string_view scope = input.substr(percent + 1);
    if (scope.empty())
      return make_unexpected(invalid_endpoint("empty IPv6 scope id"));
    const char* first = scope.data();
    const char* last = first + scope.size();
    auto parsed = std::from_chars(first, last, endpoint.scope_id_);
    if (parsed.ec != std::errc{} || parsed.ptr != last)
      return make_unexpected(
          invalid_endpoint("IPv6 scope id must be an unsigned integer"));
  }

  // inet_pton requires a NUL-terminated source. Numeric addresses are bounded
  // by their textual protocol representation; reject arbitrary long input
  // rather than allocating on this cold parse path.
  if (address.empty() || address.size() >= 64)
    return make_unexpected(invalid_endpoint("invalid numeric IP address"));
  char text[64]{};
  std::memcpy(text, address.data(), address.size());

  in_addr ipv4{};
  if (::inet_pton(AF_INET, text, &ipv4) == 1) {
    if (endpoint.scope_id_ != 0)
      return make_unexpected(invalid_endpoint("IPv4 address cannot have a scope id"));
    endpoint.family_ = AddressFamily::kIpv4;
    std::memcpy(endpoint.address_.data(), &ipv4, sizeof(ipv4));
    endpoint.valid_ = true;
    return endpoint;
  }

  in6_addr ipv6{};
  if (::inet_pton(AF_INET6, text, &ipv6) == 1) {
    endpoint.family_ = AddressFamily::kIpv6;
    std::memcpy(endpoint.address_.data(), &ipv6, sizeof(ipv6));
    endpoint.valid_ = true;
    return endpoint;
  }
  return make_unexpected(invalid_endpoint("invalid numeric IP address"));
}

bool Endpoint::is_unspecified() const noexcept {
  if (!valid_) return false;
  const std::size_t size = family_ == AddressFamily::kIpv4 ? 4 : 16;
  return all_zero(address_.data(), size);
}

bool Endpoint::is_loopback() const noexcept {
  if (!valid_) return false;
  if (family_ == AddressFamily::kIpv4)
    return address_[0] == std::byte{127};
  return all_zero(address_.data(), 15) && address_[15] == std::byte{1};
}

Result<std::size_t> Endpoint::format(char* destination, std::size_t capacity,
                                     bool with_port) const noexcept {
  if (!valid_)
    return make_unexpected(invalid_endpoint("cannot format an invalid endpoint"));
  if (destination == nullptr && capacity != 0)
    return make_unexpected(invalid_endpoint("null endpoint output buffer"));

  char address[64]{};
  const int family = family_ == AddressFamily::kIpv4 ? AF_INET : AF_INET6;
  if (::inet_ntop(family, address_.data(), address, sizeof(address)) == nullptr) {
    return make_unexpected(Status::Internal("numeric IP formatting failed"));
  }

  char complete[96]{};
  int length = 0;
  if (!with_port) {
    length = scope_id_ == 0
                 ? std::snprintf(complete, sizeof(complete), "%s", address)
                 : std::snprintf(complete, sizeof(complete), "%s%%%u", address,
                                 static_cast<unsigned>(scope_id_));
  } else if (family_ == AddressFamily::kIpv6) {
    length = scope_id_ == 0
                 ? std::snprintf(complete, sizeof(complete), "[%s]:%u", address,
                                 static_cast<unsigned>(port_))
                 : std::snprintf(complete, sizeof(complete), "[%s%%%u]:%u", address,
                                 static_cast<unsigned>(scope_id_),
                                 static_cast<unsigned>(port_));
  } else {
    length = std::snprintf(complete, sizeof(complete), "%s:%u", address,
                           static_cast<unsigned>(port_));
  }
  if (length < 0 || static_cast<std::size_t>(length) >= sizeof(complete))
    return make_unexpected(Status::Internal("endpoint formatting overflow"));

  const std::size_t needed = static_cast<std::size_t>(length);
  if (capacity != 0) {
    const std::size_t copy = (std::min)(needed, capacity - 1);
    std::memcpy(destination, complete, copy);
    destination[copy] = '\0';
  }
  return needed;
}

Result<NativeEndpoint> EndpointNativeAccess::encode(
    const Endpoint& endpoint) noexcept {
  if (!endpoint.valid_)
    return make_unexpected(invalid_endpoint("invalid endpoint"));
  NativeEndpoint native;
  if (endpoint.family_ == AddressFamily::kIpv4) {
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(endpoint.port_);
    std::memcpy(&address.sin_addr, endpoint.address_.data(),
                sizeof(address.sin_addr));
    std::memcpy(native.storage, &address, sizeof(address));
    native.size = sizeof(address);
  } else {
    sockaddr_in6 address{};
    address.sin6_family = AF_INET6;
    address.sin6_port = htons(endpoint.port_);
    address.sin6_scope_id = endpoint.scope_id_;
    std::memcpy(&address.sin6_addr, endpoint.address_.data(),
                sizeof(address.sin6_addr));
    std::memcpy(native.storage, &address, sizeof(address));
    native.size = sizeof(address);
  }
  return native;
}

Result<Endpoint> EndpointNativeAccess::decode(const void* source,
                                              std::uint32_t size) noexcept {
  if (source == nullptr)
    return make_unexpected(invalid_endpoint("null native endpoint"));
  const auto* generic = static_cast<const sockaddr*>(source);
  Endpoint endpoint;
  if (generic->sa_family == AF_INET) {
    if (size < sizeof(sockaddr_in))
      return make_unexpected(invalid_endpoint("truncated IPv4 endpoint"));
    const auto* address = static_cast<const sockaddr_in*>(source);
    endpoint.family_ = AddressFamily::kIpv4;
    endpoint.port_ = ntohs(address->sin_port);
    std::memcpy(endpoint.address_.data(), &address->sin_addr,
                sizeof(address->sin_addr));
  } else if (generic->sa_family == AF_INET6) {
    if (size < sizeof(sockaddr_in6))
      return make_unexpected(invalid_endpoint("truncated IPv6 endpoint"));
    const auto* address = static_cast<const sockaddr_in6*>(source);
    endpoint.family_ = AddressFamily::kIpv6;
    endpoint.port_ = ntohs(address->sin6_port);
    endpoint.scope_id_ = address->sin6_scope_id;
    std::memcpy(endpoint.address_.data(), &address->sin6_addr,
                sizeof(address->sin6_addr));
  } else {
    return make_unexpected(Status::InvalidArgument(
        "native endpoint is neither IPv4 nor IPv6"));
  }
  endpoint.valid_ = true;
  return endpoint;
}

}  // namespace hyle::io

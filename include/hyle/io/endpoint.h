#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "hyle/error/result.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_IO_API
#elif defined(HYLE_IO_CORE_BUILD)
#  define HYLE_IO_API HYLE_EXPORT
#else
#  define HYLE_IO_API HYLE_IMPORT
#endif

namespace hyle::io {

enum class AddressFamily : std::uint8_t {
  kIpv4 = 4,
  kIpv6 = 6,
};

// Owning, platform-neutral IP endpoint. Unlike sockaddr_storage this value has
// no native padding, native family numbers, pointer/length pair, or hidden
// lifetime. The IO backend materializes a sockaddr only at the syscall edge.
class Endpoint {
 public:
  Endpoint() noexcept = default;

  // Parse a numeric IPv4/IPv6 address. IPv6 may be bracketed and may carry a
  // numeric scope suffix ("fe80::1%3"). Host names belong to resolve().
  HYLE_IO_API static Result<Endpoint> parse(std::string_view address,
                                            std::uint16_t port) noexcept;

  static Endpoint any(AddressFamily family, std::uint16_t port = 0) noexcept {
    Endpoint endpoint;
    endpoint.family_ = family;
    endpoint.port_ = port;
    endpoint.valid_ = true;
    return endpoint;
  }

  static Endpoint loopback(AddressFamily family,
                           std::uint16_t port = 0) noexcept {
    Endpoint endpoint = any(family, port);
    if (family == AddressFamily::kIpv4) {
      endpoint.address_[0] = std::byte{127};
      endpoint.address_[3] = std::byte{1};
    } else {
      endpoint.address_[15] = std::byte{1};
    }
    return endpoint;
  }

  bool valid() const noexcept { return valid_; }
  explicit operator bool() const noexcept { return valid(); }
  AddressFamily family() const noexcept { return family_; }
  std::uint16_t port() const noexcept { return port_; }
  std::uint32_t scope_id() const noexcept { return scope_id_; }

  void set_port(std::uint16_t port) noexcept { port_ = port; }
  void set_scope_id(std::uint32_t scope_id) noexcept { scope_id_ = scope_id; }

  HYLE_IO_API bool is_unspecified() const noexcept;
  HYLE_IO_API bool is_loopback() const noexcept;

  // snprintf-style formatting. Writes a NUL when capacity > 0 and returns the
  // complete required byte count excluding it; value >= capacity means the
  // output was truncated. With_port renders IPv4 as 127.0.0.1:80 and IPv6 as
  // [::1]:80, preserving a numeric scope suffix.
  HYLE_IO_API Result<std::size_t> format(char* destination,
                                         std::size_t capacity,
                                         bool with_port = true) const noexcept;

  friend bool operator==(const Endpoint&, const Endpoint&) noexcept = default;

 private:
  friend struct EndpointNativeAccess;

  std::array<std::byte, 16> address_{};
  std::uint32_t scope_id_ = 0;
  std::uint16_t port_ = 0;
  AddressFamily family_ = AddressFamily::kIpv4;
  bool valid_ = false;
};

static_assert(sizeof(Endpoint) <= 32,
              "Endpoint must remain a small owning transport value");

}  // namespace hyle::io

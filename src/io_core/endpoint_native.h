#pragma once

#include <cstdint>

#include "hyle/error/result.h"
#include "hyle/io/endpoint.h"

namespace hyle::io {

struct NativeEndpoint {
  alignas(16) unsigned char storage[128]{};
  std::uint32_t size = 0;
};

struct EndpointNativeAccess {
  static Result<NativeEndpoint> encode(const Endpoint& endpoint) noexcept;
  static Result<Endpoint> decode(const void* address,
                                 std::uint32_t size) noexcept;
};

}  // namespace hyle::io

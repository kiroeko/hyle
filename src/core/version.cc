#include "hyle/core/version.h"

namespace hyle {

std::uint32_t version() noexcept {
  return (0u << 16) | (1u << 8) | 0u;  // 0.1.0
}

bool scaffolding_ok() noexcept { return true; }

}  // namespace hyle

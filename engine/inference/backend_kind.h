#pragma once

#include <cstdint>

namespace hyle::inference {

// Cold configuration vocabulary, independent of native endpoint headers.
enum class BackendKind : std::uint8_t { kDefault, kCpu, kCuda, kApple, kVulkan };
inline constexpr const char *backend_protocol_name(BackendKind kind) noexcept {
  switch (kind) {
    case BackendKind::kCpu: return "cpu";
    case BackendKind::kCuda: return "cuda";
    case BackendKind::kApple: return "apple";
    case BackendKind::kVulkan: return "vulkan";
    default: return "-";
  }
}
struct BackendSelection final {
  BackendKind primary = BackendKind::kDefault;
  bool cpu_assist = false;
};

} // namespace hyle::inference

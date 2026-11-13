#pragma once

#include <cstdint>

namespace hyle::inference {

// Cold configuration vocabulary, independent of native endpoint headers.
enum class BackendKind : std::uint8_t { kDefault, kCpu, kCuda, kApple, kVulkan };
struct BackendSelection final {
  BackendKind primary = BackendKind::kDefault;
  bool cpu_assist = false;
};

} // namespace hyle::inference

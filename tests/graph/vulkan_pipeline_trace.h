#pragma once
// Test-local ABI for the transparent Vulkan loader fixture; never installed.
#include <cstddef>
#include <cstdint>

namespace g09_vk {
struct Calibration {
  std::uint64_t device_ticks = 0, host_ticks = 0, max_deviation_ns = 0;
};
struct DeviceInfo {
  double timestamp_period_ns = 0;
  std::uint64_t host_frequency = 0;
  std::uint32_t timestamp_bits = 0, host_domain = 0;
  std::uint32_t matched_shader_modules = 0, matched_pipelines = 0;
  std::uint32_t fp16_controls = 0;
};
struct Dispatch {
  std::uint64_t begin = 0, end = 0;
  std::uint32_t phase = 0, query = 0;
};
struct Observation {
  Calibration after;
  std::uint32_t count = 0;
  Dispatch dispatches[64]{};
};
using Configure = bool (*)(const void *, std::size_t, const char *);
using Ready = bool (*)(DeviceInfo *);
using Begin = bool (*)(std::uint32_t, Calibration *);
using End = bool (*)(Observation *);
} // namespace g09_vk

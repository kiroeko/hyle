#pragma once

#include <cstdint>

#include "hyle/error/status.h"
#include "hyle/platform/visibility.h"

namespace hyle::platform {

// Explicit application-startup scheduling and execution-quality policy.
// This is separate from ThreadPriority because the base Platform factor must
// not depend on platform_thread and because the OS scopes are genuinely
// different. There is deliberately no realtime level: realtime scheduling can
// starve system work and needs an application-specific admission policy.
enum class ProcessPriority : std::int8_t {
  kLowest = -2,
  kLow = -1,
  kNormal = 0,
  kHigh = 1,
  kHighest = 2,
};

// Set the current application's process-startup priority policy. Call this
// before creating worker threads when process-wide behavior is required:
// Linux nice values are per-thread and are inherited by subsequently-created
// threads, while Windows and macOS apply process-scoped policy. On Windows the
// portable ladder sets both the priority class and execution-speed QoS;
// kNormal returns QoS to system-managed behavior. Elevation may fail for lack
// of privilege. This explicit control operation is not performed by default.
HYLE_PLATFORM_API Status
set_process_priority(ProcessPriority priority) noexcept;

}  // namespace hyle::platform

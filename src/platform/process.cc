#include "hyle/platform/process.h"

#include <cerrno>

#include "hyle/core/environment.h"

#if HYLE_OS_WINDOWS
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <sys/resource.h>
#endif

namespace hyle::platform {

Status set_process_priority(ProcessPriority priority) noexcept {
#if HYLE_OS_WINDOWS
  DWORD priority_class = NORMAL_PRIORITY_CLASS;
  PROCESS_POWER_THROTTLING_STATE qos{};
  qos.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
  switch (priority) {
    case ProcessPriority::kLowest:
      priority_class = IDLE_PRIORITY_CLASS;
      qos.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
      qos.StateMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
      break;
    case ProcessPriority::kLow:
      priority_class = BELOW_NORMAL_PRIORITY_CLASS;
      qos.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
      qos.StateMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
      break;
    case ProcessPriority::kNormal:
      // A zero control mask returns execution QoS to system heuristics.
      break;
    case ProcessPriority::kHigh:
      priority_class = ABOVE_NORMAL_PRIORITY_CLASS;
      qos.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
      break;
    case ProcessPriority::kHighest:
      priority_class = HIGH_PRIORITY_CLASS;
      qos.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
      break;
    default:
      return Status::InvalidArgument("process priority is invalid");
  }

  // These are two independent OS controls. Apply QoS first so a class failure
  // is still visible; Windows provides no transaction spanning both settings.
  if (!::SetProcessInformation(::GetCurrentProcess(), ProcessPowerThrottling,
                               &qos, sizeof(qos))) {
    return status_from_win32(
        ::GetLastError(),
        "SetProcessInformation(ProcessPowerThrottling)");
  }
  if (!::SetPriorityClass(::GetCurrentProcess(), priority_class)) {
    return status_from_win32(::GetLastError(), "SetPriorityClass");
  }
  return Status::Ok();
#else
  int nice_value = 0;
  switch (priority) {
    case ProcessPriority::kLowest: nice_value = 10; break;
    case ProcessPriority::kLow: nice_value = 5; break;
    case ProcessPriority::kNormal: nice_value = 0; break;
    case ProcessPriority::kHigh: nice_value = -5; break;
    case ProcessPriority::kHighest: nice_value = -10; break;
    default:
      return Status::InvalidArgument("process priority is invalid");
  }
  errno = 0;
  if (::setpriority(PRIO_PROCESS, 0, nice_value) != 0) {
    return status_from_errno(errno, "setpriority(process startup)");
  }
  return Status::Ok();
#endif
}

}  // namespace hyle::platform

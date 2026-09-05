#include <gtest/gtest.h>

#include "hyle/core/environment.h"
#include "hyle/platform/process.h"

#if HYLE_OS_WINDOWS
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <cerrno>
#  include <cstdlib>
#  include <sys/resource.h>
#  include <sys/wait.h>
#  include <unistd.h>
#endif

namespace hyle::platform {
namespace {

TEST(ProcessPriority, RejectsInvalidLevelBeforeCallingTheOS) {
  EXPECT_TRUE(set_process_priority(static_cast<ProcessPriority>(99))
                  .IsInvalidArgument());
}

#if HYLE_OS_WINDOWS
TEST(ProcessPriority, AppliesClassAndExecutionQosWithoutRealtime) {
  ASSERT_TRUE(set_process_priority(ProcessPriority::kLow).ok());
  EXPECT_EQ(::GetPriorityClass(::GetCurrentProcess()),
            BELOW_NORMAL_PRIORITY_CLASS);

  PROCESS_POWER_THROTTLING_STATE qos{};
  qos.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
  ASSERT_TRUE(::GetProcessInformation(
      ::GetCurrentProcess(), ProcessPowerThrottling, &qos, sizeof(qos)));
  EXPECT_NE(qos.ControlMask & PROCESS_POWER_THROTTLING_EXECUTION_SPEED, 0u);
  EXPECT_NE(qos.StateMask & PROCESS_POWER_THROTTLING_EXECUTION_SPEED, 0u);

  // Restore a predictable policy for the remaining tests in this process.
  ASSERT_TRUE(set_process_priority(ProcessPriority::kNormal).ok());
  EXPECT_EQ(::GetPriorityClass(::GetCurrentProcess()), NORMAL_PRIORITY_CLASS);
  qos = {};
  qos.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
  ASSERT_TRUE(::GetProcessInformation(
      ::GetCurrentProcess(), ProcessPowerThrottling, &qos, sizeof(qos)));
  EXPECT_EQ(qos.ControlMask, 0u);
  EXPECT_EQ(qos.StateMask, 0u);
}
#else
TEST(ProcessPriority, AppliesPortableNiceLevelInAnIsolatedChild) {
  const pid_t child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    const Status status = set_process_priority(ProcessPriority::kLow);
    errno = 0;
    const int nice_value = ::getpriority(PRIO_PROCESS, 0);
    const int error = errno;
    ::_exit(status.ok() && error == 0 && nice_value == 5 ? EXIT_SUCCESS
                                                         : EXIT_FAILURE);
  }
  int wait_status = 0;
  ASSERT_EQ(::waitpid(child, &wait_status, 0), child);
  ASSERT_TRUE(WIFEXITED(wait_status));
  EXPECT_EQ(WEXITSTATUS(wait_status), EXIT_SUCCESS);
}
#endif

}  // namespace
}  // namespace hyle::platform

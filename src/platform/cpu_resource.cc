#include "hyle/platform/cpu.h"

#include "cpu_internal.h"
#include "linux_cgroup_internal.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

#include "hyle/core/environment.h"

#if HYLE_OS_WINDOWS
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <powrprof.h>
#elif HYLE_OS_MACOS
#include <sys/sysctl.h>
#endif

namespace hyle::platform {
namespace {

#if HYLE_OS_LINUX
bool parse_u64(const char *text, std::uint64_t &value) noexcept {
  if (text == nullptr)
    return false;
  while (*text == ' ' || *text == '\t')
    ++text;
  if (*text < '0' || *text > '9')
    return false;
  errno = 0;
  char *end = nullptr;
  const std::uint64_t parsed = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text)
    return false;
  while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n')
    ++end;
  if (*end != '\0')
    return false;
  value = parsed;
  return true;
}

bool read_text_file(const char *path, char *output,
                    std::size_t capacity) noexcept {
  if (path == nullptr || output == nullptr || capacity < 2)
    return false;
  std::FILE *file = nullptr;
#if HYLE_OS_WINDOWS
  if (::fopen_s(&file, path, "r") != 0)
    return false;
#else
  file = std::fopen(path, "r");
#endif
  if (file == nullptr)
    return false;
  const bool ok =
      std::fgets(output, static_cast<int>(capacity), file) != nullptr;
  if (ok && std::strchr(output, '\n') == nullptr && !std::feof(file)) {
    std::fclose(file);
    output[0] = '\0';
    return false;
  }
  const bool failed = std::ferror(file) != 0;
  std::fclose(file);
  return ok && !failed;
}

bool make_child_path(char *output, std::size_t capacity, const char *directory,
                     const char *name) noexcept {
  const int written = std::snprintf(output, capacity, "%s/%s", directory, name);
  return written >= 0 && static_cast<std::size_t>(written) < capacity;
}

bool read_linux_quota_file(const char *path, CpuQuota &quota) noexcept {
  char text[192]{};
  if (!read_text_file(path, text, sizeof(text)))
    return false;
  return detail::parse_linux_cpu_max(text, quota);
}

bool read_linux_quota_pair(const char *directory, CpuQuota &quota) noexcept {
  char quota_path[detail::kLinuxCgroupPathCapacity + 32];
  char period_path[detail::kLinuxCgroupPathCapacity + 32];
  if (!make_child_path(quota_path, sizeof(quota_path), directory,
                       "cpu.cfs_quota_us") ||
      !make_child_path(period_path, sizeof(period_path), directory,
                       "cpu.cfs_period_us")) {
    return false;
  }
  char quota_text[96]{};
  char period_text[96]{};
  return read_text_file(quota_path, quota_text, sizeof(quota_text)) &&
         read_text_file(period_path, period_text, sizeof(period_text)) &&
         detail::parse_linux_cpu_quota(quota_text, period_text, quota);
}

std::optional<CpuQuota> linux_hard_cpu_quota() noexcept {
  detail::LinuxCgroupPath path;
  if (!detail::read_linux_cgroup_path("cpu", path))
    return std::nullopt;
  char directory[detail::kLinuxCgroupPathCapacity];
  std::memcpy(directory, path.leaf, sizeof(directory));
  std::optional<CpuQuota> tightest;
  for (;;) {
    CpuQuota candidate;
    bool constrained = false;
    if (path.version == detail::LinuxCgroupVersion::kV2) {
      char file[detail::kLinuxCgroupPathCapacity + 16];
      constrained = make_child_path(file, sizeof(file), directory, "cpu.max") &&
                    read_linux_quota_file(file, candidate);
    } else {
      constrained = read_linux_quota_pair(directory, candidate);
    }
    if (constrained &&
        (!tightest || detail::cpu_quota_tighter(candidate, *tightest))) {
      tightest = candidate;
    }
    if (!detail::linux_cgroup_parent(directory, path.mount_point))
      break;
  }
  return tightest;
}

bool read_linux_khz(const char *path, std::uint64_t &hz) noexcept {
  char text[96]{};
  std::uint64_t khz = 0;
  if (!read_text_file(path, text, sizeof(text)) || !parse_u64(text, khz) ||
      khz > (std::numeric_limits<std::uint64_t>::max)() / 1000u) {
    return false;
  }
  hz = khz * 1000u;
  return true;
}
#endif

#if HYLE_OS_WINDOWS
// Microsoft documents this exact CallNtPowerInformation payload but, as of
// Windows SDK 10.0.26100, still omits PROCESSOR_POWER_INFORMATION from the
// public headers. Keep the wire layout local so it cannot leak into HYLE's API.
struct WindowsProcessorPowerInformation {
  ULONG number;
  ULONG max_mhz;
  ULONG current_mhz;
  ULONG mhz_limit;
  ULONG max_idle_state;
  ULONG current_idle_state;
};
static_assert(sizeof(WindowsProcessorPowerInformation) == 6 * sizeof(ULONG));
#endif

template <class Fill>
Result<std::size_t> with_available_cpu_ids(CpuFrequencyInfo *dst,
                                           std::size_t cap,
                                           Fill &&fill) noexcept {
  if (dst == nullptr)
    cap = 0;
  auto initial = available_cpu_ids(nullptr, 0);
  if (!initial)
    return std::unexpected(std::move(initial).error());
  if (cap == 0)
    return *initial;
  const std::size_t wanted = (std::min)(cap, *initial);
  unsigned stack[256];
  unsigned *ids =
      wanted <= std::size(stack) ? stack : new (std::nothrow) unsigned[wanted];
  if (ids == nullptr) {
    return std::unexpected(Status::ResourceExhausted(
        "CPU frequency id snapshot allocation failed"));
  }
  auto current = available_cpu_ids(ids, wanted);
  if (!current) {
    if (ids != stack)
      delete[] ids;
    return std::unexpected(std::move(current).error());
  }
  const std::size_t written = (std::min)(wanted, *current);
  fill(ids, written, dst);
  if (ids != stack)
    delete[] ids;
  return *current;
}

#if HYLE_OS_WINDOWS
std::uint64_t mhz_to_hz(unsigned long mhz) noexcept {
  constexpr std::uint64_t kScale = 1000000u;
  return static_cast<std::uint64_t>(mhz) <=
                 (std::numeric_limits<std::uint64_t>::max)() / kScale
             ? static_cast<std::uint64_t>(mhz) * kScale
             : 0;
}
#endif

} // namespace

Result<ProcessCpuCapacity> process_cpu_capacity() noexcept {
  HYLE_ASSIGN_OR_RETURN(std::size_t placement, available_cpu_count());
  ProcessCpuCapacity output;
  output.placement_logical_cpus = placement;
#if HYLE_OS_LINUX
  output.hard_quota = linux_hard_cpu_quota();
#elif HYLE_OS_WINDOWS
  // QueryInformationJobObject(nullptr, ...) exposes only the immediate job.
  // Nested CPU rates are relative to their parent, so the documented API
  // cannot establish this process's effective hard ceiling without ancestor
  // handles. Preserve unknown rather than overstate available CPU bandwidth.
#endif
  return output;
}

Result<std::size_t> available_cpu_frequencies(CpuFrequencyInfo *dst,
                                              std::size_t cap) noexcept {
  if (dst == nullptr || cap == 0)
    return available_cpu_ids(nullptr, 0);
#if HYLE_OS_WINDOWS
  const DWORD count = ::GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
  WindowsProcessorPowerInformation stack[256]{};
  WindowsProcessorPowerInformation *power =
      count <= std::size(stack) ? stack
                                : new (std::nothrow)
                                      WindowsProcessorPowerInformation[count]{};
  bool have_power = false;
  if (power != nullptr && count != 0 &&
      count <= (std::numeric_limits<ULONG>::max)() /
                   sizeof(WindowsProcessorPowerInformation)) {
    have_power = ::CallNtPowerInformation(
                     ProcessorInformation, nullptr, 0, power,
                     count * sizeof(WindowsProcessorPowerInformation)) >= 0;
  }
  auto result = with_available_cpu_ids(
      dst, cap,
      [power, count, have_power](const unsigned *ids, std::size_t size,
                                 CpuFrequencyInfo *output) noexcept {
        for (std::size_t i = 0; i < size; ++i) {
          output[i] = CpuFrequencyInfo{};
          output[i].id = ids[i];
          if (!have_power || ids[i] >= count)
            continue;
          const auto &value = power[ids[i]];
          if (value.number != ids[i])
            continue;
          output[i].hardware_max_hz = mhz_to_hz(value.max_mhz);
          output[i].policy_max_hz = mhz_to_hz(value.mhz_limit);
          output[i].current_hz = mhz_to_hz(value.current_mhz);
        }
      });
  if (power != stack)
    delete[] power;
  return result;
#elif HYLE_OS_LINUX
  return with_available_cpu_ids(
      dst, cap,
      [](const unsigned *ids, std::size_t size,
         CpuFrequencyInfo *output) noexcept {
        for (std::size_t i = 0; i < size; ++i) {
          CpuFrequencyInfo info;
          info.id = ids[i];
          char path[160];
          std::snprintf(
              path, sizeof(path),
              "/sys/devices/system/cpu/cpu%u/cpufreq/cpuinfo_max_freq", ids[i]);
          (void)read_linux_khz(path, info.hardware_max_hz);
          std::snprintf(
              path, sizeof(path),
              "/sys/devices/system/cpu/cpu%u/cpufreq/scaling_max_freq", ids[i]);
          (void)read_linux_khz(path, info.policy_max_hz);
          std::snprintf(
              path, sizeof(path),
              "/sys/devices/system/cpu/cpu%u/cpufreq/cpuinfo_cur_freq", ids[i]);
          if (!read_linux_khz(path, info.current_hz)) {
            std::snprintf(
                path, sizeof(path),
                "/sys/devices/system/cpu/cpu%u/cpufreq/scaling_cur_freq",
                ids[i]);
            (void)read_linux_khz(path, info.current_hz);
          }
          output[i] = info;
        }
      });
#elif HYLE_OS_MACOS
  auto read_sysctl_hz = [](const char *name) noexcept {
    std::uint64_t value = 0;
    std::size_t size = sizeof(value);
    return ::sysctlbyname(name, &value, &size, nullptr, 0) == 0 ? value : 0;
  };
  const std::uint64_t maximum = read_sysctl_hz("hw.cpufrequency_max");
  const std::uint64_t current = read_sysctl_hz("hw.cpufrequency");
  return with_available_cpu_ids(
      dst, cap,
      [maximum, current](const unsigned *ids, std::size_t size,
                         CpuFrequencyInfo *output) noexcept {
        for (std::size_t i = 0; i < size; ++i) {
          output[i].id = ids[i];
          output[i].hardware_max_hz = maximum;
          output[i].policy_max_hz = maximum;
          output[i].current_hz = current;
        }
      });
#else
  return with_available_cpu_ids(dst, cap,
                                [](const unsigned *ids, std::size_t size,
                                   CpuFrequencyInfo *output) noexcept {
                                  for (std::size_t i = 0; i < size; ++i)
                                    output[i].id = ids[i];
                                });
#endif
}

} // namespace hyle::platform

#include "hyle/platform/host_memory.h"

#include "host_memory_internal.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "hyle/core/environment.h"

#if HYLE_OS_WINDOWS
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#  include <psapi.h>
#elif HYLE_OS_MACOS
#  include <mach/mach.h>
#  include <sys/resource.h>
#  include <sys/sysctl.h>
#else
#  include <sys/resource.h>
#  include <unistd.h>
#endif

namespace hyle::platform {
namespace {

std::uint64_t remaining(std::uint64_t limit, std::uint64_t used) noexcept {
  return used < limit ? limit - used : 0;
}

void merge_process_budget(HostMemoryInfo& info, std::uint64_t limit,
                          std::uint64_t available) noexcept {
  if (!info.process_limit_bytes || limit < *info.process_limit_bytes) {
    info.process_limit_bytes = limit;
  }
  if (!info.process_available_bytes ||
      available < *info.process_available_bytes) {
    info.process_available_bytes = available;
  }
}

#if HYLE_OS_LINUX

struct LinuxMemInfo {
  std::uint64_t total = 0;
  std::uint64_t available = 0;
  std::uint64_t free = 0;
  std::uint64_t buffers = 0;
  std::uint64_t cached = 0;
  std::uint64_t commit_limit = 0;
  std::uint64_t committed = 0;
  std::uint64_t swap_total = 0;
  std::uint64_t swap_free = 0;
  bool have_available = false;
  bool have_commit_limit = false;
  bool have_committed = false;
  bool have_swap_total = false;
  bool have_swap_free = false;
};

Result<LinuxMemInfo> read_linux_meminfo() noexcept {
  std::FILE* file = std::fopen("/proc/meminfo", "r");
  if (file == nullptr) {
    return std::unexpected(status_from_errno(errno, "open /proc/meminfo"));
  }
  LinuxMemInfo out;
  char line[512];
  while (std::fgets(line, sizeof(line), file) != nullptr) {
    std::uint64_t bytes = 0;
    if (detail::parse_linux_meminfo_kib(line, "MemTotal", bytes)) {
      out.total = bytes;
    } else if (detail::parse_linux_meminfo_kib(line, "MemAvailable", bytes)) {
      out.available = bytes;
      out.have_available = true;
    } else if (detail::parse_linux_meminfo_kib(line, "MemFree", bytes)) {
      out.free = bytes;
    } else if (detail::parse_linux_meminfo_kib(line, "Buffers", bytes)) {
      out.buffers = bytes;
    } else if (detail::parse_linux_meminfo_kib(line, "Cached", bytes)) {
      out.cached = bytes;
    } else if (detail::parse_linux_meminfo_kib(line, "CommitLimit", bytes)) {
      out.commit_limit = bytes;
      out.have_commit_limit = true;
    } else if (detail::parse_linux_meminfo_kib(line, "Committed_AS", bytes)) {
      out.committed = bytes;
      out.have_committed = true;
    } else if (detail::parse_linux_meminfo_kib(line, "SwapTotal", bytes)) {
      out.swap_total = bytes;
      out.have_swap_total = true;
    } else if (detail::parse_linux_meminfo_kib(line, "SwapFree", bytes)) {
      out.swap_free = bytes;
      out.have_swap_free = true;
    }
  }
  const int saved = errno;
  const bool io_error = std::ferror(file) != 0;
  std::fclose(file);
  if (io_error) {
    return std::unexpected(status_from_errno(saved, "read /proc/meminfo"));
  }
  if (out.total == 0) {
    return std::unexpected(
        Status::DataLoss("/proc/meminfo has no usable MemTotal"));
  }
  if (!out.have_available) {
    const std::uint64_t reclaimable =
        out.free + (out.buffers <= (std::numeric_limits<std::uint64_t>::max)() - out.free
                        ? out.buffers
                        : 0);
    out.available =
        out.cached <= (std::numeric_limits<std::uint64_t>::max)() - reclaimable
            ? reclaimable + out.cached
            : out.free;
  }
  out.available = (std::min)(out.available, out.total);
  return out;
}

bool read_u64_file(const char* path, std::uint64_t& value,
                   bool& unlimited) noexcept {
  std::FILE* file = std::fopen(path, "r");
  if (file == nullptr) return false;
  char text[96]{};
  const bool ok = std::fgets(text, sizeof(text), file) != nullptr;
  std::fclose(file);
  if (!ok) return false;
  if (std::strncmp(text, "max", 3) == 0) {
    unlimited = true;
    return true;
  }
  char* end = nullptr;
  errno = 0;
  const std::uint64_t parsed = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text) return false;
  value = parsed;
  unlimited = false;
  return true;
}

bool cgroup_path(char* output, std::size_t capacity, bool& v2) noexcept {
  std::FILE* file = std::fopen("/proc/self/cgroup", "r");
  if (file == nullptr) return false;
  char line[4096];
  bool found = false;
  while (std::fgets(line, sizeof(line), file) != nullptr) {
    char* first = std::strchr(line, ':');
    if (first == nullptr) continue;
    char* second = std::strchr(first + 1, ':');
    if (second == nullptr) continue;
    *second = '\0';
    const char* controllers = first + 1;
    char* path = second + 1;
    path[std::strcspn(path, "\r\n")] = '\0';
    const bool unified = controllers[0] == '\0';
    const bool memory = std::strstr(controllers, "memory") != nullptr;
    if (!unified && !memory) continue;
    const int written = std::snprintf(output, capacity, "%s", path);
    if (written < 0 || static_cast<std::size_t>(written) >= capacity) continue;
    v2 = unified;
    found = true;
    if (unified) break;
  }
  std::fclose(file);
  return found;
}

void add_linux_cgroup_budget(HostMemoryInfo& info) noexcept {
  char group[4096];
  bool v2 = false;
  if (!cgroup_path(group, sizeof(group), v2)) return;
  char limit_path[4608];
  char current_path[4608];
  const int limit_written = std::snprintf(
      limit_path, sizeof(limit_path),
      v2 ? "/sys/fs/cgroup%s/memory.max"
         : "/sys/fs/cgroup/memory%s/memory.limit_in_bytes",
      group);
  const int current_written = std::snprintf(
      current_path, sizeof(current_path),
      v2 ? "/sys/fs/cgroup%s/memory.current"
         : "/sys/fs/cgroup/memory%s/memory.usage_in_bytes",
      group);
  if (limit_written < 0 || current_written < 0 ||
      static_cast<std::size_t>(limit_written) >= sizeof(limit_path) ||
      static_cast<std::size_t>(current_written) >= sizeof(current_path)) {
    return;
  }
  std::uint64_t limit = 0, current = 0;
  bool unlimited = false, current_unlimited = false;
  if (!read_u64_file(limit_path, limit, unlimited) || unlimited ||
      !read_u64_file(current_path, current, current_unlimited) ||
      current_unlimited) {
    return;
  }
  // cgroup v1 commonly reports a huge sentinel for "unlimited".
  if (!v2 && limit > info.physical_total_bytes &&
      limit - info.physical_total_bytes > info.physical_total_bytes) {
    return;
  }
  merge_process_budget(info, limit, remaining(limit, current));
}

std::optional<std::uint64_t> linux_virtual_bytes() noexcept {
  std::FILE* file = std::fopen("/proc/self/statm", "r");
  if (file == nullptr) return std::nullopt;
  std::uint64_t pages = 0;
  const bool ok = std::fscanf(file, "%" SCNu64, &pages) == 1;
  std::fclose(file);
  const long page = ::sysconf(_SC_PAGESIZE);
  if (!ok || page <= 0 ||
      pages > (std::numeric_limits<std::uint64_t>::max)() /
                  static_cast<std::uint64_t>(page)) {
    return std::nullopt;
  }
  return pages * static_cast<std::uint64_t>(page);
}

#endif  // HYLE_OS_LINUX

}  // namespace

Result<HostMemoryInfo> host_memory_info() noexcept {
  HostMemoryInfo info;
#if HYLE_OS_WINDOWS
  MEMORYSTATUSEX state{};
  state.dwLength = sizeof(state);
  if (!::GlobalMemoryStatusEx(&state)) {
    return std::unexpected(
        status_from_win32(::GetLastError(), "GlobalMemoryStatusEx"));
  }
  info.physical_total_bytes = state.ullTotalPhys;
  info.physical_available_bytes = state.ullAvailPhys;
  info.commit_limit_bytes = state.ullTotalPageFile;
  info.commit_available_bytes = state.ullAvailPageFile;

  BOOL in_job = FALSE;
  if (::IsProcessInJob(::GetCurrentProcess(), nullptr, &in_job) && in_job) {
    std::uint64_t current_process_commit = 0;
    PROCESS_MEMORY_COUNTERS_EX process_memory{};
    process_memory.cb = sizeof(process_memory);
    if (::GetProcessMemoryInfo(
            ::GetCurrentProcess(),
            reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&process_memory),
            sizeof(process_memory))) {
      current_process_commit = process_memory.PrivateUsage;
    }

    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    if (::QueryInformationJobObject(nullptr, JobObjectExtendedLimitInformation,
                                    &limits, sizeof(limits), nullptr)) {
      const DWORD flags = limits.BasicLimitInformation.LimitFlags;
      if ((flags & JOB_OBJECT_LIMIT_PROCESS_MEMORY) != 0) {
        const std::uint64_t used = current_process_commit != 0
                                       ? current_process_commit
                                       : limits.PeakProcessMemoryUsed;
        merge_process_budget(
            info, limits.ProcessMemoryLimit,
            remaining(limits.ProcessMemoryLimit, used));
      }
      if ((flags & JOB_OBJECT_LIMIT_JOB_MEMORY) != 0) {
        std::uint64_t used = limits.PeakJobMemoryUsed;
        JOBOBJECT_LIMIT_VIOLATION_INFORMATION_2 live{};
        if (::QueryInformationJobObject(
                nullptr, JobObjectLimitViolationInformation2, &live,
                sizeof(live), nullptr)) {
          used = live.JobMemory;
        }
        // Older hosts may not expose live job usage. Peak is a conservative
        // fallback: it can understate budget but can never overcommit the job.
        merge_process_budget(
            info, limits.JobMemoryLimit,
            remaining(limits.JobMemoryLimit, used));
      }
    }
  }
#elif HYLE_OS_LINUX
  HYLE_ASSIGN_OR_RETURN(LinuxMemInfo native, read_linux_meminfo());
  info.physical_total_bytes = native.total;
  info.physical_available_bytes = native.available;
  if (native.have_commit_limit) info.commit_limit_bytes = native.commit_limit;
  if (native.have_commit_limit && native.have_committed) {
    info.commit_available_bytes =
        remaining(native.commit_limit, native.committed);
  }
  if (native.have_swap_total) info.swap_total_bytes = native.swap_total;
  if (native.have_swap_free) info.swap_available_bytes = native.swap_free;
  add_linux_cgroup_budget(info);

  struct rlimit limit {};
  if (::getrlimit(RLIMIT_AS, &limit) == 0 && limit.rlim_cur != RLIM_INFINITY) {
    const std::uint64_t bytes = static_cast<std::uint64_t>(limit.rlim_cur);
    const auto used = linux_virtual_bytes();
    if (used) merge_process_budget(info, bytes, remaining(bytes, *used));
  }
#elif HYLE_OS_MACOS
  std::uint64_t total = 0;
  std::size_t total_size = sizeof(total);
  if (::sysctlbyname("hw.memsize", &total, &total_size, nullptr, 0) != 0 ||
      total == 0) {
    return std::unexpected(status_from_errno(errno, "sysctl hw.memsize"));
  }
  mach_port_t host = ::mach_host_self();
  vm_size_t page = 0;
  vm_statistics64_data_t stats{};
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  const kern_return_t page_result = ::host_page_size(host, &page);
  const kern_return_t stats_result = ::host_statistics64(
      host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&stats), &count);
  ::mach_port_deallocate(::mach_task_self(), host);
  if (page_result != KERN_SUCCESS || stats_result != KERN_SUCCESS) {
    return std::unexpected(Status::Unavailable("host VM statistics failed"));
  }
  info.physical_total_bytes = total;
  const std::uint64_t available_pages =
      static_cast<std::uint64_t>(stats.free_count) + stats.inactive_count;
  info.physical_available_bytes =
      available_pages > (std::numeric_limits<std::uint64_t>::max)() / page
          ? total
          : (std::min)(total, available_pages * page);

  struct xsw_usage swap {};
  std::size_t swap_size = sizeof(swap);
  if (::sysctlbyname("vm.swapusage", &swap, &swap_size, nullptr, 0) == 0) {
    info.swap_total_bytes = swap.xsu_total;
    info.swap_available_bytes = swap.xsu_avail;
  }
  struct rlimit limit {};
  if (::getrlimit(RLIMIT_AS, &limit) == 0 && limit.rlim_cur != RLIM_INFINITY) {
    mach_task_basic_info_data_t task{};
    mach_msg_type_number_t task_count = MACH_TASK_BASIC_INFO_COUNT;
    const kern_return_t task_result = ::task_info(
        ::mach_task_self(), MACH_TASK_BASIC_INFO,
        reinterpret_cast<task_info_t>(&task), &task_count);
    const std::uint64_t bytes = static_cast<std::uint64_t>(limit.rlim_cur);
    if (task_result == KERN_SUCCESS) {
      merge_process_budget(info, bytes,
                           remaining(bytes, task.virtual_size));
    }
  }
#endif
  return info;
}

}  // namespace hyle::platform

#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/visibility.h"

namespace hyle::platform::detail {

inline constexpr std::size_t kLinuxCgroupPathCapacity = 4096;

enum class LinuxCgroupVersion : std::uint8_t {
  kNone = 0,
  kV1 = 1,
  kV2 = 2,
};

struct LinuxCgroupMembership {
  LinuxCgroupVersion version = LinuxCgroupVersion::kNone;
  char path[kLinuxCgroupPathCapacity]{};
};

struct LinuxCgroupMount {
  LinuxCgroupVersion version = LinuxCgroupVersion::kNone;
  char root[kLinuxCgroupPathCapacity]{};
  char mount_point[kLinuxCgroupPathCapacity]{};
};

struct LinuxCgroupPath {
  LinuxCgroupVersion version = LinuxCgroupVersion::kNone;
  char leaf[kLinuxCgroupPathCapacity]{};
  char mount_point[kLinuxCgroupPathCapacity]{};
};

// Pure line parsers used by the production /proc walk and by platform-neutral
// fixtures. controller must be one exact v1 token ("memory", "cpu", ...).
HYLE_HIDDEN bool
parse_linux_cgroup_membership_line(const char *line, const char *controller,
                                   LinuxCgroupMembership &output) noexcept;
HYLE_HIDDEN bool
parse_linux_cgroup_mountinfo_line(const char *line, LinuxCgroupVersion version,
                                  const char *controller,
                                  LinuxCgroupMount &output) noexcept;

// Map a cgroup namespace path through mountinfo's root/mount-point pair.
// Rejects traversal and truncation instead of constructing a best-guess path.
HYLE_HIDDEN bool
resolve_linux_cgroup_directory(const LinuxCgroupMembership &membership,
                               const LinuxCgroupMount &mount,
                               LinuxCgroupPath &output) noexcept;

// Resolve the current process' controller directory from /proc. False means
// the controller/mount/path was unavailable or untrustworthy.
HYLE_HIDDEN bool read_linux_cgroup_path(const char *controller,
                                        LinuxCgroupPath &output) noexcept;

// Move directory to its parent without crossing mount_point. Returns false
// when directory already names the visible mount root or is malformed.
HYLE_HIDDEN bool linux_cgroup_parent(char *directory,
                                     const char *mount_point) noexcept;

} // namespace hyle::platform::detail

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>

#include "linux_cgroup_internal.h"

namespace hyle::platform::detail {
namespace {

template <std::size_t N>
void set_text(char (&output)[N], const char* text) {
  const int written = std::snprintf(output, N, "%s", text);
  ASSERT_GE(written, 0);
  ASSERT_LT(static_cast<std::size_t>(written), N);
}

TEST(LinuxCgroupParser, SelectsExactV1ControllerAndUnifiedV2) {
  LinuxCgroupMembership membership;
  ASSERT_TRUE(parse_linux_cgroup_membership_line("0::/workload/a\n", "memory",
                                                 membership));
  EXPECT_EQ(membership.version, LinuxCgroupVersion::kV2);
  EXPECT_STREQ(membership.path, "/workload/a");

  ASSERT_TRUE(parse_linux_cgroup_membership_line("4:cpu,cpuacct:/batch\n",
                                                 "cpu", membership));
  EXPECT_EQ(membership.version, LinuxCgroupVersion::kV1);
  EXPECT_STREQ(membership.path, "/batch");
  EXPECT_FALSE(parse_linux_cgroup_membership_line("5:cpuacct:/batch\n", "cpu",
                                                  membership));
  EXPECT_FALSE(parse_linux_cgroup_membership_line("6:notmemory:/batch\n",
                                                  "memory", membership));
  EXPECT_FALSE(parse_linux_cgroup_membership_line("0::/work/../escape\n", "cpu",
                                                  membership));
}

TEST(LinuxCgroupParser, ReadsMountRootPointAndController) {
  LinuxCgroupMount mount;
  ASSERT_TRUE(parse_linux_cgroup_mountinfo_line(
      "36 29 0:32 /docker/id /sys/fs/cgroup rw,nosuid - cgroup2 cgroup rw\n",
      LinuxCgroupVersion::kV2, "memory", mount));
  EXPECT_STREQ(mount.root, "/docker/id");
  EXPECT_STREQ(mount.mount_point, "/sys/fs/cgroup");

  ASSERT_TRUE(parse_linux_cgroup_mountinfo_line(
      "41 29 0:37 / /sys/fs/cgroup/cpu rw - cgroup cgroup rw,cpu,cpuacct\n",
      LinuxCgroupVersion::kV1, "cpu", mount));
  EXPECT_STREQ(mount.root, "/");
  EXPECT_STREQ(mount.mount_point, "/sys/fs/cgroup/cpu");
  EXPECT_FALSE(parse_linux_cgroup_mountinfo_line(
      "41 29 0:37 / /sys/fs/cgroup/cpu rw - cgroup cgroup rw,cpuacct\n",
      LinuxCgroupVersion::kV1, "cpu", mount));

  ASSERT_TRUE(parse_linux_cgroup_mountinfo_line(
      "36 29 0:32 / /sys/fs/cgroup\\040space rw - cgroup2 cgroup rw\n",
      LinuxCgroupVersion::kV2, "memory", mount));
  EXPECT_STREQ(mount.mount_point, "/sys/fs/cgroup space");
  EXPECT_FALSE(parse_linux_cgroup_mountinfo_line(
      "36 29 0:32 / /sys/fs/cgroup\\057escape rw - cgroup2 cgroup rw\n",
      LinuxCgroupVersion::kV2, "memory", mount));
  EXPECT_FALSE(parse_linux_cgroup_mountinfo_line(
      "36 29 0:32 / /sys/fs/cgroup\\141escape rw - cgroup2 cgroup rw\n",
      LinuxCgroupVersion::kV2, "memory", mount));
}

TEST(LinuxCgroupParser, ResolvesNamespacesAndBoundsAncestorWalk) {
  LinuxCgroupMembership membership;
  membership.version = LinuxCgroupVersion::kV2;
  set_text(membership.path, "/docker/id/work/a");
  LinuxCgroupMount mount;
  mount.version = LinuxCgroupVersion::kV2;
  set_text(mount.root, "/docker/id");
  set_text(mount.mount_point, "/sys/fs/cgroup");

  LinuxCgroupPath path;
  ASSERT_TRUE(resolve_linux_cgroup_directory(membership, mount, path));
  EXPECT_STREQ(path.leaf, "/sys/fs/cgroup/work/a");
  EXPECT_STREQ(path.mount_point, "/sys/fs/cgroup");
  EXPECT_TRUE(linux_cgroup_parent(path.leaf, path.mount_point));
  EXPECT_STREQ(path.leaf, "/sys/fs/cgroup/work");
  EXPECT_TRUE(linux_cgroup_parent(path.leaf, path.mount_point));
  EXPECT_STREQ(path.leaf, "/sys/fs/cgroup");
  EXPECT_FALSE(linux_cgroup_parent(path.leaf, path.mount_point));

  set_text(membership.path, "/");
  ASSERT_TRUE(resolve_linux_cgroup_directory(membership, mount, path));
  EXPECT_STREQ(path.leaf, "/sys/fs/cgroup");

  set_text(membership.path, "/namespace/child");
  EXPECT_FALSE(resolve_linux_cgroup_directory(membership, mount, path));

  set_text(mount.root, "/");
  ASSERT_TRUE(resolve_linux_cgroup_directory(membership, mount, path));
  EXPECT_STREQ(path.leaf, "/sys/fs/cgroup/namespace/child");

  set_text(membership.path, "/../escape");
  EXPECT_FALSE(resolve_linux_cgroup_directory(membership, mount, path));
}

}  // namespace
}  // namespace hyle::platform::detail

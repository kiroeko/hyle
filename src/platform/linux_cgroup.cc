#include "linux_cgroup_internal.h"

#include <cstdio>
#include <cstring>
#include <iterator>

#include "hyle/core/environment.h"

#if HYLE_OS_LINUX
#include <unistd.h>
#endif

namespace hyle::platform::detail {
namespace {

bool copy_text(char *output, std::size_t capacity, const char *begin,
               const char *end) noexcept {
  if (output == nullptr || capacity == 0 || begin == nullptr || end < begin) {
    return false;
  }
  const std::size_t length = static_cast<std::size_t>(end - begin);
  if (length >= capacity)
    return false;
  std::memcpy(output, begin, length);
  output[length] = '\0';
  return true;
}

bool token_present(const char *begin, const char *end,
                   const char *token) noexcept {
  if (begin == nullptr || end == nullptr || token == nullptr ||
      token[0] == '\0') {
    return false;
  }
  const std::size_t wanted = std::strlen(token);
  const char *cursor = begin;
  while (cursor < end) {
    const char *next = cursor;
    while (next < end && *next != ',')
      ++next;
    if (static_cast<std::size_t>(next - cursor) == wanted &&
        std::memcmp(cursor, token, wanted) == 0) {
      return true;
    }
    cursor = next < end ? next + 1 : end;
  }
  return false;
}

bool canonical_absolute_path(const char *path) noexcept {
  if (path == nullptr || path[0] != '/')
    return false;
  const char *segment = path + 1;
  for (const char *cursor = segment;; ++cursor) {
    if (*cursor != '/' && *cursor != '\0')
      continue;
    const std::size_t length = static_cast<std::size_t>(cursor - segment);
    if ((length == 1 && segment[0] == '.') ||
        (length == 2 && segment[0] == '.' && segment[1] == '.')) {
      return false;
    }
    if (*cursor == '\0')
      return true;
    // Empty interior path components are non-canonical. A single trailing '/'
    // is accepted and normalized by the mountinfo decoder.
    if (length == 0 && cursor[1] != '\0')
      return false;
    segment = cursor + 1;
  }
}

bool decode_mountinfo_path(const char *encoded, char *output,
                           std::size_t capacity) noexcept {
  if (encoded == nullptr || output == nullptr || capacity == 0)
    return false;
  std::size_t written = 0;
  for (std::size_t i = 0; encoded[i] != '\0'; ++i) {
    unsigned char value = static_cast<unsigned char>(encoded[i]);
    if (value == '\\') {
      if (encoded[i + 1] == '\0' || encoded[i + 2] == '\0' ||
          encoded[i + 3] == '\0' || encoded[i + 1] < '0' ||
          encoded[i + 1] > '7' || encoded[i + 2] < '0' ||
          encoded[i + 2] > '7' || encoded[i + 3] < '0' ||
          encoded[i + 3] > '7') {
        return false;
      }
      value = static_cast<unsigned char>(((encoded[i + 1] - '0') << 6) |
                                         ((encoded[i + 2] - '0') << 3) |
                                         (encoded[i + 3] - '0'));
      // mountinfo defines exactly these four escapes. Accepting arbitrary
      // octal bytes would let a malformed fixture hide path syntax from the
      // canonical-path check.
      if (value != ' ' && value != '\t' && value != '\n' && value != '\\') {
        return false;
      }
      i += 3;
    }
    if (written + 1 >= capacity)
      return false;
    output[written++] = static_cast<char>(value);
  }
  while (written > 1 && output[written - 1] == '/')
    --written;
  output[written] = '\0';
  return canonical_absolute_path(output);
}

bool path_prefix(const char *path, const char *prefix) noexcept {
  const std::size_t length = std::strlen(prefix);
  if (std::strncmp(path, prefix, length) != 0)
    return false;
  return length == 1 || path[length] == '\0' || path[length] == '/';
}

#if HYLE_OS_LINUX
bool line_is_complete(std::FILE *file, const char *line) noexcept {
  if (std::strchr(line, '\n') != nullptr || std::feof(file))
    return true;
  // The record exceeded the parser's audited bound. Consume the remainder so
  // the next fgets starts on a record boundary, but never parse this prefix.
  int ch = 0;
  do {
    ch = std::fgetc(file);
  } while (ch != '\n' && ch != EOF);
  return false;
}
#endif

} // namespace

bool parse_linux_cgroup_membership_line(
    const char *line, const char *controller,
    LinuxCgroupMembership &output) noexcept {
  if (line == nullptr || controller == nullptr || controller[0] == '\0') {
    return false;
  }
  const char *first = std::strchr(line, ':');
  if (first == nullptr)
    return false;
  const char *second = std::strchr(first + 1, ':');
  if (second == nullptr)
    return false;
  const bool unified = second == first + 1;
  if (!unified && !token_present(first + 1, second, controller))
    return false;

  const char *begin = second + 1;
  const char *end = begin + std::strcspn(begin, "\r\n");
  LinuxCgroupMembership parsed;
  parsed.version = unified ? LinuxCgroupVersion::kV2 : LinuxCgroupVersion::kV1;
  if (!copy_text(parsed.path, sizeof(parsed.path), begin, end) ||
      !canonical_absolute_path(parsed.path)) {
    return false;
  }
  output = parsed;
  return true;
}

bool parse_linux_cgroup_mountinfo_line(const char *line,
                                       LinuxCgroupVersion version,
                                       const char *controller,
                                       LinuxCgroupMount &output) noexcept {
  if (line == nullptr || controller == nullptr || controller[0] == '\0' ||
      version == LinuxCgroupVersion::kNone) {
    return false;
  }
  char copy[16384];
  const std::size_t length = std::strlen(line);
  if (length >= sizeof(copy))
    return false;
  std::memcpy(copy, line, length + 1);

  char *fields[128]{};
  std::size_t count = 0;
  char *cursor = copy;
  while (*cursor != '\0' && count < std::size(fields)) {
    while (*cursor == ' ')
      ++cursor;
    if (*cursor == '\0' || *cursor == '\r' || *cursor == '\n')
      break;
    fields[count++] = cursor;
    while (*cursor != '\0' && *cursor != ' ' && *cursor != '\r' &&
           *cursor != '\n') {
      ++cursor;
    }
    if (*cursor == '\0')
      break;
    *cursor++ = '\0';
  }
  while (*cursor == ' ')
    ++cursor;
  if (*cursor != '\0' && *cursor != '\r' && *cursor != '\n')
    return false;
  if (count < 10)
    return false;
  std::size_t separator = 6;
  while (separator < count && std::strcmp(fields[separator], "-") != 0) {
    ++separator;
  }
  if (separator + 3 >= count)
    return false;
  const char *expected =
      version == LinuxCgroupVersion::kV2 ? "cgroup2" : "cgroup";
  if (std::strcmp(fields[separator + 1], expected) != 0)
    return false;
  if (version == LinuxCgroupVersion::kV1) {
    const auto has = [controller](const char *value) noexcept {
      return token_present(value, value + std::strlen(value), controller);
    };
    if (!has(fields[5]) && !has(fields[separator + 2]) &&
        !has(fields[separator + 3])) {
      return false;
    }
  }

  LinuxCgroupMount parsed;
  parsed.version = version;
  if (!decode_mountinfo_path(fields[3], parsed.root, sizeof(parsed.root)) ||
      !decode_mountinfo_path(fields[4], parsed.mount_point,
                             sizeof(parsed.mount_point))) {
    return false;
  }
  output = parsed;
  return true;
}

bool resolve_linux_cgroup_directory(const LinuxCgroupMembership &membership,
                                    const LinuxCgroupMount &mount,
                                    LinuxCgroupPath &output) noexcept {
  if (membership.version == LinuxCgroupVersion::kNone ||
      membership.version != mount.version ||
      !canonical_absolute_path(membership.path) ||
      !canonical_absolute_path(mount.root) ||
      !canonical_absolute_path(mount.mount_point)) {
    return false;
  }

  const char *suffix = membership.path;
  if (std::strcmp(membership.path, "/") == 0) {
    suffix = "";
  } else if (std::strcmp(mount.root, "/") != 0) {
    if (!path_prefix(membership.path, mount.root))
      return false;
    suffix = membership.path + std::strlen(mount.root);
  }

  LinuxCgroupPath resolved;
  resolved.version = membership.version;
  const std::size_t mount_length = std::strlen(mount.mount_point);
  const std::size_t suffix_length = std::strlen(suffix);
  const bool add_separator =
      suffix_length != 0 && suffix[0] != '/' && mount_length != 1;
  if (mount_length + (add_separator ? 1u : 0u) + suffix_length >=
      sizeof(resolved.leaf)) {
    return false;
  }
  std::memcpy(resolved.leaf, mount.mount_point, mount_length);
  std::size_t written = mount_length;
  if (add_separator)
    resolved.leaf[written++] = '/';
  if (mount_length == 1 && resolved.leaf[0] == '/' && suffix[0] == '/') {
    ++suffix;
  }
  const std::size_t adjusted_suffix = std::strlen(suffix);
  std::memcpy(resolved.leaf + written, suffix, adjusted_suffix);
  written += adjusted_suffix;
  resolved.leaf[written] = '\0';
  if (!copy_text(resolved.mount_point, sizeof(resolved.mount_point),
                 mount.mount_point, mount.mount_point + mount_length) ||
      !canonical_absolute_path(resolved.leaf) ||
      !path_prefix(resolved.leaf, resolved.mount_point)) {
    return false;
  }
  output = resolved;
  return true;
}

bool read_linux_cgroup_path(const char *controller,
                            LinuxCgroupPath &output) noexcept {
#if HYLE_OS_LINUX
  if (controller == nullptr || controller[0] == '\0')
    return false;
  std::FILE *memberships = std::fopen("/proc/self/cgroup", "r");
  if (memberships == nullptr)
    return false;
  LinuxCgroupMembership selected;
  LinuxCgroupMembership unified;
  char line[8192];
  while (std::fgets(line, sizeof(line), memberships) != nullptr) {
    if (!line_is_complete(memberships, line))
      continue;
    LinuxCgroupMembership parsed;
    if (!parse_linux_cgroup_membership_line(line, controller, parsed)) {
      continue;
    }
    if (parsed.version == LinuxCgroupVersion::kV1) {
      selected = parsed;
      break;
    }
    unified = parsed;
  }
  std::fclose(memberships);
  if (selected.version == LinuxCgroupVersion::kNone)
    selected = unified;
  if (selected.version == LinuxCgroupVersion::kNone)
    return false;

  std::FILE *mounts = std::fopen("/proc/self/mountinfo", "r");
  if (mounts == nullptr)
    return false;
  bool found = false;
  char mount_line[16384];
  while (std::fgets(mount_line, sizeof(mount_line), mounts) != nullptr) {
    if (!line_is_complete(mounts, mount_line))
      continue;
    LinuxCgroupMount mount;
    if (!parse_linux_cgroup_mountinfo_line(mount_line, selected.version,
                                           controller, mount)) {
      continue;
    }
    LinuxCgroupPath resolved;
    if (!resolve_linux_cgroup_directory(selected, mount, resolved))
      continue;
    if (::access(resolved.leaf, F_OK) != 0)
      continue;
    output = resolved;
    found = true;
    break;
  }
  std::fclose(mounts);
  return found;
#else
  (void)controller;
  (void)output;
  return false;
#endif
}

bool linux_cgroup_parent(char *directory, const char *mount_point) noexcept {
  if (!canonical_absolute_path(directory) ||
      !canonical_absolute_path(mount_point) ||
      !path_prefix(directory, mount_point) ||
      std::strcmp(directory, mount_point) == 0) {
    return false;
  }
  const std::size_t mount_length = std::strlen(mount_point);
  char *slash = std::strrchr(directory, '/');
  if (slash == nullptr)
    return false;
  if (mount_length == 1 && slash == directory) {
    directory[1] = '\0';
    return true;
  }
  if (static_cast<std::size_t>(slash - directory) <= mount_length) {
    directory[mount_length] = '\0';
    return true;
  }
  *slash = '\0';
  return true;
}

} // namespace hyle::platform::detail

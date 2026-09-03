#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace hyle::platform::detail {

// Parse one exact Linux /proc/meminfo key whose value is expressed in kB.
// Kept platform-neutral so malformed/overflow behavior is covered on every CI
// host even though the production caller exists only on Linux.
inline bool parse_linux_meminfo_kib(const char* line, const char* key,
                                    std::uint64_t& bytes) noexcept {
  if (line == nullptr || key == nullptr || key[0] == '\0') return false;
  const std::size_t key_size = std::strlen(key);
  if (std::strncmp(line, key, key_size) != 0 || line[key_size] != ':') {
    return false;
  }
  const char* cursor = line + key_size + 1u;
  while (*cursor == ' ' || *cursor == '\t') ++cursor;
  errno = 0;
  char* end = nullptr;
  const std::uint64_t kib = std::strtoull(cursor, &end, 10);
  if (errno != 0 || end == cursor) return false;
  while (*end == ' ' || *end == '\t') ++end;
  if (end[0] != 'k' || end[1] != 'B') return false;
  end += 2;
  while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') ++end;
  if (*end != '\0' ||
      kib > (std::numeric_limits<std::uint64_t>::max)() / 1024u) {
    return false;
  }
  bytes = kib * 1024u;
  return true;
}

}  // namespace hyle::platform::detail

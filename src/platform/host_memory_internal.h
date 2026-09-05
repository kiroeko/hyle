#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "hyle/visibility.h"

namespace hyle::platform::detail {

// Parse one exact Linux /proc/meminfo key whose value is expressed in kB.
// Kept platform-neutral so malformed/overflow behavior is covered on every CI
// host even though the production caller exists only on Linux.
HYLE_HIDDEN inline bool parse_linux_meminfo_kib(const char* line,
                                                const char* key,
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

// Parse one cgroup scalar. cgroup v2 spells an absent hard ceiling "max";
// cgroup v1 uses a very large numeric page-counter sentinel instead.
HYLE_HIDDEN inline bool parse_linux_cgroup_u64(const char* text,
                                               std::uint64_t& value,
                                               bool& unlimited) noexcept {
  if (text == nullptr) return false;
  while (*text == ' ' || *text == '\t') ++text;
  if (std::strncmp(text, "max", 3) == 0) {
    const char* end = text + 3;
    while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') {
      ++end;
    }
    if (*end != '\0') return false;
    unlimited = true;
    value = 0;
    return true;
  }
  if (*text < '0' || *text > '9') return false;
  errno = 0;
  char* end = nullptr;
  const std::uint64_t parsed = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text) return false;
  while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') ++end;
  if (*end != '\0') return false;
  value = parsed;
  unlimited = false;
  return true;
}

HYLE_HIDDEN inline bool linux_v1_memory_limit_is_unlimited(
    std::uint64_t value) noexcept {
  // PAGE_COUNTER_MAX is LONG_MAX rounded down to a page boundary. Allow the
  // largest 1 MiB window so page size differences do not turn its sentinel
  // into an impossible exabyte-scale process limit.
  constexpr std::uint64_t kLowestPageCounterMax =
      static_cast<std::uint64_t>((std::numeric_limits<std::int64_t>::max)()) -
      (std::uint64_t{1} << 20);
  return value >= kLowestPageCounterMax;
}

}  // namespace hyle::platform::detail

#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "host_memory_internal.h"
#include "hyle/platform/numa.h"

namespace hyle::platform::detail {

// Parse one exact Linux node meminfo line, for example
// "Node 7 MemTotal: 123 kB". Requiring the requested node prefix prevents a
// fixture or future alternate source from silently mixing node identities;
// the shared meminfo parser owns unit and overflow validation.
HYLE_HIDDEN inline bool parse_linux_node_meminfo_kib(
    const char* line, unsigned node, const char* key,
    std::uint64_t& bytes) noexcept {
  if (line == nullptr || key == nullptr || key[0] == '\0' ||
      std::strncmp(line, "Node ", 5) != 0) {
    return false;
  }
  const char* cursor = line + 5;
  if (*cursor < '0' || *cursor > '9') return false;
  errno = 0;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(cursor, &end, 10);
  if (errno != 0 || end == cursor || *end != ' ' ||
      parsed > (std::numeric_limits<unsigned>::max)() ||
      static_cast<unsigned>(parsed) != node) {
    return false;
  }
  return parse_linux_meminfo_kib(end + 1, key, bytes);
}

// Parse Linux's canonical comma/range list (for example "0-2,7") into a
// sorted unique caller buffer. maximum_id is an explicit consumer bound.
// Partial output on false is unspecified; count is set only on success.
HYLE_HIDDEN inline bool parse_linux_node_list(const char* text,
                                              unsigned maximum_id,
                                              unsigned* ids,
                                              std::size_t capacity,
                                              std::size_t& count) noexcept {
  if (text == nullptr || (ids == nullptr && capacity != 0)) return false;
  const char* cursor = text;
  while (*cursor == ' ' || *cursor == '\t') ++cursor;
  if (*cursor == '\0' || *cursor == '\r' || *cursor == '\n') return false;

  std::size_t written = 0;
  unsigned previous = 0;
  bool have_previous = false;
  for (;;) {
    if (*cursor < '0' || *cursor > '9') return false;
    unsigned first = 0;
    do {
      const unsigned digit = static_cast<unsigned>(*cursor++ - '0');
      if (first > ((std::numeric_limits<unsigned>::max)() - digit) / 10u) {
        return false;
      }
      first = first * 10u + digit;
    } while (*cursor >= '0' && *cursor <= '9');

    unsigned last = first;
    if (*cursor == '-') {
      ++cursor;
      if (*cursor < '0' || *cursor > '9') return false;
      last = 0;
      do {
        const unsigned digit = static_cast<unsigned>(*cursor++ - '0');
        if (last > ((std::numeric_limits<unsigned>::max)() - digit) / 10u) {
          return false;
        }
        last = last * 10u + digit;
      } while (*cursor >= '0' && *cursor <= '9');
      if (last < first) return false;
    }
    if (last > maximum_id || (have_previous && first <= previous)) {
      return false;
    }
    const std::size_t range = static_cast<std::size_t>(last - first) + 1u;
    if (range > capacity - written) return false;
    for (unsigned node = first;; ++node) {
      ids[written++] = node;
      if (node == last) break;
    }
    previous = last;
    have_previous = true;

    while (*cursor == ' ' || *cursor == '\t') ++cursor;
    if (*cursor == ',') {
      ++cursor;
      while (*cursor == ' ' || *cursor == '\t') ++cursor;
      continue;
    }
    if (*cursor == '\0' || *cursor == '\r' || *cursor == '\n') {
      while (*cursor == '\r' || *cursor == '\n') ++cursor;
      while (*cursor == ' ' || *cursor == '\t') ++cursor;
      if (*cursor != '\0') return false;
      count = written;
      return written != 0;
    }
    return false;
  }
}

}  // namespace hyle::platform::detail

#if defined(HYLE_TESTING) && defined(_WIN32)
namespace hyle::platform::detail {

// TEST-ONLY: inject runtime API availability into the Windows capability
// decision without replacing or invoking OS entry points.
HYLE_PLATFORM_API bool windows_selected_interleave_supported_for_testing(
    bool has_virtual_alloc_ex_numa, bool has_virtual_alloc2) noexcept;

}  // namespace hyle::platform::detail
#endif

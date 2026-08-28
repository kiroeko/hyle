#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

namespace hyle::platform::detail {

inline bool large_page_value_is_valid(std::uint64_t value,
                                      std::size_t base_page) noexcept {
  return value >= base_page &&
         value <= (std::numeric_limits<std::size_t>::max)() &&
         (value & (value - 1)) == 0;
}

inline std::string_view trim_ascii_space(std::string_view text) noexcept {
  while (!text.empty() &&
         (text.front() == ' ' || text.front() == '\t' ||
          text.front() == '\r' || text.front() == '\n')) {
    text.remove_prefix(1);
  }
  while (!text.empty() &&
         (text.back() == ' ' || text.back() == '\t' ||
          text.back() == '\r' || text.back() == '\n')) {
    text.remove_suffix(1);
  }
  return text;
}

inline std::size_t parse_linux_thp_size(std::string_view text,
                                        std::size_t base_page) noexcept {
  text = trim_ascii_space(text);
  std::uint64_t value = 0;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
      !large_page_value_is_valid(value, base_page)) {
    return 0;
  }
  return static_cast<std::size_t>(value);
}

inline std::size_t parse_linux_meminfo_huge_page_size(
    std::string_view line, std::size_t base_page) noexcept {
  constexpr std::string_view kPrefix = "Hugepagesize:";
  if (!line.starts_with(kPrefix)) return 0;
  line = trim_ascii_space(line.substr(kPrefix.size()));

  std::uint64_t kib = 0;
  const auto parsed =
      std::from_chars(line.data(), line.data() + line.size(), kib);
  if (parsed.ec != std::errc{} || parsed.ptr == line.data()) return 0;
  std::string_view unit(parsed.ptr,
                        static_cast<std::size_t>(line.data() + line.size() -
                                                 parsed.ptr));
  if (trim_ascii_space(unit) != "kB" || kib > UINT64_MAX / 1024u) return 0;
  const std::uint64_t bytes = kib * 1024u;
  return large_page_value_is_valid(bytes, base_page)
             ? static_cast<std::size_t>(bytes)
             : 0;
}

}  // namespace hyle::platform::detail

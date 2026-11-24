#pragma once

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace hyle::inference {

// The caller validates the upper bound against the loaded model's geometry.
// Keep intervals compact so parsing a requested placement never allocates in
// proportion to an untrusted layer index.
class CpuFfnLayers final {
public:
  CpuFfnLayers() = default;
  CpuFfnLayers(std::size_t prefix) {
    if (prefix) ranges_.push_back({0, prefix});
  }

  static CpuFfnLayers parse(std::string_view value) {
    if (value.find(':') == std::string_view::npos)
      return CpuFfnLayers(number(value));
    std::vector<Range> ranges;
    while (true) {
      const auto comma = value.find(',');
      const auto range = value.substr(0, comma);
      const auto colon = range.find(':');
      if (colon == std::string_view::npos) throw invalid();
      const auto first = number(range.substr(0, colon));
      const auto end = number(range.substr(colon + 1));
      if (first >= end) throw invalid();
      ranges.push_back({first, end});
      if (comma == std::string_view::npos) break;
      value.remove_prefix(comma + 1);
    }
    std::ranges::sort(ranges, {}, &Range::first);
    CpuFfnLayers result;
    for (const auto &range : ranges) {
      if (!result.ranges_.empty()) {
        auto &previous = result.ranges_.back();
        if (range.first < previous.end) throw invalid();
        if (range.first == previous.end) {
          previous.end = range.end;
          continue;
        }
      }
      result.ranges_.push_back(range);
    }
    return result;
  }

  bool empty() const noexcept { return ranges_.empty(); }
  std::size_t count() const noexcept {
    std::size_t count = 0;
    for (const auto &range : ranges_) count += range.end - range.first;
    return count;
  }
  bool contains(std::size_t layer) const noexcept {
    return std::ranges::any_of(ranges_, [layer](const auto &range) {
      return layer >= range.first && layer < range.end;
    });
  }
  bool fits(std::size_t layer_count) const noexcept {
    return empty() || ranges_.back().end <= layer_count;
  }

  std::string ranges_text() const {
    std::string result;
    for (const auto &range : ranges_) {
      if (!result.empty()) result += ',';
      result += std::to_string(range.first) + ':' + std::to_string(range.end);
    }
    return result;
  }

  friend bool operator==(const CpuFfnLayers &, const CpuFfnLayers &) = default;

private:
  struct Range final {
    std::size_t first, end;
    friend bool operator==(const Range &, const Range &) = default;
  };

  static std::invalid_argument invalid() {
    return std::invalid_argument("--cpu-ffn-layers requires a count or disjoint zero-based START:END ranges");
  }
  static std::size_t number(std::string_view value) {
    if (value.empty()) throw invalid();
    std::size_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) throw invalid();
    return result;
  }

  std::vector<Range> ranges_;
};

} // namespace hyle::inference

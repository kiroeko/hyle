// Consumer token-file input shared by direct and schedule entry points.
#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace qwen35 {

inline std::vector<std::int32_t> read_ids(const std::filesystem::path &path,
    std::int64_t vocabulary = (std::numeric_limits<std::int32_t>::max)(),
    std::size_t maximum_tokens = (std::numeric_limits<std::size_t>::max)()) {
  if (path.native().find(std::filesystem::path::value_type{}) != std::filesystem::path::string_type::npos)
    throw std::runtime_error("token IDs path contains a null byte");
  std::ifstream stream(path); if (!stream) throw std::runtime_error("cannot open token IDs");
  stream.imbue(std::locale::classic());
  if (vocabulary <= 0 || vocabulary > (std::numeric_limits<std::int32_t>::max)())
    throw std::runtime_error("invalid token vocabulary");
  std::vector<std::int32_t> ids;
  // Every supported token ID fits in eleven characters including a sign.
  // Bound lexical input too; do not allocate for an arbitrarily long word.
  std::string word;
  while (stream >> std::setw(31) >> word) {
    const auto next = stream.peek();
    if (next != std::char_traits<char>::eof() &&
        std::string_view(" \t\r\n\f\v").find(static_cast<char>(next)) == std::string_view::npos)
      throw std::runtime_error("input token length exceeds 31 characters");
    if (ids.size() == maximum_tokens)
      throw std::runtime_error("input token count exceeds configured capacity");
    std::int64_t value = 0;
    const auto end = word.data() + word.size();
    const auto parsed = std::from_chars(word.data(), end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != end) throw std::runtime_error("malformed input IDs");
    if (value < 0 || value >= vocabulary) throw std::runtime_error("input token outside vocabulary");
    ids.push_back(static_cast<std::int32_t>(value));
  }
  if (stream.bad() || !stream.eof()) throw std::runtime_error("malformed input IDs");
  return ids;
}

} // namespace qwen35

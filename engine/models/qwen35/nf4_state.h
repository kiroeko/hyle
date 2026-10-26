#pragma once

#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace qwen35::prepared {
inline constexpr std::size_t maximum_nf4_state_bytes = 512;
inline constexpr std::string_view nf4_state_suffix = ".quant_state.bitsandbytes__nf4";
inline constexpr std::string_view nf4_policy_line =
    "nf4_double_quant\t64\t256\tF16\tF16\tF32\thigh_first\tbitsandbytes-default@833649043474794b8fe7a4136e0c40faf077b2e0";
// Decode dtype is original QuantState.dtype; compute dtype is separately fixed
// by the selected source policy. Neither is an Unknown/BF16 default.
struct Nf4Policy final { bool enabled = false; };
struct Nf4State final {
  std::array<std::int64_t, 2> logical_shape{};
  std::uint32_t offset_f32_bits = 0;
  std::string original_json_sha256{};
  static constexpr std::string_view decode_dtype = "F16", compute_dtype = "F16";
};

// Deliberately limited to the fixed seven-field original QuantState object.
// It does not parse arbitrary model JSON, allocate a JSON tree, or read files.
inline Nf4State parse_nf4_state(std::string_view raw) {
  const auto fail = [] { throw std::runtime_error("invalid original NF4 quant_state"); };
  if (raw.empty() || raw.size() > maximum_nf4_state_bytes) fail();
  std::size_t cursor = 0;
  const auto space = [&] { while (cursor < raw.size() &&
      (raw[cursor] == ' ' || raw[cursor] == '\t' || raw[cursor] == '\n' || raw[cursor] == '\r')) ++cursor; };
  const auto expect = [&](char ch) { space(); if (cursor == raw.size() || raw[cursor++] != ch) fail(); };
  const auto word = [&]() {
    expect('"'); const auto first = cursor;
    while (cursor < raw.size() && raw[cursor] != '"') {
      if (static_cast<unsigned char>(raw[cursor]) < 32 || raw[cursor] == '\\') fail();
      ++cursor;
    }
    if (cursor == raw.size()) fail();
    const auto value = raw.substr(first, cursor-first); ++cursor; return value;
  };
  const auto number = [&]() {
    space(); const auto first = cursor;
    if (cursor < raw.size() && raw[cursor] == '-') ++cursor;
    if (cursor == raw.size() || raw[cursor] < '0' || raw[cursor] > '9') fail();
    if (raw[cursor++] != '0') while (cursor < raw.size() && raw[cursor] >= '0' && raw[cursor] <= '9') ++cursor;
    if (cursor < raw.size() && raw[cursor] == '.') {
      const auto start = ++cursor;
      while (cursor < raw.size() && raw[cursor] >= '0' && raw[cursor] <= '9') ++cursor;
      if (cursor == start) fail();
    }
    if (cursor < raw.size() && (raw[cursor] == 'e' || raw[cursor] == 'E')) {
      ++cursor; if (cursor < raw.size() && (raw[cursor] == '+' || raw[cursor] == '-')) ++cursor;
      const auto start = cursor;
      while (cursor < raw.size() && raw[cursor] >= '0' && raw[cursor] <= '9') ++cursor;
      if (cursor == start) fail();
    }
    return raw.substr(first, cursor-first);
  };
  const auto integer = [&]() {
    const auto token = number(); std::int64_t value = 0;
    const auto parsed = std::from_chars(token.data(), token.data()+token.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != token.data()+token.size()) fail();
    return value;
  };
  constexpr std::array<std::string_view,7> keys{"quant_type","blocksize","dtype","shape",
      "nested_blocksize","nested_dtype","nested_offset"};
  std::array<bool,7> seen{}; Nf4State result;
  expect('{');
  for (std::size_t field = 0; field < keys.size(); ++field) {
    if (field) expect(',');
    const auto key = word(); std::size_t index = 0;
    while (index < keys.size() && keys[index] != key) ++index;
    if (index == keys.size() || seen[index]) fail();
    seen[index] = true; expect(':');
    if (index == 0 || index == 2 || index == 5) {
      const auto value = word();
      if (value != (index == 0 ? "nf4" : index == 2 ? "float16" : "float32")) fail();
    } else if (index == 1 || index == 4) {
      if (integer() != (index == 1 ? 64 : 256)) fail();
    } else if (index == 3) {
      expect('['); result.logical_shape[0] = integer(); expect(',');
      result.logical_shape[1] = integer(); expect(']');
    } else {
      const auto token = number(); double value = 0;
      // This bounded token already passed the JSON number grammar. Classic
      // locale keeps decimal parsing independent of the host locale, including
      // standard libraries without floating from_chars at our deployment floor.
      std::istringstream parsed{std::string(token)};
      parsed.imbue(std::locale::classic());
      parsed >> std::noskipws;
      if (!(parsed >> value) || parsed.peek() != std::char_traits<char>::eof() || !std::isfinite(value) ||
          value > (std::numeric_limits<float>::max)() || value < -(std::numeric_limits<float>::max)()) fail();
      // Some num_get implementations silently round a double underflow to
      // zero. Preserve the original range rejection without rejecting -0.
      if (value == 0 && token.substr(0, token.find_first_of("eE")).find_first_of("123456789") !=
          std::string_view::npos) fail();
      const auto offset = static_cast<float>(value);
      if (!std::isfinite(offset)) fail();
      result.offset_f32_bits = std::bit_cast<std::uint32_t>(offset);
    }
  }
  expect('}'); space();
  const auto n = result.logical_shape[0], k = result.logical_shape[1];
  if (cursor != raw.size() || n <= 0 || k <= 0 || n > INT64_MAX/k) fail();
  return result;
}
// The loader separately authenticates the original JSON range's SHA256 against
// original_json_sha256 before calling this value/shape check.
inline void validate_nf4_state_values(std::string_view raw, const Nf4State &expected) {
  const auto actual = parse_nf4_state(raw);
  if (actual.logical_shape != expected.logical_shape || actual.offset_f32_bits != expected.offset_f32_bits)
    throw std::runtime_error("original NF4 quant_state values differ");
}
} // namespace qwen35::prepared

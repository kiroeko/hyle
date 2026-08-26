#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "hyle/container/string_ref.h"
#include "hyle/core/compiler.h"

// Allocation-free byte-string foundations for parsers, protocols, configuration,
// and path helpers. Inputs are non-owning string_ref values; transforms use
// caller-provided buffers; character rules are locale-free and deterministic.
//
// Scope:
//  * IN : ASCII classification/case conversion, hex encode/decode, C byte escape,
//         and strict UTF-8 iteration/validation/code-point counting.
//  * USE THE STANDARD LIBRARY for formatting/joining, numeric parsing, and owned
//         string composition (`std::format`, `std::from_chars`, `std::string`).
//  * APPLICATION LAYER: fuzzy matching, ranking, and UI/search-specific policy.

namespace hyle::container {

// ---- ASCII character classifiers (locale-free, constexpr) -------------------
HYLE_FORCE_INLINE constexpr bool is_ascii_space(char c) noexcept {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' ||
         c == '\f';
}
HYLE_FORCE_INLINE constexpr bool is_ascii_digit(char c) noexcept {
  return c >= '0' && c <= '9';
}
// Printable per Unix isprint, but locale-insensitive (clay isprint).
HYLE_FORCE_INLINE constexpr bool is_ascii_print(char c) noexcept {
  return static_cast<unsigned char>(c) > 0x1f &&
         static_cast<unsigned char>(c) < 0x7f;
}
HYLE_FORCE_INLINE constexpr bool is_hex_digit(char c) noexcept {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
         (c >= 'A' && c <= 'F');
}
// Value of a hex digit, or -1 if not hex (clay hex_to_int silently returned 0
// for invalid input -- a bug that hid bad data; HYLE signals it).
HYLE_FORCE_INLINE constexpr int hex_value(char c) noexcept {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}
HYLE_FORCE_INLINE constexpr char to_ascii_lower(char c) noexcept {
  return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + ('a' - 'A')) : c;
}
HYLE_FORCE_INLINE constexpr char to_ascii_upper(char c) noexcept {
  return (c >= 'a' && c <= 'z') ? static_cast<char>(c - ('a' - 'A')) : c;
}

// ---- case conversion (buffer-fill, snprintf semantics) ----------------------
// Write the lower/upper-cased `s` into out[cap], always NUL-terminating when
// cap>0. Returns the full source length (>= cap signals truncation). No alloc.
namespace detail {
HYLE_CONTAINER_API std::uint32_t str_ascii_lower(
    const char*, std::uint32_t, char*, std::uint32_t) noexcept;
HYLE_CONTAINER_API std::uint32_t str_ascii_upper(
    const char*, std::uint32_t, char*, std::uint32_t) noexcept;
}
inline std::uint32_t to_lower(string_ref s, char* out,
                              std::uint32_t cap) noexcept {
  return detail::str_ascii_lower(s.data, s.size, out, cap);
}
inline std::uint32_t to_upper(string_ref s, char* out,
                              std::uint32_t cap) noexcept {
  return detail::str_ascii_upper(s.data, s.size, out, cap);
}

// ---- hex encode / C escape (buffer-fill) ------------------------------------
// Lowercase hex of `len` bytes into out. Needs 2*len (+1 NUL) capacity; returns
// the full hex length (2*len). Truncates to whole bytes if cap is short.
inline std::size_t hex_encode(const void* data, std::size_t len, char* out,
                              std::size_t cap) noexcept {
  static constexpr char kHex[] = "0123456789abcdef";
  const auto* b = static_cast<const unsigned char*>(data);
  const std::size_t need =
      len > std::numeric_limits<std::size_t>::max() / 2
          ? std::numeric_limits<std::size_t>::max()
          : len * 2;
  std::size_t w = 0;
  for (std::size_t i = 0; i < len && w + 2 < cap; ++i) {
    out[w++] = kHex[b[i] >> 4];
    out[w++] = kHex[b[i] & 0xF];
  }
  if (cap > 0) out[w] = '\0';
  return need;
}

// Decode an even-length hexadecimal string. Returns the full decoded byte
// count, SIZE_MAX on malformed/odd input. Writes at most `cap` bytes; size-only
// calls may pass out==nullptr with cap==0.
inline std::size_t hex_decode(string_ref hex, void* out,
                              std::size_t cap) noexcept {
  if ((hex.size & 1u) != 0) return std::numeric_limits<std::size_t>::max();
  const std::size_t need = hex.size / 2;
  auto* bytes = static_cast<unsigned char*>(out);
  const std::size_t write_count = cap < need ? cap : need;
  for (std::size_t i = 0; i < write_count; ++i) {
    const int high = hex_value(hex.data[i * 2]);
    const int low = hex_value(hex.data[i * 2 + 1]);
    if (high < 0 || low < 0) return std::numeric_limits<std::size_t>::max();
    bytes[i] = static_cast<unsigned char>((high << 4) | low);
  }
  // A truncated output buffer must not weaken validation of the suffix.
  for (std::size_t i = write_count; i < need; ++i) {
    if (hex_value(hex.data[i * 2]) < 0 ||
        hex_value(hex.data[i * 2 + 1]) < 0)
      return std::numeric_limits<std::size_t>::max();
  }
  return need;
}

// C-escape arbitrary bytes into out. Named controls use \n/\t/etc, printable
// ASCII passes through, and every other byte uses an unambiguous three-digit
// octal escape (including embedded NUL and non-UTF-8 bytes). Returns the full
// escaped length (>= cap signals truncation) and emits only complete escape
// tokens. This is the allocation-free buffer-fill counterpart of Abseil
// CEscape, not a UI/display policy.
inline std::size_t c_escape(string_ref s, char* out, std::size_t cap) noexcept {
  auto esc = [](char c) -> char {
    switch (c) {
      case '\a': return 'a';
      case '\b': return 'b';
      case '\f': return 'f';
      case '\n': return 'n';
      case '\r': return 'r';
      case '\t': return 't';
      case '\v': return 'v';
      case '\\': return '\\';
      case '\'': return '\'';
      case '"': return '"';
      default: return 0;
    }
  };
  static constexpr char kOctal[] = "01234567";
  std::size_t need = 0, w = 0;
  bool writing = true;
  for (std::uint32_t i = 0; i < s.size; ++i) {
    const unsigned char byte = static_cast<unsigned char>(s.data[i]);
    const char e = esc(static_cast<char>(byte));
    const bool printable = byte >= 0x20 && byte <= 0x7e;
    const std::size_t add = e ? 2 : printable ? 1 : 4;
    // `add < cap - w` reserves one byte for NUL without the unsigned
    // `cap - 1 - w` shape that can underflow if this invariant is refactored.
    const bool fits = writing && w < cap && add < cap - w;
    if (fits) {
      if (e) {
        out[w++] = '\\';
        out[w++] = e;
      } else if (printable) {
        out[w++] = static_cast<char>(byte);
      } else {
        out[w++] = '\\';
        out[w++] = kOctal[(byte >> 6) & 0x3];
        out[w++] = kOctal[(byte >> 3) & 0x7];
        out[w++] = kOctal[byte & 0x7];
      }
    } else {
      writing = false;
    }
    need = need > std::numeric_limits<std::size_t>::max() - add
               ? std::numeric_limits<std::size_t>::max()
               : need + add;
  }
  if (cap > 0) out[w] = '\0';
  return need;
}

// ---- UTF-8 validation/count --------------------------------------------------
// Forward decoder over a byte range. next() returns the next code point, 0 at
// end, and flips valid() to false on malformed input. A null pointer is an
// empty valid range only when size is zero; null + nonzero starts invalid.
class Utf8Decoder {
 public:
  explicit Utf8Decoder(string_ref s) noexcept
      : p_(s.data),
        end_(s.data == nullptr ? nullptr : s.data + s.size),
        valid_(s.data != nullptr || s.size == 0) {}

  bool is_end() const noexcept { return p_ == end_; }
  bool valid() const noexcept { return valid_; }

  int next() noexcept {
    if (p_ == end_) return 0;
    const unsigned first = static_cast<unsigned char>(*p_++);
    if (first <= 0x7F) return static_cast<int>(first);

    unsigned value;
    unsigned continuation;
    unsigned minimum;
    if (first >= 0xC2 && first <= 0xDF) {
      value = first & 0x1F;
      continuation = 1;
      minimum = 0x80;
    } else if (first >= 0xE0 && first <= 0xEF) {
      value = first & 0x0F;
      continuation = 2;
      minimum = 0x800;
    } else if (first >= 0xF0 && first <= 0xF4) {
      value = first & 0x07;
      continuation = 3;
      minimum = 0x10000;
    } else {
      valid_ = false;  // continuation, overlong lead, or > U+10FFFF lead
      return 0;
    }
    while (continuation-- != 0) {
      if (p_ == end_) {
        valid_ = false;
        return 0;
      }
      const unsigned next = static_cast<unsigned char>(*p_++);
      if ((next & 0xC0) != 0x80) {
        valid_ = false;
        return 0;
      }
      value = (value << 6) | (next & 0x3F);
    }
    if (value < minimum || value > 0x10FFFF ||
        (value >= 0xD800 && value <= 0xDFFF)) {
      valid_ = false;
      return 0;
    }
    return static_cast<int>(value);
  }

 private:
  const char* p_;
  const char* end_;
  bool valid_ = true;
};

// Validate a UTF-8 byte range without counting it. This is the one-pass normal
// path for parsers/protocol boundaries that only need an accept/reject decision.
// Number of UTF-8 code points returns -1 on invalid input and therefore performs
// validate + count (two dispatched passes for valid input).
// A null data pointer is valid only for an empty range: {nullptr, 0} validates
// and counts as zero, while {nullptr, nonzero} returns false / -1.
//
// The shipping container target embeds reviewed simdutf v9.1 and borrows its
// verified cores plus internal runtime ISA dispatch. Utf8Decoder remains the
// scalar/reference fallback; vendor types do not cross this public seam. The
// runtime entries live in string_algo.cc.
namespace detail {
HYLE_CONTAINER_API bool utf8_validate_runtime(string_ref s) noexcept;
HYLE_CONTAINER_API std::int64_t utf8_length_runtime(string_ref s) noexcept;
}
inline bool utf8_validate(string_ref s) noexcept {
  return detail::utf8_validate_runtime(s);
}
inline std::int64_t utf8_length(string_ref s) noexcept {
  return detail::utf8_length_runtime(s);
}

}  // namespace hyle::container

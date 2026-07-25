#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "hyle/container/string_ref.h"
#include "hyle/macros.h"

// Byte-string algorithms (clay utils/string_utils.cc + sys/cstring.cc, the parts
// clay ACTUALLY implemented). clay's string_utils.h was the std-stream epicenter
// (<ostream>/<sstream>/std::string everywhere) -- HYLE re-expresses the genuine
// algorithms as string_ref-in / buffer-fill-out, pulling in zero std streams.
//
// What is and isn't here (honest scope):
//  * IN  : ASCII case classify/convert, capitalize (C-ident -> "Nice Name"),
//          hex encode/decode, C escape, UTF-8 iterate/length/validate/icmp.
//  * std : str()/join()/to_string -> std::format; stoi/stod -> std::from_chars;
//          tolower/toupper(std::string) -> these (ASCII, locale-free, no alloc).
//  * GONE: clay only-declared formatting/fuzzy helpers. Application-specific
//          ranking belongs in a search/UI layer, not this container foundation.

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

// ---- capitalize (clay cstring.cc capitalize, in-place) ----------------------
// Turns a C identifier into a UI-friendly name in place: '_'/space become a
// space, and the first letter of each word is upper-cased. Special-cases the
// "Ui" acronym -> "UI". Operates on a NUL-terminated mutable buffer.
inline void capitalize(char* s) noexcept {
  bool cap_next = true;
  while (*s != '\0') {
    if (cap_next) {
      *s = to_ascii_upper(*s);
      if (*s == 'U' && s[1] == 'i' && (s[2] == ' ' || s[2] == '_' || s[2] == 0))
        s[1] = 'I';
    }
    if (*s == '_' || *s == ' ') {
      *s = ' ';
      cap_next = true;
    } else {
      cap_next = false;
    }
    ++s;
  }
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

// C-escape `s` into out (\n \t \\ \" etc; printables pass through). Returns the
// full escaped length (>= cap signals truncation). clay escaped to std::string;
// HYLE is buffer-fill.
inline std::size_t escape(string_ref s, char* out, std::size_t cap) noexcept {
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
  std::size_t need = 0, w = 0;
  for (std::uint32_t i = 0; i < s.size; ++i) {
    char e = esc(s.data[i]);
    std::size_t add = e ? 2 : 1;
    if (w + add < cap) {
      if (e) {
        out[w++] = '\\';
        out[w++] = e;
      } else {
        out[w++] = s.data[i];
      }
    }
    need = need > std::numeric_limits<std::size_t>::max() - add
               ? std::numeric_limits<std::size_t>::max()
               : need + add;
  }
  if (cap > 0) out[w] = '\0';
  return need;
}

// ---- UTF-8 (clay utf8_decode / utf8_length / utf8_icmp) ---------------------
// Forward decoder over a byte range. next() returns the next code point, 0 at
// end, and flips valid() to false on malformed input (clay's utf8_decode, on
// string_ref instead of std::string::const_iterator).
class Utf8Decoder {
 public:
  explicit Utf8Decoder(string_ref s) noexcept
      : p_(s.data), end_(s.data == nullptr ? nullptr : s.data + s.size) {}

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

// Number of UTF-8 code points in `s`, or -1 if `s` is not valid UTF-8 (VALIDATE +
// COUNT). Dispatched: the scalar Utf8Decoder scan is the default + fallback, but
// when the container module is built with -DHYLE_SIMD_SIMDUTF it BORROWS simdutf's
// verified SIMD validate+count cores (借核不手搓, same as simd/math borrows
// Highway/xsimd). The runtime entry lives in string_algo.cc; this is a thin delegate.
namespace detail {
HYLE_CONTAINER_API std::int64_t utf8_length_runtime(string_ref s) noexcept;
}
inline std::int64_t utf8_length(string_ref s) noexcept {
  return detail::utf8_length_runtime(s);
}

// Case-insensitive (ASCII fold) UTF-8 compare: <0, 0, >0. Compares up to `limit`
// code points (0 = whole string). clay utf8_icmp.
inline int utf8_icmp(string_ref a, string_ref b, int limit = 0) noexcept {
  Utf8Decoder da(a), db(b);
  int i = 0;
  for (; (limit == 0 || i < limit) && !da.is_end() && !db.is_end(); ++i) {
    int ca = da.next();
    int cb = db.next();
    if (ca < 128) ca = to_ascii_lower(static_cast<char>(ca));
    if (cb < 128) cb = to_ascii_lower(static_cast<char>(cb));
    if (ca < cb) return -1;
    if (ca > cb) return 1;
  }
  if (limit > 0 && i == limit) return 0;
  if (da.is_end() && db.is_end()) return 0;
  return da.is_end() ? -1 : 1;
}

}  // namespace hyle::container

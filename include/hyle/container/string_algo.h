#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>  // std::swap (find_path_matches heap)

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
//  * GONE: clay DECLARED format_si / find_path_matches / cscore_path_match /
//          cstr_score_similarity but NEVER implemented them (header vanity, like
//          its fake sprintf truncation). Nothing to absorb; if HYLE ever needs
//          fuzzy path matching it is net-new design, registered as container-06c.

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

// ---- fuzzy path matching (container-06c) ------------------------------------
// Editor quick-open scoring (Sublime/VSCode/fzf Ctrl-P style): score how well a
// short `pattern` fuzzy-matches a `candidate`. NET-NEW design -- clay declared
// cscore_path_match/find_path_matches but never implemented them (see file top).
//
// ALGORITHM (greedy single-pass O(pattern + candidate), zero-alloc): every pattern
// char must appear in the candidate IN ORDER (subsequence) else score 0; a matched
// char earns scoreMatch plus a boundary/camel/consecutive bonus. The bonus SEMANTICS
// and constants are fzf's (src/algo/algo.go: bonusFor + the bonus* constants), which
// is why "cfg" ranks a filename hit over a directory hit WITHOUT any ad-hoc filename
// multiplier: fzf treats '/' as a DELIMITER boundary and keeps a string-start (white)
// boundary <= a delimiter boundary, so a match after '/' isn't beaten by one at
// index 0. Chosen over fzf's Smith-Waterman DP because the load is "10k-100k paths,
// re-scored every keystroke" -- an O(n) no-alloc pass with fzf's bonus values gives
// near-DP ranking at a fraction of the cost (DP allocates an H/C table per candidate).
// Case-insensitive over ASCII (bytes >= 0x80 compare verbatim -- ASCII fold is a no-op
// on UTF-8 continuation bytes, so safe, just not case-folding non-ASCII; phase 2).
//
// Scoring is INTEGER (not clay's double): faster, exactly reproducible, and ordering
// is all a quick-open UI needs. Higher = better. 0 = not a subsequence (no match).

namespace detail {

// fzf bonus/penalty constants (algo.go L112-162), the values tuned over fzf's life.
// Public-header inline so the scorer stays a hot-path leaf with no cross-TU call.
inline constexpr int kFuzzyMatch = 16;            // scoreMatch: base per matched char
inline constexpr int kFuzzyGapStart = -3;         // scoreGapStart: first unmatched char in a gap
inline constexpr int kFuzzyGapExt = -1;           // scoreGapExtension: each further gap char
inline constexpr int kFuzzyBoundary = 8;          // bonusBoundary: match at a word boundary (=match/2)
inline constexpr int kFuzzyBoundaryDelim = 9;     // bonusBoundaryDelimiter: after '/' ':' etc (boundary+1)
inline constexpr int kFuzzyCamel = 7;             // bonusCamel123: camelCase / alpha->digit hump
inline constexpr int kFuzzyConsecutive = 4;       // bonusConsecutive: min per consecutive-run char
inline constexpr int kFuzzyFirstCharMult = 2;     // bonusFirstCharMultiplier: 1st match's bonus x2

// Character classes (fzf charClass): the boundary bonus depends on the PREVIOUS
// char's class, so a match "after a delimiter / at a camel hump / after whitespace"
// is scored by classifying both neighbours. ASCII-only (phase 1).
enum class CharClass : unsigned char { White, Delimiter, Lower, Upper, Digit, NonWord };

HYLE_FORCE_INLINE constexpr CharClass fuzzy_class(char c) noexcept {
  if (c >= 'a' && c <= 'z') return CharClass::Lower;
  if (c >= 'A' && c <= 'Z') return CharClass::Upper;
  if (c >= '0' && c <= '9') return CharClass::Digit;
  if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f')
    return CharClass::White;
  // Path/word delimiters get the (slightly higher) delimiter-boundary bonus so a
  // filename char after '/' outranks a char at string start (fzf path scheme).
  if (c == '/' || c == '\\' || c == '_' || c == '-' || c == '.' || c == ' ' ||
      c == ':' || c == ';' || c == ',' || c == '|')
    return CharClass::Delimiter;
  return CharClass::NonWord;
}

// Boundary bonus for a `cur` char whose predecessor was `prev` (fzf bonusFor,
// algo.go L268-296). A word char right after white/delimiter/non-word starts a new
// "word" (boundary bonus); a lower->upper or ->digit transition is a camel hump.
HYLE_FORCE_INLINE constexpr int fuzzy_bonus(CharClass prev, CharClass cur) noexcept {
  const bool cur_word = (cur == CharClass::Lower || cur == CharClass::Upper ||
                         cur == CharClass::Digit);
  if (cur_word) {
    switch (prev) {
      case CharClass::White: return kFuzzyBoundary;       // start-of-string / after space
      case CharClass::Delimiter: return kFuzzyBoundaryDelim;  // after '/' etc (> white)
      case CharClass::NonWord: return kFuzzyBoundary;
      default: break;  // prev is a word char -> maybe a camel hump below
    }
  }
  // camelCase hump (lower/digit -> upper) or alpha -> digit.
  if ((prev == CharClass::Lower && cur == CharClass::Upper) ||
      (prev != CharClass::Digit && cur == CharClass::Digit)) {
    return kFuzzyCamel;
  }
  return 0;
}

// Core scorer over [cand] against [pat]. `init_class` is the class of the "character
// before index 0": White for a plain string (fuzzy_match_score) so string-start is a
// whitespace boundary; Delimiter for a path (path_match_score) so the whole path's
// first char is treated like a post-'/' filename char. Returns 0 if `pat` is not an
// ordered subsequence of `cand`. O(cand), zero-alloc.
//
// Two passes (fzf FuzzyMatchV1, algo.go L710-790): a plain left-to-right greedy match
// locks onto the FIRST occurrence of each pattern char, which misses better-aligned
// hits ("cfg" would match the 'c' in "src/..." not the one starting "config"). So we
// (1) FORWARD-scan to find the end of a full subsequence match, then (2) BACKWARD-scan
// from there to pull the START as far right as possible -- this snaps the match onto
// the tightest, most-boundary-aligned run (the 'c' of "config"). Scoring then runs
// over that tightened window. Still O(cand), still zero-alloc.
inline int fuzzy_score_impl(string_ref cand, string_ref pat,
                            CharClass init_class) noexcept {
  if (pat.size == 0) return 0;         // empty pattern: no discriminating score
  if (pat.size > cand.size) return 0;  // can't be a subsequence
  // Pass 1 (forward): find sidx (first matched char) and eidx (one past the last).
  std::uint32_t pidx = 0, sidx = cand.size, eidx = 0;
  for (std::uint32_t i = 0; i < cand.size; ++i) {
    if (to_ascii_lower(cand.data[i]) == to_ascii_lower(pat.data[pidx])) {
      if (sidx == cand.size) sidx = i;
      if (++pidx == pat.size) {
        eidx = i + 1;
        break;
      }
    }
  }
  if (eidx == 0) return 0;  // pattern is not a subsequence of cand
  // Pass 2 (backward): tighten sidx rightward -- match pattern in reverse from eidx-1
  // so the start snaps to the latest feasible position (the boundary-aligned run).
  pidx = pat.size - 1;
  for (std::uint32_t i = eidx; i-- > sidx;) {
    if (to_ascii_lower(cand.data[i]) == to_ascii_lower(pat.data[pidx])) {
      if (pidx-- == 0) {
        sidx = i;
        break;
      }
    }
  }
  // Pass 3: score over the tightened window [sidx, eidx). Every pattern char lies in
  // it, in order; walk it charging match+bonus, gaps, and consecutive runs.
  std::int64_t score = 0;
  std::uint32_t ci = sidx;
  bool first_match = true;
  for (std::uint32_t pi = 0; pi < pat.size; ++pi) {
    char pc = to_ascii_lower(pat.data[pi]);
    std::uint32_t gap = 0;
    while (ci < eidx && to_ascii_lower(cand.data[ci]) != pc) {
      ++gap;
      ++ci;
    }
    int b = kFuzzyMatch;
    CharClass prev = ci == 0 ? init_class : fuzzy_class(cand.data[ci - 1]);
    CharClass cur = fuzzy_class(cand.data[ci]);
    int bonus = fuzzy_bonus(prev, cur);
    if (first_match) {
      // fzf: the FIRST matched char's boundary bonus is multiplied (a hit at a word
      // start is worth extra), and there is no gap penalty before it.
      score += b + bonus * kFuzzyFirstCharMult;
      first_match = false;
    } else if (gap == 0) {
      // Consecutive (adjacent to the previous match): each run char earns at least
      // bonusConsecutive; a genuine boundary bonus (a new camel hump) wins if higher
      // (fzf max(consecutive, bonus)).
      score += b + (bonus > kFuzzyConsecutive ? bonus : kFuzzyConsecutive);
    } else {
      // A gap preceded this match: charge the gap, then the plain match + bonus.
      score += static_cast<std::int64_t>(kFuzzyGapStart) +
               static_cast<std::int64_t>(kFuzzyGapExt) * (gap - 1);
      score += b + bonus;
    }
    ++ci;  // consume the matched char
  }
  // Tie-breaker toward SHORTER candidates (fzf's length tiebreak, folded into the
  // score): a complete "abc" beats "abcXYZ", and "src/config.h" (12) beats
  // "config/main.cpp" (15) when their bonuses tie. Weight 1/char so it only
  // separates otherwise-equal scores, never reorders a real bonus difference.
  const std::int64_t adjusted = score - static_cast<std::int64_t>(cand.size);
  // Zero is reserved for "not a subsequence". Very long but valid candidates
  // must remain matches even when the length tie-break exceeds all bonuses.
  if (adjusted <= 0) return 1;
  if (adjusted > std::numeric_limits<int>::max())
    return std::numeric_limits<int>::max();
  return static_cast<int>(adjusted);
}

}  // namespace detail

// Fuzzy-match `pattern` against `candidate` (plain string -- string-start counts as a
// whitespace boundary). Returns an integer score (higher = better), or 0 if `pattern`
// is not an ordered case-insensitive subsequence of `candidate`. O(candidate), no alloc.
inline int fuzzy_match_score(string_ref candidate, string_ref pattern) noexcept {
  return detail::fuzzy_score_impl(candidate, pattern, detail::CharClass::White);
}

// Score `pattern` against a FILE PATH. Same scorer, but the path's first char is
// treated as if preceded by a delimiter (like a post-'/' filename char), and '/' '\\'
// are delimiter boundaries -- so a hit in the filename part outranks one buried in a
// directory, and "cfg" ranks "src/config.h" at/above "config/main.cpp" (clay
// cscore_path_match's contract) WITHOUT an ad-hoc multiplier (fzf's path scheme:
// delimiter boundaries + the length tiebreak fold that in). O(path), no alloc.
inline int path_match_score(string_ref path, string_ref pattern) noexcept {
  return detail::fuzzy_score_impl(path, pattern, detail::CharClass::Delimiter);
}

// Find the best matches of `pattern` among `candidates[0..n)`, writing their indices
// into `out_idx[0..return)` BEST-FIRST and their scores into `out_score[0..return)`.
// `out_idx` and `out_score` are caller-owned buffers of length `k` (the top-list
// size, e.g. 20) -- they double as the working min-heap, so there is ZERO heap
// allocation (charter: no hidden alloc; the caller sizes k). Non-matching candidates
// (score 0) are skipped. Returns the match count written (<= k).
//
// O(n log k): a size-k min-heap keeps the running top-k (push/replace is log k); a
// final heap-sort orders the survivors best-first. Ties break toward the lower index
// (stable, reproducible for a UI list). path_aware=true weights the filename part
// (path_match_score); false is plain fuzzy_match_score.
inline std::uint32_t find_path_matches(const string_ref* candidates, std::uint32_t n,
                                       string_ref pattern, std::uint32_t* out_idx,
                                       int* out_score, std::uint32_t k,
                                       bool path_aware = true) noexcept {
  if (k == 0 || n == 0) return 0;
  HYLE_FATAL_ASSERT(candidates != nullptr && out_idx != nullptr &&
                    out_score != nullptr);
  // (score, index) live in parallel caller buffers; [0..count) is a MIN-heap keyed on
  // score (ties: larger index is "smaller", so equal scores evict the higher index and
  // the lower index wins -- matching the tie-break). "worse" = lower score, or equal
  // score with higher index.
  auto worse = [](int sa, std::uint32_t ia, int sb, std::uint32_t ib) noexcept {
    return sa < sb || (sa == sb && ia > ib);
  };
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < n; ++i) {
    int s = path_aware ? path_match_score(candidates[i], pattern)
                       : fuzzy_match_score(candidates[i], pattern);
    if (s <= 0) continue;
    if (count < k) {
      // Push: place at the end, sift up (toward the root = the WORST element).
      std::uint32_t c = count++;
      out_idx[c] = i;
      out_score[c] = s;
      while (c > 0) {
        std::uint32_t p = (c - 1) / 2;
        if (worse(out_score[c], out_idx[c], out_score[p], out_idx[p])) {
          std::swap(out_score[c], out_score[p]);
          std::swap(out_idx[c], out_idx[p]);
          c = p;
        } else {
          break;
        }
      }
    } else if (worse(out_score[0], out_idx[0], s, i)) {
      // Window full and the new entry beats the current worst (heap root): replace
      // the root, sift down.
      out_idx[0] = i;
      out_score[0] = s;
      std::uint32_t c = 0;
      for (;;) {
        if (c >= count / 2) break;
        const std::uint32_t l = c * 2 + 1;
        std::uint32_t m = l;
        if (l + 1 < count) {
          const std::uint32_t r = l + 1;
          if (worse(out_score[r], out_idx[r], out_score[l], out_idx[l])) m = r;
        }
        if (!worse(out_score[m], out_idx[m], out_score[c], out_idx[c])) break;
        std::swap(out_score[c], out_score[m]);
        std::swap(out_idx[c], out_idx[m]);
        c = m;
      }
    }
  }
  // Heap-sort the survivors best-first: repeatedly pop the worst (root) to the
  // shrinking tail. After each pop the front [0..size) is re-heapified, so the worst
  // elements pile up at the end and [0..count) ends up best-first.
  for (std::uint32_t size = count; size > 1;) {
    --size;
    std::swap(out_score[0], out_score[size]);  // worst goes to the tail
    std::swap(out_idx[0], out_idx[size]);
    std::uint32_t c = 0;
    for (;;) {
      if (c >= size / 2) break;
      const std::uint32_t l = c * 2 + 1;
      std::uint32_t m = l;
      if (l + 1 < size) {
        const std::uint32_t r = l + 1;
        if (worse(out_score[r], out_idx[r], out_score[l], out_idx[l])) m = r;
      }
      if (!worse(out_score[m], out_idx[m], out_score[c], out_idx[c])) break;
      std::swap(out_score[c], out_score[m]);
      std::swap(out_idx[c], out_idx[m]);
      c = m;
    }
  }
  return count;
}

}  // namespace hyle::container

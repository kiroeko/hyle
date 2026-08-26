#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include "hyle/container/hash.h"  // Hash<> primary template (+ HYLE_CONTAINER_API)
#include "hyle/core/compiler.h"
#include "hyle/mem/allocator.h"

namespace hyle::container {

// SIMD search cores (string_ref.cc). Kept out-of-line/exported so the public
// header pulls in no <emmintrin.h>; the runtime branch of the corresponding
// string_ref members delegates here, the constexpr branch stays a scalar loop.
// The ~1ns call seam is negligible on the multi-KB scans these accelerate.
namespace detail {
HYLE_CONTAINER_API std::uint32_t str_count(const char* data, std::uint32_t size,
                                           char c) noexcept;
// Reverse char scan: index of the LAST `c` in data[0,size), or npos. SSE2
// backward blocks (no memrchr on Windows; the scalar backward loop lost std
// ~20x). Returns string_ref::npos-equivalent 0xFFFFFFFF when absent.
HYLE_CONTAINER_API std::uint32_t str_rfind_char(const char* data,
                                                std::uint32_t size,
                                                char c) noexcept;
// Character-set membership scans. `set` is the delimiter set (any length). RUNTIME
// uses SSE4.2 PCMPISTRI (_SIDD_CMP_EQUAL_ANY: one instruction tests 16 haystack bytes
// against a <=16-char set) when the CPU has SSE4.2 AND set.size<=16 -- the same
// hardware string-search MSVC STL's find_*_of family uses; else a 256-bool table
// scan. Return string_ref::npos-equivalent 0xFFFFFFFF when nothing matches. `want`:
// true = first/last IN the set (find_*_of), false = first/last NOT in set (not_of).
HYLE_CONTAINER_API std::uint32_t str_find_first_of(const char* data,
                                                   std::uint32_t size,
                                                   const char* set,
                                                   std::uint32_t set_size,
                                                   std::uint32_t pos,
                                                   bool want) noexcept;
HYLE_CONTAINER_API std::uint32_t str_find_last_of(const char* data,
                                                  std::uint32_t size,
                                                  const char* set,
                                                  std::uint32_t set_size,
                                                  std::uint32_t pos,
                                                  bool want) noexcept;
// Substring search. RUNTIME forwards to std::string_view (MSVC's two-way vectorized
// search is the measured SOTA; our memchr-candidate walk lost 2.5-4x). `rev`: false =
// find (first at/after pos), true = rfind (last at/before pos). npos = 0xFFFFFFFF.
HYLE_CONTAINER_API std::uint32_t str_find_substr(const char* data,
                                                 std::uint32_t size,
                                                 const char* needle,
                                                 std::uint32_t needle_size,
                                                 std::uint32_t pos,
                                                 bool rev) noexcept;
}  // namespace detail

// Non-owning string view = pointer + length + a `null_terminated` bit. The bit
// is the real increment over std::string_view (conventions §7): when set,
// `data[size] == 0` is guaranteed, so the view can be handed to a C API as a
// C string with zero copy. clay's compact 16-byte layout is kept so the struct
// remains trivially copyable across internal erased seams; the algorithms are reimplemented in
// C++23 instead of delegating to std::string_view -- string_ref IS the std
// replacement. It uses only the non-owning <string_view> algorithms where the
// platform STL provides a measured faster primitive; no owning string or stream
// type crosses the interface.
//
// `size` is uint32_t (clay parity): a string_ref addresses <= 4 GiB. The two
// uint32_t fields pack against the 8-byte pointer for a tight 16-byte POD.
struct string_ref {
  const char* data = nullptr;
  std::uint32_t size = 0;
  std::uint32_t null_terminated = 0;  // 1 => data[size]=='\0' is allocated

  // Sentinel returned by find-family when nothing matches.
  static constexpr std::uint32_t npos = 0xFFFFFFFFu;

  constexpr string_ref() noexcept = default;
  constexpr string_ref(const char* d, std::uint32_t s,
                       bool nt = false) noexcept
      : data(d), size(s), null_terminated(nt ? 1u : 0u) {}

  // From a C string: scans for the terminator, so the result is null-terminated
  // by construction. (Implicit on purpose: "foo" should flow in as a literal.)
  constexpr string_ref(const char* s) noexcept : data(s) {  // NOLINT(*-explicit)
    if (s != nullptr) {
      std::uint32_t n = 0;
      while (s[n] != '\0') {
        HYLE_FATAL_ASSERT(n != npos);
        ++n;
      }
      size = n;
    }
    null_terminated = 1;
  }

  // From any contiguous char range (std::string / std::string_view / our own
  // string types) WITHOUT including their headers: a constrained template binds
  // to anything exposing data()->const char* and size()->integral. The
  // terminator is unknown for a generic range, so we conservatively clear the
  // bit (callers that know better use the (ptr,size,true) ctor or c_str()).
  template <class R>
    requires(!std::is_same_v<std::remove_cvref_t<R>, string_ref> &&
             requires(const R& r) {
               { r.data() } -> std::convertible_to<const char*>;
               { r.size() } -> std::convertible_to<std::size_t>;
             })
  string_ref(const R& r) noexcept  // NOLINT(*-explicit)
      : data(r.data()) {
    HYLE_FATAL_ASSERT(r.size() <= npos);
    size = static_cast<std::uint32_t>(r.size());
  }

  constexpr char operator[](std::uint32_t i) const noexcept { return data[i]; }
  constexpr char front() const noexcept { return data[0]; }
  constexpr char back() const noexcept { return data[size - 1]; }
  constexpr const char* begin() const noexcept { return data; }
  constexpr const char* end() const noexcept {
    return data == nullptr ? nullptr : data + size;
  }
  constexpr bool empty() const noexcept { return size == 0; }

  // Returns data only if it is safe as a C string; fail-fast otherwise (a bug:
  // the caller must check null_terminated or use the allocating c_str(s,alloc)).
  const char* c_str() const noexcept {
    HYLE_FATAL_ASSERT(data == nullptr || null_terminated);
    return data;
  }

  // Sub-view [start, start+len). Clamped to bounds. null_terminated is inherited
  // only by the tail that still ends at our end (a prefix is not terminated).
  constexpr string_ref substr(std::uint32_t start,
                              std::uint32_t len = npos) const noexcept {
    if (start > size) start = size;
    std::uint32_t avail = size - start;
    if (len > avail) len = avail;
    bool nt = null_terminated && (start + len == size);
    return data == nullptr ? string_ref{}
                           : string_ref{data + start, len, nt};
  }

  constexpr std::uint32_t find(char c, std::uint32_t pos = 0) const noexcept {
    if (pos >= size) return npos;
    if consteval {
      for (std::uint32_t i = pos; i < size; ++i)
        if (data[i] == c) return i;
      return npos;
    }
    // The standard library's char finder is a tuned runtime-dispatched helper
    // on MSVC and maps to the platform memchr machinery elsewhere. Keep this
    // header-inline: an extra HYLE DLL hop nearly doubled a 4 KiB miss scan.
    const std::size_t result = std::string_view(data, size).find(c, pos);
    return result == std::string_view::npos
               ? npos
               : static_cast<std::uint32_t>(result);
  }
  // constexpr: naive O(n*m) window walk. RUNTIME: forwards to std::string_view's
  // two-way vectorized substring search (the measured SOTA; see string_ref.cc).
  constexpr std::uint32_t find(string_ref needle,
                               std::uint32_t pos = 0) const noexcept {
    if (needle.size == 0) return pos <= size ? pos : npos;
    if (needle.size > size) return npos;
    std::uint32_t last = size - needle.size;  // no overflow: needle.size<=size
    if (pos > last) return npos;
    if consteval {
      for (std::uint32_t i = pos; i <= last; ++i)
        if (region_eq(i, needle)) return i;
      return npos;
    }
    return detail::str_find_substr(data, size, needle.data, needle.size, pos,
                                   /*rev=*/false);
  }
  constexpr std::uint32_t rfind(char c,
                                std::uint32_t pos = npos) const noexcept {
    if (size == 0) return npos;
    std::uint32_t hi = (pos >= size) ? size - 1 : pos;
    if consteval {
      for (std::uint32_t i = hi;; --i) {
        if (data[i] == c) return i;
        if (i == 0) break;
      }
      return npos;
    }
    // RUNTIME: SSE2 backward scan over data[0, hi+1) (no memrchr on Windows;
    // the scalar backward byte loop lost std::string_view::rfind ~20x).
    return detail::str_rfind_char(data, hi + 1, c);
  }
  // Last occurrence of `needle` starting at or before `pos`.
  constexpr std::uint32_t rfind(string_ref needle,
                                std::uint32_t pos = npos) const noexcept {
    if (needle.size == 0) return size < pos ? size : pos;
    if (needle.size > size) return npos;
    std::uint32_t last = size - needle.size;  // no overflow: needle.size<=size
    std::uint32_t hi = (pos > last) ? last : pos;
    if consteval {
      for (std::uint32_t i = hi;; --i) {
        if (region_eq(i, needle)) return i;
        if (i == 0) break;
      }
      return npos;
    }
    // RUNTIME: mirror find(needle)'s memchr strategy backward -- str_rfind_char
    // (SSE2) locates first-char candidates from the high end, region_eq (memcmp)
    // verifies. Far fewer byte compares than the scalar backward window walk
    // (which lost std::string_view::rfind ~20x). Candidates live in [0, hi] on
    // the FIRST char (a match at i needs needle.data[0] at i <= hi).
    // RUNTIME: forwards to std::string_view::rfind (two-way vectorized SOTA).
    return detail::str_find_substr(data, size, needle.data, needle.size, hi,
                                   /*rev=*/true);
  }

  // First position of ANY char in `set`. constexpr: O(n*m) scan. RUNTIME delegates
  // to str_find_first_of (forwards to std::string_view::find_first_of, the measured
  // SOTA -- MSVC's vectorized char-set scan).
  constexpr std::uint32_t find_first_of(string_ref set,
                                        std::uint32_t pos = 0) const noexcept {
    if (pos >= size) return npos;
    if consteval {
      for (std::uint32_t i = pos; i < size; ++i)
        if (set.find(data[i]) != npos) return i;
      return npos;
    }
    return detail::str_find_first_of(data, size, set.data, set.size, pos,
                                     /*want=*/true);
  }
  // First position NOT in `set`. constexpr: O(n*m) scan. RUNTIME delegates to
  // str_find_first_of with want=false (SSE4.2 PCMPISTRI negated-match, else table).
  constexpr std::uint32_t find_first_not_of(
      string_ref set, std::uint32_t pos = 0) const noexcept {
    if (pos >= size) return npos;
    if consteval {
      for (std::uint32_t i = pos; i < size; ++i)
        if (set.find(data[i]) == npos) return i;
      return npos;
    }
    return detail::str_find_first_of(data, size, set.data, set.size, pos,
                                     /*want=*/false);
  }
  constexpr std::uint32_t find_last_of(string_ref set,
                                       std::uint32_t pos = npos) const noexcept {
    if (size == 0) return npos;
    if consteval {
      std::uint32_t i = (pos >= size) ? size - 1 : pos;
      for (;; --i) {
        if (set.find(data[i]) != npos) return i;
        if (i == 0) break;
      }
      return npos;
    }
    std::uint32_t hi = (pos >= size) ? size - 1 : pos;
    return detail::str_find_last_of(data, hi + 1, set.data, set.size, hi,
                                    /*want=*/true);
  }
  constexpr std::uint32_t find_last_not_of(
      string_ref set, std::uint32_t pos = npos) const noexcept {
    if (size == 0) return npos;
    if consteval {
      std::uint32_t i = (pos >= size) ? size - 1 : pos;
      for (;; --i) {
        if (set.find(data[i]) == npos) return i;
        if (i == 0) break;
      }
      return npos;
    }
    std::uint32_t hi = (pos >= size) ? size - 1 : pos;
    return detail::str_find_last_of(data, hi + 1, set.data, set.size, hi,
                                    /*want=*/false);
  }

  constexpr bool starts_with(string_ref p) const noexcept {
    return size >= p.size && region_eq(0, p);
  }
  constexpr bool ends_with(string_ref s) const noexcept {
    return size >= s.size && region_eq(size - s.size, s);
  }
  constexpr bool contains(string_ref s) const noexcept {
    return find(s) != npos;
  }
  constexpr bool contains(char c) const noexcept { return find(c) != npos; }

  // Number of occurrences of `c` (clay cstrchar). RUNTIME borrows a SIMD count
  // core (str_count in string_ref.cc: SSE2 cmpeq + psadbw accumulate, the same
  // vectorization std::count uses); the scalar accumulate stays the constexpr
  // path. std::count auto-vectorizes and beat the scalar loop ~25x, so this
  // closes that gap without pulling <emmintrin.h> into the public header.
  constexpr std::uint32_t count(char c) const noexcept {
    if consteval {
      std::uint32_t n = 0;
      for (std::uint32_t i = 0; i < size; ++i) n += (data[i] == c);
      return n;
    }
    return detail::str_count(data, size, c);
  }

  // View with leading/trailing chars from `cut` removed (default: ASCII space).
  constexpr string_ref trim(string_ref cut = " \t\r\n") const noexcept {
    std::uint32_t b = find_first_not_of(cut);
    if (b == npos) return string_ref{};
    std::uint32_t e = find_last_not_of(cut);
    return substr(b, e - b + 1);
  }

  // Copy into a caller buffer of `cap` bytes; always NUL-terminates. Returns the
  // number of chars written (excluding the terminator). Truncates if cap is too
  // small (cap==0 writes nothing).
  std::uint32_t copy_to(char* dst, std::uint32_t cap) const noexcept {
    if (cap == 0) return 0;
    std::uint32_t n = size < (cap - 1) ? size : (cap - 1);
    if (n != 0) std::memcpy(dst, data, n);
    dst[n] = '\0';
    return n;
  }

 private:
  // Bytewise equality of the window [at, at+other.size) against `other`. At
  // RUNTIME this is memcmp (libc lowers it to SIMD); the scalar loop is kept for
  // the constexpr path (memcmp is not constexpr). region_eq powers starts_with/
  // ends_with/find(needle), so the memcmp win propagates to all of them.
  constexpr bool region_eq(std::uint32_t at, string_ref other) const noexcept {
    if consteval {
      for (std::uint32_t i = 0; i < other.size; ++i)
        if (data[at + i] != other.data[i]) return false;
      return true;
    }
    return other.size == 0 ||
           std::memcmp(data + at, other.data, other.size) == 0;
  }
};

// Preferred C++ spelling. The lower-case name is retained for the existing
// implementation and source consumers; both denote the same 16-byte view.
using StringView = string_ref;

static_assert(sizeof(string_ref) == 16, "string_ref must stay a 16-byte POD");
static_assert(std::is_trivially_copyable_v<string_ref>);

// ---- comparison (value semantics, lexicographic then length) ----------------
inline bool operator==(string_ref a, string_ref b) noexcept {
  return a.size == b.size &&
         (a.size == 0 || std::memcmp(a.data, b.data, a.size) == 0);
}
inline std::strong_ordering operator<=>(string_ref a, string_ref b) noexcept {
  std::uint32_t n = a.size < b.size ? a.size : b.size;
  int c = n ? std::memcmp(a.data, b.data, n) : 0;
  if (c < 0) return std::strong_ordering::less;
  if (c > 0) return std::strong_ordering::greater;
  return a.size <=> b.size;
}

// Case-insensitive ASCII equality (clay cstrequal_ignore_case).
inline bool equal_ignore_case(string_ref a, string_ref b) noexcept {
  if (a.size != b.size) return false;
  auto lower = [](char c) -> char {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + ('a' - 'A')) : c;
  };
  for (std::uint32_t i = 0; i < a.size; ++i)
    if (lower(a.data[i]) != lower(b.data[i])) return false;
  return true;
}

// ---- allocator-backed helpers (clay cstring parity) -------------------------

// To a C string: returns data directly when already null-terminated (zero copy),
// otherwise allocates a NUL-terminated copy via `alloc` (caller owns it).
inline const char* c_str(string_ref s, mem::Allocator alloc) {
  if (s.null_terminated) return s.data;
  const std::size_t bytes = static_cast<std::size_t>(s.size) + 1;
  char* p = static_cast<char*>(alloc.allocate(bytes));
  if (p == nullptr) return nullptr;
  if (s.size != 0) std::memcpy(p, s.data, s.size);
  p[s.size] = '\0';
  return p;
}

// Owning, NUL-terminated copy via `alloc`. Free with the same allocator and
// size+1 (the result's null_terminated bit is set).
inline string_ref dup(string_ref s, mem::Allocator alloc) {
  const std::size_t bytes = static_cast<std::size_t>(s.size) + 1;
  char* p = static_cast<char*>(alloc.allocate(bytes));
  if (p == nullptr) return string_ref{};
  if (s.size) std::memcpy(p, s.data, s.size);
  p[s.size] = '\0';
  return string_ref{p, s.size, true};
}

// Owning concatenation a+b, NUL-terminated, via `alloc` (clay cstrcat).
inline string_ref concat(string_ref a, string_ref b, mem::Allocator alloc) {
  // CONTRACT: a.size + b.size fits in u32 (a string_ref's size type). This is a
  // caller obligation, asserted (not defensively branched) -- same discipline as
  // mem/'s pow2-alignment asserts: a perf library constrains callers by contract,
  // and a silent under-allocated result would only hide a >4GiB-concat caller bug.
  // The u64 sum exists solely to feed the assert without itself wrapping.
  HYLE_FATAL_ASSERT(static_cast<std::uint64_t>(a.size) + b.size <= 0xFFFFFFFFull);
  std::uint32_t n = a.size + b.size;
  const std::size_t bytes = static_cast<std::size_t>(n) + 1;
  char* p = static_cast<char*>(alloc.allocate(bytes));
  if (p == nullptr) return string_ref{};
  if (a.size) std::memcpy(p, a.data, a.size);
  if (b.size) std::memcpy(p + a.size, b.data, b.size);
  p[n] = '\0';
  return string_ref{p, n, true};
}

// Tail of `s` from `start` (no allocation). Empty if start past the end. Keeps
// the null_terminated bit -- a suffix shares our terminator.
inline constexpr string_ref slice(string_ref s, std::uint32_t start) noexcept {
  return start < s.size
             ? string_ref{s.data + start, s.size - start, s.null_terminated != 0}
             : string_ref{};
}

// Splits *s at the first `c`: returns the part before it and advances *s past
// it. If `c` is absent, returns all of *s and sets *s empty. No allocation --
// both pieces are sub-views of the original buffer (clay cstrsplit).
inline constexpr string_ref split(string_ref* s, char c) noexcept {
  std::uint32_t at = s->find(c);
  if (at == string_ref::npos) {
    string_ref all = *s;
    *s = string_ref{};
    return all;
  }
  string_ref head = s->substr(0, at);
  *s = slice(*s, at + 1);
  return head;
}

// Splits *s at the LAST `c`: returns the part after it and shortens *s to the
// part before it. Mirror of split() consuming right-to-left. No allocation
// (clay cstrrsplit).
inline constexpr string_ref rsplit(string_ref* s, char c) noexcept {
  std::uint32_t at = s->rfind(c);
  if (at == string_ref::npos) {
    string_ref all = *s;
    *s = string_ref{};
    return all;
  }
  string_ref tail = slice(*s, at + 1);
  *s = s->substr(0, at);
  return tail;
}

// ---- user-defined literal ---------------------------------------------------
namespace literals {
// "name"_sr -> string_ref. A literal is always null-terminated, so set the bit.
constexpr string_ref operator""_sr(const char* s, std::size_t n) noexcept {
  return string_ref{s, static_cast<std::uint32_t>(n), true};
}
}  // namespace literals

// ---- Hash specialization (lets string_ref be a SwissTable key) --------------
// Uses the same process-seeded string policy as std::string/string_view. Only
// instantiated when string_ref is actually used as a key, so headers that just
// pass string_ref by value never pull a link dependency on its long-input unit.
template <>
struct Hash<string_ref> : StringHash {
  using StringHash::StringHash;
};

}  // namespace hyle::container

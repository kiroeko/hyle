#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "hyle/container/hash.h"  // Hash<> primary template (+ HYLE_CONTAINER_API)
#include "hyle/macros.h"
#include "hyle/mem/allocator.h"

namespace hyle::container {

// Non-owning string view = pointer + length + a `null_terminated` bit. The bit
// is the real increment over std::string_view (conventions §7): when set,
// `data[size] == 0` is guaranteed, so the view can be handed to a C API as a
// C string with zero copy. clay's cstring.h ABI (16 bytes, C layout) is kept so
// the struct passes a C-ABI seam unchanged; the algorithms are reimplemented in
// C++23 instead of delegating to std::string_view -- string_ref IS the std
// replacement, so this header pulls in no <string>/<string_view>/<iostream>.
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
      while (s[n] != '\0') ++n;
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
      : data(r.data()), size(static_cast<std::uint32_t>(r.size())) {}

  constexpr char operator[](std::uint32_t i) const noexcept { return data[i]; }
  constexpr char front() const noexcept { return data[0]; }
  constexpr char back() const noexcept { return data[size - 1]; }
  constexpr const char* begin() const noexcept { return data; }
  constexpr const char* end() const noexcept { return data + size; }
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
    return string_ref{data + start, len, nt};
  }

  constexpr std::uint32_t find(char c, std::uint32_t pos = 0) const noexcept {
    for (std::uint32_t i = pos; i < size; ++i)
      if (data[i] == c) return i;
    return npos;
  }
  // Naive O(n*m) search (region_eq bails on the first mismatching byte, and the
  // single-char scan vectorizes). Fine for the short needles this sees; a
  // sublinear searcher is not worth the code for view-sized strings.
  constexpr std::uint32_t find(string_ref needle,
                               std::uint32_t pos = 0) const noexcept {
    if (needle.size == 0) return pos <= size ? pos : npos;
    if (needle.size > size) return npos;
    std::uint32_t last = size - needle.size;  // no overflow: needle.size<=size
    for (std::uint32_t i = pos; i <= last; ++i)
      if (region_eq(i, needle)) return i;
    return npos;
  }
  constexpr std::uint32_t rfind(char c,
                                std::uint32_t pos = npos) const noexcept {
    if (size == 0) return npos;
    std::uint32_t i = (pos >= size) ? size - 1 : pos;
    for (;; --i) {
      if (data[i] == c) return i;
      if (i == 0) break;
    }
    return npos;
  }
  // Last occurrence of `needle` starting at or before `pos`.
  constexpr std::uint32_t rfind(string_ref needle,
                                std::uint32_t pos = npos) const noexcept {
    if (needle.size == 0) return size < pos ? size : pos;
    if (needle.size > size) return npos;
    std::uint32_t last = size - needle.size;  // no overflow: needle.size<=size
    std::uint32_t i = (pos > last) ? last : pos;
    for (;; --i) {
      if (region_eq(i, needle)) return i;
      if (i == 0) break;
    }
    return npos;
  }

  constexpr std::uint32_t find_first_of(string_ref set,
                                        std::uint32_t pos = 0) const noexcept {
    for (std::uint32_t i = pos; i < size; ++i)
      if (set.find(data[i]) != npos) return i;
    return npos;
  }
  constexpr std::uint32_t find_first_not_of(
      string_ref set, std::uint32_t pos = 0) const noexcept {
    for (std::uint32_t i = pos; i < size; ++i)
      if (set.find(data[i]) == npos) return i;
    return npos;
  }
  constexpr std::uint32_t find_last_of(string_ref set,
                                       std::uint32_t pos = npos) const noexcept {
    if (size == 0) return npos;
    std::uint32_t i = (pos >= size) ? size - 1 : pos;
    for (;; --i) {
      if (set.find(data[i]) != npos) return i;
      if (i == 0) break;
    }
    return npos;
  }
  constexpr std::uint32_t find_last_not_of(
      string_ref set, std::uint32_t pos = npos) const noexcept {
    if (size == 0) return npos;
    std::uint32_t i = (pos >= size) ? size - 1 : pos;
    for (;; --i) {
      if (set.find(data[i]) == npos) return i;
      if (i == 0) break;
    }
    return npos;
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

  // Number of occurrences of `c` (clay cstrchar).
  constexpr std::uint32_t count(char c) const noexcept {
    std::uint32_t n = 0;
    for (std::uint32_t i = 0; i < size; ++i) n += (data[i] == c);
    return n;
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
    std::memcpy(dst, data, n);
    dst[n] = '\0';
    return n;
  }

 private:
  // Bytewise equality of the window [at, at+other.size) against `other`.
  constexpr bool region_eq(std::uint32_t at, string_ref other) const noexcept {
    for (std::uint32_t i = 0; i < other.size; ++i)
      if (data[at + i] != other.data[i]) return false;
    return true;
  }
};

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
  char* p = static_cast<char*>(alloc.allocate(s.size + 1));
  if (p == nullptr) return nullptr;
  std::memcpy(p, s.data, s.size);
  p[s.size] = '\0';
  return p;
}

// Owning, NUL-terminated copy via `alloc`. Free with the same allocator and
// size+1 (the result's null_terminated bit is set).
inline string_ref dup(string_ref s, mem::Allocator alloc) {
  char* p = static_cast<char*>(alloc.allocate(s.size + 1));
  if (p == nullptr) return string_ref{};
  if (s.size) std::memcpy(p, s.data, s.size);
  p[s.size] = '\0';
  return string_ref{p, s.size, true};
}

// Owning concatenation a+b, NUL-terminated, via `alloc` (clay cstrcat).
inline string_ref concat(string_ref a, string_ref b, mem::Allocator alloc) {
  std::uint32_t n = a.size + b.size;
  char* p = static_cast<char*>(alloc.allocate(n + 1));
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
// Hashes the bytes with XXH3 (hash_bytes lives in hash.cc). Only instantiated
// when string_ref is actually used as a key, so headers that just pass
// string_ref by value never pull a link dependency on the hash unit.
template <>
struct Hash<string_ref> {
  std::uint64_t operator()(string_ref s) const noexcept {
    return hash_bytes(s.data, s.size);
  }
};

}  // namespace hyle::container

#pragma once

#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>
#include <type_traits>

#include "hyle/container/hash.h"  // Hash<> primary template (+ HYLE_CONTAINER_API)
#include "hyle/core/compiler.h"

namespace hyle::container {

// SIMD search cores (string_view.cc). Kept out-of-line/exported so the public
// header pulls in no ISA headers; runtime members delegate here while constexpr
// evaluation keeps a scalar implementation.
namespace detail {
HYLE_CONTAINER_API std::uint32_t str_count(const char* data, std::uint32_t size,
                                           char c) noexcept;
// Reverse char scan: index of the LAST `c` in data[0,size), or npos. Returns
// StringView::npos-equivalent 0xFFFFFFFF when absent.
HYLE_CONTAINER_API std::uint32_t str_rfind_char(const char* data,
                                                std::uint32_t size,
                                                char c) noexcept;
// Character-set membership scans. `set` is the delimiter set (any length). RUNTIME
// uses SSE4.2 PCMPISTRI (_SIDD_CMP_EQUAL_ANY: one instruction tests 16 haystack bytes
// against a <=16-char set) when the CPU has SSE4.2 AND set.size<=16 -- the same
// hardware string-search MSVC STL's find_*_of family uses; else a 256-bool table
// scan. Return StringView::npos-equivalent 0xFFFFFFFF when nothing matches. `want`:
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
// Substring search delegates to the standard-library implementation. `rev`:
// false = find (first at/after pos), true = rfind (last at/before pos).
// npos = 0xFFFFFFFF.
HYLE_CONTAINER_API std::uint32_t str_find_substr(const char* data,
                                                 std::uint32_t size,
                                                 const char* needle,
                                                 std::uint32_t needle_size,
                                                 std::uint32_t pos,
                                                 bool rev) noexcept;
}  // namespace detail

// Standard-shaped non-owning string view plus one explicit fact that
// std::string_view cannot carry: when is_null_terminated() is true,
// data()[size()] is known to be '\0'. That permits zero-copy calls into APIs
// requiring a C string without pretending every borrowed slice is terminated.
// StringView is a trivially-copyable 16-byte value and never owns its bytes.
//
// The compact uint32_t length makes the supported maximum explicit: one view
// addresses at most 4 GiB. Inputs outside that bound fail fast at construction.
struct StringView {
 private:
  const char* data_ = nullptr;
  std::uint32_t size_ = 0;
  std::uint32_t null_terminated_ = 0;  // 1 => data_[size_]=='\0' is readable

 public:
  // Sentinel returned by find-family when nothing matches.
  static constexpr std::uint32_t npos = 0xFFFFFFFFu;

  constexpr StringView() noexcept = default;
  constexpr StringView(const char* d, std::uint32_t s,
                       bool nt = false) noexcept
      : data_(d), size_(s), null_terminated_(nt ? 1u : 0u) {
  }

  // From a C string: scans for the terminator, so the result is null-terminated
  // by construction. (Implicit on purpose: "foo" should flow in as a literal.)
  constexpr StringView(const char* s) noexcept  // NOLINT(*-explicit)
      : data_(s) {
    if (s != nullptr) {
      std::uint32_t n = 0;
      while (s[n] != '\0') {
        HYLE_FATAL_ASSERT(n != npos);
        ++n;
      }
      size_ = n;
    }
    null_terminated_ = 1;
  }

  // From any contiguous char range (std::string / std::string_view / our own
  // string types) WITHOUT including their headers: a constrained template binds
  // to anything exposing data()->const char* and size()->integral. The
  // terminator is unknown for a generic range, so we conservatively clear the
  // bit (callers that know better use the (ptr,size_,true) ctor or c_str()).
  template <class R>
    requires(!std::is_same_v<std::remove_cvref_t<R>, StringView> &&
             requires(const R& r) {
               { r.data() } noexcept -> std::convertible_to<const char*>;
               { r.size() } noexcept -> std::convertible_to<std::size_t>;
             })
  StringView(const R& r) noexcept  // NOLINT(*-explicit)
      : data_(r.data()) {
    HYLE_FATAL_ASSERT(r.size() <= npos);
    size_ = static_cast<std::uint32_t>(r.size());
  }

  using size_type = std::uint32_t;

  constexpr const char* data() const noexcept { return data_; }
  constexpr size_type size() const noexcept { return size_; }
  constexpr size_type length() const noexcept { return size_; }
  constexpr bool is_null_terminated() const noexcept {
    return null_terminated_ != 0;
  }
  constexpr std::string_view as_std() const noexcept {
    HYLE_FATAL_ASSERT(data_ != nullptr || size_ == 0);
    return data_ == nullptr ? std::string_view{}
                            : std::string_view{data_, size_};
  }
  constexpr operator std::string_view() const noexcept { return as_std(); }

  constexpr char operator[](std::uint32_t i) const noexcept { return data_[i]; }
  constexpr char front() const noexcept { return data_[0]; }
  constexpr char back() const noexcept { return data_[size_ - 1]; }
  constexpr const char* begin() const noexcept { return data_; }
  constexpr const char* end() const noexcept {
    return data_ == nullptr ? nullptr : data_ + size_;
  }
  constexpr const char* cbegin() const noexcept { return begin(); }
  constexpr const char* cend() const noexcept { return end(); }
  constexpr bool empty() const noexcept { return size_ == 0; }

  constexpr void remove_prefix(std::uint32_t count) noexcept {
    HYLE_FATAL_ASSERT(count <= size_);
    if (data_ != nullptr) data_ += count;
    size_ -= count;
  }

  constexpr void remove_suffix(std::uint32_t count) noexcept {
    HYLE_FATAL_ASSERT(count <= size_);
    size_ -= count;
    if (count != 0) null_terminated_ = 0;
  }

  // Returns data_ only if an end terminator is known; fail-fast otherwise. This
  // does not prove that the logical bytes contain no earlier NUL. Validate that
  // separately when a C API must observe exactly `size_` bytes. To obtain
  // ownership, copy into std::string/std::pmr::string or caller storage.
  const char* c_str() const noexcept {
    HYLE_FATAL_ASSERT(data_ == nullptr || null_terminated_);
    return data_;
  }

  // Sub-view [start, start+len). Clamped to bounds. null_terminated_ is inherited
  // only by the tail that still ends at our end (a prefix is not terminated).
  constexpr StringView substr(std::uint32_t start,
                              std::uint32_t len = npos) const noexcept {
    if (start > size_) start = size_;
    std::uint32_t avail = size_ - start;
    if (len > avail) len = avail;
    bool nt = null_terminated_ && (start + len == size_);
    return data_ == nullptr ? StringView{}
                           : StringView{data_ + start, len, nt};
  }

  constexpr std::uint32_t find(char c, std::uint32_t pos = 0) const noexcept {
    if (pos >= size_) return npos;
    if consteval {
      for (std::uint32_t i = pos; i < size_; ++i)
        if (data_[i] == c) return i;
      return npos;
    }
    // The standard library's char finder is a tuned runtime-dispatched helper
    // on MSVC and maps to the platform memchr machinery elsewhere. Keep this
    // header-inline: an extra HYLE DLL hop nearly doubled a 4 KiB miss scan.
    const std::size_t result = std::string_view(data_, size_).find(c, pos);
    return result == std::string_view::npos
               ? npos
               : static_cast<std::uint32_t>(result);
  }
  // constexpr: O(n*m) window walk. Runtime delegates to std::string_view.
  constexpr std::uint32_t find(StringView needle,
                               std::uint32_t pos = 0) const noexcept {
    if (needle.size_ == 0) return pos <= size_ ? pos : npos;
    if (needle.size_ > size_) return npos;
    std::uint32_t last = size_ - needle.size_;  // no overflow: needle.size_<=size_
    if (pos > last) return npos;
    if consteval {
      for (std::uint32_t i = pos; i <= last; ++i)
        if (region_eq(i, needle)) return i;
      return npos;
    }
    return detail::str_find_substr(data_, size_, needle.data_, needle.size_, pos,
                                   /*rev=*/false);
  }
  constexpr std::uint32_t rfind(char c,
                                std::uint32_t pos = npos) const noexcept {
    if (size_ == 0) return npos;
    std::uint32_t hi = (pos >= size_) ? size_ - 1 : pos;
    if consteval {
      for (std::uint32_t i = hi;; --i) {
        if (data_[i] == c) return i;
        if (i == 0) break;
      }
      return npos;
    }
    // Runtime: dispatched backward scan over data_[0, hi+1).
    return detail::str_rfind_char(data_, hi + 1, c);
  }
  // Last occurrence of `needle` starting at or before `pos`.
  constexpr std::uint32_t rfind(StringView needle,
                                std::uint32_t pos = npos) const noexcept {
    if (needle.size_ == 0) return size_ < pos ? size_ : pos;
    if (needle.size_ > size_) return npos;
    std::uint32_t last = size_ - needle.size_;  // no overflow: needle.size_<=size_
    std::uint32_t hi = (pos > last) ? last : pos;
    if consteval {
      for (std::uint32_t i = hi;; --i) {
        if (region_eq(i, needle)) return i;
        if (i == 0) break;
      }
      return npos;
    }
    // Runtime delegates to std::string_view::rfind.
    return detail::str_find_substr(data_, size_, needle.data_, needle.size_, hi,
                                   /*rev=*/true);
  }

  // First position of ANY char in `set`. constexpr: O(n*m) scan. Runtime
  // delegates to the standard-library character-set search.
  constexpr std::uint32_t find_first_of(StringView set,
                                        std::uint32_t pos = 0) const noexcept {
    if (pos >= size_) return npos;
    if consteval {
      for (std::uint32_t i = pos; i < size_; ++i)
        if (set.find(data_[i]) != npos) return i;
      return npos;
    }
    return detail::str_find_first_of(data_, size_, set.data_, set.size_, pos,
                                     /*want=*/true);
  }
  // First position NOT in `set`. constexpr: O(n*m) scan. RUNTIME delegates to
  // str_find_first_of with want=false (SSE4.2 PCMPISTRI negated-match, else table).
  constexpr std::uint32_t find_first_not_of(
      StringView set, std::uint32_t pos = 0) const noexcept {
    if (pos >= size_) return npos;
    if consteval {
      for (std::uint32_t i = pos; i < size_; ++i)
        if (set.find(data_[i]) == npos) return i;
      return npos;
    }
    return detail::str_find_first_of(data_, size_, set.data_, set.size_, pos,
                                     /*want=*/false);
  }
  constexpr std::uint32_t find_last_of(StringView set,
                                       std::uint32_t pos = npos) const noexcept {
    if (size_ == 0) return npos;
    if consteval {
      std::uint32_t i = (pos >= size_) ? size_ - 1 : pos;
      for (;; --i) {
        if (set.find(data_[i]) != npos) return i;
        if (i == 0) break;
      }
      return npos;
    }
    std::uint32_t hi = (pos >= size_) ? size_ - 1 : pos;
    return detail::str_find_last_of(data_, hi + 1, set.data_, set.size_, hi,
                                    /*want=*/true);
  }
  constexpr std::uint32_t find_last_not_of(
      StringView set, std::uint32_t pos = npos) const noexcept {
    if (size_ == 0) return npos;
    if consteval {
      std::uint32_t i = (pos >= size_) ? size_ - 1 : pos;
      for (;; --i) {
        if (set.find(data_[i]) == npos) return i;
        if (i == 0) break;
      }
      return npos;
    }
    std::uint32_t hi = (pos >= size_) ? size_ - 1 : pos;
    return detail::str_find_last_of(data_, hi + 1, set.data_, set.size_, hi,
                                    /*want=*/false);
  }

  constexpr bool starts_with(StringView p) const noexcept {
    return size_ >= p.size_ && region_eq(0, p);
  }
  constexpr bool ends_with(StringView s) const noexcept {
    return size_ >= s.size_ && region_eq(size_ - s.size_, s);
  }
  constexpr bool contains(StringView s) const noexcept {
    return find(s) != npos;
  }
  constexpr bool contains(char c) const noexcept { return find(c) != npos; }

  // Number of occurrences of `c`. Runtime uses the dispatched count core; the
  // scalar loop remains available during constant evaluation.
  constexpr std::uint32_t count(char c) const noexcept {
    if consteval {
      std::uint32_t n = 0;
      for (std::uint32_t i = 0; i < size_; ++i) n += (data_[i] == c);
      return n;
    }
    return detail::str_count(data_, size_, c);
  }

  // View with leading/trailing chars from `cut` removed (default: ASCII space).
  constexpr StringView trim(StringView cut = " \t\r\n") const noexcept {
    std::uint32_t b = find_first_not_of(cut);
    if (b == npos) return StringView{};
    std::uint32_t e = find_last_not_of(cut);
    return substr(b, e - b + 1);
  }

  // Copy into caller-owned storage and always NUL-terminate when non-empty.
  // Returns bytes written (excluding the terminator); compare with size_ to
  // detect truncation. The returned bytes remain owned by `dst`'s owner.
  std::size_t copy_to(std::span<char> dst) const noexcept {
    if (dst.empty()) return 0;
    const std::size_t n =
        static_cast<std::size_t>(size_) < dst.size() - 1 ? size_ : dst.size() - 1;
    if (n != 0) std::memmove(dst.data(), data_, n);
    dst[n] = '\0';
    return n;
  }

 private:
  // Bytewise equality of the window [at, at+other.size_) against `other`. At
  // RUNTIME this is memcmp (libc lowers it to SIMD); the scalar loop is kept for
  // the constexpr path (memcmp is not constexpr). region_eq powers starts_with/
  // ends_with/find(needle), so the memcmp win propagates to all of them.
  constexpr bool region_eq(std::uint32_t at, StringView other) const noexcept {
    if consteval {
      for (std::uint32_t i = 0; i < other.size_; ++i)
        if (data_[at + i] != other.data_[i]) return false;
      return true;
    }
    return other.size_ == 0 ||
           std::memcmp(data_ + at, other.data_, other.size_) == 0;
  }
};


static_assert(sizeof(StringView) == 16,
              "StringView must stay a 16-byte borrowed value");
static_assert(std::is_trivially_copyable_v<StringView>);
static_assert(std::is_standard_layout_v<StringView>);

// ---- comparison (value semantics, lexicographic then length) ----------------
inline bool operator==(StringView a, StringView b) noexcept {
  return a.size() == b.size() &&
         (a.size() == 0 || std::memcmp(a.data(), b.data(), a.size()) == 0);
}
inline std::strong_ordering operator<=>(StringView a, StringView b) noexcept {
  std::uint32_t n = a.size() < b.size() ? a.size() : b.size();
  int c = n ? std::memcmp(a.data(), b.data(), n) : 0;
  if (c < 0) return std::strong_ordering::less;
  if (c > 0) return std::strong_ordering::greater;
  return a.size() <=> b.size();
}

// Case-insensitive ASCII equality.
inline bool equal_ignore_case(StringView a, StringView b) noexcept {
  if (a.size() != b.size()) return false;
  auto lower = [](char c) -> char {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + ('a' - 'A')) : c;
  };
  for (std::uint32_t i = 0; i < a.size(); ++i)
    if (lower(a.data()[i]) != lower(b.data()[i])) return false;
  return true;
}

// Tail of `s` from `start` (no allocation). Empty if start past the end. Keeps
// the null_terminated bit -- a suffix shares our terminator.
inline constexpr StringView slice(StringView s, std::uint32_t start) noexcept {
  if (start > s.size() || s.data() == nullptr) return StringView{};
  return StringView{s.data() + start, s.size() - start,
                    s.is_null_terminated()};
}

// Splits *s at the first `c`: returns the part before it and advances *s past
// it. If `c` is absent, returns all of *s and sets *s empty. No allocation --
// both pieces are sub-views of the original buffer.
inline constexpr StringView split(StringView* s, char c) noexcept {
  std::uint32_t at = s->find(c);
  if (at == StringView::npos) {
    StringView all = *s;
    *s = StringView{};
    return all;
  }
  StringView head = s->substr(0, at);
  *s = slice(*s, at + 1);
  return head;
}

// Splits *s at the LAST `c`: returns the part after it and shortens *s to the
// part before it. Mirror of split() consuming right-to-left. No allocation
// consuming right-to-left.
inline constexpr StringView rsplit(StringView* s, char c) noexcept {
  std::uint32_t at = s->rfind(c);
  if (at == StringView::npos) {
    StringView all = *s;
    *s = StringView{};
    return all;
  }
  StringView tail = slice(*s, at + 1);
  *s = s->substr(0, at);
  return tail;
}

// ---- user-defined literal ---------------------------------------------------
namespace literals {
// "name"_sv -> StringView. A literal is always null-terminated, so set the bit.
constexpr StringView operator""_sv(const char* s, std::size_t n) noexcept {
  return StringView{s, static_cast<std::uint32_t>(n), true};
}
}  // namespace literals

// ---- Hash specialization (lets StringView be a SwissTable key) --------------
// Uses the same process-seeded string policy as std::string/string_view. Only
// instantiated when StringView is actually used as a key, so headers that just
// pass StringView by value never pull a link dependency on its long-input unit.
template <>
struct Hash<StringView> : StringHash {
  using StringHash::StringHash;
};

}  // namespace hyle::container

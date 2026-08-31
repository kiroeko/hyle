#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

#include "hyle/container/pod_vector.h"   // PodVector<StringView> split result
#include "hyle/container/string_view.h"   // StringView + HYLE_CONTAINER_API
#include "hyle/core/compiler.h"
#include "hyle/mem/allocator.h"

// Path string manipulation (clay sys/path.h `path_api_t` -- the pure-string half:
// no syscalls, deterministic on every platform, so it lives in `container` next
// to StringView rather than in `platform`. The fs/env/os_info half of clay's
// path/os stays in platform-03.). Read-only operations return borrowed
// StringView values. Producing operations write into caller-owned spans; no path
// function hides an allocation inside a non-owning view.
//
// THREE deliberate upgrades over clay's vtable (each registered in the coverage
// map), because clay shipped inconsistent or domain-specific behavior:
//   1. extension()/strip_extension() use Python os.path.splitext semantics (a
//      leading-dot name like ".bashrc" has NO extension) -- clay's vtable entry
//      used a naive reverse-scan that disagreed with its OWN path_extension_or_end
//      (which is splitext). splitext is the robust one: strip never empties a name.
//   2. is_absolute() is a TRUE absolute-path test (POSIX root / Win drive / UNC).
//      clay's `is_relative` actually tested the Blender "//" blend-relative prefix
//      -- a domain convention, useless for general infra, not absorbed.
//   3. normalize_path() collapses on '/' as THE separator (identical on all
//      platforms, so it is unit-testable double-platform). clay used the native
//      separator and folded in Windows drive/UNC casing -- that part is coupled to
//      the fs layer and is registered for platform-03, not done here.

namespace hyle::container {

// HYLE canonical separator: '/' is emitted by every producing op and accepted as
// a separator on input everywhere. Both '/' and '\\' are recognized on input so
// Windows-style paths parse correctly without a pre-pass.
inline constexpr char kPathSep = '/';

// The OS-native separator, for callers that must hand a path to a native API.
#if defined(_WIN32)
inline constexpr char kPathSepNative = '\\';
#else
inline constexpr char kPathSepNative = '/';
#endif

// True for either separator (clay slash_is_native_compat, minus the WIN32 gate:
// HYLE treats '\\' as a separator on all platforms for parsing robustness).
HYLE_FORCE_INLINE constexpr bool is_separator(char c) noexcept {
  return c == '/' || c == '\\';
}

// ---- extension (Python splitext semantics) ----------------------------------
// Returns the file extension of `path` INCLUDING the leading dot, or an empty
// view if there is none. A dot only starts an extension when a non-dot,
// non-separator char precedes it in the same component:
//   "test.jpg" -> ".jpg"   "a.tar.gz" -> ".gz"   "test" -> {}
//   ".bashrc"  -> {}       "a.b/c"    -> {}       "" -> {}
HYLE_FORCE_INLINE constexpr StringView extension(StringView path) noexcept {
  // Scan the LAST component only, back-to-front (like base()/directory()): stop
  // at the first separator, so a deep path touches ~one filename's worth of bytes
  // instead of the whole string. `ext` latches the LAST dot (first one seen going
  // left); the FIRST non-dot char we then meet is necessarily to its LEFT, which
  // is exactly the splitext rule ("a non-dot precedes the dot") -> return at once.
  // A leading-dot name (".bashrc") never meets such a char, so it has no ext.
  std::uint32_t ext = StringView::npos;
  for (std::uint32_t i = path.size() - 1; i < path.size(); --i) {  // wraps below 0
    char c = path.data()[i];
    if (is_separator(c)) break;                       // reached basename start
    if (c == '.') {
      if (ext == StringView::npos) ext = i;           // latch the last dot
    } else if (ext != StringView::npos) {
      return StringView{path.data() + ext, path.size() - ext,
                        path.is_null_terminated() != 0};
    }
  }
  return StringView{};
}

// `path` with its extension() removed. Pairs with extension(): for a name with
// no extension this returns the whole path unchanged (never an empty string).
HYLE_FORCE_INLINE constexpr StringView strip_extension(
    StringView path) noexcept {
  StringView e = extension(path);
  if (e.size() == 0) return path;
  return path.substr(0, path.size() - e.size());
}

// ---- base / directory (basename / dirname) ----------------------------------
// Last element of `path`. Trailing separators are stripped first. "" -> ".",
// a path of only separators -> "/". (clay base)
HYLE_FORCE_INLINE constexpr StringView base(StringView path) noexcept {
  if (path.size() == 0) return StringView{".", 1, true};
  bool found = false;
  std::uint32_t end = path.size();
  for (std::uint32_t i = path.size() - 1; i < path.size(); --i) {  // wraps below 0
    if (is_separator(path.data()[i])) {
      if (found)
        return StringView{path.data() + i + 1, end - i - 1,
                          path.is_null_terminated() != 0 && end == path.size()};
      end = i;  // trailing separator: shrink the window
    } else {
      found = true;
    }
  }
  if (found)
    return StringView{path.data(), end,
                      path.is_null_terminated() != 0 && end == path.size()};
  return StringView{"/", 1, true};
}

// "Directory" of `path`: everything before base(). Trailing separators removed
// unless it is the root. No directory / empty path -> ".". (clay directory)
HYLE_FORCE_INLINE constexpr StringView directory(StringView path) noexcept {
  if (path.size() == 0) return StringView{".", 1, true};
  bool found = false;
  for (std::uint32_t i = path.size() - 1; i < path.size(); --i) {  // wraps below 0
    if (is_separator(path.data()[i])) {
      if (found) {
        if (i == 0) return StringView{"/", 1, true};
        // Skip a run of separators back to the parent's last char. Bounds check
        // FIRST so the underflowed index never dereferences (clay read OOB here
        // on an all-separator prefix like "//x").
        std::uint32_t j = i;
        while (j < path.size() && is_separator(path.data()[j])) --j;
        if (j < path.size())
          return StringView{path.data(), j + 1,
                            path.is_null_terminated() != 0 &&
                                j + 1 == path.size()};
        return StringView{"/", 1, true};
      }
    } else {
      found = true;
    }
  }
  if (found) return StringView{".", 1, true};
  return StringView{"/", 1, true};
}

// ---- classification ----------------------------------------------------------
// True absolute path: POSIX root ("/..."), a Windows drive root ("C:\" / "C:/"),
// or a UNC path ("\\..."). A bare "C:" (drive-relative) is NOT absolute.
constexpr bool is_absolute(StringView path) noexcept {
  if (path.size() == 0) return false;
  if (is_separator(path.data()[0])) return true;  // "/..." or "\\..." (UNC)
  char c0 = path.data()[0];
  bool alpha = (c0 >= 'A' && c0 <= 'Z') || (c0 >= 'a' && c0 <= 'z');
  return path.size() >= 3 && alpha && path.data()[1] == ':' &&
         is_separator(path.data()[2]);
}

// True for a UNC path: starts with two backslashes (clay path_is_unc).
constexpr bool is_unc(StringView path) noexcept {
  return path.size() >= 2 && path.data()[0] == '\\' && path.data()[1] == '\\';
}

// ---- iteration (pop a leading / trailing element) ----------------------------
// Pops the FIRST element off *path and returns it; *path advances past it and any
// following separators. Empty *path -> ".". A leading separator pops as itself
// ("/foo" -> "/", then "foo"). (clay pop_first)
inline StringView pop_first(StringView* path) noexcept {
  if (path->empty()) return StringView{".", 1, true};
  bool found = false;
  for (std::uint32_t i = 0; i < path->size(); ++i) {
    if (is_separator((*path)[i])) {
      StringView res = found ? path->substr(0, i) : path->substr(0, 1);
      std::uint32_t after = i;
      while (after < path->size() && is_separator((*path)[after])) ++after;
      *path = (after == path->size())
                  ? StringView{}
                  : StringView{path->data() + after, path->size() - after,
                               path->is_null_terminated()};
      return res;
    }
    found = true;
  }
  StringView first = *path;
  *path = StringView{};
  return first;
}

// Pops the LAST element: returns base(*path) and sets *path to directory(*path)
// (clay pop_last).
inline StringView pop_last(StringView* path) noexcept {
  StringView b = base(*path);
  *path = directory(*path);
  return b;
}

// ---- caller-owned producing operations (compiled once in path.cc) ------------
// Joins parts into output with a single '/' between each, snprintf-style:
// always NUL-terminates when cap>0 and returns the FULL length that would be
// written (`result >= output.size()` signals truncation). A leading separator on
// parts[0] and a trailing separator on the last part are preserved; interior
// separators are collapsed. Output must not overlap the input bytes.
HYLE_CONTAINER_API std::size_t join_path(
    std::span<const StringView> parts, std::span<char> output) noexcept;

inline std::size_t join_path(StringView a, StringView b,
                             std::span<char> output) noexcept {
  const StringView parts[]{a, b};
  return join_path(std::span<const StringView>{parts}, output);
}

// Splits `path` into its elements (clay split_all). Allocates the array via
// `alloc`; each element is a view INTO `path` (no string copies).
HYLE_CONTAINER_API PodVector<StringView> split_all(StringView path,
                                                   mem::Allocator alloc);

// Replaces the extension: "test.jpg" + ".png" -> "test.png". If `path` has no
// extension, `ext` is appended ("test" + ".png" -> "test.png" -- clay returned
// null here; HYLE appends, which is the useful behavior). Uses the same
// snprintf-style return contract as join_path; output must not overlap inputs.
HYLE_CONTAINER_API std::size_t replace_extension(
    StringView path, StringView ext, std::span<char> output) noexcept;

// Copies `path` into caller-owned storage and collapses redundant separators:
//   "//" -> "/" , "/./" -> "/" , resolve "<dir>/../" -> "" .
// Lexical only (does not touch the filesystem). Treats
// '/' as the separator and emits '/'. A relative path that collapses to nothing
// becomes ".". Leading ".." in a relative path is kept (cannot go above cwd);
// ".." at an absolute root is dropped. (clay path_normalize_native, rewritten as
// a component walk -- clearer than clay's in-place memmove batching, same result.)
// The result borrows `output`; nullopt means output is smaller than
// max(path.size(), 1)+1. Exact overlap between input and output is supported.
HYLE_CONTAINER_API std::optional<StringView> normalize_path(
    StringView path, std::span<char> output) noexcept;

}  // namespace hyle::container

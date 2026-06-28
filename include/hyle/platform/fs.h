#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>

#include "hyle/macros.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_PLATFORM_API
#elif defined(HYLE_PLATFORM_BUILD)
#  define HYLE_PLATFORM_API HYLE_EXPORT
#else
#  define HYLE_PLATFORM_API HYLE_IMPORT
#endif

// Filesystem + environment syscalls (clay sys/os.h `file_system_api_t` + the
// portable helpers in os.shared.cc). This is the syscall half of clay's path/os;
// the pure-string half landed in container/path.h (platform-03a).
//
// LAYERING: every path-returning call is BUFFER-FILL (char* dst, cap) and
// directory iteration is a VISITOR -- so this header pulls in neither mem nor
// container, and fs stays in `platform` (P1, depends only on error). That is a
// real improvement over clay, which allocated every result through an Allocator*
// and returned a `strings_t*` blob for directory listings. Buffer-fill calls
// follow snprintf semantics: they always NUL-terminate when cap>0 and return the
// FULL length that would be written (a return >= cap means the path was
// truncated).

namespace hyle::platform {

// ---- query (stat) -----------------------------------------------------------
// Result of stat(): a snapshot of one path. `exists` is false for anything that
// is neither a regular file nor a directory (or on error).
struct FileInfo {
  bool exists = false;
  bool is_directory = false;
  std::uint64_t size = 0;            // bytes (regular files); 0 for directories
  std::int64_t modified_time = 0;    // seconds since the epoch (mtime)
};

// Stat `path`. Never fails loudly: a missing path returns {exists=false}.
HYLE_PLATFORM_API FileInfo stat(const char* path) noexcept;

HYLE_PLATFORM_API bool exists(const char* path) noexcept;
HYLE_PLATFORM_API bool is_directory(const char* path) noexcept;
// Size in bytes, or 0 if absent / not a regular file.
HYLE_PLATFORM_API std::uint64_t file_size(const char* path) noexcept;

// ---- directories ------------------------------------------------------------
// Create one directory. Returns true on success OR if it already exists as a
// directory (idempotent, unlike a bare mkdir which fails on EEXIST).
HYLE_PLATFORM_API bool make_directory(const char* path) noexcept;

// Create `path` and every missing parent ("mkdir -p"). `/` and `\\` both split
// components. Returns true if the full path exists as a directory afterward.
HYLE_PLATFORM_API bool make_directory_recursive(const char* path) noexcept;

// Remove an empty directory. Returns true on success.
HYLE_PLATFORM_API bool remove_directory(const char* path) noexcept;

// Remove a directory and ALL its contents ("rm -rf"). Returns true on success.
// (Hard-to-reverse: deletes recursively. Caller owns the blast radius.)
HYLE_PLATFORM_API bool remove_directory_recursive(const char* path) noexcept;

// ---- files ------------------------------------------------------------------
// Remove a single file. Returns true on success.
HYLE_PLATFORM_API bool remove_file(const char* path) noexcept;

// Rename/move `from` to `to`. Returns true on success.
HYLE_PLATFORM_API bool rename_path(const char* from, const char* to) noexcept;

// Copy a single file `from` -> `to`, overwriting `to`. Returns true on success.
HYLE_PLATFORM_API bool copy_file(const char* from, const char* to) noexcept;

// Create `path` empty if absent, or bump its mtime if present (unix `touch`).
HYLE_PLATFORM_API bool touch(const char* path) noexcept;

// ---- working / special directories (buffer-fill) ----------------------------
// Current working directory into dst[cap]. Returns the full length (>= cap =>
// truncated; 0 on failure).
HYLE_PLATFORM_API std::size_t current_directory(char* dst,
                                                std::size_t cap) noexcept;

// Change the working directory. Returns true on success.
HYLE_PLATFORM_API bool change_directory(const char* path) noexcept;

// System temp directory (honors $TMPDIR/$TMP/$TEMP, else the OS default) into
// dst[cap]. Returns the full length. clay hard-coded P_tmpdir and ignored the
// environment; honoring it is the documented platform behavior.
HYLE_PLATFORM_API std::size_t temp_directory(char* dst,
                                             std::size_t cap) noexcept;

// User home directory (Win: %USERPROFILE%; POSIX: $HOME, else the passwd entry)
// into dst[cap]. Returns the full length (0 on failure). clay's `app_folder`.
HYLE_PLATFORM_API std::size_t home_directory(char* dst,
                                             std::size_t cap) noexcept;

// ---- environment ------------------------------------------------------------
// Read environment variable `name` into dst[cap] (buffer-fill). Returns the full
// value length, or npos if the variable is unset (distinct from set-but-empty,
// which returns 0). dst is always NUL-terminated when cap>0.
inline constexpr std::size_t kEnvNotSet = static_cast<std::size_t>(-1);
HYLE_PLATFORM_API std::size_t get_env_var(const char* name, char* dst,
                                          std::size_t cap) noexcept;

// ---- directory iteration (visitor, no allocation) ---------------------------
// Type-erased core: calls `visit(name, is_dir, ctx)` once per entry in `path`,
// EXCLUDING "." and "..". `name` is just the base name (not joined with `path`).
// Stops early if `visit` returns false. Returns false if `path` can't be opened.
// Compiled once in fs.cc (no per-callback-type code).
HYLE_PLATFORM_API bool directory_iterate(
    const char* path, bool (*visit)(const char* name, bool is_dir, void* ctx),
    void* ctx) noexcept;

// Ergonomic wrapper: pass any callable `fn(const char* name, bool is_dir)->bool`
// (return false to stop early) or `->void`. Inline, so no allocation and the
// lambda body inlines into the trampoline.
template <class Fn>
bool for_each_entry(const char* path, Fn&& fn) {
  auto trampoline = [](const char* name, bool is_dir, void* ctx) -> bool {
    Fn& f = *static_cast<Fn*>(ctx);
    if constexpr (requires { { f(name, is_dir) } -> std::same_as<bool>; }) {
      return f(name, is_dir);
    } else {
      f(name, is_dir);
      return true;
    }
  };
  return directory_iterate(path, trampoline, &fn);
}

}  // namespace hyle::platform

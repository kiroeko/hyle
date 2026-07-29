#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>

#include "hyle/error/result.h"
#include "hyle/macros.h"
#include "hyle/time/unix_timestamp.h"
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

// ---- FileHandle --------------------------------------------------------------
// Zero-cost owning wrapper over a raw OS file descriptor / HANDLE (folly::File /
// seastar file_desc lineage). Move-only, closes on destruction. No
// vtable, no heap, no refcount -- one word; the compiler lowers it to a bare fd.
// The raw value is stored as intptr_t: a POSIX int fd or a Windows HANDLE both
// fit. kInvalid (-1) is the empty state (matches POSIX -1 and is never a valid
// Windows HANDLE returned by CreateFile, which yields INVALID_HANDLE_VALUE == -1).
class FileHandle {
 public:
  static constexpr std::intptr_t kInvalid = -1;

  FileHandle() noexcept = default;
  explicit FileHandle(std::intptr_t raw) noexcept : raw_(raw) {}
  HYLE_PLATFORM_API ~FileHandle();
  FileHandle(FileHandle&& o) noexcept : raw_(o.raw_) { o.raw_ = kInvalid; }
  HYLE_PLATFORM_API FileHandle& operator=(FileHandle&& o) noexcept;
  HYLE_DISABLE_COPY(FileHandle);

  bool valid() const noexcept { return raw_ != kInvalid; }
  explicit operator bool() const noexcept { return valid(); }

  // Borrow the raw handle WITHOUT transferring ownership (still closed on ~).
  std::intptr_t get() const noexcept { return raw_; }
  // Transfer ownership out: returns the raw handle and leaves *this empty (the
  // caller is now responsible for closing it).
  std::intptr_t release() noexcept {
    std::intptr_t r = raw_;
    raw_ = kInvalid;
    return r;
  }
  // Close early (also done by the destructor). Safe on an empty handle. Unlike
  // the destructor, this reports close/flush failures that may matter after a
  // write. The handle is empty after the call even when close reports failure.
  HYLE_PLATFORM_API Status close() noexcept;

  // Read up to `n` bytes; 0 is EOF and a short read is normal. EINTR is retried.
  HYLE_PLATFORM_API Result<std::size_t> read(void* buf,
                                             std::size_t n) noexcept;
  // Write up to `n` bytes; a short write is a successful result the caller may
  // continue. EINTR is retried. Neither operation uses a TLS error side channel.
  HYLE_PLATFORM_API Result<std::size_t> write(const void* buf,
                                              std::size_t n) noexcept;

 private:
  std::intptr_t raw_ = kInvalid;
};

// ---- query (stat) -----------------------------------------------------------
// Portable file kinds. Directory iteration reports the entry itself (so a
// symlink is kSymlink); stat() follows a symlink and reports its target kind.
// kUnknown is a real fallback for platform/filesystem kinds without a portable
// representation, not a missing-path sentinel.
enum class FileType : std::uint8_t {
  kUnknown = 0,
  kRegular = 1,
  kDirectory = 2,
  kSymlink = 3,
  kBlockDevice = 4,
  kCharacterDevice = 5,
  kFifo = 6,
  kSocket = 7,
};

// Result of stat(): a snapshot of one existing path. Absence and OS failures
// are carried by Result, so redundant `exists` state cannot disagree with it.
struct FileInfo {
  FileType type = FileType::kUnknown;  // target kind (stat follows symlinks)
  bool is_symlink = false;           // path itself is a symbolic link (lstat semantics)
  std::uint64_t size = 0;            // bytes (regular files); 0 for directories
  time::UnixTimestamp modified_time{};  // signed Unix mtime, nanosecond precision

  constexpr bool is_regular() const noexcept {
    return type == FileType::kRegular;
  }
  constexpr bool is_directory() const noexcept {
    return type == FileType::kDirectory;
  }
};

// Stat `path`, following symlinks for target type/size while retaining whether
// the path itself is a symlink. Missing paths return NotFound.
HYLE_PLATFORM_API Result<FileInfo> stat(const char* path) noexcept;

// Missing is the ordinary value false; permission/I/O failures remain errors.
HYLE_PLATFORM_API Result<bool> exists(const char* path) noexcept;
HYLE_PLATFORM_API Result<bool> is_directory(const char* path) noexcept;
// Size in bytes. Missing paths return NotFound; directories return
// FailedPrecondition rather than an ambiguous 0.
HYLE_PLATFORM_API Result<std::uint64_t> file_size(const char* path) noexcept;

// ---- directories ------------------------------------------------------------
// Create one directory. Already existing as a directory is OK (idempotent);
// already existing as another kind is AlreadyExists.
HYLE_PLATFORM_API Status make_directory(const char* path) noexcept;

// Create `path` and every missing parent ("mkdir -p"). `/` and `\\` both split
// components. OK means the full path exists as a directory afterward.
HYLE_PLATFORM_API Status make_directory_recursive(const char* path) noexcept;

// Remove an empty directory.
HYLE_PLATFORM_API Status remove_directory(const char* path) noexcept;

// Remove a directory and ALL its contents ("rm -rf").
// Does not follow symlinks/junctions in a caller-owned tree. Like other
// path-based recursive APIs, it cannot make traversal atomic against a process
// concurrently replacing entries; the caller must prevent mutation while it
// runs. (Hard-to-reverse: deletes recursively. Caller owns the blast radius.)
HYLE_PLATFORM_API Status remove_directory_recursive(const char* path) noexcept;

// ---- files ------------------------------------------------------------------
// Remove a single file.
HYLE_PLATFORM_API Status remove_file(const char* path) noexcept;

// Rename/move `from` to `to`.
HYLE_PLATFORM_API Status rename_path(const char* from, const char* to) noexcept;

// Copy a single file `from` -> `to`, overwriting `to`.
HYLE_PLATFORM_API Status copy_file(const char* from, const char* to) noexcept;

// Create `path` empty if absent, or bump its mtime if present (unix `touch`).
HYLE_PLATFORM_API Status touch(const char* path) noexcept;

// ---- permissions ------------------------------------------------------------
// POSIX permission bits (the low 9 of st_mode). Windows has no POSIX permission
// model -- only a read-only attribute -- so on Windows get_permissions() SYNTHESIZES
// a mode (0444 read-only, else 0666) and set_permissions() honors only the write
// bits (any owner/group/other write bit clear => set READONLY). Use the portable
// is_readonly()/set_readonly() subset when you only care about writability.
enum Perms : std::uint32_t {
  kPermOwnerRead = 0400, kPermOwnerWrite = 0200, kPermOwnerExec = 0100,
  kPermGroupRead = 0040, kPermGroupWrite = 0020, kPermGroupExec = 0010,
  kPermOtherRead = 0004, kPermOtherWrite = 0002, kPermOtherExec = 0001,
  kPermAll = 0777,
};

// Permission bits of `path` (low 9). A legitimate mode 0000 remains distinct
// from NotFound/PermissionDenied. Windows returns a synthesized mode (see above).
HYLE_PLATFORM_API Result<std::uint32_t> get_permissions(
    const char* path) noexcept;
// Set permission bits. POSIX: chmod(mode & 0777). Windows: only the write bits
// are honored (mapped to the READONLY attribute).
HYLE_PLATFORM_API Status set_permissions(const char* path,
                                         std::uint32_t mode) noexcept;
// Portable subset: is the file present and not writable by the owner?
HYLE_PLATFORM_API Result<bool> is_readonly(const char* path) noexcept;
// Portable subset: toggle owner write (POSIX: chmod +/-0200; Windows: READONLY).
HYLE_PLATFORM_API Status set_readonly(const char* path,
                                      bool readonly) noexcept;

// ---- symbolic links ---------------------------------------------------------
// Windows must encode whether a symlink targets a file or directory. kAuto is
// convenient for an existing target; pass an explicit kind for a dangling link.
// POSIX validates but otherwise ignores the kind because symlink(2) is untyped.
enum class SymlinkKind : std::uint8_t {
  kAuto = 0,
  kFile = 1,
  kDirectory = 2,
};

// Create a symlink at `link` pointing to `target`. Windows requires Developer
// Mode or admin (SeCreateSymboliclink privilege); without it returns
// PermissionDenied.
HYLE_PLATFORM_API Status create_symlink(const char* target,
                                        const char* link,
                                        SymlinkKind kind =
                                            SymlinkKind::kAuto) noexcept;
// Create a HARD link at `link` to `target` (another name for the same file
// data). POSIX link(2) / Windows CreateHardLinkW. Same-volume, regular files
// only (no directory hard links); no special privilege.
HYLE_PLATFORM_API Status create_hard_link(const char* target,
                                          const char* link) noexcept;
// True if `path` itself is a symbolic link (does not follow it).
HYLE_PLATFORM_API Result<bool> is_symlink(const char* path) noexcept;
// Read a symlink's target into dst[cap] (buffer-fill; returns the full length;
// >= cap means truncated). A non-symlink is FAILED_PRECONDITION.
HYLE_PLATFORM_API Result<std::size_t> read_symlink(
    const char* path, char* dst, std::size_t cap) noexcept;

// ---- open / read / write ----------------------------------------------------
// Flags for open_file (OR them). kRead/kWrite pick access; kCreate makes a
// missing file; kTruncate empties an existing one; kAppend puts writes at EOF.
enum OpenFlags {
  kOpenRead = 1u << 0,
  kOpenWrite = 1u << 1,
  kOpenCreate = 1u << 2,    // create if absent (needs kOpenWrite)
  kOpenTruncate = 1u << 3,  // truncate to empty on open (needs kOpenWrite)
  kOpenAppend = 1u << 4,    // writes go to end-of-file
};

// Open `path` with `flags` (OpenFlags OR'd). Wide APIs on Windows (UTF-8 path,
// no ANSI mis-decode). Exactly one or both of kOpenRead/kOpenWrite must be set.
// Failures preserve native operation/code/message. The general file-I/O
// primitive the module was missing (read a config without mmap, etc).
HYLE_PLATFORM_API Result<FileHandle> open_file(const char* path,
                                               unsigned flags) noexcept;

// ---- temporary files --------------------------------------------------------
// Atomically create a uniquely-named file under `dir` (NULL => temp_directory())
// with the given `prefix`, and return an OPEN read-write handle to it (POSIX
// mkstemp / Windows GetTempFileName+CreateFile). Returning the open handle -- not
// just a path -- is the TOCTOU-safe idiom (SEI CERT FIO21-C, Rust NamedTempFile):
// nobody can swap the file between creation and use. The actual path is written
// to dst[cap] (buffer-fill). A returned handle always owns the created file.
// The file is NOT auto-deleted: the caller unlinks it (or keeps it) via the path.
struct CreatedTempFile {
  FileHandle file;
  std::size_t path_size = 0;  // full UTF-8 length, excluding NUL
};

HYLE_PLATFORM_API Result<CreatedTempFile> create_temp_file(
    const char* dir, const char* prefix, char* dst, std::size_t cap) noexcept;

// ---- working / special directories (buffer-fill) ----------------------------
// Current working directory into dst[cap]. The Result value is the full length
// (>= cap => truncated); OS/encoding/allocation failures remain distinguishable.
HYLE_PLATFORM_API Result<std::size_t> current_directory(
    char* dst, std::size_t cap) noexcept;

// Change the process working directory.
HYLE_PLATFORM_API Status change_directory(const char* path) noexcept;

// System temp directory (POSIX: $TMPDIR/$TMP/$TEMP then /tmp; Windows:
// GetTempPathW) into dst[cap]. This is path discovery, not a writability probe;
// use create_temp_file when a usable temp location is required.
HYLE_PLATFORM_API Result<std::size_t> temp_directory(
    char* dst, std::size_t cap) noexcept;

// User home directory (Win: %USERPROFILE%; POSIX: $HOME, else the passwd entry)
// into dst[cap]. The Result value is the full length. clay's `app_folder`.
HYLE_PLATFORM_API Result<std::size_t> home_directory(
    char* dst, std::size_t cap) noexcept;

// ---- environment ------------------------------------------------------------
// Read environment variable `name` into dst[cap] (buffer-fill). The Result value
// is the full value length; unset is NotFound (distinct from set-but-empty, whose
// value is 0). dst is always NUL-terminated when cap>0 on success.
HYLE_PLATFORM_API Result<std::size_t> get_env_var(
    const char* name, char* dst, std::size_t cap) noexcept;

// Set / unset an environment variable in this process. POSIX setenv/unsetenv,
// Windows _wputenv_s. NOT thread-safe against concurrent
// get_env_var/getenv in other threads (same caveat as the C library) -- set env
// during startup, before spawning threads that read it.
// PLATFORM DIFFERENCE: set_env_var(name, "") keeps a zero-length value on POSIX
// (distinct from unset) but DELETES the variable on Windows (_wputenv_s treats an
// empty value as removal -- Windows has no "present but empty" env var).
HYLE_PLATFORM_API Status set_env_var(const char* name,
                                     const char* value) noexcept;
HYLE_PLATFORM_API Status unset_env_var(const char* name) noexcept;

// ---- directory iteration (visitor, no allocation) ---------------------------
// Type-erased core: calls `visit(name, type, ctx)` once per entry in `path`,
// EXCLUDING "." and "..". `name` is just the base name (not joined with `path`).
// `type` describes the directory entry itself and therefore does not follow
// symlinks. It comes from dirent/FindFirst metadata without a per-entry stat in
// the common case; kUnknown is possible on filesystems with no portable kind.
// Stops early if `visit` returns false; that is a successful Result(false), not
// an enumeration failure.
// Compiled once in fs.cc (no per-callback-type code).
// Result value is true when every entry was visited, false when the visitor
// stopped normally. Opening/enumerating failures are Status errors.
HYLE_PLATFORM_API Result<bool> directory_iterate(
    const char* path, bool (*visit)(const char* name, FileType type, void* ctx),
    void* ctx) noexcept;

// Ergonomic wrapper: pass any callable `fn(const char* name, FileType)->bool`
// (return false to stop early) or `->void`. Inline, so no allocation and the
// lambda body inlines into the trampoline.
template <class Fn>
Result<bool> for_each_entry(const char* path, Fn&& fn) {
  auto trampoline = [](const char* name, FileType type, void* ctx) -> bool {
    Fn& f = *static_cast<Fn*>(ctx);
    if constexpr (requires { { f(name, type) } -> std::same_as<bool>; }) {
      return f(name, type);
    } else {
      f(name, type);
      return true;
    }
  };
  return directory_iterate(path, trampoline, &fn);
}

}  // namespace hyle::platform

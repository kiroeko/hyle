#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <span>
#include <type_traits>

#include "hyle/error/result.h"
#include "hyle/core/compiler.h"
#include "hyle/platform/fs_visibility.h"
#include "hyle/time/unix_timestamp.h"

// LAYERING: every path-returning call is BUFFER-FILL (char* dst, cap) and
// directory iteration is a VISITOR -- so this header pulls in neither mem nor
// container, and filesystem mechanisms stay in platform. Pure lexical path
// algorithms belong to container/string rather than this OS boundary.
// Buffer-fill calls follow snprintf semantics: they always NUL-terminate when
// cap>0 and return the FULL length that would be written (a return >= cap means
// the path was truncated).

namespace hyle::platform {

enum class SeekOrigin : std::uint8_t {
  kBegin = 0,
  kCurrent = 1,
  kEnd = 2,
};

// Whole-file coordination between cooperating handles/processes. POSIX uses
// advisory flock locks; Windows uses the corresponding full-range LockFileEx
// lock, whose conflicting I/O enforcement is stronger. This is not an
// in-process mutex or a distributed-filesystem consensus primitive.
enum class FileLockMode : std::uint8_t {
  kShared = 0,
  kExclusive = 1,
};

// ---- FileHandle
// -------------------------------------------------------------- Zero-cost
// owning wrapper over a raw OS file descriptor / HANDLE (folly::File / seastar
// file_desc lineage). Move-only, closes on destruction. No vtable, no heap, no
// refcount -- one word; the compiler lowers it to a bare fd. The raw value is
// stored as intptr_t: a POSIX int fd or a Windows HANDLE both fit. kInvalid
// (-1) is the empty state (matches POSIX -1 and is never a valid Windows HANDLE
// returned by CreateFile, which yields INVALID_HANDLE_VALUE == -1).
class FileHandle {
public:
  static constexpr std::intptr_t kInvalid = -1;

  FileHandle() noexcept = default;
  explicit FileHandle(std::intptr_t raw) noexcept : raw_(raw) {}
  HYLE_PLATFORM_FS_API ~FileHandle();
  FileHandle(FileHandle &&o) noexcept : raw_(o.raw_) { o.raw_ = kInvalid; }
  HYLE_PLATFORM_FS_API FileHandle &operator=(FileHandle &&o) noexcept;
  FileHandle(const FileHandle&) = delete;
  FileHandle& operator=(const FileHandle&) = delete;

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
  // the destructor, this reports close failures. Call sync() before close when
  // durability matters. The handle is empty even when close reports failure.
  HYLE_PLATFORM_FS_API Status close() noexcept;

  // Read up to `n` bytes; 0 is EOF and a short read is normal. EINTR is
  // retried.
  HYLE_PLATFORM_FS_API Result<std::size_t> read(void *buf, std::size_t n) noexcept;
  // Write up to `n` bytes; a short write is a successful result the caller may
  // continue. EINTR is retried. Neither operation uses a TLS error side
  // channel.
  HYLE_PLATFORM_FS_API Result<std::size_t> write(const void *buf,
                                              std::size_t n) noexcept;

  Result<std::size_t> read(std::span<std::byte> buffer) noexcept {
    return read(buffer.data(), buffer.size());
  }
  Result<std::size_t> write(std::span<const std::byte> buffer) noexcept {
    return write(buffer.data(), buffer.size());
  }

  // Complete blocking transfers for ordinary regular files. read_exact()
  // reports DataLoss if EOF arrives before all `n` bytes; write_all() reports
  // DataLoss if the OS makes a successful zero-byte write. Both preserve the
  // first native failure and allocate nothing.
  HYLE_PLATFORM_FS_API Status read_exact(void *buf, std::size_t n) noexcept;
  HYLE_PLATFORM_FS_API Status write_all(const void *buf,
                                     std::size_t n) noexcept;
  Status read_exact(std::span<std::byte> buffer) noexcept {
    return read_exact(buffer.data(), buffer.size());
  }
  Status write_all(std::span<const std::byte> buffer) noexcept {
    return write_all(buffer.data(), buffer.size());
  }

  // Move the shared file cursor and return its resulting absolute byte
  // position. A FileHandle is intentionally a single-cursor owner; concurrent
  // positional I/O belongs to io::File (or separate handles), not a hidden lock
  // or OVERLAPPED state in this one-word type.
  HYLE_PLATFORM_FS_API Result<std::uint64_t>
  seek(std::int64_t offset, SeekOrigin origin = SeekOrigin::kBegin) noexcept;
  Result<std::uint64_t> position() noexcept {
    return seek(0, SeekOrigin::kCurrent);
  }

  // Request that buffered file data reach the OS storage boundary. `data_only`
  // may omit metadata where the platform has fdatasync; Windows and macOS use
  // their ordinary full-file flush. This is explicit because close alone is
  // not a durability request.
  HYLE_PLATFORM_FS_API Status sync(bool data_only = false) noexcept;

  // Query or change the logical end-of-file through this already-open owner,
  // avoiding a second path lookup. resize() may grow a sparse file and does not
  // move the shared cursor.
  HYLE_PLATFORM_FS_API Result<std::uint64_t> size() const noexcept;
  HYLE_PLATFORM_FS_API Status resize(std::uint64_t size) noexcept;

  // When the file is smaller than minimum_size, request backing storage before
  // growing its logical size. A file already at least this large is unchanged;
  // this does not de-sparsify it or promise a contiguous extent. Allocation and
  // EOF update are not an atomic transaction on every OS.
  HYLE_PLATFORM_FS_API Status
  preallocate(std::uint64_t minimum_size) noexcept;

  // Blocking/nonblocking whole-file coordination. try_lock() returns false
  // only for lock contention; OS/permission failures remain Status errors.
  // close()/destruction releases locks. Converting an already-held mode is not
  // promised atomic, and callers must externally synchronize one FileHandle.
  HYLE_PLATFORM_FS_API Status
  lock(FileLockMode mode = FileLockMode::kExclusive) noexcept;
  HYLE_PLATFORM_FS_API Result<bool>
  try_lock(FileLockMode mode = FileLockMode::kExclusive) noexcept;
  HYLE_PLATFORM_FS_API Status unlock() noexcept;

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
  FileType type = FileType::kUnknown; // target kind (stat follows symlinks)
  bool is_symlink = false; // path itself is a symbolic link (lstat semantics)
  std::uint64_t size = 0;  // bytes (regular files); 0 for directories
  time::UnixTimestamp
      modified_time{}; // signed Unix mtime, nanosecond precision

  constexpr bool is_regular() const noexcept {
    return type == FileType::kRegular;
  }
  constexpr bool is_directory() const noexcept {
    return type == FileType::kDirectory;
  }
};

// Stat `path`, following symlinks for target type/size while retaining whether
// the path itself is a symlink. Missing paths return NotFound.
HYLE_PLATFORM_FS_API Result<FileInfo> stat(const char *path) noexcept;

// Missing is the ordinary value false; permission/I/O failures remain errors.
HYLE_PLATFORM_FS_API Result<bool> exists(const char *path) noexcept;
HYLE_PLATFORM_FS_API Result<bool> is_directory(const char *path) noexcept;
// Size in bytes. Missing paths return NotFound; directories return
// FailedPrecondition rather than an ambiguous 0.
HYLE_PLATFORM_FS_API Result<std::uint64_t> file_size(const char *path) noexcept;

// ---- directories ------------------------------------------------------------
// Create one directory. Already existing as a directory is OK (idempotent);
// already existing as another kind is AlreadyExists.
HYLE_PLATFORM_FS_API Status make_directory(const char *path) noexcept;

// Create `path` and every missing parent ("mkdir -p"). `/` and `\\` both split
// components. OK means the full path exists as a directory afterward.
HYLE_PLATFORM_FS_API Status make_directory_recursive(const char *path) noexcept;

// Remove an empty directory.
HYLE_PLATFORM_FS_API Status remove_directory(const char *path) noexcept;

// Remove a directory and ALL its contents ("rm -rf").
// Does not follow symlinks/junctions in a caller-owned tree. Like other
// path-based recursive APIs, it cannot make traversal atomic against a process
// concurrently replacing entries; the caller must prevent mutation while it
// runs. Traversal uses a resumable native iterator frame per directory depth
// and one mutable path buffer, rather than the process call stack or a
// width-sized child queue. Auxiliary memory is O(depth + longest path);
// frame/path allocation failure or an OS per-process iterator-handle limit is
// ResourceExhausted. (Hard-to-reverse: deletes recursively. Caller owns the
// blast radius.)
HYLE_PLATFORM_FS_API Status remove_directory_recursive(const char *path) noexcept;

// ---- files ------------------------------------------------------------------
// Remove a single file.
HYLE_PLATFORM_FS_API Status remove_file(const char *path) noexcept;

// Rename/move `from` to `to`.
HYLE_PLATFORM_FS_API Status rename_path(const char *from, const char *to) noexcept;

// Copy one filesystem file `from` -> `to`, overwriting `to`. POSIX succeeds
// only when the opened objects are regular files (a symlink resolving to one is
// accepted); a successfully opened non-regular object is FailedPrecondition,
// while an earlier open failure preserves its native error (for example ENXIO
// for a FIFO destination with no reader). Virtual regular files are streamed
// to EOF rather than trusting a possibly-zero reported size. Windows delegates
// reparse-point follow/replace behavior to CopyFileW and preserves its native
// failure.
HYLE_PLATFORM_FS_API Status copy_file(const char *from, const char *to) noexcept;

// Create `path` empty if absent, or bump its mtime if present (unix `touch`).
HYLE_PLATFORM_FS_API Status touch(const char *path) noexcept;

// ---- permissions ------------------------------------------------------------
// POSIX permission bits (the low 9 of st_mode). Windows has no POSIX permission
// model -- only a read-only attribute -- so on Windows get_permissions()
// SYNTHESIZES a mode (0444 read-only, else 0666) and set_permissions() honors
// only the write bits (no owner/group/other write bits set => set READONLY).
// Use the portable is_readonly()/set_readonly() subset when you only care about
// writability.
enum Perms : std::uint32_t {
  kPermOwnerRead = 0400,
  kPermOwnerWrite = 0200,
  kPermOwnerExec = 0100,
  kPermGroupRead = 0040,
  kPermGroupWrite = 0020,
  kPermGroupExec = 0010,
  kPermOtherRead = 0004,
  kPermOtherWrite = 0002,
  kPermOtherExec = 0001,
  kPermAll = 0777,
};

// Permission bits of `path` (low 9). A legitimate mode 0000 remains distinct
// from NotFound/PermissionDenied. Windows returns a synthesized mode (see
// above).
HYLE_PLATFORM_FS_API Result<std::uint32_t>
get_permissions(const char *path) noexcept;
// Set permission bits. POSIX: chmod(mode & 0777). Windows: only the write bits
// are honored (mapped to the READONLY attribute).
HYLE_PLATFORM_FS_API Status set_permissions(const char *path,
                                         std::uint32_t mode) noexcept;
// Portable subset: is the file present and not writable by the owner?
HYLE_PLATFORM_FS_API Result<bool> is_readonly(const char *path) noexcept;
// Portable subset: toggle owner write (POSIX: chmod +/-0200; Windows:
// READONLY).
HYLE_PLATFORM_FS_API Status set_readonly(const char *path, bool readonly) noexcept;

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
HYLE_PLATFORM_FS_API Status
create_symlink(const char *target, const char *link,
               SymlinkKind kind = SymlinkKind::kAuto) noexcept;
// Create a HARD link at `link` to `target` (another name for the same file
// data). POSIX link(2) / Windows CreateHardLinkW. Same-volume, regular files
// only (no directory hard links); no special privilege.
HYLE_PLATFORM_FS_API Status create_hard_link(const char *target,
                                          const char *link) noexcept;
// True if `path` itself is a symbolic link (does not follow it).
HYLE_PLATFORM_FS_API Result<bool> is_symlink(const char *path) noexcept;
// Read a symlink's target into dst[cap] (buffer-fill; returns the full length;
// >= cap means truncated). The implementation grows its syscall scratch when a
// POSIX readlink result fills the whole buffer, so it never reports a possibly
// truncated target as complete. A non-symlink is FAILED_PRECONDITION.
HYLE_PLATFORM_FS_API Result<std::size_t> read_symlink(const char *path, char *dst,
                                                   std::size_t cap) noexcept;

// ---- open / read / write ----------------------------------------------------
// Flags for open_file (OR them). kRead/kWrite pick access; kCreate makes a
// missing file; kTruncate empties an existing one; kAppend puts writes at EOF.
enum OpenFlags {
  kOpenRead = 1u << 0,
  kOpenWrite = 1u << 1,
  kOpenCreate = 1u << 2,   // create if absent (needs kOpenWrite)
  kOpenTruncate = 1u << 3, // truncate to empty on open (needs kOpenWrite)
  kOpenAppend = 1u << 4,   // writes go to end-of-file
};

// Open `path` with `flags` (OpenFlags OR'd). Wide APIs on Windows (UTF-8 path,
// no ANSI mis-decode). Exactly one or both of kOpenRead/kOpenWrite must be set.
// Failures preserve native operation/code/message. The general file-I/O
// primitive the module was missing (read a config without mmap, etc).
HYLE_PLATFORM_FS_API Result<FileHandle> open_file(const char *path,
                                               unsigned flags) noexcept;

// ---- temporary files --------------------------------------------------------
// Atomically create a uniquely-named file under `dir` (NULL =>
// temp_directory()) with the given `prefix`, and return an OPEN read-write
// handle to it (POSIX mkostemp/mkstemp / Windows CreateFile(CREATE_NEW)).
// Returning the open handle
// -- not just a path -- is the TOCTOU-safe idiom (SEI CERT FIO21-C, Rust
// NamedTempFile):
// nobody can swap the file between creation and use. The actual path is written
// to dst[cap] (buffer-fill). The prefix is preserved in the generated basename;
// it must be one basename fragment (no path separators). Invalid Windows UTF-8,
// filename metacharacters, or an overlong prefix are errors, never silently
// substituted. A non-NULL `dir` must not be empty. Path construction uses
// checked dynamic scratch rather than imposing MAX_PATH/1024-byte library
// ceilings; native filesystem component/path limits still apply. A returned
// handle always owns the created file and is
// non-inheritable across process execution (FD_CLOEXEC on POSIX). The file is
// NOT auto-deleted: the caller unlinks it (or keeps it) via the path.
struct CreatedTempFile {
  FileHandle file;
  std::size_t path_size = 0; // full UTF-8 length, excluding NUL
};

HYLE_PLATFORM_FS_API Result<CreatedTempFile>
create_temp_file(const char *dir, const char *prefix, char *dst,
                 std::size_t cap) noexcept;

// ---- working / special directories (buffer-fill) ----------------------------
// Current working directory into dst[cap]. The Result value is the full length
// (>= cap => truncated); OS/encoding/allocation failures remain
// distinguishable.
HYLE_PLATFORM_FS_API Result<std::size_t>
current_directory(char *dst, std::size_t cap) noexcept;

// Change the process working directory.
HYLE_PLATFORM_FS_API Status change_directory(const char *path) noexcept;

// System temp directory (POSIX: $TMPDIR/$TMP/$TEMP then /tmp; Windows:
// GetTempPathW) into dst[cap]. This is path discovery, not a writability probe;
// use create_temp_file when a usable temp location is required.
HYLE_PLATFORM_FS_API Result<std::size_t> temp_directory(char *dst,
                                                     std::size_t cap) noexcept;

// User home directory (Win: %USERPROFILE%; POSIX: $HOME, else the passwd entry)
// into dst[cap]. The Result value is the full length.
HYLE_PLATFORM_FS_API Result<std::size_t> home_directory(char *dst,
                                                     std::size_t cap) noexcept;

// ---- directory iteration (visitor, no allocation) ---------------------------
// Type-erased core: calls `visit(name, type, ctx)` once per entry in `path`,
// EXCLUDING "." and "..". `name` is just the base name (not joined with
// `path`). `type` describes the directory entry itself and therefore does not
// follow symlinks. It comes from dirent/FindFirst metadata without a per-entry
// stat in the common case; kUnknown is possible on filesystems with no portable
// kind. Stops early if `visit` returns false; that is a successful
// Result(false), not an enumeration failure. Compiled once in fs.cc (no
// per-callback-type code). Result value is true when every entry was visited,
// false when the visitor stopped normally. Opening/enumerating failures are
// Status errors. A callback passed directly to this allocation-free core must
// not throw because the compiled iterator is a noexcept resource-cleanup
// boundary. Ordinary C++ callers should use for_each_entry(), which preserves
// callback exceptions after the native iterator has been closed.
HYLE_PLATFORM_FS_API Result<bool>
directory_iterate(const char *path,
                  bool (*visit)(const char *name, FileType type, void *ctx),
                  void *ctx) noexcept;

// Ergonomic wrapper: pass any callable `fn(const char* name, FileType)->bool`
// (return false to stop early) or `->void`. Inline, so no allocation and the
// lambda body inlines into the trampoline.
template <class Fn> Result<bool> for_each_entry(const char *path, Fn &&fn) {
  using Callable = std::remove_reference_t<Fn>;
  struct Context {
    Callable *callable;
#if HYLE_HAS_EXCEPTIONS
    std::exception_ptr exception;
#endif
  };
  Context context{std::addressof(fn)
#if HYLE_HAS_EXCEPTIONS
                  , {}
#endif
  };
  auto trampoline = [](const char *name, FileType type,
                       void *opaque) noexcept -> bool {
    auto &context = *static_cast<Context *>(opaque);
#if HYLE_HAS_EXCEPTIONS
    try {
#endif
    Callable &f = *context.callable;
    if constexpr (requires {
                    { f(name, type) } -> std::same_as<bool>;
                  }) {
      return f(name, type);
    } else {
      f(name, type);
      return true;
    }
#if HYLE_HAS_EXCEPTIONS
    } catch (...) {
      context.exception = std::current_exception();
      return false;
    }
#endif
  };
  auto result = directory_iterate(path, trampoline, &context);
#if HYLE_HAS_EXCEPTIONS
  if (context.exception) std::rethrow_exception(context.exception);
#endif
  return result;
}

} // namespace hyle::platform

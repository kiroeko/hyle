// _GNU_SOURCE exposes copy_file_range (glibc 2.27+) for the Linux fast path;
// must precede any system header. We build with -std=c++23 (extensions OFF), so
// it isn't defined for us. Harmless on non-glibc / Windows.
#if defined(__linux__)
#  define _GNU_SOURCE 1
#endif

// std::fopen in touch() is flagged C4996 by MSVC; the "ab" use here is safe and
// portable, so opt out of the CRT deprecation nag (no global override exists).
#if defined(_WIN32)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

#include "hyle/platform/fs.h"

#include "fs_internal.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "hyle/core/environment.h"  // HYLE_OS_* for the branches below (the _GNU_SOURCE/_CRT
                            // feature macros above MUST stay raw -- they precede ALL
                            // includes, before platform.h is parsed).
#include "hyle/platform/environment.h"
#include "hyle/platform/random.h"

#if HYLE_OS_WINDOWS
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <winioctl.h>  // FSCTL_GET_REPARSE_POINT for read_symlink
#  include <cwchar>  // std::wcslen for the directory_iterate pattern
#else
#  include <dirent.h>
#  include <fcntl.h>
#  include <pwd.h>
#  include <sys/file.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace hyle::platform {
namespace {

// Buffer-fill with snprintf semantics: copy [src, src+len) into dst[cap], always
// NUL-terminating when cap>0, and return `len` (the full length, so a caller sees
// truncation when the return is >= cap).
std::size_t fill(char* dst, std::size_t cap, const char* src,
                 std::size_t len) noexcept {
  if (dst == nullptr) cap = 0;
  if (cap == 0) return len;
  std::size_t n = len < cap - 1 ? len : cap - 1;
  if (n) std::memcpy(dst, src, n);
  dst[n] = '\0';
  return len;
}

void clear_output(char* dst, std::size_t cap) noexcept {
  if (dst != nullptr && cap != 0) dst[0] = '\0';
}

// `create_temp_file(dir, prefix, ...)` promises that the result stays directly
// under dir. Treat prefix as one basename fragment, never as a path supplied by
// the caller. Windows additionally rejects filename metacharacters here so a
// colon cannot turn a would-be temp file into an alternate data stream.
bool valid_temp_prefix(const char* prefix) noexcept {
  if (prefix == nullptr) return true;
#if HYLE_OS_WINDOWS
  return std::strpbrk(prefix, "<>:\"/\\|?*") == nullptr;
#else
  // Reject both separators for cross-platform behavior even though backslash
  // is an ordinary POSIX filename byte.
  return std::strchr(prefix, '/') == nullptr &&
         std::strchr(prefix, '\\') == nullptr;
#endif
}

bool checked_size_add(std::size_t left, std::size_t right,
                      std::size_t& result) noexcept {
  if (right > (std::numeric_limits<std::size_t>::max)() - left) return false;
  result = left + right;
  return true;
}

bool checked_offset_add(std::uint64_t offset, std::size_t delta,
                        std::uint64_t& result) noexcept {
  if (delta > (std::numeric_limits<std::uint64_t>::max)() - offset) {
    return false;
  }
  result = offset + static_cast<std::uint64_t>(delta);
  return true;
}

template <class T>
class MallocArray {
 public:
  MallocArray() noexcept = default;
  ~MallocArray() { std::free(data_); }
  MallocArray(const MallocArray&) = delete;
  MallocArray& operator=(const MallocArray&) = delete;

  bool resize(std::size_t count) noexcept {
    overflowed_ = count > (std::numeric_limits<std::size_t>::max)() / sizeof(T);
    if (overflowed_) return false;
    void* replacement = std::realloc(data_, count * sizeof(T));
    if (replacement == nullptr && count != 0) return false;
    data_ = static_cast<T*>(replacement);
    return true;
  }

  T* data() noexcept { return data_; }
  const T* data() const noexcept { return data_; }
  bool overflowed() const noexcept { return overflowed_; }

 private:
  T* data_ = nullptr;
  bool overflowed_ = false;
};

// Resolve an explicit directory or copy the process temp directory without a
// fixed path ceiling. The second query is checked and retried because process
// environment/configuration can change between count and fill.
class TempDirectoryInput {
 public:
  Status resolve(const char* explicit_directory) noexcept {
    if (explicit_directory != nullptr) {
      if (explicit_directory[0] == '\0') {
        return Status::InvalidArgument(
            "create temp file: explicit directory is empty");
      }
      path_ = explicit_directory;
      return Status::Ok();
    }

    char* output = stack_;
    std::size_t capacity = sizeof(stack_);
    for (;;) {
      auto length = temp_directory(output, capacity);
      if (!length) return std::move(length).error();
      if (*length == 0) {
        return Status::FailedPrecondition(
            "create temp file: temp directory is empty");
      }
      if (*length < capacity) {
        path_ = output;
        return Status::Ok();
      }
      std::size_t required;
      if (!checked_size_add(*length, 1, required)) {
        return Status::OutOfRange(
            "create temp file: temp directory is too long");
      }
      if (!heap_.resize(required)) {
        return heap_.overflowed()
                   ? Status::OutOfRange(
                         "create temp file: temp directory is too long")
                   : Status::ResourceExhausted(
                         "create temp file: temp directory allocation failed");
      }
      output = heap_.data();
      capacity = required;
    }
  }

  const char* path() const noexcept { return path_; }

 private:
  char stack_[512];
  MallocArray<char> heap_;
  const char* path_ = nullptr;
};

// Join "a/b" into dst[cap]. Returns false (no write) if it would not fit. Bounded
// and allocation-free -- used to rebuild child paths during recursion without
// snprintf (whose worst-case length trips GCC -Wformat-truncation under -Werror).
#if !HYLE_OS_WINDOWS
bool join_path(char* dst, std::size_t cap, const char* a,
               const char* b) noexcept {
  std::size_t la = std::strlen(a), lb = std::strlen(b);
  if (la + 1 + lb + 1 > cap) return false;
  std::memcpy(dst, a, la);
  dst[la] = '/';
  std::memcpy(dst + la + 1, b, lb);
  dst[la + 1 + lb] = '\0';
  return true;
}
#endif

// Raw file-handle operations. Defined once per OS and shared by FileHandle.
Status close_raw(std::intptr_t raw) noexcept;
Result<std::intptr_t> duplicate_raw(std::intptr_t raw) noexcept;
Result<FileInfo> info_raw(std::intptr_t raw) noexcept;
// Raw read/write on an fd/HANDLE (per-OS defs below), declared for the
// platform-agnostic FileHandle::read/write tail.
Result<std::size_t> read_raw(std::intptr_t raw, void* buf,
                             std::size_t n) noexcept;
Result<std::size_t> write_raw(std::intptr_t raw, const void* buf,
                              std::size_t n) noexcept;
Result<std::size_t> read_at_raw(std::intptr_t raw, std::uint64_t offset,
                                void* buf, std::size_t n) noexcept;
Result<std::size_t> write_at_raw(std::intptr_t raw, std::uint64_t offset,
                                 const void* buf, std::size_t n) noexcept;
Result<std::uint64_t> seek_raw(std::intptr_t raw, std::int64_t offset,
                               SeekOrigin origin) noexcept;
Status sync_raw(std::intptr_t raw, bool data_only) noexcept;
Result<std::uint64_t> size_raw(std::intptr_t raw) noexcept;
Status resize_raw(std::intptr_t raw, std::uint64_t size) noexcept;
Status preallocate_raw(std::intptr_t raw,
                       std::uint64_t minimum_size) noexcept;
Status lock_raw(std::intptr_t raw, FileLockMode mode) noexcept;
Result<bool> try_lock_raw(std::intptr_t raw, FileLockMode mode) noexcept;
Status unlock_raw(std::intptr_t raw) noexcept;

}  // namespace

#if defined(HYLE_TESTING) && !HYLE_OS_WINDOWS
namespace detail {
namespace {
ReadSymlinkFunctionForTesting g_read_symlink_function_for_testing = nullptr;
}

void set_read_symlink_function_for_testing(
    ReadSymlinkFunctionForTesting function) noexcept {
  g_read_symlink_function_for_testing = function;
}
} // namespace detail
#endif

#if HYLE_OS_WINDOWS

// ---- Windows ----------------------------------------------------------------
//
// All path I/O goes through the wide (...W) API: the ...A variants interpret
// bytes as the active code page, NOT UTF-8, so a UTF-8 path with non-ASCII chars
// would be mis-decoded. We convert UTF-8 <-> UTF-16 at the boundary so callers
// pass UTF-8 everywhere (matching POSIX, which is byte-transparent).

namespace {

Status win32_fs_status(const char* operation, DWORD code) noexcept {
  return status_from_win32(code, operation);
}

// UTF-8 const char* -> UTF-16, with a stack buffer and heap fallback for long
// paths (RAII frees the heap copy). Self-contained in platform: no mem/container
// dependency. On conversion failure `w()` is nullptr (callers treat as error).
//
// Long ABSOLUTE paths (>= kLongThreshold) are rewritten to the `\\?\` extended
// form so the Win32 ...W APIs aren't capped at MAX_PATH (260) -- without the
// prefix even the wide APIs reject longer paths unless a manifest opts in (a DLL
// can't). Short paths are converted verbatim (zero behavior change). `\\?\`
// requires backslash separators and no `.`/`..` (not resolved under it), so the
// body is slash-normalized; callers needing collapse should normalize first.
class WidePath {
 public:
  explicit WidePath(const char* utf8, bool extended_path = true) noexcept {
    if (utf8 == nullptr) return;
    int need = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8, -1,
                                   nullptr, 0);
    if (need <= 0) return;  // includes the NUL
    int wlen = need - 1;

    if (extended_path && wlen >= kLongThreshold) {
      auto is_alpha = [](char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
      };
      bool already = (utf8[0] == '\\' || utf8[0] == '/') &&
                     (utf8[1] == '\\' || utf8[1] == '/') && utf8[2] == '?';
      if (!already) {
        if (is_alpha(utf8[0]) && utf8[1] == ':' &&
            (utf8[2] == '\\' || utf8[2] == '/')) {
          if (build_extended(utf8, need, L"\\\\?\\", 0)) return;  // C:\... -> \\?\C:\...
        } else if ((utf8[0] == '\\' || utf8[0] == '/') &&
                   (utf8[1] == '\\' || utf8[1] == '/')) {
          // UNC \\server\share -> \\?\UNC\server\share (code present; untested,
          // needs a network share -- registered in the coverage map).
          if (build_extended(utf8, need, L"\\\\?\\UNC\\", 2)) return;
        }
        if (allocation_failed_) return;
        std::free(heap_);
        heap_ = nullptr;
      }
      // not absolute (or build failed): fall through to a plain conversion.
    }

    wchar_t* dst = stack_;
    if (static_cast<std::size_t>(need) > kStack) {
      heap_ = static_cast<wchar_t*>(std::malloc(need * sizeof(wchar_t)));
      if (heap_ == nullptr) {
        allocation_failed_ = true;
        return;
      }
      dst = heap_;
    }
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8, -1, dst,
                            need) <= 0) {
      return;
    }
    w_ = dst;
  }
  ~WidePath() { std::free(heap_); }
  WidePath(const WidePath&) = delete;
  WidePath& operator=(const WidePath&) = delete;

  const wchar_t* w() const noexcept { return w_; }
  bool allocation_failed() const noexcept { return allocation_failed_; }

 private:
  // Build prefix + (body minus `skip` leading chars), slash-normalized to '\'.
  bool build_extended(const char* utf8, int need, const wchar_t* prefix,
                      int skip) noexcept {
    int plen = static_cast<int>(std::wcslen(prefix));
    int cap = plen + need;  // prefix + full body + NUL (skip only shrinks it)
    wchar_t* dst = stack_;
    if (static_cast<std::size_t>(cap) > kStack) {
      heap_ = static_cast<wchar_t*>(std::malloc(cap * sizeof(wchar_t)));
      if (heap_ == nullptr) {
        allocation_failed_ = true;
        return false;
      }
      dst = heap_;
    }
    for (int i = 0; i < plen; ++i) dst[i] = prefix[i];
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8, -1,
                            dst + plen, need) <= 0) {
      return false;
    }
    if (skip) {  // drop the leading "\\" of a UNC path (prefix supplies its own)
      std::memmove(dst + plen, dst + plen + skip,
                   (need - skip) * sizeof(wchar_t));
    }
    for (wchar_t* p = dst + plen; *p != L'\0'; ++p)
      if (*p == L'/') *p = L'\\';  // \\?\ form requires backslashes
    w_ = dst;
    return true;
  }

  static constexpr std::size_t kStack = 512;
  // Prefix below 260 (MAX_PATH) with margin for the API's own additions (e.g.
  // FindFirstFile appends "\*"); over-prefixing a <260 path is harmless.
  static constexpr int kLongThreshold = 248;
  wchar_t stack_[kStack];
  wchar_t* heap_ = nullptr;
  const wchar_t* w_ = nullptr;
  bool allocation_failed_ = false;
};

// UTF-16 -> UTF-8 buffer-fill (snprintf semantics): write into dst[cap], always
// NUL-terminate when cap>0, return the FULL UTF-8 length (>= cap => truncated).
// `wlen` is the UTF-16 length excluding any terminator (-1 lets it scan).
Result<std::size_t> fill_w(char* dst, std::size_t cap, const wchar_t* src,
                           int wlen) noexcept {
  if (dst == nullptr) cap = 0;
  int need = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, src, wlen,
                                 nullptr, 0, nullptr, nullptr);
  if (need <= 0) {
    return std::unexpected(
        win32_fs_status("WideCharToMultiByte(size)", ::GetLastError()));
  }
  const std::size_t full =
      static_cast<std::size_t>(need - (wlen == -1 ? 1 : 0));
  if (cap == 0) return full;
  if (full < cap) {
    int got = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, src, wlen, dst,
                                  need, nullptr, nullptr);
    if (got <= 0) {
      dst[0] = '\0';
      return std::unexpected(
          win32_fs_status("WideCharToMultiByte", ::GetLastError()));
    }
    const std::size_t written =
        static_cast<std::size_t>(got - (wlen == -1 ? 1 : 0));
    dst[written] = '\0';
    return full;
  }
  // Won't fit: convert into a heap scratch, then buffer-fill (truncates).
  char* tmp = static_cast<char*>(std::malloc(need + 1));
  if (tmp == nullptr) {
    dst[0] = '\0';
    return std::unexpected(
        Status::ResourceExhausted("UTF-16 to UTF-8 scratch allocation failed"));
  }
  int got = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, src, wlen, tmp,
                                need, nullptr, nullptr);
  if (got <= 0) {
    const DWORD code = ::GetLastError();
    std::free(tmp);
    dst[0] = '\0';
    return std::unexpected(win32_fs_status("WideCharToMultiByte", code));
  }
  const std::size_t converted =
      static_cast<std::size_t>(got - (wlen == -1 ? 1 : 0));
  tmp[converted] = '\0';
  std::size_t r = fill(dst, cap, tmp, converted);
  std::free(tmp);
  return r;
}

}  // namespace

Result<FileInfo> stat(const char* path) noexcept {
  FileInfo r;
  WidePath wp(path);
  if (wp.w() == nullptr) {
    return std::unexpected(
        Status::InvalidArgument("stat: path is null or not valid UTF-8"));
  }
  WIN32_FILE_ATTRIBUTE_DATA d;
  if (!GetFileAttributesExW(wp.w(), GetFileExInfoStandard, &d)) {
    return std::unexpected(
        win32_fs_status("GetFileAttributesExW", ::GetLastError()));
  }
  const bool link = (d.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
  if (link) {
    // GetFileAttributesExW can describe the link itself even when its target is
    // missing. Open without OPEN_REPARSE_POINT so stat's target-following
    // contract is real; keep this extra work off the overwhelmingly common
    // non-link path.
    HANDLE h = CreateFileW(
        wp.w(), FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
        OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
      return std::unexpected(
          win32_fs_status("CreateFileW(stat target)", ::GetLastError()));
    }
    BY_HANDLE_FILE_INFORMATION target;
    if (!GetFileInformationByHandle(h, &target)) {
      const DWORD code = ::GetLastError();
      ::CloseHandle(h);
      return std::unexpected(
          win32_fs_status("GetFileInformationByHandle", code));
    }
    const BOOL closed = ::CloseHandle(h);
    if (!closed) {
      return std::unexpected(
          win32_fs_status("CloseHandle(stat)", ::GetLastError()));
    }
    d.dwFileAttributes = target.dwFileAttributes;
    d.ftLastWriteTime = target.ftLastWriteTime;
    d.nFileSizeHigh = target.nFileSizeHigh;
    d.nFileSizeLow = target.nFileSizeLow;
  }
  bool dir = (d.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
  r.type = dir ? FileType::kDirectory : FileType::kRegular;
  r.is_symlink = link;
  r.size = dir ? 0
               : (static_cast<std::uint64_t>(d.nFileSizeHigh) << 32) |
                     d.nFileSizeLow;
  // FILETIME is 100ns ticks since 1601; convert to seconds since 1970.
  std::uint64_t t = (static_cast<std::uint64_t>(d.ftLastWriteTime.dwHighDateTime)
                     << 32) |
                    d.ftLastWriteTime.dwLowDateTime;
  r.modified_time = time::unix_timestamp_from_windows_filetime_ticks(t);
  return r;
}

Status make_directory(const char* path) noexcept {
  WidePath wp(path);
  if (wp.w() == nullptr) {
    return Status::InvalidArgument(
        "make directory: path is null or not valid UTF-8");
  }
  if (CreateDirectoryW(wp.w(), nullptr)) return Status::Ok();
  const DWORD code = GetLastError();
  if (code == ERROR_ALREADY_EXISTS) {
    auto directory = is_directory(path);
    if (directory && *directory) return Status::Ok();
  }
  return win32_fs_status("CreateDirectoryW", code);
}

Status remove_directory(const char* path) noexcept {
  WidePath wp(path);
  if (wp.w() == nullptr) {
    return Status::InvalidArgument(
        "remove directory: path is null or not valid UTF-8");
  }
  if (RemoveDirectoryW(wp.w()) != 0) return Status::Ok();
  return win32_fs_status("RemoveDirectoryW", ::GetLastError());
}

Status remove_file(const char* path) noexcept {
  WidePath wp(path);
  if (wp.w() == nullptr) {
    return Status::InvalidArgument(
        "remove file: path is null or not valid UTF-8");
  }
  if (DeleteFileW(wp.w()) != 0) return Status::Ok();
  return win32_fs_status("DeleteFileW", ::GetLastError());
}

// Remove a symlink/junction ITSELF, never its target. A directory reparse
// (junction / dir symlink) needs RemoveDirectoryW; a file symlink needs
// DeleteFileW. Used by the recursive delete so it never follows a link out of
// the tree (platform-audit #1 data-loss guard).
Status remove_symlink(const char* path) noexcept {
  WidePath wp(path);
  if (wp.w() == nullptr) {
    return Status::InvalidArgument(
        "remove symlink: path is null or not valid UTF-8");
  }
  DWORD a = GetFileAttributesW(wp.w());
  if (a == INVALID_FILE_ATTRIBUTES) {
    return win32_fs_status("GetFileAttributesW symlink", ::GetLastError());
  }
  const BOOL removed = (a & FILE_ATTRIBUTE_DIRECTORY)
                           ? RemoveDirectoryW(wp.w())
                           : DeleteFileW(wp.w());
  if (removed) return Status::Ok();
  return win32_fs_status("remove symlink", ::GetLastError());
}

Status rename_path(const char* from, const char* to) noexcept {
  WidePath wf(from), wt(to);
  if (wf.w() == nullptr || wt.w() == nullptr) {
    return Status::InvalidArgument(
        "rename: path is null or not valid UTF-8");
  }
  if (MoveFileExW(wf.w(), wt.w(),
                  MOVEFILE_REPLACE_EXISTING | MOVEFILE_COPY_ALLOWED) != 0) {
    return Status::Ok();
  }
  return win32_fs_status("MoveFileExW", ::GetLastError());
}

Status copy_file(const char* from, const char* to) noexcept {
  WidePath wf(from), wt(to);
  if (wf.w() == nullptr || wt.w() == nullptr) {
    return Status::InvalidArgument(
        "copy file: path is null or not valid UTF-8");
  }
  if (CopyFileW(wf.w(), wt.w(), /*bFailIfExists=*/FALSE) != 0) {
    return Status::Ok();
  }
  return win32_fs_status("CopyFileW", ::GetLastError());
}

Result<std::size_t> current_directory(char* dst, std::size_t cap) noexcept {
  clear_output(dst, cap);
  wchar_t stack[512];
  wchar_t* buf = stack;
  wchar_t* heap = nullptr;
  DWORD buf_cap = static_cast<DWORD>(sizeof(stack) / sizeof(stack[0]));
  DWORD n = GetCurrentDirectoryW(buf_cap, buf);
  if (n >= buf_cap) {
    // The too-small result includes the terminator. CWD can exceed MAX_PATH
    // after SetCurrentDirectoryW on an extended path, so size dynamically.
    buf_cap = n;
    heap = static_cast<wchar_t*>(
        std::malloc(static_cast<std::size_t>(buf_cap) * sizeof(wchar_t)));
    if (heap == nullptr) {
      return std::unexpected(Status::ResourceExhausted(
          "current directory: UTF-16 buffer allocation failed"));
    }
    buf = heap;
    n = GetCurrentDirectoryW(buf_cap, buf);
  }
  if (n == 0) {
    const DWORD code = ::GetLastError();
    std::free(heap);
    return std::unexpected(win32_fs_status("GetCurrentDirectoryW", code));
  }
  if (n >= buf_cap) {
    std::free(heap);
    return std::unexpected(Status::Aborted(
        "current directory changed while it was being read"));
  }
  auto out = fill_w(dst, cap, buf, static_cast<int>(n));
  std::free(heap);
  return out;
}

Status change_directory(const char* path) noexcept {
  WidePath wp(path);
  if (wp.w() == nullptr) {
    return Status::InvalidArgument(
        "change directory: path is null or not valid UTF-8");
  }
  if (SetCurrentDirectoryW(wp.w()) != 0) return Status::Ok();
  return win32_fs_status("SetCurrentDirectoryW", ::GetLastError());
}

Result<std::size_t> temp_directory(char* dst, std::size_t cap) noexcept {
  clear_output(dst, cap);
  wchar_t stack[512];
  wchar_t* buf = stack;
  wchar_t* heap = nullptr;
  DWORD buf_cap = static_cast<DWORD>(sizeof(stack) / sizeof(stack[0]));
  DWORD n = GetTempPathW(buf_cap, buf);  // already honors TMP/TEMP
  if (n >= buf_cap) {
    buf_cap = n + 1;
    heap = static_cast<wchar_t*>(
        std::malloc(static_cast<std::size_t>(buf_cap) * sizeof(wchar_t)));
    if (heap == nullptr) {
      return std::unexpected(Status::ResourceExhausted(
          "temp directory: UTF-16 buffer allocation failed"));
    }
    buf = heap;
    n = GetTempPathW(buf_cap, buf);
  }
  if (n == 0) {
    const DWORD code = ::GetLastError();
    std::free(heap);
    return std::unexpected(win32_fs_status("GetTempPathW", code));
  }
  if (n >= buf_cap) {
    std::free(heap);
    return std::unexpected(
        Status::Aborted("temp directory changed while it was being read"));
  }
  while (n > 1 && (buf[n - 1] == L'\\' || buf[n - 1] == L'/')) --n;  // trim sep
  auto out = fill_w(dst, cap, buf, static_cast<int>(n));
  std::free(heap);
  return out;
}

Result<std::size_t> home_directory(char* dst, std::size_t cap) noexcept {
  return get_env_var("USERPROFILE", dst, cap);
}

Result<bool> directory_iterate(const char* path,
                               bool (*visit)(const char*, FileType, void*),
                               void* ctx) noexcept {
  if (path == nullptr || visit == nullptr) {
    return std::unexpected(
        Status::InvalidArgument("directory iterate: invalid argument"));
  }
  // Build the UTF-16 search pattern "<path>\*". Heap-size it off the path length
  // so long paths aren't capped at MAX_PATH.
  WidePath wp(path);
  if (wp.w() == nullptr) {
    return std::unexpected(
        Status::InvalidArgument("directory iterate: path is not valid UTF-8"));
  }
  std::size_t wl = std::wcslen(wp.w());
  wchar_t stackp[MAX_PATH];
  wchar_t* pat = stackp;
  wchar_t* heapp = nullptr;
  if (wl + 3 > MAX_PATH) {
    heapp = static_cast<wchar_t*>(std::malloc((wl + 3) * sizeof(wchar_t)));
    if (heapp == nullptr) {
      return std::unexpected(Status::ResourceExhausted(
          "directory iterate: search-pattern allocation failed"));
    }
    pat = heapp;
  }
  std::memcpy(pat, wp.w(), wl * sizeof(wchar_t));
  pat[wl] = L'\\';
  pat[wl + 1] = L'*';
  pat[wl + 2] = L'\0';

  WIN32_FIND_DATAW fd;
  // FindFirstFileExW + FindExInfoBasic skips the legacy 8.3 short-name the plain
  // FindFirstFileW computes -- work we never use. FIND_FIRST_EX_LARGE_FETCH asks the
  // FS to batch more entries per kernel round-trip. This is what MSVC's own
  // std::filesystem does (FindExInfoBasic, flags 0).
  // NOTE: measured statistical tie vs std::filesystem (~31.8 vs 32.3 us, ratio 1.034,
  // gap 3.4% < spread 9.2%). The per-entry work (one WideCharToMultiByte + visitor) is
  // dwarfed by the FindNextFileW advance -- this loop is syscall-bound. Kept for
  // strictly-less work + STL parity, not a bench win. (The per-entry convert was
  // trimmed from two WCTMB calls to one; still tie -- confirms the syscall dominates.)
  HANDLE h = FindFirstFileExW(pat, FindExInfoBasic, &fd, FindExSearchNameMatch,
                              nullptr, FIND_FIRST_EX_LARGE_FETCH);
  std::free(heapp);
  if (h == INVALID_HANDLE_VALUE) {
    return std::unexpected(
        win32_fs_status("FindFirstFileExW", ::GetLastError()));
  }
  bool stopped = false;
  Status iteration_status = Status::Ok();
  do {
    const wchar_t* wname = fd.cFileName;
    if (wname[0] == L'.' &&
        (wname[1] == L'\0' || (wname[1] == L'.' && wname[2] == L'\0')))
      continue;
    char name[1024];
    // cFileName is bounded to MAX_PATH (260) wchar, so its UTF-8 form is <= 780
    // bytes + NUL < 1024 -- it ALWAYS fits. So skip fill_w's length-probe call
    // (WideCharToMultiByte(...,nullptr,0,...)) and convert in ONE call straight
    // into the fixed buffer. wlen=-1 processes the terminator too, so the output
    // is NUL-terminated and `got` includes it. A malformed UTF-16 name is an
    // explicit enumeration error instead of reaching the visitor as "".
    int got = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wname, -1, name,
                                  static_cast<int>(sizeof(name)), nullptr, nullptr);
    if (got <= 0) {
      iteration_status =
          win32_fs_status("WideCharToMultiByte(directory entry)", ::GetLastError());
      break;
    }
    FileType type = FileType::kRegular;
    if ((fd.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0) {
      type = FileType::kSymlink;
    } else if ((fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) {
      type = FileType::kDirectory;
    } else if ((fd.dwFileAttributes & FILE_ATTRIBUTE_DEVICE) != 0) {
      type = FileType::kCharacterDevice;
    }
    if (!visit(name, type, ctx)) {
      stopped = true;
      break;
    }
  } while (FindNextFileW(h, &fd));
  const DWORD end_error =
      (stopped || !iteration_status.ok()) ? ERROR_NO_MORE_FILES : GetLastError();
  const BOOL closed = FindClose(h);
  if (!iteration_status.ok()) return std::unexpected(std::move(iteration_status));
  if (!stopped && end_error != ERROR_NO_MORE_FILES) {
    return std::unexpected(win32_fs_status("FindNextFileW", end_error));
  }
  if (!closed) {
    return std::unexpected(
        win32_fs_status("FindClose", ::GetLastError()));
  }
  return !stopped;
}

namespace {
Status close_raw(std::intptr_t raw) noexcept {
  if (raw == FileHandle::kInvalid) return Status::Ok();
  if (::CloseHandle(reinterpret_cast<HANDLE>(raw))) return Status::Ok();
  return win32_fs_status("CloseHandle", ::GetLastError());
}
Result<std::intptr_t> duplicate_raw(std::intptr_t raw) noexcept {
  if (raw == FileHandle::kInvalid) {
    return std::unexpected(
        Status::FailedPrecondition("file duplicate: handle is closed"));
  }
  HANDLE duplicate = INVALID_HANDLE_VALUE;
  if (!::DuplicateHandle(::GetCurrentProcess(), reinterpret_cast<HANDLE>(raw),
                         ::GetCurrentProcess(), &duplicate, 0, FALSE,
                         DUPLICATE_SAME_ACCESS)) {
    return std::unexpected(
        win32_fs_status("DuplicateHandle", ::GetLastError()));
  }
  return reinterpret_cast<std::intptr_t>(duplicate);
}
Result<FileInfo> info_raw(std::intptr_t raw) noexcept {
  if (raw == FileHandle::kInvalid) {
    return std::unexpected(
        Status::FailedPrecondition("file info: handle is closed"));
  }
  const HANDLE handle = reinterpret_cast<HANDLE>(raw);
  ::SetLastError(ERROR_SUCCESS);
  const DWORD native_type = ::GetFileType(handle);
  if (native_type == FILE_TYPE_UNKNOWN) {
    const DWORD code = ::GetLastError();
    if (code != ERROR_SUCCESS) {
      return std::unexpected(win32_fs_status("GetFileType(file info)", code));
    }
    return FileInfo{};
  }

  FileInfo out;
  if (native_type == FILE_TYPE_CHAR) {
    out.type = FileType::kCharacterDevice;
    return out;
  }
  if (native_type == FILE_TYPE_PIPE) {
    out.type = FileType::kFifo;
    return out;
  }

  BY_HANDLE_FILE_INFORMATION native{};
  if (!::GetFileInformationByHandle(handle, &native)) {
    return std::unexpected(win32_fs_status(
        "GetFileInformationByHandle(file info)", ::GetLastError()));
  }
  if (native_type == FILE_TYPE_DISK) {
    out.type = (native.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0
                   ? FileType::kDirectory
                   : FileType::kRegular;
  }
  if (out.is_regular()) {
    out.size = (static_cast<std::uint64_t>(native.nFileSizeHigh) << 32) |
               native.nFileSizeLow;
  }
  const std::uint64_t ticks =
      (static_cast<std::uint64_t>(native.ftLastWriteTime.dwHighDateTime)
       << 32) |
      native.ftLastWriteTime.dwLowDateTime;
  out.modified_time = time::unix_timestamp_from_windows_filetime_ticks(ticks);
  return out;
}
Result<std::size_t> read_raw(std::intptr_t raw, void* buf,
                             std::size_t n) noexcept {
  if (raw == FileHandle::kInvalid) {
    return std::unexpected(
        Status::FailedPrecondition("file read: handle is closed"));
  }
  DWORD want = n > 0xFFFFFFFFull ? 0xFFFFFFFFu : static_cast<DWORD>(n);
  DWORD got = 0;
  if (!ReadFile(reinterpret_cast<HANDLE>(raw), buf, want, &got, nullptr)) {
    return std::unexpected(win32_fs_status("ReadFile", ::GetLastError()));
  }
  return static_cast<std::size_t>(got);
}
Result<std::size_t> write_raw(std::intptr_t raw, const void* buf,
                              std::size_t n) noexcept {
  if (raw == FileHandle::kInvalid) {
    return std::unexpected(
        Status::FailedPrecondition("file write: handle is closed"));
  }
  DWORD want = n > 0xFFFFFFFFull ? 0xFFFFFFFFu : static_cast<DWORD>(n);
  DWORD put = 0;
  if (!WriteFile(reinterpret_cast<HANDLE>(raw), buf, want, &put, nullptr)) {
    return std::unexpected(win32_fs_status("WriteFile", ::GetLastError()));
  }
  return static_cast<std::size_t>(put);
}
Result<std::size_t> read_at_raw(std::intptr_t raw, std::uint64_t offset,
                                void* buf, std::size_t n) noexcept {
  if (raw == FileHandle::kInvalid) {
    return std::unexpected(
        Status::FailedPrecondition("file read at: handle is closed"));
  }
  if (offset > static_cast<std::uint64_t>(
                   (std::numeric_limits<std::int64_t>::max)())) {
    return std::unexpected(
        Status::OutOfRange("file read at: offset exceeds native range"));
  }
  OVERLAPPED overlapped{};
  overlapped.Offset = static_cast<DWORD>(offset);
  overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);
  const DWORD want = n > 0xFFFFFFFFull ? 0xFFFFFFFFu : static_cast<DWORD>(n);
  DWORD got = 0;
  if (!::ReadFile(reinterpret_cast<HANDLE>(raw), buf, want, &got,
                  &overlapped)) {
    const DWORD code = ::GetLastError();
    if (code == ERROR_HANDLE_EOF) return std::size_t{0};
    return std::unexpected(win32_fs_status("ReadFile(offset)", code));
  }
  return static_cast<std::size_t>(got);
}
Result<std::size_t> write_at_raw(std::intptr_t raw, std::uint64_t offset,
                                 const void* buf, std::size_t n) noexcept {
  if (raw == FileHandle::kInvalid) {
    return std::unexpected(
        Status::FailedPrecondition("file write at: handle is closed"));
  }
  if (offset > static_cast<std::uint64_t>(
                   (std::numeric_limits<std::int64_t>::max)())) {
    return std::unexpected(
        Status::OutOfRange("file write at: offset exceeds native range"));
  }
  OVERLAPPED overlapped{};
  overlapped.Offset = static_cast<DWORD>(offset);
  overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);
  const DWORD want = n > 0xFFFFFFFFull ? 0xFFFFFFFFu : static_cast<DWORD>(n);
  DWORD put = 0;
  if (!::WriteFile(reinterpret_cast<HANDLE>(raw), buf, want, &put,
                   &overlapped)) {
    return std::unexpected(
        win32_fs_status("WriteFile(offset)", ::GetLastError()));
  }
  return static_cast<std::size_t>(put);
}
Result<std::uint64_t> seek_raw(std::intptr_t raw, std::int64_t offset,
                               SeekOrigin origin) noexcept {
  if (raw == FileHandle::kInvalid) {
    return std::unexpected(
        Status::FailedPrecondition("file seek: handle is closed"));
  }
  DWORD method = FILE_BEGIN;
  switch (origin) {
    case SeekOrigin::kBegin: method = FILE_BEGIN; break;
    case SeekOrigin::kCurrent: method = FILE_CURRENT; break;
    case SeekOrigin::kEnd: method = FILE_END; break;
    default:
      return std::unexpected(
          Status::InvalidArgument("file seek: invalid origin"));
  }
  if (origin == SeekOrigin::kBegin && offset < 0) {
    return std::unexpected(
        Status::OutOfRange("file seek: negative absolute position"));
  }
  LARGE_INTEGER distance{};
  distance.QuadPart = offset;
  LARGE_INTEGER position{};
  if (!::SetFilePointerEx(reinterpret_cast<HANDLE>(raw), distance, &position,
                          method)) {
    return std::unexpected(
        win32_fs_status("SetFilePointerEx", ::GetLastError()));
  }
  if (position.QuadPart < 0) {
    return std::unexpected(
        Status::OutOfRange("file seek: resulting position is negative"));
  }
  return static_cast<std::uint64_t>(position.QuadPart);
}
Status sync_raw(std::intptr_t raw, bool data_only) noexcept {
  (void)data_only;
  if (raw == FileHandle::kInvalid) {
    return Status::FailedPrecondition("file sync: handle is closed");
  }
  if (!::FlushFileBuffers(reinterpret_cast<HANDLE>(raw))) {
    return win32_fs_status("FlushFileBuffers", ::GetLastError());
  }
  return Status::Ok();
}
Result<std::uint64_t> size_raw(std::intptr_t raw) noexcept {
  if (raw == FileHandle::kInvalid) {
    return std::unexpected(
        Status::FailedPrecondition("file size: handle is closed"));
  }
  LARGE_INTEGER size{};
  if (!::GetFileSizeEx(reinterpret_cast<HANDLE>(raw), &size)) {
    return std::unexpected(
        win32_fs_status("GetFileSizeEx", ::GetLastError()));
  }
  if (size.QuadPart < 0) {
    return std::unexpected(
        Status::DataLoss("GetFileSizeEx returned a negative size"));
  }
  return static_cast<std::uint64_t>(size.QuadPart);
}
Status resize_raw(std::intptr_t raw, std::uint64_t size) noexcept {
  if (raw == FileHandle::kInvalid) {
    return Status::FailedPrecondition("file resize: handle is closed");
  }
  if (size > static_cast<std::uint64_t>(
                 (std::numeric_limits<std::int64_t>::max)())) {
    return Status::OutOfRange("file resize exceeds native offset range");
  }
  FILE_END_OF_FILE_INFO end{};
  end.EndOfFile.QuadPart = static_cast<LONGLONG>(size);
  if (!::SetFileInformationByHandle(reinterpret_cast<HANDLE>(raw),
                                    FileEndOfFileInfo, &end, sizeof(end))) {
    return win32_fs_status("SetFileInformationByHandle(end of file)",
                           ::GetLastError());
  }
  return Status::Ok();
}
Status preallocate_raw(std::intptr_t raw,
                       std::uint64_t minimum_size) noexcept {
  if (raw == FileHandle::kInvalid) {
    return Status::FailedPrecondition("file preallocate: handle is closed");
  }
  if (minimum_size > static_cast<std::uint64_t>(
                         (std::numeric_limits<std::int64_t>::max)())) {
    return Status::OutOfRange("file preallocate exceeds native offset range");
  }
  FILE_STANDARD_INFO current{};
  if (!::GetFileInformationByHandleEx(reinterpret_cast<HANDLE>(raw),
                                      FileStandardInfo, &current,
                                      sizeof(current))) {
    return win32_fs_status(
        "GetFileInformationByHandleEx(preallocate)", ::GetLastError());
  }
  const LONGLONG target = static_cast<LONGLONG>(minimum_size);
  if (current.EndOfFile.QuadPart >= target) return Status::Ok();

  if (current.AllocationSize.QuadPart < target) {
    FILE_ALLOCATION_INFO allocation{};
    allocation.AllocationSize.QuadPart = target;
    if (!::SetFileInformationByHandle(reinterpret_cast<HANDLE>(raw),
                                      FileAllocationInfo, &allocation,
                                      sizeof(allocation))) {
      return win32_fs_status("SetFileInformationByHandle(allocation)",
                             ::GetLastError());
    }
  }
  FILE_END_OF_FILE_INFO end{};
  end.EndOfFile.QuadPart = target;
  if (!::SetFileInformationByHandle(reinterpret_cast<HANDLE>(raw),
                                    FileEndOfFileInfo, &end, sizeof(end))) {
    return win32_fs_status(
        "SetFileInformationByHandle(preallocate end of file)",
        ::GetLastError());
  }
  return Status::Ok();
}
Status lock_raw(std::intptr_t raw, FileLockMode mode) noexcept {
  if (raw == FileHandle::kInvalid) {
    return Status::FailedPrecondition("file lock: handle is closed");
  }
  DWORD flags = mode == FileLockMode::kExclusive
                    ? LOCKFILE_EXCLUSIVE_LOCK
                    : 0;
  OVERLAPPED overlapped{};
  if (!::LockFileEx(reinterpret_cast<HANDLE>(raw), flags, 0, MAXDWORD,
                    MAXDWORD, &overlapped)) {
    return win32_fs_status("LockFileEx", ::GetLastError());
  }
  return Status::Ok();
}
Result<bool> try_lock_raw(std::intptr_t raw, FileLockMode mode) noexcept {
  if (raw == FileHandle::kInvalid) {
    return std::unexpected(
        Status::FailedPrecondition("file try lock: handle is closed"));
  }
  DWORD flags = LOCKFILE_FAIL_IMMEDIATELY;
  if (mode == FileLockMode::kExclusive) flags |= LOCKFILE_EXCLUSIVE_LOCK;
  OVERLAPPED overlapped{};
  if (::LockFileEx(reinterpret_cast<HANDLE>(raw), flags, 0, MAXDWORD,
                   MAXDWORD, &overlapped)) {
    return true;
  }
  const DWORD code = ::GetLastError();
  if (code == ERROR_LOCK_VIOLATION) return false;
  return std::unexpected(win32_fs_status("LockFileEx(try)", code));
}
Status unlock_raw(std::intptr_t raw) noexcept {
  if (raw == FileHandle::kInvalid) {
    return Status::FailedPrecondition("file unlock: handle is closed");
  }
  OVERLAPPED overlapped{};
  if (!::UnlockFileEx(reinterpret_cast<HANDLE>(raw), 0, MAXDWORD, MAXDWORD,
                      &overlapped)) {
    return win32_fs_status("UnlockFileEx", ::GetLastError());
  }
  return Status::Ok();
}
}  // namespace

Result<FileHandle> open_file(const char* path, unsigned flags) noexcept {
  constexpr unsigned kKnown = kOpenRead | kOpenWrite | kOpenCreate |
                              kOpenTruncate | kOpenAppend;
  if (path == nullptr || path[0] == '\0' || (flags & ~kKnown) != 0 ||
      (flags & (kOpenRead | kOpenWrite)) == 0 ||
      ((flags & (kOpenCreate | kOpenTruncate | kOpenAppend)) != 0 &&
       (flags & kOpenWrite) == 0)) {
    return std::unexpected(
        Status::InvalidArgument("open file: invalid path or flag combination"));
  }
  WidePath wp(path);
  if (wp.w() == nullptr) {
    return std::unexpected(
        Status::InvalidArgument("open file: path is not valid UTF-8"));
  }
  DWORD access = 0;
  if (flags & kOpenRead) access |= GENERIC_READ;
  if (flags & kOpenWrite) {
    access |= (flags & kOpenAppend) ? FILE_APPEND_DATA : GENERIC_WRITE;
  }
  DWORD creation;
  if (flags & kOpenTruncate)
    creation = (flags & kOpenCreate) ? CREATE_ALWAYS : TRUNCATE_EXISTING;
  else
    creation = (flags & kOpenCreate) ? OPEN_ALWAYS : OPEN_EXISTING;
  // Match POSIX's unlink/rename-while-open lifetime: an owning FileHandle keeps
  // the file object alive but must not block atomic replacement of its path.
  // libuv uses the same share mask for its portable file surface.
  HANDLE h = CreateFileW(
      wp.w(), access, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
      nullptr, creation, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE) {
    return std::unexpected(win32_fs_status("CreateFileW", ::GetLastError()));
  }
  return FileHandle{reinterpret_cast<std::intptr_t>(h)};
}

Result<std::uint32_t> get_permissions(const char* path) noexcept {
  WidePath wp(path);
  if (wp.w() == nullptr) {
    return std::unexpected(Status::InvalidArgument(
        "get permissions: path is null or not valid UTF-8"));
  }
  DWORD a = GetFileAttributesW(wp.w());
  if (a == INVALID_FILE_ATTRIBUTES) {
    return std::unexpected(
        win32_fs_status("GetFileAttributesW", ::GetLastError()));
  }
  // No POSIX permission model on Windows: synthesize from the READONLY attribute.
  return (a & FILE_ATTRIBUTE_READONLY) ? 0444u : 0666u;
}

Status set_permissions(const char* path, std::uint32_t mode) noexcept {
  WidePath wp(path);
  if (wp.w() == nullptr) {
    return Status::InvalidArgument(
        "set permissions: path is null or not valid UTF-8");
  }
  DWORD a = GetFileAttributesW(wp.w());
  if (a == INVALID_FILE_ATTRIBUTES) {
    return win32_fs_status("GetFileAttributesW", ::GetLastError());
  }
  // Honor only the write bits: any owner/group/other write clear => READONLY.
  const bool writable = (mode & 0222u) != 0;
  if (writable) a &= ~static_cast<DWORD>(FILE_ATTRIBUTE_READONLY);
  else a |= FILE_ATTRIBUTE_READONLY;
  if (SetFileAttributesW(wp.w(), a) != 0) return Status::Ok();
  return win32_fs_status("SetFileAttributesW", ::GetLastError());
}

Result<bool> is_readonly(const char* path) noexcept {
  HYLE_ASSIGN_OR_RETURN(std::uint32_t mode, get_permissions(path));
  return (mode & 0222u) == 0;
}

Status set_readonly(const char* path, bool readonly) noexcept {
  return set_permissions(path, readonly ? 0444u : 0666u);
}

Status create_symlink(const char* target, const char* link,
                      SymlinkKind kind) noexcept {
  WidePath wt(target), wl(link);
  if (wt.w() == nullptr || wl.w() == nullptr ||
      (kind != SymlinkKind::kAuto && kind != SymlinkKind::kFile &&
       kind != SymlinkKind::kDirectory)) {
    return Status::InvalidArgument(
        "create symlink: invalid path, UTF-8, or kind");
  }
  // Flag bit is target type; add UNPRIVILEGED_CREATE so Developer Mode suffices.
  DWORD flags = SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE;
  bool directory = kind == SymlinkKind::kDirectory;
  if (kind == SymlinkKind::kAuto) {
    auto target_is_directory = is_directory(target);
    if (!target_is_directory) return std::move(target_is_directory).error();
    directory = *target_is_directory;
  }
  if (directory) {
    flags |= SYMBOLIC_LINK_FLAG_DIRECTORY;
  }
  if (CreateSymbolicLinkW(wl.w(), wt.w(), flags) != 0) return Status::Ok();
  return win32_fs_status("CreateSymbolicLinkW", ::GetLastError());
}

Status create_hard_link(const char* target, const char* link) noexcept {
  WidePath wt(target), wl(link);
  if (wt.w() == nullptr || wl.w() == nullptr) {
    return Status::InvalidArgument(
        "create hard link: path is null or not valid UTF-8");
  }
  // Hard link: another directory entry for the SAME file data (same volume,
  // files only -- no directory hard links). No special privilege needed.
  if (CreateHardLinkW(wl.w(), wt.w(), nullptr) != 0) return Status::Ok();
  return win32_fs_status("CreateHardLinkW", ::GetLastError());
}

Result<bool> is_symlink(const char* path) noexcept {
  WidePath wp(path);
  if (wp.w() == nullptr) {
    return std::unexpected(Status::InvalidArgument(
        "is symlink: path is null or not valid UTF-8"));
  }
  const DWORD attributes = ::GetFileAttributesW(wp.w());
  if (attributes != INVALID_FILE_ATTRIBUTES) {
    return (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
  }
  const DWORD code = ::GetLastError();
  if (code == ERROR_FILE_NOT_FOUND || code == ERROR_PATH_NOT_FOUND) return false;
  return std::unexpected(win32_fs_status("GetFileAttributesW symlink", code));
}

Result<std::size_t> read_symlink(const char* path, char* dst,
                                 std::size_t cap) noexcept {
  WidePath wp(path);
  if (wp.w() == nullptr) {
    return std::unexpected(Status::InvalidArgument(
        "read symlink: path is null or not valid UTF-8"));
  }
  // Open the link ITSELF (OPEN_REPARSE_POINT) and pull its stored target out of
  // the raw reparse buffer -- GetFinalPathNameByHandle would resolve THROUGH the
  // link to the final object, not report the link's own target string.
  HANDLE h = CreateFileW(wp.w(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                         OPEN_EXISTING,
                         FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT,
                         nullptr);
  if (h == INVALID_HANDLE_VALUE) {
    return std::unexpected(
        win32_fs_status("CreateFileW(OPEN_REPARSE_POINT)", ::GetLastError()));
  }

  // REPARSE_DATA_BUFFER (ntifs.h; not in the plain Win32 SDK headers, so lay it
  // out locally -- the shape is stable ABI).
  struct ReparseHeader {
    ULONG ReparseTag;
    USHORT ReparseDataLength;
    USHORT Reserved;
    USHORT SubstituteNameOffset;
    USHORT SubstituteNameLength;
    USHORT PrintNameOffset;
    USHORT PrintNameLength;
    ULONG Flags;
    wchar_t PathBuffer[1];
  };
  alignas(ReparseHeader) unsigned char raw[16 * 1024];  // MAXIMUM_REPARSE_DATA_BUFFER_SIZE
  DWORD got = 0;
  BOOL ok = DeviceIoControl(h, FSCTL_GET_REPARSE_POINT, nullptr, 0, raw,
                            sizeof(raw), &got, nullptr);
  const DWORD operation_error = ok ? ERROR_SUCCESS : GetLastError();
  const BOOL closed = CloseHandle(h);
  if (!ok) {
    return std::unexpected(
        win32_fs_status("FSCTL_GET_REPARSE_POINT", operation_error));
  }
  if (!closed) {
    return std::unexpected(
        win32_fs_status("CloseHandle", ::GetLastError()));
  }
  auto* rep = reinterpret_cast<ReparseHeader*>(raw);
  if (rep->ReparseTag != IO_REPARSE_TAG_SYMLINK) {
    return std::unexpected(
        Status::FailedPrecondition("read symlink: path is not a symlink"));
  }
  // Prefer the human-readable PrintName; fall back to SubstituteName.
  USHORT off = rep->PrintNameLength ? rep->PrintNameOffset
                                    : rep->SubstituteNameOffset;
  USHORT len = rep->PrintNameLength ? rep->PrintNameLength
                                    : rep->SubstituteNameLength;
  // off/len come straight from the (kernel-supplied but still untrusted-shaped)
  // reparse buffer. Bound them against the bytes DeviceIoControl actually wrote
  // so a malformed record can't drive an OOB read past `raw`. PathBuffer starts
  // at offsetof(PathBuffer); the name occupies [that+off, that+off+len).
  const std::size_t name_base = offsetof(ReparseHeader, PathBuffer);
  const std::size_t name_end = name_base + static_cast<std::size_t>(off) + len;
  if (name_end > got || name_end > sizeof(raw)) {
    return std::unexpected(
        Status::DataLoss("read symlink: malformed reparse record"));
  }
  const wchar_t* name =
      reinterpret_cast<const wchar_t*>(
          reinterpret_cast<const unsigned char*>(rep->PathBuffer) + off);
  return fill_w(dst, cap, name, static_cast<int>(len / sizeof(wchar_t)));
}

Result<CreatedTempFile> create_temp_file(const char* dir, const char* prefix,
                                         char* dst, std::size_t cap) noexcept {
  clear_output(dst, cap);
  if (!valid_temp_prefix(prefix)) {
    return std::unexpected(Status::InvalidArgument(
        "create temp file: prefix must be one filename fragment"));
  }
  TempDirectoryInput directory;
  Status directory_status = directory.resolve(dir);
  if (!directory_status.ok())
    return std::unexpected(std::move(directory_status));

  WidePath wd(directory.path());
  if (wd.w() == nullptr) {
    return std::unexpected(
        wd.allocation_failed()
            ? Status::ResourceExhausted(
                  "create temp file: directory conversion allocation failed")
            : Status::InvalidArgument(
                  "create temp file: invalid UTF-8 directory"));
  }
  const std::size_t dl = std::wcslen(wd.w());
  const char* p = prefix != nullptr ? prefix : "tmp";
  WidePath wp(p, false);
  if (wp.w() == nullptr) {
    return std::unexpected(
        wp.allocation_failed()
            ? Status::ResourceExhausted(
                  "create temp file: prefix conversion allocation failed")
            : Status::InvalidArgument(
                  "create temp file: prefix is not valid UTF-8"));
  }
  const wchar_t* wpre = wp.w();
  const std::size_t pl = std::wcslen(wpre);

  const bool needs_separator =
      wd.w()[dl - 1] != L'\\' && wd.w()[dl - 1] != L'/';
  std::size_t path_chars = dl;
  if ((needs_separator && !checked_size_add(path_chars, 1, path_chars)) ||
      !checked_size_add(path_chars, pl, path_chars) ||
      !checked_size_add(path_chars, 16, path_chars) ||
      !checked_size_add(path_chars, 4, path_chars) ||
      !checked_size_add(path_chars, 1, path_chars)) {
    return std::unexpected(
        Status::OutOfRange("create temp file: generated path is too long"));
  }
  MallocArray<wchar_t> wpath;
  if (!wpath.resize(path_chars)) {
    return std::unexpected(
        wpath.overflowed()
            ? Status::OutOfRange(
                  "create temp file: generated path is too long")
            : Status::ResourceExhausted(
                  "create temp file: path allocation failed"));
  }
  std::memcpy(wpath.data(), wd.w(), dl * sizeof(wchar_t));
  std::size_t random_offset = dl;
  if (needs_separator) wpath.data()[random_offset++] = L'\\';
  std::memcpy(wpath.data() + random_offset, wpre, pl * sizeof(wchar_t));
  random_offset += pl;
  constexpr wchar_t kExt[] = L".tmp";
  std::memcpy(wpath.data() + random_offset + 16, kExt, sizeof(kExt));

  // TRUE atomic create (mkstemp-equivalent): generate a high-entropy random name
  // and CreateFileW with CREATE_NEW, which fails if the name already exists. On
  // collision (ERROR_FILE_EXISTS) we retry a fresh name; any other error aborts.
  // No GetTempFileName (which pre-creates the file with a predictable 16-bit
  // counter, then needs a racy reopen and can leave an orphan). CREATE_NEW gives
  // us the file exclusively in one syscall -> no TOCTOU window, no orphan on fail.
  constexpr int kMaxTries = 128;
  const wchar_t kHex[] = L"0123456789abcdef";
  for (int attempt = 0; attempt < kMaxTries; ++attempt) {
    // One OS entropy request supplies 64 unpredictable bits for 16 hex chars.
    std::size_t w = random_offset;
    std::uint64_t random = 0;
    Status entropy = secure_random(std::span<std::byte>(
        reinterpret_cast<std::byte*>(&random), sizeof(random)));
    if (!entropy.ok()) {
      return std::unexpected(
          std::move(entropy).WithContext("create temp file name"));
    }
    for (int i = 0; i < 16; ++i) {
      wpath.data()[w++] = kHex[random & 0xF];
      random >>= 4;
    }

    HANDLE h = CreateFileW(wpath.data(), GENERIC_READ | GENERIC_WRITE, 0,
                           nullptr, CREATE_NEW, FILE_ATTRIBUTE_TEMPORARY,
                           nullptr);
    if (h != INVALID_HANDLE_VALUE) {
      auto output = fill_w(dst, cap, wpath.data(), -1);
      if (!output) {
        Status conversion_status = std::move(output).error();
        ::CloseHandle(h);
        ::DeleteFileW(wpath.data());
        return std::unexpected(std::move(conversion_status));
      }
      return CreatedTempFile{
          FileHandle{reinterpret_cast<std::intptr_t>(h)}, *output};
    }
    const DWORD code = GetLastError();
    if (code != ERROR_FILE_EXISTS && code != ERROR_ALREADY_EXISTS) {
      return std::unexpected(win32_fs_status("CreateFileW(CREATE_NEW)", code));
    }
    // else: name collided -> loop and try a fresh random name.
  }
  return std::unexpected(Status::AlreadyExists(
      "create temp file: exhausted unique-name retries"));
}

#else

// ---- POSIX ------------------------------------------------------------------

namespace {

Status posix_fs_status(const char* operation, int code) noexcept {
  // The error module owns the generic errno taxonomy. Filesystem operations
  // override only ENOTDIR, whose path lookup contract means "a required path
  // component does not exist as a directory". The canonical factory still
  // owns strerror formatting and native-detail capture.
  Status status = status_from_errno(code, operation);
  if (code != ENOTDIR) return status;
  return Status::FromNative(StatusCode::NotFound, status.native_error(),
                            status.message());
}

FileType file_type_from_mode(mode_t mode) noexcept {
  if (S_ISREG(mode)) return FileType::kRegular;
  if (S_ISDIR(mode)) return FileType::kDirectory;
  if (S_ISLNK(mode)) return FileType::kSymlink;
  if (S_ISBLK(mode)) return FileType::kBlockDevice;
  if (S_ISCHR(mode)) return FileType::kCharacterDevice;
  if (S_ISFIFO(mode)) return FileType::kFifo;
  if (S_ISSOCK(mode)) return FileType::kSocket;
  return FileType::kUnknown;
}

FileType file_type_from_dirent(unsigned char type) noexcept {
  switch (type) {
    case DT_REG: return FileType::kRegular;
    case DT_DIR: return FileType::kDirectory;
    case DT_LNK: return FileType::kSymlink;
    case DT_BLK: return FileType::kBlockDevice;
    case DT_CHR: return FileType::kCharacterDevice;
    case DT_FIFO: return FileType::kFifo;
    case DT_SOCK: return FileType::kSocket;
    default: return FileType::kUnknown;
  }
}

}  // namespace

Result<FileInfo> stat(const char* path) noexcept {
  if (path == nullptr) {
    return std::unexpected(Status::InvalidArgument("stat: path is null"));
  }
  struct ::stat s;
  FileInfo r;
  if (detail::retry_eintr_negative(
          [&]() noexcept { return ::stat(path, &s); }) != 0) {
    return std::unexpected(posix_fs_status("stat", errno));
  }
  r.type = file_type_from_mode(s.st_mode);
  r.size = r.is_regular() ? static_cast<std::uint64_t>(s.st_size) : 0;
#  if HYLE_OS_MACOS
  r.modified_time = time::unix_timestamp(
      static_cast<std::int64_t>(s.st_mtimespec.tv_sec),
      static_cast<std::uint32_t>(s.st_mtimespec.tv_nsec));
#  else
  r.modified_time = time::unix_timestamp(
      static_cast<std::int64_t>(s.st_mtim.tv_sec),
      static_cast<std::uint32_t>(s.st_mtim.tv_nsec));
#  endif
  // stat() follows symlinks; an extra lstat tells us if the path ITSELF is a link
  // (a symlink to a directory therefore has type=kDirectory + is_symlink=true).
  struct ::stat ls;
  if (detail::retry_eintr_negative(
          [&]() noexcept { return ::lstat(path, &ls); }) == 0 &&
      S_ISLNK(ls.st_mode)) {
    r.is_symlink = true;
  }
  return r;
}

Status make_directory(const char* path) noexcept {
  if (path == nullptr) {
    return Status::InvalidArgument("make directory: path is null");
  }
  if (::mkdir(path, 0777) == 0) return Status::Ok();
  const int code = errno;
  if (code == EEXIST) {
    auto directory = is_directory(path);
    if (directory && *directory) return Status::Ok();
  }
  return posix_fs_status("mkdir", code);
}

Status remove_directory(const char* path) noexcept {
  if (path == nullptr) {
    return Status::InvalidArgument("remove directory: path is null");
  }
  if (::rmdir(path) == 0) return Status::Ok();
  return posix_fs_status("rmdir", errno);
}

Status remove_file(const char* path) noexcept {
  if (path == nullptr) return Status::InvalidArgument("remove file: path is null");
  if (::unlink(path) == 0) return Status::Ok();
  return posix_fs_status("unlink", errno);
}

// Remove a symlink ITSELF, never its target: unlink works on any symlink,
// including a symlink-to-directory (rmdir would fail with ENOTDIR). Used by the
// recursive delete so it never follows a link out of the tree (platform-audit #1/#2).
Status remove_symlink(const char* path) noexcept {
  if (::unlink(path) == 0) return Status::Ok();
  return posix_fs_status("unlink symlink", errno);
}

Status rename_path(const char* from, const char* to) noexcept {
  if (from == nullptr || to == nullptr) {
    return Status::InvalidArgument("rename: path is null");
  }
  if (::rename(from, to) == 0) return Status::Ok();
  return posix_fs_status("rename", errno);
}

Status copy_file(const char* from, const char* to) noexcept {
  if (from == nullptr || to == nullptr) {
    return Status::InvalidArgument("copy file: path is null");
  }
  // A verified regular-file self-copy is a successful no-op. Do not return
  // before opening and classifying the source: equal missing/non-regular paths
  // are still errors. Distinct names may also be hard links to the same inode.
  const bool same_path = std::strcmp(from, to) == 0;
  // O_NONBLOCK prevents a path concurrently replaced by a FIFO/device from
  // hanging this synchronous regular-file API before fstat can reject it. It
  // has no effect on regular-file reads.
  int in = detail::retry_eintr_negative([&]() noexcept {
    return ::open(from, O_RDONLY | O_CLOEXEC | O_NONBLOCK);
  });
  if (in < 0) return posix_fs_status("open copy source", errno);
  struct ::stat in_stat;
  if (detail::retry_eintr_negative(
          [&]() noexcept { return ::fstat(in, &in_stat); }) != 0) {
    const int code = errno;
    ::close(in);
    return posix_fs_status("fstat copy source", code);
  }
  if (!S_ISREG(in_stat.st_mode)) {
    ::close(in);
    return Status::FailedPrecondition(
        "copy file: source is not a regular file");
  }
  if (same_path) return close_raw(in);
  // Open without O_TRUNC, then compare the opened descriptors. Checking paths
  // before open has a TOCTOU race and misses hard-link aliases; truncating first
  // would destroy the source before we discover that both fds name one inode.
  int out = detail::retry_eintr_negative([&]() noexcept {
    return ::open(to, O_WRONLY | O_CREAT | O_CLOEXEC | O_NONBLOCK,
                  in_stat.st_mode & 0777);
  });
  if (out < 0) {
    const int code = errno;
    ::close(in);
    return posix_fs_status("open copy destination", code);
  }
  struct ::stat out_stat;
  if (detail::retry_eintr_negative(
          [&]() noexcept { return ::fstat(out, &out_stat); }) != 0) {
    const int code = errno;
    ::close(in);
    ::close(out);
    return posix_fs_status("fstat copy destination", code);
  }
  if (!S_ISREG(out_stat.st_mode)) {
    ::close(in);
    ::close(out);
    return Status::FailedPrecondition(
        "copy file: destination is not a regular file");
  }
  if (in_stat.st_dev == out_stat.st_dev &&
      in_stat.st_ino == out_stat.st_ino) {
    Status status = close_raw(in);
    status.Update(close_raw(out));
    return status;
  }
  if (detail::retry_eintr_negative(
          [&]() noexcept { return ::ftruncate(out, 0); }) != 0) {
    const int code = errno;
    ::close(in);
    ::close(out);
    return posix_fs_status("ftruncate copy destination", code);
  }

  Status status = Status::Ok();
#if HYLE_OS_LINUX
  // Fast path: copy_file_range keeps bytes in the kernel (and can reflink on
  // btrfs/xfs) -- no user-space bounce. Drive it to EOF instead of trusting
  // st_size: procfs/sysfs regular files can report size zero while yielding
  // content. Unsupported-operation failures fall back at the descriptors'
  // current offsets, so a partial kernel copy is never duplicated.
  bool use_userspace = false;
  bool copied_any = false;
  constexpr std::size_t kCopyFileRangeChunk = std::size_t{1} << 30;
  for (;;) {
    ssize_t n =
        ::copy_file_range(in, nullptr, out, nullptr, kCopyFileRangeChunk, 0);
    if (n < 0) {
      if (errno == EINTR) continue;
      const int code = errno;
      if (code == EXDEV || code == ENOSYS || code == EINVAL ||
          code == EOPNOTSUPP) {
        use_userspace = true;
        break;
      }
      status = posix_fs_status("copy_file_range", code);
      break;
    }
    if (n == 0) {
      // A zero-size virtual regular file can still yield bytes through read(2),
      // while copy_file_range reports zero immediately. Probe it in user space;
      // a genuinely empty file costs only one extra EOF read.
      if (!copied_any && in_stat.st_size == 0) use_userspace = true;
      break;
    }
    copied_any = true;
  }
  if (use_userspace) {
#else
  {
#endif
    char buf[64 * 1024];
    for (;;) {
      ssize_t got = ::read(in, buf, sizeof(buf));
      if (got < 0) {
        if (errno == EINTR) continue;  // interrupted by a signal: retry
        status = posix_fs_status("read copy source", errno);
        break;
      }
      if (got == 0) break;
      char* p = buf;
      ssize_t left = got;
      while (left > 0) {
        ssize_t wrote = ::write(out, p, static_cast<std::size_t>(left));
        if (wrote < 0) {
          if (errno == EINTR) continue;  // retry the same chunk
          status = posix_fs_status("write copy destination", errno);
          break;
        }
        if (wrote == 0) {
          status = Status::Unavailable(
              "write copy destination made no forward progress");
          break;
        }
        left -= wrote;
        p += wrote;
      }
      if (!status.ok()) break;
    }
  }
  status.Update(close_raw(in));
  status.Update(close_raw(out));
  return status;
}

Result<std::size_t> current_directory(char* dst, std::size_t cap) noexcept {
  clear_output(dst, cap);
  char stack[512];
  char* buf = stack;
  char* heap = nullptr;
  std::size_t buf_cap = sizeof(stack);
  for (;;) {
    if (::getcwd(buf, buf_cap) != nullptr) {
      const std::size_t result = fill(dst, cap, buf, std::strlen(buf));
      std::free(heap);
      return result;
    }
    const int code = errno;
    if (code != ERANGE) {
      std::free(heap);
      return std::unexpected(posix_fs_status("getcwd", code));
    }
    if (buf_cap > std::numeric_limits<std::size_t>::max() / 2) {
      std::free(heap);
      return std::unexpected(
          Status::OutOfRange("current directory path is too long"));
    }
    buf_cap *= 2;
    char* grown = static_cast<char*>(std::realloc(heap, buf_cap));
    if (grown == nullptr) {
      std::free(heap);
      return std::unexpected(Status::ResourceExhausted(
          "current directory buffer allocation failed"));
    }
    heap = grown;
    buf = heap;
  }
}

Status change_directory(const char* path) noexcept {
  if (path == nullptr) {
    return Status::InvalidArgument("change directory: path is null");
  }
  if (::chdir(path) == 0) return Status::Ok();
  return posix_fs_status("chdir", errno);
}

Result<std::size_t> temp_directory(char* dst, std::size_t cap) noexcept {
  clear_output(dst, cap);
  const char* t = std::getenv("TMPDIR");
  if (t == nullptr || t[0] == '\0') t = std::getenv("TMP");
  if (t == nullptr || t[0] == '\0') t = std::getenv("TEMP");
  if (t == nullptr || t[0] == '\0') t = "/tmp";
  std::size_t n = std::strlen(t);
  while (n > 1 && t[n - 1] == '/') --n;  // trim trailing slash
  return fill(dst, cap, t, n);
}

Result<std::size_t> home_directory(char* dst, std::size_t cap) noexcept {
  clear_output(dst, cap);
  const char* h = std::getenv("HOME");
  if (h != nullptr && h[0] != '\0') return fill(dst, cap, h, std::strlen(h));
  // getpwuid returns a pointer into shared static storage (not thread-safe);
  // getpwuid_r fills caller buffers instead.
  struct passwd pw;
  struct passwd* result = nullptr;
  char stack[1024];
  char* buf = stack;
  char* heap = nullptr;
  std::size_t buf_cap = sizeof(stack);
  for (;;) {
    const int code = ::getpwuid_r(::getuid(), &pw, buf, buf_cap, &result);
    if (code == 0 && result != nullptr && result->pw_dir != nullptr &&
        result->pw_dir[0] != '\0') {
      const std::size_t size =
          fill(dst, cap, result->pw_dir, std::strlen(result->pw_dir));
      std::free(heap);
      return size;
    }
    if (code == 0) {
      std::free(heap);
      return std::unexpected(
          Status::NotFound("home directory: user has no passwd entry"));
    }
    if (code != ERANGE) {
      std::free(heap);
      return std::unexpected(posix_fs_status("getpwuid_r", code));
    }
    if (buf_cap > std::numeric_limits<std::size_t>::max() / 2) {
      std::free(heap);
      return std::unexpected(
          Status::OutOfRange("home directory passwd entry is too large"));
    }
    buf_cap *= 2;
    char* grown = static_cast<char*>(std::realloc(heap, buf_cap));
    if (grown == nullptr) {
      std::free(heap);
      return std::unexpected(Status::ResourceExhausted(
          "home directory passwd buffer allocation failed"));
    }
    heap = grown;
    buf = heap;
  }
}

namespace {
Status close_raw(std::intptr_t raw) noexcept {
  if (raw == FileHandle::kInvalid) return Status::Ok();
  if (::close(static_cast<int>(raw)) == 0) return Status::Ok();
  return posix_fs_status("close", errno);
}
Result<std::intptr_t> duplicate_raw(std::intptr_t raw) noexcept {
  if (raw == FileHandle::kInvalid) {
    return std::unexpected(
        Status::FailedPrecondition("file duplicate: handle is closed"));
  }
  const int source = static_cast<int>(raw);
  int duplicate = -1;
#ifdef F_DUPFD_CLOEXEC
  duplicate = detail::retry_eintr_negative(
      [&]() noexcept { return ::fcntl(source, F_DUPFD_CLOEXEC, 0); });
  if (duplicate < 0) {
    return std::unexpected(posix_fs_status("fcntl(F_DUPFD_CLOEXEC)", errno));
  }
#else
  duplicate = detail::retry_eintr_negative(
      [&]() noexcept { return ::dup(source); });
  if (duplicate < 0) {
    return std::unexpected(posix_fs_status("dup", errno));
  }
  if (detail::retry_eintr_negative([&]() noexcept {
        return ::fcntl(duplicate, F_SETFD, FD_CLOEXEC);
      }) != 0) {
    const int code = errno;
    (void)::close(duplicate);
    return std::unexpected(posix_fs_status("fcntl(FD_CLOEXEC)", code));
  }
#endif
  return static_cast<std::intptr_t>(duplicate);
}
Result<FileInfo> info_raw(std::intptr_t raw) noexcept {
  if (raw == FileHandle::kInvalid) {
    return std::unexpected(
        Status::FailedPrecondition("file info: handle is closed"));
  }
  struct ::stat native {};
  if (detail::retry_eintr_negative([&]() noexcept {
        return ::fstat(static_cast<int>(raw), &native);
      }) != 0) {
    return std::unexpected(posix_fs_status("fstat(file info)", errno));
  }
  FileInfo out;
  out.type = file_type_from_mode(native.st_mode);
  if (out.is_regular()) {
    if (native.st_size < 0) {
      return std::unexpected(
          Status::DataLoss("fstat returned a negative file size"));
    }
    out.size = static_cast<std::uint64_t>(native.st_size);
  }
#if HYLE_OS_MACOS
  out.modified_time = time::unix_timestamp(
      static_cast<std::int64_t>(native.st_mtimespec.tv_sec),
      static_cast<std::uint32_t>(native.st_mtimespec.tv_nsec));
#else
  out.modified_time = time::unix_timestamp(
      static_cast<std::int64_t>(native.st_mtim.tv_sec),
      static_cast<std::uint32_t>(native.st_mtim.tv_nsec));
#endif
  return out;
}
Result<std::size_t> read_raw(std::intptr_t raw, void* buf,
                             std::size_t n) noexcept {
  if (raw == FileHandle::kInvalid) {
    return std::unexpected(
        Status::FailedPrecondition("file read: handle is closed"));
  }
  const std::size_t max_io =
      static_cast<std::size_t>(std::numeric_limits<ssize_t>::max());
  const std::size_t want = n < max_io ? n : max_io;
  for (;;) {
    const ssize_t moved = ::read(static_cast<int>(raw), buf, want);
    if (moved >= 0) return static_cast<std::size_t>(moved);
    if (errno != EINTR) return std::unexpected(posix_fs_status("read", errno));
  }
}
Result<std::size_t> write_raw(std::intptr_t raw, const void* buf,
                              std::size_t n) noexcept {
  if (raw == FileHandle::kInvalid) {
    return std::unexpected(
        Status::FailedPrecondition("file write: handle is closed"));
  }
  const std::size_t max_io =
      static_cast<std::size_t>(std::numeric_limits<ssize_t>::max());
  const std::size_t want = n < max_io ? n : max_io;
  for (;;) {
    const ssize_t moved = ::write(static_cast<int>(raw), buf, want);
    if (moved >= 0) return static_cast<std::size_t>(moved);
    if (errno != EINTR) return std::unexpected(posix_fs_status("write", errno));
  }
}
Result<std::size_t> read_at_raw(std::intptr_t raw, std::uint64_t offset,
                                void* buf, std::size_t n) noexcept {
  if (raw == FileHandle::kInvalid) {
    return std::unexpected(
        Status::FailedPrecondition("file read at: handle is closed"));
  }
  if (offset > static_cast<std::uint64_t>(
                   (std::numeric_limits<std::int64_t>::max)())) {
    return std::unexpected(
        Status::OutOfRange("file read at: offset exceeds native range"));
  }
  const std::size_t max_io =
      static_cast<std::size_t>(std::numeric_limits<ssize_t>::max());
  const std::size_t want = n < max_io ? n : max_io;
  for (;;) {
    const ssize_t moved = ::pread(static_cast<int>(raw), buf, want,
                                  static_cast<off_t>(offset));
    if (moved >= 0) return static_cast<std::size_t>(moved);
    if (errno != EINTR) {
      return std::unexpected(posix_fs_status("pread", errno));
    }
  }
}
Result<std::size_t> write_at_raw(std::intptr_t raw, std::uint64_t offset,
                                 const void* buf, std::size_t n) noexcept {
  if (raw == FileHandle::kInvalid) {
    return std::unexpected(
        Status::FailedPrecondition("file write at: handle is closed"));
  }
  if (offset > static_cast<std::uint64_t>(
                   (std::numeric_limits<std::int64_t>::max)())) {
    return std::unexpected(
        Status::OutOfRange("file write at: offset exceeds native range"));
  }
  const std::size_t max_io =
      static_cast<std::size_t>(std::numeric_limits<ssize_t>::max());
  const std::size_t want = n < max_io ? n : max_io;
  for (;;) {
    const ssize_t moved = ::pwrite(static_cast<int>(raw), buf, want,
                                   static_cast<off_t>(offset));
    if (moved >= 0) return static_cast<std::size_t>(moved);
    if (errno != EINTR) {
      return std::unexpected(posix_fs_status("pwrite", errno));
    }
  }
}
Result<std::uint64_t> seek_raw(std::intptr_t raw, std::int64_t offset,
                               SeekOrigin origin) noexcept {
  if (raw == FileHandle::kInvalid) {
    return std::unexpected(
        Status::FailedPrecondition("file seek: handle is closed"));
  }
  int whence = SEEK_SET;
  switch (origin) {
    case SeekOrigin::kBegin: whence = SEEK_SET; break;
    case SeekOrigin::kCurrent: whence = SEEK_CUR; break;
    case SeekOrigin::kEnd: whence = SEEK_END; break;
    default:
      return std::unexpected(
          Status::InvalidArgument("file seek: invalid origin"));
  }
  if (origin == SeekOrigin::kBegin && offset < 0) {
    return std::unexpected(
        Status::OutOfRange("file seek: negative absolute position"));
  }
  static_assert(sizeof(off_t) >= sizeof(std::int64_t));
  const off_t position = detail::retry_eintr_negative([&]() noexcept {
    return ::lseek(static_cast<int>(raw), static_cast<off_t>(offset), whence);
  });
  if (position < 0) {
    return std::unexpected(posix_fs_status("lseek", errno));
  }
  return static_cast<std::uint64_t>(position);
}
Status sync_raw(std::intptr_t raw, bool data_only) noexcept {
  if (raw == FileHandle::kInvalid) {
    return Status::FailedPrecondition("file sync: handle is closed");
  }
  int rc = -1;
#if HYLE_OS_MACOS
  (void)data_only;
  rc = detail::retry_eintr_negative(
      [&]() noexcept { return ::fsync(static_cast<int>(raw)); });
#else
  rc = detail::retry_eintr_negative([&]() noexcept {
    return data_only ? ::fdatasync(static_cast<int>(raw))
                     : ::fsync(static_cast<int>(raw));
  });
#endif
  if (rc != 0) return posix_fs_status("file sync", errno);
  return Status::Ok();
}
Result<std::uint64_t> size_raw(std::intptr_t raw) noexcept {
  if (raw == FileHandle::kInvalid) {
    return std::unexpected(
        Status::FailedPrecondition("file size: handle is closed"));
  }
  struct stat info {};
  if (detail::retry_eintr_negative([&]() noexcept {
        return ::fstat(static_cast<int>(raw), &info);
      }) != 0) {
    return std::unexpected(posix_fs_status("fstat(file size)", errno));
  }
  if (!S_ISREG(info.st_mode)) {
    return std::unexpected(
        Status::FailedPrecondition("file size: handle is not a regular file"));
  }
  if (info.st_size < 0) {
    return std::unexpected(
        Status::DataLoss("fstat returned a negative file size"));
  }
  return static_cast<std::uint64_t>(info.st_size);
}
Status resize_raw(std::intptr_t raw, std::uint64_t size) noexcept {
  if (raw == FileHandle::kInvalid) {
    return Status::FailedPrecondition("file resize: handle is closed");
  }
  if (size > static_cast<std::uint64_t>(
                 (std::numeric_limits<std::int64_t>::max)())) {
    return Status::OutOfRange("file resize exceeds native offset range");
  }
  static_assert(sizeof(off_t) >= sizeof(std::int64_t));
  if (detail::retry_eintr_negative([&]() noexcept {
        return ::ftruncate(static_cast<int>(raw), static_cast<off_t>(size));
      }) != 0) {
    return posix_fs_status("ftruncate(file resize)", errno);
  }
  return Status::Ok();
}
Status preallocate_raw(std::intptr_t raw,
                       std::uint64_t minimum_size) noexcept {
  if (raw == FileHandle::kInvalid) {
    return Status::FailedPrecondition("file preallocate: handle is closed");
  }
  if (minimum_size > static_cast<std::uint64_t>(
                         (std::numeric_limits<std::int64_t>::max)())) {
    return Status::OutOfRange("file preallocate exceeds native offset range");
  }
  auto current = size_raw(raw);
  if (!current) return std::move(current).error();
  if (*current >= minimum_size) return Status::Ok();

  static_assert(sizeof(off_t) >= sizeof(std::int64_t));
  const off_t length = static_cast<off_t>(minimum_size - *current);
#if HYLE_OS_MACOS
  fstore_t store{};
  store.fst_flags = F_ALLOCATECONTIG;
  store.fst_posmode = F_PEOFPOSMODE;
  store.fst_offset = 0;
  store.fst_length = length;
  if (detail::retry_eintr_negative([&]() noexcept {
        return ::fcntl(static_cast<int>(raw), F_PREALLOCATE, &store);
      }) != 0) {
    store.fst_flags = F_ALLOCATEALL;
    if (detail::retry_eintr_negative([&]() noexcept {
          return ::fcntl(static_cast<int>(raw), F_PREALLOCATE, &store);
        }) != 0) {
      return posix_fs_status("fcntl(F_PREALLOCATE)", errno);
    }
  }
  if (detail::retry_eintr_negative([&]() noexcept {
        return ::ftruncate(static_cast<int>(raw),
                           static_cast<off_t>(minimum_size));
      }) != 0) {
    return posix_fs_status("ftruncate(file preallocate)", errno);
  }
#else
  const off_t begin = static_cast<off_t>(*current);
  const int allocated = detail::retry_eintr_error_code([&]() noexcept {
    return ::posix_fallocate(static_cast<int>(raw), begin, length);
  });
  if (allocated != 0) {
    return posix_fs_status("posix_fallocate", allocated);
  }
#endif
  return Status::Ok();
}
Status lock_raw(std::intptr_t raw, FileLockMode mode) noexcept {
  if (raw == FileHandle::kInvalid) {
    return Status::FailedPrecondition("file lock: handle is closed");
  }
  const int operation =
      mode == FileLockMode::kExclusive ? LOCK_EX : LOCK_SH;
  for (;;) {
    if (::flock(static_cast<int>(raw), operation) == 0) return Status::Ok();
    if (errno != EINTR) return posix_fs_status("flock", errno);
  }
}
Result<bool> try_lock_raw(std::intptr_t raw, FileLockMode mode) noexcept {
  if (raw == FileHandle::kInvalid) {
    return std::unexpected(
        Status::FailedPrecondition("file try lock: handle is closed"));
  }
  const int operation =
      (mode == FileLockMode::kExclusive ? LOCK_EX : LOCK_SH) | LOCK_NB;
  for (;;) {
    if (::flock(static_cast<int>(raw), operation) == 0) return true;
    const int code = errno;
    if (code == EINTR) continue;
    if (code == EWOULDBLOCK || code == EAGAIN) return false;
    return std::unexpected(posix_fs_status("flock(try)", code));
  }
}
Status unlock_raw(std::intptr_t raw) noexcept {
  if (raw == FileHandle::kInvalid) {
    return Status::FailedPrecondition("file unlock: handle is closed");
  }
  for (;;) {
    if (::flock(static_cast<int>(raw), LOCK_UN) == 0) return Status::Ok();
    if (errno != EINTR) return posix_fs_status("flock(unlock)", errno);
  }
}
}  // namespace

Result<FileHandle> open_file(const char* path, unsigned flags) noexcept {
  constexpr unsigned kKnown = kOpenRead | kOpenWrite | kOpenCreate |
                              kOpenTruncate | kOpenAppend;
  if (path == nullptr || path[0] == '\0' || (flags & ~kKnown) != 0 ||
      (flags & (kOpenRead | kOpenWrite)) == 0 ||
      ((flags & (kOpenCreate | kOpenTruncate | kOpenAppend)) != 0 &&
       (flags & kOpenWrite) == 0)) {
    return std::unexpected(
        Status::InvalidArgument("open file: invalid path or flag combination"));
  }
  int f;
  const bool rd = (flags & kOpenRead) != 0, wr = (flags & kOpenWrite) != 0;
  if (rd && wr) f = O_RDWR;
  else if (wr) f = O_WRONLY;
  else f = O_RDONLY;
  if (flags & kOpenCreate) f |= O_CREAT;
  if (flags & kOpenTruncate) f |= O_TRUNC;
  if (flags & kOpenAppend) f |= O_APPEND;
  f |= O_CLOEXEC;
  int fd = detail::retry_eintr_negative([&]() noexcept {
    return (flags & kOpenCreate) ? ::open(path, f, 0644) : ::open(path, f);
  });
  if (fd < 0) return std::unexpected(posix_fs_status("open", errno));
  return FileHandle{static_cast<std::intptr_t>(fd)};
}

Result<std::uint32_t> get_permissions(const char* path) noexcept {
  if (path == nullptr) {
    return std::unexpected(Status::InvalidArgument(
        "get permissions: path is null"));
  }
  struct ::stat s;
  if (detail::retry_eintr_negative(
          [&]() noexcept { return ::stat(path, &s); }) != 0) {
    return std::unexpected(posix_fs_status("stat permissions", errno));
  }
  return static_cast<std::uint32_t>(s.st_mode & 0777);
}

Status set_permissions(const char* path, std::uint32_t mode) noexcept {
  if (path == nullptr) {
    return Status::InvalidArgument("set permissions: path is null");
  }
  if (::chmod(path, static_cast<mode_t>(mode & 0777)) == 0) {
    return Status::Ok();
  }
  return posix_fs_status("chmod", errno);
}

Result<bool> is_readonly(const char* path) noexcept {
  HYLE_ASSIGN_OR_RETURN(std::uint32_t mode, get_permissions(path));
  return (mode & S_IWUSR) == 0;
}

Status set_readonly(const char* path, bool readonly) noexcept {
  if (path == nullptr) {
    return Status::InvalidArgument("set readonly: path is null");
  }
  struct ::stat s;
  if (detail::retry_eintr_negative(
          [&]() noexcept { return ::stat(path, &s); }) != 0) {
    return posix_fs_status("stat readonly", errno);
  }
  mode_t m = readonly ? (s.st_mode & ~static_cast<mode_t>(0222))
                      : (s.st_mode | S_IWUSR);
  if (::chmod(path, m & 0777) == 0) return Status::Ok();
  return posix_fs_status("chmod readonly", errno);
}

Status create_symlink(const char* target, const char* link,
                      SymlinkKind kind) noexcept {
  if (target == nullptr || link == nullptr ||
      (kind != SymlinkKind::kAuto && kind != SymlinkKind::kFile &&
       kind != SymlinkKind::kDirectory)) {
    return Status::InvalidArgument("create symlink: invalid path or kind");
  }
  if (::symlink(target, link) == 0) return Status::Ok();
  return posix_fs_status("symlink", errno);
}

Status create_hard_link(const char* target, const char* link) noexcept {
  if (target == nullptr || link == nullptr) {
    return Status::InvalidArgument("create hard link: path is null");
  }
  if (::link(target, link) == 0) return Status::Ok();
  return posix_fs_status("link", errno);
}

Result<bool> is_symlink(const char* path) noexcept {
  if (path == nullptr) {
    return std::unexpected(Status::InvalidArgument("is symlink: path is null"));
  }
  struct ::stat ls;
  if (detail::retry_eintr_negative(
          [&]() noexcept { return ::lstat(path, &ls); }) == 0) {
    return S_ISLNK(ls.st_mode);
  }
  if (errno == ENOENT || errno == ENOTDIR) return false;
  return std::unexpected(posix_fs_status("lstat symlink", errno));
}

Result<std::size_t> read_symlink(const char* path, char* dst,
                                 std::size_t cap) noexcept {
  if (path == nullptr) {
    return std::unexpected(Status::InvalidArgument("read symlink: path is null"));
  }
  char stack[256];
  char* buffer = stack;
  std::size_t buffer_capacity = sizeof(stack);
  MallocArray<char> heap;
  for (;;) {
    std::ptrdiff_t raw_result;
    int code = 0;
#if defined(HYLE_TESTING)
    if (detail::g_read_symlink_function_for_testing != nullptr) {
      raw_result = detail::g_read_symlink_function_for_testing(
          path, buffer, buffer_capacity, &code);
    } else
#endif
    {
      ssize_t result;
      do {
        result = ::readlink(path, buffer, buffer_capacity);
      } while (result < 0 && errno == EINTR);
      raw_result = static_cast<std::ptrdiff_t>(result);
      if (result < 0) code = errno;
    }

    if (raw_result < 0) {
      if (code == EINVAL) {
        return std::unexpected(
            Status::FailedPrecondition("read symlink: path is not a symlink"));
      }
      return std::unexpected(posix_fs_status("readlink", code));
    }
    const std::size_t length = static_cast<std::size_t>(raw_result);
    if (length < buffer_capacity) {
      return fill(dst, cap, buffer, length);
    }

    // readlink does not append NUL and n==capacity is indistinguishable from a
    // truncated result. Grow and retry until one call leaves spare capacity.
    if (buffer_capacity >
        (std::numeric_limits<std::size_t>::max)() / 2) {
      return std::unexpected(
          Status::OutOfRange("read symlink: target is too long"));
    }
    buffer_capacity *= 2;
    if (!heap.resize(buffer_capacity)) {
      return std::unexpected(
          heap.overflowed()
              ? Status::OutOfRange("read symlink: target is too long")
              : Status::ResourceExhausted(
                    "read symlink: target buffer allocation failed"));
    }
    buffer = heap.data();
  }
}

Result<CreatedTempFile> create_temp_file(const char* dir, const char* prefix,
                                         char* dst, std::size_t cap) noexcept {
  clear_output(dst, cap);
  if (!valid_temp_prefix(prefix)) {
    return std::unexpected(Status::InvalidArgument(
        "create temp file: prefix must be one filename fragment"));
  }
  TempDirectoryInput directory;
  Status directory_status = directory.resolve(dir);
  if (!directory_status.ok())
    return std::unexpected(std::move(directory_status));
  dir = directory.path();

  const char* p = prefix != nullptr ? prefix : "tmp";
  // Build "<dir>/<prefix>XXXXXX" (mkstemp replaces the 6 X's atomically).
  const std::size_t dl = std::strlen(dir);
  const std::size_t pl = std::strlen(p);
  const bool needs_separator = dir[dl - 1] != '/';
  std::size_t path_chars = dl;
  if ((needs_separator && !checked_size_add(path_chars, 1, path_chars)) ||
      !checked_size_add(path_chars, pl, path_chars) ||
      !checked_size_add(path_chars, 6, path_chars) ||
      !checked_size_add(path_chars, 1, path_chars)) {
    return std::unexpected(
        Status::OutOfRange("create temp file: generated path is too long"));
  }
  MallocArray<char> tmpl;
  if (!tmpl.resize(path_chars)) {
    return std::unexpected(
        tmpl.overflowed()
            ? Status::OutOfRange(
                  "create temp file: generated path is too long")
            : Status::ResourceExhausted(
                  "create temp file: path allocation failed"));
  }
  std::memcpy(tmpl.data(), dir, dl);
  std::size_t w = dl;
  if (needs_separator) tmpl.data()[w++] = '/';
  std::memcpy(tmpl.data() + w, p, pl);
  w += pl;
  std::memcpy(tmpl.data() + w, "XXXXXX", 6);
  w += 6;
  tmpl.data()[w] = '\0';
#if HYLE_OS_LINUX
  // Keep the descriptor non-inheritable from the instant it is created. A
  // later fcntl leaves a fork+exec race in multi-threaded programs.
  int fd = ::mkostemp(tmpl.data(), O_CLOEXEC);
  if (fd < 0) return std::unexpected(posix_fs_status("mkostemp", errno));
#else
  int fd = ::mkstemp(tmpl.data());
  if (fd < 0) return std::unexpected(posix_fs_status("mkstemp", errno));
  // macOS has no mkostemp. Close the inheritance window as soon as possible,
  // and remove the newly-created file if descriptor setup fails.
  int descriptor_flags;
  descriptor_flags = detail::retry_eintr_negative(
      [&]() noexcept { return ::fcntl(fd, F_GETFD); });
  int close_on_exec_result = -1;
  if (descriptor_flags >= 0) {
    close_on_exec_result = detail::retry_eintr_negative([&]() noexcept {
      return ::fcntl(fd, F_SETFD, descriptor_flags | FD_CLOEXEC);
    });
  }
  if (descriptor_flags < 0 || close_on_exec_result < 0) {
    const int code = errno;
    ::close(fd);
    ::unlink(tmpl.data());
    return std::unexpected(
        posix_fs_status("fcntl temp file close-on-exec", code));
  }
#endif
  const std::size_t path_size = std::strlen(tmpl.data());
  fill(dst, cap, tmpl.data(), path_size);
  return CreatedTempFile{FileHandle{static_cast<std::intptr_t>(fd)}, path_size};
}

Result<bool> directory_iterate(const char* path,
                               bool (*visit)(const char*, FileType, void*),
                               void* ctx) noexcept {
  if (path == nullptr || visit == nullptr) {
    return std::unexpected(
        Status::InvalidArgument("directory iterate: invalid argument"));
  }
  DIR* d = detail::retry_eintr_null(
      [&]() noexcept { return ::opendir(path); });
  if (d == nullptr) {
    return std::unexpected(posix_fs_status("opendir", errno));
  }
  bool stopped = false;
  Status status = Status::Ok();
  for (;;) {
    struct dirent* de = detail::retry_eintr_null(
        [&]() noexcept { return ::readdir(d); });
    if (de == nullptr) {
      if (errno != 0) status = posix_fs_status("readdir", errno);
      break;
    }
    const char* name = de->d_name;
    if (name[0] == '.' &&
        (name[1] == '\0' || (name[1] == '.' && name[2] == '\0')))
      continue;
    // d_type may be DT_UNKNOWN on some filesystems; fall back to lstat then.
    FileType type = file_type_from_dirent(de->d_type);
    if (type == FileType::kUnknown) {
      // lstat, NOT stat: report the entry itself and never follow a symlink.
      char child[4096];
      struct ::stat st;
      if (join_path(child, sizeof(child), path, name) &&
          detail::retry_eintr_negative(
              [&]() noexcept { return ::lstat(child, &st); }) == 0) {
        type = file_type_from_mode(st.st_mode);
      }
    }
    if (!visit(name, type, ctx)) {
      stopped = true;
      break;
    }
  }
  if (::closedir(d) != 0 && status.ok()) {
    status = posix_fs_status("closedir", errno);
  }
  if (!status.ok()) return std::unexpected(std::move(status));
  return !stopped;
}

#endif

// ---- portable (built on the primitives above) -------------------------------

Result<RandomAccessFile> open_random_access_file(const char* path,
                                                  unsigned flags) noexcept {
  if ((flags & kOpenAppend) != 0) {
    return std::unexpected(Status::InvalidArgument(
        "open random access file: append conflicts with explicit offsets"));
  }
  auto opened = open_file(path, flags);
  if (!opened) {
    return std::unexpected(std::move(opened).error());
  }
  return RandomAccessFile{std::move(*opened)};
}

// FileHandle: platform-agnostic RAII; close_raw() is the per-OS close (defined in
// each section above, declared at the top of the file).
FileHandle::~FileHandle() { (void)close_raw(raw_); }

FileHandle& FileHandle::operator=(FileHandle&& o) noexcept {
  if (this != &o) {
    (void)close_raw(raw_);
    raw_ = o.raw_;
    o.raw_ = kInvalid;
  }
  return *this;
}

Status FileHandle::close() noexcept {
  Status status = close_raw(raw_);
  raw_ = kInvalid;
  return status;
}

Result<FileHandle> FileHandle::duplicate() const noexcept {
  auto copy = duplicate_raw(raw_);
  if (!copy) return std::unexpected(std::move(copy).error());
  return FileHandle{*copy};
}

Result<FileInfo> FileHandle::info() const noexcept {
  return info_raw(raw_);
}

Result<std::size_t> FileHandle::read(void* buf, std::size_t n) noexcept {
  if (n != 0 && buf == nullptr) {
    return std::unexpected(
        Status::InvalidArgument("file read: null buffer for nonzero length"));
  }
  return read_raw(raw_, buf, n);
}
Result<std::size_t> FileHandle::write(const void* buf, std::size_t n) noexcept {
  if (n != 0 && buf == nullptr) {
    return std::unexpected(
        Status::InvalidArgument("file write: null buffer for nonzero length"));
  }
  return write_raw(raw_, buf, n);
}

Status FileHandle::read_exact(void* buf, std::size_t n) noexcept {
  if (n != 0 && buf == nullptr) {
    return Status::InvalidArgument(
        "file read exact: null buffer for nonzero length");
  }
  auto* bytes = static_cast<std::byte*>(buf);
  std::size_t done = 0;
  while (done != n) {
    auto moved = read_raw(raw_, bytes + done, n - done);
    if (!moved) {
      return std::move(moved).error().WithContext("file read exact");
    }
    if (*moved == 0) {
      return Status::DataLoss("file read exact: unexpected end of file");
    }
    done += *moved;
  }
  return Status::Ok();
}

Status FileHandle::write_all(const void* buf, std::size_t n) noexcept {
  if (n != 0 && buf == nullptr) {
    return Status::InvalidArgument(
        "file write all: null buffer for nonzero length");
  }
  const auto* bytes = static_cast<const std::byte*>(buf);
  std::size_t done = 0;
  while (done != n) {
    auto moved = write_raw(raw_, bytes + done, n - done);
    if (!moved) {
      return std::move(moved).error().WithContext("file write all");
    }
    if (*moved == 0) {
      return Status::DataLoss("file write all: write made no progress");
    }
    done += *moved;
  }
  return Status::Ok();
}

Result<std::uint64_t> FileHandle::seek(std::int64_t offset,
                                       SeekOrigin origin) noexcept {
  return seek_raw(raw_, offset, origin);
}

Status FileHandle::sync(bool data_only) noexcept {
  return sync_raw(raw_, data_only);
}

Result<std::uint64_t> FileHandle::size() const noexcept {
  return size_raw(raw_);
}

Status FileHandle::resize(std::uint64_t size) noexcept {
  return resize_raw(raw_, size);
}

Status FileHandle::preallocate(std::uint64_t minimum_size) noexcept {
  return preallocate_raw(raw_, minimum_size);
}

Status FileHandle::lock(FileLockMode mode) noexcept {
  if (mode != FileLockMode::kShared && mode != FileLockMode::kExclusive) {
    return Status::InvalidArgument("file lock: invalid mode");
  }
  return lock_raw(raw_, mode);
}

Result<bool> FileHandle::try_lock(FileLockMode mode) noexcept {
  if (mode != FileLockMode::kShared && mode != FileLockMode::kExclusive) {
    return std::unexpected(Status::InvalidArgument(
        "file try lock: invalid mode"));
  }
  return try_lock_raw(raw_, mode);
}

Status FileHandle::unlock() noexcept {
  return unlock_raw(raw_);
}

Result<RandomAccessFile> RandomAccessFile::duplicate() const noexcept {
  auto copy = file_.duplicate();
  if (!copy) return std::unexpected(std::move(copy).error());
  return RandomAccessFile{std::move(*copy)};
}

Result<std::size_t> RandomAccessFile::read_at(
    std::uint64_t offset, void* buf, std::size_t n) noexcept {
  if (n != 0 && buf == nullptr) {
    return std::unexpected(Status::InvalidArgument(
        "file read at: null buffer for nonzero length"));
  }
  return read_at_raw(file_.get(), offset, buf, n);
}

Result<std::size_t> RandomAccessFile::write_at(
    std::uint64_t offset, const void* buf, std::size_t n) noexcept {
  if (n != 0 && buf == nullptr) {
    return std::unexpected(Status::InvalidArgument(
        "file write at: null buffer for nonzero length"));
  }
  return write_at_raw(file_.get(), offset, buf, n);
}

Status RandomAccessFile::read_exact_at(
    std::uint64_t offset, void* buf, std::size_t n) noexcept {
  if (n != 0 && buf == nullptr) {
    return Status::InvalidArgument(
        "file read exact at: null buffer for nonzero length");
  }
  auto* bytes = static_cast<std::byte*>(buf);
  std::size_t done = 0;
  while (done != n) {
    std::uint64_t current = 0;
    if (!checked_offset_add(offset, done, current)) {
      return Status::OutOfRange("file read exact at: offset overflow");
    }
    auto moved = read_at_raw(file_.get(), current, bytes + done, n - done);
    if (!moved) {
      return std::move(moved).error().WithContext("file read exact at");
    }
    if (*moved == 0) {
      return Status::DataLoss("file read exact at: unexpected end of file");
    }
    done += *moved;
  }
  return Status::Ok();
}

Status RandomAccessFile::write_all_at(
    std::uint64_t offset, const void* buf, std::size_t n) noexcept {
  if (n != 0 && buf == nullptr) {
    return Status::InvalidArgument(
        "file write all at: null buffer for nonzero length");
  }
  const auto* bytes = static_cast<const std::byte*>(buf);
  std::size_t done = 0;
  while (done != n) {
    std::uint64_t current = 0;
    if (!checked_offset_add(offset, done, current)) {
      return Status::OutOfRange("file write all at: offset overflow");
    }
    auto moved = write_at_raw(file_.get(), current, bytes + done, n - done);
    if (!moved) {
      return std::move(moved).error().WithContext("file write all at");
    }
    if (*moved == 0) {
      return Status::DataLoss("file write all at: write made no progress");
    }
    done += *moved;
  }
  return Status::Ok();
}

Result<bool> exists(const char* path) noexcept {
  auto info = stat(path);
  if (info) return true;
  if (info.error().IsNotFound()) return false;
  return std::unexpected(std::move(info).error());
}

Result<bool> is_directory(const char* path) noexcept {
  auto info = stat(path);
  if (info) return info->is_directory();
  if (info.error().IsNotFound()) return false;
  return std::unexpected(std::move(info).error());
}

Result<std::uint64_t> file_size(const char* path) noexcept {
  HYLE_ASSIGN_OR_RETURN(FileInfo info, stat(path));
  if (info.is_directory()) {
    return std::unexpected(
        Status::FailedPrecondition("file size: path is a directory"));
  }
  return info.size;
}

Status touch(const char* path) noexcept {
  if (path == nullptr) return Status::InvalidArgument("touch: path is null");
#if HYLE_OS_WINDOWS
  // CreateFileW with OPEN_ALWAYS creates the file if absent, opens it if present
  // -- no std::fopen (which is ANSI on Windows and would mis-decode a UTF-8
  // path). Then stamp mtime to now.
  WidePath wp(path);
  if (wp.w() == nullptr) {
    return Status::InvalidArgument("touch: path is not valid UTF-8");
  }
  HANDLE h = CreateFileW(wp.w(), FILE_WRITE_ATTRIBUTES, FILE_SHARE_READ, nullptr,
                         OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE) {
    return win32_fs_status("CreateFileW touch", ::GetLastError());
  }
  FILETIME now;
  GetSystemTimeAsFileTime(&now);
  Status status = Status::Ok();
  if (SetFileTime(h, nullptr, nullptr, &now) == 0) {
    status = win32_fs_status("SetFileTime", ::GetLastError());
  }
  if (CloseHandle(h) == 0) {
    status.Update(win32_fs_status("CloseHandle touch", ::GetLastError()));
  }
  return status;
#else
  std::FILE* f = std::fopen(path, "ab");  // append: create if absent, no truncate
  if (f == nullptr) return posix_fs_status("fopen touch", errno);
  if (std::fclose(f) != 0) return posix_fs_status("fclose touch", errno);
  // Bump mtime to now even when the file already existed and append wrote 0 bytes.
  if (::utimensat(AT_FDCWD, path, nullptr, 0) == 0) return Status::Ok();
  return posix_fs_status("utimensat touch", errno);
#endif
}

Status make_directory_recursive(const char* path) noexcept {
  if (path == nullptr || path[0] == '\0') {
    return Status::InvalidArgument("make directories: path is empty");
  }
  auto already_directory = is_directory(path);
  if (!already_directory) return std::move(already_directory).error();
  if (*already_directory) return Status::Ok();
  std::size_t n = std::strlen(path);
  std::size_t required = 0;
  if (!checked_size_add(n, 1, required)) {
    return Status::OutOfRange("make directories: path size overflows");
  }
  char stack[4096];
  char* buf = stack;
  MallocArray<char> heap;
  if (required > sizeof(stack)) {
    if (!heap.resize(required)) {
      return heap.overflowed()
                 ? Status::OutOfRange("make directories: path is too long")
                 : Status::ResourceExhausted(
                       "make directories: path allocation failed");
    }
    buf = heap.data();
  }
  std::memcpy(buf, path, n + 1);
  // Create each ancestor in turn by temporarily terminating at each separator.
  for (std::size_t i = 1; i < n; ++i) {
    if (buf[i] == '/' || buf[i] == '\\') {
      char sep = buf[i];
      buf[i] = '\0';
      if (buf[0] != '\0') {
        Status made = make_directory(buf);
        // Tolerate a drive root ("C:") or existing dir; only fail if not a dir.
        if (!made.ok() && std::strcmp(buf, ".") != 0 &&
            !(i == 2 && buf[1] == ':')) {
          return made;
        }
      }
      buf[i] = sep;
    }
  }
  return make_directory(buf);
}

namespace {
struct RmPath {
  char stack[4096];
  char* data = stack;
  std::size_t size = 0;
  std::size_t capacity = sizeof(stack);
};

Status rm_path_reserve(RmPath* path, std::size_t required) noexcept {
  if (required <= path->capacity) return Status::Ok();
  const std::size_t max_size = (std::numeric_limits<std::size_t>::max)();
  std::size_t grown_capacity = path->capacity;
  while (grown_capacity < required) {
    if (grown_capacity > max_size / 2) {
      grown_capacity = required;
      break;
    }
    grown_capacity *= 2;
  }
  char* grown = nullptr;
  if (path->data == path->stack) {
    grown = static_cast<char*>(std::malloc(grown_capacity));
    if (grown != nullptr) std::memcpy(grown, path->stack, path->size + 1);
  } else {
    grown = static_cast<char*>(std::realloc(path->data, grown_capacity));
  }
  if (grown == nullptr) {
    return Status::ResourceExhausted(
        "recursive remove: path buffer allocation failed");
  }
  path->data = grown;
  path->capacity = grown_capacity;
  return Status::Ok();
}

Status rm_path_set(RmPath* path, const char* value) noexcept {
  const std::size_t size = std::strlen(value);
  if (size == (std::numeric_limits<std::size_t>::max)()) {
    return Status::OutOfRange("recursive remove: root path size overflow");
  }
  HYLE_RETURN_IF_ERROR(rm_path_reserve(path, size + 1));
  std::memcpy(path->data, value, size + 1);
  path->size = size;
  return Status::Ok();
}

Status rm_path_append(RmPath* path, const char* name) noexcept {
  const std::size_t name_size = std::strlen(name);
  const bool separator =
      path->size != 0 && path->data[path->size - 1] != '/' &&
      path->data[path->size - 1] != '\\';
  const std::size_t separator_size = separator ? 1u : 0u;
  const std::size_t max_size = (std::numeric_limits<std::size_t>::max)();
  if (path->size > max_size - separator_size) {
    return Status::OutOfRange("recursive remove: child path size overflow");
  }
  const std::size_t prefix_size = path->size + separator_size;
  if (name_size >= max_size - prefix_size) {
    return Status::OutOfRange("recursive remove: child path size overflow");
  }
  const std::size_t child_size = prefix_size + name_size;
  HYLE_RETURN_IF_ERROR(rm_path_reserve(path, child_size + 1));
  std::size_t cursor = path->size;
  if (separator) path->data[cursor++] = '/';
  std::memcpy(path->data + cursor, name, name_size + 1);
  path->size = child_size;
  return Status::Ok();
}

void rm_path_truncate(RmPath* path, std::size_t size) noexcept {
  path->size = size;
  path->data[size] = '\0';
}

void rm_path_destroy(RmPath* path) noexcept {
  if (path->data != path->stack) std::free(path->data);
  path->data = path->stack;
  path->size = 0;
  path->capacity = sizeof(path->stack);
}

#if HYLE_OS_WINDOWS
struct RmFrame {
  HANDLE iterator = INVALID_HANDLE_VALUE;
  WIN32_FIND_DATAW entry{};
  std::size_t path_size = 0;
  bool first_pending = false;
};
#else
struct RmFrame {
  DIR* iterator = nullptr;
  std::size_t path_size = 0;
};
#endif

struct RmFrames {
  RmFrame* data = nullptr;
  std::size_t size = 0;
  std::size_t capacity = 0;
};

Status rm_frames_reserve(RmFrames* frames, std::size_t required) noexcept {
  if (required <= frames->capacity) return Status::Ok();
  const std::size_t max_size = (std::numeric_limits<std::size_t>::max)();
  std::size_t grown_capacity = frames->capacity == 0 ? 16 : frames->capacity;
  while (grown_capacity < required) {
    if (grown_capacity > max_size / 2) {
      grown_capacity = required;
      break;
    }
    grown_capacity *= 2;
  }
  if (grown_capacity > max_size / sizeof(RmFrame)) {
    return Status::OutOfRange("recursive remove: iterator stack size overflow");
  }
  void* grown =
      std::realloc(frames->data, grown_capacity * sizeof(RmFrame));
  if (grown == nullptr) {
    return Status::ResourceExhausted(
        "recursive remove: iterator stack allocation failed");
  }
  frames->data = static_cast<RmFrame*>(grown);
  frames->capacity = grown_capacity;
  return Status::Ok();
}

Status rm_open_frame(const char* path, std::size_t path_size,
                     RmFrame* frame) noexcept {
#if HYLE_OS_WINDOWS
  WidePath wide_path(path);
  if (wide_path.w() == nullptr) {
    return wide_path.allocation_failed()
               ? Status::ResourceExhausted(
                     "recursive remove: path conversion allocation failed")
               : Status::InvalidArgument(
                     "recursive remove: path is not valid UTF-8");
  }
  const std::size_t wide_size = std::wcslen(wide_path.w());
  const bool separator =
      wide_size != 0 && wide_path.w()[wide_size - 1] != L'\\' &&
      wide_path.w()[wide_size - 1] != L'/';
  const std::size_t pattern_size = wide_size + (separator ? 1u : 0u) + 2;
  wchar_t pattern_stack[MAX_PATH];
  wchar_t* pattern = pattern_stack;
  if (pattern_size > MAX_PATH) {
    if (pattern_size >
        (std::numeric_limits<std::size_t>::max)() / sizeof(wchar_t)) {
      return Status::OutOfRange(
          "recursive remove: search pattern size overflow");
    }
    pattern = static_cast<wchar_t*>(
        std::malloc(pattern_size * sizeof(wchar_t)));
    if (pattern == nullptr) {
      return Status::ResourceExhausted(
          "recursive remove: search pattern allocation failed");
    }
  }
  std::memcpy(pattern, wide_path.w(), wide_size * sizeof(wchar_t));
  std::size_t cursor = wide_size;
  if (separator) pattern[cursor++] = L'\\';
  pattern[cursor++] = L'*';
  pattern[cursor] = L'\0';
  HANDLE iterator = FindFirstFileExW(
      pattern, FindExInfoBasic, &frame->entry, FindExSearchNameMatch, nullptr,
      FIND_FIRST_EX_LARGE_FETCH);
  const DWORD code = iterator == INVALID_HANDLE_VALUE ? ::GetLastError() : 0;
  if (pattern != pattern_stack) std::free(pattern);
  if (iterator == INVALID_HANDLE_VALUE) {
    return win32_fs_status("FindFirstFileExW(recursive remove)", code);
  }
  frame->iterator = iterator;
  frame->path_size = path_size;
  frame->first_pending = true;
  return Status::Ok();
#else
  DIR* iterator = detail::retry_eintr_null(
      [&]() noexcept { return ::opendir(path); });
  if (iterator == nullptr) return posix_fs_status("opendir recursive remove", errno);
  frame->iterator = iterator;
  frame->path_size = path_size;
  return Status::Ok();
#endif
}

Status rm_close_frame(RmFrame* frame) noexcept {
#if HYLE_OS_WINDOWS
  if (frame->iterator == INVALID_HANDLE_VALUE) return Status::Ok();
  const BOOL closed = ::FindClose(frame->iterator);
  const DWORD code = closed ? 0 : ::GetLastError();
  frame->iterator = INVALID_HANDLE_VALUE;
  return closed ? Status::Ok()
                : win32_fs_status("FindClose(recursive remove)", code);
#else
  if (frame->iterator == nullptr) return Status::Ok();
  DIR* iterator = frame->iterator;
  frame->iterator = nullptr;
  if (::closedir(iterator) == 0) return Status::Ok();
  return posix_fs_status("closedir recursive remove", errno);
#endif
}

Status rm_next_entry(RmFrame* frame, char* name, std::size_t name_capacity,
                     FileType* type, bool* has_entry) noexcept {
  *has_entry = false;
#if HYLE_OS_WINDOWS
  for (;;) {
    if (frame->first_pending) {
      frame->first_pending = false;
    } else if (!::FindNextFileW(frame->iterator, &frame->entry)) {
      const DWORD code = ::GetLastError();
      return code == ERROR_NO_MORE_FILES
                 ? Status::Ok()
                 : win32_fs_status("FindNextFileW(recursive remove)", code);
    }
    const wchar_t* wide_name = frame->entry.cFileName;
    if (wide_name[0] == L'.' &&
        (wide_name[1] == L'\0' ||
         (wide_name[1] == L'.' && wide_name[2] == L'\0'))) {
      continue;
    }
    const int converted = ::WideCharToMultiByte(
        CP_UTF8, WC_ERR_INVALID_CHARS, wide_name, -1, name,
        static_cast<int>(name_capacity), nullptr, nullptr);
    if (converted <= 0) {
      return win32_fs_status("WideCharToMultiByte(recursive remove entry)",
                             ::GetLastError());
    }
    *type = FileType::kRegular;
    if ((frame->entry.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0) {
      *type = FileType::kSymlink;
    } else if ((frame->entry.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) {
      *type = FileType::kDirectory;
    } else if ((frame->entry.dwFileAttributes & FILE_ATTRIBUTE_DEVICE) != 0) {
      *type = FileType::kCharacterDevice;
    }
    *has_entry = true;
    return Status::Ok();
  }
#else
  for (;;) {
    struct dirent* entry = detail::retry_eintr_null(
        [&]() noexcept { return ::readdir(frame->iterator); });
    if (entry == nullptr) {
      const int code = errno;
      return code == 0 ? Status::Ok()
                       : posix_fs_status("readdir recursive remove", code);
    }
    const char* entry_name = entry->d_name;
    if (entry_name[0] == '.' &&
        (entry_name[1] == '\0' ||
         (entry_name[1] == '.' && entry_name[2] == '\0'))) {
      continue;
    }
    const std::size_t name_size = std::strlen(entry_name);
    if (name_size >= name_capacity) {
      return Status::OutOfRange(
          "recursive remove: directory entry name is too long");
    }
    std::memcpy(name, entry_name, name_size + 1);
    *type = file_type_from_dirent(entry->d_type);
    if (*type == FileType::kUnknown) {
      struct ::stat metadata;
      if (detail::retry_eintr_negative([&]() noexcept {
            return ::fstatat(::dirfd(frame->iterator), entry_name, &metadata,
                             AT_SYMLINK_NOFOLLOW);
          }) != 0) {
        return posix_fs_status("fstatat recursive remove entry", errno);
      }
      *type = file_type_from_mode(metadata.st_mode);
    }
    *has_entry = true;
    return Status::Ok();
  }
#endif
}

Status rm_push_frame(RmFrames* frames, const char* path,
                     std::size_t path_size) noexcept {
  HYLE_RETURN_IF_ERROR(rm_frames_reserve(frames, frames->size + 1));
  RmFrame* frame = &frames->data[frames->size];
  *frame = RmFrame{};
  HYLE_RETURN_IF_ERROR(rm_open_frame(path, path_size, frame));
  ++frames->size;
  return Status::Ok();
}

Status rm_finish(RmFrames* frames, RmPath* path, Status status) noexcept {
  while (frames->size != 0) {
    --frames->size;
    status.Update(rm_close_frame(&frames->data[frames->size]));
  }
  std::free(frames->data);
  frames->data = nullptr;
  frames->capacity = 0;
  rm_path_destroy(path);
  return status;
}
}  // namespace

Status remove_directory_recursive(const char* path) noexcept {
  if (path == nullptr || path[0] == '\0') {
    return Status::InvalidArgument("recursive remove: path is empty");
  }

  // Inspect the root itself before asking whether it is a directory. This is
  // the data-loss guard: is_directory follows links, while is_symlink/lstat
  // detects the link or junction that must be unlinked instead.
  HYLE_ASSIGN_OR_RETURN(bool root_link, is_symlink(path));
  if (root_link) return remove_symlink(path);
  HYLE_ASSIGN_OR_RETURN(bool root_directory, is_directory(path));
  if (!root_directory) return remove_file(path);

  RmPath current_path;
  RmFrames frames;
  Status status = rm_path_set(&current_path, path);
  if (status.ok()) {
    status = rm_push_frame(&frames, current_path.data, current_path.size);
  }
  if (!status.ok()) return rm_finish(&frames, &current_path, std::move(status));

  // Resumable depth-first postorder traversal. One native iterator remains open
  // per ancestor and one mutable path buffer names the current entry. Siblings
  // are never queued, so auxiliary memory is O(depth + longest path), not width.
  char entry_name[4096];
  while (frames.size != 0) {
    RmFrame* frame = &frames.data[frames.size - 1];
    rm_path_truncate(&current_path, frame->path_size);

    FileType type = FileType::kUnknown;
    bool has_entry = false;
    status = rm_next_entry(frame, entry_name, sizeof(entry_name), &type,
                           &has_entry);
    if (!status.ok()) {
      return rm_finish(&frames, &current_path, std::move(status));
    }
    if (!has_entry) {
      status = rm_close_frame(frame);
      --frames.size;
      if (status.ok()) status = remove_directory(current_path.data);
      if (!status.ok()) {
        return rm_finish(&frames, &current_path, std::move(status));
      }
      continue;
    }

    const std::size_t parent_size = frame->path_size;
    status = rm_path_append(&current_path, entry_name);
    if (!status.ok()) {
      return rm_finish(&frames, &current_path, std::move(status));
    }

    if (type == FileType::kSymlink) {
      status = remove_symlink(current_path.data);
    } else if (type == FileType::kDirectory) {
      // Re-check immediately before descent. This preserves the link check and
      // reduces stale-type risk under the API's caller-no-mutation contract.
      auto link = is_symlink(current_path.data);
      if (!link) {
        status = std::move(link).error();
      } else if (*link) {
        status = remove_symlink(current_path.data);
      } else {
        auto directory = is_directory(current_path.data);
        if (!directory) {
          status = std::move(directory).error();
        } else if (!*directory) {
          status = remove_file(current_path.data);
        } else {
          status =
              rm_push_frame(&frames, current_path.data, current_path.size);
          if (status.ok()) continue;
        }
      }
    } else {
      status = remove_file(current_path.data);
    }

    rm_path_truncate(&current_path, parent_size);
    if (!status.ok()) {
      return rm_finish(&frames, &current_path, std::move(status));
    }
  }
  return rm_finish(&frames, &current_path, Status::Ok());
}

}  // namespace hyle::platform

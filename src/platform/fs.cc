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
#  define _CRT_RAND_S  // expose rand_s (RtlGenRandom-backed) for temp file names
#endif

#include "hyle/platform/fs.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "hyle/platform.h"  // HYLE_OS_* for the branches below (the _GNU_SOURCE/_CRT
                            // feature macros above MUST stay raw -- they precede ALL
                            // includes, before platform.h is parsed).

#if defined(HYLE_OS_WINDOWS)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <winioctl.h>  // FSCTL_GET_REPARSE_POINT for read_symlink
#  include <cwchar>  // std::wcslen for the directory_iterate pattern
#else
#  include <dirent.h>
#  include <fcntl.h>
#  include <pwd.h>
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

// Join "a/b" into dst[cap]. Returns false (no write) if it would not fit. Bounded
// and allocation-free -- used to rebuild child paths during recursion without
// snprintf (whose worst-case length trips GCC -Wformat-truncation under -Werror).
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

// Raw file-handle operations. Defined once per OS and shared by FileHandle.
Status close_raw(std::intptr_t raw) noexcept;
// Raw read/write on an fd/HANDLE (per-OS defs below), declared for the
// platform-agnostic FileHandle::read/write tail.
Result<std::size_t> read_raw(std::intptr_t raw, void* buf,
                             std::size_t n) noexcept;
Result<std::size_t> write_raw(std::intptr_t raw, const void* buf,
                              std::size_t n) noexcept;

}  // namespace

#if defined(HYLE_OS_WINDOWS)

// ---- Windows ----------------------------------------------------------------
//
// All path I/O goes through the wide (...W) API: the ...A variants interpret
// bytes as the active code page, NOT UTF-8, so a UTF-8 path with non-ASCII chars
// would be mis-decoded. We convert UTF-8 <-> UTF-16 at the boundary so callers
// pass UTF-8 everywhere (matching POSIX, which is byte-transparent).

namespace {

const char* win32_fs_error_name(DWORD code) noexcept {
  switch (code) {
    case ERROR_FILE_NOT_FOUND: return "ERROR_FILE_NOT_FOUND";
    case ERROR_PATH_NOT_FOUND: return "ERROR_PATH_NOT_FOUND";
    case ERROR_ACCESS_DENIED: return "ERROR_ACCESS_DENIED";
    case ERROR_PRIVILEGE_NOT_HELD: return "ERROR_PRIVILEGE_NOT_HELD";
    case ERROR_ALREADY_EXISTS: return "ERROR_ALREADY_EXISTS";
    case ERROR_FILE_EXISTS: return "ERROR_FILE_EXISTS";
    case ERROR_DISK_FULL: return "ERROR_DISK_FULL";
    case ERROR_NOT_ENOUGH_MEMORY: return "ERROR_NOT_ENOUGH_MEMORY";
    case ERROR_INVALID_HANDLE: return "ERROR_INVALID_HANDLE";
    case ERROR_INVALID_PARAMETER: return "ERROR_INVALID_PARAMETER";
    case ERROR_FILENAME_EXCED_RANGE: return "ERROR_FILENAME_EXCED_RANGE";
    default: return "WIN32_ERROR";
  }
}

Status win32_fs_status(const char* operation, DWORD code) noexcept {
  char description[192]{};
  DWORD length = ::FormatMessageA(
      FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, code,
      0, description, static_cast<DWORD>(sizeof(description)), nullptr);
  while (length != 0 &&
         (description[length - 1] == '\r' || description[length - 1] == '\n')) {
    description[--length] = '\0';
  }
  char message[320];
  std::snprintf(message, sizeof(message), "windows %s failed: %s (%lu)%s%s",
                operation, win32_fs_error_name(code),
                static_cast<unsigned long>(code), length == 0 ? "" : ": ",
                length == 0 ? "" : description);
  if (code == ERROR_FILE_NOT_FOUND || code == ERROR_PATH_NOT_FOUND) {
    return Status::NotFound(message);
  }
  if (code == ERROR_ACCESS_DENIED || code == ERROR_PRIVILEGE_NOT_HELD ||
      code == ERROR_SHARING_VIOLATION || code == ERROR_WRITE_PROTECT) {
    return Status::PermissionDenied(message);
  }
  if (code == ERROR_ALREADY_EXISTS || code == ERROR_FILE_EXISTS) {
    return Status::AlreadyExists(message);
  }
  if (code == ERROR_DISK_FULL || code == ERROR_HANDLE_DISK_FULL ||
      code == ERROR_NOT_ENOUGH_MEMORY || code == ERROR_OUTOFMEMORY ||
      code == ERROR_TOO_MANY_OPEN_FILES) {
    return Status::ResourceExhausted(message);
  }
  if (code == ERROR_INVALID_HANDLE) return Status::FailedPrecondition(message);
  if (code == ERROR_INVALID_PARAMETER || code == ERROR_INVALID_NAME) {
    return Status::InvalidArgument(message);
  }
  if (code == ERROR_FILENAME_EXCED_RANGE) return Status::OutOfRange(message);
  if (code == ERROR_NOT_SUPPORTED || code == ERROR_CALL_NOT_IMPLEMENTED) {
    return Status::Unimplemented(message);
  }
  return Status::Unavailable(message);
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
  explicit WidePath(const char* utf8) noexcept {
    if (utf8 == nullptr) return;
    int need = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, nullptr, 0);
    if (need <= 0) return;  // includes the NUL
    int wlen = need - 1;

    if (wlen >= kLongThreshold) {
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
      }
      // not absolute (or build failed): fall through to a plain conversion.
    }

    wchar_t* dst = stack_;
    if (static_cast<std::size_t>(need) > kStack) {
      heap_ = static_cast<wchar_t*>(std::malloc(need * sizeof(wchar_t)));
      if (heap_ == nullptr) return;
      dst = heap_;
    }
    if (MultiByteToWideChar(CP_UTF8, 0, utf8, -1, dst, need) <= 0) return;
    w_ = dst;
  }
  ~WidePath() { std::free(heap_); }
  WidePath(const WidePath&) = delete;
  WidePath& operator=(const WidePath&) = delete;

  const wchar_t* w() const noexcept { return w_; }

 private:
  // Build prefix + (body minus `skip` leading chars), slash-normalized to '\'.
  bool build_extended(const char* utf8, int need, const wchar_t* prefix,
                      int skip) noexcept {
    int plen = static_cast<int>(std::wcslen(prefix));
    int cap = plen + need;  // prefix + full body + NUL (skip only shrinks it)
    wchar_t* dst = stack_;
    if (static_cast<std::size_t>(cap) > kStack) {
      heap_ = static_cast<wchar_t*>(std::malloc(cap * sizeof(wchar_t)));
      if (heap_ == nullptr) return false;
      dst = heap_;
    }
    for (int i = 0; i < plen; ++i) dst[i] = prefix[i];
    if (MultiByteToWideChar(CP_UTF8, 0, utf8, -1, dst + plen, need) <= 0)
      return false;
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
};

// UTF-16 -> UTF-8 buffer-fill (snprintf semantics): write into dst[cap], always
// NUL-terminate when cap>0, return the FULL UTF-8 length (>= cap => truncated).
// `wlen` is the UTF-16 length excluding any terminator (-1 lets it scan).
std::size_t fill_w(char* dst, std::size_t cap, const wchar_t* src,
                   int wlen) noexcept {
  if (dst == nullptr) cap = 0;
  int need = WideCharToMultiByte(CP_UTF8, 0, src, wlen, nullptr, 0, nullptr,
                                 nullptr);
  if (need <= 0) return fill(dst, cap, "", 0);
  const std::size_t full =
      static_cast<std::size_t>(need - (wlen == -1 ? 1 : 0));
  if (cap == 0) return full;
  if (full < cap) {
    int got =
        WideCharToMultiByte(CP_UTF8, 0, src, wlen, dst, need, nullptr, nullptr);
    if (got <= 0) {
      dst[0] = '\0';
      return full;
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
    return full;
  }
  int got = WideCharToMultiByte(CP_UTF8, 0, src, wlen, tmp, need, nullptr,
                                nullptr);
  if (got <= 0) {
    std::free(tmp);
    dst[0] = '\0';
    return full;
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
  bool dir = (d.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
  r.exists = true;
  r.is_directory = dir;
  r.is_symlink = (d.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
  r.size = dir ? 0
               : (static_cast<std::uint64_t>(d.nFileSizeHigh) << 32) |
                     d.nFileSizeLow;
  // FILETIME is 100ns ticks since 1601; convert to seconds since 1970.
  std::uint64_t t = (static_cast<std::uint64_t>(d.ftLastWriteTime.dwHighDateTime)
                     << 32) |
                    d.ftLastWriteTime.dwLowDateTime;
  r.modified_time = static_cast<std::int64_t>(t / 10000000ull) - 11644473600ll;
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

std::size_t current_directory(char* dst, std::size_t cap) noexcept {
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
    if (heap == nullptr) return fill(dst, cap, "", 0);
    buf = heap;
    n = GetCurrentDirectoryW(buf_cap, buf);
  }
  std::size_t out =
      (n == 0 || n >= buf_cap)
          ? fill(dst, cap, "", 0)
          : fill_w(dst, cap, buf, static_cast<int>(n));
  std::free(heap);
  return out;
}

bool change_directory(const char* path) noexcept {
  WidePath wp(path);
  return wp.w() != nullptr && SetCurrentDirectoryW(wp.w()) != 0;
}

std::size_t temp_directory(char* dst, std::size_t cap) noexcept {
  wchar_t stack[512];
  wchar_t* buf = stack;
  wchar_t* heap = nullptr;
  DWORD buf_cap = static_cast<DWORD>(sizeof(stack) / sizeof(stack[0]));
  DWORD n = GetTempPathW(buf_cap, buf);  // already honors TMP/TEMP
  if (n >= buf_cap) {
    buf_cap = n + 1;
    heap = static_cast<wchar_t*>(
        std::malloc(static_cast<std::size_t>(buf_cap) * sizeof(wchar_t)));
    if (heap == nullptr) return fill(dst, cap, "", 0);
    buf = heap;
    n = GetTempPathW(buf_cap, buf);
  }
  if (n == 0 || n >= buf_cap) {
    std::free(heap);
    return fill(dst, cap, "", 0);
  }
  while (n > 1 && (buf[n - 1] == L'\\' || buf[n - 1] == L'/')) --n;  // trim sep
  std::size_t out = fill_w(dst, cap, buf, static_cast<int>(n));
  std::free(heap);
  return out;
}

std::size_t home_directory(char* dst, std::size_t cap) noexcept {
  wchar_t stack[512];
  wchar_t* buf = stack;
  wchar_t* heap = nullptr;
  DWORD buf_cap = static_cast<DWORD>(sizeof(stack) / sizeof(stack[0]));
  DWORD n = GetEnvironmentVariableW(L"USERPROFILE", buf, buf_cap);
  if (n >= buf_cap) {
    buf_cap = n;
    heap = static_cast<wchar_t*>(
        std::malloc(static_cast<std::size_t>(buf_cap) * sizeof(wchar_t)));
    if (heap == nullptr) return fill(dst, cap, "", 0);
    buf = heap;
    n = GetEnvironmentVariableW(L"USERPROFILE", buf, buf_cap);
  }
  std::size_t out =
      (n == 0 || n >= buf_cap)
          ? fill(dst, cap, "", 0)
          : fill_w(dst, cap, buf, static_cast<int>(n));
  std::free(heap);
  return out;
}

std::size_t get_env_var(const char* name, char* dst, std::size_t cap) noexcept {
  if (name == nullptr) return kEnvNotSet;
  WidePath wn(name);
  if (wn.w() == nullptr) return kEnvNotSet;
  // Query length first. 0 with ERROR_ENVVAR_NOT_FOUND => unset (vs set-empty).
  DWORD need = GetEnvironmentVariableW(wn.w(), nullptr, 0);
  if (need == 0) {
    if (GetLastError() == ERROR_ENVVAR_NOT_FOUND) return kEnvNotSet;
    return fill(dst, cap, "", 0);  // set but empty
  }
  wchar_t stackw[512];
  wchar_t* wbuf = stackw;
  wchar_t* heapw = nullptr;
  if (need > 512) {
    heapw = static_cast<wchar_t*>(std::malloc(need * sizeof(wchar_t)));
    if (heapw == nullptr) return fill(dst, cap, "", 0);
    wbuf = heapw;
  }
  DWORD got = GetEnvironmentVariableW(wn.w(), wbuf, need);
  // Race guard: if another thread grew the var between the size probe and here,
  // got >= need and wbuf is NOT filled -> return empty rather than read past need.
  std::size_t r = (got != 0 && got < need)
                      ? fill_w(dst, cap, wbuf, static_cast<int>(got))
                      : fill(dst, cap, "", 0);
  std::free(heapw);
  return r;
}

Result<bool> directory_iterate(const char* path,
                               bool (*visit)(const char*, bool, void*),
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
    // is NUL-terminated and `got` includes it. Fall back to fill_w only on the
    // impossible failure (0 = bad input / would-truncate), keeping correctness.
    int got = WideCharToMultiByte(CP_UTF8, 0, wname, -1, name,
                                  static_cast<int>(sizeof(name)), nullptr, nullptr);
    if (got <= 0) fill_w(name, sizeof(name), wname, -1);  // defensive, never hit here
    bool dir = (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
    if (!visit(name, dir, ctx)) {
      stopped = true;
      break;
    }
  } while (FindNextFileW(h, &fd));
  const DWORD end_error = stopped ? ERROR_NO_MORE_FILES : GetLastError();
  const BOOL closed = FindClose(h);
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
}  // namespace

Result<FileHandle> open_file(const char* path, unsigned flags) noexcept {
  constexpr unsigned kKnown = kOpenRead | kOpenWrite | kOpenCreate |
                              kOpenTruncate | kOpenAppend;
  if (path == nullptr || (flags & ~kKnown) != 0 ||
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
  HANDLE h = CreateFileW(wp.w(), access, FILE_SHARE_READ | FILE_SHARE_WRITE,
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

Status create_symlink(const char* target, const char* link) noexcept {
  WidePath wt(target), wl(link);
  if (wt.w() == nullptr || wl.w() == nullptr) {
    return Status::InvalidArgument(
        "create symlink: path is null or not valid UTF-8");
  }
  // Flag bit is target type; add UNPRIVILEGED_CREATE so Developer Mode suffices.
  DWORD flags = SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE;
  auto target_is_directory = is_directory(target);
  if (!target_is_directory) return std::move(target_is_directory).error();
  if (*target_is_directory) {
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
  auto info = stat(path);
  if (info) return info->is_symlink;
  if (info.error().IsNotFound()) return false;
  return std::unexpected(std::move(info).error());
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

Result<FileHandle> create_temp_file(const char* dir, const char* prefix,
                                    char* dst, std::size_t cap) noexcept {
  char dirbuf[1024];
  if (dir == nullptr) {
    std::size_t n = temp_directory(dirbuf, sizeof(dirbuf));
    if (n == 0 || n >= sizeof(dirbuf)) {
      return std::unexpected(
          Status::OutOfRange("create temp file: temp directory is too long"));
    }
    dir = dirbuf;
  }
  WidePath wd(dir);
  if (wd.w() == nullptr) {
    return std::unexpected(
        Status::InvalidArgument("create temp file: invalid UTF-8 directory"));
  }
  std::size_t dl = std::wcslen(wd.w());
  wchar_t wpre[16];
  const char* p = prefix != nullptr ? prefix : "tmp";
  int pn = MultiByteToWideChar(CP_UTF8, 0, p, -1, wpre, 16);
  if (pn <= 0) { wpre[0] = L't'; wpre[1] = L'\0'; }
  std::size_t pl = std::wcslen(wpre);

  // TRUE atomic create (mkstemp-equivalent): generate a high-entropy random name
  // and CreateFileW with CREATE_NEW, which fails if the name already exists. On
  // collision (ERROR_FILE_EXISTS) we retry a fresh name; any other error aborts.
  // No GetTempFileName (which pre-creates the file with a predictable 16-bit
  // counter, then needs a racy reopen and can leave an orphan). CREATE_NEW gives
  // us the file exclusively in one syscall -> no TOCTOU window, no orphan on fail.
  constexpr int kMaxTries = 128;
  const wchar_t kHex[] = L"0123456789abcdef";
  for (int attempt = 0; attempt < kMaxTries; ++attempt) {
    // dir + '\' + prefix + 16 hex + ".tmp" + NUL
    if (dl + 1 + pl + 16 + 4 + 1 > MAX_PATH) {
      return std::unexpected(
          Status::OutOfRange("create temp file: generated path is too long"));
    }
    wchar_t wpath[MAX_PATH];
    std::memcpy(wpath, wd.w(), dl * sizeof(wchar_t));
    std::size_t w = dl;
    if (w == 0 || (wpath[w - 1] != L'\\' && wpath[w - 1] != L'/')) wpath[w++] = L'\\';
    std::memcpy(wpath + w, wpre, pl * sizeof(wchar_t));
    w += pl;
    // 64 random bits -> 16 hex chars (rand_s is RtlGenRandom-backed, unpredictable).
    for (int word = 0; word < 2; ++word) {
      unsigned r = 0;
      if (rand_s(&r) != 0) r = static_cast<unsigned>(GetTickCount()) ^ (attempt << 16);
      for (int i = 0; i < 8; ++i) { wpath[w++] = kHex[r & 0xF]; r >>= 4; }
    }
    const wchar_t kExt[] = L".tmp";
    std::memcpy(wpath + w, kExt, sizeof(kExt));  // includes NUL
    w += 4;

    HANDLE h = CreateFileW(wpath, GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                           CREATE_NEW, FILE_ATTRIBUTE_TEMPORARY, nullptr);
    if (h != INVALID_HANDLE_VALUE) {
      char u8[MAX_PATH * 3];
      if (WideCharToMultiByte(CP_UTF8, 0, wpath, -1, u8, sizeof(u8), nullptr,
                              nullptr) > 0) {
        fill(dst, cap, u8, std::strlen(u8));
      }
      return FileHandle{reinterpret_cast<std::intptr_t>(h)};
    }
    const DWORD code = GetLastError();
    if (code != ERROR_FILE_EXISTS) {
      return std::unexpected(win32_fs_status("CreateFileW(CREATE_NEW)", code));
    }
    // else: name collided -> loop and try a fresh random name.
  }
  return std::unexpected(Status::AlreadyExists(
      "create temp file: exhausted unique-name retries"));
}

bool set_env_var(const char* name, const char* value) noexcept {
  return name != nullptr &&
         _putenv_s(name, value != nullptr ? value : "") == 0;
}

bool unset_env_var(const char* name) noexcept {
  return name != nullptr && _putenv_s(name, "") == 0;
}

#else

// ---- POSIX ------------------------------------------------------------------

namespace {

[[maybe_unused]] const char* strerror_text(int result, char* buffer) noexcept {
  return result == 0 ? buffer : nullptr;
}
[[maybe_unused]] const char* strerror_text(char* result, char*) noexcept {
  return result;
}

Status posix_fs_status(const char* operation, int code) noexcept {
  char description[192]{};
  const char* detail = strerror_text(
      ::strerror_r(code, description, sizeof(description)), description);
  char message[320];
  std::snprintf(message, sizeof(message), "posix %s failed: errno %d%s%s",
                operation, code,
                detail == nullptr || detail[0] == '\0' ? "" : ": ",
                detail == nullptr ? "" : detail);
  if (code == ENOENT || code == ENOTDIR) return Status::NotFound(message);
  if (code == EACCES || code == EPERM || code == EROFS) {
    return Status::PermissionDenied(message);
  }
  if (code == EEXIST) return Status::AlreadyExists(message);
  if (code == ENOMEM || code == ENOSPC || code == EDQUOT || code == EMFILE ||
      code == ENFILE) {
    return Status::ResourceExhausted(message);
  }
  if (code == EBADF) return Status::FailedPrecondition(message);
  if (code == EINVAL || code == EFAULT) return Status::InvalidArgument(message);
  if (code == ENAMETOOLONG || code == EFBIG || code == EOVERFLOW) {
    return Status::OutOfRange(message);
  }
  if (code == ENOSYS || code == ENOTSUP) return Status::Unimplemented(message);
  return Status::Unavailable(message);
}

}  // namespace

Result<FileInfo> stat(const char* path) noexcept {
  if (path == nullptr) {
    return std::unexpected(Status::InvalidArgument("stat: path is null"));
  }
  struct ::stat s;
  FileInfo r;
  if (::stat(path, &s) != 0) {
    return std::unexpected(posix_fs_status("stat", errno));
  }
  bool reg = S_ISREG(s.st_mode);
  r.is_directory = S_ISDIR(s.st_mode);
  r.exists = true;
  r.size = reg ? static_cast<std::uint64_t>(s.st_size) : 0;
  r.modified_time = static_cast<std::int64_t>(s.st_mtime);
  // stat() follows symlinks; an extra lstat tells us if the path ITSELF is a link
  // (a symlink to a dir still reports is_directory=true above, as callers expect).
  struct ::stat ls;
  if (::lstat(path, &ls) == 0 && S_ISLNK(ls.st_mode)) r.is_symlink = true;
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
  // Self-copy is a successful no-op. This string check is only a fast path:
  // distinct names may still be hard links to the same inode.
  if (std::strcmp(from, to) == 0) return Status::Ok();
  int in = ::open(from, O_RDONLY | O_CLOEXEC);
  if (in < 0) return posix_fs_status("open copy source", errno);
  struct ::stat in_stat;
  if (::fstat(in, &in_stat) != 0) {
    const int code = errno;
    ::close(in);
    return posix_fs_status("fstat copy source", code);
  }
  // Open without O_TRUNC, then compare the opened descriptors. Checking paths
  // before open has a TOCTOU race and misses hard-link aliases; truncating first
  // would destroy the source before we discover that both fds name one inode.
  int out =
      ::open(to, O_WRONLY | O_CREAT | O_CLOEXEC, in_stat.st_mode & 0777);
  if (out < 0) {
    const int code = errno;
    ::close(in);
    return posix_fs_status("open copy destination", code);
  }
  struct ::stat out_stat;
  if (::fstat(out, &out_stat) != 0) {
    const int code = errno;
    ::close(in);
    ::close(out);
    return posix_fs_status("fstat copy destination", code);
  }
  if (in_stat.st_dev == out_stat.st_dev &&
      in_stat.st_ino == out_stat.st_ino) {
    Status status = close_raw(in);
    status.Update(close_raw(out));
    return status;
  }
  if (::ftruncate(out, 0) != 0) {
    const int code = errno;
    ::close(in);
    ::close(out);
    return posix_fs_status("ftruncate copy destination", code);
  }

  Status status = Status::Ok();
#if defined(HYLE_OS_LINUX)
  // Fast path: copy_file_range keeps bytes in the kernel (and can reflink on
  // btrfs/xfs) -- no user-space bounce. Only fall back to read/write if it fails
  // on the FIRST call (EXDEV cross-fs / ENOSYS old kernel / EINVAL), i.e. before
  // any bytes are written, so the fallback can't double-copy.
  bool use_userspace = false;
  off_t remaining = in_stat.st_size;
  bool any_copied = false;
  while (remaining > 0) {
    ssize_t n = ::copy_file_range(in, nullptr, out, nullptr,
                                  static_cast<std::size_t>(remaining), 0);
    if (n < 0) {
      if (errno == EINTR) continue;
      if (!any_copied) {
        use_userspace = true;  // unsupported here; retry whole copy in user space
        break;
      }
      status = posix_fs_status("copy_file_range", errno);
      break;
    }
    if (n == 0) break;  // EOF
    any_copied = true;
    remaining -= n;
  }
  if (use_userspace) {
    ::lseek(in, 0, SEEK_SET);
    ::lseek(out, 0, SEEK_SET);
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

std::size_t current_directory(char* dst, std::size_t cap) noexcept {
  char buf[4096];
  if (::getcwd(buf, sizeof(buf)) == nullptr) return fill(dst, cap, "", 0);
  return fill(dst, cap, buf, std::strlen(buf));
}

bool change_directory(const char* path) noexcept {
  return ::chdir(path) == 0;
}

std::size_t temp_directory(char* dst, std::size_t cap) noexcept {
  const char* t = std::getenv("TMPDIR");
  if (t == nullptr || t[0] == '\0') t = std::getenv("TMP");
  if (t == nullptr || t[0] == '\0') t = std::getenv("TEMP");
  if (t == nullptr || t[0] == '\0') t = "/tmp";
  std::size_t n = std::strlen(t);
  while (n > 1 && t[n - 1] == '/') --n;  // trim trailing slash
  return fill(dst, cap, t, n);
}

std::size_t home_directory(char* dst, std::size_t cap) noexcept {
  const char* h = std::getenv("HOME");
  if (h != nullptr && h[0] != '\0') return fill(dst, cap, h, std::strlen(h));
  // getpwuid returns a pointer into shared static storage (not thread-safe);
  // getpwuid_r fills caller buffers instead.
  struct passwd pw;
  struct passwd* result = nullptr;
  char buf[1024];
  if (::getpwuid_r(::getuid(), &pw, buf, sizeof(buf), &result) == 0 &&
      result != nullptr && result->pw_dir != nullptr) {
    return fill(dst, cap, result->pw_dir, std::strlen(result->pw_dir));
  }
  return fill(dst, cap, "", 0);
}

std::size_t get_env_var(const char* name, char* dst, std::size_t cap) noexcept {
  if (name == nullptr) return kEnvNotSet;
  const char* v = std::getenv(name);
  if (v == nullptr) return kEnvNotSet;
  return fill(dst, cap, v, std::strlen(v));
}

namespace {
Status close_raw(std::intptr_t raw) noexcept {
  if (raw == FileHandle::kInvalid) return Status::Ok();
  if (::close(static_cast<int>(raw)) == 0) return Status::Ok();
  return posix_fs_status("close", errno);
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
}  // namespace

Result<FileHandle> open_file(const char* path, unsigned flags) noexcept {
  constexpr unsigned kKnown = kOpenRead | kOpenWrite | kOpenCreate |
                              kOpenTruncate | kOpenAppend;
  if (path == nullptr || (flags & ~kKnown) != 0 ||
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
  int fd = (flags & kOpenCreate) ? ::open(path, f, 0644) : ::open(path, f);
  if (fd < 0) return std::unexpected(posix_fs_status("open", errno));
  return FileHandle{static_cast<std::intptr_t>(fd)};
}

Result<std::uint32_t> get_permissions(const char* path) noexcept {
  if (path == nullptr) {
    return std::unexpected(Status::InvalidArgument(
        "get permissions: path is null"));
  }
  struct ::stat s;
  if (::stat(path, &s) != 0) {
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
  if (::stat(path, &s) != 0) {
    return posix_fs_status("stat readonly", errno);
  }
  mode_t m = readonly ? (s.st_mode & ~static_cast<mode_t>(0222))
                      : (s.st_mode | S_IWUSR);
  if (::chmod(path, m & 0777) == 0) return Status::Ok();
  return posix_fs_status("chmod readonly", errno);
}

Status create_symlink(const char* target, const char* link) noexcept {
  if (target == nullptr || link == nullptr) {
    return Status::InvalidArgument("create symlink: path is null");
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
  if (::lstat(path, &ls) == 0) return S_ISLNK(ls.st_mode);
  if (errno == ENOENT || errno == ENOTDIR) return false;
  return std::unexpected(posix_fs_status("lstat symlink", errno));
}

Result<std::size_t> read_symlink(const char* path, char* dst,
                                 std::size_t cap) noexcept {
  if (path == nullptr) {
    return std::unexpected(Status::InvalidArgument("read symlink: path is null"));
  }
  char buf[4096];
  ssize_t n = ::readlink(path, buf, sizeof(buf));
  if (n < 0) {
    const int code = errno;
    if (code == EINVAL) {
      return std::unexpected(
          Status::FailedPrecondition("read symlink: path is not a symlink"));
    }
    return std::unexpected(posix_fs_status("readlink", code));
  }
  return fill(dst, cap, buf, static_cast<std::size_t>(n));
}

Result<FileHandle> create_temp_file(const char* dir, const char* prefix,
                                    char* dst, std::size_t cap) noexcept {
  char dirbuf[1024];
  if (dir == nullptr) {
    std::size_t n = temp_directory(dirbuf, sizeof(dirbuf));
    if (n == 0 || n >= sizeof(dirbuf)) {
      return std::unexpected(
          Status::OutOfRange("create temp file: temp directory is too long"));
    }
    dir = dirbuf;
  }
  const char* p = prefix != nullptr ? prefix : "tmp";
  // Build "<dir>/<prefix>XXXXXX" (mkstemp replaces the 6 X's atomically).
  char tmpl[1024];
  std::size_t dl = std::strlen(dir), pl = std::strlen(p);
  if (dl + 1 + pl + 6 + 1 > sizeof(tmpl)) {
    return std::unexpected(
        Status::OutOfRange("create temp file: generated path is too long"));
  }
  std::memcpy(tmpl, dir, dl);
  std::size_t w = dl;
  if (w == 0 || tmpl[w - 1] != '/') tmpl[w++] = '/';
  std::memcpy(tmpl + w, p, pl);
  w += pl;
  std::memcpy(tmpl + w, "XXXXXX", 6);
  w += 6;
  tmpl[w] = '\0';
  int fd = ::mkstemp(tmpl);
  if (fd < 0) return std::unexpected(posix_fs_status("mkstemp", errno));
  fill(dst, cap, tmpl, std::strlen(tmpl));
  return FileHandle{static_cast<std::intptr_t>(fd)};
}

bool set_env_var(const char* name, const char* value) noexcept {
  return name != nullptr &&
         ::setenv(name, value != nullptr ? value : "", 1) == 0;
}

bool unset_env_var(const char* name) noexcept {
  return name != nullptr && ::unsetenv(name) == 0;
}

Result<bool> directory_iterate(const char* path,
                               bool (*visit)(const char*, bool, void*),
                               void* ctx) noexcept {
  if (path == nullptr || visit == nullptr) {
    return std::unexpected(
        Status::InvalidArgument("directory iterate: invalid argument"));
  }
  DIR* d = ::opendir(path);
  if (d == nullptr) {
    return std::unexpected(posix_fs_status("opendir", errno));
  }
  bool stopped = false;
  Status status = Status::Ok();
  for (;;) {
    errno = 0;
    struct dirent* de = ::readdir(d);
    if (de == nullptr) {
      if (errno != 0) status = posix_fs_status("readdir", errno);
      break;
    }
    const char* name = de->d_name;
    if (name[0] == '.' &&
        (name[1] == '\0' || (name[1] == '.' && name[2] == '\0')))
      continue;
    // d_type may be DT_UNKNOWN on some filesystems; fall back to lstat then.
    bool dir;
    if (de->d_type == DT_DIR) {
      dir = true;
    } else if (de->d_type == DT_UNKNOWN) {
      // lstat, NOT stat: a symlink-to-directory must report as a non-directory
      // (false), matching the DT_LNK fast path. Otherwise remove_directory_
      // recursive would follow the link and delete OUTSIDE the tree.
      char child[4096];
      struct ::stat st;
      dir = join_path(child, sizeof(child), path, name) &&
            ::lstat(child, &st) == 0 && S_ISDIR(st.st_mode);
    } else {
      dir = false;
    }
    if (!visit(name, dir, ctx)) {
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

Result<bool> exists(const char* path) noexcept {
  auto info = stat(path);
  if (info) return true;
  if (info.error().IsNotFound()) return false;
  return std::unexpected(std::move(info).error());
}

Result<bool> is_directory(const char* path) noexcept {
  auto info = stat(path);
  if (info) return info->is_directory;
  if (info.error().IsNotFound()) return false;
  return std::unexpected(std::move(info).error());
}

Result<std::uint64_t> file_size(const char* path) noexcept {
  HYLE_ASSIGN_OR_RETURN(FileInfo info, stat(path));
  if (info.is_directory) {
    return std::unexpected(
        Status::FailedPrecondition("file size: path is a directory"));
  }
  return info.size;
}

Status touch(const char* path) noexcept {
  if (path == nullptr) return Status::InvalidArgument("touch: path is null");
#if defined(HYLE_OS_WINDOWS)
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
  char buf[4096];
  std::size_t n = std::strlen(path);
  if (n >= sizeof(buf)) {
    return Status::OutOfRange("make directories: path exceeds 4095 bytes");
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
// Visitor state for remove_directory_recursive: remembers the parent path so it
// can rebuild each child's full path, and an `ok` flag latched on first failure.
struct RmCtx {
  const char* dir;
  Status status;
  unsigned depth;
};

// Recursion depth cap: a pathological deep tree would blow the stack
// (~several KB/frame). 512 is far past any real nesting. Fwd-decl for rm_visit.
constexpr unsigned kRmMaxDepth = 512;
Status rm_recursive_depth(const char* path, unsigned depth) noexcept;

bool rm_visit(const char* name, bool is_dir, void* vctx) noexcept {
  auto* c = static_cast<RmCtx*>(vctx);
  char child[4096];
  if (!join_path(child, sizeof(child), c->dir, name)) {
    c->status.Update(
        Status::OutOfRange("recursive remove: child path is too long"));
    return false;
  }
  // A symlink/junction child: delete the LINK, never recurse into its target
  // (which may live outside the tree). is_symlink tests the entry ITSELF (lstat
  // on POSIX / reparse bit on Windows), so this guards the Windows case where
  // directory_iterate reports a directory junction as is_dir=true (platform-audit #1).
  auto link = is_symlink(child);
  if (!link) {
    c->status.Update(std::move(link).error());
    return false;
  }
  Status status;
  if (*link) status = remove_symlink(child);
  else if (is_dir) status = rm_recursive_depth(child, c->depth + 1);
  else status = remove_file(child);
  c->status.Update(std::move(status));
  return c->status.ok();
}

// Depth-capped recursive core; both the public entry and rm_visit route here.
Status rm_recursive_depth(const char* path, unsigned depth) noexcept {
  if (depth > kRmMaxDepth) {
    return Status::OutOfRange("recursive remove: directory tree is too deep");
  }
  // A symlink/junction at this level: remove the LINK itself, never follow it into
  // a target outside the tree (platform-audit #1/#2 data-loss guard). is_symlink
  // tests the path itself (lstat / reparse bit), before is_directory (follows).
  HYLE_ASSIGN_OR_RETURN(bool link, is_symlink(path));
  if (link) return remove_symlink(path);
  HYLE_ASSIGN_OR_RETURN(bool directory, is_directory(path));
  if (!directory) return remove_file(path);  // a plain file: just unlink
  RmCtx c{path, Status::Ok(), depth};
  auto completed = directory_iterate(path, rm_visit, &c);
  if (!completed) return std::move(completed).error();
  if (!c.status.ok()) return std::move(c.status);
  return remove_directory(path);
}
}  // namespace

Status remove_directory_recursive(const char* path) noexcept {
  if (path == nullptr || path[0] == '\0') {
    return Status::InvalidArgument("recursive remove: path is empty");
  }
  return rm_recursive_depth(path, 0);
}

}  // namespace hyle::platform

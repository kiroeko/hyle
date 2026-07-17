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

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

// Close a raw fd/HANDLE. DEFINED in each platform section below (close() vs
// CloseHandle); DECLARED here so the platform-agnostic OwnedFd tail can call it.
void close_raw(std::intptr_t raw) noexcept;

}  // namespace

#if defined(HYLE_OS_WINDOWS)

// ---- Windows ----------------------------------------------------------------
//
// All path I/O goes through the wide (...W) API: the ...A variants interpret
// bytes as the active code page, NOT UTF-8, so a UTF-8 path with non-ASCII chars
// would be mis-decoded. We convert UTF-8 <-> UTF-16 at the boundary so callers
// pass UTF-8 everywhere (matching POSIX, which is byte-transparent).

namespace {

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
  int need = WideCharToMultiByte(CP_UTF8, 0, src, wlen, nullptr, 0, nullptr,
                                 nullptr);
  if (need <= 0) return fill(dst, cap, "", 0);
  if (static_cast<std::size_t>(need) < cap) {
    int got = WideCharToMultiByte(CP_UTF8, 0, src, wlen, dst,
                                  static_cast<int>(cap - 1), nullptr, nullptr);
    if (got < 0) got = 0;
    dst[got] = '\0';
    return static_cast<std::size_t>(got);
  }
  // Won't fit: convert into a heap scratch, then buffer-fill (truncates).
  char* tmp = static_cast<char*>(std::malloc(need + 1));
  if (tmp == nullptr) return fill(dst, cap, "", 0);
  int got = WideCharToMultiByte(CP_UTF8, 0, src, wlen, tmp, need, nullptr,
                                nullptr);
  if (got < 0) got = 0;
  tmp[got] = '\0';
  std::size_t r = fill(dst, cap, tmp, static_cast<std::size_t>(got));
  std::free(tmp);
  return r;
}

}  // namespace

FileInfo stat(const char* path) noexcept {
  FileInfo r;
  WidePath wp(path);
  if (wp.w() == nullptr) return r;
  WIN32_FILE_ATTRIBUTE_DATA d;
  if (!GetFileAttributesExW(wp.w(), GetFileExInfoStandard, &d)) return r;
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

bool make_directory(const char* path) noexcept {
  WidePath wp(path);
  if (wp.w() == nullptr) return false;
  if (CreateDirectoryW(wp.w(), nullptr)) return true;
  return GetLastError() == ERROR_ALREADY_EXISTS && is_directory(path);
}

bool remove_directory(const char* path) noexcept {
  WidePath wp(path);
  return wp.w() != nullptr && RemoveDirectoryW(wp.w()) != 0;
}

bool remove_file(const char* path) noexcept {
  WidePath wp(path);
  return wp.w() != nullptr && DeleteFileW(wp.w()) != 0;
}

bool rename_path(const char* from, const char* to) noexcept {
  WidePath wf(from), wt(to);
  if (wf.w() == nullptr || wt.w() == nullptr) return false;
  return MoveFileExW(wf.w(), wt.w(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_COPY_ALLOWED) != 0;
}

bool copy_file(const char* from, const char* to) noexcept {
  WidePath wf(from), wt(to);
  if (wf.w() == nullptr || wt.w() == nullptr) return false;
  return CopyFileW(wf.w(), wt.w(), /*bFailIfExists=*/FALSE) != 0;
}

std::size_t current_directory(char* dst, std::size_t cap) noexcept {
  wchar_t buf[MAX_PATH];
  // On overflow GetCurrentDirectoryW returns the REQUIRED length (> buf) without
  // filling buf -- guard against converting uninitialized bytes.
  DWORD n = GetCurrentDirectoryW(MAX_PATH, buf);
  if (n == 0 || n > MAX_PATH) return fill(dst, cap, "", 0);
  return fill_w(dst, cap, buf, static_cast<int>(n));
}

bool change_directory(const char* path) noexcept {
  WidePath wp(path);
  return wp.w() != nullptr && SetCurrentDirectoryW(wp.w()) != 0;
}

std::size_t temp_directory(char* dst, std::size_t cap) noexcept {
  wchar_t buf[MAX_PATH + 1];
  DWORD n = GetTempPathW(MAX_PATH + 1, buf);  // already honors TMP/TEMP
  if (n == 0 || n > MAX_PATH + 1) return fill(dst, cap, "", 0);
  while (n > 1 && (buf[n - 1] == L'\\' || buf[n - 1] == L'/')) --n;  // trim sep
  return fill_w(dst, cap, buf, static_cast<int>(n));
}

std::size_t home_directory(char* dst, std::size_t cap) noexcept {
  wchar_t buf[MAX_PATH];
  DWORD n = GetEnvironmentVariableW(L"USERPROFILE", buf, MAX_PATH);
  if (n == 0 || n >= MAX_PATH) return fill(dst, cap, "", 0);
  return fill_w(dst, cap, buf, static_cast<int>(n));
}

std::size_t get_env_var(const char* name, char* dst, std::size_t cap) noexcept {
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
  std::size_t r = fill_w(dst, cap, wbuf, static_cast<int>(got));
  std::free(heapw);
  return r;
}

bool directory_iterate(const char* path,
                       bool (*visit)(const char*, bool, void*),
                       void* ctx) noexcept {
  // Build the UTF-16 search pattern "<path>\*". Heap-size it off the path length
  // so long paths aren't capped at MAX_PATH.
  WidePath wp(path);
  if (wp.w() == nullptr) return false;
  std::size_t wl = std::wcslen(wp.w());
  wchar_t stackp[MAX_PATH];
  wchar_t* pat = stackp;
  wchar_t* heapp = nullptr;
  if (wl + 3 > MAX_PATH) {
    heapp = static_cast<wchar_t*>(std::malloc((wl + 3) * sizeof(wchar_t)));
    if (heapp == nullptr) return false;
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
  if (h == INVALID_HANDLE_VALUE) return false;
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
    if (!visit(name, dir, ctx)) break;
  } while (FindNextFileW(h, &fd));
  FindClose(h);
  return true;
}

namespace {
void close_raw(std::intptr_t raw) noexcept {
  if (raw != OwnedFd::kInvalid) CloseHandle(reinterpret_cast<HANDLE>(raw));
}
}  // namespace

std::uint32_t get_permissions(const char* path) noexcept {
  WidePath wp(path);
  if (wp.w() == nullptr) return 0;
  DWORD a = GetFileAttributesW(wp.w());
  if (a == INVALID_FILE_ATTRIBUTES) return 0;
  // No POSIX permission model on Windows: synthesize from the READONLY attribute.
  return (a & FILE_ATTRIBUTE_READONLY) ? 0444u : 0666u;
}

bool set_permissions(const char* path, std::uint32_t mode) noexcept {
  WidePath wp(path);
  if (wp.w() == nullptr) return false;
  DWORD a = GetFileAttributesW(wp.w());
  if (a == INVALID_FILE_ATTRIBUTES) return false;
  // Honor only the write bits: any owner/group/other write clear => READONLY.
  const bool writable = (mode & 0222u) != 0;
  if (writable) a &= ~static_cast<DWORD>(FILE_ATTRIBUTE_READONLY);
  else a |= FILE_ATTRIBUTE_READONLY;
  return SetFileAttributesW(wp.w(), a) != 0;
}

bool is_readonly(const char* path) noexcept {
  WidePath wp(path);
  if (wp.w() == nullptr) return false;
  DWORD a = GetFileAttributesW(wp.w());
  return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_READONLY) != 0;
}

bool set_readonly(const char* path, bool readonly) noexcept {
  return set_permissions(path, readonly ? 0444u : 0666u);
}

bool create_symlink(const char* target, const char* link) noexcept {
  WidePath wt(target), wl(link);
  if (wt.w() == nullptr || wl.w() == nullptr) return false;
  // Flag bit is target type; add UNPRIVILEGED_CREATE so Developer Mode suffices.
  DWORD flags = SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE;
  if (is_directory(target)) flags |= SYMBOLIC_LINK_FLAG_DIRECTORY;
  // Needs Developer Mode or admin; returns 0 without the privilege (soft-fail).
  return CreateSymbolicLinkW(wl.w(), wt.w(), flags) != 0;
}

bool is_symlink(const char* path) noexcept { return stat(path).is_symlink; }

std::size_t read_symlink(const char* path, char* dst, std::size_t cap) noexcept {
  WidePath wp(path);
  if (wp.w() == nullptr) return 0;
  // Open the link ITSELF (OPEN_REPARSE_POINT) and pull its stored target out of
  // the raw reparse buffer -- GetFinalPathNameByHandle would resolve THROUGH the
  // link to the final object, not report the link's own target string.
  HANDLE h = CreateFileW(wp.w(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                         OPEN_EXISTING,
                         FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT,
                         nullptr);
  if (h == INVALID_HANDLE_VALUE) return 0;

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
  CloseHandle(h);
  if (!ok) return 0;
  auto* rep = reinterpret_cast<ReparseHeader*>(raw);
  if (rep->ReparseTag != IO_REPARSE_TAG_SYMLINK) return 0;  // not a symlink
  // Prefer the human-readable PrintName; fall back to SubstituteName.
  USHORT off = rep->PrintNameLength ? rep->PrintNameOffset
                                    : rep->SubstituteNameOffset;
  USHORT len = rep->PrintNameLength ? rep->PrintNameLength
                                    : rep->SubstituteNameLength;
  const wchar_t* name =
      reinterpret_cast<const wchar_t*>(
          reinterpret_cast<const unsigned char*>(rep->PathBuffer) + off);
  return fill_w(dst, cap, name, static_cast<int>(len / sizeof(wchar_t)));
}

OwnedFd create_temp_file(const char* dir, const char* prefix, char* dst,
                         std::size_t cap) noexcept {
  char dirbuf[1024];
  if (dir == nullptr) {
    std::size_t n = temp_directory(dirbuf, sizeof(dirbuf));
    if (n == 0 || n >= sizeof(dirbuf)) return OwnedFd{};
    dir = dirbuf;
  }
  WidePath wd(dir);
  if (wd.w() == nullptr) return OwnedFd{};
  wchar_t wpre[16];
  const char* p = prefix != nullptr ? prefix : "tmp";
  int pn = MultiByteToWideChar(CP_UTF8, 0, p, -1, wpre, 16);
  if (pn <= 0) { wpre[0] = L't'; wpre[1] = L'\0'; }
  // GetTempFileNameW creates a unique 0-byte file and returns its path.
  wchar_t wpath[MAX_PATH];
  if (GetTempFileNameW(wd.w(), wpre, 0, wpath) == 0) return OwnedFd{};
  // Reopen it for read/write, keeping the on-disk file (it already exists).
  HANDLE h = CreateFileW(wpath, GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                         OPEN_EXISTING, FILE_ATTRIBUTE_TEMPORARY, nullptr);
  if (h == INVALID_HANDLE_VALUE) return OwnedFd{};
  int un = WideCharToMultiByte(CP_UTF8, 0, wpath, -1, nullptr, 0, nullptr, nullptr);
  if (un > 0) {
    // reuse fill via a stack UTF-8 copy
    char u8[MAX_PATH * 3];
    if (WideCharToMultiByte(CP_UTF8, 0, wpath, -1, u8, sizeof(u8), nullptr,
                            nullptr) > 0) {
      fill(dst, cap, u8, std::strlen(u8));
    }
  }
  return OwnedFd{reinterpret_cast<std::intptr_t>(h)};
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

FileInfo stat(const char* path) noexcept {
  struct ::stat s;
  FileInfo r;
  if (::stat(path, &s) != 0) return r;
  bool reg = S_ISREG(s.st_mode);
  r.is_directory = S_ISDIR(s.st_mode);
  r.exists = reg || r.is_directory;
  r.size = reg ? static_cast<std::uint64_t>(s.st_size) : 0;
  r.modified_time = static_cast<std::int64_t>(s.st_mtime);
  // stat() follows symlinks; an extra lstat tells us if the path ITSELF is a link
  // (a symlink to a dir still reports is_directory=true above, as callers expect).
  struct ::stat ls;
  if (::lstat(path, &ls) == 0 && S_ISLNK(ls.st_mode)) r.is_symlink = true;
  return r;
}

bool make_directory(const char* path) noexcept {
  if (::mkdir(path, 0777) == 0) return true;
  return errno == EEXIST && is_directory(path);
}

bool remove_directory(const char* path) noexcept {
  return ::rmdir(path) == 0;
}

bool remove_file(const char* path) noexcept { return ::unlink(path) == 0; }

bool rename_path(const char* from, const char* to) noexcept {
  return ::rename(from, to) == 0;
}

bool copy_file(const char* from, const char* to) noexcept {
  int in = ::open(from, O_RDONLY | O_CLOEXEC);
  if (in < 0) return false;
  struct ::stat s;
  if (::fstat(in, &s) != 0) {
    ::close(in);
    return false;
  }
  int out = ::open(to, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, s.st_mode & 0777);
  if (out < 0) {
    ::close(in);
    return false;
  }

  bool ok = true;
#if defined(HYLE_OS_LINUX)
  // Fast path: copy_file_range keeps bytes in the kernel (and can reflink on
  // btrfs/xfs) -- no user-space bounce. Only fall back to read/write if it fails
  // on the FIRST call (EXDEV cross-fs / ENOSYS old kernel / EINVAL), i.e. before
  // any bytes are written, so the fallback can't double-copy.
  bool use_userspace = false;
  off_t remaining = s.st_size;
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
      ok = false;  // failed mid-copy: a real error, don't restart
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
        ok = false;
        break;
      }
      if (got == 0) break;
      char* p = buf;
      ssize_t left = got;
      while (left > 0) {
        ssize_t wrote = ::write(out, p, static_cast<std::size_t>(left));
        if (wrote < 0) {
          if (errno == EINTR) continue;  // retry the same chunk
          ok = false;
          break;
        }
        left -= wrote;
        p += wrote;
      }
      if (!ok) break;
    }
  }
  ::close(in);
  ::close(out);
  return ok;
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
  if (h == nullptr || h[0] == '\0') {
    struct passwd* pw = ::getpwuid(::getuid());
    h = (pw != nullptr) ? pw->pw_dir : nullptr;
  }
  if (h == nullptr) return fill(dst, cap, "", 0);
  return fill(dst, cap, h, std::strlen(h));
}

std::size_t get_env_var(const char* name, char* dst, std::size_t cap) noexcept {
  const char* v = std::getenv(name);
  if (v == nullptr) return kEnvNotSet;
  return fill(dst, cap, v, std::strlen(v));
}

namespace {
void close_raw(std::intptr_t raw) noexcept {
  if (raw != OwnedFd::kInvalid) ::close(static_cast<int>(raw));
}
}  // namespace

std::uint32_t get_permissions(const char* path) noexcept {
  struct ::stat s;
  if (::stat(path, &s) != 0) return 0;
  return static_cast<std::uint32_t>(s.st_mode & 0777);
}

bool set_permissions(const char* path, std::uint32_t mode) noexcept {
  return ::chmod(path, static_cast<mode_t>(mode & 0777)) == 0;
}

bool is_readonly(const char* path) noexcept {
  struct ::stat s;
  if (::stat(path, &s) != 0) return false;
  return (s.st_mode & S_IWUSR) == 0;
}

bool set_readonly(const char* path, bool readonly) noexcept {
  struct ::stat s;
  if (::stat(path, &s) != 0) return false;
  mode_t m = readonly ? (s.st_mode & ~static_cast<mode_t>(0222))
                      : (s.st_mode | S_IWUSR);
  return ::chmod(path, m & 0777) == 0;
}

bool create_symlink(const char* target, const char* link) noexcept {
  return ::symlink(target, link) == 0;
}

bool is_symlink(const char* path) noexcept {
  struct ::stat ls;
  return ::lstat(path, &ls) == 0 && S_ISLNK(ls.st_mode);
}

std::size_t read_symlink(const char* path, char* dst, std::size_t cap) noexcept {
  char buf[4096];
  ssize_t n = ::readlink(path, buf, sizeof(buf));
  if (n < 0) return 0;
  return fill(dst, cap, buf, static_cast<std::size_t>(n));
}

OwnedFd create_temp_file(const char* dir, const char* prefix, char* dst,
                         std::size_t cap) noexcept {
  char dirbuf[1024];
  if (dir == nullptr) {
    std::size_t n = temp_directory(dirbuf, sizeof(dirbuf));
    if (n == 0 || n >= sizeof(dirbuf)) return OwnedFd{};
    dir = dirbuf;
  }
  const char* p = prefix != nullptr ? prefix : "tmp";
  // Build "<dir>/<prefix>XXXXXX" (mkstemp replaces the 6 X's atomically).
  char tmpl[1024];
  std::size_t dl = std::strlen(dir), pl = std::strlen(p);
  if (dl + 1 + pl + 6 + 1 > sizeof(tmpl)) return OwnedFd{};
  std::memcpy(tmpl, dir, dl);
  std::size_t w = dl;
  if (w == 0 || tmpl[w - 1] != '/') tmpl[w++] = '/';
  std::memcpy(tmpl + w, p, pl);
  w += pl;
  std::memcpy(tmpl + w, "XXXXXX", 6);
  w += 6;
  tmpl[w] = '\0';
  int fd = ::mkstemp(tmpl);
  if (fd < 0) return OwnedFd{};
  fill(dst, cap, tmpl, std::strlen(tmpl));
  return OwnedFd{static_cast<std::intptr_t>(fd)};
}

bool set_env_var(const char* name, const char* value) noexcept {
  return name != nullptr &&
         ::setenv(name, value != nullptr ? value : "", 1) == 0;
}

bool unset_env_var(const char* name) noexcept {
  return name != nullptr && ::unsetenv(name) == 0;
}

bool directory_iterate(const char* path,
                       bool (*visit)(const char*, bool, void*),
                       void* ctx) noexcept {
  DIR* d = ::opendir(path);
  if (d == nullptr) return false;
  for (;;) {
    struct dirent* de = ::readdir(d);
    if (de == nullptr) break;
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
    if (!visit(name, dir, ctx)) break;
  }
  ::closedir(d);
  return true;
}

#endif

// ---- portable (built on the primitives above) -------------------------------

// OwnedFd: platform-agnostic RAII; close_raw() is the per-OS close (defined in
// each section above, declared at the top of the file).
OwnedFd::~OwnedFd() { close_raw(raw_); }

OwnedFd& OwnedFd::operator=(OwnedFd&& o) noexcept {
  if (this != &o) {
    close_raw(raw_);
    raw_ = o.raw_;
    o.raw_ = kInvalid;
  }
  return *this;
}

void OwnedFd::reset() noexcept {
  close_raw(raw_);
  raw_ = kInvalid;
}

bool exists(const char* path) noexcept { return stat(path).exists; }
bool is_directory(const char* path) noexcept { return stat(path).is_directory; }
std::uint64_t file_size(const char* path) noexcept { return stat(path).size; }

bool touch(const char* path) noexcept {
#if defined(HYLE_OS_WINDOWS)
  // CreateFileW with OPEN_ALWAYS creates the file if absent, opens it if present
  // -- no std::fopen (which is ANSI on Windows and would mis-decode a UTF-8
  // path). Then stamp mtime to now.
  WidePath wp(path);
  if (wp.w() == nullptr) return false;
  HANDLE h = CreateFileW(wp.w(), FILE_WRITE_ATTRIBUTES, FILE_SHARE_READ, nullptr,
                         OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE) return false;
  FILETIME now;
  GetSystemTimeAsFileTime(&now);
  SetFileTime(h, nullptr, nullptr, &now);
  CloseHandle(h);
  return true;
#else
  std::FILE* f = std::fopen(path, "ab");  // append: create if absent, no truncate
  if (f == nullptr) return false;
  std::fclose(f);
  // Bump mtime to now even when the file already existed and append wrote 0 bytes.
  ::utimensat(AT_FDCWD, path, nullptr, 0);  // nullptr times => now
  return true;
#endif
}

bool make_directory_recursive(const char* path) noexcept {
  if (is_directory(path)) return true;
  char buf[4096];
  std::size_t n = std::strlen(path);
  if (n >= sizeof(buf)) return false;
  std::memcpy(buf, path, n + 1);
  // Create each ancestor in turn by temporarily terminating at each separator.
  for (std::size_t i = 1; i < n; ++i) {
    if (buf[i] == '/' || buf[i] == '\\') {
      char sep = buf[i];
      buf[i] = '\0';
      if (buf[0] != '\0' && !make_directory(buf)) {
        // Tolerate a drive root ("C:") or existing dir; only fail if not a dir.
        if (!is_directory(buf) && std::strcmp(buf, ".") != 0 &&
            !(i == 2 && buf[1] == ':')) {
          return false;
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
  bool ok;
};

bool rm_visit(const char* name, bool is_dir, void* vctx) noexcept {
  auto* c = static_cast<RmCtx*>(vctx);
  char child[4096];
  if (!join_path(child, sizeof(child), c->dir, name)) {
    c->ok = false;
    return true;
  }
  bool r = is_dir ? remove_directory_recursive(child) : remove_file(child);
  if (!r) c->ok = false;
  return true;  // keep going; report failure via ok
}
}  // namespace

bool remove_directory_recursive(const char* path) noexcept {
  if (!is_directory(path)) return remove_file(path);  // a plain file: just unlink
  RmCtx c{path, true};
  if (!directory_iterate(path, rm_visit, &c)) return false;
  if (!c.ok) return false;
  return remove_directory(path);
}

}  // namespace hyle::platform

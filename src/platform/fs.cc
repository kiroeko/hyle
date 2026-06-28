// std::fopen in touch() is flagged C4996 by MSVC; the "ab" use here is safe and
// portable, so opt out of the CRT deprecation nag (no global override exists).
#if defined(_WIN32)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

#include "hyle/platform/fs.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
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

}  // namespace

#if defined(_WIN32)

// ---- Windows ----------------------------------------------------------------

FileInfo stat(const char* path) noexcept {
  FileInfo r;
  WIN32_FILE_ATTRIBUTE_DATA d;
  if (!GetFileAttributesExA(path, GetFileExInfoStandard, &d)) return r;
  bool dir = (d.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
  r.exists = true;
  r.is_directory = dir;
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
  if (CreateDirectoryA(path, nullptr)) return true;
  return GetLastError() == ERROR_ALREADY_EXISTS && is_directory(path);
}

bool remove_directory(const char* path) noexcept {
  return RemoveDirectoryA(path) != 0;
}

bool remove_file(const char* path) noexcept { return DeleteFileA(path) != 0; }

bool rename_path(const char* from, const char* to) noexcept {
  return MoveFileExA(from, to, MOVEFILE_REPLACE_EXISTING | MOVEFILE_COPY_ALLOWED) != 0;
}

bool copy_file(const char* from, const char* to) noexcept {
  return CopyFileA(from, to, /*bFailIfExists=*/FALSE) != 0;
}

std::size_t current_directory(char* dst, std::size_t cap) noexcept {
  char buf[MAX_PATH];
  DWORD n = GetCurrentDirectoryA(sizeof(buf), buf);
  if (n == 0) return fill(dst, cap, "", 0);
  return fill(dst, cap, buf, n);
}

bool change_directory(const char* path) noexcept {
  return SetCurrentDirectoryA(path) != 0;
}

std::size_t temp_directory(char* dst, std::size_t cap) noexcept {
  char buf[MAX_PATH + 1];
  DWORD n = GetTempPathA(sizeof(buf), buf);  // already honors TMP/TEMP
  if (n == 0) return fill(dst, cap, "", 0);
  while (n > 1 && (buf[n - 1] == '\\' || buf[n - 1] == '/')) --n;  // trim slash
  return fill(dst, cap, buf, n);
}

std::size_t home_directory(char* dst, std::size_t cap) noexcept {
  char buf[MAX_PATH];
  DWORD n = GetEnvironmentVariableA("USERPROFILE", buf, sizeof(buf));
  if (n == 0 || n >= sizeof(buf)) return fill(dst, cap, "", 0);
  return fill(dst, cap, buf, n);
}

std::size_t get_env_var(const char* name, char* dst, std::size_t cap) noexcept {
  // Two-call pattern: first query the length, then read. GetEnvironmentVariable
  // returns 0 with ERROR_ENVVAR_NOT_FOUND when unset (vs set-but-empty -> 0 too,
  // disambiguated by the error code).
  DWORD need = GetEnvironmentVariableA(name, nullptr, 0);
  if (need == 0) {
    if (GetLastError() == ERROR_ENVVAR_NOT_FOUND) return kEnvNotSet;
    return fill(dst, cap, "", 0);  // set but empty
  }
  // `need` includes the NUL. Read into a scratch buffer sized to need.
  if (need <= cap) {
    DWORD got = GetEnvironmentVariableA(name, dst, static_cast<DWORD>(cap));
    return got;  // chars written (excl NUL) == full length when it fit
  }
  // Caller buffer too small: report the full length (need-1) after filling.
  char* tmp = static_cast<char*>(std::malloc(need));
  if (tmp == nullptr) return fill(dst, cap, "", 0);
  DWORD got = GetEnvironmentVariableA(name, tmp, need);
  std::size_t r = fill(dst, cap, tmp, got);
  std::free(tmp);
  return r;
}

bool directory_iterate(const char* path,
                       bool (*visit)(const char*, bool, void*),
                       void* ctx) noexcept {
  char pattern[MAX_PATH];
  std::size_t n = std::strlen(path);
  if (n + 3 >= sizeof(pattern)) return false;
  std::memcpy(pattern, path, n);
  pattern[n] = '\\';
  pattern[n + 1] = '*';
  pattern[n + 2] = '\0';
  WIN32_FIND_DATAA fd;
  HANDLE h = FindFirstFileA(pattern, &fd);
  if (h == INVALID_HANDLE_VALUE) return false;
  do {
    const char* name = fd.cFileName;
    if (name[0] == '.' &&
        (name[1] == '\0' || (name[1] == '.' && name[2] == '\0')))
      continue;
    bool dir = (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
    if (!visit(name, dir, ctx)) break;
  } while (FindNextFileA(h, &fd));
  FindClose(h);
  return true;
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
  int in = ::open(from, O_RDONLY);
  if (in < 0) return false;
  struct ::stat s;
  if (::fstat(in, &s) != 0) {
    ::close(in);
    return false;
  }
  int out = ::open(to, O_WRONLY | O_CREAT | O_TRUNC, s.st_mode & 0777);
  if (out < 0) {
    ::close(in);
    return false;
  }
  char buf[64 * 1024];
  bool ok = true;
  for (;;) {
    ssize_t got = ::read(in, buf, sizeof(buf));
    if (got < 0) {
      ok = false;
      break;
    }
    if (got == 0) break;
    char* p = buf;
    ssize_t left = got;
    while (left > 0) {
      ssize_t wrote = ::write(out, p, static_cast<std::size_t>(left));
      if (wrote < 0) {
        ok = false;
        break;
      }
      left -= wrote;
      p += wrote;
    }
    if (!ok) break;
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
    // d_type may be DT_UNKNOWN on some filesystems; fall back to stat then.
    bool dir;
    if (de->d_type == DT_DIR) {
      dir = true;
    } else if (de->d_type == DT_UNKNOWN) {
      char child[4096];
      dir = join_path(child, sizeof(child), path, name) && is_directory(child);
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

bool exists(const char* path) noexcept { return stat(path).exists; }
bool is_directory(const char* path) noexcept { return stat(path).is_directory; }
std::uint64_t file_size(const char* path) noexcept { return stat(path).size; }

bool touch(const char* path) noexcept {
  std::FILE* f = std::fopen(path, "ab");  // append: create if absent, no truncate
  if (f == nullptr) return false;
  std::fclose(f);
  // Bump mtime to now even when the file already existed and append wrote 0 bytes.
#if defined(_WIN32)
  HANDLE h = CreateFileA(path, FILE_WRITE_ATTRIBUTES, FILE_SHARE_READ, nullptr,
                         OPEN_EXISTING, 0, nullptr);
  if (h != INVALID_HANDLE_VALUE) {
    FILETIME now;
    GetSystemTimeAsFileTime(&now);
    SetFileTime(h, nullptr, nullptr, &now);
    CloseHandle(h);
  }
#else
  ::utimensat(AT_FDCWD, path, nullptr, 0);  // nullptr times => now
#endif
  return true;
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

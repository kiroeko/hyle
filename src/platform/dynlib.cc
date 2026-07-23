#include "hyle/platform/dynlib.h"

#include <cstdio>
#include <new>

#include "hyle/platform.h"  // HYLE_OS_* (project convention over raw _WIN32/__APPLE__)

#if defined(HYLE_OS_WINDOWS)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace hyle::platform {

namespace {
#if defined(HYLE_OS_WINDOWS)
const char* win32_loader_error_name(DWORD code) noexcept {
  switch (code) {
    case ERROR_MOD_NOT_FOUND: return "ERROR_MOD_NOT_FOUND";
    case ERROR_PROC_NOT_FOUND: return "ERROR_PROC_NOT_FOUND";
    case ERROR_BAD_EXE_FORMAT: return "ERROR_BAD_EXE_FORMAT";
    case ERROR_ACCESS_DENIED: return "ERROR_ACCESS_DENIED";
    case ERROR_NOT_ENOUGH_MEMORY: return "ERROR_NOT_ENOUGH_MEMORY";
    case ERROR_OUTOFMEMORY: return "ERROR_OUTOFMEMORY";
    default: return "WIN32_LOADER_ERROR";
  }
}

Status win32_loader_error(const char* operation, DWORD code) noexcept {
  char description[192]{};
  DWORD length = ::FormatMessageA(
      FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, code,
      0, description, static_cast<DWORD>(sizeof(description)), nullptr);
  while (length != 0 &&
         (description[length - 1] == '\r' || description[length - 1] == '\n')) {
    description[--length] = '\0';
  }
  char message[320];
  std::snprintf(message, sizeof(message),
                "windows %s failed: %s (%lu)%s%s", operation,
                win32_loader_error_name(code), static_cast<unsigned long>(code),
                length == 0 ? "" : ": ", length == 0 ? "" : description);
  if (code == ERROR_MOD_NOT_FOUND || code == ERROR_PROC_NOT_FOUND) {
    return Status::NotFound(message);
  }
  if (code == ERROR_BAD_EXE_FORMAT) return Status::InvalidArgument(message);
  if (code == ERROR_ACCESS_DENIED) return Status::PermissionDenied(message);
  if (code == ERROR_NOT_ENOUGH_MEMORY || code == ERROR_OUTOFMEMORY) {
    return Status::ResourceExhausted(message);
  }
  return Status::Unavailable(message);
}

struct WideString {
  wchar_t stack[512];
  wchar_t* ptr = nullptr;
  wchar_t* heap = nullptr;
  bool out_of_memory = false;

  explicit WideString(const char* utf8) noexcept {
    if (utf8 == nullptr) return;
    const int need = ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8,
                                           -1, nullptr, 0);
    if (need <= 0) return;
    ptr = stack;
    if (need > static_cast<int>(sizeof(stack) / sizeof(stack[0]))) {
      heap = new (std::nothrow) wchar_t[static_cast<std::size_t>(need)];
      ptr = heap;
      out_of_memory = ptr == nullptr;
    }
    if (ptr == nullptr ||
        ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8, -1, ptr,
                              need) != need) {
      delete[] heap;
      heap = ptr = nullptr;
    }
  }
  ~WideString() { delete[] heap; }
  WideString(const WideString&) = delete;
  WideString& operator=(const WideString&) = delete;
};
#else
Status posix_loader_error(const char* operation, const char* detail) noexcept {
  char message[320];
  std::snprintf(message, sizeof(message), "posix %s failed%s%s", operation,
                detail == nullptr || detail[0] == '\0' ? "" : ": ",
                detail == nullptr ? "" : detail);
  return Status::NotFound(message);
}
#endif
}  // namespace

const char* dynlib_extension() noexcept {
#if defined(HYLE_OS_WINDOWS)
  return "dll";
#elif defined(HYLE_OS_MACOS)
  return "dylib";
#else
  return "so";
#endif
}

Result<DynLib> DynLib::open(const char* path) noexcept {
  DynLib lib;
  if (path == nullptr || path[0] == '\0') {
    return std::unexpected(Status::InvalidArgument("dynlib open: empty path"));
  }
#if defined(HYLE_OS_WINDOWS)
  WideString w(path);
  if (!w.ptr) {
    return std::unexpected(
        w.out_of_memory
            ? Status::ResourceExhausted("dynlib open: path conversion allocation failed")
            : Status::InvalidArgument("dynlib open: path is not valid UTF-8"));
  }
  HMODULE h = LoadLibraryW(w.ptr);
  if (h == nullptr) {
    return std::unexpected(win32_loader_error("LoadLibraryW", ::GetLastError()));
  }
  lib.handle_ = reinterpret_cast<void*>(h);
#else
  // RTLD_NOW: resolve all symbols at load so a missing symbol fails here, not
  // mid-call. RTLD_LOCAL: don't leak this lib's symbols into the global scope.
  void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (h == nullptr) {
    return std::unexpected(posix_loader_error("dlopen", dlerror()));
  }
  lib.handle_ = h;
#endif
  lib.owns_ = true;
  return lib;
}

Result<DynLib> DynLib::get(const char* path) noexcept {
  DynLib lib;
  if (path == nullptr || path[0] == '\0') {
    return std::unexpected(Status::InvalidArgument("dynlib get: empty path"));
  }
#if defined(HYLE_OS_WINDOWS)
  // Already-mapped modules only; does not load and does not bump the refcount.
  WideString w(path);
  if (!w.ptr) {
    return std::unexpected(
        w.out_of_memory
            ? Status::ResourceExhausted("dynlib get: path conversion allocation failed")
            : Status::InvalidArgument("dynlib get: path is not valid UTF-8"));
  }
  HMODULE h = GetModuleHandleW(w.ptr);
  if (h == nullptr) {
    return std::unexpected(
        win32_loader_error("GetModuleHandleW", ::GetLastError()));
  }
  lib.handle_ = reinterpret_cast<void*>(h);
  lib.owns_ = false;  // must not FreeLibrary a GetModuleHandle result
#else
  // RTLD_NOLOAD: succeed only if already loaded; never map a new image.
  // A successful NOLOAD still increments the dlopen refcount -> we own a close.
  void* h = dlopen(path, RTLD_NOW | RTLD_NOLOAD);
  if (h == nullptr) {
    const char* e = dlerror();
    return std::unexpected(posix_loader_error(
        "dlopen(RTLD_NOLOAD)",
        e != nullptr ? e : "library not loaded in this process"));
  }
  lib.handle_ = h;
  lib.owns_ = true;
#endif
  return lib;
}

Result<DynLib> DynLib::current() noexcept {
  DynLib lib;
#if defined(HYLE_OS_WINDOWS)
  lib.handle_ = reinterpret_cast<void*>(GetModuleHandleW(nullptr));
  if (lib.handle_ == nullptr) {
    return std::unexpected(
        win32_loader_error("GetModuleHandleW(current)", ::GetLastError()));
  }
  lib.owns_ = false;  // GetModuleHandle does not increment the refcount
#else
  lib.handle_ = dlopen(nullptr, RTLD_NOW | RTLD_LOCAL);
  if (lib.handle_ == nullptr) {
    return std::unexpected(posix_loader_error("dlopen(current)", dlerror()));
  }
  // dlopen(nullptr) returns a closeable handle and increments the loader's
  // reference count just like other successful dlopen calls.
  lib.owns_ = true;
#endif
  return lib;
}

Result<void*> DynLib::symbol(const char* name) const noexcept {
  if (handle_ == nullptr) {
    return std::unexpected(
        Status::FailedPrecondition("dynlib symbol: library is not open"));
  }
  if (name == nullptr || name[0] == '\0') {
    return std::unexpected(Status::InvalidArgument("dynlib symbol: empty name"));
  }
#if defined(HYLE_OS_WINDOWS)
  FARPROC p = GetProcAddress(reinterpret_cast<HMODULE>(handle_), name);
  if (p == nullptr) {
    return std::unexpected(
        win32_loader_error("GetProcAddress", ::GetLastError()));
  }
  return reinterpret_cast<void*>(p);
#else
  dlerror();  // clear stale error so a NULL-but-valid symbol is distinguishable
  void* p = dlsym(handle_, name);
  const char* e = dlerror();
  if (e != nullptr) {
    return std::unexpected(posix_loader_error("dlsym", e));
  }
  return p;
#endif
}

void DynLib::close() noexcept {
  if (handle_ != nullptr && owns_) {
#if defined(HYLE_OS_WINDOWS)
    FreeLibrary(reinterpret_cast<HMODULE>(handle_));
#else
    dlclose(handle_);
#endif
  }
  handle_ = nullptr;
  owns_ = false;
}

DynLib::~DynLib() { close(); }

DynLib& DynLib::operator=(DynLib&& o) noexcept {
  if (this != &o) {
    close();
    handle_ = o.handle_;
    owns_ = o.owns_;
    o.handle_ = nullptr;
    o.owns_ = false;
  }
  return *this;
}

}  // namespace hyle::platform

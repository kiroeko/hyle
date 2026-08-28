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
Status win32_loader_error(const char* operation, DWORD code) noexcept {
  return status_from_win32(code, operation);
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

Status posix_loader_close_error(const char* detail) noexcept {
  char message[320];
  std::snprintf(message, sizeof(message), "posix dlclose failed%s%s",
                detail == nullptr || detail[0] == '\0' ? "" : ": ",
                detail == nullptr ? "" : detail);
  return Status::Unavailable(message);
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
  // Already-mapped modules only. GetModuleHandleEx pins the returned image by
  // incrementing its loader refcount, closing the unload-vs-symbol race present
  // with a borrowed GetModuleHandleW result.
  WideString w(path);
  if (!w.ptr) {
    return std::unexpected(
        w.out_of_memory
            ? Status::ResourceExhausted("dynlib get: path conversion allocation failed")
            : Status::InvalidArgument("dynlib get: path is not valid UTF-8"));
  }
  HMODULE h = nullptr;
  if (!::GetModuleHandleExW(0, w.ptr, &h)) {
    return std::unexpected(
        win32_loader_error("GetModuleHandleExW", ::GetLastError()));
  }
  lib.handle_ = reinterpret_cast<void*>(h);
  lib.owns_ = true;
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

Status DynLib::close() noexcept {
  void* handle = handle_;
  const bool owns = owns_;
  handle_ = nullptr;
  owns_ = false;
  if (handle != nullptr && owns) {
#if defined(HYLE_OS_WINDOWS)
    if (!::FreeLibrary(reinterpret_cast<HMODULE>(handle))) {
      return win32_loader_error("FreeLibrary", ::GetLastError());
    }
#else
    if (::dlclose(handle) != 0) {
      return posix_loader_close_error(::dlerror());
    }
#endif
  }
  return Status::Ok();
}

DynLib::~DynLib() { (void)close(); }

DynLib& DynLib::operator=(DynLib&& o) noexcept {
  if (this != &o) {
    (void)close();
    handle_ = o.handle_;
    owns_ = o.owns_;
    o.handle_ = nullptr;
    o.owns_ = false;
  }
  return *this;
}

}  // namespace hyle::platform

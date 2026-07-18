#include "hyle/platform/dynlib.h"

#include <cstring>
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
// Thread-local error buffer: dynlib errors are reported per-thread without
// allocating (clay returned char* via an Allocator). 256 bytes covers OS
// messages; longer ones truncate.
thread_local char g_error[256] = {0};

void set_error(const char* msg) noexcept {
  if (msg == nullptr) {
    g_error[0] = '\0';
    return;
  }
  std::size_t n = std::strlen(msg);
  if (n >= sizeof(g_error)) n = sizeof(g_error) - 1;
  std::memcpy(g_error, msg, n);
  g_error[n] = '\0';
}

#if defined(HYLE_OS_WINDOWS)
// Format the last Win32 error into the thread-local buffer.
void set_error_win() noexcept {
  DWORD code = GetLastError();
  char buf[256];
  DWORD len = FormatMessageA(
      FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, code,
      0, buf, sizeof(buf), nullptr);
  if (len == 0) {
    set_error("unknown error");
  } else {
    set_error(buf);
  }
}

struct WideString {
  wchar_t stack[512];
  wchar_t* ptr = nullptr;
  wchar_t* heap = nullptr;

  explicit WideString(const char* utf8) noexcept {
    if (utf8 == nullptr) return;
    const int need = ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8,
                                           -1, nullptr, 0);
    if (need <= 0) return;
    ptr = stack;
    if (need > static_cast<int>(sizeof(stack) / sizeof(stack[0]))) {
      heap = new (std::nothrow) wchar_t[static_cast<std::size_t>(need)];
      ptr = heap;
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

const char* dynlib_last_error() noexcept { return g_error; }
void dynlib_set_last_error(const char* msg) noexcept { set_error(msg); }

DynLib DynLib::open(const char* path) noexcept {
  DynLib lib;
#if defined(HYLE_OS_WINDOWS)
  WideString w(path);
  if (!w.ptr) {
    set_error("dynlib open: null path, invalid UTF-8, or out of memory");
    return lib;
  }
  HMODULE h = LoadLibraryW(w.ptr);
  if (h == nullptr) {
    set_error_win();
    return lib;
  }
  lib.handle_ = reinterpret_cast<void*>(h);
#else
  // RTLD_NOW: resolve all symbols at load so a missing symbol fails here, not
  // mid-call. RTLD_LOCAL: don't leak this lib's symbols into the global scope.
  void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (h == nullptr) {
    set_error(dlerror());
    return lib;
  }
  lib.handle_ = h;
#endif
  lib.owns_ = true;
  set_error(nullptr);  // clear on success
  return lib;
}

DynLib DynLib::get(const char* path) noexcept {
  DynLib lib;
  if (path == nullptr || path[0] == '\0') {
    set_error("dynlib get: empty path");
    return lib;
  }
#if defined(HYLE_OS_WINDOWS)
  // Already-mapped modules only; does not load and does not bump the refcount.
  WideString w(path);
  if (!w.ptr) {
    set_error("dynlib get: invalid UTF-8 or out of memory");
    return lib;
  }
  HMODULE h = GetModuleHandleW(w.ptr);
  if (h == nullptr) {
    set_error_win();
    return lib;
  }
  lib.handle_ = reinterpret_cast<void*>(h);
  lib.owns_ = false;  // must not FreeLibrary a GetModuleHandle result
#else
  // RTLD_NOLOAD: succeed only if already loaded; never map a new image.
  // A successful NOLOAD still increments the dlopen refcount -> we own a close.
  void* h = dlopen(path, RTLD_NOW | RTLD_NOLOAD);
  if (h == nullptr) {
    const char* e = dlerror();
    set_error(e != nullptr ? e : "library not loaded in this process");
    return lib;
  }
  lib.handle_ = h;
  lib.owns_ = true;
#endif
  set_error(nullptr);
  return lib;
}

DynLib DynLib::current() noexcept {
  DynLib lib;
#if defined(HYLE_OS_WINDOWS)
  lib.handle_ = reinterpret_cast<void*>(GetModuleHandleW(nullptr));
  if (lib.handle_ == nullptr) {
    set_error_win();
    return lib;
  }
  lib.owns_ = false;  // GetModuleHandle does not increment the refcount
#else
  lib.handle_ = dlopen(nullptr, RTLD_NOW | RTLD_LOCAL);
  if (lib.handle_ == nullptr) {
    set_error(dlerror());
    return lib;
  }
  // dlopen(nullptr) returns a closeable handle and increments the loader's
  // reference count just like other successful dlopen calls.
  lib.owns_ = true;
#endif
  set_error(nullptr);
  return lib;
}

void* DynLib::symbol(const char* name) const noexcept {
  if (handle_ == nullptr) {
    set_error("symbol() on an invalid library");
    return nullptr;
  }
  if (name == nullptr || name[0] == '\0') {
    set_error("symbol(): empty name");
    return nullptr;
  }
#if defined(HYLE_OS_WINDOWS)
  FARPROC p = GetProcAddress(reinterpret_cast<HMODULE>(handle_), name);
  if (p == nullptr) {
    set_error_win();
  } else {
    set_error(nullptr);
  }
  return reinterpret_cast<void*>(p);
#else
  dlerror();  // clear stale error so a NULL-but-valid symbol is distinguishable
  void* p = dlsym(handle_, name);
  const char* e = dlerror();
  if (e != nullptr) {
    set_error(e);
    return nullptr;
  }
  set_error(nullptr);
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

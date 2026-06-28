#include "hyle/platform/dynlib.h"

#include <cstring>

#if defined(_WIN32)
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

#if defined(_WIN32)
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
#endif
}  // namespace

const char* dynlib_extension() noexcept {
#if defined(_WIN32)
  return "dll";
#elif defined(__APPLE__)
  return "dylib";
#else
  return "so";
#endif
}

const char* dynlib_last_error() noexcept { return g_error; }

DynLib DynLib::open(const char* path) noexcept {
  DynLib lib;
#if defined(_WIN32)
  HMODULE h = LoadLibraryA(path);
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

DynLib DynLib::current() noexcept {
  DynLib lib;
#if defined(_WIN32)
  lib.handle_ = reinterpret_cast<void*>(GetModuleHandleA(nullptr));
#else
  lib.handle_ = dlopen(nullptr, RTLD_NOW | RTLD_LOCAL);
#endif
  lib.owns_ = false;  // process image: not ours to free
  return lib;
}

void* DynLib::symbol(const char* name) const noexcept {
  if (handle_ == nullptr) {
    set_error("symbol() on an invalid library");
    return nullptr;
  }
#if defined(_WIN32)
  FARPROC p = GetProcAddress(reinterpret_cast<HMODULE>(handle_), name);
  if (p == nullptr) set_error_win();
  return reinterpret_cast<void*>(p);
#else
  dlerror();  // clear stale error so a NULL-but-valid symbol is distinguishable
  void* p = dlsym(handle_, name);
  if (p == nullptr) {
    const char* e = dlerror();
    if (e != nullptr) set_error(e);
  }
  return p;
#endif
}

void DynLib::close() noexcept {
  if (handle_ != nullptr && owns_) {
#if defined(_WIN32)
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

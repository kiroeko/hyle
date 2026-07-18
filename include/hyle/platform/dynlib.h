#pragma once

#include "hyle/macros.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_PLATFORM_API
#elif defined(HYLE_PLATFORM_BUILD)
#  define HYLE_PLATFORM_API HYLE_EXPORT
#else
#  define HYLE_PLATFORM_API HYLE_IMPORT
#endif

// Dynamic library loading (clay dynlib). Cross-platform dlopen/LoadLibrary. RAII
// over clay's manual open/close: a DynLib closes on destruction, so a forgotten
// close can't leak the OS handle. Error detail is a thread-local string (no
// allocator, unlike clay's char*(Allocator*)).
//
// STATUS (platform-audit followup #21): the original motivation — dlopen the
// CPUID-scored best SIMD backend as a separate loadable unit (conventions §10) —
// did NOT materialize. SIMD dispatch is in-process: compile-time variant TUs
// selected by the cpu_features() runtime table (simd/runtime.cc), no dlopen. So
// dynlib now stands as (1) a general FFI-embedder primitive — load a host plugin
// or user-supplied backend at runtime — and (2) groundwork for any future
// out-of-process backend. Kept because it's a clean, tested platform capability
// with real FFI use, not because SIMD needs it.

namespace hyle::platform {

// Platform shared-library file extension WITHOUT the dot: "dll" / "so" / "dylib".
HYLE_PLATFORM_API const char* dynlib_extension() noexcept;

// The last dynlib error on THIS thread (open/symbol failure), or "" if none.
// Valid until the next failing dynlib call on the same thread.
HYLE_PLATFORM_API const char* dynlib_last_error() noexcept;

// An owned handle to a loaded shared library. Move-only; closes on destruction.
class DynLib {
 public:
  // Load the library at `path`. On failure returns an invalid DynLib (valid()
  // == false); see dynlib_last_error() for why.
  HYLE_PLATFORM_API static DynLib open(const char* path) noexcept;

  // Attach to a library ALREADY mapped into this process (clay dynlib_get).
  // Does NOT load if absent -- returns invalid instead (Win GetModuleHandle /
  // POSIX dlopen(RTLD_NOLOAD)). Use when another component already loaded the
  // DLL and you only need a handle for symbol lookup.
  // Ownership: Windows handle is non-owning (GetModuleHandle does not pin);
  // POSIX NOLOAD bumps the dlopen refcount so the DynLib owns a close.
  // NAME MATCHING: RTLD_NOLOAD matches an already-loaded image by its INSTALL
  // NAME / canonical path, NOT by the search aliases a plain open() accepts. On
  // macOS especially, pass the install-name (e.g. "libSystem.B.dylib", not
  // "libSystem.dylib") -- the alias resolves under open() but NOLOAD returns
  // invalid for it. Prefer the exact name/path the library was loaded under.
  HYLE_PLATFORM_API static DynLib get(const char* path) noexcept;

  // A handle to the current process image (POSIX dlopen(nullptr) / Win
  // GetModuleHandle(nullptr)) for resolving already-loaded symbols. Does NOT own
  // a refcount to free -- its destructor is a no-op.
  HYLE_PLATFORM_API static DynLib current() noexcept;

  DynLib() noexcept = default;
  HYLE_PLATFORM_API ~DynLib();
  DynLib(DynLib&& o) noexcept
      : handle_(o.handle_), owns_(o.owns_) {
    o.handle_ = nullptr;
    o.owns_ = false;
  }
  HYLE_PLATFORM_API DynLib& operator=(DynLib&& o) noexcept;
  HYLE_DISABLE_COPY(DynLib);

  bool valid() const noexcept { return handle_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }

  // Address of exported `name`, or nullptr if absent (sets dynlib_last_error()).
  HYLE_PLATFORM_API void* symbol(const char* name) const noexcept;

  // Typed symbol lookup: dl.symbol_as<int(*)(void)>("foo"). Caller asserts the
  // signature (the loader cannot check it).
  template <class Fn>
  Fn symbol_as(const char* name) const noexcept {
    return reinterpret_cast<Fn>(symbol(name));
  }

  // Close early (also done by the destructor). Safe to call on an invalid lib.
  HYLE_PLATFORM_API void close() noexcept;

 private:
  void* handle_ = nullptr;  // HMODULE on Windows, void* from dlopen on POSIX
  bool owns_ = false;       // current() handles are not freed on close
};

}  // namespace hyle::platform

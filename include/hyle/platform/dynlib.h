#pragma once

#include <type_traits>
#include <utility>

#include "hyle/error/result.h"
#include "hyle/core/compiler.h"
#include "hyle/platform/visibility.h"

// Cross-platform dlopen/LoadLibrary ownership. DynLib closes its loader
// reference on destruction, so a forgotten explicit close cannot leak the OS
// handle. Recoverable loader failures are returned as Result values; there is
// no module-local last-error side channel.
//
// This is an in-process optional-native-backend/plugin primitive. It does not
// define a plugin ABI, an FFI surface, or an out-of-process protocol; those
// contracts belong to the consumer that chooses to load a library.

namespace hyle::platform {

// Platform shared-library file extension WITHOUT the dot: "dll" / "so" / "dylib".
HYLE_PLATFORM_API const char* dynlib_extension() noexcept;

// An owned handle to a loaded shared library. Move-only; closes on destruction.
class DynLib {
 public:
  // Load the library at `path`.
  HYLE_PLATFORM_API static Result<DynLib> open(const char* path) noexcept;

  // Attach to a library ALREADY mapped into this process.
  // Does NOT load if absent -- returns NotFound (Win GetModuleHandle / POSIX
  // dlopen(RTLD_NOLOAD)). Use when another component already loaded the DLL and
  // you only need a handle for symbol lookup.
  // A successful get pins/increments the loader reference on every platform,
  // so symbol lookup cannot race another owner unloading the image. The
  // returned DynLib owns that reference and balances it in close/destruction.
  // NAME MATCHING: RTLD_NOLOAD matches an already-loaded image by its INSTALL
  // NAME / canonical path, NOT by the search aliases a plain open() accepts. On
  // macOS especially, pass the install-name (e.g. "libSystem.B.dylib", not
  // "libSystem.dylib") -- the alias resolves under open() but NOLOAD returns
  // invalid for it. Prefer the exact name/path the library was loaded under.
  HYLE_PLATFORM_API static Result<DynLib> get(const char* path) noexcept;

  // A handle to the current process image (POSIX dlopen(nullptr) / Win
  // GetModuleHandle(nullptr)) for resolving already-loaded symbols. Windows is
  // non-owning; POSIX dlopen increments a refcount, so the destructor balances
  // it with dlclose.
  HYLE_PLATFORM_API static Result<DynLib> current() noexcept;

  DynLib() noexcept = default;
  HYLE_PLATFORM_API ~DynLib();
  DynLib(DynLib&& o) noexcept
      : handle_(o.handle_), owns_(o.owns_) {
    o.handle_ = nullptr;
    o.owns_ = false;
  }
  HYLE_PLATFORM_API DynLib& operator=(DynLib&& o) noexcept;
  DynLib(const DynLib&) = delete;
  DynLib& operator=(const DynLib&) = delete;

  bool valid() const noexcept { return handle_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }

  // Address of exported `name`. A platform may legally export a null-valued
  // symbol, so success is represented by the Result state rather than pointer
  // truthiness.
  HYLE_PLATFORM_API Result<void*> symbol(const char* name) const noexcept;

  // Typed symbol lookup: dl.symbol_as<int(*)(void)>("foo"). Caller asserts the
  // signature (the loader cannot check it).
  template <class Pointer>
    requires std::is_pointer_v<Pointer>
  Result<Pointer> symbol_as(const char* name) const noexcept {
    Result<void*> address = symbol(name);
    if (!address) return std::unexpected(std::move(address).error());
    return reinterpret_cast<Pointer>(*address);
  }

  // Close early (also done by the destructor). Safe to call on an invalid lib.
  // The object is empty after the call even if the native loader reports an
  // unload failure; use this explicit form when that diagnostic matters.
  HYLE_PLATFORM_API Status close() noexcept;

 private:
  void* handle_ = nullptr;  // HMODULE on Windows, void* from dlopen on POSIX
  bool owns_ = false;       // close/destructor must release the loader reference
};

}  // namespace hyle::platform

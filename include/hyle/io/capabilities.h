#pragma once

// The coroutine transport is a product-shape capability, not a property that
// can be inferred from the presence of the public header tree: HYLE installs
// the reviewed headers as one surface even when the target has no sched/
// fcontext implementation.  Linking hyle::io supplies the authoritative 0/1
// definition in both add_subdirectory() and installed-package consumption.
//
// Raw headers + library are also a supported delivery form.  When no CMake
// target supplied the product definition, derive the same public target fact
// as root CMake: x64 on the Microsoft ABI, Linux, or macOS; and arm64 on Linux
// or macOS.  In particular, MinGW x64 and Windows arm64 are not claimed until
// HYLE ships matching fcontext assembly/build integration.
#include <cstdint>

#include "hyle/core/environment.h"

#if HYLE_ARCH_X64 &&                                           \
    ((HYLE_OS_WINDOWS && defined(_MSC_VER)) ||                 \
     HYLE_OS_LINUX || HYLE_OS_MACOS)
#  define HYLE_IO_DETAIL_DERIVED_COROUTINE_TRANSPORT 1
#elif HYLE_ARCH_ARM64 &&                                      \
    (HYLE_OS_LINUX || HYLE_OS_MACOS)
#  define HYLE_IO_DETAIL_DERIVED_COROUTINE_TRANSPORT 1
#else
#  define HYLE_IO_DETAIL_DERIVED_COROUTINE_TRANSPORT 0
#endif

// A linked source/install target is authoritative, but it must never disagree
// with the public compiler/ABI facts used by a raw headers+library consumer.
// Such a mismatch means the package was generated for a different target, not
// an optional feature that can safely degrade.
#if defined(HYLE_IO_HAS_COROUTINE_TRANSPORT)
#  if HYLE_IO_HAS_COROUTINE_TRANSPORT != HYLE_IO_DETAIL_DERIVED_COROUTINE_TRANSPORT
#    error "HYLE io coroutine-transport target/header capability mismatch; use a library package built for this OS/architecture/compiler ABI."
#  endif
#else
#  define HYLE_IO_HAS_COROUTINE_TRANSPORT \
    HYLE_IO_DETAIL_DERIVED_COROUTINE_TRANSPORT
#endif

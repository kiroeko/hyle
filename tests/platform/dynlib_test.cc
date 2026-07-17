#include <gtest/gtest.h>

#include <cstring>

#include "hyle/platform/dynlib.h"

using hyle::platform::DynLib;
using hyle::platform::dynlib_extension;
using hyle::platform::dynlib_last_error;

// A known system library + an exported symbol, per platform.
#if defined(_WIN32)
static const char* kSysLib = "kernel32.dll";
static const char* kSysSym = "GetCurrentProcessId";
static const char* kExpectExt = "dll";
#elif defined(__APPLE__)
// Use the library's real INSTALL NAME, not the "libSystem.dylib" search alias:
// macOS RTLD_NOLOAD (DynLib::get) matches already-loaded images by their
// install-name / canonical path, and libSystem's install-name is libSystem.B.
// The search alias resolves under a plain dlopen but is NOT what NOLOAD keys on
// (verified on macOS 26: NOLOAD "libSystem.dylib" -> NULL, "libSystem.B.dylib" -> ok).
static const char* kSysLib = "libSystem.B.dylib";
static const char* kSysSym = "getpid";
static const char* kExpectExt = "dylib";
#else
static const char* kSysLib = "libm.so.6";
static const char* kSysSym = "cos";
static const char* kExpectExt = "so";
#endif

TEST(DynLib, Extension) {
  EXPECT_STREQ(dynlib_extension(), kExpectExt);
}

TEST(DynLib, OpenSystemLibAndFindSymbol) {
  DynLib lib = DynLib::open(kSysLib);
  ASSERT_TRUE(lib.valid()) << "open failed: " << dynlib_last_error();
  EXPECT_TRUE(static_cast<bool>(lib));
  void* sym = lib.symbol(kSysSym);
  EXPECT_NE(sym, nullptr) << "symbol failed: " << dynlib_last_error();
}

TEST(DynLib, OpenNonexistentFails) {
  DynLib lib = DynLib::open("this_library_does_not_exist_zzz.xyz");
  EXPECT_FALSE(lib.valid());
  EXPECT_STRNE(dynlib_last_error(), "");  // error string was set
}

TEST(DynLib, MissingSymbolReturnsNull) {
  DynLib lib = DynLib::open(kSysLib);
  ASSERT_TRUE(lib.valid());
  EXPECT_EQ(lib.symbol("definitely_not_a_real_symbol_zzz"), nullptr);
}

TEST(DynLib, TypedSymbolCallable) {
  DynLib lib = DynLib::open(kSysLib);
  ASSERT_TRUE(lib.valid());
#if defined(_WIN32)
  auto fn = lib.symbol_as<unsigned long (*)()>(kSysSym);  // GetCurrentProcessId
  ASSERT_NE(fn, nullptr);
  EXPECT_NE(fn(), 0u);  // our PID is nonzero
#else
  auto fn = lib.symbol_as<double (*)(double)>(kSysSym);  // cos / (getpid on mac)
  ASSERT_NE(fn, nullptr);
#  if !defined(__APPLE__)
  EXPECT_DOUBLE_EQ(fn(0.0), 1.0);  // cos(0) == 1
#  endif
#endif
}

TEST(DynLib, MoveTransfersOwnership) {
  DynLib a = DynLib::open(kSysLib);
  ASSERT_TRUE(a.valid());
  DynLib b = std::move(a);
  EXPECT_FALSE(a.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_TRUE(b.valid());
  EXPECT_NE(b.symbol(kSysSym), nullptr);
}

TEST(DynLib, CloseInvalidates) {
  DynLib lib = DynLib::open(kSysLib);
  ASSERT_TRUE(lib.valid());
  lib.close();
  EXPECT_FALSE(lib.valid());
  lib.close();  // double close is safe
}

TEST(DynLib, CurrentProcessImage) {
  DynLib self = DynLib::current();
  EXPECT_TRUE(self.valid());
  // The process image can resolve a libc symbol it links (POSIX dlopen(NULL));
  // on Windows GetModuleHandle(NULL) is the exe -- it may not export libc, so
  // only assert validity here, not a specific symbol.
}

// get(path): attach only if already mapped -- never load a missing library.
TEST(DynLib, GetMissingDoesNotLoad) {
  DynLib lib = DynLib::get("this_library_does_not_exist_zzz.xyz");
  EXPECT_FALSE(lib.valid());
  EXPECT_STRNE(dynlib_last_error(), "");
}

// System libs are almost always already mapped into the process.
TEST(DynLib, GetAlreadyLoadedSystemLib) {
  DynLib lib = DynLib::get(kSysLib);
  ASSERT_TRUE(lib.valid()) << "get failed: " << dynlib_last_error();
  void* sym = lib.symbol(kSysSym);
  EXPECT_NE(sym, nullptr) << "symbol failed: " << dynlib_last_error();
}

// open then get: get attaches to the same already-resident image.
TEST(DynLib, GetAfterOpen) {
  DynLib loaded = DynLib::open(kSysLib);
  ASSERT_TRUE(loaded.valid());
  DynLib attached = DynLib::get(kSysLib);
  ASSERT_TRUE(attached.valid()) << dynlib_last_error();
  EXPECT_NE(attached.symbol(kSysSym), nullptr);
}

TEST(DynLib, GetEmptyPathFails) {
  EXPECT_FALSE(DynLib::get("").valid());
  EXPECT_FALSE(DynLib::get(nullptr).valid());
}

TEST(DynLib, DoubleCloseSafeAfterGet) {
  DynLib lib = DynLib::get(kSysLib);
  ASSERT_TRUE(lib.valid());
  lib.close();
  EXPECT_FALSE(lib.valid());
  lib.close();
}

TEST(DynLib, CurrentCloseIsNoOp) {
  DynLib self = DynLib::current();
  ASSERT_TRUE(self.valid());
  self.close();  // owns_=false
  EXPECT_FALSE(self.valid());
}
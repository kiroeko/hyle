#include <gtest/gtest.h>

#include <cstring>
#include <string_view>

#include "hyle/platform/dynlib.h"

using hyle::platform::DynLib;
using hyle::platform::dynlib_extension;

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

static DynLib open_system_library() {
  auto result = DynLib::open(kSysLib);
  EXPECT_TRUE(result.has_value())
      << (result ? "" : result.error().message());
  return result ? *std::move(result) : DynLib{};
}

static DynLib get_system_library() {
  auto result = DynLib::get(kSysLib);
  EXPECT_TRUE(result.has_value())
      << (result ? "" : result.error().message());
  return result ? *std::move(result) : DynLib{};
}

TEST(DynLib, Extension) {
  EXPECT_STREQ(dynlib_extension(), kExpectExt);
}

TEST(DynLib, OpenSystemLibAndFindSymbol) {
  DynLib lib = open_system_library();
  ASSERT_TRUE(lib.valid());
  EXPECT_TRUE(static_cast<bool>(lib));
  auto sym = lib.symbol(kSysSym);
  ASSERT_TRUE(sym.has_value()) << sym.error().message();
  EXPECT_NE(*sym, nullptr);
}

TEST(DynLib, OpenNonexistentFails) {
  auto lib = DynLib::open("this_library_does_not_exist_zzz.xyz");
  ASSERT_FALSE(lib.has_value());
  EXPECT_TRUE(lib.error().IsNotFound());
  EXPECT_FALSE(lib.error().message().empty());
#if defined(_WIN32)
  EXPECT_NE(lib.error().message().find("LoadLibraryW"), std::string_view::npos);
  EXPECT_EQ(lib.error().native_error().domain, hyle::NativeErrorDomain::Win32);
  EXPECT_EQ(lib.error().native_error().code, 126);  // ERROR_MOD_NOT_FOUND
#else
  EXPECT_NE(lib.error().message().find("dlopen"), std::string_view::npos);
#endif
}

TEST(DynLib, OpenEmptyPathFails) {
  auto empty = DynLib::open("");
  auto null_path = DynLib::open(nullptr);
  ASSERT_FALSE(empty.has_value());
  ASSERT_FALSE(null_path.has_value());
  EXPECT_TRUE(empty.error().IsInvalidArgument());
  EXPECT_TRUE(null_path.error().IsInvalidArgument());
}

TEST(DynLib, MissingSymbolReturnsNull) {
  DynLib lib = open_system_library();
  ASSERT_TRUE(lib.valid());
  auto symbol = lib.symbol("definitely_not_a_real_symbol_zzz");
  ASSERT_FALSE(symbol.has_value());
  EXPECT_TRUE(symbol.error().IsNotFound());
#if defined(_WIN32)
  EXPECT_NE(symbol.error().message().find("GetProcAddress"),
            std::string_view::npos);
#else
  EXPECT_NE(symbol.error().message().find("dlsym"), std::string_view::npos);
#endif
}

TEST(DynLib, InvalidSymbolArgumentsReturnStatus) {
  DynLib lib = open_system_library();
  ASSERT_TRUE(lib.valid());
  auto missing = lib.symbol("definitely_not_a_real_symbol_zzz");
  EXPECT_FALSE(missing.has_value());
  auto found = lib.symbol(kSysSym);
  EXPECT_TRUE(found.has_value());
  auto null_name = lib.symbol(nullptr);
  ASSERT_FALSE(null_name.has_value());
  EXPECT_TRUE(null_name.error().IsInvalidArgument());
}

TEST(DynLib, TypedSymbolCallable) {
  DynLib lib = open_system_library();
  ASSERT_TRUE(lib.valid());
#if defined(_WIN32)
  auto fn = lib.symbol_as<unsigned long (*)()>(kSysSym);  // GetCurrentProcessId
  ASSERT_TRUE(fn.has_value());
  EXPECT_NE((*fn)(), 0u);  // our PID is nonzero
#else
  auto fn = lib.symbol_as<double (*)(double)>(kSysSym);  // cos / (getpid on mac)
  ASSERT_TRUE(fn.has_value());
#  if !defined(__APPLE__)
  EXPECT_DOUBLE_EQ((*fn)(0.0), 1.0);  // cos(0) == 1
#  endif
#endif
}

TEST(DynLib, MoveTransfersOwnership) {
  DynLib a = open_system_library();
  ASSERT_TRUE(a.valid());
  DynLib b = std::move(a);
  EXPECT_FALSE(a.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_TRUE(b.valid());
  EXPECT_TRUE(b.symbol(kSysSym).has_value());
}

TEST(DynLib, CloseInvalidates) {
  DynLib lib = open_system_library();
  ASSERT_TRUE(lib.valid());
  lib.close();
  EXPECT_FALSE(lib.valid());
  lib.close();  // double close is safe
  auto symbol = lib.symbol(kSysSym);
  ASSERT_FALSE(symbol.has_value());
  EXPECT_TRUE(symbol.error().IsFailedPrecondition());
}

TEST(DynLib, CurrentProcessImage) {
  auto result = DynLib::current();
  ASSERT_TRUE(result.has_value());
  DynLib self = *std::move(result);
  EXPECT_TRUE(self.valid());
  // The process image can resolve a libc symbol it links (POSIX dlopen(NULL));
  // on Windows GetModuleHandle(NULL) is the exe -- it may not export libc, so
  // only assert validity here, not a specific symbol.
}

// get(path): attach only if already mapped -- never load a missing library.
TEST(DynLib, GetMissingDoesNotLoad) {
  auto lib = DynLib::get("this_library_does_not_exist_zzz.xyz");
  ASSERT_FALSE(lib.has_value());
  EXPECT_TRUE(lib.error().IsNotFound());
}

// System libs are almost always already mapped into the process.
TEST(DynLib, GetAlreadyLoadedSystemLib) {
  DynLib lib = get_system_library();
  ASSERT_TRUE(lib.valid());
  auto sym = lib.symbol(kSysSym);
  ASSERT_TRUE(sym.has_value()) << sym.error().message();
  EXPECT_NE(*sym, nullptr);
}

// open then get: get attaches to the same already-resident image.
TEST(DynLib, GetAfterOpen) {
  DynLib loaded = open_system_library();
  ASSERT_TRUE(loaded.valid());
  DynLib attached = get_system_library();
  ASSERT_TRUE(attached.valid());
  EXPECT_TRUE(attached.symbol(kSysSym).has_value());
}

TEST(DynLib, GetEmptyPathFails) {
  auto empty = DynLib::get("");
  auto null_path = DynLib::get(nullptr);
  ASSERT_FALSE(empty.has_value());
  ASSERT_FALSE(null_path.has_value());
  EXPECT_TRUE(empty.error().IsInvalidArgument());
  EXPECT_TRUE(null_path.error().IsInvalidArgument());
}

TEST(DynLib, DoubleCloseSafeAfterGet) {
  DynLib lib = get_system_library();
  ASSERT_TRUE(lib.valid());
  lib.close();
  EXPECT_FALSE(lib.valid());
  lib.close();
}

TEST(DynLib, CurrentCanBeClosed) {
  auto result = DynLib::current();
  ASSERT_TRUE(result.has_value());
  DynLib self = *std::move(result);
  ASSERT_TRUE(self.valid());
  self.close();  // POSIX balances dlopen(nullptr); Windows drops its view only.
  EXPECT_FALSE(self.valid());
}

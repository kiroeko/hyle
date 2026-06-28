#include <gtest/gtest.h>

#include <cstdint>
#include <string_view>

#include "hyle/util/type_name.h"

using hyle::util::type_hash;
using hyle::util::type_name;

namespace {
struct MyWidget {};
enum class Color { Red };
}  // namespace

// Names are compiler-spelled (MSVC adds "class"/"struct"), so assert by
// substring containment, not exact equality.
TEST(TypeName, ContainsFundamentalName) {
  EXPECT_NE(type_name<int>().find("int"), std::string_view::npos);
  EXPECT_NE(type_name<double>().find("double"), std::string_view::npos);
}

TEST(TypeName, ContainsUserTypeName) {
  EXPECT_NE(type_name<MyWidget>().find("MyWidget"), std::string_view::npos);
  EXPECT_NE(type_name<Color>().find("Color"), std::string_view::npos);
}

TEST(TypeName, DistinctTypesDistinctNames) {
  EXPECT_NE(type_name<int>(), type_name<double>());
  EXPECT_NE(type_name<int>(), type_name<unsigned>());
  EXPECT_NE(type_name<MyWidget>(), type_name<Color>());
}

TEST(TypeName, NotEmptyOnSupportedCompiler) {
  // GCC/Clang/MSVC are all supported; the slice must have found something.
  EXPECT_FALSE(type_name<int>().empty());
}

// Pointer/const/reference qualified types are distinguished.
TEST(TypeName, QualifiersDistinguished) {
  EXPECT_NE(type_name<int>(), type_name<int*>());
  EXPECT_NE(type_name<int>(), type_name<const int>());
  EXPECT_NE(type_name<int*>(), type_name<int**>());
}

TEST(TypeHash, StableAcrossCalls) {
  constexpr std::uint64_t a = type_hash<MyWidget>();
  constexpr std::uint64_t b = type_hash<MyWidget>();
  EXPECT_EQ(a, b);
}

TEST(TypeHash, DistinctTypesDistinctHashes) {
  EXPECT_NE(type_hash<int>(), type_hash<double>());
  EXPECT_NE(type_hash<MyWidget>(), type_hash<Color>());
  EXPECT_NE(type_hash<int>(), type_hash<unsigned>());
  EXPECT_NE(type_hash<int*>(), type_hash<int>());
}

TEST(TypeHash, NonZeroForRealType) {
  EXPECT_NE(type_hash<MyWidget>(), 0u);
}

// type_name and type_hash are usable in constant expressions (compile-time IDs).
TEST(TypeName, UsableAtCompileTime) {
  static_assert(!type_name<int>().empty());
  static_assert(type_hash<int>() != type_hash<long>());
  SUCCEED();
}

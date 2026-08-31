#include "hyle/container/path.h"

#include <gtest/gtest.h>

#include <cstring>
#include <string>

#include "hyle/container/pod_vector.h"
#include "hyle/container/string_view.h"
#include "hyle/mem/allocator.h"

using namespace hyle::container;
using namespace hyle::container::literals;
using hyle::mem::system_allocator;

// normalize_path() writes into caller storage and returns a borrowed view, so
// tests can read like a truth table without any hidden allocation.
static std::string norm(const char* in) {
  char buf[256];
  const auto normalized = normalize_path(StringView{in}, buf);
  EXPECT_TRUE(normalized.has_value());
  if (!normalized) return {};
  EXPECT_EQ(normalized->data(), buf);
  EXPECT_TRUE(normalized->is_null_terminated());
  EXPECT_EQ(normalized->size(), std::strlen(buf));
  return std::string(normalized->data(), normalized->size());
}

// ---- extension (splitext semantics) -----------------------------------------
TEST(Path, Extension) {
  EXPECT_TRUE(extension("test.jpg"_sv) == ".jpg"_sv);
  EXPECT_TRUE(extension("a.tar.gz"_sv) == ".gz"_sv);  // last dot only
  EXPECT_TRUE(extension("archive.TAR"_sv) == ".TAR"_sv);
  EXPECT_TRUE(extension("test"_sv).empty());          // no dot
  EXPECT_TRUE(extension(".bashrc"_sv).empty());       // leading dot != extension
  EXPECT_TRUE(extension("dir.d/file"_sv).empty());    // dot is in a parent dir
  EXPECT_TRUE(extension(""_sv).empty());
  EXPECT_TRUE(extension("a/b.c"_sv) == ".c"_sv);
  // Back-to-front scan edge cases (guard the reverse-scan rewrite):
  EXPECT_TRUE(extension("..foo"_sv).empty());         // all leading dots, no ext
  EXPECT_TRUE(extension("..."_sv).empty());           // dots only
  EXPECT_TRUE(extension("a..b"_sv) == ".b"_sv);       // empty middle component ok
  EXPECT_TRUE(extension("x."_sv) == "."_sv);          // trailing dot = empty ext
  EXPECT_TRUE(extension("/a/b/c.h"_sv) == ".h"_sv);   // deep path, last comp only
  EXPECT_TRUE(extension("/a.b/c"_sv).empty());        // dot before last separator
}

TEST(Path, StripExtension) {
  EXPECT_TRUE(strip_extension("test.jpg"_sv) == "test"_sv);
  EXPECT_TRUE(strip_extension("a.tar.gz"_sv) == "a.tar"_sv);
  EXPECT_TRUE(strip_extension("test"_sv) == "test"_sv);  // unchanged, not empty
  EXPECT_TRUE(strip_extension(".bashrc"_sv) == ".bashrc"_sv);
  EXPECT_TRUE(strip_extension("a/b.c"_sv) == "a/b"_sv);
}

// ---- base / directory --------------------------------------------------------
TEST(Path, Base) {
  EXPECT_TRUE(base("/foo/bar.txt"_sv) == "bar.txt"_sv);
  EXPECT_TRUE(base("/foo/bar/"_sv) == "bar"_sv);   // trailing slash stripped
  EXPECT_TRUE(base("bar.txt"_sv) == "bar.txt"_sv);
  EXPECT_TRUE(base(""_sv) == "."_sv);              // empty -> "."
  EXPECT_TRUE(base("/"_sv) == "/"_sv);             // only slashes -> "/"
  EXPECT_TRUE(base("///"_sv) == "/"_sv);
  EXPECT_TRUE(base("foo"_sv) == "foo"_sv);
  EXPECT_TRUE(base("a\\b"_sv) == "b"_sv);          // backslash is a separator
}

TEST(Path, Directory) {
  EXPECT_TRUE(directory("/foo/bar.txt"_sv) == "/foo"_sv);
  EXPECT_TRUE(directory("/foo/bar/"_sv) == "/foo"_sv);
  EXPECT_TRUE(directory("bar.txt"_sv) == "."_sv);  // no dir -> "."
  EXPECT_TRUE(directory(""_sv) == "."_sv);
  EXPECT_TRUE(directory("/foo"_sv) == "/"_sv);     // parent is root
  EXPECT_TRUE(directory("/"_sv) == "/"_sv);
  EXPECT_TRUE(directory("//foo"_sv) == "/"_sv);    // clay read OOB here
  EXPECT_TRUE(directory("a/b/c"_sv) == "a/b"_sv);
}

// ---- classification ----------------------------------------------------------
TEST(Path, IsAbsolute) {
  EXPECT_TRUE(is_absolute("/foo"_sv));
  EXPECT_TRUE(is_absolute("/"_sv));
  EXPECT_TRUE(is_absolute("C:\\win"_sv));   // Windows drive root
  EXPECT_TRUE(is_absolute("c:/win"_sv));    // either separator
  EXPECT_TRUE(is_absolute("\\\\srv\\sh"_sv));  // UNC
  EXPECT_FALSE(is_absolute("foo/bar"_sv));
  EXPECT_FALSE(is_absolute(""_sv));
  EXPECT_FALSE(is_absolute("C:"_sv));       // drive-relative, NOT absolute
  EXPECT_FALSE(is_absolute("C:foo"_sv));
}

TEST(Path, IsUnc) {
  EXPECT_TRUE(is_unc("\\\\server\\share"_sv));
  EXPECT_FALSE(is_unc("/foo"_sv));
  EXPECT_FALSE(is_unc("\\foo"_sv));   // single backslash
  EXPECT_FALSE(is_unc(""_sv));
}

// ---- iteration ---------------------------------------------------------------
TEST(Path, PopFirst) {
  StringView p = "/usr/local/bin"_sv;
  EXPECT_TRUE(pop_first(&p) == "/"_sv);     // leading sep pops as "/"
  EXPECT_TRUE(pop_first(&p) == "usr"_sv);
  EXPECT_TRUE(pop_first(&p) == "local"_sv);
  EXPECT_TRUE(pop_first(&p) == "bin"_sv);
  EXPECT_TRUE(p.empty());
  EXPECT_TRUE(pop_first(&p) == "."_sv);     // empty -> "."
}

TEST(Path, PopLast) {
  StringView p = "/usr/local/bin"_sv;
  EXPECT_TRUE(pop_last(&p) == "bin"_sv);
  EXPECT_TRUE(p == "/usr/local"_sv);
  EXPECT_TRUE(pop_last(&p) == "local"_sv);
  EXPECT_TRUE(p == "/usr"_sv);
}

// ---- normalize (table-driven) ------------------------------------------------
TEST(Path, NormalizeCollapsesRedundant) {
  EXPECT_EQ(norm("a//b"), "a/b");
  EXPECT_EQ(norm("a/./b"), "a/b");
  EXPECT_EQ(norm("./a"), "a");
  EXPECT_EQ(norm("a/b/"), "a/b");        // trailing slash dropped
  EXPECT_EQ(norm("a///b//c"), "a/b/c");
  EXPECT_EQ(norm("/a/./b/"), "/a/b");
}

TEST(Path, NormalizeResolvesParent) {
  EXPECT_EQ(norm("a/b/../c"), "a/c");
  EXPECT_EQ(norm("a/../b"), "b");
  EXPECT_EQ(norm("a/b/c/../../d"), "a/d");
  EXPECT_EQ(norm("/a/../../b"), "/b");   // can't pop above absolute root
  EXPECT_EQ(norm("/.."), "/");
}

TEST(Path, NormalizeKeepsIrreducibleParent) {
  EXPECT_EQ(norm("../b"), "../b");        // relative: leading .. is kept
  EXPECT_EQ(norm("a/../../b"), "../b");
  EXPECT_EQ(norm("../../x"), "../../x");
}

TEST(Path, NormalizeEdges) {
  EXPECT_EQ(norm(""), ".");              // empty relative -> "."
  EXPECT_EQ(norm("."), ".");
  EXPECT_EQ(norm("/"), "/");
  EXPECT_EQ(norm("///"), "/");
  EXPECT_EQ(norm("a/.."), ".");          // collapses to nothing -> "."
}

// ---- caller-owned producing helpers ------------------------------------------
TEST(Path, Join) {
  char output[32];
  EXPECT_EQ(join_path("a"_sv, "b"_sv, output), 3u);
  EXPECT_STREQ(output, "a/b");

  EXPECT_EQ(join_path("/usr"_sv, "bin"_sv, output), 8u);
  EXPECT_STREQ(output, "/usr/bin");
}

TEST(Path, JoinPathRange) {
  char buf[64];
  StringView parts[] = {"/usr"_sv, "local"_sv, "bin"_sv};
  std::size_t n = join_path(parts, buf);
  EXPECT_EQ(n, std::strlen(buf));
  EXPECT_STREQ(buf, "/usr/local/bin");

  // Interior separators collapse; leading on [0] and trailing on last kept.
  StringView parts2[] = {"a/"_sv, "/b/"_sv, "c/"_sv};
  join_path(parts2, buf);
  EXPECT_STREQ(buf, "a/b/c/");
}

TEST(Path, JoinPathReportsRequiredSizeWhenTruncated) {
  char buf[8];
  StringView parts[] = {"/usr"_sv, "local"_sv, "bin"_sv};
  std::size_t need = join_path(parts, buf);
  EXPECT_EQ(need, 14u);                   // full "/usr/local/bin"
  EXPECT_EQ(join_path(parts, std::span<char>{}), need);  // size query
  EXPECT_EQ(std::strlen(buf), 7u);        // 7 written + NUL
  EXPECT_EQ(std::strncmp(buf, "/usr/lo", 7), 0);
}

TEST(Path, SplitAll) {
  PodVector<StringView> parts =
      split_all("/usr/local/bin"_sv, system_allocator());
  ASSERT_EQ(parts.size(), 4u);
  EXPECT_TRUE(parts[0] == "/"_sv);
  EXPECT_TRUE(parts[1] == "usr"_sv);
  EXPECT_TRUE(parts[2] == "local"_sv);
  EXPECT_TRUE(parts[3] == "bin"_sv);
}

TEST(Path, ReplaceExtension) {
  char output[32];
  EXPECT_EQ(replace_extension("test.jpg"_sv, ".png"_sv, output), 8u);
  EXPECT_STREQ(output, "test.png");

  // No existing extension: ext is appended (clay returned null; HYLE appends).
  EXPECT_EQ(replace_extension("test"_sv, ".png"_sv, output), 8u);
  EXPECT_STREQ(output, "test.png");
  EXPECT_EQ(replace_extension("test"_sv, ".png"_sv, std::span<char>{}), 8u);
}

TEST(Path, NormalizePathRejectsSmallStorageAndSupportsExactOverlap) {
  char too_small[4];
  EXPECT_FALSE(normalize_path("a/../b"_sv, too_small).has_value());

  char in_place[] = "a/../b";
  const auto normalized = normalize_path(StringView{in_place}, in_place);
  ASSERT_TRUE(normalized.has_value());
  EXPECT_EQ(normalized->data(), in_place);
  EXPECT_TRUE(*normalized == "b"_sv);
}

TEST(Path, NormalizePathUsesViewLengthPastEmbeddedNul) {
  const char input[] = {'a', '\0', 'b', '/', '.', '/', 'c'};
  char output[sizeof(input) + 1];
  const auto normalized =
      normalize_path(StringView{input, sizeof(input), false}, output);
  ASSERT_TRUE(normalized.has_value());
  const char expected[] = {'a', '\0', 'b', '/', 'c'};
  EXPECT_EQ(normalized->size(), sizeof(expected));
  EXPECT_EQ(std::memcmp(normalized->data(), expected, sizeof(expected)), 0);
  EXPECT_EQ(output[sizeof(expected)], '\0');
}

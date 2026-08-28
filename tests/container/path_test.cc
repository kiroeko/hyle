#include "hyle/container/path.h"

#include <gtest/gtest.h>

#include <cstring>
#include <string>

#include "hyle/container/pod_vector.h"
#include "hyle/container/string_ref.h"
#include "hyle/mem/allocator.h"

using namespace hyle::container;
using namespace hyle::container::literals;
using hyle::mem::system_allocator;

// normalize_path() writes into caller storage and returns a borrowed view, so
// tests can read like a truth table without any hidden allocation.
static std::string norm(const char* in) {
  char buf[256];
  const auto normalized = normalize_path(string_ref{in}, buf);
  EXPECT_TRUE(normalized.has_value());
  if (!normalized) return {};
  EXPECT_EQ(normalized->data, buf);
  EXPECT_TRUE(normalized->null_terminated);
  EXPECT_EQ(normalized->size, std::strlen(buf));
  return std::string(normalized->data, normalized->size);
}

// ---- extension (splitext semantics) -----------------------------------------
TEST(Path, Extension) {
  EXPECT_TRUE(extension("test.jpg"_sr) == ".jpg"_sr);
  EXPECT_TRUE(extension("a.tar.gz"_sr) == ".gz"_sr);  // last dot only
  EXPECT_TRUE(extension("archive.TAR"_sr) == ".TAR"_sr);
  EXPECT_TRUE(extension("test"_sr).empty());          // no dot
  EXPECT_TRUE(extension(".bashrc"_sr).empty());       // leading dot != extension
  EXPECT_TRUE(extension("dir.d/file"_sr).empty());    // dot is in a parent dir
  EXPECT_TRUE(extension(""_sr).empty());
  EXPECT_TRUE(extension("a/b.c"_sr) == ".c"_sr);
  // Back-to-front scan edge cases (guard the reverse-scan rewrite):
  EXPECT_TRUE(extension("..foo"_sr).empty());         // all leading dots, no ext
  EXPECT_TRUE(extension("..."_sr).empty());           // dots only
  EXPECT_TRUE(extension("a..b"_sr) == ".b"_sr);       // empty middle component ok
  EXPECT_TRUE(extension("x."_sr) == "."_sr);          // trailing dot = empty ext
  EXPECT_TRUE(extension("/a/b/c.h"_sr) == ".h"_sr);   // deep path, last comp only
  EXPECT_TRUE(extension("/a.b/c"_sr).empty());        // dot before last separator
}

TEST(Path, StripExtension) {
  EXPECT_TRUE(strip_extension("test.jpg"_sr) == "test"_sr);
  EXPECT_TRUE(strip_extension("a.tar.gz"_sr) == "a.tar"_sr);
  EXPECT_TRUE(strip_extension("test"_sr) == "test"_sr);  // unchanged, not empty
  EXPECT_TRUE(strip_extension(".bashrc"_sr) == ".bashrc"_sr);
  EXPECT_TRUE(strip_extension("a/b.c"_sr) == "a/b"_sr);
}

// ---- base / directory --------------------------------------------------------
TEST(Path, Base) {
  EXPECT_TRUE(base("/foo/bar.txt"_sr) == "bar.txt"_sr);
  EXPECT_TRUE(base("/foo/bar/"_sr) == "bar"_sr);   // trailing slash stripped
  EXPECT_TRUE(base("bar.txt"_sr) == "bar.txt"_sr);
  EXPECT_TRUE(base(""_sr) == "."_sr);              // empty -> "."
  EXPECT_TRUE(base("/"_sr) == "/"_sr);             // only slashes -> "/"
  EXPECT_TRUE(base("///"_sr) == "/"_sr);
  EXPECT_TRUE(base("foo"_sr) == "foo"_sr);
  EXPECT_TRUE(base("a\\b"_sr) == "b"_sr);          // backslash is a separator
}

TEST(Path, Directory) {
  EXPECT_TRUE(directory("/foo/bar.txt"_sr) == "/foo"_sr);
  EXPECT_TRUE(directory("/foo/bar/"_sr) == "/foo"_sr);
  EXPECT_TRUE(directory("bar.txt"_sr) == "."_sr);  // no dir -> "."
  EXPECT_TRUE(directory(""_sr) == "."_sr);
  EXPECT_TRUE(directory("/foo"_sr) == "/"_sr);     // parent is root
  EXPECT_TRUE(directory("/"_sr) == "/"_sr);
  EXPECT_TRUE(directory("//foo"_sr) == "/"_sr);    // clay read OOB here
  EXPECT_TRUE(directory("a/b/c"_sr) == "a/b"_sr);
}

// ---- classification ----------------------------------------------------------
TEST(Path, IsAbsolute) {
  EXPECT_TRUE(is_absolute("/foo"_sr));
  EXPECT_TRUE(is_absolute("/"_sr));
  EXPECT_TRUE(is_absolute("C:\\win"_sr));   // Windows drive root
  EXPECT_TRUE(is_absolute("c:/win"_sr));    // either separator
  EXPECT_TRUE(is_absolute("\\\\srv\\sh"_sr));  // UNC
  EXPECT_FALSE(is_absolute("foo/bar"_sr));
  EXPECT_FALSE(is_absolute(""_sr));
  EXPECT_FALSE(is_absolute("C:"_sr));       // drive-relative, NOT absolute
  EXPECT_FALSE(is_absolute("C:foo"_sr));
}

TEST(Path, IsUnc) {
  EXPECT_TRUE(is_unc("\\\\server\\share"_sr));
  EXPECT_FALSE(is_unc("/foo"_sr));
  EXPECT_FALSE(is_unc("\\foo"_sr));   // single backslash
  EXPECT_FALSE(is_unc(""_sr));
}

// ---- iteration ---------------------------------------------------------------
TEST(Path, PopFirst) {
  string_ref p = "/usr/local/bin"_sr;
  EXPECT_TRUE(pop_first(&p) == "/"_sr);     // leading sep pops as "/"
  EXPECT_TRUE(pop_first(&p) == "usr"_sr);
  EXPECT_TRUE(pop_first(&p) == "local"_sr);
  EXPECT_TRUE(pop_first(&p) == "bin"_sr);
  EXPECT_TRUE(p.empty());
  EXPECT_TRUE(pop_first(&p) == "."_sr);     // empty -> "."
}

TEST(Path, PopLast) {
  string_ref p = "/usr/local/bin"_sr;
  EXPECT_TRUE(pop_last(&p) == "bin"_sr);
  EXPECT_TRUE(p == "/usr/local"_sr);
  EXPECT_TRUE(pop_last(&p) == "local"_sr);
  EXPECT_TRUE(p == "/usr"_sr);
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
  EXPECT_EQ(join_path("a"_sr, "b"_sr, output), 3u);
  EXPECT_STREQ(output, "a/b");

  EXPECT_EQ(join_path("/usr"_sr, "bin"_sr, output), 8u);
  EXPECT_STREQ(output, "/usr/bin");
}

TEST(Path, JoinPathRange) {
  char buf[64];
  string_ref parts[] = {"/usr"_sr, "local"_sr, "bin"_sr};
  std::size_t n = join_path(parts, buf);
  EXPECT_EQ(n, std::strlen(buf));
  EXPECT_STREQ(buf, "/usr/local/bin");

  // Interior separators collapse; leading on [0] and trailing on last kept.
  string_ref parts2[] = {"a/"_sr, "/b/"_sr, "c/"_sr};
  join_path(parts2, buf);
  EXPECT_STREQ(buf, "a/b/c/");
}

TEST(Path, JoinPathReportsRequiredSizeWhenTruncated) {
  char buf[8];
  string_ref parts[] = {"/usr"_sr, "local"_sr, "bin"_sr};
  std::size_t need = join_path(parts, buf);
  EXPECT_EQ(need, 14u);                   // full "/usr/local/bin"
  EXPECT_EQ(join_path(parts, std::span<char>{}), need);  // size query
  EXPECT_EQ(std::strlen(buf), 7u);        // 7 written + NUL
  EXPECT_EQ(std::strncmp(buf, "/usr/lo", 7), 0);
}

TEST(Path, SplitAll) {
  PodVector<string_ref> parts =
      split_all("/usr/local/bin"_sr, system_allocator());
  ASSERT_EQ(parts.size(), 4u);
  EXPECT_TRUE(parts[0] == "/"_sr);
  EXPECT_TRUE(parts[1] == "usr"_sr);
  EXPECT_TRUE(parts[2] == "local"_sr);
  EXPECT_TRUE(parts[3] == "bin"_sr);
}

TEST(Path, ReplaceExtension) {
  char output[32];
  EXPECT_EQ(replace_extension("test.jpg"_sr, ".png"_sr, output), 8u);
  EXPECT_STREQ(output, "test.png");

  // No existing extension: ext is appended (clay returned null; HYLE appends).
  EXPECT_EQ(replace_extension("test"_sr, ".png"_sr, output), 8u);
  EXPECT_STREQ(output, "test.png");
  EXPECT_EQ(replace_extension("test"_sr, ".png"_sr, std::span<char>{}), 8u);
}

TEST(Path, NormalizePathRejectsSmallStorageAndSupportsExactOverlap) {
  char too_small[4];
  EXPECT_FALSE(normalize_path("a/../b"_sr, too_small).has_value());

  char in_place[] = "a/../b";
  const auto normalized = normalize_path(string_ref{in_place}, in_place);
  ASSERT_TRUE(normalized.has_value());
  EXPECT_EQ(normalized->data, in_place);
  EXPECT_TRUE(*normalized == "b"_sr);
}

TEST(Path, NormalizePathUsesViewLengthPastEmbeddedNul) {
  const char input[] = {'a', '\0', 'b', '/', '.', '/', 'c'};
  char output[sizeof(input) + 1];
  const auto normalized =
      normalize_path(string_ref{input, sizeof(input), false}, output);
  ASSERT_TRUE(normalized.has_value());
  const char expected[] = {'a', '\0', 'b', '/', 'c'};
  EXPECT_EQ(normalized->size, sizeof(expected));
  EXPECT_EQ(std::memcmp(normalized->data, expected, sizeof(expected)), 0);
  EXPECT_EQ(output[sizeof(expected)], '\0');
}

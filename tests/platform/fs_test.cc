// The test writes files with std::fopen to check sizes/contents; opt out of
// MSVC's C4996 CRT deprecation (the uses here are safe).
#if defined(_WIN32)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

#include "hyle/platform/fs.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if !defined(_WIN32)
#  include <unistd.h>  // symlink() for the recursive-delete safety test
#endif

using namespace hyle::platform;

namespace {

// A unique scratch directory under the system temp dir, created fresh for each
// test and recursively removed on teardown -- so the suite touches the real
// filesystem (the whole point) without leaking state between runs.
class FsTest : public ::testing::Test {
 protected:
  std::string root;

  void SetUp() override {
    char tmp[4096];
    std::size_t n = temp_directory(tmp, sizeof(tmp));
    ASSERT_GT(n, 0u);
    ASSERT_LT(n, sizeof(tmp));
    // std::string concat (not snprintf into a fixed buffer, which GCC flags as
    // possibly-truncated since tmp can be up to 4096 chars).
    root = std::string(tmp, n) + "/hyle_fs_test_" +
           std::to_string(reinterpret_cast<std::uintptr_t>(this));
    remove_directory_recursive(root.c_str());  // clear any stale leftover
    ASSERT_TRUE(make_directory(root.c_str()));
  }
  void TearDown() override { remove_directory_recursive(root.c_str()); }

  std::string path(const char* rel) const { return root + "/" + rel; }
};

}  // namespace

TEST_F(FsTest, StatExistsDirectoryFile) {
  EXPECT_TRUE(exists(root.c_str()));
  EXPECT_TRUE(is_directory(root.c_str()));

  std::string f = path("a.txt");
  EXPECT_FALSE(exists(f.c_str()));
  ASSERT_TRUE(touch(f.c_str()));
  EXPECT_TRUE(exists(f.c_str()));
  EXPECT_FALSE(is_directory(f.c_str()));

  FileInfo s = stat(f.c_str());
  EXPECT_TRUE(s.exists);
  EXPECT_FALSE(s.is_directory);
  EXPECT_EQ(s.size, 0u);

  // A missing path stats as {exists=false}, never throws/aborts.
  FileInfo m = stat(path("nope").c_str());
  EXPECT_FALSE(m.exists);
}

TEST_F(FsTest, MakeDirectoryIdempotent) {
  std::string d = path("sub");
  EXPECT_TRUE(make_directory(d.c_str()));
  EXPECT_TRUE(make_directory(d.c_str()));  // already exists -> still true
  EXPECT_TRUE(is_directory(d.c_str()));
}

TEST_F(FsTest, MakeDirectoryRecursive) {
  std::string deep = path("x/y/z");
  EXPECT_TRUE(make_directory_recursive(deep.c_str()));
  EXPECT_TRUE(is_directory(deep.c_str()));
  EXPECT_TRUE(is_directory(path("x/y").c_str()));
  EXPECT_TRUE(is_directory(path("x").c_str()));
}

TEST_F(FsTest, FileSizeReflectsWrite) {
  std::string f = path("data.bin");
  std::FILE* fp = std::fopen(f.c_str(), "wb");
  ASSERT_NE(fp, nullptr);
  const char payload[] = "0123456789";
  std::fwrite(payload, 1, 10, fp);
  std::fclose(fp);
  EXPECT_EQ(file_size(f.c_str()), 10u);
}

TEST_F(FsTest, RenameAndRemoveFile) {
  std::string a = path("a.txt"), b = path("b.txt");
  ASSERT_TRUE(touch(a.c_str()));
  EXPECT_TRUE(rename_path(a.c_str(), b.c_str()));
  EXPECT_FALSE(exists(a.c_str()));
  EXPECT_TRUE(exists(b.c_str()));
  EXPECT_TRUE(remove_file(b.c_str()));
  EXPECT_FALSE(exists(b.c_str()));
  EXPECT_FALSE(remove_file(b.c_str()));  // already gone
}

TEST_F(FsTest, CopyFile) {
  std::string a = path("src.txt"), b = path("dst.txt");
  std::FILE* fp = std::fopen(a.c_str(), "wb");
  ASSERT_NE(fp, nullptr);
  std::fwrite("hello", 1, 5, fp);
  std::fclose(fp);
  EXPECT_TRUE(copy_file(a.c_str(), b.c_str()));
  EXPECT_EQ(file_size(b.c_str()), 5u);
  char got[8] = {0};
  fp = std::fopen(b.c_str(), "rb");
  ASSERT_NE(fp, nullptr);
  EXPECT_EQ(std::fread(got, 1, 5, fp), 5u);  // also silences warn_unused_result
  std::fclose(fp);
  EXPECT_STREQ(got, "hello");
}

// Larger-than-one-chunk copy: exercises the Linux copy_file_range loop (and the
// userspace 64 KiB-buffer loop elsewhere) over a multi-block, byte-patterned
// file, then verifies the bytes survive intact.
TEST_F(FsTest, CopyFileLargeContentIntact) {
  std::string a = path("big_src.bin"), b = path("big_dst.bin");
  constexpr std::size_t kN = 300 * 1024;  // > 64 KiB buffer, > 1 page
  std::vector<unsigned char> data(kN);
  for (std::size_t i = 0; i < kN; ++i) data[i] = static_cast<unsigned char>(i * 31 + 7);
  std::FILE* fp = std::fopen(a.c_str(), "wb");
  ASSERT_NE(fp, nullptr);
  ASSERT_EQ(std::fwrite(data.data(), 1, kN, fp), kN);
  std::fclose(fp);

  ASSERT_TRUE(copy_file(a.c_str(), b.c_str()));
  EXPECT_EQ(file_size(b.c_str()), kN);
  std::vector<unsigned char> back(kN, 0);
  fp = std::fopen(b.c_str(), "rb");
  ASSERT_NE(fp, nullptr);
  EXPECT_EQ(std::fread(back.data(), 1, kN, fp), kN);
  std::fclose(fp);
  EXPECT_EQ(back, data);
}

TEST_F(FsTest, DirectoryIterateVisitsEntriesNotDotDot) {
  ASSERT_TRUE(touch(path("f1").c_str()));
  ASSERT_TRUE(touch(path("f2").c_str()));
  ASSERT_TRUE(make_directory(path("d1").c_str()));

  std::vector<std::string> files, dirs;
  bool ok = for_each_entry(root.c_str(), [&](const char* name, bool is_dir) {
    if (is_dir)
      dirs.emplace_back(name);
    else
      files.emplace_back(name);
  });
  ASSERT_TRUE(ok);
  EXPECT_EQ(files.size(), 2u);
  EXPECT_EQ(dirs.size(), 1u);
  // "." / ".." must be excluded.
  for (const auto& n : files) EXPECT_NE(n, ".");
  for (const auto& n : dirs) EXPECT_EQ(n, "d1");
}

TEST_F(FsTest, ForEachEntryEarlyStop) {
  for (int i = 0; i < 5; ++i)
    ASSERT_TRUE(touch(path(("e" + std::to_string(i)).c_str()).c_str()));
  int seen = 0;
  for_each_entry(root.c_str(), [&](const char*, bool) -> bool {
    ++seen;
    return seen < 2;  // stop after the second entry
  });
  EXPECT_EQ(seen, 2);
}

TEST_F(FsTest, IterateMissingDirReturnsFalse) {
  bool ok = for_each_entry(path("ghost").c_str(),
                           [](const char*, bool) { /* never called */ });
  EXPECT_FALSE(ok);
}

TEST_F(FsTest, RemoveDirectoryRecursive) {
  ASSERT_TRUE(make_directory_recursive(path("tree/inner").c_str()));
  ASSERT_TRUE(touch(path("tree/top.txt").c_str()));
  ASSERT_TRUE(touch(path("tree/inner/leaf.txt").c_str()));
  // A non-empty dir cannot be removed by the plain (empty-only) call.
  EXPECT_FALSE(remove_directory(path("tree").c_str()));
  EXPECT_TRUE(remove_directory_recursive(path("tree").c_str()));
  EXPECT_FALSE(exists(path("tree").c_str()));
}

// A directory whose TOTAL path exceeds MAX_PATH (260). On Windows this only
// works if fs prefixes "\\?\" (the ...W APIs otherwise reject > 260); without
// it make_directory_recursive would fail partway. On POSIX it just exercises a
// deep mkdir (limit is 4096). Builds many short segments so each component is
// legal but the cumulative path is long, then creates/stats/removes it.
TEST_F(FsTest, LongPathBeyondMaxPath) {
  std::string deep = path("L");
  while (deep.size() < 320) deep += "/abcdefgh";  // push well past 260
  ASSERT_LT(deep.size(), 1024u);
  ASSERT_TRUE(make_directory_recursive(deep.c_str()));
  EXPECT_TRUE(is_directory(deep.c_str()));
  // A file at the deep leaf round-trips too (touch -> exists).
  std::string leaf = deep + "/leaf.txt";
  ASSERT_TRUE(touch(leaf.c_str()));
  EXPECT_TRUE(exists(leaf.c_str()));
  // Recursive remove must also handle the long path (iterate + rebuild children).
  EXPECT_TRUE(remove_directory_recursive(path("L").c_str()));
  EXPECT_FALSE(exists(path("L").c_str()));
}

#if !defined(_WIN32)
// Recursive delete must NOT follow a symlink-to-directory and delete the target
// (blast-radius safety). POSIX-only: symlink creation on Windows needs a
// privilege; this guards the DT_LNK / lstat-on-DT_UNKNOWN path in fs.cc.
TEST_F(FsTest, RemoveDirectoryRecursiveDoesNotFollowSymlink) {
  // outside/keep.txt is the target that must SURVIVE.
  ASSERT_TRUE(make_directory(path("outside").c_str()));
  ASSERT_TRUE(touch(path("outside/keep.txt").c_str()));
  // victim/ contains a symlink "link" -> ../outside.
  ASSERT_TRUE(make_directory(path("victim").c_str()));
  std::string link = path("victim/link");
  std::string target = path("outside");
  ASSERT_EQ(::symlink(target.c_str(), link.c_str()), 0);

  EXPECT_TRUE(remove_directory_recursive(path("victim").c_str()));
  EXPECT_FALSE(exists(path("victim").c_str()));
  // The link's target and its contents must be untouched.
  EXPECT_TRUE(exists(path("outside").c_str()));
  EXPECT_TRUE(exists(path("outside/keep.txt").c_str()));
}
#endif

TEST_F(FsTest, SpecialDirectoriesNonEmpty) {
  char buf[4096];
  EXPECT_GT(current_directory(buf, sizeof(buf)), 0u);
  EXPECT_GT(temp_directory(buf, sizeof(buf)), 0u);
  EXPECT_GT(home_directory(buf, sizeof(buf)), 0u);
}

// Non-ASCII UTF-8 path round-trips through every fs op. POSIX is byte-
// transparent; Windows must convert UTF-8 <-> UTF-16 (else the ...A APIs would
// mis-decode under the active code page). The directory name mixes CJK + emoji.
TEST_F(FsTest, Utf8NonAsciiPath) {
  std::string dir = path("\xE6\xB5\x8B\xE8\xAF\x95_\xF0\x9F\x9A\x80");  // 测试_🚀
  ASSERT_TRUE(make_directory(dir.c_str()));
  EXPECT_TRUE(is_directory(dir.c_str()));

  std::string file = dir + "/\xE6\x96\x87\xE4\xBB\xB6.txt";  // 文件.txt
  ASSERT_TRUE(touch(file.c_str()));
  EXPECT_TRUE(exists(file.c_str()));

  // The non-ASCII entry name comes back intact through the visitor (UTF-8).
  bool found = false;
  for_each_entry(dir.c_str(), [&](const char* name, bool is_dir) {
    if (!is_dir && std::strcmp(name, "\xE6\x96\x87\xE4\xBB\xB6.txt") == 0)
      found = true;
  });
  EXPECT_TRUE(found);
  EXPECT_TRUE(remove_directory_recursive(dir.c_str()));
}

TEST_F(FsTest, ChangeDirectoryRoundTrip) {
  char before[4096];
  ASSERT_GT(current_directory(before, sizeof(before)), 0u);
  ASSERT_TRUE(change_directory(root.c_str()));
  char now[4096];
  ASSERT_GT(current_directory(now, sizeof(now)), 0u);
  // cwd now ends with our scratch dir's base name.
  EXPECT_NE(std::strstr(now, "hyle_fs_test_"), nullptr);
  EXPECT_TRUE(change_directory(before));  // restore for other tests
}

TEST(FsEnv, GetEnvVarSetAndUnset) {
  char buf[4096];
  // PATH is set on every platform CI runs on.
  std::size_t n = get_env_var("PATH", buf, sizeof(buf));
  EXPECT_NE(n, kEnvNotSet);
  EXPECT_GT(n, 0u);

  // A name that cannot exist -> kEnvNotSet (distinct from set-but-empty).
  EXPECT_EQ(get_env_var("HYLE_DEFINITELY_UNSET_XYZZY_123", buf, sizeof(buf)),
            kEnvNotSet);
}

TEST(FsEnv, BufferFillTruncates) {
  // current_directory into a tiny buffer: NUL-terminated, returns full length.
  char tiny[4];
  std::size_t full = current_directory(tiny, sizeof(tiny));
  EXPECT_EQ(std::strlen(tiny), 3u);  // 3 chars + NUL
  EXPECT_GE(full, std::strlen(tiny));
}

TEST_F(FsTest, MissingPathDoesNotExist) {
  std::string missing = root + "/no_such_file_xyz";
  EXPECT_FALSE(exists(missing.c_str()));
  FileInfo info = stat(missing.c_str());
  EXPECT_FALSE(info.exists);
}

TEST_F(FsTest, TouchCreatesEmptyFile) {
  std::string f = root + "/touched.txt";
  ASSERT_TRUE(touch(f.c_str()));
  EXPECT_TRUE(exists(f.c_str()));
  EXPECT_EQ(file_size(f.c_str()), 0u);
  // second touch is idempotent (updates mtime or no-op success)
  EXPECT_TRUE(touch(f.c_str()));
}

TEST_F(FsTest, RemoveNonEmptyDirectoryFails) {
  // Non-recursive remove_directory requires empty dir.
  std::string sub = root + "/nonempty";
  ASSERT_TRUE(make_directory(sub.c_str()));
  ASSERT_TRUE(touch((sub + "/x").c_str()));
  EXPECT_FALSE(remove_directory(sub.c_str()));
  EXPECT_TRUE(exists(sub.c_str()));
  EXPECT_TRUE(remove_directory_recursive(sub.c_str()));
}

TEST_F(FsTest, IsDirectoryDistinguishesFile) {
  std::string f = root + "/plain.bin";
  ASSERT_TRUE(touch(f.c_str()));
  EXPECT_TRUE(exists(f.c_str()));
  EXPECT_FALSE(is_directory(f.c_str()));
  EXPECT_TRUE(is_directory(root.c_str()));
}

TEST(FsEnv, EnvVarTinyBufferTruncates) {
  char tiny[4];
  std::size_t n = get_env_var("PATH", tiny, sizeof(tiny));
  if (n == kEnvNotSet) {
    GTEST_SKIP() << "PATH unset in this environment";
  }
  EXPECT_EQ(std::strlen(tiny), 3u);
  EXPECT_GE(n, 3u);
}

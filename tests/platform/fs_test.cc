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

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>  // WriteFile to prove the temp-file handle is writable
#else
#  include <unistd.h>   // symlink() + write() to prove the temp-file fd is writable
#endif

using namespace hyle::platform;

namespace {
// Write `n` bytes through a raw OS fd/HANDLE (from create_temp_file). Returns
// true if all bytes were written -- proves the handle is genuinely open for write.
bool write_all_fd(std::intptr_t fd, const void* data, std::size_t n) {
#if defined(_WIN32)
  DWORD wrote = 0;
  return WriteFile(reinterpret_cast<HANDLE>(fd), data, static_cast<DWORD>(n),
                   &wrote, nullptr) && wrote == n;
#else
  return ::write(static_cast<int>(fd), data, n) == static_cast<ssize_t>(n);
#endif
}
}  // namespace

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

TEST(FsBoundary, NullInputsAndInvalidOpenFlagsFailSafely) {
  EXPECT_FALSE(make_directory_recursive(nullptr));
  EXPECT_FALSE(remove_directory_recursive(nullptr));
  EXPECT_FALSE(touch(nullptr));
  EXPECT_EQ(get_env_var(nullptr, nullptr, 0), kEnvNotSet);
  EXPECT_FALSE(open_file(nullptr, kOpenRead).has_value());
  EXPECT_FALSE(open_file("unused", 0).has_value());
  EXPECT_FALSE(
      open_file("unused", kOpenCreate | kOpenTruncate).has_value());

  char tmp[4] = {'x', 'x', 'x', '\0'};
  const std::size_t need = temp_directory(nullptr, sizeof(tmp));
  EXPECT_GT(need, 0u);
}

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

#if !defined(_WIN32)
TEST_F(FsTest, CopyFileHardLinkAliasIsNoOp) {
  std::string a = path("hardlink_src.txt"), b = path("hardlink_dst.txt");
  std::FILE* fp = std::fopen(a.c_str(), "wb");
  ASSERT_NE(fp, nullptr);
  ASSERT_EQ(std::fwrite("keep me", 1, 7, fp), 7u);
  std::fclose(fp);
  ASSERT_EQ(::link(a.c_str(), b.c_str()), 0);

  ASSERT_TRUE(copy_file(a.c_str(), b.c_str()));
  EXPECT_EQ(file_size(a.c_str()), 7u);
  EXPECT_EQ(file_size(b.c_str()), 7u);
  char got[8] = {};
  fp = std::fopen(a.c_str(), "rb");
  ASSERT_NE(fp, nullptr);
  EXPECT_EQ(std::fread(got, 1, 7, fp), 7u);
  std::fclose(fp);
  EXPECT_STREQ(got, "keep me");
}
#endif

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

// Recursive delete must not follow a symlink/junction-to-directory -- at the ROOT
// or as a child -- and wipe its target (data-loss guard; platform-audit #1: the
// Windows junction path was untested since the case above is POSIX-only; #2: root
// symlink). Cross-platform via create_symlink, which needs privilege on Windows
// (Developer Mode/admin) -> SKIP if unavailable.
TEST_F(FsTest, RemoveRecursiveDoesNotFollowDirLink) {
  ASSERT_TRUE(make_directory(path("keep").c_str()));
  ASSERT_TRUE(touch(path("keep/data.txt").c_str()));

  std::string rootlink = path("rootlink");
  if (!create_symlink(path("keep").c_str(), rootlink.c_str())) {
    GTEST_SKIP() << "create_symlink unavailable (needs privilege on this OS)";
  }
  ASSERT_TRUE(is_symlink(rootlink.c_str()));
  // (a) ROOT is a dir-link: remove the LINK itself, keep the target (audit #2).
  EXPECT_TRUE(remove_directory_recursive(rootlink.c_str()));
  EXPECT_FALSE(exists(rootlink.c_str()));
  EXPECT_TRUE(exists(path("keep/data.txt").c_str()));  // target survived

  // (b) CHILD dir-link inside a tree: recurse must not follow it (audit #1).
  ASSERT_TRUE(make_directory(path("tree").c_str()));
  ASSERT_TRUE(create_symlink(path("keep").c_str(), path("tree/link").c_str()));
  EXPECT_TRUE(remove_directory_recursive(path("tree").c_str()));
  EXPECT_FALSE(exists(path("tree").c_str()));
  EXPECT_TRUE(exists(path("keep/data.txt").c_str()));  // still there
}

TEST_F(FsTest, SpecialDirectoriesNonEmpty) {
  char buf[4096];
  EXPECT_GT(current_directory(buf, sizeof(buf)), 0u);
  EXPECT_GT(temp_directory(buf, sizeof(buf)), 0u);
  EXPECT_GT(home_directory(buf, sizeof(buf)), 0u);
}

TEST(FsEnv, HomeDirectorySupportsLongEnvironmentValue) {
#if defined(_WIN32)
  const char* name = "USERPROFILE";
#else
  const char* name = "HOME";
#endif
  char old[4096];
  const std::size_t old_len = get_env_var(name, old, sizeof(old));
  ASSERT_NE(old_len, kEnvNotSet);
  ASSERT_LT(old_len, sizeof(old));
  std::string value(700, 'h');
  ASSERT_TRUE(set_env_var(name, value.c_str()));
  char got[1024];
  EXPECT_EQ(home_directory(got, sizeof(got)), value.size());
  EXPECT_STREQ(got, value.c_str());
  EXPECT_TRUE(set_env_var(name, old));
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

// ---- v4: permissions --------------------------------------------------------

TEST_F(FsTest, PermissionsReadWriteRoundTrip) {
  ASSERT_TRUE(make_directory_recursive(root.c_str()));
  std::string f = root + "/perm.txt";
  ASSERT_TRUE(touch(f.c_str()));

  // A fresh file is writable.
  EXPECT_FALSE(is_readonly(f.c_str()));
  EXPECT_TRUE(set_readonly(f.c_str(), true));
  EXPECT_TRUE(is_readonly(f.c_str()));
  EXPECT_TRUE(set_readonly(f.c_str(), false));
  EXPECT_FALSE(is_readonly(f.c_str()));

  // get/set_permissions round-trip (POSIX exact; Windows synthesizes RO/RW).
  std::uint32_t p = get_permissions(f.c_str());
  EXPECT_NE(p, 0u);  // file exists
#if !defined(_WIN32)
  EXPECT_TRUE(set_permissions(f.c_str(), 0640));
  EXPECT_EQ(get_permissions(f.c_str()) & 0777u, 0640u);
  set_permissions(f.c_str(), 0644);  // restore so cleanup can delete
#endif
}

TEST_F(FsTest, PermissionsAbsentPathIsZero) {
  std::string f = root + "/nope.txt";
  EXPECT_EQ(get_permissions(f.c_str()), 0u);
  EXPECT_FALSE(is_readonly(f.c_str()));
}

// ---- v4: symbolic links -----------------------------------------------------

TEST_F(FsTest, SymlinkCreateReadAndDetect) {
  ASSERT_TRUE(make_directory_recursive(root.c_str()));
  std::string target = root + "/real.txt";
  std::string link = root + "/link.txt";
  ASSERT_TRUE(touch(target.c_str()));

  // create_symlink needs privilege on Windows (Developer Mode/admin); if it
  // fails there, skip the rest rather than fail the suite.
  if (!create_symlink(target.c_str(), link.c_str())) {
    GTEST_SKIP() << "create_symlink unavailable (needs privilege on this OS)";
  }
  EXPECT_TRUE(is_symlink(link.c_str()));
  EXPECT_FALSE(is_symlink(target.c_str()));
  EXPECT_TRUE(stat(link.c_str()).is_symlink);

  char buf[4096];
  std::size_t n = read_symlink(link.c_str(), buf, sizeof(buf));
  EXPECT_GT(n, 0u);
  EXPECT_LT(n, sizeof(buf));
  // The link resolves to (something ending in) the target name.
  EXPECT_NE(std::string(buf).find("real.txt"), std::string::npos);

  // buffer-fill truncation contract: a tiny buffer still returns the FULL length
  // (>= cap => truncated) and NUL-terminates what it did write.
  char tiny[4];
  std::size_t need = read_symlink(link.c_str(), tiny, sizeof(tiny));
  EXPECT_EQ(need, n);                     // full length reported regardless of cap
  EXPECT_EQ(std::strlen(tiny), 3u);       // 3 bytes + NUL
}

// ---- v4: temp file + FileHandle ----------------------------------------------

TEST_F(FsTest, CreateTempFileIsUniqueAndOpen) {
  ASSERT_TRUE(make_directory_recursive(root.c_str()));
  char p1[4096], p2[4096];
  auto ar = create_temp_file(root.c_str(), "hyle", p1, sizeof(p1));
  auto br = create_temp_file(root.c_str(), "hyle", p2, sizeof(p2));
  ASSERT_TRUE(ar.has_value()) << ar.error().message();
  ASSERT_TRUE(br.has_value()) << br.error().message();
  FileHandle a = std::move(*ar);
  FileHandle b = std::move(*br);
  ASSERT_TRUE(a.valid());
  ASSERT_TRUE(b.valid());
  EXPECT_STRNE(p1, p2);              // unique names
  EXPECT_TRUE(exists(p1));          // the file really exists on disk
  EXPECT_TRUE(exists(p2));
  EXPECT_NE(a.get(), FileHandle::kInvalid);
  // Prove the returned handle is genuinely OPEN FOR WRITE (not just a path):
  // write through it, then read the bytes back from the path.
  const char msg[] = "hyle-temp";
  ASSERT_TRUE(write_all_fd(a.get(), msg, sizeof(msg) - 1));
  EXPECT_TRUE(a.close().ok());
  std::FILE* f = std::fopen(p1, "rb");
  ASSERT_NE(f, nullptr);
  char buf[16] = {0};
  std::size_t got = std::fread(buf, 1, sizeof(buf) - 1, f);
  std::fclose(f);
  EXPECT_EQ(got, sizeof(msg) - 1);
  EXPECT_STREQ(buf, "hyle-temp");
}

TEST_F(FsTest, HardLinkSharesData) {
  ASSERT_TRUE(make_directory_recursive(root.c_str()));
  std::string target = root + "/orig.txt";
  std::string link = root + "/hardlink.txt";
  {
    std::FILE* f = std::fopen(target.c_str(), "wb");
    ASSERT_NE(f, nullptr);
    std::fputs("shared", f);
    std::fclose(f);
  }
  ASSERT_TRUE(create_hard_link(target.c_str(), link.c_str()));
  // The hard link is a second name for the SAME data: not a symlink, same size,
  // and it survives deletion of the original.
  EXPECT_TRUE(exists(link.c_str()));
  EXPECT_FALSE(is_symlink(link.c_str()));
  EXPECT_EQ(file_size(link.c_str()), 6u);
  ASSERT_TRUE(remove_file(target.c_str()));
  EXPECT_TRUE(exists(link.c_str()));            // data persists via the other name
  EXPECT_EQ(file_size(link.c_str()), 6u);
}

TEST(FsEnv, SetEmptyValueSemantics) {
  const char* kName = "HYLE_FS_EMPTY_VAR";
  ASSERT_TRUE(set_env_var(kName, ""));
  char buf[8];
  std::size_t n = get_env_var(kName, buf, sizeof(buf));
#if defined(_WIN32)
  // PLATFORM DIFFERENCE: Windows _putenv_s(name,"") DELETES the variable (there
  // is no "set to empty" -- an empty value IS removal), so a get reports unset.
  EXPECT_EQ(n, kEnvNotSet);
#else
  // POSIX setenv(name,"",1) keeps the variable present with a zero-length value,
  // distinct from unset. This is the "set-but-empty vs unset" distinction.
  EXPECT_NE(n, kEnvNotSet);
  EXPECT_EQ(n, 0u);
#endif
  EXPECT_TRUE(unset_env_var(kName));
  EXPECT_EQ(get_env_var(kName, buf, sizeof(buf)), kEnvNotSet);  // truly unset
}

TEST(FileHandle, MoveAndRelease) {
  char path[4096];
  auto created = create_temp_file(nullptr, "hyle_file", path, sizeof(path));
  ASSERT_TRUE(created.has_value()) << created.error().message();
  FileHandle a = std::move(*created);
  ASSERT_TRUE(a.valid());
  std::intptr_t raw = a.get();

  FileHandle b = std::move(a);       // move transfers ownership
  EXPECT_FALSE(a.valid());
  EXPECT_TRUE(b.valid());
  EXPECT_EQ(b.get(), raw);

  std::intptr_t released = b.release();  // give up ownership without closing
  EXPECT_FALSE(b.valid());
  EXPECT_EQ(released, raw);
  // We now own `released`; hand it to a fresh handle so it is closed.
  FileHandle c{released};
  EXPECT_TRUE(c.valid());
  remove_file(path);
}

TEST(FileHandle, OpenFileWriteThenReadRoundTrip) {
  char dir[4096];
  ASSERT_NE(temp_directory(dir, sizeof(dir)), 0u);
  std::string p = std::string(dir) + "/hyle_openfile_test.txt";
  const char msg[] = "hello platform io";
  const std::size_t len = sizeof(msg) - 1;  // exclude NUL
  {
    auto opened = open_file(
        p.c_str(), kOpenWrite | kOpenCreate | kOpenTruncate);
    ASSERT_TRUE(opened.has_value()) << opened.error().message();
    auto wrote = opened->write(msg, len);
    ASSERT_TRUE(wrote.has_value()) << wrote.error().message();
    EXPECT_EQ(*wrote, len);
    EXPECT_TRUE(opened->close().ok());
  }
  {
    auto opened = open_file(p.c_str(), kOpenRead);
    ASSERT_TRUE(opened.has_value()) << opened.error().message();
    char buf[64] = {0};
    auto got = opened->read(buf, sizeof(buf));
    ASSERT_TRUE(got.has_value()) << got.error().message();
    EXPECT_EQ(*got, len);
    EXPECT_EQ(std::string(buf, *got), "hello platform io");
  }
  // Missing file and operations on a closed handle preserve structured errors.
  auto missing =
      open_file((std::string(dir) + "/hyle_no_such.txt").c_str(), kOpenRead);
  EXPECT_FALSE(missing.has_value());
  EXPECT_TRUE(missing.error().IsNotFound());
  FileHandle empty;
  char b[4];
  auto closed_read = empty.read(b, sizeof(b));
  auto closed_write = empty.write(b, sizeof(b));
  EXPECT_TRUE(closed_read.error().IsFailedPrecondition());
  EXPECT_TRUE(closed_write.error().IsFailedPrecondition());
  EXPECT_TRUE(empty.read(nullptr, 1).error().IsInvalidArgument());
  EXPECT_TRUE(empty.write(nullptr, 1).error().IsInvalidArgument());
  remove_file(p.c_str());
}

// ---- v4: environment write --------------------------------------------------

TEST(FsEnv, SetGetUnsetRoundTrip) {
  const char* kName = "HYLE_FS_TEST_VAR";
  EXPECT_TRUE(set_env_var(kName, "hello-42"));
  char buf[64];
  std::size_t n = get_env_var(kName, buf, sizeof(buf));
  ASSERT_NE(n, kEnvNotSet);
  EXPECT_STREQ(buf, "hello-42");
  EXPECT_TRUE(unset_env_var(kName));
  EXPECT_EQ(get_env_var(kName, buf, sizeof(buf)), kEnvNotSet);
}

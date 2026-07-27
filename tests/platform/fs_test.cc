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
#  include <fcntl.h>
#  include <sys/stat.h>
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

bool set_one_tick_before_unix_epoch(const char* path) {
#if defined(_WIN32)
  HANDLE file = ::CreateFileA(path, FILE_WRITE_ATTRIBUTES,
                              FILE_SHARE_READ | FILE_SHARE_WRITE |
                                  FILE_SHARE_DELETE,
                              nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE) return false;
  constexpr std::uint64_t kUnixEpochTicks =
      116'444'736'000'000'000ull;
  const std::uint64_t raw = kUnixEpochTicks - 1;
  FILETIME value{static_cast<DWORD>(raw),
                 static_cast<DWORD>(raw >> 32)};
  const bool ok = ::SetFileTime(file, nullptr, nullptr, &value) != 0;
  ::CloseHandle(file);
  return ok;
#else
  ::timespec values[2] = {{-1, 999'999'999}, {-1, 999'999'999}};
  return ::utimensat(AT_FDCWD, path, values, 0) == 0;
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
    auto temp = temp_directory(tmp, sizeof(tmp));
    ASSERT_TRUE(temp.has_value()) << temp.error().message();
    ASSERT_GT(*temp, 0u);
    ASSERT_LT(*temp, sizeof(tmp));
    // std::string concat (not snprintf into a fixed buffer, which GCC flags as
    // possibly-truncated since tmp can be up to 4096 chars).
    root = std::string(tmp, *temp) + "/hyle_fs_test_" +
           std::to_string(reinterpret_cast<std::uintptr_t>(this));
    (void)remove_directory_recursive(root.c_str());  // clear any stale leftover
    ASSERT_TRUE(make_directory(root.c_str()).ok());
  }
  void TearDown() override { (void)remove_directory_recursive(root.c_str()); }

  std::string path(const char* rel) const { return root + "/" + rel; }
};

}  // namespace

TEST(FsBoundary, NullInputsAndInvalidOpenFlagsFailSafely) {
  EXPECT_TRUE(make_directory_recursive(nullptr).IsInvalidArgument());
  EXPECT_TRUE(remove_directory_recursive(nullptr).IsInvalidArgument());
  EXPECT_TRUE(touch(nullptr).IsInvalidArgument());
  EXPECT_TRUE(stat(nullptr).error().IsInvalidArgument());
  EXPECT_TRUE(exists(nullptr).error().IsInvalidArgument());
  EXPECT_TRUE(file_size(nullptr).error().IsInvalidArgument());
  EXPECT_TRUE(get_permissions(nullptr).error().IsInvalidArgument());
  EXPECT_TRUE(read_symlink(nullptr, nullptr, 0).error().IsInvalidArgument());
  EXPECT_TRUE(get_env_var(nullptr, nullptr, 0).error().IsInvalidArgument());
  EXPECT_TRUE(get_env_var("", nullptr, 0).error().IsInvalidArgument());
  EXPECT_TRUE(get_env_var("A=B", nullptr, 0).error().IsInvalidArgument());
  EXPECT_TRUE(set_env_var(nullptr, "value").IsInvalidArgument());
  EXPECT_TRUE(set_env_var("NAME", nullptr).IsInvalidArgument());
  EXPECT_TRUE(set_env_var("A=B", "value").IsInvalidArgument());
  EXPECT_TRUE(unset_env_var(nullptr).IsInvalidArgument());
  EXPECT_TRUE(change_directory(nullptr).IsInvalidArgument());
  EXPECT_FALSE(open_file(nullptr, kOpenRead).has_value());
  EXPECT_FALSE(open_file("unused", 0).has_value());
  EXPECT_FALSE(
      open_file("unused", kOpenCreate | kOpenTruncate).has_value());

  char tmp[4] = {'x', 'x', 'x', '\0'};
  auto need = temp_directory(nullptr, sizeof(tmp));
  ASSERT_TRUE(need.has_value());
  EXPECT_GT(*need, 0u);
}

TEST_F(FsTest, StatExistsDirectoryFile) {
  EXPECT_TRUE(exists(root.c_str()).value_or(false));
  EXPECT_TRUE(is_directory(root.c_str()).value_or(false));

  std::string f = path("a.txt");
  EXPECT_FALSE(exists(f.c_str()).value_or(true));
  ASSERT_TRUE(touch(f.c_str()).ok());
  EXPECT_TRUE(exists(f.c_str()).value_or(false));
  EXPECT_FALSE(is_directory(f.c_str()).value_or(true));

  FileInfo s = stat(f.c_str()).value();
  EXPECT_EQ(s.type, FileType::kRegular);
  EXPECT_FALSE(s.is_directory());
  EXPECT_EQ(s.size, 0u);

  // A missing path is an explicit NotFound, not an all-zero pseudo-stat.
  EXPECT_TRUE(stat(path("nope").c_str()).error().IsNotFound());
}

TEST_F(FsTest, StatPreservesPreEpochTimeAndSubsecondPrecision) {
  const std::string file = path("before-epoch.txt");
  ASSERT_TRUE(touch(file.c_str()).ok());
  ASSERT_TRUE(set_one_tick_before_unix_epoch(file.c_str()));

  const FileInfo info = stat(file.c_str()).value();
  EXPECT_TRUE(hyle::time::unix_timestamp_valid(info.modified_time));
  EXPECT_EQ(info.modified_time.seconds, -1);
#if defined(_WIN32)
  EXPECT_EQ(info.modified_time.nanoseconds, 999'999'900u);
#else
  EXPECT_EQ(info.modified_time.nanoseconds, 999'999'999u);
#endif
}

TEST_F(FsTest, MakeDirectoryIdempotent) {
  std::string d = path("sub");
  EXPECT_TRUE(make_directory(d.c_str()).ok());
  EXPECT_TRUE(make_directory(d.c_str()).ok());  // already exists -> still OK
  EXPECT_TRUE(is_directory(d.c_str()).value_or(false));
}

TEST_F(FsTest, MakeDirectoryRecursive) {
  std::string deep = path("x/y/z");
  EXPECT_TRUE(make_directory_recursive(deep.c_str()).ok());
  EXPECT_TRUE(is_directory(deep.c_str()).value_or(false));
  EXPECT_TRUE(is_directory(path("x/y").c_str()).value_or(false));
  EXPECT_TRUE(is_directory(path("x").c_str()).value_or(false));
}

TEST_F(FsTest, FileSizeReflectsWrite) {
  std::string f = path("data.bin");
  std::FILE* fp = std::fopen(f.c_str(), "wb");
  ASSERT_NE(fp, nullptr);
  const char payload[] = "0123456789";
  std::fwrite(payload, 1, 10, fp);
  std::fclose(fp);
  EXPECT_EQ(file_size(f.c_str()).value(), 10u);
}

TEST_F(FsTest, FileSizeRejectsDirectory) {
  auto size = file_size(root.c_str());
  ASSERT_FALSE(size.has_value());
  EXPECT_TRUE(size.error().IsFailedPrecondition());
}

TEST_F(FsTest, RenameAndRemoveFile) {
  std::string a = path("a.txt"), b = path("b.txt");
  ASSERT_TRUE(touch(a.c_str()).ok());
  EXPECT_TRUE(rename_path(a.c_str(), b.c_str()).ok());
  EXPECT_FALSE(exists(a.c_str()).value_or(true));
  EXPECT_TRUE(exists(b.c_str()).value_or(false));
  EXPECT_TRUE(remove_file(b.c_str()).ok());
  EXPECT_FALSE(exists(b.c_str()).value_or(true));
  EXPECT_TRUE(remove_file(b.c_str()).IsNotFound());
}

TEST_F(FsTest, CopyFile) {
  std::string a = path("src.txt"), b = path("dst.txt");
  std::FILE* fp = std::fopen(a.c_str(), "wb");
  ASSERT_NE(fp, nullptr);
  std::fwrite("hello", 1, 5, fp);
  std::fclose(fp);
  EXPECT_TRUE(copy_file(a.c_str(), b.c_str()).ok());
  EXPECT_EQ(file_size(b.c_str()).value(), 5u);
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

  ASSERT_TRUE(copy_file(a.c_str(), b.c_str()).ok());
  EXPECT_EQ(file_size(a.c_str()).value(), 7u);
  EXPECT_EQ(file_size(b.c_str()).value(), 7u);
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

  ASSERT_TRUE(copy_file(a.c_str(), b.c_str()).ok());
  EXPECT_EQ(file_size(b.c_str()).value(), kN);
  std::vector<unsigned char> back(kN, 0);
  fp = std::fopen(b.c_str(), "rb");
  ASSERT_NE(fp, nullptr);
  EXPECT_EQ(std::fread(back.data(), 1, kN, fp), kN);
  std::fclose(fp);
  EXPECT_EQ(back, data);
}

TEST_F(FsTest, DirectoryIterateVisitsEntriesNotDotDot) {
  ASSERT_TRUE(touch(path("f1").c_str()).ok());
  ASSERT_TRUE(touch(path("f2").c_str()).ok());
  ASSERT_TRUE(make_directory(path("d1").c_str()).ok());

  std::vector<std::string> files, dirs;
  auto completed = for_each_entry(root.c_str(), [&](const char* name, FileType type) {
    if (type == FileType::kDirectory)
      dirs.emplace_back(name);
    else
      files.emplace_back(name);
  });
  ASSERT_TRUE(completed.has_value());
  EXPECT_TRUE(*completed);
  EXPECT_EQ(files.size(), 2u);
  EXPECT_EQ(dirs.size(), 1u);
  // "." / ".." must be excluded.
  for (const auto& n : files) EXPECT_NE(n, ".");
  for (const auto& n : dirs) EXPECT_EQ(n, "d1");
}

TEST_F(FsTest, ForEachEntryEarlyStop) {
  for (int i = 0; i < 5; ++i)
    ASSERT_TRUE(touch(path(("e" + std::to_string(i)).c_str()).c_str()).ok());
  int seen = 0;
  auto completed = for_each_entry(root.c_str(), [&](const char*, FileType) -> bool {
    ++seen;
    return seen < 2;  // stop after the second entry
  });
  ASSERT_TRUE(completed.has_value());
  EXPECT_FALSE(*completed);
  EXPECT_EQ(seen, 2);
}

TEST_F(FsTest, IterateMissingDirReturnsNotFound) {
  auto result = for_each_entry(path("ghost").c_str(),
                               [](const char*, FileType) { /* never called */ });
  ASSERT_FALSE(result.has_value());
  EXPECT_TRUE(result.error().IsNotFound());
}

TEST_F(FsTest, RemoveDirectoryRecursive) {
  ASSERT_TRUE(make_directory_recursive(path("tree/inner").c_str()).ok());
  ASSERT_TRUE(touch(path("tree/top.txt").c_str()).ok());
  ASSERT_TRUE(touch(path("tree/inner/leaf.txt").c_str()).ok());
  // A non-empty dir cannot be removed by the plain (empty-only) call.
  EXPECT_FALSE(remove_directory(path("tree").c_str()).ok());
  EXPECT_TRUE(remove_directory_recursive(path("tree").c_str()).ok());
  EXPECT_FALSE(exists(path("tree").c_str()).value_or(true));
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
  ASSERT_TRUE(make_directory_recursive(deep.c_str()).ok());
  EXPECT_TRUE(is_directory(deep.c_str()).value_or(false));
  // A file at the deep leaf round-trips too (touch -> exists).
  std::string leaf = deep + "/leaf.txt";
  ASSERT_TRUE(touch(leaf.c_str()).ok());
  EXPECT_TRUE(exists(leaf.c_str()).value_or(false));
  // Recursive remove must also handle the long path (iterate + rebuild children).
  EXPECT_TRUE(remove_directory_recursive(path("L").c_str()).ok());
  EXPECT_FALSE(exists(path("L").c_str()).value_or(true));
}

#if !defined(_WIN32)
// Recursive delete must NOT follow a symlink-to-directory and delete the target
// (blast-radius safety). POSIX-only: symlink creation on Windows needs a
// privilege; this guards the DT_LNK / lstat-on-DT_UNKNOWN path in fs.cc.
TEST_F(FsTest, RemoveDirectoryRecursiveDoesNotFollowSymlink) {
  // outside/keep.txt is the target that must SURVIVE.
  ASSERT_TRUE(make_directory(path("outside").c_str()).ok());
  ASSERT_TRUE(touch(path("outside/keep.txt").c_str()).ok());
  // victim/ contains a symlink "link" -> ../outside.
  ASSERT_TRUE(make_directory(path("victim").c_str()).ok());
  std::string link = path("victim/link");
  std::string target = path("outside");
  ASSERT_EQ(::symlink(target.c_str(), link.c_str()), 0);

  EXPECT_TRUE(remove_directory_recursive(path("victim").c_str()).ok());
  EXPECT_FALSE(exists(path("victim").c_str()).value_or(true));
  // The link's target and its contents must be untouched.
  EXPECT_TRUE(exists(path("outside").c_str()).value_or(false));
  EXPECT_TRUE(exists(path("outside/keep.txt").c_str()).value_or(false));
}
#endif

// Recursive delete must not follow a symlink/junction-to-directory -- at the ROOT
// or as a child -- and wipe its target (data-loss guard; platform-audit #1: the
// Windows junction path was untested since the case above is POSIX-only; #2: root
// symlink). Cross-platform via create_symlink, which needs privilege on Windows
// (Developer Mode/admin) -> SKIP if unavailable.
TEST_F(FsTest, RemoveRecursiveDoesNotFollowDirLink) {
  ASSERT_TRUE(make_directory(path("keep").c_str()).ok());
  ASSERT_TRUE(touch(path("keep/data.txt").c_str()).ok());

  std::string rootlink = path("rootlink");
  if (!create_symlink(path("keep").c_str(), rootlink.c_str()).ok()) {
    GTEST_SKIP() << "create_symlink unavailable (needs privilege on this OS)";
  }
  ASSERT_TRUE(is_symlink(rootlink.c_str()).value_or(false));
  // (a) ROOT is a dir-link: remove the LINK itself, keep the target (audit #2).
  EXPECT_TRUE(remove_directory_recursive(rootlink.c_str()).ok());
  EXPECT_FALSE(exists(rootlink.c_str()).value_or(true));
  EXPECT_TRUE(exists(path("keep/data.txt").c_str()).value_or(false));

  // (b) CHILD dir-link inside a tree: recurse must not follow it (audit #1).
  ASSERT_TRUE(make_directory(path("tree").c_str()).ok());
  ASSERT_TRUE(create_symlink(path("keep").c_str(), path("tree/link").c_str()).ok());
  EXPECT_TRUE(remove_directory_recursive(path("tree").c_str()).ok());
  EXPECT_FALSE(exists(path("tree").c_str()).value_or(true));
  EXPECT_TRUE(exists(path("keep/data.txt").c_str()).value_or(false));
}

TEST_F(FsTest, SpecialDirectoriesNonEmpty) {
  char buf[4096];
  EXPECT_GT(current_directory(buf, sizeof(buf)).value(), 0u);
  EXPECT_GT(temp_directory(buf, sizeof(buf)).value(), 0u);
  EXPECT_GT(home_directory(buf, sizeof(buf)).value(), 0u);
}

TEST(FsEnv, HomeDirectorySupportsLongEnvironmentValue) {
#if defined(_WIN32)
  const char* name = "USERPROFILE";
#else
  const char* name = "HOME";
#endif
  char old[4096];
  auto old_len = get_env_var(name, old, sizeof(old));
  ASSERT_TRUE(old_len.has_value()) << old_len.error().message();
  ASSERT_LT(*old_len, sizeof(old));
  std::string value(700, 'h');
  ASSERT_TRUE(set_env_var(name, value.c_str()).ok());
  char got[1024];
  EXPECT_EQ(home_directory(got, sizeof(got)).value(), value.size());
  EXPECT_STREQ(got, value.c_str());
  EXPECT_TRUE(set_env_var(name, old).ok());
}

// Non-ASCII UTF-8 path round-trips through every fs op. POSIX is byte-
// transparent; Windows must convert UTF-8 <-> UTF-16 (else the ...A APIs would
// mis-decode under the active code page). The directory name mixes CJK + emoji.
TEST_F(FsTest, Utf8NonAsciiPath) {
  std::string dir = path("\xE6\xB5\x8B\xE8\xAF\x95_\xF0\x9F\x9A\x80");  // 测试_🚀
  ASSERT_TRUE(make_directory(dir.c_str()).ok());
  EXPECT_TRUE(is_directory(dir.c_str()).value_or(false));

  std::string file = dir + "/\xE6\x96\x87\xE4\xBB\xB6.txt";  // 文件.txt
  ASSERT_TRUE(touch(file.c_str()).ok());
  EXPECT_TRUE(exists(file.c_str()).value_or(false));

  // The non-ASCII entry name comes back intact through the visitor (UTF-8).
  bool found = false;
  auto completed = for_each_entry(dir.c_str(), [&](const char* name, FileType type) {
    if (type == FileType::kRegular &&
        std::strcmp(name, "\xE6\x96\x87\xE4\xBB\xB6.txt") == 0)
      found = true;
  });
  ASSERT_TRUE(completed.has_value());
  EXPECT_TRUE(found);
  EXPECT_TRUE(remove_directory_recursive(dir.c_str()).ok());
}

TEST_F(FsTest, ChangeDirectoryRoundTrip) {
  char before[4096];
  ASSERT_GT(current_directory(before, sizeof(before)).value(), 0u);
  ASSERT_TRUE(change_directory(root.c_str()).ok());
  char now[4096];
  ASSERT_GT(current_directory(now, sizeof(now)).value(), 0u);
  // cwd now ends with our scratch dir's base name.
  EXPECT_NE(std::strstr(now, "hyle_fs_test_"), nullptr);
  EXPECT_TRUE(change_directory(before).ok());  // restore for other tests
}

TEST_F(FsTest, ChangeDirectoryMissingIsNotFound) {
  EXPECT_TRUE(change_directory(path("missing-cwd").c_str()).IsNotFound());
}

TEST(FsEnv, GetEnvVarSetAndUnset) {
  char buf[4096];
  // PATH is set on every platform CI runs on.
  auto n = get_env_var("PATH", buf, sizeof(buf));
  ASSERT_TRUE(n.has_value()) << n.error().message();
  EXPECT_GT(*n, 0u);

  // A name that cannot exist -> NotFound (distinct from set-but-empty).
  EXPECT_TRUE(get_env_var("HYLE_DEFINITELY_UNSET_XYZZY_123", buf, sizeof(buf))
                  .error()
                  .IsNotFound());
}

TEST(FsEnv, BufferFillTruncates) {
  // current_directory into a tiny buffer: NUL-terminated, returns full length.
  char tiny[4];
  std::size_t full = current_directory(tiny, sizeof(tiny)).value();
  EXPECT_EQ(std::strlen(tiny), 3u);  // 3 chars + NUL
  EXPECT_GE(full, std::strlen(tiny));
}

TEST_F(FsTest, MissingPathDoesNotExist) {
  std::string missing = root + "/no_such_file_xyz";
  EXPECT_FALSE(exists(missing.c_str()).value_or(true));
  EXPECT_TRUE(stat(missing.c_str()).error().IsNotFound());
}

TEST_F(FsTest, TouchCreatesEmptyFile) {
  std::string f = root + "/touched.txt";
  ASSERT_TRUE(touch(f.c_str()).ok());
  EXPECT_TRUE(exists(f.c_str()).value_or(false));
  EXPECT_EQ(file_size(f.c_str()).value(), 0u);
  // second touch is idempotent (updates mtime or no-op success)
  EXPECT_TRUE(touch(f.c_str()).ok());
}

TEST_F(FsTest, RemoveNonEmptyDirectoryFails) {
  // Non-recursive remove_directory requires empty dir.
  std::string sub = root + "/nonempty";
  ASSERT_TRUE(make_directory(sub.c_str()).ok());
  ASSERT_TRUE(touch((sub + "/x").c_str()).ok());
  EXPECT_FALSE(remove_directory(sub.c_str()).ok());
  EXPECT_TRUE(exists(sub.c_str()).value_or(false));
  EXPECT_TRUE(remove_directory_recursive(sub.c_str()).ok());
}

TEST_F(FsTest, IsDirectoryDistinguishesFile) {
  std::string f = root + "/plain.bin";
  ASSERT_TRUE(touch(f.c_str()).ok());
  EXPECT_TRUE(exists(f.c_str()).value_or(false));
  EXPECT_FALSE(is_directory(f.c_str()).value_or(true));
  EXPECT_TRUE(is_directory(root.c_str()).value_or(false));
}

TEST(FsEnv, EnvVarTinyBufferTruncates) {
  char tiny[4];
  auto n = get_env_var("PATH", tiny, sizeof(tiny));
  if (!n && n.error().IsNotFound()) {
    GTEST_SKIP() << "PATH unset in this environment";
  }
  ASSERT_TRUE(n.has_value()) << n.error().message();
  EXPECT_EQ(std::strlen(tiny), 3u);
  EXPECT_GE(*n, 3u);
}

// ---- v4: permissions --------------------------------------------------------

TEST_F(FsTest, PermissionsReadWriteRoundTrip) {
  ASSERT_TRUE(make_directory_recursive(root.c_str()).ok());
  std::string f = root + "/perm.txt";
  ASSERT_TRUE(touch(f.c_str()).ok());

  // A fresh file is writable.
  EXPECT_FALSE(is_readonly(f.c_str()).value_or(true));
  EXPECT_TRUE(set_readonly(f.c_str(), true).ok());
  EXPECT_TRUE(is_readonly(f.c_str()).value_or(false));
  EXPECT_TRUE(set_readonly(f.c_str(), false).ok());
  EXPECT_FALSE(is_readonly(f.c_str()).value_or(true));

  // get/set_permissions round-trip (POSIX exact; Windows synthesizes RO/RW).
  std::uint32_t p = get_permissions(f.c_str()).value();
  EXPECT_NE(p, 0u);  // file exists
#if !defined(_WIN32)
  EXPECT_TRUE(set_permissions(f.c_str(), 0640).ok());
  EXPECT_EQ(get_permissions(f.c_str()).value() & 0777u, 0640u);
  (void)set_permissions(f.c_str(), 0644);  // restore so cleanup can delete
#endif
}

TEST_F(FsTest, PermissionsAbsentPathIsNotFound) {
  std::string f = root + "/nope.txt";
  EXPECT_TRUE(get_permissions(f.c_str()).error().IsNotFound());
  EXPECT_TRUE(is_readonly(f.c_str()).error().IsNotFound());
}

// ---- v4: symbolic links -----------------------------------------------------

TEST_F(FsTest, SymlinkCreateReadAndDetect) {
  ASSERT_TRUE(make_directory_recursive(root.c_str()).ok());
  std::string target = root + "/real.txt";
  std::string link = root + "/link.txt";
  ASSERT_TRUE(touch(target.c_str()).ok());

  // create_symlink needs privilege on Windows (Developer Mode/admin); if it
  // fails there, skip the rest rather than fail the suite.
  if (!create_symlink(target.c_str(), link.c_str()).ok()) {
    GTEST_SKIP() << "create_symlink unavailable (needs privilege on this OS)";
  }
  EXPECT_TRUE(is_symlink(link.c_str()).value_or(false));
  EXPECT_FALSE(is_symlink(target.c_str()).value_or(true));
  FileInfo link_info = stat(link.c_str()).value();
  EXPECT_TRUE(link_info.is_symlink);
  EXPECT_EQ(link_info.type, FileType::kRegular);  // stat follows the link

  bool saw_link = false;
  auto iterated = for_each_entry(
      root.c_str(), [&](const char* name, FileType type) {
        if (std::strcmp(name, "link.txt") == 0) {
          saw_link = true;
          EXPECT_EQ(type, FileType::kSymlink);  // iteration does not follow it
        }
      });
  ASSERT_TRUE(iterated.has_value());
  EXPECT_TRUE(saw_link);

  char buf[4096];
  std::size_t n = read_symlink(link.c_str(), buf, sizeof(buf)).value();
  EXPECT_GT(n, 0u);
  EXPECT_LT(n, sizeof(buf));
  // The link resolves to (something ending in) the target name.
  EXPECT_NE(std::string(buf).find("real.txt"), std::string::npos);

  // buffer-fill truncation contract: a tiny buffer still returns the FULL length
  // (>= cap => truncated) and NUL-terminates what it did write.
  char tiny[4];
  std::size_t need = read_symlink(link.c_str(), tiny, sizeof(tiny)).value();
  EXPECT_EQ(need, n);                     // full length reported regardless of cap
  EXPECT_EQ(std::strlen(tiny), 3u);       // 3 bytes + NUL
}

TEST_F(FsTest, ExplicitSymlinkKindSupportsDanglingDirectoryTarget) {
  const std::string target = path("directory-created-later");
  const std::string link = path("dangling-directory-link");
  auto created = create_symlink(target.c_str(), link.c_str(),
                                SymlinkKind::kDirectory);
  if (!created.ok()) {
    GTEST_SKIP() << "create_symlink unavailable (needs privilege on this OS)";
  }
  EXPECT_TRUE(is_symlink(link.c_str()).value_or(false));
  EXPECT_TRUE(stat(link.c_str()).error().IsNotFound());
  EXPECT_TRUE(remove_directory_recursive(link.c_str()).ok());
  EXPECT_FALSE(is_symlink(link.c_str()).value_or(true));
}

TEST_F(FsTest, CreateSymlinkRejectsUnknownKind) {
  EXPECT_TRUE(create_symlink("target", "link", static_cast<SymlinkKind>(99))
                  .IsInvalidArgument());
}

// ---- v4: temp file + FileHandle ----------------------------------------------

TEST_F(FsTest, CreateTempFileIsUniqueAndOpen) {
  ASSERT_TRUE(make_directory_recursive(root.c_str()).ok());
  char p1[4096], p2[4096];
  auto ar = create_temp_file(root.c_str(), "hyle", p1, sizeof(p1));
  auto br = create_temp_file(root.c_str(), "hyle", p2, sizeof(p2));
  ASSERT_TRUE(ar.has_value()) << ar.error().message();
  ASSERT_TRUE(br.has_value()) << br.error().message();
  EXPECT_EQ(ar->path_size, std::strlen(p1));
  EXPECT_EQ(br->path_size, std::strlen(p2));
  FileHandle a = std::move(ar->file);
  FileHandle b = std::move(br->file);
  ASSERT_TRUE(a.valid());
  ASSERT_TRUE(b.valid());
  EXPECT_STRNE(p1, p2);              // unique names
  EXPECT_TRUE(exists(p1).value_or(false));  // the file really exists on disk
  EXPECT_TRUE(exists(p2).value_or(false));
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
  ASSERT_TRUE(make_directory_recursive(root.c_str()).ok());
  std::string target = root + "/orig.txt";
  std::string link = root + "/hardlink.txt";
  {
    std::FILE* f = std::fopen(target.c_str(), "wb");
    ASSERT_NE(f, nullptr);
    std::fputs("shared", f);
    std::fclose(f);
  }
  ASSERT_TRUE(create_hard_link(target.c_str(), link.c_str()).ok());
  // The hard link is a second name for the SAME data: not a symlink, same size,
  // and it survives deletion of the original.
  EXPECT_TRUE(exists(link.c_str()).value_or(false));
  EXPECT_FALSE(is_symlink(link.c_str()).value_or(true));
  EXPECT_EQ(file_size(link.c_str()).value(), 6u);
  ASSERT_TRUE(remove_file(target.c_str()).ok());
  EXPECT_TRUE(exists(link.c_str()).value_or(false));
  EXPECT_EQ(file_size(link.c_str()).value(), 6u);
}

TEST(FsEnv, SetEmptyValueSemantics) {
  const char* kName = "HYLE_FS_EMPTY_VAR";
  ASSERT_TRUE(set_env_var(kName, "").ok());
  char buf[8];
  auto n = get_env_var(kName, buf, sizeof(buf));
#if defined(_WIN32)
  // PLATFORM DIFFERENCE: Windows _wputenv_s(name,"") DELETES the variable (there
  // is no "set to empty" -- an empty value IS removal), so a get reports unset.
  ASSERT_FALSE(n.has_value());
  EXPECT_TRUE(n.error().IsNotFound());
#else
  // POSIX setenv(name,"",1) keeps the variable present with a zero-length value,
  // distinct from unset. This is the "set-but-empty vs unset" distinction.
  ASSERT_TRUE(n.has_value());
  EXPECT_EQ(*n, 0u);
#endif
  EXPECT_TRUE(unset_env_var(kName).ok());
  EXPECT_TRUE(get_env_var(kName, buf, sizeof(buf)).error().IsNotFound());
}

TEST(FileHandle, MoveAndRelease) {
  char path[4096];
  auto created = create_temp_file(nullptr, "hyle_file", path, sizeof(path));
  ASSERT_TRUE(created.has_value()) << created.error().message();
  EXPECT_EQ(created->path_size, std::strlen(path));
  FileHandle a = std::move(created->file);
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
  (void)remove_file(path);
}

TEST(FileHandle, OpenFileWriteThenReadRoundTrip) {
  char dir[4096];
  auto temp = temp_directory(dir, sizeof(dir));
  ASSERT_TRUE(temp.has_value()) << temp.error().message();
  ASSERT_NE(*temp, 0u);
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
  (void)remove_file(p.c_str());
}

TEST(FileHandle, OpenHandleDoesNotBlockAtomicRenameOrDelete) {
  char dir[4096];
  auto temp = temp_directory(dir, sizeof(dir));
  ASSERT_TRUE(temp.has_value()) << temp.error().message();
  std::string original = std::string(dir) + "/hyle_share_delete.txt";
  std::string renamed = original + ".renamed";
  ASSERT_TRUE(touch(original.c_str()).ok());

  auto opened = open_file(original.c_str(), kOpenRead);
  ASSERT_TRUE(opened.has_value()) << opened.error().message();
  EXPECT_TRUE(rename_path(original.c_str(), renamed.c_str()).ok());
  EXPECT_TRUE(remove_file(renamed.c_str()).ok());
  EXPECT_TRUE(opened->valid());
  EXPECT_TRUE(opened->close().ok());
}

// ---- v4: environment write --------------------------------------------------

TEST(FsEnv, SetGetUnsetRoundTrip) {
  const char* kName = "HYLE_FS_TEST_VAR";
  EXPECT_TRUE(set_env_var(kName, "hello-42").ok());
  char buf[64];
  auto n = get_env_var(kName, buf, sizeof(buf));
  ASSERT_TRUE(n.has_value()) << n.error().message();
  EXPECT_STREQ(buf, "hello-42");
  EXPECT_TRUE(unset_env_var(kName).ok());
  EXPECT_TRUE(get_env_var(kName, buf, sizeof(buf)).error().IsNotFound());
}

TEST(FsEnv, Utf8NameAndValueRoundTrip) {
  const char* kName = "HYLE_\xE7\x8E\xAF\xE5\xA2\x83";  // HYLE_环境
  const char* kValue = "\xE5\x80\xBC_\xF0\x9F\x9A\x80";  // 值_🚀
  ASSERT_TRUE(set_env_var(kName, kValue).ok());
  char buf[64];
  auto size = get_env_var(kName, buf, sizeof(buf));
  ASSERT_TRUE(size.has_value()) << size.error().message();
  EXPECT_EQ(*size, std::strlen(kValue));
  EXPECT_STREQ(buf, kValue);
  EXPECT_TRUE(unset_env_var(kName).ok());
}

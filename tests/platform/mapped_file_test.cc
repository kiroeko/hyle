// Real-filesystem tests for platform::MappedFile (RO / RW / COW mapping).
#if defined(_WIN32)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

#include "hyle/platform/mapped_file.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>  // std::abort in the death test
#include <cstring>
#include <string>

#include "hyle/platform/fs.h"
#include "hyle/platform/virtual_memory.h"  // page_size()

using hyle::platform::MappedFile;
using hyle::platform::page_size;
namespace fs = hyle::platform;

namespace {

// A scratch file seeded with `bytes` of content, removed on teardown.
class MapTest : public ::testing::Test {
 protected:
  std::string path;

  void SetUp() override {
    char tmp[4096];
    std::size_t n = fs::temp_directory(tmp, sizeof(tmp));
    ASSERT_GT(n, 0u);
    path = std::string(tmp, n) + "/hyle_map_test_" +
           std::to_string(reinterpret_cast<std::uintptr_t>(this));
  }
  void TearDown() override { fs::remove_file(path.c_str()); }

  void write_file(const void* data, std::size_t len) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    ASSERT_NE(f, nullptr);
    ASSERT_EQ(std::fwrite(data, 1, len, f), len);
    std::fclose(f);
  }
  std::string read_file() {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return {};
    std::string out;
    char buf[4096];
    std::size_t got;
    while ((got = std::fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, got);
    std::fclose(f);
    return out;
  }
};

}  // namespace

TEST_F(MapTest, ReadOnlyMapsContent) {
  const char* msg = "hello mapped world";
  write_file(msg, std::strlen(msg));

  MappedFile m = MappedFile::open(path.c_str(), MappedFile::Mode::kReadOnly);
  ASSERT_TRUE(m.valid());
  EXPECT_EQ(m.size(), std::strlen(msg));
  EXPECT_EQ(std::memcmp(m.data(), msg, std::strlen(msg)), 0);
}

TEST_F(MapTest, ReadWriteFlushPersists) {
  std::string init(64, 'A');
  write_file(init.data(), init.size());

  {
    MappedFile m = MappedFile::open(path.c_str(), MappedFile::Mode::kReadWrite);
    ASSERT_TRUE(m.valid());
    ASSERT_EQ(m.size(), 64u);
    static_cast<char*>(m.data())[0] = 'Z';
    static_cast<char*>(m.data())[63] = 'Y';
    EXPECT_TRUE(m.flush());
  }  // unmap

  std::string after = read_file();
  ASSERT_EQ(after.size(), 64u);
  EXPECT_EQ(after[0], 'Z');
  EXPECT_EQ(after[63], 'Y');
}

TEST_F(MapTest, CopyOnWriteDoesNotTouchFile) {
  std::string init(32, 'A');
  write_file(init.data(), init.size());

  {
    MappedFile m = MappedFile::open(path.c_str(), MappedFile::Mode::kCopyOnWrite);
    ASSERT_TRUE(m.valid());
    static_cast<char*>(m.data())[0] = 'X';       // private write
    EXPECT_EQ(static_cast<char*>(m.data())[0], 'X');  // visible in-process
    EXPECT_TRUE(m.flush());                       // no-op for COW
  }

  // File on disk is UNCHANGED (private pages never written back).
  std::string after = read_file();
  ASSERT_EQ(after.size(), 32u);
  EXPECT_EQ(after[0], 'A');
}

TEST_F(MapTest, LargeFileMultiPage) {
  const std::size_t n = page_size() * 4 + 123;  // spans several pages
  std::string init(n, '\0');
  for (std::size_t i = 0; i < n; ++i) init[i] = static_cast<char>(i & 0x7F);
  write_file(init.data(), n);

  MappedFile m = MappedFile::open(path.c_str(), MappedFile::Mode::kReadOnly);
  ASSERT_TRUE(m.valid());
  ASSERT_EQ(m.size(), n);
  EXPECT_EQ(std::memcmp(m.data(), init.data(), n), 0);
}

TEST_F(MapTest, OffsetMapping) {
  const std::size_t pg = page_size();
  std::string init(pg * 2, 'a');
  init[pg] = 'M';  // marker at the start of page 1
  write_file(init.data(), init.size());

  // Map starting at page 1 (page-aligned offset), to end.
  MappedFile m = MappedFile::open(path.c_str(), MappedFile::Mode::kReadOnly, pg, 0);
  ASSERT_TRUE(m.valid());
  EXPECT_EQ(m.size(), pg);
  EXPECT_EQ(static_cast<char*>(m.data())[0], 'M');
}

TEST_F(MapTest, InvalidPathIsInvalid) {
  MappedFile m = MappedFile::open((path + "_nope").c_str());
  EXPECT_FALSE(m.valid());
  EXPECT_EQ(m.data(), nullptr);
}

TEST_F(MapTest, LengthPastEofRejected) {
  std::string init(64, 'A');
  write_file(init.data(), init.size());
  // An explicit length beyond the file must be refused (no silent file growth
  // on RW, no past-EOF SIGBUS region). File is 64 bytes; ask for 4096.
  MappedFile rw = MappedFile::open(path.c_str(), MappedFile::Mode::kReadWrite, 0,
                                   4096);
  EXPECT_FALSE(rw.valid());
  // The file must NOT have been grown by the rejected request.
  EXPECT_EQ(read_file().size(), 64u);
  // offset past EOF is likewise rejected.
  MappedFile ro = MappedFile::open(path.c_str(), MappedFile::Mode::kReadOnly, 4096, 0);
  EXPECT_FALSE(ro.valid());
}

TEST_F(MapTest, MoveTransfersMapping) {
  const char* msg = "movable";
  write_file(msg, std::strlen(msg));

  MappedFile a = MappedFile::open(path.c_str());
  ASSERT_TRUE(a.valid());
  void* addr = a.data();
  MappedFile b = std::move(a);
  EXPECT_FALSE(a.valid());
  EXPECT_TRUE(b.valid());
  EXPECT_EQ(b.data(), addr);
  EXPECT_EQ(std::memcmp(b.data(), msg, std::strlen(msg)), 0);
}

// Prove a read-only mapping actually traps on write (not just "opened ok"). Runs
// in a forked child so the fault is captured as expected death. Uses its own
// scratch path (fixtures aren't shared into the death child cleanly).
TEST(MapDeathTest, ReadOnlyWriteFaults) {
  char tmp[4096];
  std::size_t n = fs::temp_directory(tmp, sizeof(tmp));
  ASSERT_GT(n, 0u);
  std::string p = std::string(tmp, n) + "/hyle_map_death";
  {
    std::FILE* f = std::fopen(p.c_str(), "wb");
    ASSERT_NE(f, nullptr);
    std::fputs("readonly", f);
    std::fclose(f);
  }
  EXPECT_DEATH(
      {
        MappedFile m = MappedFile::open(p.c_str(), MappedFile::Mode::kReadOnly);
        if (!m.valid()) std::abort();
        volatile auto* d = static_cast<volatile char*>(m.data());
        d[0] = 'X';  // write to a PROT_READ / PAGE_READONLY mapping -> trap
      },
      ".*");
  fs::remove_file(p.c_str());
}

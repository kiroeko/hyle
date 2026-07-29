// Real-filesystem tests for platform::MappedFile (RO / RW / COW mapping).
#if defined(_WIN32)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

#include "hyle/platform/mapped_file.h"

#include <gtest/gtest.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>  // std::abort in the death test
#include <cstring>
#include <string>
#include <string_view>
#include <utility>

#include "hyle/platform/fs.h"
#include "hyle/platform/virtual_memory.h"  // page_size()

using hyle::platform::MappedFile;
using hyle::platform::page_size;
using hyle::Status;
namespace fs = hyle::platform;

namespace {

// A scratch file seeded with `bytes` of content, removed on teardown.
class MapTest : public ::testing::Test {
 protected:
  std::string path;

  void SetUp() override {
    char tmp[4096];
    auto temp = fs::temp_directory(tmp, sizeof(tmp));
    ASSERT_TRUE(temp.has_value()) << temp.error().message();
    ASSERT_GT(*temp, 0u);
    path = std::string(tmp, *temp) + "/hyle_map_test_" +
           std::to_string(reinterpret_cast<std::uintptr_t>(this));
  }
  void TearDown() override { (void)fs::remove_file(path.c_str()); }

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

  MappedFile open_map(MappedFile::Mode mode = MappedFile::Mode::kReadOnly,
                      std::size_t offset = 0, std::size_t length = 0) {
    auto result = MappedFile::open(path.c_str(), mode, offset, length);
    EXPECT_TRUE(result.has_value())
        << (result ? "" : result.error().message());
    return result ? *std::move(result) : MappedFile{};
  }
};

}  // namespace

TEST_F(MapTest, ReadOnlyMapsContent) {
  const char* msg = "hello mapped world";
  write_file(msg, std::strlen(msg));

  MappedFile m = open_map();
  ASSERT_TRUE(m.valid());
  EXPECT_EQ(m.size(), std::strlen(msg));
  EXPECT_EQ(std::memcmp(m.data(), msg, std::strlen(msg)), 0);
}

TEST_F(MapTest, OpenMappingDoesNotBlockAtomicRenameOrDelete) {
  const char* msg = "replaceable mapped content";
  write_file(msg, std::strlen(msg));
  MappedFile m = open_map();
  ASSERT_TRUE(m.valid());

  const std::string renamed = path + ".renamed";
  ASSERT_TRUE(fs::rename_path(path.c_str(), renamed.c_str()).ok());
  ASSERT_TRUE(fs::remove_file(renamed.c_str()).ok());
  EXPECT_EQ(std::memcmp(m.data(), msg, std::strlen(msg)), 0);
}

TEST_F(MapTest, AdviceCoversWholeAndBoundedRanges) {
  std::string init(page_size() * 2, 'A');
  write_file(init.data(), init.size());
  MappedFile m = open_map();

  EXPECT_TRUE(m.advise(MappedFile::Advice::kNormal).ok());
  EXPECT_TRUE(
      m.advise(MappedFile::Advice::kNormal, 7, page_size()).ok());
  Status sequential = m.advise(MappedFile::Advice::kSequential);
  EXPECT_TRUE(sequential.ok() || sequential.IsUnimplemented());
  Status prefetch = m.prefetch();
  EXPECT_TRUE(prefetch.ok() || prefetch.IsUnimplemented());
  EXPECT_TRUE(m.advise(MappedFile::Advice::kNormal, m.size(), 0).ok());
  EXPECT_TRUE(m.advise(MappedFile::Advice::kNormal, m.size() + 1, 0)
                  .IsOutOfRange());
  EXPECT_TRUE(m.advise(MappedFile::Advice::kNormal, m.size() - 1, 2)
                  .IsOutOfRange());
  EXPECT_TRUE(m.advise(static_cast<MappedFile::Advice>(255))
                  .IsInvalidArgument());
}

TEST_F(MapTest, PageLockIsExplicitAndIdempotentWhenAvailable) {
  std::string init(page_size(), 'A');
  write_file(init.data(), init.size());
  MappedFile m = open_map();
  EXPECT_FALSE(m.locked());

  Status locked = m.lock();
  if (locked.ok()) {
    EXPECT_TRUE(m.locked());
    EXPECT_TRUE(m.lock().ok());

    MappedFile moved = std::move(m);
    EXPECT_FALSE(m.locked());
    EXPECT_TRUE(moved.locked());
    EXPECT_TRUE(moved.unlock().ok());
    EXPECT_FALSE(moved.locked());
    EXPECT_TRUE(moved.unlock().ok());
  } else {
    // Containers and restricted CI commonly set a zero/small memlock quota.
    EXPECT_TRUE(locked.IsPermissionDenied() || locked.IsResourceExhausted() ||
                locked.IsUnavailable());
    EXPECT_FALSE(m.locked());
  }
}

TEST_F(MapTest, ReadWriteFlushPersists) {
  std::string init(64, 'A');
  write_file(init.data(), init.size());

  {
    MappedFile m = open_map(MappedFile::Mode::kReadWrite);
    ASSERT_TRUE(m.valid());
    ASSERT_EQ(m.size(), 64u);
    static_cast<char*>(m.data())[0] = 'Z';
    static_cast<char*>(m.data())[63] = 'Y';
    EXPECT_TRUE(m.flush().ok());
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
    MappedFile m = open_map(MappedFile::Mode::kCopyOnWrite);
    ASSERT_TRUE(m.valid());
    static_cast<char*>(m.data())[0] = 'X';       // private write
    EXPECT_EQ(static_cast<char*>(m.data())[0], 'X');  // visible in-process
    EXPECT_TRUE(m.flush().ok());                  // no-op for COW
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

  MappedFile m = open_map();
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
  MappedFile m = open_map(MappedFile::Mode::kReadOnly, pg, 0);
  ASSERT_TRUE(m.valid());
  EXPECT_EQ(m.size(), pg);
  EXPECT_EQ(static_cast<char*>(m.data())[0], 'M');
}

// A NON-page-aligned offset must still work: the header rounds the offset down to
// the mapping granularity internally, but data() has to point at the EXACT byte
// the caller asked for -- not the rounded base. Seed a positional pattern so a
// stray granularity offset in data() would show up as wrong content.
TEST_F(MapTest, UnalignedOffsetToEnd) {
  const std::size_t pg = page_size();
  const std::size_t n = pg * 2;
  std::string init(n, '\0');
  for (std::size_t i = 0; i < n; ++i) init[i] = static_cast<char>(i & 0x7F);
  write_file(init.data(), n);

  const std::size_t off = 100;  // deliberately unaligned (not a page/granularity)
  MappedFile m = open_map(MappedFile::Mode::kReadOnly, off, 0);
  ASSERT_TRUE(m.valid()) << "unaligned offset must map (rounded internally)";
  EXPECT_EQ(m.size(), n - off);  // length 0 => exactly [off, EOF)
  auto* p = static_cast<const unsigned char*>(m.data());
  // data()[0] is the byte AT the requested offset, not the rounded-down base.
  EXPECT_EQ(p[0], static_cast<unsigned char>(off & 0x7F));
  EXPECT_EQ(p[m.size() - 1], static_cast<unsigned char>((n - 1) & 0x7F));
  EXPECT_EQ(std::memcmp(m.data(), init.data() + off, m.size()), 0);
}

// Unaligned offset AND an explicit bounded length that crosses a page boundary:
// the window [pg+50, pg+50+200) starts unaligned and past the first page. data()
// must land exactly on pg+50 and size() must be the requested 200.
TEST_F(MapTest, UnalignedOffsetExplicitLength) {
  const std::size_t pg = page_size();
  const std::size_t n = pg * 2;
  std::string init(n, '\0');
  for (std::size_t i = 0; i < n; ++i) init[i] = static_cast<char>(i & 0x7F);
  write_file(init.data(), n);

  const std::size_t off = pg + 50;  // unaligned, and past the first page
  const std::size_t len = 200;      // within EOF (pg+250 < 2*pg)
  MappedFile m = open_map(MappedFile::Mode::kReadOnly, off, len);
  ASSERT_TRUE(m.valid());
  EXPECT_EQ(m.size(), len);
  EXPECT_EQ(std::memcmp(m.data(), init.data() + off, len), 0);
}

TEST_F(MapTest, InvalidPathIsInvalid) {
  auto m = MappedFile::open((path + "_nope").c_str());
  ASSERT_FALSE(m.has_value());
  EXPECT_TRUE(m.error().IsNotFound());
#if defined(_WIN32)
  EXPECT_NE(m.error().message().find("CreateFileW"), std::string_view::npos);
  EXPECT_EQ(m.error().native_error().domain, hyle::NativeErrorDomain::Win32);
  EXPECT_EQ(m.error().native_error().code, 2);  // ERROR_FILE_NOT_FOUND
#else
  EXPECT_NE(m.error().message().find("open"), std::string_view::npos);
  EXPECT_EQ(m.error().native_error().domain,
            hyle::NativeErrorDomain::PosixErrno);
  EXPECT_EQ(m.error().native_error().code, ENOENT);
#endif
}

TEST_F(MapTest, InvalidArgumentsAndEmptyMappingReturnStatus) {
  auto empty_path = MappedFile::open("");
  auto null_path = MappedFile::open(nullptr);
  auto invalid_mode = MappedFile::open(
      path.c_str(), static_cast<MappedFile::Mode>(255));
  ASSERT_FALSE(empty_path.has_value());
  ASSERT_FALSE(null_path.has_value());
  ASSERT_FALSE(invalid_mode.has_value());
  EXPECT_TRUE(empty_path.error().IsInvalidArgument());
  EXPECT_TRUE(null_path.error().IsInvalidArgument());
  EXPECT_TRUE(invalid_mode.error().IsInvalidArgument());

  write_file("", 0);
  auto empty_file = MappedFile::open(path.c_str());
  ASSERT_FALSE(empty_file.has_value());
  EXPECT_TRUE(empty_file.error().IsFailedPrecondition());

  MappedFile unopened;
  EXPECT_TRUE(unopened.flush().IsFailedPrecondition());
  EXPECT_TRUE(unopened.advise(MappedFile::Advice::kNormal)
                  .IsFailedPrecondition());
  EXPECT_TRUE(unopened.lock().IsFailedPrecondition());
  EXPECT_TRUE(unopened.unlock().IsFailedPrecondition());
}

TEST_F(MapTest, LengthPastEofRejected) {
  std::string init(64, 'A');
  write_file(init.data(), init.size());
  // An explicit length beyond the file must be refused (no silent file growth
  // on RW, no past-EOF SIGBUS region). File is 64 bytes; ask for 4096.
  auto rw = MappedFile::open(path.c_str(), MappedFile::Mode::kReadWrite, 0,
                             4096);
  ASSERT_FALSE(rw.has_value());
  EXPECT_TRUE(rw.error().IsOutOfRange());
  // The file must NOT have been grown by the rejected request.
  EXPECT_EQ(read_file().size(), 64u);
  // offset past EOF is likewise rejected.
  auto ro = MappedFile::open(path.c_str(), MappedFile::Mode::kReadOnly, 4096, 0);
  ASSERT_FALSE(ro.has_value());
  EXPECT_TRUE(ro.error().IsOutOfRange());
}

TEST_F(MapTest, MoveTransfersMapping) {
  const char* msg = "movable";
  write_file(msg, std::strlen(msg));

  MappedFile a = open_map();
  ASSERT_TRUE(a.valid());
  void* addr = a.data();
  MappedFile b = std::move(a);
  EXPECT_FALSE(a.valid());
  EXPECT_TRUE(b.valid());
  EXPECT_EQ(b.data(), addr);
  EXPECT_EQ(std::memcmp(b.data(), msg, std::strlen(msg)), 0);
}

// Move-ASSIGN (operator=(MappedFile&&)) onto an already-live mapping: the target
// must drop its previous mapping and take over the source's. Two distinct files
// so the transfer of content is observable; the source is left !valid().
TEST_F(MapTest, MoveAssignReleasesPrevious) {
  const char* src_msg = "source-mapping-content";
  write_file(src_msg, std::strlen(src_msg));  // `path` holds the SOURCE content

  // A second scratch file for the assign target's initial (to-be-dropped) mapping.
  std::string dst_path = path + "_dst";
  const char* dst_msg = "destination-initial";
  {
    std::FILE* f = std::fopen(dst_path.c_str(), "wb");
    ASSERT_NE(f, nullptr);
    ASSERT_EQ(std::fwrite(dst_msg, 1, std::strlen(dst_msg), f), std::strlen(dst_msg));
    std::fclose(f);
  }

  MappedFile src = open_map();
  auto dst_result = MappedFile::open(dst_path.c_str());
  ASSERT_TRUE(dst_result.has_value());
  MappedFile dst = *std::move(dst_result);
  ASSERT_TRUE(src.valid());
  ASSERT_TRUE(dst.valid());
  void* src_addr = src.data();
  std::size_t src_size = src.size();

  dst = std::move(src);  // drops dst's old mapping, adopts src's

  EXPECT_FALSE(src.valid());
  EXPECT_EQ(src.data(), nullptr);
  ASSERT_TRUE(dst.valid());
  EXPECT_EQ(dst.data(), src_addr);
  EXPECT_EQ(dst.size(), src_size);
  EXPECT_EQ(std::memcmp(dst.data(), src_msg, std::strlen(src_msg)), 0);

  (void)fs::remove_file(dst_path.c_str());
}

// Prove a read-only mapping actually traps on write (not just "opened ok"). Runs
// in a forked child so the fault is captured as expected death. Uses its own
// scratch path (fixtures aren't shared into the death child cleanly).
TEST(MapDeathTest, ReadOnlyWriteFaults) {
  char tmp[4096];
  auto temp = fs::temp_directory(tmp, sizeof(tmp));
  ASSERT_TRUE(temp.has_value()) << temp.error().message();
  ASSERT_GT(*temp, 0u);
  std::string p = std::string(tmp, *temp) + "/hyle_map_death";
  {
    std::FILE* f = std::fopen(p.c_str(), "wb");
    ASSERT_NE(f, nullptr);
    std::fputs("readonly", f);
    std::fclose(f);
  }
  EXPECT_DEATH(
      {
        auto result = MappedFile::open(p.c_str(), MappedFile::Mode::kReadOnly);
        if (!result) std::abort();
        MappedFile m = *std::move(result);
        volatile auto* d = static_cast<volatile char*>(m.data());
        d[0] = 'X';  // write to a PROT_READ / PAGE_READONLY mapping -> trap
      },
      ".*");
  (void)fs::remove_file(p.c_str());
}

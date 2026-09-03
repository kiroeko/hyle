// Real-filesystem tests for mem::FileMapping (RO / RW / COW mapping).
#if defined(_WIN32)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

#include "hyle/mem/mapping.h"

#include <gtest/gtest.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>  // std::abort in the death test
#include <cstring>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "hyle/platform/fs.h"
#include "hyle/core/environment.h"
#include "hyle/platform/virtual_memory.h"  // page_size()

using hyle::mem::FileMapping;
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

  FileMapping open_map(hyle::mem::MappingAccess mode = hyle::mem::MappingAccess::kReadOnly,
                      std::size_t offset = 0, std::size_t length = 0) {
    auto result = FileMapping::open(path.c_str(), mode, offset, length);
    EXPECT_TRUE(result.has_value())
        << (result ? "" : result.error().message());
    return result ? *std::move(result) : FileMapping{};
  }
};

}  // namespace

TEST_F(MapTest, ReadOnlyMapsContent) {
  const char* msg = "hello mapped world";
  write_file(msg, std::strlen(msg));

  FileMapping m = open_map();
  ASSERT_TRUE(m.valid());
  EXPECT_EQ(m.size(), std::strlen(msg));
  EXPECT_EQ(m.access(), hyle::mem::MappingAccess::kReadOnly);
  EXPECT_EQ(m.bytes().size(), std::strlen(msg));
  EXPECT_EQ(std::memcmp(m.bytes().data(), msg, std::strlen(msg)), 0);
  EXPECT_TRUE(m.writable_bytes().error().IsFailedPrecondition());
  static_assert(std::is_same_v<decltype(m.data()), const void*>);
}

TEST_F(MapTest, ConsumesExistingFileHandle) {
  const char* msg = "already opened";
  write_file(msg, std::strlen(msg));
  auto file = fs::open_file(path.c_str(), fs::kOpenRead);
  ASSERT_TRUE(file.has_value()) << file.error().message();

  auto mapped = FileMapping::map(std::move(*file));
  ASSERT_TRUE(mapped.has_value()) << mapped.error().message();
  EXPECT_FALSE(file->valid());
  EXPECT_EQ(std::memcmp(mapped->bytes().data(), msg, std::strlen(msg)), 0);
}

TEST_F(MapTest, MapConsumesHandleOnFailure) {
  write_file("x", 1);
  auto file = fs::open_file(path.c_str(), fs::kOpenRead);
  ASSERT_TRUE(file.has_value()) << file.error().message();
  auto mapped = FileMapping::map(
      std::move(*file), static_cast<hyle::mem::MappingAccess>(255));
  EXPECT_FALSE(mapped.has_value());
  EXPECT_TRUE(mapped.error().IsInvalidArgument());
  EXPECT_FALSE(file->valid());
}

TEST_F(MapTest, OpenMappingDoesNotBlockAtomicRenameOrDelete) {
  const char* msg = "replaceable mapped content";
  write_file(msg, std::strlen(msg));
  FileMapping m = open_map();
  ASSERT_TRUE(m.valid());

  const std::string renamed = path + ".renamed";
  ASSERT_TRUE(fs::rename_path(path.c_str(), renamed.c_str()).ok());
  ASSERT_TRUE(fs::remove_file(renamed.c_str()).ok());
  EXPECT_EQ(std::memcmp(m.data(), msg, std::strlen(msg)), 0);
}

TEST_F(MapTest, AdviceCoversWholeAndBoundedRanges) {
  std::string init(page_size() * 2, 'A');
  write_file(init.data(), init.size());
  FileMapping m = open_map();

  EXPECT_TRUE(m.advise(hyle::mem::MappingAdvice::kNormal).ok());
  EXPECT_TRUE(
      m.advise(hyle::mem::MappingAdvice::kNormal, 7, page_size()).ok());
  Status sequential = m.advise(hyle::mem::MappingAdvice::kSequential);
  EXPECT_TRUE(sequential.ok() || sequential.IsUnimplemented());
  Status prefetch = m.prefetch();
  EXPECT_TRUE(prefetch.ok() || prefetch.IsUnimplemented());
  EXPECT_TRUE(m.populate().ok());
  EXPECT_TRUE(m.populate(7, page_size()).ok());
  EXPECT_TRUE(m.populate(m.size(), 0).ok());
  EXPECT_TRUE(m.populate(m.size() + 1, 0).IsOutOfRange());
  EXPECT_TRUE(m.populate(m.size() - 1, 2).IsOutOfRange());
  EXPECT_TRUE(m.advise(hyle::mem::MappingAdvice::kNormal, m.size(), 0).ok());
  EXPECT_TRUE(m.advise(hyle::mem::MappingAdvice::kNormal, m.size() + 1, 0)
                  .IsOutOfRange());
  EXPECT_TRUE(m.advise(hyle::mem::MappingAdvice::kNormal, m.size() - 1, 2)
                  .IsOutOfRange());
  EXPECT_TRUE(m.advise(static_cast<hyle::mem::MappingAdvice>(255))
                  .IsInvalidArgument());
}

TEST_F(MapTest, PageLockGrowsMonotonicallyAndMovesWhenAvailable) {
  std::string init(page_size() * 3, 'A');
  write_file(init.data(), init.size());
  FileMapping m = open_map();
  EXPECT_FALSE(m.locked());
  EXPECT_EQ(m.locked_size(), 0u);
  EXPECT_TRUE(m.lock_prefix(0).ok());
  EXPECT_FALSE(m.locked());

  Status locked = m.lock_prefix(1);
  if (locked.ok()) {
    EXPECT_TRUE(m.locked());
    ASSERT_GE(m.locked_size(), 1u);
    ASSERT_LE(m.locked_size(), m.size());
    const std::size_t first = m.locked_size();
    EXPECT_TRUE(m.lock_prefix(1).ok());
    EXPECT_EQ(m.locked_size(), first);

    if (first < m.size()) {
      Status grown = m.lock_prefix(first + 1);
      if (grown.ok()) {
        EXPECT_GT(m.locked_size(), first);
        const std::size_t grown_size = m.locked_size();
        EXPECT_TRUE(m.lock_prefix(1).ok());
        EXPECT_EQ(m.locked_size(), grown_size);
      } else {
        // A quota failure on the new native page must not forget the already
        // applied prefix.
        EXPECT_TRUE(grown.IsPermissionDenied() ||
                    grown.IsResourceExhausted() || grown.IsUnavailable());
        EXPECT_EQ(m.locked_size(), first);
      }
    }

    hyle::mem::ResidencyOptions different_mode;
    different_mode.mode = hyle::mem::ResidencyMode::kOnFault;
    EXPECT_TRUE(m.lock_prefix(1, different_mode).IsFailedPrecondition());

    FileMapping moved = std::move(m);
    EXPECT_FALSE(m.locked());
    EXPECT_EQ(m.locked_size(), 0u);
    EXPECT_TRUE(moved.locked());
    EXPECT_GE(moved.locked_size(), first);
    EXPECT_TRUE(moved.unlock().ok());
    EXPECT_FALSE(moved.locked());
    EXPECT_EQ(moved.locked_size(), 0u);
    EXPECT_TRUE(moved.unlock().ok());
  } else {
    // Containers and restricted CI commonly set a zero/small memlock quota.
    EXPECT_TRUE(locked.IsPermissionDenied() || locked.IsResourceExhausted() ||
                locked.IsUnavailable());
    EXPECT_FALSE(m.locked());
  }
}

TEST_F(MapTest, PageLockSupportsEveryAccessModeAndResetClearsState) {
  std::string init(page_size(), 'A');
  write_file(init.data(), init.size());

  constexpr hyle::mem::MappingAccess kModes[] = {
      hyle::mem::MappingAccess::kReadOnly, hyle::mem::MappingAccess::kReadWrite,
      hyle::mem::MappingAccess::kCopyOnWrite};
  for (hyle::mem::MappingAccess mode : kModes) {
    FileMapping m = open_map(mode);
    Status locked = m.lock_prefix(1);
    if (locked.ok()) {
      EXPECT_TRUE(m.locked());
      EXPECT_GE(m.locked_size(), 1u);
      m.reset();
      EXPECT_FALSE(m.valid());
      EXPECT_FALSE(m.locked());
      EXPECT_EQ(m.locked_size(), 0u);
    } else {
      EXPECT_TRUE(locked.IsPermissionDenied() ||
                  locked.IsResourceExhausted() || locked.IsUnavailable());
    }
  }
}

TEST_F(MapTest, WholeLockRemainsTheNaturalConvenience) {
  std::string init(page_size(), 'A');
  write_file(init.data(), init.size());
  FileMapping m = open_map();
  Status locked = m.lock();
  if (locked.ok()) {
    EXPECT_EQ(m.locked_size(), m.size());
    EXPECT_TRUE(m.unlock().ok());
  } else {
    EXPECT_TRUE(locked.IsPermissionDenied() || locked.IsResourceExhausted() ||
                locked.IsUnavailable());
  }
}

TEST_F(MapTest, PrefixLockAccountsForUnalignedMappingOffset) {
  const std::size_t pg = page_size();
  std::string init(pg * 2, 'A');
  write_file(init.data(), init.size());
  FileMapping m = open_map(hyle::mem::MappingAccess::kReadOnly, 100, pg);
  Status locked = m.lock_prefix(1);
  if (locked.ok()) {
    EXPECT_GE(m.locked_size(), 1u);
    EXPECT_LE(m.locked_size(), m.size());
    EXPECT_TRUE(m.unlock().ok());
  } else {
    EXPECT_TRUE(locked.IsPermissionDenied() || locked.IsResourceExhausted() ||
                locked.IsUnavailable());
  }
}

TEST_F(MapTest, OnFaultLockIsExplicitlySupportedOrRejected) {
  std::string init(page_size(), 'A');
  write_file(init.data(), init.size());
  FileMapping m = open_map();
  hyle::mem::ResidencyOptions options;
  options.mode = hyle::mem::ResidencyMode::kOnFault;
  Status locked = m.lock_prefix(1, options);
#if HYLE_OS_LINUX
  if (locked.ok()) {
    EXPECT_TRUE(m.locked());
    EXPECT_GE(m.locked_size(), 1u);
    EXPECT_TRUE(m.unlock().ok());
  } else {
    // Old kernels/libcs and restricted containers may lack the primitive or
    // the memlock budget. All are observable instead of an eager fallback.
    EXPECT_TRUE(locked.IsUnimplemented() || locked.IsPermissionDenied() ||
                locked.IsResourceExhausted() || locked.IsUnavailable());
  }
#else
  EXPECT_TRUE(locked.IsUnimplemented());
  EXPECT_FALSE(m.locked());
#endif
}

TEST_F(MapTest, ExplicitLimitAdjustmentPreservesStateOnFailure) {
  std::string init(page_size() * 512, 'A');
  write_file(init.data(), init.size());
  FileMapping m = open_map();
  hyle::mem::ResidencyOptions options;
  options.limit_policy = hyle::mem::ResidencyLimitPolicy::kAdjustIfNeeded;
  Status locked = m.lock(options);
  if (locked.ok()) {
    EXPECT_EQ(m.locked_size(), m.size());
    EXPECT_TRUE(m.unlock().ok());
  } else {
    EXPECT_TRUE(locked.IsPermissionDenied() || locked.IsResourceExhausted() ||
                locked.IsUnavailable() || locked.IsOutOfRange());
    EXPECT_FALSE(m.locked());
    EXPECT_EQ(m.locked_size(), 0u);
  }
}

TEST_F(MapTest, ReadWriteFlushPersists) {
  std::string init(64, 'A');
  write_file(init.data(), init.size());

  {
    FileMapping m = open_map(hyle::mem::MappingAccess::kReadWrite);
    ASSERT_TRUE(m.valid());
    ASSERT_EQ(m.size(), 64u);
    auto bytes = m.writable_bytes();
    ASSERT_TRUE(bytes.has_value()) << bytes.error().message();
    (*bytes)[0] = std::byte{'Z'};
    (*bytes)[63] = std::byte{'Y'};
    EXPECT_TRUE(m.flush(0, 1).ok());
    EXPECT_TRUE(m.flush(63, 1).ok());
    EXPECT_TRUE(m.flush(m.size(), 0).ok());
    EXPECT_TRUE(m.flush(m.size() + 1, 0).IsOutOfRange());
    EXPECT_TRUE(m.flush(m.size() - 1, 2).IsOutOfRange());
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
    FileMapping m = open_map(hyle::mem::MappingAccess::kCopyOnWrite);
    ASSERT_TRUE(m.valid());
    auto bytes = m.writable_bytes();
    ASSERT_TRUE(bytes.has_value()) << bytes.error().message();
    (*bytes)[0] = std::byte{'X'};  // private write
    EXPECT_EQ((*bytes)[0], std::byte{'X'});  // visible in-process
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

  FileMapping m = open_map();
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
  FileMapping m = open_map(hyle::mem::MappingAccess::kReadOnly, pg, 0);
  ASSERT_TRUE(m.valid());
  EXPECT_EQ(m.size(), pg);
  EXPECT_EQ(static_cast<const char*>(m.data())[0], 'M');
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
  FileMapping m = open_map(hyle::mem::MappingAccess::kReadOnly, off, 0);
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
  FileMapping m = open_map(hyle::mem::MappingAccess::kReadOnly, off, len);
  ASSERT_TRUE(m.valid());
  EXPECT_EQ(m.size(), len);
  EXPECT_EQ(std::memcmp(m.data(), init.data() + off, len), 0);
}

TEST_F(MapTest, InvalidPathIsInvalid) {
  auto m = FileMapping::open((path + "_nope").c_str());
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
  auto empty_path = FileMapping::open("");
  auto null_path = FileMapping::open(nullptr);
  auto invalid_mode = FileMapping::open(
      path.c_str(), static_cast<hyle::mem::MappingAccess>(255));
  ASSERT_FALSE(empty_path.has_value());
  ASSERT_FALSE(null_path.has_value());
  ASSERT_FALSE(invalid_mode.has_value());
  EXPECT_TRUE(empty_path.error().IsInvalidArgument());
  EXPECT_TRUE(null_path.error().IsInvalidArgument());
  EXPECT_TRUE(invalid_mode.error().IsInvalidArgument());

  write_file("", 0);
  auto empty_file = FileMapping::open(path.c_str());
  ASSERT_FALSE(empty_file.has_value());
  EXPECT_TRUE(empty_file.error().IsFailedPrecondition());

  FileMapping unopened;
  EXPECT_TRUE(unopened.flush().IsFailedPrecondition());
  EXPECT_TRUE(unopened.advise(hyle::mem::MappingAdvice::kNormal)
                  .IsFailedPrecondition());
  EXPECT_TRUE(unopened.populate().IsFailedPrecondition());
  EXPECT_TRUE(unopened.lock().IsFailedPrecondition());
  EXPECT_TRUE(unopened.lock_prefix(1).IsFailedPrecondition());
  EXPECT_TRUE(unopened.unlock().IsFailedPrecondition());

  write_file("abcd", 4);
  FileMapping mapped = open_map();
  EXPECT_TRUE(mapped.lock_prefix(mapped.size() + 1).IsOutOfRange());
  hyle::mem::ResidencyOptions invalid_lock_mode;
  invalid_lock_mode.mode = static_cast<hyle::mem::ResidencyMode>(255);
  EXPECT_TRUE(
      mapped.lock_prefix(1, invalid_lock_mode).IsInvalidArgument());
  hyle::mem::ResidencyOptions invalid_limit;
  invalid_limit.limit_policy =
      static_cast<hyle::mem::ResidencyLimitPolicy>(255);
  EXPECT_TRUE(mapped.lock_prefix(1, invalid_limit).IsInvalidArgument());
}

TEST_F(MapTest, LengthPastEofRejected) {
  std::string init(64, 'A');
  write_file(init.data(), init.size());
  // An explicit length beyond the file must be refused (no silent file growth
  // on RW, no past-EOF SIGBUS region). File is 64 bytes; ask for 4096.
  auto rw = FileMapping::open(path.c_str(), hyle::mem::MappingAccess::kReadWrite, 0,
                             4096);
  ASSERT_FALSE(rw.has_value());
  EXPECT_TRUE(rw.error().IsOutOfRange());
  // The file must NOT have been grown by the rejected request.
  EXPECT_EQ(read_file().size(), 64u);
  // offset past EOF is likewise rejected.
  auto ro = FileMapping::open(path.c_str(), hyle::mem::MappingAccess::kReadOnly, 4096, 0);
  ASSERT_FALSE(ro.has_value());
  EXPECT_TRUE(ro.error().IsOutOfRange());
}

TEST_F(MapTest, MoveTransfersMapping) {
  const char* msg = "movable";
  write_file(msg, std::strlen(msg));

  FileMapping a = open_map();
  ASSERT_TRUE(a.valid());
  const void* addr = a.data();
  FileMapping b = std::move(a);
  EXPECT_FALSE(a.valid());
  EXPECT_TRUE(b.valid());
  EXPECT_EQ(b.data(), addr);
  EXPECT_EQ(std::memcmp(b.data(), msg, std::strlen(msg)), 0);
}

// Move-ASSIGN (operator=(FileMapping&&)) onto an already-live mapping: the target
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

  FileMapping src = open_map();
  auto dst_result = FileMapping::open(dst_path.c_str());
  ASSERT_TRUE(dst_result.has_value());
  FileMapping dst = *std::move(dst_result);
  ASSERT_TRUE(src.valid());
  ASSERT_TRUE(dst.valid());
  const void* src_addr = src.data();
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
        auto result = FileMapping::open(p.c_str(), hyle::mem::MappingAccess::kReadOnly);
        if (!result) std::abort();
        FileMapping m = *std::move(result);
        volatile auto* d = static_cast<volatile char*>(
            const_cast<void*>(m.data()));
        d[0] = 'X';  // write to a PROT_READ / PAGE_READONLY mapping -> trap
      },
      ".*");
  (void)fs::remove_file(p.c_str());
}

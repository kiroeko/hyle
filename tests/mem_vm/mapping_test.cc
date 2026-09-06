#if defined(_WIN32)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>

#include "hyle/mem/mapping.h"
#include "hyle/platform/fs.h"
#include "hyle/platform/virtual_memory.h"

namespace {

using namespace hyle::mem;
using hyle::Status;
namespace fs = hyle::platform;

class FileMappingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    char temporary[4096];
    auto directory = fs::temp_directory(temporary, sizeof(temporary));
    ASSERT_TRUE(directory.has_value()) << directory.error().message();
    path_ = std::string(temporary, *directory) + "/hyle_mem_mapping_" +
            std::to_string(reinterpret_cast<std::uintptr_t>(this));
  }

  void TearDown() override { fs::remove_file(path_.c_str()).IgnoreError(); }

  void seed(const void* data, std::size_t size) {
    std::FILE* file = std::fopen(path_.c_str(), "wb");
    ASSERT_NE(file, nullptr);
    ASSERT_EQ(std::fwrite(data, 1, size, file), size);
    std::fclose(file);
  }

  std::string path_;
};

TEST(VirtualMapping, OwnsAnonymousRangesAndForwardsPageControls) {
  const std::size_t page = fs::page_size();
  auto allocated = VirtualMapping::allocate_rw(page + 1);
  ASSERT_TRUE(allocated.has_value()) << allocated.error().message();
  VirtualMapping mapping = std::move(*allocated);
  ASSERT_TRUE(mapping.valid());
  EXPECT_EQ(mapping.size(), page * 2);

  auto* bytes = static_cast<std::byte*>(mapping.data());
  bytes[0] = std::byte{0x2a};
  EXPECT_TRUE(
      fs::vm_advise(mapping.data(), mapping.size(), MappingAdvice::kNormal)
          .ok());
  EXPECT_TRUE(mapping.protect(0, page, MappingProtection::kReadWrite).ok());
  EXPECT_TRUE(mapping.decommit(page, page).ok());
  EXPECT_TRUE(mapping.commit(page, page).ok());
  EXPECT_TRUE(mapping.release().ok());
  EXPECT_FALSE(mapping.valid());
}

TEST(VirtualMapping, RejectsInvalidRangesAndPolicies) {
  const std::size_t page = fs::page_size();
  VirtualMapping mapping = VirtualMapping::reserve(page).value();
  EXPECT_TRUE(mapping.commit(page + 1, 1).IsOutOfRange());
  EXPECT_TRUE(mapping
                  .protect(0, page, static_cast<MappingProtection>(255))
                  .IsInvalidArgument());
  EXPECT_TRUE(fs::vm_advise(mapping.data(), page,
                            static_cast<MappingAdvice>(255))
                  .IsInvalidArgument());
  ResidencyOptions options;
  options.mode = static_cast<ResidencyMode>(255);
  EXPECT_TRUE(fs::vm_lock(mapping.data(), page, options).IsInvalidArgument());
}

TEST_F(FileMappingTest, ReadOnlyMappingProvidesConstBufferJourney) {
  constexpr char contents[] = "mapped model bytes";
  seed(contents, sizeof(contents));

  auto opened = FileMapping::open(path_.c_str());
  ASSERT_TRUE(opened.has_value()) << opened.error().message();
  FileMapping mapping = std::move(*opened);
  ASSERT_TRUE(mapping.valid());
  EXPECT_EQ(mapping.size(), sizeof(contents));
  EXPECT_TRUE(mapping.writable_bytes().error().IsPermissionDenied());

  std::array<std::byte, sizeof(contents)> copied{};
  auto view = mapping.view();
  ASSERT_TRUE(view.has_value()) << view.error().message();
  ASSERT_TRUE(view->read(copied).ok());
  EXPECT_EQ(std::memcmp(copied.data(), contents, sizeof(contents)), 0);
  EXPECT_TRUE(mapping.advise(MappingAdvice::kNormal).ok());
}

TEST_F(FileMappingTest, ReadWriteMappingFlushesExplicitly) {
  std::array<std::byte, 64> contents{};
  seed(contents.data(), contents.size());

  FileMapping mapping =
      FileMapping::open(path_.c_str(), MappingAccess::kReadWrite).value();
  auto writable = mapping.writable_bytes();
  ASSERT_TRUE(writable.has_value()) << writable.error().message();
  (*writable)[7] = std::byte{0x5a};
  EXPECT_TRUE(mapping.flush(7, 1).ok());
}

TEST(SharedMapping, NamedOwnersObserveTheSameStorage) {
  int marker = 0;
  const std::string name =
      "hsm_o_" +
      std::to_string(reinterpret_cast<std::uintptr_t>(&marker));
  SharedMapping::unlink(name.c_str()).IgnoreError();

  auto created = SharedMapping::create(name.c_str(), 128);
  ASSERT_TRUE(created.has_value()) << created.error().message();
  SharedMapping owner = std::move(*created);
  EXPECT_EQ(owner.access(), MappingAccess::kReadWrite);
  auto duplicate = SharedMapping::create(name.c_str(), 128);
  EXPECT_FALSE(duplicate.has_value());
  EXPECT_TRUE(duplicate.error().IsAlreadyExists());

  auto opened = SharedMapping::open(name.c_str(), 128);
  ASSERT_TRUE(opened.has_value()) << opened.error().message();
  SharedMapping peer = std::move(*opened);
  EXPECT_EQ(peer.access(), MappingAccess::kReadOnly);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(owner.data()) % fs::page_size(),
            0u);
  auto owner_view = owner.writable_view();
  ASSERT_TRUE(owner_view.has_value()) << owner_view.error().message();
  ASSERT_TRUE(owner_view->fill(std::byte{0x39}).ok());
  EXPECT_TRUE(peer.writable_bytes().error().IsPermissionDenied());
  EXPECT_TRUE(peer.flush().IsFailedPrecondition());

  std::array<std::byte, 128> observed{};
  ASSERT_EQ(peer.size(), observed.size());
  const Status read =
      static_cast<const SharedMapping&>(peer).view().read(observed);
  ASSERT_TRUE(read.ok()) << read.message();
  for (std::byte value : observed) EXPECT_EQ(value, std::byte{0x39});
  EXPECT_TRUE(owner.flush().ok());

  Status unlink = SharedMapping::unlink(name.c_str());
  EXPECT_TRUE(unlink.ok() || unlink.IsUnimplemented()) << unlink.message();
}

TEST(SharedMapping, ReadOnlyPeerObservesStorageAcrossProcessBoundary) {
  int marker = 0;
  const std::string name =
      "hsm_p_" +
      std::to_string(reinterpret_cast<std::uintptr_t>(&marker));
  SharedMapping::unlink(name.c_str()).IgnoreError();

  constexpr std::size_t kSize = 4096;
  auto created = SharedMapping::create(name.c_str(), kSize);
  ASSERT_TRUE(created.has_value()) << created.error().message();
  SharedMapping owner = std::move(*created);
  auto bytes = owner.writable_bytes();
  ASSERT_TRUE(bytes.has_value()) << bytes.error().message();
  std::fill(bytes->begin(), bytes->end(), std::byte{0x39});
  ASSERT_TRUE(owner.flush().ok());

#if defined(_WIN32)
  const std::string command =
      "\"\"" HYLE_MEM_VM_SHARED_PEER_PATH "\" \"" + name + "\" " +
      std::to_string(kSize) + " 57\"";
#else
  const std::string command =
      "\"" HYLE_MEM_VM_SHARED_PEER_PATH "\" \"" + name + "\" " +
      std::to_string(kSize) + " 57";
#endif
  EXPECT_EQ(std::system(command.c_str()), 0);

  EXPECT_TRUE(owner.reset().ok());
  const Status unlink = SharedMapping::unlink(name.c_str());
  EXPECT_TRUE(unlink.ok() || unlink.IsUnimplemented()) << unlink.message();
}

TEST(SharedMapping, ExplicitProtocolRejectsAmbiguousSizeAndAccess) {
  int marker = 0;
  const std::string name =
      "hsm_c_" +
      std::to_string(reinterpret_cast<std::uintptr_t>(&marker));
  SharedMapping::unlink(name.c_str()).IgnoreError();

  EXPECT_TRUE(SharedMapping::create(name.c_str(), 0)
                  .error()
                  .IsInvalidArgument());
  EXPECT_TRUE(SharedMapping::open(name.c_str(), 0)
                  .error()
                  .IsInvalidArgument());
  EXPECT_TRUE(SharedMapping::open(
                  name.c_str(), 64, MappingAccess::kCopyOnWrite)
                  .error()
                  .IsInvalidArgument());

  auto created = SharedMapping::create(name.c_str(), 64);
  ASSERT_TRUE(created.has_value()) << created.error().message();
  SharedMapping owner = std::move(*created);

  auto too_large = SharedMapping::open(name.c_str(), 65);
#if HYLE_OS_WINDOWS || HYLE_OS_MACOS
  // Win32 sections and Darwin POSIX shm objects report capacity rounded to a
  // native page boundary. The payload size is therefore an external protocol
  // fact, not exactly recoverable through the mapping API without a header.
  ASSERT_TRUE(too_large.has_value()) << too_large.error().message();
  EXPECT_TRUE(too_large->reset().ok());

  const std::size_t native_page = hyle::platform::page_size();
  ASSERT_GT(native_page, 64u);
  auto rounded_capacity = SharedMapping::open(name.c_str(), native_page);
  ASSERT_TRUE(rounded_capacity.has_value())
      << rounded_capacity.error().message();
  EXPECT_TRUE(rounded_capacity->reset().ok());
  auto past_rounded_capacity =
      SharedMapping::open(name.c_str(), native_page + 1);
  EXPECT_FALSE(past_rounded_capacity.has_value());
#else
  // Linux tmpfs shm preserves the exact ftruncate size.
  EXPECT_FALSE(too_large.has_value());
  EXPECT_TRUE(too_large.error().IsUnavailable());
#endif

  auto read_write =
      SharedMapping::open(name.c_str(), 64, MappingAccess::kReadWrite);
  ASSERT_TRUE(read_write.has_value()) << read_write.error().message();
  auto writable = read_write->writable_bytes();
  ASSERT_TRUE(writable.has_value()) << writable.error().message();
  (*writable)[7] = std::byte{0x4a};
  EXPECT_EQ(owner.bytes()[7], std::byte{0x4a});

  EXPECT_TRUE(owner.reset().ok());
  EXPECT_TRUE(read_write->reset().ok());
  const Status unlink = SharedMapping::unlink(name.c_str());
  EXPECT_TRUE(unlink.ok() || unlink.IsUnimplemented()) << unlink.message();
}

TEST(SharedMapping, InvalidNamesAreStructuredErrors) {
  EXPECT_TRUE(SharedMapping::create(nullptr, 64).error().IsInvalidArgument());
  EXPECT_TRUE(SharedMapping::create("", 64).error().IsInvalidArgument());
  EXPECT_TRUE(SharedMapping::open(nullptr, 64).error().IsInvalidArgument());
#if !HYLE_OS_WINDOWS
  EXPECT_TRUE(SharedMapping::create("nested/name", 64)
                  .error()
                  .IsInvalidArgument());
  const std::string too_long(512, 'x');
  EXPECT_TRUE(SharedMapping::create(too_long.c_str(), 64)
                  .error()
                  .IsInvalidArgument());
#endif
  SharedMapping empty;
  EXPECT_TRUE(empty.writable_data().error().IsFailedPrecondition());
  EXPECT_TRUE(empty.flush().IsFailedPrecondition());
  EXPECT_TRUE(empty.reset().ok());
}

}  // namespace

#if defined(_WIN32)
#  define _CRT_SECURE_NO_WARNINGS 1
#endif

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
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
  auto allocated = VirtualMapping::allocate_read_write(page + 1);
  ASSERT_TRUE(allocated.has_value()) << allocated.error().message();
  VirtualMapping mapping = std::move(*allocated);
  ASSERT_TRUE(mapping.valid());
  EXPECT_EQ(mapping.size(), page * 2);

  auto* bytes = static_cast<std::byte*>(mapping.data());
  bytes[0] = std::byte{0x2a};
  EXPECT_TRUE(mapping.advise(MappingAdvice::kNormal).ok());
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
  EXPECT_TRUE(mapping
                  .advise(static_cast<MappingAdvice>(255))
                  .IsInvalidArgument());
  ResidencyOptions options;
  options.mode = static_cast<ResidencyMode>(255);
  EXPECT_TRUE(mapping.lock(0, page, options).IsInvalidArgument());
}

TEST_F(FileMappingTest, ReadOnlyMappingProvidesConstBufferJourney) {
  constexpr char contents[] = "mapped model bytes";
  seed(contents, sizeof(contents));

  auto opened = FileMapping::open(path_.c_str());
  ASSERT_TRUE(opened.has_value()) << opened.error().message();
  FileMapping mapping = std::move(*opened);
  ASSERT_TRUE(mapping.valid());
  EXPECT_EQ(mapping.size(), sizeof(contents));
  EXPECT_TRUE(mapping.writable_bytes().error().IsFailedPrecondition());

  std::array<std::byte, sizeof(contents)> copied{};
  ASSERT_TRUE(mapping.view().read(copied).ok());
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
      "hyle_mem_shared_" +
      std::to_string(reinterpret_cast<std::uintptr_t>(&marker));
  SharedMapping::unlink(name.c_str()).IgnoreError();

  auto created = SharedMapping::create(name.c_str(), 128);
  ASSERT_TRUE(created.has_value()) << created.error().message();
  SharedMapping owner = std::move(*created);
  auto duplicate = SharedMapping::create(name.c_str(), 128);
  EXPECT_FALSE(duplicate.has_value());
  EXPECT_TRUE(duplicate.error().IsAlreadyExists());

  auto opened = SharedMapping::open(name.c_str());
  ASSERT_TRUE(opened.has_value()) << opened.error().message();
  SharedMapping peer = std::move(*opened);
  ASSERT_TRUE(owner.view().fill(std::byte{0x39}).ok());

  std::array<std::byte, 128> observed{};
  ASSERT_TRUE(static_cast<const SharedMapping&>(peer).view().read(observed).ok());
  for (std::byte value : observed) EXPECT_EQ(value, std::byte{0x39});
  EXPECT_TRUE(owner.flush().ok());

  Status unlink = SharedMapping::unlink(name.c_str());
  EXPECT_TRUE(unlink.ok() || unlink.IsUnimplemented()) << unlink.message();
}

}  // namespace

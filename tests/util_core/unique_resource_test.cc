#include "hyle/util/unique_resource.h"

#include <gtest/gtest.h>

#include <type_traits>
#include <utility>

namespace {

using hyle::util::UniqueResource;
using hyle::util::make_unique_resource_checked;

struct TokenDeleter {
  int* deleted = nullptr;

  void operator()(int token) const noexcept { *deleted += token; }
};

struct ThrowingCopyToken {
  int value;
  bool* throw_on_copy;

  ThrowingCopyToken(int input, bool* should_throw) noexcept
      : value(input), throw_on_copy(should_throw) {}
  ThrowingCopyToken(const ThrowingCopyToken& other)
      : value(other.value), throw_on_copy(other.throw_on_copy) {
    if (*throw_on_copy) throw 17;
  }
  ThrowingCopyToken(ThrowingCopyToken&& other) noexcept(false)
      : value(other.value), throw_on_copy(other.throw_on_copy) {}
  ThrowingCopyToken& operator=(ThrowingCopyToken&&) noexcept = default;

  friend bool operator==(const ThrowingCopyToken& token,
                         int invalid) noexcept {
    return token.value == invalid;
  }
};

struct ThrowingCopyTokenDeleter {
  int* deleted;
  void operator()(ThrowingCopyToken& token) const noexcept {
    *deleted += token.value;
  }
};

static_assert(!std::copy_constructible<UniqueResource<int, TokenDeleter>>);
static_assert(std::move_constructible<UniqueResource<int, TokenDeleter>>);

TEST(UniqueResource, DeletesValidTokenExactlyOnce) {
  int deleted = 0;
  {
    UniqueResource owner{7, TokenDeleter{&deleted}};
    EXPECT_TRUE(owner.allocated());
    EXPECT_TRUE(static_cast<bool>(owner));
    EXPECT_EQ(owner.get(), 7);
    EXPECT_EQ(owner.get_deleter().deleted, &deleted);
  }
  EXPECT_EQ(deleted, 7);
}

TEST(UniqueResource, CheckedFactoryDoesNotOwnInvalidSentinel) {
  int deleted = 0;
  {
    auto invalid = make_unique_resource_checked(-1, -1,
                                                TokenDeleter{&deleted});
    EXPECT_FALSE(invalid.allocated());
    EXPECT_EQ(invalid.get(), -1);
  }
  EXPECT_EQ(deleted, 0);

  {
    auto valid = make_unique_resource_checked(5, -1,
                                              TokenDeleter{&deleted});
    EXPECT_TRUE(valid.allocated());
  }
  EXPECT_EQ(deleted, 5);
}

TEST(UniqueResource, ReleaseAndResetHaveExplicitOwnership) {
  int deleted = 0;
  UniqueResource owner{3, TokenDeleter{&deleted}};
  const int transferred = owner.get();
  owner.release();
  EXPECT_FALSE(owner);
  EXPECT_EQ(transferred, 3);
  EXPECT_EQ(deleted, 0);

  owner.reset(4);
  EXPECT_TRUE(owner);
  owner.reset();
  EXPECT_FALSE(owner);
  EXPECT_EQ(deleted, 4);
  owner.reset();
  EXPECT_EQ(deleted, 4);
}

TEST(UniqueResource, MoveTransfersTheSingleCleanup) {
  int deleted = 0;
  {
    UniqueResource source{9, TokenDeleter{&deleted}};
    UniqueResource moved{std::move(source)};
    EXPECT_FALSE(source);  // NOLINT(bugprone-use-after-move)
    EXPECT_TRUE(moved);

    UniqueResource destination{2, TokenDeleter{&deleted}};
    destination = std::move(moved);
    EXPECT_EQ(deleted, 2);
    EXPECT_FALSE(moved);  // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(destination.get(), 9);
  }
  EXPECT_EQ(deleted, 11);
}

TEST(UniqueResource, FailedStorageConstructionReclaimsTheAcquisition) {
  int deleted = 0;
  bool throw_on_copy = true;
  ThrowingCopyToken token{13, &throw_on_copy};
  EXPECT_THROW(
      ([&] {
        UniqueResource<ThrowingCopyToken, ThrowingCopyTokenDeleter> owner{
            token, ThrowingCopyTokenDeleter{&deleted}};
        (void)owner;
      }()),
      int);
  EXPECT_EQ(deleted, 13);
}

TEST(UniqueResource, InvalidSentinelIsNotDeletedWhenStorageCopyThrows) {
  int deleted = 0;
  bool throw_on_copy = true;
  ThrowingCopyToken invalid{-1, &throw_on_copy};
  EXPECT_THROW(
      ([&] {
        auto owner = make_unique_resource_checked(
            invalid, -1, ThrowingCopyTokenDeleter{&deleted});
        (void)owner;
      }()),
      int);
  EXPECT_EQ(deleted, 0);
}

}  // namespace

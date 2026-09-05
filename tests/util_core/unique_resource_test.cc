#include "hyle/util/unique_resource.h"

#include <gtest/gtest.h>

#include <memory>
#include <type_traits>
#include <utility>

namespace {

using hyle::util::make_unique_resource_checked;
using hyle::util::UniqueResource;

struct TokenDeleter {
  int* deleted = nullptr;

  void operator()(int token) const noexcept { *deleted += token; }
};

struct MoveOnlyTokenDeleter {
  MoveOnlyTokenDeleter(int* deleted, int weight)
      : deleted(deleted), weight(std::make_unique<int>(weight)) {}
  MoveOnlyTokenDeleter(MoveOnlyTokenDeleter&&) noexcept = default;
  MoveOnlyTokenDeleter& operator=(MoveOnlyTokenDeleter&&) noexcept = default;
  MoveOnlyTokenDeleter(const MoveOnlyTokenDeleter&) = delete;
  MoveOnlyTokenDeleter& operator=(const MoveOnlyTokenDeleter&) = delete;

  void operator()(int token) const noexcept { *deleted += token * *weight; }

  int* deleted;
  std::unique_ptr<int> weight;
};

struct CopyFallbackDeleter {
  CopyFallbackDeleter(int* deleted, int* copies, int* moves) noexcept
      : deleted(deleted), copies(copies), moves(moves) {}
  CopyFallbackDeleter(const CopyFallbackDeleter& other) noexcept
      : deleted(other.deleted), copies(other.copies), moves(other.moves) {
    ++*copies;
  }
  CopyFallbackDeleter(CopyFallbackDeleter&& other) noexcept(false)
      : deleted(other.deleted), copies(other.copies), moves(other.moves) {
    ++*moves;
  }

  void operator()(int token) const noexcept { *deleted += token; }

  int* deleted;
  int* copies;
  int* moves;
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

  friend bool operator==(const ThrowingCopyToken& token, int invalid) noexcept {
    return token.value == invalid;
  }
};

struct ThrowingCopyTokenDeleter {
  int* deleted;
  void operator()(ThrowingCopyToken& token) const noexcept {
    *deleted += token.value;
  }
};

struct ThrowingBoolConversion final {
  operator bool() const noexcept(false) { throw 7; }
};

struct ThrowingBoolSentinelToken final {
  ThrowingBoolConversion operator==(int) const noexcept { return {}; }
};

struct ThrowingBoolSentinelDeleter final {
  void operator()(ThrowingBoolSentinelToken&) const noexcept {}
};

template <class Resource, class Invalid, class Deleter>
concept HasCheckedUniqueResourceFactory =
    requires(Resource&& resource, const Invalid& invalid, Deleter&& deleter) {
      make_unique_resource_checked(std::forward<Resource>(resource), invalid,
                                   std::forward<Deleter>(deleter));
    };

struct MoveOnlyThrowingCopyTokenDeleter {
  explicit MoveOnlyThrowingCopyTokenDeleter(int* deleted)
      : deleted(deleted), state(std::make_unique<int>(1)) {}
  MoveOnlyThrowingCopyTokenDeleter(
      MoveOnlyThrowingCopyTokenDeleter&&) noexcept = default;
  MoveOnlyThrowingCopyTokenDeleter& operator=(
      MoveOnlyThrowingCopyTokenDeleter&&) noexcept = default;
  MoveOnlyThrowingCopyTokenDeleter(const MoveOnlyThrowingCopyTokenDeleter&) =
      delete;
  MoveOnlyThrowingCopyTokenDeleter& operator=(
      const MoveOnlyThrowingCopyTokenDeleter&) = delete;

  void operator()(ThrowingCopyToken& token) const noexcept {
    *deleted += token.value * *state;
  }

  int* deleted;
  std::unique_ptr<int> state;
};

static_assert(!std::copy_constructible<UniqueResource<int, TokenDeleter>>);
static_assert(std::move_constructible<UniqueResource<int, TokenDeleter>>);
static_assert(
    std::move_constructible<UniqueResource<int, MoveOnlyTokenDeleter>>);
static_assert(noexcept(UniqueResource<int, MoveOnlyTokenDeleter>(
    std::declval<int&&>(), std::declval<MoveOnlyTokenDeleter&&>())));
static_assert(noexcept(make_unique_resource_checked(
    std::declval<int&&>(), std::declval<const int&>(),
    std::declval<MoveOnlyTokenDeleter&&>())));
static_assert(!noexcept(make_unique_resource_checked(
    std::declval<ThrowingCopyToken&>(), std::declval<const int&>(),
    std::declval<ThrowingCopyTokenDeleter&&>())));
static_assert(!HasCheckedUniqueResourceFactory<
              ThrowingBoolSentinelToken, int,
              ThrowingBoolSentinelDeleter>);

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
    auto invalid = make_unique_resource_checked(-1, -1, TokenDeleter{&deleted});
    EXPECT_FALSE(invalid.allocated());
    EXPECT_EQ(invalid.get(), -1);
  }
  EXPECT_EQ(deleted, 0);

  {
    auto valid = make_unique_resource_checked(5, -1, TokenDeleter{&deleted});
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

TEST(UniqueResource, SupportsMoveOnlyNothrowDeleter) {
  int deleted = 0;
  {
    UniqueResource source{9, MoveOnlyTokenDeleter{&deleted, 2}};
    UniqueResource moved{std::move(source)};
    EXPECT_FALSE(source);  // NOLINT(bugprone-use-after-move)
    EXPECT_TRUE(moved);
    EXPECT_EQ(moved.get(), 9);
  }
  EXPECT_EQ(deleted, 18);

  auto invalid =
      make_unique_resource_checked(-1, -1, MoveOnlyTokenDeleter{&deleted, 3});
  EXPECT_FALSE(invalid);
  EXPECT_EQ(deleted, 18);
}

TEST(UniqueResource, CopiesDeleterWhenItsMoveCanThrow) {
  int deleted = 0;
  int copies = 0;
  int moves = 0;
  {
    UniqueResource source{5, CopyFallbackDeleter{&deleted, &copies, &moves}};
    UniqueResource moved{std::move(source)};
    EXPECT_FALSE(source);  // NOLINT(bugprone-use-after-move)
    EXPECT_TRUE(moved);
  }
  EXPECT_EQ(deleted, 5);
  EXPECT_EQ(copies, 2);
  EXPECT_EQ(moves, 0);
}

TEST(UniqueResource, FailedOwnerMoveWithMovedDeleterReclaimsSource) {
  int deleted = 0;
  bool throw_on_copy = false;
  ThrowingCopyToken token{17, &throw_on_copy};
  UniqueResource<ThrowingCopyToken, MoveOnlyThrowingCopyTokenDeleter> source{
      token, MoveOnlyThrowingCopyTokenDeleter{&deleted}};

  throw_on_copy = true;
  EXPECT_THROW(
      ([&] {
        UniqueResource<ThrowingCopyToken, MoveOnlyThrowingCopyTokenDeleter>
            moved{std::move(source)};
        (void)moved;
      }()),
      int);
  EXPECT_FALSE(source);  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(deleted, 17);
}

TEST(UniqueResource, FailedOwnerMoveWithCopiedDeleterLeavesSourceOwning) {
  int deleted = 0;
  bool throw_on_copy = false;
  ThrowingCopyToken token{19, &throw_on_copy};
  {
    UniqueResource<ThrowingCopyToken, ThrowingCopyTokenDeleter> source{
        token, ThrowingCopyTokenDeleter{&deleted}};

    throw_on_copy = true;
    EXPECT_THROW(
        ([&] {
          UniqueResource<ThrowingCopyToken, ThrowingCopyTokenDeleter> moved{
              std::move(source)};
          (void)moved;
        }()),
        int);
    EXPECT_TRUE(source);  // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(deleted, 0);
  }
  EXPECT_EQ(deleted, 19);
}

TEST(UniqueResource, FailedAcquisitionWithMovedDeleterReclaimsToken) {
  int deleted = 0;
  bool throw_on_copy = true;
  ThrowingCopyToken token{23, &throw_on_copy};
  EXPECT_THROW(
      ([&] {
        UniqueResource<ThrowingCopyToken, MoveOnlyThrowingCopyTokenDeleter>
            owner{token, MoveOnlyThrowingCopyTokenDeleter{&deleted}};
        (void)owner;
      }()),
      int);
  EXPECT_EQ(deleted, 23);
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
  EXPECT_THROW(([&] {
                 auto owner = make_unique_resource_checked(
                     invalid, -1, ThrowingCopyTokenDeleter{&deleted});
                 (void)owner;
               }()),
               int);
  EXPECT_EQ(deleted, 0);
}

}  // namespace

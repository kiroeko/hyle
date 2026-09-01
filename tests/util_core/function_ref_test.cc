#include "hyle/util/function_ref.h"

#include <gtest/gtest.h>

#include <concepts>
#include <cstddef>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace {

using hyle::util::FunctionRef;

int increment(int value) { return value + 1; }
int increment_noexcept(int value) noexcept { return value + 1; }
short narrow_increment(short value) { return static_cast<short>(value + 1); }

int invoke_twice(FunctionRef<int(int)> function, int value) {
  return function(function(value));
}

int invoke_twice_noexcept(FunctionRef<int(int) noexcept> function,
                          int value) noexcept {
  return function(function(value));
}

struct ThrowingCallable {
  int operator()(int value) const { return value; }
};

struct NoexceptCallable {
  int operator()(int value) const noexcept { return value; }
};

struct MemberCallable {
  int add(int input) const { return value + input; }
  int value;
};

static_assert(sizeof(FunctionRef<int(int)>) == 2 * sizeof(void*));
static_assert(std::is_trivially_copyable_v<FunctionRef<int(int)>>);
static_assert(!std::default_initializable<FunctionRef<int(int)>>);
static_assert(!std::constructible_from<FunctionRef<int(int)>, std::nullptr_t>);
static_assert(
    std::constructible_from<FunctionRef<int(int) noexcept>, NoexceptCallable&>);
static_assert(!std::constructible_from<FunctionRef<int(int) noexcept>,
                                       ThrowingCallable&>);
static_assert(noexcept(std::declval<FunctionRef<int(int) noexcept>>()(1)));

TEST(FunctionRef, InvokesFunctionPointerWithoutPointerVariableIndirection) {
  FunctionRef function{&increment};
  EXPECT_EQ(function(4), 5);
  EXPECT_EQ(invoke_twice(&increment, 4), 6);

  FunctionRef<int(int)> compatible = &narrow_increment;
  EXPECT_EQ(compatible(6), 7);
}

TEST(FunctionRef, BorrowsConstMutableAndMoveOnlyCallables) {
  const auto constant = [offset = 3](int value) { return value + offset; };
  EXPECT_EQ(invoke_twice(constant, 1), 7);

  int calls = 0;
  auto mutable_callable = [&calls](int value) mutable {
    ++calls;
    return value + 1;
  };
  EXPECT_EQ(invoke_twice(mutable_callable, 5), 7);
  EXPECT_EQ(calls, 2);

  auto move_only = [offset = std::make_unique<int>(4)](int value) {
    return value + *offset;
  };
  EXPECT_EQ(invoke_twice(move_only, 1), 9);
}

TEST(FunctionRef, TemporaryCallableLivesThroughSynchronousParameterCall) {
  EXPECT_EQ(invoke_twice([](int value) { return value * 2; }, 3), 12);
}

TEST(FunctionRef, SupportsVoidAndReferenceSignatures) {
  int observed = 0;
  auto returns_value = [&observed](int value) {
    observed = value;
    return value + 1;
  };
  FunctionRef<void(int)> consume = returns_value;
  consume(8);
  EXPECT_EQ(observed, 8);

  int value = 11;
  auto identity = [](int& input) -> int& { return input; };
  FunctionRef<int&(int&)> borrow = identity;
  EXPECT_EQ(&borrow(value), &value);
}

TEST(FunctionRef, UsesInvokeSemanticsForMemberPointers) {
  MemberCallable callable{7};
  const auto member_pointer = &MemberCallable::add;
  FunctionRef<int(const MemberCallable&, int)> member = member_pointer;
  EXPECT_EQ(member(callable, 4), 11);
}

TEST(FunctionRef, PreservesNoexceptSignature) {
  EXPECT_EQ(invoke_twice_noexcept(&increment_noexcept, 4), 6);
  NoexceptCallable callable;
  EXPECT_EQ(invoke_twice_noexcept(callable, 9), 9);
}

TEST(FunctionRefDeathTest, RejectsDynamicNullFunctionPointer) {
  EXPECT_DEATH(([] {
                 int (*function)(int) = nullptr;
                 FunctionRef<int(int)> reference(function);
                 (void)reference;
               }()),
               "");
}

TEST(FunctionRefDeathTest, RejectsEmptyOwningCallable) {
  EXPECT_DEATH(([] {
                 std::function<int(int)> empty;
                 FunctionRef<int(int)> reference(empty);
                 (void)reference;
               }()),
               "");
}

TEST(FunctionRefDeathTest, RejectsNullMemberPointer) {
  EXPECT_DEATH(([] {
                 struct Owner {
                   int invoke(int value) { return value; }
                 };
                 int (Owner::*member)(int) = nullptr;
                 FunctionRef<int(Owner&, int)> reference(member);
                 (void)reference;
               }()),
               "");
}

}  // namespace

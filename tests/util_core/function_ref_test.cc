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
using hyle::util::nontype;

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

struct MutableOnlyCallable {
  int operator()(int input) { return ++calls + input; }
  int calls = 0;
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
static_assert(sizeof(FunctionRef<int(int) const>) == 2 * sizeof(void*));
static_assert(sizeof(FunctionRef<int(int) const noexcept>) == 2 * sizeof(void*));
static_assert(std::is_trivially_copyable_v<FunctionRef<int(int) const>>);
static_assert(std::constructible_from<FunctionRef<int(int) const>,
                                      const NoexceptCallable&>);
static_assert(!std::constructible_from<FunctionRef<int(int) const>,
                                       MutableOnlyCallable&>);
static_assert(std::constructible_from<FunctionRef<int(int) const noexcept>,
                                      const NoexceptCallable&>);
static_assert(!std::constructible_from<FunctionRef<int(int) const noexcept>,
                                       const ThrowingCallable&>);
static_assert(
    noexcept(std::declval<FunctionRef<int(int) const noexcept>>()(1)));

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

TEST(FunctionRef, ConstSignatureRejectsMutationAndInvokesConstTarget) {
  const NoexceptCallable callable;
  FunctionRef<int(int) const> throwing_contract = callable;
  FunctionRef<int(int) const noexcept> no_throw_contract = callable;
  EXPECT_EQ(throwing_contract(7), 7);
  EXPECT_EQ(no_throw_contract(9), 9);

  MutableOnlyCallable mutable_only;
  FunctionRef<int(int)> mutable_contract = mutable_only;
  EXPECT_EQ(mutable_contract(4), 5);
  EXPECT_EQ(mutable_only.calls, 1);
}

TEST(FunctionRef, NontypeBindingAvoidsStoredRuntimeCallablePointer) {
  FunctionRef direct{nontype<&increment>};
  EXPECT_EQ(direct(4), 5);

  MemberCallable owner{7};
  FunctionRef<int(const MemberCallable&, int)> unbound{
      nontype<&MemberCallable::add>};
  EXPECT_EQ(unbound(owner, 5), 12);

  FunctionRef<int(int) const> bound{nontype<&MemberCallable::add>, owner};
  EXPECT_EQ(bound(6), 13);

  const MemberCallable* owner_pointer = &owner;
  FunctionRef<int(int) const> bound_pointer{nontype<&MemberCallable::add>,
                                             owner_pointer};
  EXPECT_EQ(bound_pointer(8), 15);
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

TEST(FunctionRefDeathTest, RejectsNullNontypeBoundObject) {
  EXPECT_DEATH(([] {
                 const MemberCallable* owner = nullptr;
                 FunctionRef<int(int) const> reference{
                     nontype<&MemberCallable::add>, owner};
                 (void)reference;
               }()),
               "");
}

}  // namespace

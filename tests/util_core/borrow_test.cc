#include "hyle/util/borrow.h"

#include <gtest/gtest.h>

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <unordered_set>

namespace {

using hyle::util::NotNull;

struct Base {
  int value = 0;
};

struct Derived : Base {};

int plus_one(int value) { return value + 1; }

constexpr int kConstexprValue = 5;
constexpr NotNull<const int*> kConstexprPointer = &kConstexprValue;

static_assert(sizeof(NotNull<int*>) == sizeof(int*));
static_assert(std::is_trivially_copyable_v<NotNull<int*>>);
static_assert(std::is_standard_layout_v<NotNull<int*>>);
static_assert(!std::default_initializable<NotNull<int*>>);
static_assert(!std::constructible_from<NotNull<int*>, std::nullptr_t>);
static_assert(std::constructible_from<NotNull<Base*>, Derived*>);
static_assert(!std::constructible_from<NotNull<Derived*>, Base*>);
static_assert(std::constructible_from<NotNull<const void*>, const int*>);
static_assert(*kConstexprPointer == 5);

TEST(NotNull, ProvidesRequiredPointerOperations) {
  Base value{7};
  NotNull<Base*> pointer = &value;
  EXPECT_EQ(pointer.get(), &value);
  EXPECT_EQ((*pointer).value, 7);
  pointer->value = 9;
  EXPECT_EQ(value.value, 9);

  Base* raw = pointer;
  EXPECT_EQ(raw, &value);
  EXPECT_EQ(pointer, raw);
}

TEST(NotNull, ConvertsConstAndBasePointersWithoutRechecking) {
  Derived value;
  value.value = 3;
  NotNull<Derived*> derived = &value;
  NotNull<Base*> base = derived;
  NotNull<const Base*> read_only = base;
  EXPECT_EQ(read_only->value, 3);
  EXPECT_EQ(base, derived);
}

TEST(NotNull, HashesPointerIdentity) {
  int first = 1;
  int second = 2;
  NotNull<int*> first_pointer = &first;
  std::unordered_set<NotNull<int*>> pointers;
  pointers.insert(first_pointer);
  pointers.insert(NotNull<int*>{&first});
  pointers.insert(NotNull<int*>{&second});
  EXPECT_EQ(pointers.size(), 2u);
  EXPECT_TRUE(pointers.contains(first_pointer));
}

TEST(NotNull, SupportsFunctionPointers) {
  NotNull<int (*)(int)> function = &plus_one;
  EXPECT_EQ((*function)(8), 9);
}

TEST(NotNull, SupportsOpaqueRequiredContext) {
  const int value = 4;
  NotNull<const void*> context = &value;
  EXPECT_EQ(context.get(), static_cast<const void*>(&value));
}

TEST(NotNullDeathTest, RejectsDynamicNull) {
  EXPECT_DEATH(([] {
                 int* pointer = nullptr;
                 NotNull<int*> required(pointer);
                 (void)required;
               }()),
               "");
}

}  // namespace

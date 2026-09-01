#include <gtest/gtest.h>

#include <atomic>
#include <limits>
#include <set>
#include <unordered_set>
#include <thread>
#include <type_traits>
#include <vector>

#include "hyle/util/ref_counted.h"

using hyle::util::adopt_ref;
using hyle::util::make_ref;
using hyle::util::Ref;
using hyle::util::RefCompatible;
using hyle::util::RefCounted;
using hyle::util::ThreadUnsafeRefCounted;
using hyle::util::retain_ref;
using hyle::util::const_pointer_cast;
using hyle::util::dynamic_pointer_cast;
using hyle::util::static_pointer_cast;

namespace {

// Tracks live instances so tests can assert destruction happened.
struct Widget : RefCounted<Widget> {
  static inline std::atomic<int> live{0};
  int value;
  explicit Widget(int v) : value(v) { live.fetch_add(1); }
  ~Widget() { live.fetch_sub(1); }
};

// Subclass with a CUSTOM destroy() -- the clay-bug regression guard. clay's
// unqualified destroy() in decref would call the BASE version; HYLE static_casts
// to the subclass, so this custom path must actually run.
struct CustomTeardown : RefCounted<CustomTeardown> {
  static inline std::atomic<int> custom_destroys{0};
  // destroy() must be reachable by RefCounted<CustomTeardown>.
  void destroy() {
    custom_destroys.fetch_add(1);
    delete this;
  }
};

struct ThreadConfinedWidget : ThreadUnsafeRefCounted<ThreadConfinedWidget> {
  static inline int live = 0;
  explicit ThreadConfinedWidget(int initial) : value(initial) { ++live; }
  ~ThreadConfinedWidget() { --live; }
  int value;
};

}  // namespace

TEST(RefCounted, MakeRefStartsAtOneAndDestroys) {
  Widget::live = 0;
  {
    Ref<Widget> r = make_ref<Widget>(42);
    EXPECT_EQ(r->value, 42);
    EXPECT_EQ(r->use_count(), 1u);
    EXPECT_TRUE(r->is_unique());
    EXPECT_EQ(Widget::live.load(), 1);
  }
  EXPECT_EQ(Widget::live.load(), 0);  // destroyed at scope exit
}

TEST(RefCounted, ThreadUnsafePolicyKeepsTheSameOwnerSurface) {
  ThreadConfinedWidget::live = 0;
  {
    Ref<ThreadConfinedWidget> first = make_ref<ThreadConfinedWidget>(12);
    EXPECT_EQ(first->use_count(), 1u);
    Ref<ThreadConfinedWidget> second = first;
    EXPECT_EQ(first->use_count(), 2u);
    second.reset();
    EXPECT_TRUE(first->is_unique());
    first->incref(3);
    EXPECT_EQ(first->use_count(), 4u);
    first->decref(3);
    EXPECT_EQ(first->use_count(), 1u);
  }
  EXPECT_EQ(ThreadConfinedWidget::live, 0);
}

TEST(RefCounted, CopyShares) {
  Widget::live = 0;
  Ref<Widget> a = make_ref<Widget>(1);
  {
    Ref<Widget> b = a;  // copy -> incref
    EXPECT_EQ(a->use_count(), 2u);
    EXPECT_EQ(b.get(), a.get());
    EXPECT_FALSE(a->is_unique());
  }
  EXPECT_EQ(a->use_count(), 1u);  // b dropped
  EXPECT_EQ(Widget::live.load(), 1);
}

TEST(RefCounted, MoveStealsNoIncref) {
  Widget::live = 0;
  Ref<Widget> a = make_ref<Widget>(7);
  Widget* raw = a.get();
  Ref<Widget> b = std::move(a);
  EXPECT_EQ(a.get(), nullptr);       // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(b.get(), raw);
  EXPECT_EQ(b->use_count(), 1u);     // move did not bump the count
  EXPECT_EQ(Widget::live.load(), 1);
}

TEST(RefCounted, CustomDestroyDispatchesViaCRTP) {
  CustomTeardown::custom_destroys = 0;
  {
    Ref<CustomTeardown> r = make_ref<CustomTeardown>();
    EXPECT_EQ(CustomTeardown::custom_destroys.load(), 0);
  }
  // If decref called the base destroy (clay's bug), this would be 0.
  EXPECT_EQ(CustomTeardown::custom_destroys.load(), 1);
}

TEST(RefCounted, ResetAndReleaseAndAssign) {
  Widget::live = 0;
  Ref<Widget> a = make_ref<Widget>(1);
  a.reset();
  EXPECT_FALSE(a);
  EXPECT_EQ(Widget::live.load(), 0);

  Ref<Widget> b = make_ref<Widget>(2);
  Widget* raw = b.release();          // gives us the +1
  EXPECT_FALSE(b);
  EXPECT_EQ(Widget::live.load(), 1);
  Ref<Widget> c = adopt_ref(raw);     // take it back, no extra incref
  EXPECT_EQ(c->use_count(), 1u);

  Ref<Widget> d = make_ref<Widget>(3);
  d = c;                              // copy-assign: decref old, share c
  EXPECT_EQ(c->use_count(), 2u);
  EXPECT_EQ(Widget::live.load(), 1);  // d's old Widget(3) was destroyed
}

TEST(RefCounted, RetainRefAddsReference) {
  Widget::live = 0;
  Ref<Widget> a = make_ref<Widget>(5);
  {
    Ref<Widget> b = retain_ref(a.get());  // borrow -> +1
    EXPECT_EQ(a->use_count(), 2u);
  }
  EXPECT_EQ(a->use_count(), 1u);
}

TEST(RefCounted, NullCompare) {
  Ref<Widget> a;
  EXPECT_EQ(a, nullptr);
  EXPECT_FALSE(a);
  a = make_ref<Widget>(1);
  EXPECT_NE(a, nullptr);
  EXPECT_TRUE(a);
}

// Upcast Ref<Derived> -> Ref<Base>.
namespace {
struct Base : RefCounted<Base> {
  virtual ~Base() = default;
  virtual int kind() const { return 0; }
};
struct Derived : Base {
  int kind() const override { return 1; }
};
struct NonVirtualBase : RefCounted<NonVirtualBase> {};
struct UnsafeDerived : NonVirtualBase {};
static_assert(!RefCompatible<UnsafeDerived>);
static_assert(RefCompatible<Derived>);
static_assert(RefCompatible<const Widget>);
}  // namespace

TEST(RefCounted, UpcastDerivedToBase) {
  Ref<Derived> d = make_ref<Derived>();
  Ref<Base> b = d;             // copy upcast -> incref
  EXPECT_EQ(d->use_count(), 2u);
  EXPECT_EQ(b->kind(), 1);     // virtual dispatch through base ref
  Ref<Base> b2 = std::move(d); // move upcast -> steal
  EXPECT_EQ(b2->use_count(), 2u);
  EXPECT_EQ(d.get(), nullptr); // NOLINT(bugprone-use-after-move)
}

TEST(RefCounted, ConvertingAssignmentAndConstOwnership) {
  Ref<Derived> derived = make_ref<Derived>();
  Ref<Base> base;
  base = derived;
  EXPECT_EQ(base->kind(), 1);
  EXPECT_EQ(base->use_count(), 2u);

  Ref<const Base> read_only;
  read_only = base;
  EXPECT_EQ(read_only->kind(), 1);
  EXPECT_EQ(read_only->use_count(), 3u);

  Ref<const Base> moved;
  moved = std::move(derived);
  EXPECT_FALSE(derived);  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(moved->kind(), 1);
  EXPECT_EQ(moved->use_count(), 3u);
}

TEST(RefCounted, HeterogeneousIdentityComparisonDoesNotRetain) {
  Ref<Derived> derived = make_ref<Derived>();
  Ref<Base> base = derived;
  const auto count = base->use_count();
  EXPECT_EQ(base, derived);
  EXPECT_EQ(derived, base);
  EXPECT_EQ(base <=> derived,
            std::compare_three_way{}(base.get(), derived.get()));
  EXPECT_EQ(base->use_count(), count);
}

TEST(RefCounted, PointerCastsPreserveOwnership) {
  Ref<Base> base = make_ref<Derived>();
  EXPECT_EQ(base->use_count(), 1u);

  Ref<Derived> copied = dynamic_pointer_cast<Derived>(base);
  ASSERT_TRUE(copied);
  EXPECT_EQ(copied->kind(), 1);
  EXPECT_EQ(base->use_count(), 2u);

  Ref<const Base> read_only = base;
  Ref<Base> mutable_again = const_pointer_cast<Base>(read_only);
  EXPECT_EQ(mutable_again.get(), base.get());

  Ref<Base> moved_source = std::move(base);
  const auto before_move = moved_source->use_count();
  Ref<Derived> moved = static_pointer_cast<Derived>(std::move(moved_source));
  EXPECT_FALSE(moved_source);  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(moved->use_count(), before_move);

  Ref<Base> plain = make_ref<Base>();
  Base* plain_pointer = plain.get();
  Ref<Derived> failed = dynamic_pointer_cast<Derived>(std::move(plain));
  EXPECT_FALSE(failed);
  EXPECT_EQ(plain.get(), plain_pointer);  // failed rvalue cast does not consume
}

TEST(RefCounted, HashUsesPointerIdentity) {
  Ref<Widget> widget = make_ref<Widget>(4);
  std::unordered_set<Ref<Widget>> owners;
  owners.insert(widget);
  EXPECT_TRUE(owners.contains(widget));
  EXPECT_TRUE(owners.contains(retain_ref(widget.get())));
  EXPECT_EQ(owners.size(), 1u);
}

TEST(RefCounted, OrderedContainersUsePointerIdentity) {
  Ref<Widget> first = make_ref<Widget>(1);
  Ref<Widget> second = make_ref<Widget>(2);
  std::set<Ref<Widget>> owners;
  owners.insert(first);
  owners.insert(second);
  owners.insert(retain_ref(first.get()));
  EXPECT_EQ(owners.size(), 2u);
  EXPECT_TRUE((first <=> nullptr) == std::strong_ordering::greater);
  EXPECT_EQ(first <=> second,
            std::compare_three_way{}(first.get(), second.get()));
}

// Concurrent incref/decref must not corrupt the count (TSan gate on WSL). Many
// threads each take and drop a reference repeatedly; the object stays alive
// because the test holds one ref throughout, and use_count returns to 1.
TEST(RefCounted, ConcurrentIncrefDecref) {
  Widget::live = 0;
  Ref<Widget> keep = make_ref<Widget>(0);
  Widget* w = keep.get();
  constexpr int kThreads = 8;
  constexpr int kIters = 50000;
  std::vector<std::thread> ts;
  for (int t = 0; t < kThreads; ++t) {
    ts.emplace_back([w] {
      for (int i = 0; i < kIters; ++i) {
        w->incref();
        w->decref();
      }
    });
  }
  for (auto& th : ts) th.join();
  EXPECT_EQ(w->use_count(), 1u);   // all transient refs balanced out
  EXPECT_EQ(Widget::live.load(), 1);
}

TEST(RefCounted, BatchedReferenceChanges) {
  Widget::live = 0;
  Widget* widget = new Widget(9);
  widget->incref(7);
  EXPECT_EQ(widget->use_count(), 8u);
  widget->decref(7);
  EXPECT_EQ(widget->use_count(), 1u);
  widget->decref();
  EXPECT_EQ(Widget::live.load(), 0);
}

TEST(RefCountedDeathTest, RejectsZeroOverflowAndUnderflow) {
  EXPECT_DEATH(
      ([] {
        auto* widget = new Widget(1);
        widget->incref(0);
      }()),
      "");
  EXPECT_DEATH(
      ([] {
        auto* widget = new Widget(1);
        widget->decref(0);
      }()),
      "");
  EXPECT_DEATH(
      ([] {
        auto* widget = new Widget(1);
        widget->incref(std::numeric_limits<std::uint32_t>::max() - 1);
        widget->incref(1);
      }()),
      "");
  EXPECT_DEATH(
      ([] {
        auto* widget = new Widget(1);
        widget->decref(2);
      }()),
      "");
}

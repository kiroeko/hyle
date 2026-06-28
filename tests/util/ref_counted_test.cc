#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

#include "hyle/util/ref_counted.h"

using hyle::util::adopt_ref;
using hyle::util::make_ref;
using hyle::util::Ref;
using hyle::util::RefCounted;
using hyle::util::retain_ref;

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

#include <gtest/gtest.h>

#include <memory>

#include "hyle/container/string_ref.h"
#include "hyle/util/registry.h"

using hyle::container::string_ref;
using hyle::util::RegistryPriority;

namespace {

// A plugin hierarchy to register by name.
struct Shape {
  virtual ~Shape() = default;
  virtual int sides() const = 0;
};
struct Triangle : Shape {
  int sides() const override { return 3; }
};
struct Square : Shape {
  int sides() const override { return 4; }
};

// A fallback and a preferred implementation of the same key (priority test).
struct CheapWidget {
  virtual ~CheapWidget() = default;
  virtual int quality() const { return 1; }
};
struct FancyWidget : CheapWidget {
  int quality() const override { return 10; }
};

}  // namespace

// Declare + define a shape registry (normally DECLARE in a header, DEFINE in one
// .cc; here both in the test TU). ExportAPI empty for a single-TU test.
HYLE_DECLARE_REGISTRY(, ShapeRegistry, Shape*);
HYLE_DEFINE_REGISTRY(ShapeRegistry, Shape*);

// Register two classes by name via the macro (static-init self-registration).
HYLE_REGISTER_CLASS(ShapeRegistry, Triangle, Triangle);
HYLE_REGISTER_CLASS(ShapeRegistry, Square, Square);

TEST(Registry, CreateByName) {
  std::unique_ptr<Shape> t(ShapeRegistry()->create("Triangle"));
  ASSERT_NE(t, nullptr);
  EXPECT_EQ(t->sides(), 3);
  std::unique_ptr<Shape> s(ShapeRegistry()->create("Square"));
  ASSERT_NE(s, nullptr);
  EXPECT_EQ(s->sides(), 4);
}

TEST(Registry, MissingKeyReturnsNull) {
  EXPECT_EQ(ShapeRegistry()->create("Hexagon"), nullptr);
  EXPECT_FALSE(ShapeRegistry()->has("Hexagon"));
  EXPECT_TRUE(ShapeRegistry()->has("Triangle"));
}

TEST(Registry, HelpTextIsTypeName) {
  // REGISTER_CLASS stores type_name<Derived>() as help.
  string_ref h = ShapeRegistry()->help("Triangle");
  EXPECT_NE(h.find("Triangle"), string_ref::npos);
}

TEST(Registry, RegisteredKeysVisible) {
  int count = 0;
  bool saw_triangle = false;
  ShapeRegistry()->for_each_key([&](string_ref k) {
    ++count;
    if (k == string_ref{"Triangle"}) saw_triangle = true;
  });
  EXPECT_GE(count, 2);
  EXPECT_TRUE(saw_triangle);
}

// Priority: a kPreferred creator overrides a kFallback one for the same key.
TEST(Registry, PriorityOverride) {
  hyle::util::Registry<CheapWidget*> reg;
  reg.register_creator(
      "widget", [](void) -> CheapWidget* { return new CheapWidget(); },
      RegistryPriority::kFallback);
  EXPECT_EQ(reg.size(), 1u);
  {
    std::unique_ptr<CheapWidget> w(reg.create("widget"));
    EXPECT_EQ(w->quality(), 1);  // fallback active
  }
  // Higher priority overrides.
  reg.register_creator(
      "widget", [](void) -> CheapWidget* { return new FancyWidget(); },
      RegistryPriority::kPreferred);
  EXPECT_EQ(reg.size(), 1u);  // same key, replaced not added
  {
    std::unique_ptr<CheapWidget> w(reg.create("widget"));
    EXPECT_EQ(w->quality(), 10);  // preferred won
  }
  // A later fallback does NOT displace the preferred one.
  reg.register_creator(
      "widget", [](void) -> CheapWidget* { return new CheapWidget(); },
      RegistryPriority::kFallback);
  {
    std::unique_ptr<CheapWidget> w(reg.create("widget"));
    EXPECT_EQ(w->quality(), 10);  // still preferred
  }
}

// Same-priority double registration is a wiring bug -> fail-fast (abort).
TEST(RegistryDeathTest, SamePriorityCollisionAborts) {
  auto fn = [] {
    hyle::util::Registry<CheapWidget*> reg;
    reg.register_creator("dup", [](void) -> CheapWidget* { return nullptr; },
                         RegistryPriority::kDefault);
    reg.register_creator("dup", [](void) -> CheapWidget* { return nullptr; },
                         RegistryPriority::kDefault);  // same priority -> abort
  };
  EXPECT_DEATH(fn(), "");
}

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "hyle/container/string_ref.h"
#include "hyle/util/ref_counted.h"
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

struct RefBase : hyle::util::RefCounted<RefBase> {
  virtual ~RefBase() = default;
  virtual int value() const { return 1; }
};
struct RefDerived : RefBase {
  int value() const override { return 7; }
};

struct MoveWidget : hyle::util::RefCounted<MoveWidget> {
  explicit MoveWidget(std::unique_ptr<int> input) : value(*input) {}
  int value;
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

TEST(Registry, DefaultCreatorSupportsIntrusiveRef) {
  using Ptr = hyle::util::Ref<RefBase>;
  using Reg = hyle::util::Registry<Ptr>;
  using Registerer = hyle::util::Registerer<Ptr>;
  Reg registry;
  registry.register_creator(
      "derived", &Registerer::default_creator<RefDerived>);
  Ptr value = registry.create("derived");
  ASSERT_TRUE(value);
  EXPECT_EQ(value->value(), 7);
}

TEST(Registry, MoveOnlyFactoryArgumentsAreForwarded) {
  using Ptr = hyle::util::Ref<MoveWidget>;
  using Reg = hyle::util::Registry<Ptr, std::unique_ptr<int>>;
  using Registerer =
      hyle::util::Registerer<Ptr, std::unique_ptr<int>>;
  Reg registry;
  registry.register_creator(
      "move", &Registerer::default_creator<MoveWidget>);
  Ptr value =
      registry.create("move", std::make_unique<int>(42));
  ASSERT_TRUE(value);
  EXPECT_EQ(value->value, 42);
}

namespace {

using ReentrantRegistry = hyle::util::Registry<int*>;
ReentrantRegistry* g_reentrant_registry = nullptr;

int* create_reentrantly() {
  EXPECT_TRUE(g_reentrant_registry->has("reentrant"));
  EXPECT_EQ(g_reentrant_registry->size(), 1u);
  static int value = 19;
  return &value;
}

}  // namespace

TEST(Registry, CreatorMayReenterRegistry) {
  ReentrantRegistry registry;
  g_reentrant_registry = &registry;
  registry.register_creator("reentrant", &create_reentrantly);
  EXPECT_EQ(registry.create("reentrant"), create_reentrantly());
  g_reentrant_registry = nullptr;
}

TEST(Registry, ConcurrentRegistrationAndReadsAreSafe) {
  constexpr std::size_t kKeys = 128;
  hyle::util::Registry<CheapWidget*> registry;
  std::array<std::array<char, 16>, kKeys> key_storage{};
  for (std::size_t i = 0; i < kKeys; ++i) {
    std::snprintf(key_storage[i].data(), key_storage[i].size(), "key-%zu", i);
  }
  auto creator = +[]() -> CheapWidget* { return new CheapWidget(); };
  std::atomic<bool> start{false};
  std::atomic<bool> finished{false};
  std::thread reader([&] {
    while (!start.load(std::memory_order_acquire)) {}
    while (!finished.load(std::memory_order_acquire)) {
      (void)registry.has("missing");
      (void)registry.size();
    }
  });
  std::vector<std::thread> writers;
  for (std::size_t worker = 0; worker < 4; ++worker) {
    writers.emplace_back([&, worker] {
      while (!start.load(std::memory_order_acquire)) {}
      for (std::size_t i = worker; i < kKeys; i += 4) {
        const auto length = static_cast<std::uint32_t>(
            std::strlen(key_storage[i].data()));
        registry.register_creator(
            string_ref{key_storage[i].data(), length, true}, creator);
      }
    });
  }
  start.store(true, std::memory_order_release);
  for (auto& writer : writers) writer.join();
  finished.store(true, std::memory_order_release);
  reader.join();

  EXPECT_EQ(registry.size(), kKeys);
  for (std::size_t i = 0; i < kKeys; ++i) {
    std::unique_ptr<CheapWidget> value(
        registry.create(key_storage[i].data()));
    ASSERT_NE(value, nullptr);
  }
}

TEST(RegistryDeathTest, RejectsNullCreatorAndInvalidPriority) {
  EXPECT_DEATH(
      ([] {
        hyle::util::Registry<CheapWidget*> registry;
        registry.register_creator("null", nullptr);
      }()),
      "");
  EXPECT_DEATH(
      ([] {
        hyle::util::Registry<CheapWidget*> registry;
        registry.register_creator(
            "priority",
            +[]() -> CheapWidget* { return nullptr; },
            static_cast<RegistryPriority>(99));
      }()),
      "");
}

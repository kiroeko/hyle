#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <limits>
#include <string>
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

struct FailingAllocatorState {
  hyle::mem::Allocator backing = hyle::mem::system_allocator();
  std::size_t calls = 0;
  std::size_t fail_after = 0;
  std::size_t last_allocation_size = 0;
};

void* failing_allocate(void* context, std::size_t size,
                       std::size_t alignment) noexcept {
  auto* state = static_cast<FailingAllocatorState*>(context);
  state->last_allocation_size = size;
  if (state->calls++ >= state->fail_after) return nullptr;
  return state->backing.allocate(size, alignment);
}

void failing_deallocate(void* context, void* pointer, std::size_t size,
                        std::size_t alignment) noexcept {
  static_cast<FailingAllocatorState*>(context)->backing.deallocate(
      pointer, size, alignment);
}

void* failing_reallocate(void* context, void* pointer, std::size_t old_size,
                         std::size_t new_size,
                         std::size_t alignment) noexcept {
  auto* state = static_cast<FailingAllocatorState*>(context);
  if (state->calls++ >= state->fail_after) return nullptr;
  return state->backing.reallocate(pointer, old_size, new_size, alignment);
}

hyle::mem::Allocator failing_allocator(FailingAllocatorState* state) {
  return hyle::mem::Allocator(&failing_allocate, &failing_deallocate,
                              &failing_reallocate, state);
}

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

TEST(Registry, ResultFactoryPreservesLookupAndCreatorErrors) {
  using FactoryResult = hyle::Result<CheapWidget*>;
  hyle::util::Registry<FactoryResult, bool> registry;
  auto creator = +[](bool fail) -> FactoryResult {
    if (fail) {
      return hyle::make_unexpected(
          hyle::Status::Unavailable("backend initialization failed"));
    }
    return new FancyWidget();
  };
  ASSERT_TRUE(registry.register_creator("fallible", creator).ok());

  FactoryResult missing = registry.create("missing", false);
  ASSERT_FALSE(missing.has_value());
  EXPECT_TRUE(missing.error().IsNotFound());

  FactoryResult failed = registry.create("fallible", true);
  ASSERT_FALSE(failed.has_value());
  EXPECT_TRUE(failed.error().IsUnavailable());

  FactoryResult created = registry.create("fallible", false);
  ASSERT_TRUE(created.has_value());
  std::unique_ptr<CheapWidget> object(*created);
  EXPECT_EQ(object->quality(), 10);

  const auto frozen = registry.freeze();
  FactoryResult frozen_failed = frozen.create("fallible", true);
  ASSERT_FALSE(frozen_failed.has_value());
  EXPECT_TRUE(frozen_failed.error().IsUnavailable());
  FactoryResult frozen_missing = frozen.create("missing", false);
  ASSERT_FALSE(frozen_missing.has_value());
  EXPECT_TRUE(frozen_missing.error().IsNotFound());
}

TEST(Registry, ResultDefaultCreatorSupportsIntrusiveRef) {
  using Value = hyle::util::Ref<RefBase>;
  using FactoryResult = hyle::Result<Value>;
  using Registry = hyle::util::Registry<FactoryResult>;
  using Registerer = hyle::util::Registerer<FactoryResult>;
  static_assert(sizeof(Registry) == sizeof(hyle::util::Registry<Value>));

  Registry registry;
  ASSERT_TRUE(registry.register_creator(
      "derived", &Registerer::default_creator<RefDerived>).ok());
  FactoryResult created = registry.create("derived");
  ASSERT_TRUE(created.has_value());
  ASSERT_TRUE(*created);
  EXPECT_EQ((*created)->value(), 7);
}

TEST(Registry, HelpTextIsTypeName) {
  // REGISTER_CLASS stores type_name<Derived>() as help.
  string_ref h = ShapeRegistry()->help("Triangle");
  EXPECT_NE(h.find("Triangle"), string_ref::npos);
}

TEST(Registry, RegisteredKeysVisible) {
  int count = 0;
  bool saw_triangle = false;
  const hyle::Status status = ShapeRegistry()->for_each_key([&](string_ref k) {
    ++count;
    if (k == string_ref{"Triangle"}) saw_triangle = true;
  });
  ASSERT_TRUE(status.ok());
  EXPECT_GE(count, 2);
  EXPECT_TRUE(saw_triangle);
}

// Priority: a kPreferred creator overrides a kFallback one for the same key.
TEST(Registry, PriorityOverride) {
  hyle::util::Registry<CheapWidget*> reg;
  ASSERT_TRUE(reg.register_creator(
      "widget", [](void) -> CheapWidget* { return new CheapWidget(); },
      RegistryPriority::kFallback).ok());
  EXPECT_EQ(reg.size(), 1u);
  {
    std::unique_ptr<CheapWidget> w(reg.create("widget"));
    EXPECT_EQ(w->quality(), 1);  // fallback active
  }
  // Higher priority overrides.
  ASSERT_TRUE(reg.register_creator(
      "widget", [](void) -> CheapWidget* { return new FancyWidget(); },
      RegistryPriority::kPreferred).ok());
  EXPECT_EQ(reg.size(), 1u);  // same key, replaced not added
  {
    std::unique_ptr<CheapWidget> w(reg.create("widget"));
    EXPECT_EQ(w->quality(), 10);  // preferred won
  }
  // A later fallback does NOT displace the preferred one.
  ASSERT_TRUE(reg.register_creator(
      "widget", [](void) -> CheapWidget* { return new CheapWidget(); },
      RegistryPriority::kFallback).ok());
  {
    std::unique_ptr<CheapWidget> w(reg.create("widget"));
    EXPECT_EQ(w->quality(), 10);  // still preferred
  }
}

TEST(Registry, SamePriorityCollisionIsRecoverable) {
  hyle::util::Registry<CheapWidget*> reg;
  ASSERT_TRUE(reg.register_creator(
      "dup", [](void) -> CheapWidget* { return nullptr; },
      RegistryPriority::kDefault).ok());
  const hyle::Status status = reg.register_creator(
      "dup", [](void) -> CheapWidget* { return nullptr; },
      RegistryPriority::kDefault);
  EXPECT_TRUE(status.IsAlreadyExists());
}

TEST(Registry, DefaultCreatorSupportsIntrusiveRef) {
  using Ptr = hyle::util::Ref<RefBase>;
  using Reg = hyle::util::Registry<Ptr>;
  using Registerer = hyle::util::Registerer<Ptr>;
  Reg registry;
  ASSERT_TRUE(registry.register_creator(
      "derived", &Registerer::default_creator<RefDerived>).ok());
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
  ASSERT_TRUE(registry.register_creator(
      "move", &Registerer::default_creator<MoveWidget>).ok());
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
  ASSERT_TRUE(
      registry.register_creator("reentrant", &create_reentrantly).ok());
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
        ASSERT_TRUE(registry.register_creator(
            string_ref{key_storage[i].data(), length, true}, creator).ok());
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

TEST(Registry, RejectsInvalidRegistration) {
  hyle::util::Registry<CheapWidget*> registry;
  EXPECT_TRUE(registry.register_creator("null", nullptr).IsInvalidArgument());
  EXPECT_TRUE(registry.register_creator(
      "priority", +[]() -> CheapWidget* { return nullptr; },
      static_cast<RegistryPriority>(99)).IsInvalidArgument());
  EXPECT_TRUE(registry.register_creator(
      "", +[]() -> CheapWidget* { return nullptr; }).IsInvalidArgument());
}

TEST(Registry, OwnsDynamicKeysAndHelpAndKeepsOldViewsStable) {
  hyle::util::Registry<CheapWidget*> registry;
  string_ref old_help;
  {
    std::string key = "dynamic-widget";
    std::string help = "fallback implementation";
    ASSERT_TRUE(registry.register_creator(
        key, +[]() -> CheapWidget* { return new CheapWidget(); },
        RegistryPriority::kFallback, help).ok());
    old_help = registry.help("dynamic-widget");
  }
  EXPECT_EQ(old_help, string_ref{"fallback implementation"});
  std::unique_ptr<CheapWidget> value(registry.create("dynamic-widget"));
  ASSERT_NE(value, nullptr);

  ASSERT_TRUE(registry.register_creator(
      "dynamic-widget", +[]() -> CheapWidget* { return new FancyWidget(); },
      RegistryPriority::kPreferred, "preferred implementation").ok());
  EXPECT_EQ(old_help, string_ref{"fallback implementation"});
  EXPECT_EQ(registry.help("dynamic-widget"),
            string_ref{"preferred implementation"});
}

TEST(Registry, MutableEnumerationIsSnapshotBasedAndReentrant) {
  hyle::util::Registry<CheapWidget*> registry;
  auto creator = +[]() -> CheapWidget* { return nullptr; };
  ASSERT_TRUE(registry.register_creator("first", creator).ok());

  std::size_t visited = 0;
  const hyle::Status status = registry.for_each(
      [&](hyle::util::Registry<CheapWidget*>::EntryView entry) {
        ++visited;
        EXPECT_TRUE(registry.has(entry.key));
        if (entry.key == string_ref{"first"}) {
          EXPECT_TRUE(registry.register_creator("second", creator).ok());
        }
      });
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(visited, 1u);
  EXPECT_TRUE(registry.has("second"));
}

TEST(Registry, FreezePublishesLockFreeReadPhase) {
  hyle::util::Registry<CheapWidget*> registry;
  ASSERT_TRUE(registry.reserve(32).ok());
  auto creator = +[]() -> CheapWidget* { return nullptr; };
  ASSERT_TRUE(registry.register_creator("frozen", creator, RegistryPriority::kDefault,
                                        "immutable entry").ok());
  const auto frozen = registry.freeze();
  const auto frozen_again = registry.freeze();
  (void)frozen_again;
  EXPECT_TRUE(registry.frozen());
  EXPECT_TRUE(frozen.has("frozen"));
  EXPECT_EQ(frozen.help("frozen"), string_ref{"immutable entry"});
  EXPECT_EQ(frozen.size(), 1u);
  EXPECT_EQ(frozen.create("frozen"), nullptr);
  EXPECT_TRUE(registry.has("frozen"));
  EXPECT_EQ(registry.help("frozen"), string_ref{"immutable entry"});
  EXPECT_TRUE(registry.register_creator("late", creator).IsFailedPrecondition());
  EXPECT_TRUE(registry.reserve(64).IsFailedPrecondition());

  std::atomic<std::size_t> observations{0};
  std::vector<std::thread> readers;
  for (int worker = 0; worker < 8; ++worker) {
    readers.emplace_back([&] {
      for (int i = 0; i < 10'000; ++i) {
        if (registry.has("frozen") && registry.size() == 1) {
          observations.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }
  for (auto& reader : readers) reader.join();
  EXPECT_EQ(observations.load(), 80'000u);

  std::size_t visited = 0;
  ASSERT_TRUE(registry.for_each([&](auto entry) {
    ++visited;
    EXPECT_TRUE(registry.has(entry.key));
  }).ok());
  EXPECT_EQ(visited, 1u);

  visited = 0;
  frozen.for_each([&](auto entry) {
    ++visited;
    EXPECT_EQ(entry.key, string_ref{"frozen"});
  });
  EXPECT_EQ(visited, 1u);
}

TEST(Registry, AllocationFailureIsRecoverableAndFreezeDoesNotAllocate) {
  auto creator = +[]() -> CheapWidget* { return nullptr; };
  FailingAllocatorState immediate_failure;
  hyle::util::Registry<CheapWidget*> empty(
      failing_allocator(&immediate_failure));
  EXPECT_TRUE(empty.reserve(1).IsResourceExhausted());
  EXPECT_TRUE(empty.register_creator("entry", creator).IsResourceExhausted());
  EXPECT_FALSE(empty.frozen());

  FailingAllocatorState freeze_failure;
  freeze_failure.fail_after = (std::numeric_limits<std::size_t>::max)();
  hyle::util::Registry<CheapWidget*> populated(
      failing_allocator(&freeze_failure));
  ASSERT_TRUE(populated.register_creator("entry", creator).ok());
  const std::size_t calls_before_freeze = freeze_failure.calls;
  freeze_failure.fail_after = freeze_failure.calls;
  const auto frozen = populated.freeze();
  (void)frozen;
  EXPECT_EQ(freeze_failure.calls, calls_before_freeze);
  EXPECT_TRUE(populated.frozen());
  EXPECT_TRUE(populated.has("entry"));
  // Idempotence also performs no allocation.
  const auto frozen_again = populated.freeze();
  (void)frozen_again;
  EXPECT_EQ(freeze_failure.calls, calls_before_freeze);
  EXPECT_TRUE(populated.frozen());
  EXPECT_TRUE(populated.has("entry"));
}

TEST(Registry, MaximumStringLengthWidensBeforeNulByteAllocation) {
  auto creator = +[]() -> CheapWidget* { return nullptr; };
  FailingAllocatorState failure;
  hyle::util::Registry<CheapWidget*> registry(failing_allocator(&failure));

  // The empty Swiss index returns before hashing. The backing allocator then
  // rejects the first string-region request, so the one-byte placeholder is
  // never read and this test does not allocate or copy a 4 GiB string. Before
  // widening, uint32_t(size + 1) wrapped to zero and Region never called the
  // backing allocator at all.
  const char placeholder = 'x';
  const string_ref maximum_key{&placeholder, string_ref::npos};
  const hyle::Status status =
      registry.register_creator(maximum_key, creator);

  EXPECT_TRUE(status.IsResourceExhausted());
  EXPECT_EQ(failure.calls, 1u);
  EXPECT_GT(failure.last_allocation_size,
            static_cast<std::size_t>(string_ref::npos));
}

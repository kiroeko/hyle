#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "hyle/container/string_view.h"
#include "hyle/util/ref_counted.h"
#include "hyle/util/registry.h"

using hyle::container::StringView;
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
  explicit MoveWidget(std::unique_ptr<int> input) noexcept : value(*input) {}
  int value;
};

Shape* create_triangle() noexcept { return new (std::nothrow) Triangle(); }
Shape* create_square() noexcept { return new (std::nothrow) Square(); }

hyle::Status register_shapes(hyle::util::Registry<Shape*>* registry) noexcept {
  hyle::Status status = registry->register_creator(
      "triangle", &create_triangle, RegistryPriority::kDefault,
      "three-sided shape");
  if (!status.ok()) return status;
  return registry->register_creator("square", &create_square,
                                    RegistryPriority::kDefault,
                                    "four-sided shape");
}

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

TEST(Registry, CreateByName) {
  hyle::util::Registry<Shape*> registry;
  ASSERT_TRUE(register_shapes(&registry).ok());
  std::unique_ptr<Shape> t(registry.create("triangle"));
  ASSERT_NE(t, nullptr);
  EXPECT_EQ(t->sides(), 3);
  std::unique_ptr<Shape> s(registry.create("square"));
  ASSERT_NE(s, nullptr);
  EXPECT_EQ(s->sides(), 4);
}

TEST(Registry, FindResolvesFactoryMetadataWithoutCreating) {
  hyle::util::Registry<Shape*> registry;
  ASSERT_TRUE(register_shapes(&registry).ok());

  const auto triangle = registry.find("triangle");
  ASSERT_TRUE(triangle.has_value());
  EXPECT_EQ(triangle->key, StringView{"triangle"});
  EXPECT_EQ(triangle->creator, &create_triangle);
  EXPECT_EQ(triangle->priority, RegistryPriority::kDefault);
  EXPECT_EQ(triangle->help, StringView{"three-sided shape"});
  EXPECT_FALSE(registry.find("hexagon").has_value());

  std::unique_ptr<Shape> object(triangle->creator());
  ASSERT_NE(object, nullptr);
  EXPECT_EQ(object->sides(), 3);

  const auto frozen = registry.freeze();
  const auto square = frozen.find("square");
  ASSERT_TRUE(square.has_value());
  EXPECT_EQ(square->key, StringView{"square"});
  EXPECT_EQ(square->creator, &create_square);
  EXPECT_EQ(square->help, StringView{"four-sided shape"});
  EXPECT_FALSE(frozen.find("hexagon").has_value());
}

TEST(Registry, MissingKeyReturnsNull) {
  hyle::util::Registry<Shape*> registry;
  ASSERT_TRUE(register_shapes(&registry).ok());
  EXPECT_EQ(registry.create("hexagon"), nullptr);
  EXPECT_FALSE(registry.has("hexagon"));
  EXPECT_TRUE(registry.has("triangle"));
}

TEST(Registry, ResultFactoryPreservesLookupAndCreatorErrors) {
  using FactoryResult = hyle::Result<CheapWidget*>;
  hyle::util::Registry<FactoryResult, bool> registry;
  auto creator = +[](bool fail) noexcept -> FactoryResult {
    if (fail) {
      return hyle::make_unexpected(
          hyle::Status::Unavailable("backend initialization failed"));
    }
    auto* value = new (std::nothrow) FancyWidget();
    if (value == nullptr) {
      return hyle::make_unexpected(
          hyle::Status::ResourceExhausted("backend allocation failed"));
    }
    return value;
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

TEST(Registry, ExplicitFallibleCreatorSupportsIntrusiveRef) {
  using Value = hyle::util::Ref<RefBase>;
  using FactoryResult = hyle::Result<Value>;
  using Registry = hyle::util::Registry<FactoryResult>;
  static_assert(sizeof(Registry) == sizeof(hyle::util::Registry<Value>));

  Registry registry;
  auto creator = +[]() noexcept -> FactoryResult {
    auto* object = new (std::nothrow) RefDerived();
    if (object == nullptr) {
      return hyle::make_unexpected(
          hyle::Status::ResourceExhausted("object allocation failed"));
    }
    Value value = hyle::util::adopt_ref(object);
    return FactoryResult{std::in_place, std::move(value)};
  };
  ASSERT_TRUE(registry.register_creator("derived", creator).ok());
  FactoryResult created = registry.create("derived");
  ASSERT_TRUE(created.has_value());
  ASSERT_TRUE(*created);
  EXPECT_EQ((*created)->value(), 7);
}

TEST(Registry, HelpTextIsExplicitConsumerDescription) {
  hyle::util::Registry<Shape*> registry;
  ASSERT_TRUE(register_shapes(&registry).ok());
  EXPECT_EQ(registry.help("triangle"), StringView{"three-sided shape"});
}

TEST(Registry, RegisteredKeysVisible) {
  hyle::util::Registry<Shape*> registry;
  ASSERT_TRUE(register_shapes(&registry).ok());
  int count = 0;
  bool saw_triangle = false;
  const hyle::Status status = registry.for_each_key([&](StringView k) {
    ++count;
    if (k == StringView{"triangle"}) saw_triangle = true;
  });
  ASSERT_TRUE(status.ok());
  EXPECT_GE(count, 2);
  EXPECT_TRUE(saw_triangle);
}

// Priority: a kPreferred creator overrides a kFallback one for the same key.
TEST(Registry, PriorityOverride) {
  hyle::util::Registry<CheapWidget*> reg;
  ASSERT_TRUE(reg.register_creator(
      "widget", [](void) noexcept -> CheapWidget* {
        return new (std::nothrow) CheapWidget();
      },
      RegistryPriority::kFallback).ok());
  EXPECT_EQ(reg.size(), 1u);
  {
    std::unique_ptr<CheapWidget> w(reg.create("widget"));
    EXPECT_EQ(w->quality(), 1);  // fallback active
  }
  // Higher priority overrides.
  ASSERT_TRUE(reg.register_creator(
      "widget", [](void) noexcept -> CheapWidget* {
        return new (std::nothrow) FancyWidget();
      },
      RegistryPriority::kPreferred).ok());
  EXPECT_EQ(reg.size(), 1u);  // same key, replaced not added
  {
    std::unique_ptr<CheapWidget> w(reg.create("widget"));
    EXPECT_EQ(w->quality(), 10);  // preferred won
  }
  // A later fallback does NOT displace the preferred one.
  ASSERT_TRUE(reg.register_creator(
      "widget", [](void) noexcept -> CheapWidget* {
        return new (std::nothrow) CheapWidget();
      },
      RegistryPriority::kFallback).ok());
  {
    std::unique_ptr<CheapWidget> w(reg.create("widget"));
    EXPECT_EQ(w->quality(), 10);  // still preferred
  }
}

TEST(Registry, SamePriorityCollisionIsRecoverable) {
  hyle::util::Registry<CheapWidget*> reg;
  ASSERT_TRUE(reg.register_creator(
      "dup", [](void) noexcept -> CheapWidget* { return nullptr; },
      RegistryPriority::kDefault).ok());
  const hyle::Status status = reg.register_creator(
      "dup", [](void) noexcept -> CheapWidget* { return nullptr; },
      RegistryPriority::kDefault);
  EXPECT_TRUE(status.IsAlreadyExists());
}

TEST(Registry, ExplicitNullableCreatorSupportsIntrusiveRef) {
  using Ptr = hyle::util::Ref<RefBase>;
  using Reg = hyle::util::Registry<Ptr>;
  Reg registry;
  auto creator = +[]() noexcept -> Ptr {
    return hyle::util::adopt_ref(new (std::nothrow) RefDerived());
  };
  ASSERT_TRUE(registry.register_creator("derived", creator).ok());
  Ptr value = registry.create("derived");
  ASSERT_TRUE(value);
  EXPECT_EQ(value->value(), 7);
}

TEST(Registry, MoveOnlyFactoryArgumentsAreForwarded) {
  using Ptr = hyle::util::Ref<MoveWidget>;
  using Reg = hyle::util::Registry<Ptr, std::unique_ptr<int>>;
  Reg registry;
  auto creator = +[](std::unique_ptr<int> input) noexcept -> Ptr {
    return hyle::util::adopt_ref(
        new (std::nothrow) MoveWidget(std::move(input)));
  };
  ASSERT_TRUE(registry.register_creator("move", creator).ok());
  Ptr value =
      registry.create("move", std::make_unique<int>(42));
  ASSERT_TRUE(value);
  EXPECT_EQ(value->value, 42);
}

namespace {

using ReentrantRegistry = hyle::util::Registry<int*>;
ReentrantRegistry* g_reentrant_registry = nullptr;

int* create_reentrantly() noexcept {
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
  auto creator = +[]() noexcept -> CheapWidget* {
    return new (std::nothrow) CheapWidget();
  };
  std::atomic<bool> start{false};
  std::atomic<bool> finished{false};
  std::thread reader([&] {
    while (!start.load(std::memory_order_acquire)) {}
    while (!finished.load(std::memory_order_acquire)) {
      (void)registry.has("missing");
      (void)registry.find("missing");
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
            StringView{key_storage[i].data(), length, true}, creator).ok());
      }
    });
  }
  start.store(true, std::memory_order_release);
  for (auto& writer : writers) writer.join();
  finished.store(true, std::memory_order_release);
  reader.join();

  EXPECT_EQ(registry.size(), kKeys);
  for (std::size_t i = 0; i < kKeys; ++i) {
    const auto entry = registry.find(key_storage[i].data());
    ASSERT_TRUE(entry.has_value());
    EXPECT_EQ(entry->creator, creator);
    std::unique_ptr<CheapWidget> value(
        entry->creator());
    ASSERT_NE(value, nullptr);
  }
}

TEST(Registry, RejectsInvalidRegistration) {
  hyle::util::Registry<CheapWidget*> registry;
  EXPECT_TRUE(registry.register_creator("null", nullptr).IsInvalidArgument());
  EXPECT_TRUE(registry.register_creator(
      "priority", +[]() noexcept -> CheapWidget* { return nullptr; },
      static_cast<RegistryPriority>(99)).IsInvalidArgument());
  EXPECT_TRUE(
      registry
          .register_creator(
              "", +[]() noexcept -> CheapWidget* { return nullptr; })
          .IsInvalidArgument());
}

TEST(Registry, OwnsDynamicKeysAndHelpAndKeepsOldViewsStable) {
  hyle::util::Registry<CheapWidget*> registry;
  auto fallback = +[]() noexcept -> CheapWidget* {
    return new (std::nothrow) CheapWidget();
  };
  auto preferred = +[]() noexcept -> CheapWidget* {
    return new (std::nothrow) FancyWidget();
  };
  StringView old_help;
  std::optional<hyle::util::Registry<CheapWidget*>::EntryView> old_entry;
  {
    std::string key = "dynamic-widget";
    std::string help = "fallback implementation";
    ASSERT_TRUE(registry.register_creator(
        key, fallback, RegistryPriority::kFallback, help).ok());
    old_help = registry.help("dynamic-widget");
    old_entry = registry.find("dynamic-widget");
  }
  ASSERT_TRUE(old_entry.has_value());
  EXPECT_EQ(old_entry->creator, fallback);
  EXPECT_EQ(old_entry->priority, RegistryPriority::kFallback);
  EXPECT_EQ(old_entry->key, StringView{"dynamic-widget"});
  EXPECT_EQ(old_help, StringView{"fallback implementation"});
  std::unique_ptr<CheapWidget> value(registry.create("dynamic-widget"));
  ASSERT_NE(value, nullptr);

  ASSERT_TRUE(registry.register_creator(
      "dynamic-widget", preferred, RegistryPriority::kPreferred,
      "preferred implementation").ok());
  const auto current_entry = registry.find("dynamic-widget");
  ASSERT_TRUE(current_entry.has_value());
  EXPECT_EQ(current_entry->creator, preferred);
  EXPECT_EQ(current_entry->priority, RegistryPriority::kPreferred);
  EXPECT_EQ(current_entry->help, StringView{"preferred implementation"});
  EXPECT_EQ(old_entry->creator, fallback);
  EXPECT_EQ(old_entry->priority, RegistryPriority::kFallback);
  EXPECT_EQ(old_entry->help, StringView{"fallback implementation"});
  EXPECT_EQ(old_help, StringView{"fallback implementation"});
  EXPECT_EQ(registry.help("dynamic-widget"),
            StringView{"preferred implementation"});
}

TEST(Registry, MutableEnumerationIsSnapshotBasedAndReentrant) {
  hyle::util::Registry<CheapWidget*> registry;
  auto creator = +[]() noexcept -> CheapWidget* { return nullptr; };
  ASSERT_TRUE(registry.register_creator("first", creator).ok());

  std::size_t visited = 0;
  const hyle::Status status = registry.for_each(
      [&](hyle::util::Registry<CheapWidget*>::EntryView entry) {
        ++visited;
        EXPECT_TRUE(registry.has(entry.key));
        if (entry.key == StringView{"first"}) {
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
  auto creator = +[]() noexcept -> CheapWidget* { return nullptr; };
  ASSERT_TRUE(registry.register_creator("frozen", creator, RegistryPriority::kDefault,
                                        "immutable entry").ok());
  const auto frozen = registry.freeze();
  const auto frozen_again = registry.freeze();
  (void)frozen_again;
  EXPECT_TRUE(registry.frozen());
  EXPECT_TRUE(frozen.has("frozen"));
  const auto found = frozen.find("frozen");
  ASSERT_TRUE(found.has_value());
  EXPECT_EQ(found->creator, creator);
  EXPECT_EQ(found->priority, RegistryPriority::kDefault);
  EXPECT_EQ(found->help, StringView{"immutable entry"});
  EXPECT_EQ(frozen.help("frozen"), StringView{"immutable entry"});
  EXPECT_EQ(frozen.size(), 1u);
  EXPECT_EQ(frozen.create("frozen"), nullptr);
  EXPECT_TRUE(registry.has("frozen"));
  EXPECT_EQ(registry.help("frozen"), StringView{"immutable entry"});
  EXPECT_TRUE(registry.register_creator("late", creator).IsFailedPrecondition());
  EXPECT_TRUE(registry.reserve(64).IsFailedPrecondition());

  std::atomic<std::size_t> observations{0};
  std::vector<std::thread> readers;
  for (int worker = 0; worker < 8; ++worker) {
    readers.emplace_back([&] {
      for (int i = 0; i < 10'000; ++i) {
        if (registry.has("frozen") && registry.size() == 1) {
          const auto entry = frozen.find("frozen");
          EXPECT_TRUE(entry.has_value());
          EXPECT_EQ(entry->creator, creator);
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
    EXPECT_EQ(entry.key, StringView{"frozen"});
  });
  EXPECT_EQ(visited, 1u);
}

TEST(Registry, AllocationFailureIsRecoverableAndFreezeDoesNotAllocate) {
  auto creator = +[]() noexcept -> CheapWidget* { return nullptr; };
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
  EXPECT_EQ(freeze_failure.calls, calls_before_freeze);
  EXPECT_TRUE(populated.frozen());
  EXPECT_TRUE(populated.has("entry"));
  EXPECT_TRUE(frozen.has("entry"));
  EXPECT_TRUE(populated.find("entry").has_value());
  EXPECT_TRUE(frozen.find("entry").has_value());
  EXPECT_FALSE(frozen.find("missing").has_value());
  EXPECT_EQ(frozen.help("entry"), StringView{});
  EXPECT_EQ(frozen.size(), 1u);
  EXPECT_EQ(frozen.create("entry"), nullptr);
  std::size_t visited = 0;
  frozen.for_each([&](auto) { ++visited; });
  EXPECT_EQ(visited, 1u);
  EXPECT_EQ(freeze_failure.calls, calls_before_freeze);
  // Idempotence also performs no allocation.
  const auto frozen_again = populated.freeze();
  (void)frozen_again;
  EXPECT_EQ(freeze_failure.calls, calls_before_freeze);
  EXPECT_TRUE(populated.frozen());
  EXPECT_TRUE(populated.has("entry"));
}

TEST(Registry, EveryInitialAllocationFailureIsAtomicAndRetryable) {
  auto creator = +[]() noexcept -> CheapWidget* { return nullptr; };
  FailingAllocatorState probe;
  probe.fail_after = (std::numeric_limits<std::size_t>::max)();
  {
    hyle::util::Registry<CheapWidget*> registry(failing_allocator(&probe));
    ASSERT_TRUE(registry.register_creator("entry", creator,
                                          RegistryPriority::kDefault,
                                          "diagnostics")
                    .ok());
  }
  ASSERT_GT(probe.calls, 0u);

  // Discover the implementation's current backing-call count, then exercise
  // every recoverable boundary. The semantic assertion is independent of how
  // Swiss capacity happens to package its own storage.
  for (std::size_t fail_after = 0; fail_after < probe.calls; ++fail_after) {
    FailingAllocatorState failure;
    failure.fail_after = fail_after;
    hyle::util::Registry<CheapWidget*> registry(failing_allocator(&failure));

    EXPECT_TRUE(registry.register_creator("entry", creator,
                                          RegistryPriority::kDefault,
                                          "diagnostics")
                    .IsResourceExhausted());
    EXPECT_EQ(registry.size(), 0u);
    EXPECT_FALSE(registry.has("entry"));

    failure.fail_after = (std::numeric_limits<std::size_t>::max)();
    EXPECT_TRUE(registry.register_creator("entry", creator,
                                         RegistryPriority::kDefault,
                                         "diagnostics")
                    .ok());
    EXPECT_EQ(registry.size(), 1u);
    EXPECT_EQ(registry.help("entry"), StringView{"diagnostics"});
  }
}

TEST(Registry, FailedPriorityReplacementKeepsPreviousEntry) {
  auto fallback = +[]() noexcept -> CheapWidget* {
    return new (std::nothrow) CheapWidget();
  };
  auto preferred = +[]() noexcept -> CheapWidget* {
    return new (std::nothrow) FancyWidget();
  };
  FailingAllocatorState failure;
  failure.fail_after = (std::numeric_limits<std::size_t>::max)();
  hyle::util::Registry<CheapWidget*> registry(failing_allocator(&failure));
  ASSERT_TRUE(registry.register_creator("widget", fallback,
                                        RegistryPriority::kFallback,
                                        "fallback")
                  .ok());

  failure.fail_after = failure.calls;
  const std::string large_help(20 * 1024, 'x');
  EXPECT_TRUE(registry.register_creator("widget", preferred,
                                        RegistryPriority::kPreferred,
                                        large_help)
                  .IsResourceExhausted());
  EXPECT_EQ(registry.help("widget"), StringView{"fallback"});
  std::unique_ptr<CheapWidget> value(registry.create("widget"));
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(value->quality(), 1);
}

TEST(Registry, MaximumStringLengthWidensBeforeNulByteAllocation) {
  auto creator = +[]() noexcept -> CheapWidget* { return nullptr; };
  FailingAllocatorState probe;
  probe.fail_after = (std::numeric_limits<std::size_t>::max)();
  {
    hyle::util::Registry<CheapWidget*> ordinary(failing_allocator(&probe));
    ASSERT_TRUE(ordinary.register_creator("ordinary", creator).ok());
  }
  ASSERT_GT(probe.calls, 0u);

  FailingAllocatorState failure;
  // The combined entry storage is the last backing allocation after table
  // preparation. Fail there so no bytes are read from the synthetic huge view.
  failure.fail_after = probe.calls - 1;
  hyle::util::Registry<CheapWidget*> registry(failing_allocator(&failure));

  // The first call prepares the empty Swiss index. The backing allocator then
  // rejects the combined entry storage request, so the one-byte placeholder is
  // never read and this test does not copy a 4 GiB string. Widening must happen
  // before adding the terminator and Entry bytes.
  const char placeholder = 'x';
  const StringView maximum_key{&placeholder, StringView::npos};
  const hyle::Status status =
      registry.register_creator(maximum_key, creator);

  EXPECT_TRUE(status.IsResourceExhausted());
  EXPECT_EQ(failure.calls, probe.calls);
  EXPECT_GT(failure.last_allocation_size,
            static_cast<std::size_t>(StringView::npos));
}

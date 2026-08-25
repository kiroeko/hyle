#pragma once

#include <atomic>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/container/string_ref.h"
#include "hyle/container/swiss_table.h"
#include "hyle/error/result.h"
#include "hyle/error/status.h"
#include "hyle/macros.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/region.h"
#include "hyle/util/ref_counted.h"
#include "hyle/util/type_name.h"

// Self-registration registry (clay registry.h): map a name -> a factory, so
// plugins register themselves at static-init time and code creates objects by
// name without a compile-time dependency on the concrete types. Used for
// backend/plugin selection (conventions §9).
//
// Divergences from clay (deliberate, per conventions §1/§9):
//  - mutable storage is a compact HYLE flat index, NOT std::unordered_map
//    (clay used THREE of them). §1 铁律: no std hash containers in production.
//  - Creator is a FUNCTION POINTER, not std::function. So Entry is trivially
//    copyable and lives INLINE in the table (no heap, no type-erasure alloc),
//    and it is ABI-friendlier (§4). Registry creators are stateless type
//    constructors (`new Derived(args...)`) -- they never need capture.
//  - normal registration returns Status, so dynamic/plugin wiring can recover.
//    The static REGISTER macros still fail fast because static initialization
//    has no caller to receive an error.
//  - help text uses type_name<T>() (util-02), not clay's demangle.
//
// The registry owns copies of keys and help strings. Returned views stay valid
// until registry destruction, including after a higher-priority replacement.
// Mutable reads are synchronized; creators and visitors always run outside the
// lock and may re-enter. freeze() closes registration permanently and publishes
// the existing compact Swiss index, after which reads are lock-free expected
// O(1), freeze itself is O(1), and enumeration performs no allocation. This
// matches the real init-heavy/read-heavy lifecycle of named backend selection.

namespace hyle::util {

namespace detail {

// A nullable factory result keeps the traditional fast "missing -> empty"
// contract.  A Result<T> factory is the error-aware acceleration-library
// shape: lookup failure becomes NotFound and the creator's Status propagates
// unchanged.  Both use exactly the same table/layout; callers choose the
// failure contract in the Registry template argument instead of every entry
// paying for a tag, a second function pointer, or benchmark-only metadata.
template <class FactoryResult>
struct RegistryResultTraits {
  static FactoryResult missing() { return FactoryResult{}; }

  template <class Derived, class... Args>
  static FactoryResult default_create(Args&&... args) {
    if constexpr (std::is_pointer_v<FactoryResult>) {
      return new Derived(std::forward<Args>(args)...);
    } else if constexpr (requires {
                           FactoryResult{
                               new Derived(std::forward<Args>(args)...)};
                         }) {
      return FactoryResult{new Derived(std::forward<Args>(args)...)};
    } else {
      static_assert(
          std::convertible_to<Ref<Derived>, FactoryResult>,
          "Registry result must own a Derived* or accept Ref<Derived>");
      return make_ref<Derived>(std::forward<Args>(args)...);
    }
  }
};

template <class Value>
struct RegistryResultTraits<std::expected<Value, Status>> {
  static Result<Value> missing() {
    return make_unexpected(Status::NotFound("registry key is not registered"));
  }

  template <class Derived, class... Args>
  static Result<Value> default_create(Args&&... args) {
    Value value = RegistryResultTraits<Value>::template default_create<Derived>(
        std::forward<Args>(args)...);
    return Result<Value>{std::in_place, std::move(value)};
  }
};

}  // namespace detail

enum class RegistryPriority : std::uint8_t {
  kFallback = 1,  // lowest: a default that anything else overrides
  kDefault = 2,
  kPreferred = 3,  // highest: explicitly preferred implementation
};

template <class Ptr, class... Args>
class Registry {
 private:
  struct Entry;
  struct IndexValue;

 public:
  using Creator = Ptr (*)(Args...);  // function pointer (no capture, no alloc)

  struct EntryView {
    container::string_ref key;
    Creator creator;
    RegistryPriority priority;
    container::string_ref help;
  };

  // A non-owning handle to the registry's immutable phase. Cache this handle
  // in dispatch-heavy consumers after freeze(): unlike Registry::create(), it
  // does not need to re-check the lifecycle publication atomic on every read.
  // The originating Registry must outlive the view and must not be destroyed
  // while any view operation is in flight.
  class FrozenView {
   public:
    HYLE_FORCE_INLINE Ptr create(const container::string_ref& key,
                                 Args... args) const {
      return Registry::create_frozen_from(index_, key,
                                          std::forward<Args>(args)...);
    }

    bool has(const container::string_ref& key) const noexcept {
      return Registry::frozen_table(index_)->find(key) != nullptr;
    }

    container::string_ref help(
        const container::string_ref& key) const noexcept {
      if (const IndexValue* value = Registry::frozen_table(index_)->find(key)) {
        return value->entry->help;
      }
      return {};
    }

    std::size_t size() const noexcept {
      return Registry::frozen_table(index_)->size();
    }

    template <class Fn>
    void for_each(Fn&& fn) const {
      auto&& visitor = fn;
      Registry::visit_frozen_table(visitor, Registry::frozen_table(index_));
    }

   private:
    friend class Registry;
    explicit FrozenView(const void* index) noexcept : index_(index) {}
    const void* index_;
  };

  explicit Registry(mem::Allocator alloc = mem::system_allocator())
      : alloc_(alloc), keys_(alloc, string_region_options()),
        metadata_(alloc, string_region_options()), entries_(alloc) {}
  ~Registry() = default;
  HYLE_DISABLE_COPY_AND_MOVE(Registry);

  // Register `creator` under `key`. Higher priority wins; on equal priority for
  // the same key returns AlreadyExists. Lower priority is a successful no-op.
  // `help` is optional descriptive text. Key/help are copied before return.
  // Registration after freeze() returns FailedPrecondition.
  [[nodiscard]] Status register_creator(
      container::string_ref key, Creator creator,
      RegistryPriority priority = RegistryPriority::kDefault,
      container::string_ref help = {}) noexcept {
    if (key.data == nullptr || key.size == 0) {
      return Status::InvalidArgument("registry key is null or empty");
    }
    if (help.data == nullptr && help.size != 0) {
      return Status::InvalidArgument("registry help has null data");
    }
    if (creator == nullptr) {
      return Status::InvalidArgument("registry creator is null");
    }
    if (!valid_priority(priority)) {
      return Status::InvalidArgument("registry priority is invalid");
    }

    std::lock_guard<std::mutex> lock(mu_);
    if (frozen_index_.load(std::memory_order_relaxed) != nullptr) {
      return Status::FailedPrecondition("registry is frozen");
    }
    if (IndexValue* index = entries_.find(key)) {
      Entry* e = index->entry;
      if (priority > e->priority) {
        container::string_ref owned_help;
        if (!copy_string(help, &owned_help, metadata_)) {
          return Status::ResourceExhausted(
              "registry help allocation failed");
        }
        e->creator = creator;
        index->creator = creator;
        e->priority = priority;
        e->help = owned_help;
      } else if (priority == e->priority) {
        return Status::AlreadyExists(
            "registry key already has the same priority");
      }
      return Status::Ok();
    } else {
      container::string_ref owned_key;
      container::string_ref owned_help;
      if (!copy_string(key, &owned_key, keys_) ||
          !copy_string(help, &owned_help, metadata_)) {
        return Status::ResourceExhausted(
            "registry string allocation failed");
      }
      Entry* entry =
          allocate_entry(owned_key, creator, priority, owned_help);
      if (entry == nullptr) {
        return Status::ResourceExhausted("registry entry allocation failed");
      }
      using InsertResult = typename Table::InsertResult;
      if (entries_.try_insert_or_assign(
              owned_key, IndexValue{entry, creator}) ==
          InsertResult::kOutOfMemory) {
        return Status::ResourceExhausted("registry table allocation failed");
      }
    }
    return Status::Ok();
  }

  // Optional startup tuning. It changes no semantics and is rejected after the
  // immutable read phase begins.
  [[nodiscard]] Status reserve(std::size_t count) noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    if (frozen_index_.load(std::memory_order_relaxed) != nullptr) {
      return Status::FailedPrecondition("registry is frozen");
    }
    return entries_.reserve(count)
               ? Status::Ok()
               : Status::ResourceExhausted("registry reserve failed");
  }

  // One-way lifecycle transition. The startup Swiss map already owns compact
  // keys and inlines the lookup/call hot fields, so publication is O(1), does
  // not allocate, and preserves exactly the same mature probing shape.
  // Idempotent so independently composed startup code may call it.
  [[nodiscard]] FrozenView freeze() noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    if (frozen_index_.load(std::memory_order_relaxed) != nullptr) {
      return FrozenView{&entries_};
    }

    // The pointer is both lifecycle state and publication token. Readers need
    // one acquire load, rather than a state load followed by an index load.
    frozen_index_.store(&entries_, std::memory_order_release);
    return FrozenView{&entries_};
  }

  bool frozen() const noexcept {
    return frozen_index_.load(std::memory_order_acquire) != nullptr;
  }

  bool has(container::string_ref key) const noexcept {
    if (const auto* index = frozen_index_.load(std::memory_order_acquire)) {
      return index->find(key) != nullptr;
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (const auto* index = frozen_index_.load(std::memory_order_relaxed)) {
      return index->find(key) != nullptr;
    }
    return entries_.find(key) != nullptr;
  }

  // Create the object registered under `key`. A nullable Ptr returns its empty
  // value when absent. Registry<Result<T>, ...> instead returns NotFound and
  // propagates the selected creator's Status without changing the table shape.
  HYLE_FORCE_INLINE Ptr create(container::string_ref key, Args... args) const {
    if (const auto* frozen_slots =
            frozen_index_.load(std::memory_order_acquire)) {
      if (const IndexValue* value = frozen_slots->find(key)) {
        return value->creator(std::forward<Args>(args)...);
      }
      return detail::RegistryResultTraits<Ptr>::missing();
    }
    return create_mutable(key, std::forward<Args>(args)...);
  }

  // Help text for `key` (empty if absent or none given).
  container::string_ref help(container::string_ref key) const noexcept {
    if (const auto* frozen_slots =
            frozen_index_.load(std::memory_order_acquire)) {
      if (const IndexValue* value = frozen_slots->find(key)) {
        return value->entry->help;
      }
      return {};
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (const auto* published_slots =
            frozen_index_.load(std::memory_order_relaxed)) {
      if (const IndexValue* value = published_slots->find(key)) {
        return value->entry->help;
      }
    } else if (const IndexValue* found = entries_.find(key)) {
      return found->entry->help;
    }
    return {};
  }

  std::size_t size() const noexcept {
    if (const auto* index = frozen_index_.load(std::memory_order_acquire)) {
      return index->size();
    }
    std::lock_guard<std::mutex> lock(mu_);
    return entries_.size();
  }

  // Visit a point-in-time snapshot in unspecified order. A callback returning
  // bool may stop early by returning false; a void callback always continues.
  // In mutable mode one temporary array is allocated, then the lock is released
  // before callbacks. Frozen mode is lock- and allocation-free.
  template <class Fn>
  [[nodiscard]] Status for_each(Fn&& fn) const {
    auto&& visitor = fn;
    if (const auto* index = frozen_index_.load(std::memory_order_acquire)) {
      visit_frozen(visitor, index);
      return Status::Ok();
    }

    SnapshotStorage snapshot(alloc_);
    bool became_frozen = false;
    {
      std::lock_guard<std::mutex> lock(mu_);
      became_frozen =
          frozen_index_.load(std::memory_order_relaxed) != nullptr;
      if (became_frozen) {
        // freeze() published immutable storage before releasing this mutex.
      } else if (!snapshot.allocate(entries_.size())) {
        return Status::ResourceExhausted(
            "registry enumeration snapshot allocation failed");
      } else {
        entries_.for_each([&](container::string_ref,
                              const IndexValue& index) {
          const Entry& entry = *index.entry;
          snapshot.push(EntryView{entry.key, entry.creator, entry.priority,
                                  entry.help});
        });
      }
    }
    if (became_frozen) {
      visit_frozen(visitor,
                   frozen_index_.load(std::memory_order_relaxed));
      return Status::Ok();
    }
    for (std::size_t i = 0; i < snapshot.size(); ++i) {
      if (!invoke_visitor(visitor, snapshot[i])) break;
    }
    return Status::Ok();
  }

  template <class Fn>
  [[nodiscard]] Status for_each_key(Fn&& fn) const {
    auto&& visitor = fn;
    return for_each([&](EntryView entry) -> decltype(auto) {
      return visitor(entry.key);
    });
  }

 private:
  struct Entry {
    container::string_ref key;
    Creator creator;
    RegistryPriority priority;
    container::string_ref help;
  };

  // The same value shape serves startup mutation and frozen reads. Creator is
  // inline on the lookup path; Entry retains cold priority/help metadata.
  struct IndexValue {
    Entry* entry;
    Creator creator;
  };
  static_assert(std::is_trivially_copyable_v<IndexValue>);

  using Table = container::FlatHashMap<
      container::string_ref, IndexValue, container::Hash<container::string_ref>,
      std::equal_to<>, mem::Allocator>;

  class SnapshotStorage {
   public:
    explicit SnapshotStorage(mem::Allocator alloc) noexcept : alloc_(alloc) {}
    ~SnapshotStorage() {
      if (data_ != nullptr) {
        alloc_.deallocate(data_, capacity_ * sizeof(EntryView),
                          alignof(EntryView));
      }
    }
    HYLE_DISABLE_COPY_AND_MOVE(SnapshotStorage);

    bool allocate(std::size_t count) noexcept {
      capacity_ = count;
      if (count == 0) return true;
      if (count > (std::numeric_limits<std::size_t>::max)() /
                      sizeof(EntryView)) {
        return false;
      }
      data_ = static_cast<EntryView*>(
          alloc_.allocate(count * sizeof(EntryView), alignof(EntryView)));
      return data_ != nullptr;
    }
    void push(EntryView entry) noexcept {
      HYLE_FATAL_ASSERT(size_ < capacity_);
      data_[size_++] = entry;
    }
    std::size_t size() const noexcept { return size_; }
    const EntryView& operator[](std::size_t index) const noexcept {
      return data_[index];
    }

   private:
    mem::Allocator alloc_;
    EntryView* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t capacity_ = 0;
  };

  static constexpr bool valid_priority(RegistryPriority priority) noexcept {
    return priority >= RegistryPriority::kFallback &&
           priority <= RegistryPriority::kPreferred;
  }

  static constexpr mem::RegionOptions string_region_options() noexcept {
    mem::RegionOptions options;
    options.initial_block_size = 4 * 1024;
    options.max_block_size = 64 * 1024;
    options.large_allocation_threshold = 16 * 1024;
    return options;
  }

  static bool copy_string(container::string_ref source,
                          container::string_ref* out,
                          mem::Region& region) noexcept {
    if (source.size == 0) {
      *out = container::string_ref{"", 0, true};
      return true;
    }
    const std::size_t bytes = static_cast<std::size_t>(source.size) + 1;
    auto* storage = static_cast<char*>(region.allocate(bytes, 1));
    if (storage == nullptr) return false;
    std::memcpy(storage, source.data, source.size);
    storage[source.size] = '\0';
    *out = container::string_ref{storage, source.size, true};
    return true;
  }

  Entry* allocate_entry(container::string_ref key, Creator creator,
                        RegistryPriority priority,
                        container::string_ref help) noexcept {
    void* storage = metadata_.allocate(sizeof(Entry), alignof(Entry));
    if (storage == nullptr) return nullptr;
    return ::new (storage) Entry{key, creator, priority, help};
  }

  static const Table* frozen_table(const void* index) noexcept {
    return static_cast<const Table*>(index);
  }

  HYLE_FORCE_INLINE static Ptr create_frozen_from(
      const void* index, const container::string_ref& key, Args... args) {
    if (const IndexValue* value = frozen_table(index)->find(key)) {
      return value->creator(std::forward<Args>(args)...);
    }
    return detail::RegistryResultTraits<Ptr>::missing();
  }

  // Kept out of the post-freeze instruction stream. Registry is deliberately
  // init-heavy/read-heavy: after publication, lookup should inline to one
  // acquire, one ordinary Swiss find, and the creator call. The mutex/race
  // handoff remains available here for callers that read during startup.
#if defined(_MSC_VER)
  HYLE_FORCE_INLINE
#else
  HYLE_NOINLINE
#endif
  Ptr create_mutable(container::string_ref key, Args... args) const {
    Creator creator = nullptr;
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (const auto* published_slots =
              frozen_index_.load(std::memory_order_relaxed)) {
        if (const IndexValue* value = published_slots->find(key)) {
          creator = value->creator;
        }
      } else if (const IndexValue* found = entries_.find(key)) {
        creator = found->creator;
      }
    }
    if (creator != nullptr) return creator(std::forward<Args>(args)...);
    return detail::RegistryResultTraits<Ptr>::missing();
  }

  template <class Fn>
  void visit_frozen(Fn& fn, const Table* index) const {
    visit_frozen_table(fn, index);
  }

  template <class Fn>
  static void visit_frozen_table(Fn& fn, const Table* index) {
    bool keep_going = true;
    index->for_each([&](container::string_ref, const IndexValue& value) {
      if (keep_going) {
        const Entry* entry = value.entry;
        keep_going = invoke_visitor(
            fn, EntryView{entry->key, entry->creator, entry->priority,
                          entry->help});
      }
    });
  }

  template <class Fn>
  static bool invoke_visitor(Fn& fn, EntryView entry) {
    if constexpr (std::is_convertible_v<std::invoke_result_t<Fn&, EntryView>,
                                        bool>) {
      return static_cast<bool>(fn(entry));
    } else {
      fn(entry);
      return true;
    }
  }

  mem::Allocator alloc_;
  mutable std::mutex mu_;
  // Lookup touches key bytes but not help/priority metadata. Separate arenas
  // keep short names densely packed instead of interleaving each key with cold
  // diagnostics and Entry objects.
  mem::Region keys_;
  mem::Region metadata_;
  Table entries_;
  mutable std::atomic<const Table*> frozen_index_{nullptr};
};

// Helper whose constructor registers a creator -- the static-init hook the
// REGISTER macros instantiate. DefaultCreator<Derived> news a Derived.
template <class Ptr, class... Args>
struct Registerer {
  Registerer(container::string_ref key, Registry<Ptr, Args...>* registry,
             typename Registry<Ptr, Args...>::Creator creator,
             container::string_ref help = {},
             RegistryPriority priority = RegistryPriority::kDefault) {
    Status status = registry->register_creator(key, creator, priority, help);
    if (!status.ok()) {
      char message[256];
      status.ToString(message, sizeof(message));
      std::fprintf(stderr, "HYLE static registry registration failed: %s\n",
                   message);
      HYLE_FATAL_ASSERT(false);
    }
  }

  template <class Derived>
  static Ptr default_creator(Args... args) {
    return detail::RegistryResultTraits<Ptr>::template default_create<Derived>(
        std::forward<Args>(args)...);
  }
};

}  // namespace hyle::util

// ---- declare/define macros --------------------------------------------------
// A registry is accessed through a function returning Registry*. DECLARE goes in
// a header (dllimport), DEFINE in exactly one .cc (dllexport, Meyers singleton).
// The export/import is explicit (not HYLE_<MOD>_API) because a registry is used
// across downstream shared libs -- same rationale as clay.

#define HYLE_DECLARE_REGISTRY(ExportAPI, RegistryName, Ptr, ...)              \
  ExportAPI ::hyle::util::Registry<Ptr __VA_OPT__(,) __VA_ARGS__>*            \
  RegistryName();                                                              \
  using Registerer##RegistryName =                                             \
      ::hyle::util::Registerer<Ptr __VA_OPT__(,) __VA_ARGS__>

// Meyers function-local static (conventions §6: drop clay's leaky `new`). Thread-
// safe init since C++11; destroyed at exit (frees the indexes and bulk-owned
// entry/string region).
#define HYLE_DEFINE_REGISTRY(RegistryName, Ptr, ...)                          \
  ::hyle::util::Registry<Ptr __VA_OPT__(,) __VA_ARGS__>* RegistryName() {      \
    static ::hyle::util::Registry<Ptr __VA_OPT__(,) __VA_ARGS__> instance;     \
    return &instance;                                                         \
  }

// Register a stateless creator function under a stringized key.
#define HYLE_REGISTER_CREATOR(RegistryName, key, ...)                         \
  static Registerer##RegistryName HYLE_CONCAT(g_##RegistryName##_, __LINE__)( \
      #key, RegistryName(), __VA_ARGS__)

// Register class `Derived` under a stringized key, with type_name as help.
#define HYLE_REGISTER_CLASS(RegistryName, key, Derived)                       \
  static Registerer##RegistryName HYLE_CONCAT(g_##RegistryName##_, __LINE__)( \
      #key, RegistryName(),                                                   \
      Registerer##RegistryName::default_creator<Derived>,                     \
      ::hyle::util::type_name<Derived>())

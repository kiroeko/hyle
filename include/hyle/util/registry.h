#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/container/flat_hash_map.h"
#include "hyle/container/string_view.h"
#include "hyle/error/result.h"
#include "hyle/error/status.h"
#include "hyle/core/compiler.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/region.h"

// Explicit startup registry: map a name -> a factory so an owner can compose
// compiled backends and create one without a compile-time dependency on its
// concrete type. Registration is deliberately explicit: static-init macros are
// unreliable in archives (the TU can be dead-stripped) and cannot propagate a
// recoverable startup error to the application.
//
// Divergences from clay (deliberate, per conventions §1/§9):
//  - mutable storage is a compact HYLE flat index, NOT std::unordered_map
//    (clay used THREE of them). §1 铁律: no std hash containers in production.
//  - Creator is a noexcept FUNCTION POINTER, not std::function. Entry is
//    trivially copyable and no type-erasure allocation/lifetime is added.
//    Fallible factories express failure in Ptr (normally Result<T>).
//  - registration returns Status so startup wiring can recover from invalid
//    configuration, collision, or allocation failure.
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
};

template <class Value>
struct RegistryResultTraits<Result<Value>> {
  static Result<Value> missing() {
    return make_unexpected(Status::NotFound("registry key is not registered"));
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
  // Factory code is part of HYLE's no-throw boundary. Operational failure is a
  // value (normally Result<T>); captured state belongs in explicit Args/owners.
  using Creator = Ptr (*)(Args...) noexcept;

  struct EntryView {
    container::StringView key;
    Creator creator;
    RegistryPriority priority;
    container::StringView help;
  };

  // A non-owning handle to the registry's immutable phase. Cache this handle
  // in dispatch-heavy consumers after freeze(): unlike Registry::create(), it
  // does not need to re-check the lifecycle publication atomic on every read.
  // The originating Registry must outlive the view and must not be destroyed
  // while any view operation is in flight.
  class FrozenView {
   public:
    HYLE_FORCE_INLINE Ptr create(const container::StringView& key,
                                 Args... args) const {
      return Registry::create_frozen_from(index_, key,
                                          std::forward<Args>(args)...);
    }

    bool has(const container::StringView& key) const noexcept {
      return Registry::frozen_table(index_)->find(key) != nullptr;
    }

    container::StringView help(
        const container::StringView& key) const noexcept {
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
      : alloc_(alloc), storage_(alloc, storage_region_options()),
        entries_(alloc) {}
  ~Registry() = default;
  Registry(const Registry&) = delete;
  Registry& operator=(const Registry&) = delete;
  Registry(Registry&&) = delete;
  Registry& operator=(Registry&&) = delete;

  // Register `creator` under `key`. Higher priority wins; on equal priority for
  // the same key returns AlreadyExists. Lower priority is a successful no-op.
  // `help` is optional descriptive text. Key/help are copied before return.
  // Registration after freeze() returns FailedPrecondition.
  [[nodiscard]] Status register_creator(
      container::StringView key, Creator creator,
      RegistryPriority priority = RegistryPriority::kDefault,
      container::StringView help = {}) noexcept {
    if (key.data() == nullptr || key.empty()) {
      return Status::InvalidArgument("registry key is null or empty");
    }
    if (help.data() == nullptr && !help.empty()) {
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
        container::StringView owned_help;
        if (!copy_string(help, &owned_help)) {
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
      // Prepare the only independently allocating structure first. Once this
      // succeeds, one arena allocation owns Entry+key+help and insertion cannot
      // fail. A rejected registration therefore never leaves unreachable bytes
      // in the monotonic registry storage.
      if (!entries_.try_prepare_insert()) {
        return Status::ResourceExhausted("registry table allocation failed");
      }
      Entry* entry = allocate_entry(key, creator, priority, help);
      if (entry == nullptr) {
        return Status::ResourceExhausted(
            "registry entry storage allocation failed");
      }
      using InsertResult = typename Table::InsertResult;
      const InsertResult result = entries_.try_insert_or_assign(
          entry->key, IndexValue{entry, creator});
      HYLE_FATAL_ASSERT(result == InsertResult::kInserted);
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
    return entries_.try_reserve(count)
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

  bool has(container::StringView key) const noexcept {
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
  HYLE_FORCE_INLINE Ptr create(container::StringView key, Args... args) const {
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
  container::StringView help(container::StringView key) const noexcept {
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
        entries_.for_each([&](container::StringView,
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
    container::StringView key;
    Creator creator;
    RegistryPriority priority;
    container::StringView help;
  };
  static_assert(std::is_trivially_destructible_v<Entry>);

  // The same value shape serves startup mutation and frozen reads. Creator is
  // inline on the lookup path; Entry retains cold priority/help metadata.
  struct IndexValue {
    Entry* entry;
    Creator creator;
  };
  static_assert(std::is_trivially_copyable_v<IndexValue>);

  using Table = container::FlatHashMap<
      container::StringView, IndexValue, container::Hash<container::StringView>,
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
    SnapshotStorage(const SnapshotStorage&) = delete;
    SnapshotStorage& operator=(const SnapshotStorage&) = delete;
    SnapshotStorage(SnapshotStorage&&) = delete;
    SnapshotStorage& operator=(SnapshotStorage&&) = delete;

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

  static constexpr mem::RegionOptions storage_region_options() noexcept {
    mem::RegionOptions options;
    options.initial_block_size = 4 * 1024;
    options.max_block_size = 64 * 1024;
    options.large_allocation_threshold = 16 * 1024;
    return options;
  }

  bool copy_string(container::StringView source,
                   container::StringView* out) noexcept {
    if (source.empty()) {
      *out = container::StringView{"", 0, true};
      return true;
    }
    std::size_t bytes = static_cast<std::size_t>(source.size());
    if (bytes == (std::numeric_limits<std::size_t>::max)()) return false;
    ++bytes;
    auto* storage = static_cast<char*>(storage_.allocate(bytes, 1));
    if (storage == nullptr) return false;
    std::memcpy(storage, source.data(), source.size());
    storage[source.size()] = '\0';
    *out = container::StringView{storage, source.size(), true};
    return true;
  }

  Entry* allocate_entry(container::StringView key, Creator creator,
                        RegistryPriority priority,
                        container::StringView help) noexcept {
    const std::size_t maximum = (std::numeric_limits<std::size_t>::max)();
    std::size_t key_bytes = static_cast<std::size_t>(key.size());
    std::size_t help_bytes = static_cast<std::size_t>(help.size());
    if (key_bytes == maximum || (help_bytes != 0 && help_bytes == maximum)) {
      return nullptr;
    }
    ++key_bytes;  // owned keys are always NUL-terminated
    if (help_bytes != 0) ++help_bytes;

    std::size_t total = sizeof(Entry);
    if (key_bytes > maximum - total) return nullptr;
    total += key_bytes;
    if (help_bytes > maximum - total) return nullptr;
    total += help_bytes;

    void* allocation = storage_.allocate(total, alignof(Entry));
    if (allocation == nullptr) return nullptr;
    auto* cursor = static_cast<char*>(allocation) + sizeof(Entry);
    std::memcpy(cursor, key.data(), key.size());
    cursor[key.size()] = '\0';
    const container::StringView owned_key{cursor, key.size(), true};
    cursor += key_bytes;

    container::StringView owned_help{"", 0, true};
    if (!help.empty()) {
      std::memcpy(cursor, help.data(), help.size());
      cursor[help.size()] = '\0';
      owned_help = container::StringView{cursor, help.size(), true};
    }
    return ::new (allocation)
        Entry{owned_key, creator, priority, owned_help};
  }

  static const Table* frozen_table(const void* index) noexcept {
    return static_cast<const Table*>(index);
  }

  HYLE_FORCE_INLINE static Ptr create_frozen_from(
      const void* index, const container::StringView& key, Args... args) {
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
  Ptr create_mutable(container::StringView key, Args... args) const {
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
    index->for_each([&](container::StringView, const IndexValue& value) {
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
  // Each initial registration owns Entry+key+help in one allocation. This keeps
  // failure transactional and the normal 4--32-entry startup working set in one
  // small region instead of eagerly reserving two independent 4 KiB arenas.
  // Replacement help remains append-only so previously returned views stay
  // valid until Registry destruction.
  mem::Region storage_;
  Table entries_;
  mutable std::atomic<const Table*> frozen_index_{nullptr};
};

}  // namespace hyle::util

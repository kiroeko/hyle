#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>

#include "hyle/container/flat_hash_map.h"
#include "hyle/container/string_view.h"
#include "hyle/core/compiler.h"
#include "hyle/error/result.h"
#include "hyle/error/status.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/region.h"

namespace hyle::util {

namespace detail {

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
  kFallback = 1,
  kDefault = 2,
  kPreferred = 3,
};

// Explicit startup registration followed by optional immutable publication.
// Owns accepted callables and key/help bytes. Every accepted entry stays alive
// until destruction, including entries superseded by a higher priority. This
// keeps cached creators and in-flight calls valid without per-call refcounts.
//
// Thread-safe registration and reads; factories/visitors run outside the mutex
// and may re-enter. Captured shared state needs its own synchronization. The
// allocator must outlive the Registry and support concurrent snapshot release
// during enumeration, or the caller must serialize those operations. Allocator
// callbacks and callable construction/destruction must not re-enter Registry.
//
// Destruction requires quiescence of all operations and borrowed
// views/creators. Registry does not own returned objects, stop threads or
// unload libraries. A factory and its returned objects must each retain any
// code image they need.
template <class Ptr, class... Args>
class Registry {
 private:
  struct Entry;
  using Table = container::FlatHashMap<container::StringView, Entry*,
                                       container::Hash<container::StringView>,
                                       std::equal_to<>, mem::AllocatorRef>;

 public:
  // Borrowed callable identity, not an owning function or a raw function
  // address. Copies remain valid across replacement but never outlive the
  // Registry.
  class Creator {
   public:
    Creator() noexcept = default;
    explicit operator bool() const noexcept { return entry_ != nullptr; }
    bool operator==(const Creator&) const noexcept = default;

    Ptr operator()(Args... args) const noexcept {
      HYLE_FATAL_ASSERT(entry_ != nullptr);
      return entry_->invoke(entry_, std::forward<Args>(args)...);
    }

   private:
    friend class Registry;
    explicit Creator(const Entry* entry) noexcept : entry_(entry) {}
    const Entry* entry_ = nullptr;
  };

  // Value snapshot: priority is copied, key/help/creator borrow immutable
  // state.
  struct EntryView {
    container::StringView key;
    Creator creator;
    RegistryPriority priority;
    container::StringView help;
  };

  // Non-owning published table. Transfer this view to another thread using the
  // caller's normal synchronization. Registry must outlive every operation.
  class FrozenView {
   public:
    Ptr create(container::StringView key, Args... args) const {
      if (const auto entry = find(key)) {
        return entry->creator(std::forward<Args>(args)...);
      }
      return detail::RegistryResultTraits<Ptr>::missing();
    }
    bool has(container::StringView key) const noexcept {
      return index_->find(key) != nullptr;
    }
    std::optional<EntryView> find(container::StringView key) const noexcept {
      return Registry::entry_view(index_->find(key));
    }
    container::StringView help(container::StringView key) const noexcept {
      const auto entry = find(key);
      return entry ? entry->help : container::StringView{};
    }
    std::size_t size() const noexcept { return index_->size(); }

    template <class Fn>
    void for_each(Fn&& fn) const {
      Registry::visit_frozen(fn, index_);
    }

   private:
    friend class Registry;
    explicit FrozenView(const Table* index) noexcept : index_(index) {}
    const Table* index_;
  };

  explicit Registry(mem::AllocatorRef alloc = mem::system_allocator())
      : alloc_(alloc), storage_(alloc, storage_options()), entries_(alloc) {}
  ~Registry() {
    // No lock is held while user-owned captures are destroyed. As with other
    // owning containers, destruction is not a supported re-entry phase.
    while (owned_ != nullptr) {
      Entry* entry = owned_;
      owned_ = entry->next;
      entry->destroy(entry);
    }
  }
  Registry(const Registry&) = delete;
  Registry& operator=(const Registry&) = delete;
  Registry(Registry&&) = delete;
  Registry& operator=(Registry&&) = delete;

  // Accepts ordinary noexcept function pointers and owning, const-invocable
  // callables (including move-only captures). No user exception crosses this
  // boundary: transfer, destruction and invocation must all be non-throwing.
  //
  // Higher priority replaces; equal returns AlreadyExists; lower is an OK
  // no-op. Errors/no-op do not move from a named rvalue callable. All fallible
  // storage preparation precedes transfer. A failed insertion may retain index
  // capacity, but exposes no entry and retains no callable or unreachable
  // bytes. After freeze, registration returns FailedPrecondition.
  template <class Fn>
    requires(
        std::is_nothrow_constructible_v<std::decay_t<Fn>, Fn &&> &&
        std::is_nothrow_destructible_v<std::decay_t<Fn>> &&
        std::is_nothrow_invocable_r_v<Ptr, const std::decay_t<Fn>&, Args...>)
  [[nodiscard]] Status register_creator(
      container::StringView key, Fn&& creator,
      RegistryPriority priority = RegistryPriority::kDefault,
      container::StringView help = {}) noexcept {
    if (key.data() == nullptr || key.empty()) {
      return Status::InvalidArgument("registry key is null or empty");
    }
    if (help.data() == nullptr && !help.empty()) {
      return Status::InvalidArgument("registry help has null data");
    }
    if constexpr (std::is_pointer_v<std::remove_reference_t<Fn>>) {
      if (creator == nullptr) {
        return Status::InvalidArgument("registry creator is null");
      }
    }
    if (priority < RegistryPriority::kFallback ||
        priority > RegistryPriority::kPreferred) {
      return Status::InvalidArgument("registry priority is invalid");
    }

    std::lock_guard lock(mu_);
    if (frozen_index_.load(std::memory_order_relaxed) != nullptr) {
      return Status::FailedPrecondition("registry is frozen");
    }
    Entry** existing = entries_.find(key);
    if (existing != nullptr) {
      if (priority == (*existing)->priority) {
        return Status::AlreadyExists(
            "registry key already has the same priority");
      }
      if (priority < (*existing)->priority) return Status::Ok();
    } else if (!entries_.try_prepare_insert()) {
      return Status::ResourceExhausted("registry table allocation failed");
    }

    Entry* entry =
        allocate_entry(key, std::forward<Fn>(creator), priority, help);
    if (entry == nullptr) {
      return Status::ResourceExhausted(
          "registry entry storage allocation failed");
    }
    if (existing != nullptr) {
      *existing = entry;
    } else {
      const auto result = entries_.try_insert_or_assign(entry->key, entry);
      HYLE_FATAL_ASSERT(result == Table::InsertResult::kInserted);
    }
    owned_ = entry;
    return Status::Ok();
  }

  [[nodiscard]] Status register_creator(
      container::StringView, std::nullptr_t,
      RegistryPriority = RegistryPriority::kDefault,
      container::StringView = {}) noexcept {
    return Status::InvalidArgument("registry creator is null");
  }

  [[nodiscard]] Status reserve(std::size_t count) noexcept {
    std::lock_guard lock(mu_);
    if (frozen_index_.load(std::memory_order_relaxed) != nullptr) {
      return Status::FailedPrecondition("registry is frozen");
    }
    return entries_.try_reserve(count)
               ? Status::Ok()
               : Status::ResourceExhausted("registry reserve failed");
  }

  // One-way, idempotent O(1) publication of the same index; zero allocation.
  // No subsequent operation mutates entries_ or any accepted entry.
  [[nodiscard]] FrozenView freeze() noexcept {
    std::lock_guard lock(mu_);
    frozen_index_.store(&entries_, std::memory_order_release);
    return FrozenView{&entries_};
  }
  bool frozen() const noexcept {
    return frozen_index_.load(std::memory_order_acquire) != nullptr;
  }

  std::optional<EntryView> find(container::StringView key) const noexcept {
    if (const auto* index = frozen_index_.load(std::memory_order_acquire)) {
      return entry_view(index->find(key));
    }
    std::lock_guard lock(mu_);
    // Whether or not freeze won the race, this mutex observes the same table.
    return entry_view(entries_.find(key));
  }
  bool has(container::StringView key) const noexcept {
    return find(key).has_value();
  }

  // Nullable results use their default empty value for missing; Result<T>
  // returns NotFound and otherwise preserves the creator's error unchanged.
  Ptr create(container::StringView key, Args... args) const {
    const auto entry =
        find(key);  // releases the startup lock before invocation
    if (entry) return entry->creator(std::forward<Args>(args)...);
    return detail::RegistryResultTraits<Ptr>::missing();
  }
  container::StringView help(container::StringView key) const noexcept {
    const auto entry = find(key);
    return entry ? entry->help : container::StringView{};
  }
  std::size_t size() const noexcept {
    if (const auto* index = frozen_index_.load(std::memory_order_acquire)) {
      return index->size();
    }
    std::lock_guard lock(mu_);
    return entries_.size();
  }

  // Unspecified order; bool visitors may stop callback delivery early. Mutable
  // enumeration allocates one point-in-time snapshot; frozen enumeration does
  // not allocate. Callback exceptions propagate with snapshot cleanup.
  template <class Fn>
  [[nodiscard]] Status for_each(Fn&& fn) const {
    if (const auto* index = frozen_index_.load(std::memory_order_acquire)) {
      visit_frozen(fn, index);
      return Status::Ok();
    }
    SnapshotStorage snapshot(alloc_);
    const Table* published = nullptr;
    {
      std::lock_guard lock(mu_);
      published = frozen_index_.load(std::memory_order_relaxed);
      if (published == nullptr) {
        if (!snapshot.allocate(entries_.size())) {
          return Status::ResourceExhausted(
              "registry enumeration snapshot allocation failed");
        }
        entries_.for_each([&](container::StringView, Entry* entry) {
          snapshot.push(view_of(entry));
        });
      }
    }
    if (published != nullptr) {
      visit_frozen(fn, published);
    } else {
      for (std::size_t i = 0; i < snapshot.size(); ++i) {
        if (!invoke_visitor(fn, snapshot[i])) break;
      }
    }
    return Status::Ok();
  }

  template <class Fn>
  [[nodiscard]] Status for_each_key(Fn&& fn) const {
    return for_each(
        [&](EntryView entry) -> decltype(auto) { return fn(entry.key); });
  }

 private:
  struct Entry {
    container::StringView key;
    container::StringView help;
    RegistryPriority priority;
    Entry* next;
    Ptr (*invoke)(const Entry*, Args...) noexcept;
    void (*destroy)(Entry*) noexcept;
  };

  template <class Fn>
  struct StoredEntry : Entry {
    Fn factory;
  };

  static EntryView view_of(const Entry* entry) noexcept {
    return {entry->key, Creator{entry}, entry->priority, entry->help};
  }
  static std::optional<EntryView> entry_view(Entry* const* value) noexcept {
    if (value == nullptr) return std::nullopt;
    return view_of(*value);
  }

  class SnapshotStorage {
   public:
    explicit SnapshotStorage(mem::AllocatorRef alloc) noexcept
        : alloc_(alloc) {}
    ~SnapshotStorage() {
      if (data_ != nullptr) {
        alloc_.deallocate(data_, capacity_ * sizeof(EntryView),
                          alignof(EntryView));
      }
    }
    SnapshotStorage(const SnapshotStorage&) = delete;
    SnapshotStorage& operator=(const SnapshotStorage&) = delete;
    bool allocate(std::size_t count) noexcept {
      if (count == 0) return true;
      if (count >
          (std::numeric_limits<std::size_t>::max)() / sizeof(EntryView)) {
        return false;
      }
      capacity_ = count;
      data_ = static_cast<EntryView*>(
          alloc_.allocate(count * sizeof(EntryView), alignof(EntryView)));
      return data_ != nullptr;
    }
    void push(EntryView entry) noexcept {
      HYLE_FATAL_ASSERT(size_ < capacity_);
      ::new (data_ + size_++) EntryView(entry);
    }
    std::size_t size() const noexcept { return size_; }
    const EntryView& operator[](std::size_t index) const noexcept {
      return data_[index];
    }

   private:
    mem::AllocatorRef alloc_;
    EntryView* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t capacity_ = 0;
  };

  static constexpr mem::ArenaOptions storage_options() noexcept {
    mem::ArenaOptions options;
    options.initial_block_size = 4 * 1024;
    options.max_block_size = 64 * 1024;
    options.large_allocation_threshold = 16 * 1024;
    return options;
  }

  template <class Fn>
  Entry* allocate_entry(container::StringView key, Fn&& creator,
                        RegistryPriority priority,
                        container::StringView help) noexcept {
    using Stored = StoredEntry<std::decay_t<Fn>>;
    const auto maximum = (std::numeric_limits<std::size_t>::max)();
    const std::size_t key_size = key.size();
    const std::size_t help_size = help.size();
    if (key_size == maximum || help_size == maximum) return nullptr;
    const std::size_t key_bytes = key_size + 1;
    const std::size_t help_bytes = help_size + 1;
    if (key_bytes > maximum - sizeof(Stored) ||
        help_bytes > maximum - sizeof(Stored) - key_bytes)
      return nullptr;
    void* allocation = storage_.allocate(
        sizeof(Stored) + key_bytes + help_bytes, alignof(Stored));
    if (allocation == nullptr) return nullptr;

    auto* key_data = static_cast<char*>(allocation) + sizeof(Stored);
    std::memcpy(key_data, key.data(), key_size);
    key_data[key_size] = '\0';
    char* help_data = key_data + key_bytes;
    if (help_size != 0) std::memcpy(help_data, help.data(), help_size);
    help_data[help_size] = '\0';

    // From here onward nothing can fail. In particular the callable is not
    // moved before all table/arena storage has been prepared successfully.
    return ::new (allocation) Stored{
        Entry{container::StringView{key_data, key.size(), true},
              container::StringView{help_data, help.size(), true}, priority,
              owned_,
              +[](const Entry* entry, Args... args) noexcept -> Ptr {
                return std::invoke(static_cast<const Stored*>(entry)->factory,
                                   std::forward<Args>(args)...);
              },
              +[](Entry* entry) noexcept {
                static_cast<Stored*>(entry)->~Stored();
              }},
        std::forward<Fn>(creator)};
  }

  template <class Fn>
  static void visit_frozen(Fn& fn, const Table* index) {
    bool keep_going = true;
    index->for_each([&](container::StringView, Entry* entry) {
      if (keep_going) keep_going = invoke_visitor(fn, view_of(entry));
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

  mem::AllocatorRef alloc_;
  mutable std::mutex mu_;
  mem::Arena storage_;
  Table entries_;
  Entry* owned_ = nullptr;
  std::atomic<const Table*> frozen_index_{nullptr};
};

}  // namespace hyle::util

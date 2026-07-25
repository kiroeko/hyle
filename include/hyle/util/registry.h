#pragma once

#include <cstddef>
#include <cstdio>
#include <mutex>
#include <type_traits>
#include <utility>

#include "hyle/container/string_ref.h"
#include "hyle/container/swiss_table.h"
#include "hyle/macros.h"
#include "hyle/mem/allocator.h"
#include "hyle/util/ref_counted.h"
#include "hyle/util/type_name.h"

// Self-registration registry (clay registry.h): map a name -> a factory, so
// plugins register themselves at static-init time and code creates objects by
// name without a compile-time dependency on the concrete types. Used for
// backend/plugin selection (conventions §9).
//
// Divergences from clay (deliberate, per conventions §1/§9):
//  - storage is HYLE SwissTable, NOT std::unordered_map (clay used THREE of
//    them). §1 铁律: no std hash containers in production.
//  - Creator is a FUNCTION POINTER, not std::function. So Entry is trivially
//    copyable and lives INLINE in the table (no heap, no type-erasure alloc),
//    and it is ABI-friendlier (§4). Registry creators are stateless type
//    constructors (`new Derived(args...)`) -- they never need capture.
//  - a same-priority collision is a wiring BUG -> HYLE_FATAL_ASSERT (loud abort
//    with file/line), replacing clay's std::exit(1)/throw (§0 fail-fast, §5
//    no-throw; "禁 std::exit").
//  - help text uses type_name<T>() (util-02), not clay's demangle.
//
// Keys are string_ref. The REGISTER macros stringize the key (#name), so it is a
// string literal with static lifetime -- safe to store as a view. A dynamic key
// must outlive the registry (contract).
//
// Threading: every table access is synchronized. create() copies the selected
// function pointer under the lock and invokes it after unlocking, so a creator
// may safely re-enter this registry. for_each_key() deliberately invokes the
// visitor under the lock to avoid allocating a snapshot; its visitor therefore
// must not call back into this registry.

namespace hyle::util {

enum class RegistryPriority : std::uint8_t {
  kFallback = 1,  // lowest: a default that anything else overrides
  kDefault = 2,
  kPreferred = 3,  // highest: explicitly preferred implementation
};

template <class Ptr, class... Args>
class Registry {
 public:
  using Creator = Ptr (*)(Args...);  // function pointer (no capture, no alloc)

  explicit Registry(mem::Allocator alloc = mem::system_allocator())
      : entries_(alloc) {}
  HYLE_DISABLE_COPY_AND_MOVE(Registry);

  // Register `creator` under `key`. Higher priority wins; on equal priority for
  // the same key it fail-fasts (two implementations claiming the same slot is a
  // bug). `help` is optional descriptive text (e.g. the concrete type name).
  void register_creator(container::string_ref key, Creator creator,
                        RegistryPriority priority = RegistryPriority::kDefault,
                        container::string_ref help = {}) {
    std::lock_guard<std::mutex> lock(mu_);
    HYLE_FATAL_ASSERT(key.data != nullptr || key.size == 0);
    HYLE_FATAL_ASSERT(help.data != nullptr || help.size == 0);
    HYLE_FATAL_ASSERT(creator != nullptr);
    HYLE_FATAL_ASSERT(priority >= RegistryPriority::kFallback &&
                      priority <= RegistryPriority::kPreferred);
    if (Entry* e = entries_.find(key)) {
      if (priority > e->priority) {
        *e = Entry{creator, priority, help};  // higher priority overrides
      } else if (priority == e->priority) {
        std::fprintf(stderr, "HYLE registry: key '%.*s' registered twice at "
                             "the same priority\n",
                     static_cast<int>(key.size), key.data);
        HYLE_FATAL_ASSERT(false);  // wiring bug -> abort (was clay std::exit)
      }
      // lower priority: keep the existing higher-priority creator (no-op)
    } else {
      entries_.insert(key, Entry{creator, priority, help});
    }
  }

  bool has(container::string_ref key) const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    return entries_.find(key) != nullptr;
  }

  // Create the object registered under `key`, or a default-constructed Ptr
  // (e.g. nullptr) if `key` is absent.
  Ptr create(container::string_ref key, Args... args) {
    Creator creator = nullptr;
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (Entry* e = entries_.find(key)) creator = e->creator;
    }
    if (creator != nullptr) {
      return creator(std::forward<Args>(args)...);
    }
    return Ptr{};
  }

  // Help text for `key` (empty if absent or none given).
  container::string_ref help(container::string_ref key) const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    if (Entry* e = entries_.find(key)) return e->help;
    return {};
  }

  std::size_t size() const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    return entries_.size();
  }

  // Visit every (key, creator-entry). Order unspecified.
  template <class Fn>
  void for_each_key(Fn&& fn) {
    std::lock_guard<std::mutex> lock(mu_);
    entries_.for_each(
        [&](container::string_ref k, Entry&) { fn(k); });
  }

 private:
  struct Entry {  // trivially copyable -> stored inline in the SwissTable
    Creator creator;
    RegistryPriority priority;
    container::string_ref help;
  };

  mutable std::mutex mu_;
  mutable container::SwissTable<container::string_ref, Entry,
                                container::Hash<container::string_ref>,
                                std::equal_to<container::string_ref>,
                                mem::Allocator>
      entries_;
};

// Helper whose constructor registers a creator -- the static-init hook the
// REGISTER macros instantiate. DefaultCreator<Derived> news a Derived.
template <class Ptr, class... Args>
struct Registerer {
  Registerer(container::string_ref key, Registry<Ptr, Args...>* registry,
             typename Registry<Ptr, Args...>::Creator creator,
             container::string_ref help = {},
             RegistryPriority priority = RegistryPriority::kDefault) {
    registry->register_creator(key, creator, priority, help);
  }

  template <class Derived>
  static Ptr default_creator(Args... args) {
    if constexpr (std::is_pointer_v<Ptr>) {
      return new Derived(std::forward<Args>(args)...);
    } else if constexpr (requires {
                           Ptr{new Derived(std::forward<Args>(args)...)};
                         }) {
      return Ptr{new Derived(std::forward<Args>(args)...)};
    } else {
      static_assert(
          std::convertible_to<Ref<Derived>, Ptr>,
          "Registry Ptr must own a Derived* or accept Ref<Derived>");
      return make_ref<Derived>(std::forward<Args>(args)...);
    }
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
// safe init since C++11; destroyed at exit (frees the SwissTable; entries are
// trivially copyable so nothing per-entry to clean up).
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

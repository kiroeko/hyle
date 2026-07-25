#include "hyle/c/util.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <type_traits>

#include "hyle/container/string_ref.h"
#include "hyle/container/swiss_table.h"
#include "hyle/mem/allocator.h"
#include "hyle/util/demangle.h"
#include "hyle/util/random.h"
#include "hyle/util/registry.h"
#include "hyle/util/spin_lock.h"

struct hyle_util_ref_o {
  std::atomic<std::uint32_t> count{1};
  void* object = nullptr;
  hyle_util_ref_destroy_fn destroy_object = nullptr;
  void* destroy_ctx = nullptr;
};

struct hyle_util_spin_lock_o {
  hyle::util::SpinLock lock;
};

struct hyle_util_scope_guard_o {
  hyle_util_scope_guard_fn callback;
  void* ctx;
  bool active;
};

struct CRegistryEntry {
  hyle_util_registry_create_fn creator;
  void* creator_ctx;
  hyle::container::string_ref help;
  hyle_util_registry_priority priority;
};

struct hyle_util_registry_o {
  mutable std::mutex mutex;
  mutable hyle::container::SwissTable<
      hyle::container::string_ref, CRegistryEntry,
      hyle::container::Hash<hyle::container::string_ref>,
      std::equal_to<hyle::container::string_ref>, hyle::mem::Allocator>
      entries{hyle::mem::system_allocator()};

  ~hyle_util_registry_o() {
    entries.for_each([](hyle::container::string_ref key,
                        CRegistryEntry& entry) {
      delete[] key.data;
      delete[] entry.help.data;
    });
  }
};

namespace {

thread_local char g_last_error[256] = {};

void clear_error() noexcept { g_last_error[0] = '\0'; }

void set_error(const char* message) noexcept {
  std::snprintf(g_last_error, sizeof(g_last_error), "%s", message);
}

char* copy_string(const char* source) noexcept {
  if (source == nullptr) source = "";
  const std::size_t length = std::strlen(source);
  if (length == std::numeric_limits<std::size_t>::max()) return nullptr;
  auto* copy = new (std::nothrow) char[length + 1];
  if (copy == nullptr) return nullptr;
  std::memcpy(copy, source, length + 1);
  return copy;
}

std::size_t fill_string(const char* source, char* out,
                        std::size_t capacity) noexcept {
  const std::size_t length = std::strlen(source);
  if (capacity != 0) {
    const std::size_t copied =
        length < capacity - 1 ? length : capacity - 1;
    std::memcpy(out, source, copied);
    out[copied] = '\0';
  }
  return length;
}

bool valid_c_state(const hyle_util_rng* rng) noexcept {
  return rng != nullptr &&
         (rng->state[0] | rng->state[1] | rng->state[2] | rng->state[3]) != 0;
}

bool load_rng(const hyle_util_rng* source, hyle::util::Rng& destination) noexcept {
  if (source == nullptr) {
    set_error("rng is NULL");
    return false;
  }
  hyle::util::RngState state{{
      source->state[0],
      source->state[1],
      source->state[2],
      source->state[3],
  }};
  if (!hyle::util::Rng::valid_state(state)) {
    set_error("rng has the invalid all-zero state; call hyle_util_rng_seed");
    return false;
  }
  destination.set_state(state);
  return true;
}

void store_rng(const hyle::util::Rng& source,
               hyle_util_rng* destination) noexcept {
  const hyle::util::RngState state = source.state();
  for (int lane = 0; lane < 4; ++lane) {
    destination->state[lane] = state.words[lane];
  }
}

}  // namespace

static_assert(std::is_standard_layout_v<hyle_util_rng>);
static_assert(std::is_trivially_copyable_v<hyle_util_rng>);
static_assert(sizeof(hyle_util_rng) == sizeof(hyle::util::RngState));
static_assert(alignof(hyle_util_rng) == alignof(hyle::util::RngState));
static_assert(std::is_standard_layout_v<CRegistryEntry>);
static_assert(std::is_trivially_copyable_v<CRegistryEntry>);
static_assert(HYLE_UTIL_REGISTRY_FALLBACK ==
              static_cast<int>(hyle::util::RegistryPriority::kFallback));
static_assert(HYLE_UTIL_REGISTRY_DEFAULT ==
              static_cast<int>(hyle::util::RegistryPriority::kDefault));
static_assert(HYLE_UTIL_REGISTRY_PREFERRED ==
              static_cast<int>(hyle::util::RegistryPriority::kPreferred));

extern "C" {

const char* hyle_util_last_error(void) { return g_last_error; }

size_t hyle_util_demangle(const char* mangled, char* out, size_t cap) {
  if (cap != 0 && out == nullptr) {
    set_error("out is NULL while cap is non-zero");
    return 0;
  }
  clear_error();
  return hyle::util::demangle(mangled, out, cap);
}

int hyle_util_rng_seed(hyle_util_rng* rng, uint64_t seed) {
  if (rng == nullptr) {
    set_error("rng is NULL");
    return 0;
  }
  hyle::util::Rng native{seed};
  store_rng(native, rng);
  clear_error();
  return 1;
}

int hyle_util_rng_is_valid(const hyle_util_rng* rng) {
  if (rng == nullptr) {
    set_error("rng is NULL");
    return 0;
  }
  if (!valid_c_state(rng)) {
    set_error("rng has the invalid all-zero state; call hyle_util_rng_seed");
    return 0;
  }
  clear_error();
  return 1;
}

uint64_t hyle_util_rng_next_u64(hyle_util_rng* rng) {
  hyle::util::Rng native;
  if (!load_rng(rng, native)) return 0;
  const uint64_t value = native.next_u64();
  store_rng(native, rng);
  clear_error();
  return value;
}

uint32_t hyle_util_rng_next_u32(hyle_util_rng* rng) {
  hyle::util::Rng native;
  if (!load_rng(rng, native)) return 0;
  const uint32_t value = native.next_u32();
  store_rng(native, rng);
  clear_error();
  return value;
}

int hyle_util_rng_next_bool(hyle_util_rng* rng) {
  hyle::util::Rng native;
  if (!load_rng(rng, native)) return 0;
  const int value = native.next_bool() ? 1 : 0;
  store_rng(native, rng);
  clear_error();
  return value;
}

float hyle_util_rng_next_float(hyle_util_rng* rng) {
  hyle::util::Rng native;
  if (!load_rng(rng, native)) return 0.0f;
  const float value = native.next_float();
  store_rng(native, rng);
  clear_error();
  return value;
}

double hyle_util_rng_next_double(hyle_util_rng* rng) {
  hyle::util::Rng native;
  if (!load_rng(rng, native)) return 0.0;
  const double value = native.next_double();
  store_rng(native, rng);
  clear_error();
  return value;
}

uint64_t hyle_util_rng_range_u64(hyle_util_rng* rng, uint64_t lo,
                                 uint64_t hi) {
  if (lo > hi) {
    set_error("lo is greater than hi");
    return 0;
  }
  hyle::util::Rng native;
  if (!load_rng(rng, native)) return 0;
  const uint64_t value = native.range_u64(lo, hi);
  store_rng(native, rng);
  clear_error();
  return value;
}

uint32_t hyle_util_rng_range_u32(hyle_util_rng* rng, uint32_t lo,
                                 uint32_t hi) {
  if (lo > hi) {
    set_error("lo is greater than hi");
    return 0;
  }
  hyle::util::Rng native;
  if (!load_rng(rng, native)) return 0;
  const uint32_t value = native.range_u32(lo, hi);
  store_rng(native, rng);
  clear_error();
  return value;
}

double hyle_util_rng_range_double(hyle_util_rng* rng, double lo, double hi) {
  if (!std::isfinite(lo) || !std::isfinite(hi) || lo > hi) {
    set_error("floating range requires finite lo <= hi");
    return 0.0;
  }
  hyle::util::Rng native;
  if (!load_rng(rng, native)) return 0.0;
  const double value = native.range_double(lo, hi);
  store_rng(native, rng);
  clear_error();
  return value;
}

int hyle_util_rng_fill_u64(hyle_util_rng* rng, uint64_t* out, size_t count) {
  if (count != 0 && out == nullptr) {
    set_error("out is NULL while count is non-zero");
    return 0;
  }
  hyle::util::Rng native;
  if (!load_rng(rng, native)) return 0;
  for (size_t i = 0; i < count; ++i) out[i] = native.next_u64();
  store_rng(native, rng);
  clear_error();
  return 1;
}

int hyle_util_rng_fill_float(hyle_util_rng* rng, float* out, size_t count) {
  if (count != 0 && out == nullptr) {
    set_error("out is NULL while count is non-zero");
    return 0;
  }
  hyle::util::Rng native;
  if (!load_rng(rng, native)) return 0;
  for (size_t i = 0; i < count; ++i) out[i] = native.next_float();
  store_rng(native, rng);
  clear_error();
  return 1;
}

int hyle_util_rng_fill_double(hyle_util_rng* rng, double* out, size_t count) {
  if (count != 0 && out == nullptr) {
    set_error("out is NULL while count is non-zero");
    return 0;
  }
  hyle::util::Rng native;
  if (!load_rng(rng, native)) return 0;
  for (size_t i = 0; i < count; ++i) out[i] = native.next_double();
  store_rng(native, rng);
  clear_error();
  return 1;
}

int hyle_util_rng_jump(hyle_util_rng* rng) {
  hyle::util::Rng native;
  if (!load_rng(rng, native)) return 0;
  native.jump();
  store_rng(native, rng);
  clear_error();
  return 1;
}

int hyle_util_rng_long_jump(hyle_util_rng* rng) {
  hyle::util::Rng native;
  if (!load_rng(rng, native)) return 0;
  native.long_jump();
  store_rng(native, rng);
  clear_error();
  return 1;
}

int hyle_util_rng_shuffle(hyle_util_rng* rng, uint64_t count,
                          hyle_util_rng_swap_fn swap, void* ctx) {
  if (count > 1 && swap == nullptr) {
    set_error("swap is NULL while count is greater than one");
    return 0;
  }
  hyle::util::Rng native;
  if (!load_rng(rng, native)) return 0;
  native.shuffle(count, [&](uint64_t left, uint64_t right) {
    swap(ctx, left, right);
  });
  store_rng(native, rng);
  clear_error();
  return 1;
}

hyle_util_rng* hyle_util_thread_rng(void) {
  static thread_local hyle_util_rng state = [] {
    // The C POD must own an independent stream: callers mutate it directly, so
    // aliasing the native C++ TLS generator would require unsafe layout casts or
    // surprising two-way synchronization. Draw a seed from the same unique
    // per-process sequence without advancing hyle::util::thread_rng().
    hyle::util::Rng native{hyle::util::detail::next_thread_seed()};
    const hyle::util::RngState snapshot = native.state();
    hyle_util_rng result{};
    for (int lane = 0; lane < 4; ++lane) {
      result.state[lane] = snapshot.words[lane];
    }
    return result;
  }();
  clear_error();
  return &state;
}

hyle_util_ref* hyle_util_ref_create(
    void* object, hyle_util_ref_destroy_fn destroy_object, void* destroy_ctx) {
  auto* ref = new (std::nothrow)
      hyle_util_ref_o{{1}, object, destroy_object, destroy_ctx};
  if (ref == nullptr) {
    set_error("out of memory creating reference handle");
    return nullptr;
  }
  clear_error();
  return ref;
}

hyle_util_ref* hyle_util_ref_retain(hyle_util_ref* ref) {
  if (ref == nullptr) {
    set_error("ref is NULL");
    return nullptr;
  }
  std::uint32_t count = ref->count.load(std::memory_order_relaxed);
  for (;;) {
    if (count == 0 ||
        count == std::numeric_limits<std::uint32_t>::max()) {
      set_error("reference count is zero or would overflow");
      return nullptr;
    }
    if (ref->count.compare_exchange_weak(
            count, count + 1, std::memory_order_relaxed,
            std::memory_order_relaxed)) {
      clear_error();
      return ref;
    }
  }
}

void hyle_util_ref_release(hyle_util_ref* ref) {
  if (ref == nullptr) return;
  const std::uint32_t old =
      ref->count.fetch_sub(1, std::memory_order_release);
  // A zero old count requires use-after-final-release by the caller; the handle
  // memory is already invalid in that case and no ABI can recover from it.
  if (old == 1) {
    std::atomic_thread_fence(std::memory_order_acquire);
    if (ref->destroy_object != nullptr) {
      ref->destroy_object(ref->object, ref->destroy_ctx);
    }
    delete ref;
  }
}

void* hyle_util_ref_get(const hyle_util_ref* ref) {
  return ref != nullptr ? ref->object : nullptr;
}

uint32_t hyle_util_ref_use_count(const hyle_util_ref* ref) {
  return ref != nullptr
             ? ref->count.load(std::memory_order_relaxed)
             : 0;
}

int hyle_util_ref_is_unique(const hyle_util_ref* ref) {
  return ref != nullptr &&
         ref->count.load(std::memory_order_acquire) == 1;
}

hyle_util_scope_guard* hyle_util_scope_guard_create(
    hyle_util_scope_guard_fn callback, void* ctx) {
  if (callback == nullptr) {
    set_error("scope guard callback is NULL");
    return nullptr;
  }
  auto* guard = new (std::nothrow)
      hyle_util_scope_guard_o{callback, ctx, true};
  if (guard == nullptr) {
    set_error("out of memory creating scope guard");
    return nullptr;
  }
  clear_error();
  return guard;
}

int hyle_util_scope_guard_run(hyle_util_scope_guard* guard) {
  if (guard == nullptr) {
    set_error("scope guard is NULL");
    return 0;
  }
  if (guard->active) {
    guard->active = false;
    guard->callback(guard->ctx);
  }
  clear_error();
  return 1;
}

int hyle_util_scope_guard_dismiss(hyle_util_scope_guard* guard) {
  if (guard == nullptr) {
    set_error("scope guard is NULL");
    return 0;
  }
  guard->active = false;
  clear_error();
  return 1;
}

int hyle_util_scope_guard_is_active(const hyle_util_scope_guard* guard) {
  if (guard == nullptr) {
    set_error("scope guard is NULL");
    return 0;
  }
  const bool active = guard->active;
  clear_error();
  return active ? 1 : 0;
}

void hyle_util_scope_guard_destroy(hyle_util_scope_guard* guard) {
  if (guard == nullptr) return;
  if (guard->active) {
    guard->active = false;
    guard->callback(guard->ctx);
  }
  delete guard;
}

hyle_util_registry* hyle_util_registry_create(void) {
  auto* registry = new (std::nothrow) hyle_util_registry_o;
  if (registry == nullptr) {
    set_error("out of memory creating registry");
    return nullptr;
  }
  clear_error();
  return registry;
}

void hyle_util_registry_destroy(hyle_util_registry* registry) {
  delete registry;
}

int hyle_util_registry_register(
    hyle_util_registry* registry, const char* key,
    hyle_util_registry_create_fn creator, void* creator_ctx,
    hyle_util_registry_priority priority, const char* help) {
  if (registry == nullptr) {
    set_error("registry is NULL");
    return 0;
  }
  if (key == nullptr || key[0] == '\0') {
    set_error("registry key is NULL or empty");
    return 0;
  }
  if (creator == nullptr) {
    set_error("registry creator is NULL");
    return 0;
  }
  if (priority < HYLE_UTIL_REGISTRY_FALLBACK ||
      priority > HYLE_UTIL_REGISTRY_PREFERRED) {
    set_error("registry priority is invalid");
    return 0;
  }

  char* key_copy = copy_string(key);
  char* help_copy = copy_string(help);
  if (key_copy == nullptr || help_copy == nullptr) {
    delete[] key_copy;
    delete[] help_copy;
    set_error("out of memory copying registry strings");
    return 0;
  }
  hyle::container::string_ref copied_key{key_copy};
  hyle::container::string_ref copied_help{help_copy};

  std::lock_guard<std::mutex> lock(registry->mutex);
  if (CRegistryEntry* existing = registry->entries.find(copied_key)) {
    delete[] key_copy;
    if (priority == existing->priority) {
      delete[] help_copy;
      set_error("registry key already has the same priority");
      return 0;
    }
    if (priority < existing->priority) {
      delete[] help_copy;
      clear_error();
      return 1;
    }
    delete[] existing->help.data;
    *existing = CRegistryEntry{
        creator, creator_ctx, copied_help, priority};
  } else {
    registry->entries.insert(
        copied_key,
        CRegistryEntry{creator, creator_ctx, copied_help, priority});
  }
  clear_error();
  return 1;
}

int hyle_util_registry_has(const hyle_util_registry* registry,
                           const char* key) {
  if (registry == nullptr || key == nullptr) {
    set_error(registry == nullptr ? "registry is NULL" : "key is NULL");
    return 0;
  }
  std::lock_guard<std::mutex> lock(registry->mutex);
  const bool found =
      registry->entries.find(hyle::container::string_ref{key}) != nullptr;
  clear_error();
  return found ? 1 : 0;
}

void* hyle_util_registry_create_object(
    hyle_util_registry* registry, const char* key, const void* create_args) {
  if (registry == nullptr || key == nullptr) {
    set_error(registry == nullptr ? "registry is NULL" : "key is NULL");
    return nullptr;
  }
  hyle_util_registry_create_fn creator = nullptr;
  void* creator_ctx = nullptr;
  {
    std::lock_guard<std::mutex> lock(registry->mutex);
    if (CRegistryEntry* entry =
            registry->entries.find(hyle::container::string_ref{key})) {
      creator = entry->creator;
      creator_ctx = entry->creator_ctx;
    }
  }
  if (creator == nullptr) {
    set_error("registry key is not registered");
    return nullptr;
  }
  clear_error();
  return creator(creator_ctx, create_args);
}

size_t hyle_util_registry_help(
    const hyle_util_registry* registry, const char* key, char* out,
    size_t cap) {
  if (registry == nullptr || key == nullptr ||
      (cap != 0 && out == nullptr)) {
    if (registry == nullptr) {
      set_error("registry is NULL");
    } else if (key == nullptr) {
      set_error("key is NULL");
    } else {
      set_error("out is NULL while cap is non-zero");
    }
    return 0;
  }
  std::lock_guard<std::mutex> lock(registry->mutex);
  CRegistryEntry* entry =
      registry->entries.find(hyle::container::string_ref{key});
  if (entry == nullptr) {
    set_error("registry key is not registered");
    return 0;
  }
  const size_t length = fill_string(entry->help.data, out, cap);
  clear_error();
  return length;
}

size_t hyle_util_registry_size(const hyle_util_registry* registry) {
  if (registry == nullptr) {
    set_error("registry is NULL");
    return 0;
  }
  std::lock_guard<std::mutex> lock(registry->mutex);
  const size_t size = registry->entries.size();
  clear_error();
  return size;
}

int hyle_util_registry_for_each(
    hyle_util_registry* registry, hyle_util_registry_visit_fn visit,
    void* visit_ctx) {
  if (registry == nullptr || visit == nullptr) {
    set_error(registry == nullptr ? "registry is NULL" : "visit is NULL");
    return 0;
  }
  std::lock_guard<std::mutex> lock(registry->mutex);
  bool keep_going = true;
  registry->entries.for_each(
      [&](hyle::container::string_ref key, CRegistryEntry& entry) {
        if (keep_going) {
          keep_going =
              visit(visit_ctx, key.data, entry.help.data, entry.priority) != 0;
        }
      });
  clear_error();
  return 1;
}

hyle_util_spin_lock* hyle_util_spin_lock_create(void) {
  auto* lock = new (std::nothrow) hyle_util_spin_lock_o;
  if (lock == nullptr) {
    set_error("out of memory creating spin lock");
    return nullptr;
  }
  clear_error();
  return lock;
}

int hyle_util_spin_lock_destroy(hyle_util_spin_lock* lock) {
  if (lock == nullptr) {
    clear_error();
    return 1;
  }
  if (lock->lock.is_locked()) {
    set_error("cannot destroy a locked spin lock");
    return 0;
  }
  delete lock;
  clear_error();
  return 1;
}

int hyle_util_spin_lock_lock(hyle_util_spin_lock* lock) {
  if (lock == nullptr) {
    set_error("spin lock is NULL");
    return 0;
  }
  lock->lock.lock();
  clear_error();
  return 1;
}

int hyle_util_spin_lock_try_lock(hyle_util_spin_lock* lock) {
  if (lock == nullptr) {
    set_error("spin lock is NULL");
    return 0;
  }
  const bool acquired = lock->lock.try_lock();
  clear_error();
  return acquired ? 1 : 0;
}

int hyle_util_spin_lock_unlock(hyle_util_spin_lock* lock) {
  if (lock == nullptr) {
    set_error("spin lock is NULL");
    return 0;
  }
  lock->lock.unlock();
  clear_error();
  return 1;
}

int hyle_util_spin_lock_is_locked(const hyle_util_spin_lock* lock) {
  if (lock == nullptr) {
    set_error("spin lock is NULL");
    return 0;
  }
  const bool locked = lock->lock.is_locked();
  clear_error();
  return locked ? 1 : 0;
}

}  // extern "C"

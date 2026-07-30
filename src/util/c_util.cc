#include "hyle/c/util.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/container/string_ref.h"
#include "hyle/container/swiss_table.h"
#include "hyle/error/status.h"
#include "hyle/macros.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/region.h"
#include "hyle/util/demangle.h"
#include "hyle/util/random.h"
#include "hyle/util/registry.h"
#include "hyle/util/spin_lock.h"
#include "../error/status_access.h"

namespace {

using hyle::Status;
namespace hc = hyle::container;
namespace hm = hyle::mem;

hyle_status to_c_status(Status status) noexcept {
  return hyle_status{hyle::detail::StatusAccess::release(status)};
}

bool valid_backing(const hyle_allocator& backing) noexcept {
  return backing.alloc != nullptr && backing.free != nullptr &&
         backing.realloc != nullptr;
}

void* c_backing_alloc(void* state, std::size_t size,
                      std::size_t alignment) noexcept {
  auto* backing = static_cast<hyle_allocator*>(state);
  return backing->alloc(backing->state, size, alignment);
}

void c_backing_free(void* state, void* pointer, std::size_t size,
                    std::size_t alignment) noexcept {
  auto* backing = static_cast<hyle_allocator*>(state);
  backing->free(backing->state, pointer, size, alignment);
}

void* c_backing_realloc(void* state, void* pointer, std::size_t old_size,
                        std::size_t new_size,
                        std::size_t alignment) noexcept {
  auto* backing = static_cast<hyle_allocator*>(state);
  return backing->realloc(backing->state, pointer, old_size, new_size,
                          alignment);
}

hm::Allocator to_cpp_allocator(hyle_allocator* backing) noexcept {
  return hm::Allocator(&c_backing_alloc, &c_backing_free, &c_backing_realloc,
                       backing);
}

std::size_t fill_string(hc::string_ref source, char* out,
                        std::size_t capacity) noexcept {
  const std::size_t length = source.size;
  if (capacity != 0) {
    const std::size_t copied = length < capacity - 1 ? length : capacity - 1;
    std::memcpy(out, source.data, copied);
    out[copied] = '\0';
  }
  return length;
}

HYLE_FORCE_INLINE bool c_string_view(hyle_container_string_view source,
                                     hc::string_ref* out) noexcept {
  if (source.size >= hc::string_ref::npos ||
      (source.data == nullptr && source.size != 0)) {
    return false;
  }
  *out = hc::string_ref{source.data, source.size,
                        source.null_terminated != 0};
  return true;
}

hyle_container_string_view to_c_string_view(hc::string_ref source) noexcept {
  return hyle_container_string_view{
      source.data, source.size, source.null_terminated ? 1u : 0u};
}

bool valid_c_state(const hyle_util_rng* rng) noexcept {
  return rng != nullptr &&
         (rng->state[0] | rng->state[1] | rng->state[2] | rng->state[3]) != 0;
}

bool load_rng(const hyle_util_rng* source,
              hyle::util::Rng& destination) noexcept {
  if (!valid_c_state(source)) return false;
  destination.set_state(hyle::util::RngState{{
      source->state[0], source->state[1], source->state[2], source->state[3]}});
  return true;
}

void store_rng(const hyle::util::Rng& source,
               hyle_util_rng* destination) noexcept {
  const hyle::util::RngState state = source.state();
  for (int lane = 0; lane < 4; ++lane) {
    destination->state[lane] = state.words[lane];
  }
}

constexpr hm::RegionOptions registry_string_options() noexcept {
  hm::RegionOptions options;
  options.initial_block_size = 4 * 1024;
  options.max_block_size = 64 * 1024;
  options.large_allocation_threshold = 16 * 1024;
  return options;
}

struct CRegistryEntry {
  hc::string_ref key;
  hyle_util_registry_create_fn creator = nullptr;
  void* creator_ctx = nullptr;
  hc::string_ref help;
  hyle_util_registry_priority priority = HYLE_UTIL_REGISTRY_DEFAULT;
};

struct CRegistryIndexValue {
  CRegistryEntry* entry = nullptr;
  hyle_util_registry_create_fn creator = nullptr;
  void* creator_ctx = nullptr;
};
static_assert(std::is_trivially_copyable_v<CRegistryIndexValue>);

using CRegistryTable = hc::FlatHashMap<
    hc::string_ref, CRegistryIndexValue, hc::Hash<hc::string_ref>,
    std::equal_to<>, hm::Allocator>;

struct CRegistrySnapshot {
  CRegistryEntry entry;
};

class CRegistrySnapshotStorage {
 public:
  explicit CRegistrySnapshotStorage(hm::Allocator allocator) noexcept
      : allocator_(allocator) {}
  ~CRegistrySnapshotStorage() {
    if (data_ != nullptr) {
      allocator_.deallocate(data_, capacity_ * sizeof(CRegistrySnapshot),
                            alignof(CRegistrySnapshot));
    }
  }
  HYLE_DISABLE_COPY_AND_MOVE(CRegistrySnapshotStorage);

  bool allocate(std::size_t count) noexcept {
    capacity_ = count;
    if (count == 0) return true;
    if (count > (std::numeric_limits<std::size_t>::max)() /
                    sizeof(CRegistrySnapshot)) {
      return false;
    }
    data_ = static_cast<CRegistrySnapshot*>(allocator_.allocate(
        count * sizeof(CRegistrySnapshot), alignof(CRegistrySnapshot)));
    return data_ != nullptr;
  }

  void push(CRegistrySnapshot entry) noexcept {
    HYLE_FATAL_ASSERT(size_ < capacity_);
    data_[size_++] = entry;
  }

  std::size_t size() const noexcept { return size_; }
  const CRegistrySnapshot& operator[](std::size_t index) const noexcept {
    return data_[index];
  }

 private:
  hm::Allocator allocator_;
  CRegistrySnapshot* data_ = nullptr;
  std::size_t size_ = 0;
  std::size_t capacity_ = 0;
};

}  // namespace

struct hyle_util_ref_o {
  explicit hyle_util_ref_o(hyle_allocator allocator, void* value,
                           hyle_util_ref_destroy_fn destroy, void* context)
      : backing(allocator), object(value), destroy_object(destroy),
        destroy_ctx(context) {}

  hyle_allocator backing;
  std::atomic<std::uint32_t> count{1};
  void* object = nullptr;
  hyle_util_ref_destroy_fn destroy_object = nullptr;
  void* destroy_ctx = nullptr;
};

struct hyle_util_spin_lock_o {
  explicit hyle_util_spin_lock_o(bool owned,
                                 hyle_allocator allocator = {}) noexcept
      : heap_owned(owned), backing(allocator) {}

  bool heap_owned = false;
  hyle_allocator backing{};
  hyle::util::SpinLock lock;
};

struct hyle_util_registry_o {
  explicit hyle_util_registry_o(hyle_allocator allocator) noexcept
      : backing(allocator), native_allocator(to_cpp_allocator(&backing)),
        keys(native_allocator, registry_string_options()),
        metadata(native_allocator, registry_string_options()),
        entries(native_allocator) {}

  static Status copy_string(hc::string_ref source, hc::string_ref* out,
                            hm::Region& region) noexcept {
    if (source.size == 0) {
      *out = hc::string_ref{"", 0, true};
      return Status::Ok();
    }
    auto* storage =
        static_cast<char*>(region.allocate(source.size + 1, alignof(char)));
    if (storage == nullptr) {
      return Status::ResourceExhausted("registry string allocation failed");
    }
    std::memcpy(storage, source.data, source.size);
    storage[source.size] = '\0';
    *out = hc::string_ref{storage, source.size, true};
    return Status::Ok();
  }

  CRegistryEntry* allocate_entry(hc::string_ref key,
                                 hyle_util_registry_create_fn creator,
                                 void* creator_ctx, hc::string_ref help,
                                 hyle_util_registry_priority priority) noexcept {
    void* storage = metadata.allocate(sizeof(CRegistryEntry),
                                      alignof(CRegistryEntry));
    if (storage == nullptr) return nullptr;
    return ::new (storage)
        CRegistryEntry{key, creator, creator_ctx, help, priority};
  }

  hyle_allocator backing;
  hm::Allocator native_allocator;
  hm::Region keys;
  hm::Region metadata;
  mutable std::mutex mutex;
  CRegistryTable entries;
  std::atomic<const CRegistryTable*> frozen_index{nullptr};
};

HYLE_NOINLINE bool registry_find_creator_mutable(
    const hyle_util_registry* registry, hc::string_ref key,
    hyle_util_registry_create_fn* out_creator, void** out_context) noexcept {
  std::lock_guard<std::mutex> lock(registry->mutex);
  if (const CRegistryTable* frozen_slots =
          registry->frozen_index.load(std::memory_order_relaxed)) {
    if (const CRegistryIndexValue* value = frozen_slots->find(key)) {
      *out_creator = value->creator;
      *out_context = value->creator_ctx;
      return true;
    }
  } else if (const CRegistryIndexValue* found = registry->entries.find(key)) {
    *out_creator = found->creator;
    *out_context = found->creator_ctx;
    return true;
  }
  return false;
}

HYLE_FORCE_INLINE bool registry_find_creator(
    const hyle_util_registry* registry, hc::string_ref key,
    hyle_util_registry_create_fn* out_creator, void** out_context) noexcept {
  if (const CRegistryTable* frozen_slots =
          registry->frozen_index.load(std::memory_order_acquire)) {
    if (const CRegistryIndexValue* value = frozen_slots->find(key)) {
      *out_creator = value->creator;
      *out_context = value->creator_ctx;
      return true;
    }
    return false;
  }
  return registry_find_creator_mutable(registry, key, out_creator,
                                       out_context);
}

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

hyle_status hyle_util_demangle(size_t* out_size, const char* mangled,
                               char* out, size_t cap) {
  if (out_size == nullptr) {
    return to_c_status(Status::InvalidArgument("out_size is null"));
  }
  *out_size = 0;
  if (cap != 0 && out == nullptr) {
    return to_c_status(Status::InvalidArgument(
        "out is null while cap is non-zero"));
  }
  *out_size = hyle::util::demangle(mangled, out, cap);
  return to_c_status(Status::Ok());
}

int hyle_util_rng_seed(hyle_util_rng* rng, uint64_t seed) {
  if (rng == nullptr) return 0;
  hyle::util::Rng native{seed};
  store_rng(native, rng);
  return 1;
}

int hyle_util_rng_is_valid(const hyle_util_rng* rng) {
  return valid_c_state(rng) ? 1 : 0;
}

uint64_t hyle_util_rng_next_u64(hyle_util_rng* rng) {
  hyle::util::Rng native;
  if (!load_rng(rng, native)) return 0;
  const uint64_t value = native.next_u64();
  store_rng(native, rng);
  return value;
}

uint32_t hyle_util_rng_next_u32(hyle_util_rng* rng) {
  hyle::util::Rng native;
  if (!load_rng(rng, native)) return 0;
  const uint32_t value = native.next_u32();
  store_rng(native, rng);
  return value;
}

int hyle_util_rng_next_bool(hyle_util_rng* rng) {
  hyle::util::Rng native;
  if (!load_rng(rng, native)) return 0;
  const int value = native.next_bool() ? 1 : 0;
  store_rng(native, rng);
  return value;
}

float hyle_util_rng_next_float(hyle_util_rng* rng) {
  hyle::util::Rng native;
  if (!load_rng(rng, native)) return 0.0f;
  const float value = native.next_float();
  store_rng(native, rng);
  return value;
}

double hyle_util_rng_next_double(hyle_util_rng* rng) {
  hyle::util::Rng native;
  if (!load_rng(rng, native)) return 0.0;
  const double value = native.next_double();
  store_rng(native, rng);
  return value;
}

uint64_t hyle_util_rng_range_u64(hyle_util_rng* rng, uint64_t lo,
                                 uint64_t hi) {
  if (lo > hi) return 0;
  hyle::util::Rng native;
  if (!load_rng(rng, native)) return 0;
  const uint64_t value = native.range_u64(lo, hi);
  store_rng(native, rng);
  return value;
}

uint32_t hyle_util_rng_range_u32(hyle_util_rng* rng, uint32_t lo,
                                 uint32_t hi) {
  if (lo > hi) return 0;
  hyle::util::Rng native;
  if (!load_rng(rng, native)) return 0;
  const uint32_t value = native.range_u32(lo, hi);
  store_rng(native, rng);
  return value;
}

double hyle_util_rng_range_double(hyle_util_rng* rng, double lo, double hi) {
  if (!std::isfinite(lo) || !std::isfinite(hi) || lo > hi) return 0.0;
  hyle::util::Rng native;
  if (!load_rng(rng, native)) return 0.0;
  const double value = native.range_double(lo, hi);
  store_rng(native, rng);
  return value;
}

int hyle_util_rng_fill_u64(hyle_util_rng* rng, uint64_t* out, size_t count) {
  if (count != 0 && out == nullptr) return 0;
  hyle::util::Rng native;
  if (!load_rng(rng, native)) return 0;
  for (size_t i = 0; i < count; ++i) out[i] = native.next_u64();
  store_rng(native, rng);
  return 1;
}

int hyle_util_rng_fill_float(hyle_util_rng* rng, float* out, size_t count) {
  if (count != 0 && out == nullptr) return 0;
  hyle::util::Rng native;
  if (!load_rng(rng, native)) return 0;
  for (size_t i = 0; i < count; ++i) out[i] = native.next_float();
  store_rng(native, rng);
  return 1;
}

int hyle_util_rng_fill_double(hyle_util_rng* rng, double* out, size_t count) {
  if (count != 0 && out == nullptr) return 0;
  hyle::util::Rng native;
  if (!load_rng(rng, native)) return 0;
  for (size_t i = 0; i < count; ++i) out[i] = native.next_double();
  store_rng(native, rng);
  return 1;
}

int hyle_util_rng_jump(hyle_util_rng* rng) {
  hyle::util::Rng native;
  if (!load_rng(rng, native)) return 0;
  native.jump();
  store_rng(native, rng);
  return 1;
}

int hyle_util_rng_long_jump(hyle_util_rng* rng) {
  hyle::util::Rng native;
  if (!load_rng(rng, native)) return 0;
  native.long_jump();
  store_rng(native, rng);
  return 1;
}

int hyle_util_rng_shuffle(hyle_util_rng* rng, uint64_t count,
                          hyle_util_rng_swap_fn swap, void* ctx) {
  if (count > 1 && swap == nullptr) return 0;
  hyle::util::Rng native;
  if (!load_rng(rng, native)) return 0;
  native.shuffle(count, [&](uint64_t left, uint64_t right) {
    swap(ctx, left, right);
  });
  store_rng(native, rng);
  return 1;
}

hyle_util_rng* hyle_util_thread_rng(void) {
  static thread_local hyle_util_rng state = [] {
    hyle::util::Rng native{hyle::util::detail::next_thread_seed()};
    const hyle::util::RngState snapshot = native.state();
    hyle_util_rng result{};
    for (int lane = 0; lane < 4; ++lane) {
      result.state[lane] = snapshot.words[lane];
    }
    return result;
  }();
  return &state;
}

hyle_status hyle_util_ref_create(
    hyle_util_ref** out, void* object,
    hyle_util_ref_destroy_fn destroy_object, void* destroy_ctx,
    hyle_allocator backing) {
  if (out == nullptr) {
    return to_c_status(Status::InvalidArgument("out is null"));
  }
  *out = nullptr;
  if (!valid_backing(backing)) {
    return to_c_status(Status::InvalidArgument("backing allocator is invalid"));
  }
  void* storage = backing.alloc(backing.state, sizeof(hyle_util_ref_o),
                                alignof(hyle_util_ref_o));
  if (storage == nullptr) {
    return to_c_status(Status::ResourceExhausted(
        "reference control allocation failed"));
  }
  *out = ::new (storage)
      hyle_util_ref_o(backing, object, destroy_object, destroy_ctx);
  return to_c_status(Status::Ok());
}

hyle_util_ref* hyle_util_ref_retain(hyle_util_ref* ref) {
  if (ref == nullptr) return nullptr;
  std::uint32_t count = ref->count.load(std::memory_order_relaxed);
  for (;;) {
    if (count == 0 || count == (std::numeric_limits<std::uint32_t>::max)()) {
      return nullptr;
    }
    if (ref->count.compare_exchange_weak(
            count, count + 1, std::memory_order_relaxed,
            std::memory_order_relaxed)) {
      return ref;
    }
  }
}

void hyle_util_ref_release(hyle_util_ref* ref) {
  if (ref == nullptr) return;
  const std::uint32_t old =
      ref->count.fetch_sub(1, std::memory_order_release);
  HYLE_FATAL_ASSERT(old != 0);
  if (old == 1) {
    std::atomic_thread_fence(std::memory_order_acquire);
    if (ref->destroy_object != nullptr) {
      ref->destroy_object(ref->object, ref->destroy_ctx);
    }
    const hyle_allocator backing = ref->backing;
    ref->~hyle_util_ref_o();
    backing.free(backing.state, ref, sizeof(hyle_util_ref_o),
                 alignof(hyle_util_ref_o));
  }
}

void* hyle_util_ref_get(const hyle_util_ref* ref) {
  return ref != nullptr ? ref->object : nullptr;
}

uint32_t hyle_util_ref_use_count(const hyle_util_ref* ref) {
  return ref != nullptr ? ref->count.load(std::memory_order_relaxed) : 0;
}

int hyle_util_ref_is_unique(const hyle_util_ref* ref) {
  return ref != nullptr &&
         ref->count.load(std::memory_order_acquire) == 1;
}

hyle_status hyle_util_registry_create(hyle_util_registry** out,
                                      hyle_allocator backing) {
  if (out == nullptr) {
    return to_c_status(Status::InvalidArgument("out is null"));
  }
  *out = nullptr;
  if (!valid_backing(backing)) {
    return to_c_status(Status::InvalidArgument("backing allocator is invalid"));
  }
  void* storage = backing.alloc(backing.state, sizeof(hyle_util_registry_o),
                                alignof(hyle_util_registry_o));
  if (storage == nullptr) {
    return to_c_status(Status::ResourceExhausted(
        "registry control allocation failed"));
  }
  *out = ::new (storage) hyle_util_registry_o(backing);
  return to_c_status(Status::Ok());
}

void hyle_util_registry_destroy(hyle_util_registry* registry) {
  if (registry == nullptr) return;
  const hyle_allocator backing = registry->backing;
  registry->~hyle_util_registry_o();
  backing.free(backing.state, registry, sizeof(hyle_util_registry_o),
               alignof(hyle_util_registry_o));
}

hyle_status hyle_util_registry_reserve(hyle_util_registry* registry,
                                       size_t count) {
  if (registry == nullptr) {
    return to_c_status(Status::InvalidArgument("registry is null"));
  }
  std::lock_guard<std::mutex> lock(registry->mutex);
  if (registry->frozen_index.load(std::memory_order_relaxed) != nullptr) {
    return to_c_status(Status::FailedPrecondition("registry is frozen"));
  }
  if (!registry->entries.reserve(count)) {
    return to_c_status(Status::ResourceExhausted("registry reserve failed"));
  }
  return to_c_status(Status::Ok());
}

hyle_status hyle_util_registry_register(
    hyle_util_registry* registry, hyle_container_string_view key,
    hyle_util_registry_create_fn creator, void* creator_ctx,
    hyle_util_registry_priority priority,
    hyle_container_string_view help) {
  if (registry == nullptr) {
    return to_c_status(Status::InvalidArgument("registry is null"));
  }
  hc::string_ref key_view;
  if (!c_string_view(key, &key_view) || key_view.empty()) {
    return to_c_status(Status::InvalidArgument(
        "registry key is null, empty, or too long"));
  }
  if (creator == nullptr) {
    return to_c_status(Status::InvalidArgument("registry creator is null"));
  }
  if (priority < HYLE_UTIL_REGISTRY_FALLBACK ||
      priority > HYLE_UTIL_REGISTRY_PREFERRED) {
    return to_c_status(Status::InvalidArgument("registry priority is invalid"));
  }
  hc::string_ref help_view;
  if (!c_string_view(help, &help_view)) {
    return to_c_status(Status::InvalidArgument("registry help is too long"));
  }

  std::lock_guard<std::mutex> lock(registry->mutex);
  if (registry->frozen_index.load(std::memory_order_relaxed) != nullptr) {
    return to_c_status(Status::FailedPrecondition("registry is frozen"));
  }
  if (CRegistryIndexValue* index = registry->entries.find(key_view)) {
    CRegistryEntry* existing = index->entry;
    if (priority == existing->priority) {
      return to_c_status(Status::AlreadyExists(
          "registry key already has the same priority"));
    }
    if (priority < existing->priority) return to_c_status(Status::Ok());
    hc::string_ref owned_help;
    Status copied = registry->copy_string(help_view, &owned_help,
                                          registry->metadata);
    if (!copied.ok()) return to_c_status(std::move(copied));
    existing->creator = creator;
    existing->creator_ctx = creator_ctx;
    index->creator = creator;
    index->creator_ctx = creator_ctx;
    existing->help = owned_help;
    existing->priority = priority;
    return to_c_status(Status::Ok());
  }

  hc::string_ref owned_key;
  hc::string_ref owned_help;
  Status copied = registry->copy_string(key_view, &owned_key, registry->keys);
  if (!copied.ok()) return to_c_status(std::move(copied));
  copied = registry->copy_string(help_view, &owned_help, registry->metadata);
  if (!copied.ok()) return to_c_status(std::move(copied));
  CRegistryEntry* entry = registry->allocate_entry(
      owned_key, creator, creator_ctx, owned_help, priority);
  if (entry == nullptr) {
    return to_c_status(Status::ResourceExhausted(
        "registry entry allocation failed"));
  }
  using InsertResult = CRegistryTable::InsertResult;
  if (registry->entries.try_insert_or_assign(
          owned_key, CRegistryIndexValue{entry, creator, creator_ctx}) ==
      InsertResult::kOutOfMemory) {
    return to_c_status(Status::ResourceExhausted(
        "registry table allocation failed"));
  }
  return to_c_status(Status::Ok());
}

hyle_status hyle_util_registry_freeze(hyle_util_registry* registry) {
  if (registry == nullptr) {
    return to_c_status(Status::InvalidArgument("registry is null"));
  }
  std::lock_guard<std::mutex> lock(registry->mutex);
  if (registry->frozen_index.load(std::memory_order_relaxed) != nullptr) {
    return to_c_status(Status::Ok());
  }
  registry->frozen_index.store(&registry->entries,
                               std::memory_order_release);
  return to_c_status(Status::Ok());
}

int hyle_util_registry_is_frozen(const hyle_util_registry* registry) {
  return registry != nullptr &&
         registry->frozen_index.load(std::memory_order_acquire) != nullptr;
}

int hyle_util_registry_has(const hyle_util_registry* registry,
                           hyle_container_string_view key) {
  if (registry == nullptr) return 0;
  hc::string_ref key_view;
  if (!c_string_view(key, &key_view)) return 0;
  if (const CRegistryTable* index =
          registry->frozen_index.load(std::memory_order_acquire)) {
    return index->find(key_view) != nullptr;
  }
  std::lock_guard<std::mutex> lock(registry->mutex);
  if (const CRegistryTable* index =
          registry->frozen_index.load(std::memory_order_relaxed)) {
    return index->find(key_view) != nullptr;
  }
  return registry->entries.find(key_view) != nullptr;
}

hyle_status hyle_util_registry_create_object(
    void** out, const hyle_util_registry* registry,
    hyle_container_string_view key, const void* create_args) {
  if (out == nullptr) {
    return to_c_status(Status::InvalidArgument("out is null"));
  }
  *out = nullptr;
  if (registry == nullptr) {
    return to_c_status(Status::InvalidArgument("registry is null"));
  }
  hc::string_ref key_view;
  if (!c_string_view(key, &key_view)) {
    return to_c_status(Status::InvalidArgument("registry key is null or too long"));
  }

  hyle_util_registry_create_fn creator = nullptr;
  void* creator_ctx = nullptr;
  if (!registry_find_creator(registry, key_view, &creator, &creator_ctx)) {
    return to_c_status(Status::NotFound("registry key is not registered"));
  }
  *out = creator(creator_ctx, create_args);
  return to_c_status(Status::Ok());
}

void* hyle_util_registry_create_or_null(
    const hyle_util_registry* registry, hyle_container_string_view key,
    const void* create_args) {
  if (registry == nullptr) return nullptr;
  hc::string_ref key_view;
  if (!c_string_view(key, &key_view)) return nullptr;
  hyle_util_registry_create_fn creator = nullptr;
  void* creator_ctx = nullptr;
  if (!registry_find_creator(registry, key_view, &creator, &creator_ctx)) {
    return nullptr;
  }
  return creator(creator_ctx, create_args);
}

hyle_status hyle_util_registry_help(
    size_t* out_size, const hyle_util_registry* registry,
    hyle_container_string_view key, char* out, size_t cap) {
  if (out_size == nullptr) {
    return to_c_status(Status::InvalidArgument("out_size is null"));
  }
  *out_size = 0;
  if (registry == nullptr) {
    return to_c_status(Status::InvalidArgument("registry is null"));
  }
  if (cap != 0 && out == nullptr) {
    return to_c_status(Status::InvalidArgument(
        "out is null while cap is non-zero"));
  }
  hc::string_ref key_view;
  if (!c_string_view(key, &key_view)) {
    return to_c_status(Status::InvalidArgument("registry key is null or too long"));
  }

  hc::string_ref help_view;
  bool found = false;
  if (const CRegistryTable* frozen_slots =
          registry->frozen_index.load(std::memory_order_acquire)) {
    if (const CRegistryIndexValue* value = frozen_slots->find(key_view)) {
      help_view = value->entry->help;
      found = true;
    }
  } else {
    std::lock_guard<std::mutex> lock(registry->mutex);
    if (const CRegistryTable* published_slots =
            registry->frozen_index.load(std::memory_order_relaxed)) {
      if (const CRegistryIndexValue* value = published_slots->find(key_view)) {
        help_view = value->entry->help;
        found = true;
      }
    } else if (const CRegistryIndexValue* found_entry =
                   registry->entries.find(key_view)) {
      help_view = found_entry->entry->help;
      found = true;
    }
  }
  if (!found) {
    return to_c_status(Status::NotFound("registry key is not registered"));
  }
  *out_size = fill_string(help_view, out, cap);
  return to_c_status(Status::Ok());
}

size_t hyle_util_registry_size(const hyle_util_registry* registry) {
  if (registry == nullptr) return 0;
  if (const CRegistryTable* index =
          registry->frozen_index.load(std::memory_order_acquire)) {
    return index->size();
  }
  std::lock_guard<std::mutex> lock(registry->mutex);
  return registry->entries.size();
}

hyle_status hyle_util_registry_for_each(
    int* out_completed, const hyle_util_registry* registry,
    hyle_util_registry_visit_fn visit, void* visit_ctx) {
  if (out_completed != nullptr) *out_completed = 0;
  if (registry == nullptr) {
    return to_c_status(Status::InvalidArgument("registry is null"));
  }
  if (visit == nullptr) {
    return to_c_status(Status::InvalidArgument("visitor is null"));
  }

  bool completed = true;
  if (const CRegistryTable* frozen_slots =
          registry->frozen_index.load(std::memory_order_acquire)) {
    frozen_slots->for_each(
        [&](hc::string_ref, const CRegistryIndexValue& value) {
      if (completed) {
        const CRegistryEntry* entry = value.entry;
        completed = visit(visit_ctx, to_c_string_view(entry->key),
                          to_c_string_view(entry->help), entry->priority) != 0;
      }
    });
  } else {
    CRegistrySnapshotStorage snapshot(registry->native_allocator);
    bool became_frozen = false;
    {
      std::lock_guard<std::mutex> lock(registry->mutex);
      became_frozen =
          registry->frozen_index.load(std::memory_order_relaxed) != nullptr;
      if (became_frozen) {
        // freeze() published immutable storage before releasing this mutex.
      } else if (!snapshot.allocate(registry->entries.size())) {
        return to_c_status(Status::ResourceExhausted(
            "registry enumeration snapshot allocation failed"));
      } else {
        registry->entries.for_each(
          [&](hc::string_ref, const CRegistryIndexValue& index) {
            snapshot.push(CRegistrySnapshot{*index.entry});
          });
      }
    }
    if (became_frozen) {
      const CRegistryTable* published_slots =
          registry->frozen_index.load(std::memory_order_relaxed);
      published_slots->for_each(
          [&](hc::string_ref, const CRegistryIndexValue& value) {
        if (completed) {
          const CRegistryEntry* entry = value.entry;
          completed = visit(visit_ctx, to_c_string_view(entry->key),
                            to_c_string_view(entry->help),
                            entry->priority) != 0;
        }
      });
    } else {
      for (std::size_t i = 0; i < snapshot.size(); ++i) {
        const CRegistryEntry& entry = snapshot[i].entry;
        if (visit(visit_ctx, to_c_string_view(entry.key),
                  to_c_string_view(entry.help), entry.priority) == 0) {
          completed = false;
          break;
        }
      }
    }
  }
  if (out_completed != nullptr) *out_completed = completed ? 1 : 0;
  return to_c_status(Status::Ok());
}

hyle_status hyle_util_spin_lock_create(hyle_util_spin_lock** out,
                                       hyle_allocator backing) {
  if (out == nullptr) {
    return to_c_status(Status::InvalidArgument("out is null"));
  }
  *out = nullptr;
  if (!valid_backing(backing)) {
    return to_c_status(Status::InvalidArgument("backing allocator is invalid"));
  }
  void* storage = backing.alloc(backing.state, sizeof(hyle_util_spin_lock_o),
                                alignof(hyle_util_spin_lock_o));
  if (storage == nullptr) {
    return to_c_status(Status::ResourceExhausted(
        "spin lock allocation failed"));
  }
  *out = ::new (storage) hyle_util_spin_lock_o(true, backing);
  return to_c_status(Status::Ok());
}

size_t hyle_util_spin_lock_storage_size(void) {
  return sizeof(hyle_util_spin_lock_o);
}

size_t hyle_util_spin_lock_storage_alignment(void) {
  return alignof(hyle_util_spin_lock_o);
}

hyle_status hyle_util_spin_lock_init(hyle_util_spin_lock** out, void* storage,
                                     size_t storage_size) {
  if (out == nullptr) {
    return to_c_status(Status::InvalidArgument("out is null"));
  }
  *out = nullptr;
  if (storage == nullptr || storage_size < sizeof(hyle_util_spin_lock_o) ||
      reinterpret_cast<std::uintptr_t>(storage) %
              alignof(hyle_util_spin_lock_o) !=
          0) {
    return to_c_status(Status::InvalidArgument(
        "spin lock storage is null, too small, or misaligned"));
  }
  *out = ::new (storage) hyle_util_spin_lock_o(false);
  return to_c_status(Status::Ok());
}

hyle_status hyle_util_spin_lock_destroy(hyle_util_spin_lock* lock) {
  if (lock == nullptr) return to_c_status(Status::Ok());
  if (!lock->heap_owned) {
    return to_c_status(Status::FailedPrecondition(
        "externally stored spin lock requires deinit"));
  }
  if (lock->lock.is_locked()) {
    return to_c_status(Status::FailedPrecondition(
        "cannot destroy a locked spin lock"));
  }
  const hyle_allocator backing = lock->backing;
  lock->~hyle_util_spin_lock_o();
  backing.free(backing.state, lock, sizeof(hyle_util_spin_lock_o),
               alignof(hyle_util_spin_lock_o));
  return to_c_status(Status::Ok());
}

hyle_status hyle_util_spin_lock_deinit(hyle_util_spin_lock* lock) {
  if (lock == nullptr) return to_c_status(Status::Ok());
  if (lock->heap_owned) {
    return to_c_status(Status::FailedPrecondition(
        "heap-owned spin lock requires destroy"));
  }
  if (lock->lock.is_locked()) {
    return to_c_status(Status::FailedPrecondition(
        "cannot deinit a locked spin lock"));
  }
  lock->~hyle_util_spin_lock_o();
  return to_c_status(Status::Ok());
}

hyle_status hyle_util_spin_lock_lock(hyle_util_spin_lock* lock) {
  if (lock == nullptr) {
    return to_c_status(Status::InvalidArgument("spin lock is null"));
  }
  lock->lock.lock();
  return to_c_status(Status::Ok());
}

hyle_status hyle_util_spin_lock_try_lock(int* out_acquired,
                                         hyle_util_spin_lock* lock) {
  if (out_acquired == nullptr) {
    return to_c_status(Status::InvalidArgument("out_acquired is null"));
  }
  *out_acquired = 0;
  if (lock == nullptr) {
    return to_c_status(Status::InvalidArgument("spin lock is null"));
  }
  *out_acquired = lock->lock.try_lock() ? 1 : 0;
  return to_c_status(Status::Ok());
}

hyle_status hyle_util_spin_lock_unlock(hyle_util_spin_lock* lock) {
  if (lock == nullptr) {
    return to_c_status(Status::InvalidArgument("spin lock is null"));
  }
  lock->lock.unlock();
  return to_c_status(Status::Ok());
}

int hyle_util_spin_lock_is_locked(const hyle_util_spin_lock* lock) {
  return lock != nullptr && lock->lock.is_locked();
}

}  // extern "C"

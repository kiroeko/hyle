#include "hyle/c/container.h"

#include "c_options.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/container/bounded_queue.h"
#include "hyle/container/byte_ring_buffer.h"
#include "hyle/container/carray.h"
#include "hyle/container/concurrent_vector.h"
#include "hyle/container/detail/adaptive_rw_lock.h"
#include "hyle/container/dynamic_bitset.h"
#include "hyle/container/path.h"
#include "hyle/container/probe.h"
#include "hyle/container/sharded_queue.h"
#include "hyle/container/spsc_queue.h"
#include "hyle/container/sparse_set.h"
#include "hyle/container/string_algo.h"
#include "hyle/container/string_interner.h"
#include "hyle/container/swiss_table.h"
#include "hyle/container/work_stealing_deque.h"
#include "hyle/container/ws_fifo_queue.h"
#include "hyle/core/cache.h"
#include "hyle/mem/allocator.h"

namespace hc = hyle::container;

namespace {

bool is_power_of_two(std::size_t value) noexcept {
  return value != 0 && (value & (value - 1)) == 0;
}

bool valid_backing(const hyle_allocator& backing) noexcept {
  return backing.alloc != nullptr && backing.free != nullptr &&
         backing.realloc != nullptr;
}

bool valid_hash_callbacks(hyle_container_hash_fn hash,
                          hyle_container_equal_fn equal) noexcept {
  // A custom equivalence relation can merge byte-distinct keys, so the
  // process-seeded byte hash is not necessarily compatible with it. A custom
  // hash with the default byte equality remains valid.
  return equal == nullptr || hash != nullptr;
}

bool decode_vector_options(const hyle_container_vector_options* input,
                           hyle_container_vector_options& output) noexcept {
  output = hyle_container_vector_options HYLE_CONTAINER_VECTOR_OPTIONS_INIT;
  std::uint32_t declared_size = 0;
  if (!hc::c_detail::read_required_options_prefix(
          input, HYLE_CONTAINER_VECTOR_OPTIONS_REQUIRED_SIZE, output,
          declared_size))
    return false;
  hc::c_detail::read_optional_options_field(
      input, declared_size,
      offsetof(hyle_container_vector_options, initial_capacity),
      output.initial_capacity);
  hc::c_detail::read_optional_options_field(
      input, declared_size, offsetof(hyle_container_vector_options, backing),
      output.backing);
  return output.reserved == 0;
}

bool decode_hash_map_options(
    const hyle_container_hash_map_options* input,
    hyle_container_hash_map_options& output) noexcept {
  output =
      hyle_container_hash_map_options HYLE_CONTAINER_HASH_MAP_OPTIONS_INIT;
  std::uint32_t declared_size = 0;
  if (!hc::c_detail::read_required_options_prefix(
          input, HYLE_CONTAINER_HASH_MAP_OPTIONS_REQUIRED_SIZE, output,
          declared_size))
    return false;
  hc::c_detail::read_optional_options_field(
      input, declared_size, offsetof(hyle_container_hash_map_options, hash),
      output.hash);
  hc::c_detail::read_optional_options_field(
      input, declared_size, offsetof(hyle_container_hash_map_options, equal),
      output.equal);
  hc::c_detail::read_optional_options_field(
      input, declared_size,
      offsetof(hyle_container_hash_map_options, context), output.context);
  hc::c_detail::read_optional_options_field(
      input, declared_size,
      offsetof(hyle_container_hash_map_options, backing), output.backing);
  return output.reserved == 0;
}

bool decode_hash_set_options(
    const hyle_container_hash_set_options* input,
    hyle_container_hash_set_options& output) noexcept {
  output =
      hyle_container_hash_set_options HYLE_CONTAINER_HASH_SET_OPTIONS_INIT;
  std::uint32_t declared_size = 0;
  if (!hc::c_detail::read_required_options_prefix(
          input, HYLE_CONTAINER_HASH_SET_OPTIONS_REQUIRED_SIZE, output,
          declared_size))
    return false;
  hc::c_detail::read_optional_options_field(
      input, declared_size, offsetof(hyle_container_hash_set_options, hash),
      output.hash);
  hc::c_detail::read_optional_options_field(
      input, declared_size, offsetof(hyle_container_hash_set_options, equal),
      output.equal);
  hc::c_detail::read_optional_options_field(
      input, declared_size,
      offsetof(hyle_container_hash_set_options, context), output.context);
  hc::c_detail::read_optional_options_field(
      input, declared_size,
      offsetof(hyle_container_hash_set_options, backing), output.backing);
  return output.reserved == 0;
}

bool checked_align_up(std::size_t value, std::size_t alignment,
                      std::size_t& out) noexcept {
  if (!is_power_of_two(alignment) ||
      value > SIZE_MAX - (alignment - 1))
    return false;
  out = (value + alignment - 1) & ~(alignment - 1);
  return true;
}

bool checked_add(std::size_t left, std::size_t right,
                 std::size_t& out) noexcept {
  if (right > SIZE_MAX - left) return false;
  out = left + right;
  return true;
}

bool valid_hash_slot_layout(std::size_t key_size,
                            std::size_t key_alignment,
                            std::size_t value_size,
                            std::size_t value_alignment) noexcept {
  if (key_size == 0 || !is_power_of_two(key_alignment) ||
      !is_power_of_two(value_alignment))
    return false;
  std::size_t key_offset;
  std::size_t after_key;
  std::size_t value_offset;
  std::size_t after_value;
  std::size_t block_size;
  const std::size_t block_alignment =
      std::max({alignof(std::uint64_t), key_alignment, value_alignment});
  return checked_align_up(sizeof(std::uint64_t), key_alignment, key_offset) &&
         checked_add(key_offset, key_size, after_key) &&
         checked_align_up(after_key, value_alignment, value_offset) &&
         checked_add(value_offset, value_size, after_value) &&
         checked_align_up(after_value, block_alignment, block_size) &&
         block_size != 0;
}

void* c_backing_alloc(void* state, std::size_t size,
                      std::size_t alignment) noexcept {
  const auto* backing = static_cast<const hyle_allocator*>(state);
  return backing->alloc(backing->state, size, alignment);
}
void c_backing_free(void* state, void* pointer, std::size_t size,
                    std::size_t alignment) noexcept {
  const auto* backing = static_cast<const hyle_allocator*>(state);
  backing->free(backing->state, pointer, size, alignment);
}
void* c_backing_realloc(void* state, void* pointer, std::size_t old_size,
                        std::size_t new_size,
                        std::size_t alignment) noexcept {
  const auto* backing = static_cast<const hyle_allocator*>(state);
  return backing->realloc(backing->state, pointer, old_size, new_size,
                          alignment);
}
hyle::mem::Allocator to_cpp(const hyle_allocator* backing) noexcept {
  return hyle::mem::Allocator(&c_backing_alloc, &c_backing_free,
                              &c_backing_realloc,
                              const_cast<hyle_allocator*>(backing));
}

bool valid_range(const void* pointer, std::size_t size,
                 const char*) noexcept {
  return size == 0 || pointer != nullptr;
}

bool valid_output(void* pointer, std::size_t capacity) noexcept {
  return valid_range(pointer, capacity, "non-zero output capacity requires out");
}

hc::string_ref to_cpp(hyle_container_string_view value) noexcept {
  return hc::string_ref{value.data, value.size, value.null_terminated != 0};
}

hyle_container_string_view from_cpp(hc::string_ref value) noexcept {
  return {value.data, value.size, value.null_terminated};
}

std::size_t copy_concat(hc::string_ref left, hc::string_ref right, char* out,
                        std::size_t capacity, bool slash) noexcept {
  const std::size_t separator = slash ? 1 : 0;
  std::size_t need = left.size;
  if (need > std::numeric_limits<std::size_t>::max() - separator - right.size)
    return std::numeric_limits<std::size_t>::max();
  need += separator + right.size;
  std::size_t written = 0;
  auto append = [&](const char* data, std::size_t size) {
    const std::size_t room =
        capacity > written ? capacity - written - 1 : 0;
    const std::size_t count = std::min(room, size);
    if (count != 0) std::memmove(out + written, data, count);
    written += count;
  };
  append(left.data, left.size);
  if (slash) append("/", 1);
  append(right.data, right.size);
  if (capacity != 0) out[written] = '\0';
  return need;
}

struct RawMap {
  RawMap(const hyle_container_hash_map_options& options,
         hyle::mem::Allocator alloc)
      : options(options), alloc(alloc), seed(hc::detail::process_hash_seed()) {
    key_offset = align_up(sizeof(std::uint64_t), options.key_alignment);
    value_offset =
        align_up(key_offset + options.key_size, options.value_alignment);
    block_alignment =
        std::max({alignof(std::uint64_t), options.key_alignment,
                  options.value_alignment});
    // Every slot, not just slot zero, must preserve the requested key/value
    // alignment. Round the stride to the maximum alignment.
    block_size =
        align_up(value_offset + options.value_size, block_alignment);
  }

  ~RawMap() {
    hc::detail::SwissStorage storage{ctrl, slots, cap, mask, size,
                                     growth_left};
    hc::detail::swiss_free(storage, block_size, block_alignment, &alloc,
                           &deallocate);
  }

  static std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
  }
  void* slot(std::size_t index) noexcept {
    return slots + index * block_size;
  }
  const void* slot(std::size_t index) const noexcept {
    return slots + index * block_size;
  }
  void* key(std::size_t index) noexcept {
    return static_cast<std::byte*>(slot(index)) + key_offset;
  }
  const void* key(std::size_t index) const noexcept {
    return static_cast<const std::byte*>(slot(index)) + key_offset;
  }
  void* value(std::size_t index) noexcept {
    return static_cast<std::byte*>(slot(index)) + value_offset;
  }
  const void* value(std::size_t index) const noexcept {
    return static_cast<const std::byte*>(slot(index)) + value_offset;
  }
  std::uint64_t stored_hash(std::size_t index) const noexcept {
    std::uint64_t result;
    std::memcpy(&result, slot(index), sizeof(result));
    return result;
  }
  void set_hash(std::size_t index, std::uint64_t hash) noexcept {
    std::memcpy(slot(index), &hash, sizeof(hash));
  }
  std::uint64_t hash(const void* input) const noexcept {
    return options.hash != nullptr
               ? options.hash(input, options.key_size, options.context)
               : hc::hash_bytes(input, options.key_size, seed);
  }
  bool equal(const void* left, const void* right) const noexcept {
    return options.equal != nullptr
               ? options.equal(left, right, options.key_size,
                               options.context) != 0
               : std::memcmp(left, right, options.key_size) == 0;
  }
  std::size_t find_slot(const void* input, std::uint64_t hash) const noexcept {
    if (cap == 0) return kNoSlot;
    std::size_t position = hc::detail::h1(hash) & mask;
    std::size_t stride = 0;
    const hc::detail::ctrl_t wanted = hc::detail::h2(hash);
    for (;;) {
      hc::detail::Group group(ctrl + position);
      for (std::uint32_t lane : group.match(wanted)) {
        const std::size_t index = (position + lane) & mask;
        if (stored_hash(index) == hash && equal(key(index), input))
          return index;
      }
      if (group.match_empty()) return kNoSlot;
      stride += hc::detail::kGroupWidth;
      position = (position + stride) & mask;
    }
  }
  std::size_t find_insert_slot(std::uint64_t hash) const noexcept {
    std::size_t position = hc::detail::h1(hash) & mask;
    std::size_t stride = 0;
    for (;;) {
      hc::detail::Group group(ctrl + position);
      if (hc::detail::GroupMask reusable = group.match_empty_or_deleted())
        return (position + reusable.lowest()) & mask;
      stride += hc::detail::kGroupWidth;
      position = (position + stride) & mask;
    }
  }
  void set_ctrl(std::size_t index, hc::detail::ctrl_t value) noexcept {
    ctrl[index] = value;
    if (index < hc::detail::kGroupWidth) ctrl[cap + index] = value;
  }
  void erase_slot(std::size_t index) noexcept {
    const std::size_t before = (index - hc::detail::kGroupWidth) & mask;
    const hc::detail::GroupMask empty_after =
        hc::detail::Group(ctrl + index).match_empty();
    const hc::detail::GroupMask empty_before =
        hc::detail::Group(ctrl + before).match_empty();
    const bool never_full =
        empty_before && empty_after &&
        empty_after.trailing_zeros() +
                empty_before.leading_zeros(hc::detail::kGroupWidth) <
            hc::detail::kGroupWidth;
    if (never_full) {
      set_ctrl(index, hc::detail::kEmpty);
      ++growth_left;
    } else {
      set_ctrl(index, hc::detail::kDeleted);
    }
    --size;
  }
  static std::uint64_t slot_hash(const void* raw,
                                 const void*) noexcept {
    std::uint64_t result;
    std::memcpy(&result, raw, sizeof(result));
    return result;
  }
  static void* allocate(void* state, std::size_t bytes,
                        std::size_t alignment) noexcept {
    return static_cast<hyle::mem::Allocator*>(state)->allocate(bytes,
                                                               alignment);
  }
  static void deallocate(void* state, void* pointer, std::size_t bytes,
                         std::size_t alignment) noexcept {
    static_cast<hyle::mem::Allocator*>(state)->deallocate(pointer, bytes,
                                                           alignment);
  }
  static void relocate(void* destination, void* source,
                       std::size_t bytes) noexcept {
    std::memcpy(destination, source, bytes);
  }
  bool resize(std::size_t new_capacity) noexcept {
    hc::detail::SwissStorage storage{ctrl, slots, cap, mask, size,
                                     growth_left};
    if (!hc::detail::swiss_resize(
            storage, new_capacity, block_size, block_alignment, &alloc,
            &allocate, &deallocate, &slot_hash, nullptr, &relocate))
      return false;
    ctrl = storage.ctrl;
    slots = static_cast<std::byte*>(storage.slots);
    cap = storage.cap;
    mask = storage.mask;
    size = storage.size;
    growth_left = storage.growth_left;
    return true;
  }
  bool reserve(std::size_t count) noexcept {
    if (cap != 0) {
      const std::size_t max_live = cap - cap / 8;
      if (count <= max_live) {
        if (count <= size + growth_left) return true;
        return resize(cap);
      }
    } else if (count == 0) {
      return true;
    }
    if (count > SIZE_MAX / 8) return false;
    const std::size_t required = (count * 8 + 6) / 7;
    std::size_t new_capacity = hc::detail::kGroupWidth;
    while (new_capacity < required) {
      if (new_capacity > SIZE_MAX / 2) return false;
      new_capacity *= 2;
    }
    return resize(new_capacity);
  }
  bool aliases_slots(const void* pointer, std::size_t bytes) const noexcept {
    if (slots == nullptr || pointer == nullptr || bytes == 0) return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(slots);
    const auto end = begin + cap * block_size;
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    return address >= begin && address < end && bytes <= end - address;
  }
  void clear() noexcept {
    if (ctrl != nullptr)
      std::memset(ctrl, static_cast<int>(hc::detail::kEmpty),
                  cap + hc::detail::kGroupWidth);
    size = 0;
    growth_left = cap - cap / 8;
  }

  static constexpr std::size_t kNoSlot = SIZE_MAX;
  hyle_container_hash_map_options options;
  hyle::mem::Allocator alloc;
  std::uint64_t seed;
  hc::detail::ctrl_t* ctrl = nullptr;
  std::byte* slots = nullptr;
  std::size_t cap = 0;
  std::size_t mask = 0;
  std::size_t size = 0;
  std::size_t growth_left = 0;
  std::size_t key_offset = 0;
  std::size_t value_offset = 0;
  std::size_t block_size = 0;
  std::size_t block_alignment = 0;
};

struct alignas(hyle::core::kDestructiveInterferenceSize) ConcurrentRawShard {
  ConcurrentRawShard(const hyle_container_hash_map_options& options,
                     hyle::mem::Allocator alloc)
      : value(options, alloc) {}

  hc::detail::AdaptiveRwLock lock;
  RawMap value;
};

hyle_container_hash_map_options raw_set_options(
    const hyle_container_hash_set_options& options) noexcept {
  hyle_container_hash_map_options raw = HYLE_CONTAINER_HASH_MAP_OPTIONS_INIT;
  raw.key_size = options.key_size;
  raw.key_alignment = options.key_alignment;
  raw.value_size = 0;
  raw.value_alignment = 1;
  raw.hash = options.hash;
  raw.equal = options.equal;
  raw.context = options.context;
  raw.backing = options.backing;
  return raw;
}

template <class T, class... Args>
T* allocate_handle(hyle_allocator backing, Args&&... args) noexcept {
  void* memory =
      backing.alloc(backing.state, sizeof(T), alignof(T));
  if (memory == nullptr) return nullptr;
  try {
    return ::new (memory) T(std::forward<Args>(args)...);
  } catch (...) {
    backing.free(backing.state, memory, sizeof(T), alignof(T));
    return nullptr;
  }
}

template <class T>
void free_handle(T* handle) noexcept {
  if (handle == nullptr) return;
  const hyle_allocator backing = handle->backing;
  handle->~T();
  backing.free(backing.state, handle, sizeof(T), alignof(T));
}

}  // namespace

struct hyle_container_string_interner_o {
  hyle_allocator backing;
  hc::StringInterner value;
  explicit hyle_container_string_interner_o(hyle_allocator allocator)
      : backing(allocator), value(to_cpp(&backing)) {}
};

struct hyle_container_dynamic_bitset_o {
  hyle_allocator backing;
  explicit hyle_container_dynamic_bitset_o(hyle_allocator allocator)
      : backing(allocator), value(to_cpp(&backing)) {}
  hc::DynamicBitSet value;
};

struct hyle_container_sparse_u32_set_o {
  hyle_allocator backing;
  explicit hyle_container_sparse_u32_set_o(hyle_allocator allocator)
      : backing(allocator), value(to_cpp(&backing)) {}
  hc::SparseSet<std::uint32_t> value;
};

struct hyle_container_byte_ring_o {
  hyle_allocator backing;
  explicit hyle_container_byte_ring_o(hyle_allocator allocator)
      : backing(allocator), value(to_cpp(&backing)) {}
  hc::ByteRingBuffer value;
};

struct hyle_container_vector_o {
  hyle_allocator backing{};
  void* data = nullptr;
  std::size_t item_size = 0;
  std::size_t item_alignment = 0;
};
struct hyle_container_hash_map_o {
  hyle_allocator backing;
  explicit hyle_container_hash_map_o(
      const hyle_container_hash_map_options& options)
      : backing(options.backing), value(options, to_cpp(&backing)) {}
  RawMap value;
};
struct hyle_container_hash_set_o {
  hyle_allocator backing;
  explicit hyle_container_hash_set_o(
      const hyle_container_hash_set_options& options)
      : backing(options.backing),
        value(raw_set_options(options), to_cpp(&backing)) {}
  RawMap value;
};
struct hyle_container_concurrent_hash_map_o {
  hyle_allocator backing;
  std::size_t shard_count = 0;
  ConcurrentRawShard* shards = nullptr;

  hyle_container_concurrent_hash_map_o(
      const hyle_container_hash_map_options& options,
      std::size_t requested_shards)
      : backing(options.backing) {
    if (requested_shards > SIZE_MAX / sizeof(ConcurrentRawShard)) return;
    const std::size_t bytes = requested_shards * sizeof(ConcurrentRawShard);
    void* storage = backing.alloc(backing.state, bytes,
                                  alignof(ConcurrentRawShard));
    if (storage == nullptr) return;
    shards = static_cast<ConcurrentRawShard*>(storage);
    const hyle::mem::Allocator alloc = to_cpp(&backing);
    for (; shard_count < requested_shards; ++shard_count)
      ::new (static_cast<void*>(shards + shard_count))
          ConcurrentRawShard(options, alloc);
  }

  ~hyle_container_concurrent_hash_map_o() {
    if (shards == nullptr) return;
    const std::size_t count = shard_count;
    while (shard_count != 0) shards[--shard_count].~ConcurrentRawShard();
    backing.free(backing.state, shards, count * sizeof(ConcurrentRawShard),
                 alignof(ConcurrentRawShard));
  }

  ConcurrentRawShard& shard(std::uint64_t hash) const noexcept {
    return shards[(hash >> 32) & (shard_count - 1)];
  }
};
struct hyle_container_bounded_queue_o {
  hyle_allocator backing;
  hyle_container_bounded_queue_o(hyle_allocator allocator,
                                 std::size_t capacity)
      : backing(allocator), value(to_cpp(&backing), capacity) {}
  hc::BoundedQueue<void*, hyle::mem::Allocator> value;
};
struct hyle_container_spsc_queue_o {
  hyle_allocator backing;
  hyle_container_spsc_queue_o(hyle_allocator allocator,
                              std::size_t capacity)
      : backing(allocator), value(to_cpp(&backing), capacity) {}
  hc::SpscQueue<void*, hyle::mem::Allocator> value;
};
struct hyle_container_sharded_queue_o {
  hyle_allocator backing;
  hyle_container_sharded_queue_o(hyle_allocator allocator,
                                 std::size_t shards, std::size_t capacity)
      : backing(allocator), value(to_cpp(&backing), shards, capacity) {}
  hc::ShardedQueue<void*, hyle::mem::Allocator> value;
};
struct hyle_container_work_stealing_deque_o {
  hyle_allocator backing;
  hyle_container_work_stealing_deque_o(hyle_allocator allocator,
                                       std::size_t capacity)
      : backing(allocator), value(to_cpp(&backing), capacity) {}
  hc::WorkStealingDeque<void*, hyle::mem::Allocator> value;
};
struct hyle_container_ws_fifo_queue_o {
  hyle_allocator backing;
  hyle_container_ws_fifo_queue_o(hyle_allocator allocator,
                                 std::size_t capacity)
      : backing(allocator), value(to_cpp(&backing), capacity) {}
  hc::WsFifoQueue<void*, hyle::mem::Allocator> value;
};
struct hyle_container_concurrent_ptr_vector_o {
  hyle_allocator backing;
  hyle_container_concurrent_ptr_vector_o(hyle_allocator allocator,
                                         std::size_t capacity)
      : backing(allocator), value(to_cpp(&backing), capacity) {}
  hc::ConcurrentVector<void*, hyle::mem::Allocator> value;
};

namespace {

enum class VectorSource { kExternal, kLiveAlias, kInvalid };

VectorSource classify_vector_source(const hyle_container_vector* vector,
                                    const void* items, std::size_t count,
                                    std::size_t& byte_offset) noexcept {
  byte_offset = 0;
  if (count > SIZE_MAX / vector->item_size) return VectorSource::kInvalid;
  if (vector->data == nullptr) return VectorSource::kExternal;
  const auto begin = reinterpret_cast<std::uintptr_t>(vector->data);
  const std::size_t capacity_bytes =
      hc::carray_capacity(vector->data) * vector->item_size;
  if (capacity_bytes > UINTPTR_MAX - begin) return VectorSource::kInvalid;
  const auto address = reinterpret_cast<std::uintptr_t>(items);
  if (address < begin || address >= begin + capacity_bytes)
    return VectorSource::kExternal;
  const std::size_t live_bytes =
      hc::carray_size(vector->data) * vector->item_size;
  const std::size_t input_bytes = count * vector->item_size;
  byte_offset = static_cast<std::size_t>(address - begin);
  if (byte_offset > live_bytes || input_bytes > live_bytes - byte_offset)
    return VectorSource::kInvalid;
  return VectorSource::kLiveAlias;
}

std::size_t vector_max_capacity(
    const hyle_container_vector* vector) noexcept {
  return (SIZE_MAX - sizeof(hc::CArrayHeader) - vector->item_alignment) /
         vector->item_size;
}

hyle_status vector_reserve_for(hyle_container_vector* vector,
                               std::size_t requested) {
  const std::size_t current = hc::carray_capacity(vector->data);
  if (requested <= current) return hyle_status_ok();
  const std::size_t maximum = vector_max_capacity(vector);
  if (requested > maximum)
    return hyle_status_out_of_range(
        "vector capacity overflows address space");
  std::size_t target;
  if (current == 0) {
    target = requested < 4 ? 4 : requested;
  } else if (current > SIZE_MAX / 2) {
    target = requested;
  } else {
    target = std::max(requested, current * 2);
  }
  if (target > maximum) target = requested;
  return hyle_container_vector_reserve(vector, target);
}

void fill_vector_items(void* destination, std::size_t count,
                       const void* item, std::size_t item_size) noexcept {
  if (count == 0) return;
  std::memmove(destination, item, item_size);
  auto* output = static_cast<std::byte*>(destination);
  std::size_t completed = 1;
  while (completed < count) {
    const std::size_t batch = std::min(completed, count - completed);
    std::memcpy(output + completed * item_size, output,
                batch * item_size);
    completed += batch;
  }
}

}  // namespace

static_assert(std::is_standard_layout_v<hyle_allocator>);
static_assert(std::is_trivially_copyable_v<hyle_allocator>);
static_assert(std::is_standard_layout_v<hyle_container_string_view>);
static_assert(std::is_trivially_copyable_v<hyle_container_string_view>);
static_assert(sizeof(hyle_container_string_view) == sizeof(hc::string_ref));
static_assert(offsetof(hyle_container_string_view, data) ==
              offsetof(hc::string_ref, data));
static_assert(offsetof(hyle_container_string_view, size) ==
              offsetof(hc::string_ref, size));
static_assert(offsetof(hyle_container_string_view, null_terminated) ==
              offsetof(hc::string_ref, null_terminated));
static_assert(
    std::is_standard_layout_v<hyle_container_vector_options> &&
    std::is_trivially_copyable_v<hyle_container_vector_options> &&
    offsetof(hyle_container_vector_options, struct_size) == 0);
static_assert(
    std::is_standard_layout_v<hyle_container_hash_map_options> &&
    std::is_trivially_copyable_v<hyle_container_hash_map_options> &&
    offsetof(hyle_container_hash_map_options, struct_size) == 0);
static_assert(
    std::is_standard_layout_v<hyle_container_hash_set_options> &&
    std::is_trivially_copyable_v<hyle_container_hash_set_options> &&
    offsetof(hyle_container_hash_set_options, struct_size) == 0);
static_assert(static_cast<int>(hc::StealStatus::kOk) ==
              HYLE_CONTAINER_STEAL_OK);
static_assert(static_cast<int>(hc::StealStatus::kEmpty) ==
              HYLE_CONTAINER_STEAL_EMPTY);
static_assert(static_cast<int>(hc::StealStatus::kRetry) ==
              HYLE_CONTAINER_STEAL_RETRY);
static_assert(sizeof(hyle_container_insert_result) == sizeof(std::uint32_t));
static_assert(sizeof(hyle_container_steal_status) == sizeof(std::uint32_t));

extern "C" {

uint64_t hyle_container_hash_bytes(const void* data, size_t size,
                                   uint64_t seed) {
  if (!valid_range(data, size, "non-zero hash size requires data")) return 0;
  return hc::hash_bytes(data, size, seed);
}

uint64_t hyle_container_siphash24(const void* data, size_t size,
                                  uint64_t key0, uint64_t key1) {
  if (!valid_range(data, size, "non-zero hash size requires data")) return 0;
  return hc::siphash24(data, size, key0, key1);
}

int hyle_container_string_compare(hyle_container_string_view left,
                                  hyle_container_string_view right) {
  if (!valid_range(left.data, left.size, "invalid left string view") ||
      !valid_range(right.data, right.size, "invalid right string view"))
    return 0;
  const auto result = to_cpp(left) <=> to_cpp(right);
  return result < 0 ? -1 : result > 0 ? 1 : 0;
}

int hyle_container_string_equal_ignore_case(
    hyle_container_string_view left, hyle_container_string_view right) {
  if (!valid_range(left.data, left.size, "invalid left string view") ||
      !valid_range(right.data, right.size, "invalid right string view"))
    return 0;
  return hc::equal_ignore_case(to_cpp(left), to_cpp(right));
}

#define HYLE_STRING_SEARCH_IMPL(name, method)                                  \
  uint32_t name(hyle_container_string_view text,                               \
                hyle_container_string_view needle, uint32_t position) {        \
    if (!valid_range(text.data, text.size, "invalid text view") ||             \
        !valid_range(needle.data, needle.size, "invalid needle view"))         \
      return HYLE_CONTAINER_NPOS;                                              \
    return to_cpp(text).method(to_cpp(needle), position);                      \
  }
HYLE_STRING_SEARCH_IMPL(hyle_container_string_find, find)
HYLE_STRING_SEARCH_IMPL(hyle_container_string_rfind, rfind)
#undef HYLE_STRING_SEARCH_IMPL

uint32_t hyle_container_string_find_char(hyle_container_string_view text,
                                         char needle, uint32_t position) {
  if (!valid_range(text.data, text.size, "invalid text view"))
    return HYLE_CONTAINER_NPOS;
  return to_cpp(text).find(needle, position);
}
uint32_t hyle_container_string_rfind_char(hyle_container_string_view text,
                                          char needle, uint32_t position) {
  if (!valid_range(text.data, text.size, "invalid text view"))
    return HYLE_CONTAINER_NPOS;
  return to_cpp(text).rfind(needle, position);
}
uint32_t hyle_container_string_count_char(hyle_container_string_view text,
                                          char needle) {
  if (!valid_range(text.data, text.size, "invalid text view")) return 0;
  return to_cpp(text).count(needle);
}

#define HYLE_STRING_SET_SEARCH_IMPL(name, method)                              \
  uint32_t name(hyle_container_string_view text,                              \
                hyle_container_string_view set, uint32_t position) {          \
    if (!valid_range(text.data, text.size, "invalid text view") ||            \
        !valid_range(set.data, set.size, "invalid character set view"))       \
      return HYLE_CONTAINER_NPOS;                                             \
    return to_cpp(text).method(to_cpp(set), position);                        \
  }
HYLE_STRING_SET_SEARCH_IMPL(hyle_container_string_find_first_of, find_first_of)
HYLE_STRING_SET_SEARCH_IMPL(hyle_container_string_find_first_not_of,
                            find_first_not_of)
HYLE_STRING_SET_SEARCH_IMPL(hyle_container_string_find_last_of, find_last_of)
HYLE_STRING_SET_SEARCH_IMPL(hyle_container_string_find_last_not_of,
                            find_last_not_of)
#undef HYLE_STRING_SET_SEARCH_IMPL

#define HYLE_STRING_PREDICATE_IMPL(name, method)                               \
  int name(hyle_container_string_view text,                                   \
           hyle_container_string_view part) {                                 \
    if (!valid_range(text.data, text.size, "invalid text view") ||            \
        !valid_range(part.data, part.size, "invalid string view"))            \
      return 0;                                                               \
    return to_cpp(text).method(to_cpp(part));                                 \
  }
HYLE_STRING_PREDICATE_IMPL(hyle_container_string_starts_with, starts_with)
HYLE_STRING_PREDICATE_IMPL(hyle_container_string_ends_with, ends_with)
HYLE_STRING_PREDICATE_IMPL(hyle_container_string_contains, contains)
#undef HYLE_STRING_PREDICATE_IMPL

hyle_container_string_view hyle_container_string_slice(
    hyle_container_string_view text, uint32_t start, uint32_t length) {
  if (!valid_range(text.data, text.size, "invalid text view")) return {};
  return from_cpp(to_cpp(text).substr(start, length));
}
hyle_container_string_view hyle_container_string_split(
    hyle_container_string_view* text, char delimiter) {
  if (text == nullptr ||
      !valid_range(text->data, text->size, "invalid text view")) {
    return {};
  }
  hc::string_ref remainder = to_cpp(*text);
  const hc::string_ref result = hc::split(&remainder, delimiter);
  *text = from_cpp(remainder);
  return from_cpp(result);
}
hyle_container_string_view hyle_container_string_rsplit(
    hyle_container_string_view* text, char delimiter) {
  if (text == nullptr ||
      !valid_range(text->data, text->size, "invalid text view")) {
    return {};
  }
  hc::string_ref remainder = to_cpp(*text);
  const hc::string_ref result = hc::rsplit(&remainder, delimiter);
  *text = from_cpp(remainder);
  return from_cpp(result);
}
hyle_container_string_view hyle_container_string_trim(
    hyle_container_string_view text, hyle_container_string_view cut) {
  if (!valid_range(text.data, text.size, "invalid text view") ||
      !valid_range(cut.data, cut.size, "invalid cut view"))
    return {};
  return from_cpp(to_cpp(text).trim(to_cpp(cut)));
}

size_t hyle_container_string_copy(hyle_container_string_view text, char* out,
                                  size_t capacity) {
  if (!valid_range(text.data, text.size, "invalid text view") ||
      !valid_output(out, capacity))
    return 0;
  const size_t written =
      capacity == 0 ? 0 : std::min<size_t>(text.size, capacity - 1);
  if (written != 0) std::memmove(out, text.data, written);
  if (capacity != 0) out[written] = '\0';
  return text.size;
}
size_t hyle_container_string_concat(hyle_container_string_view left,
                                    hyle_container_string_view right,
                                    char* out, size_t capacity) {
  if (!valid_range(left.data, left.size, "invalid left string view") ||
      !valid_range(right.data, right.size, "invalid right string view") ||
      !valid_output(out, capacity))
    return 0;
  return copy_concat(to_cpp(left), to_cpp(right), out, capacity, false);
}

size_t hyle_container_string_lower(hyle_container_string_view text, char* out,
                                   size_t capacity) {
  if (!valid_range(text.data, text.size, "invalid text view") ||
      !valid_output(out, capacity))
    return 0;
  if (capacity > UINT32_MAX) capacity = UINT32_MAX;
  return hc::to_lower(to_cpp(text), out, static_cast<uint32_t>(capacity));
}
size_t hyle_container_string_upper(hyle_container_string_view text, char* out,
                                   size_t capacity) {
  if (!valid_range(text.data, text.size, "invalid text view") ||
      !valid_output(out, capacity))
    return 0;
  if (capacity > UINT32_MAX) capacity = UINT32_MAX;
  return hc::to_upper(to_cpp(text), out, static_cast<uint32_t>(capacity));
}
size_t hyle_container_hex_encode(const void* data, size_t size, char* out,
                                 size_t capacity) {
  if (!valid_range(data, size, "non-zero input size requires data") ||
      !valid_output(out, capacity))
    return 0;
  return hc::hex_encode(data, size, out, capacity);
}
size_t hyle_container_hex_decode(hyle_container_string_view hex, void* out,
                                 size_t capacity) {
  if (!valid_range(hex.data, hex.size, "invalid hexadecimal view") ||
      !valid_output(out, capacity))
    return SIZE_MAX;
  const size_t result = hc::hex_decode(to_cpp(hex), out, capacity);
  return result;
}
size_t hyle_container_string_c_escape(hyle_container_string_view text,
                                      char* out, size_t capacity) {
  if (!valid_range(text.data, text.size, "invalid text view") ||
      !valid_output(out, capacity))
    return 0;
  return hc::c_escape(to_cpp(text), out, capacity);
}
int hyle_container_utf8_validate(hyle_container_string_view text) {
  if (!valid_range(text.data, text.size, "invalid text view")) return 0;
  return hc::utf8_validate(to_cpp(text)) ? 1 : 0;
}
int64_t hyle_container_utf8_length(hyle_container_string_view text) {
  if (!valid_range(text.data, text.size, "invalid text view")) return -1;
  return hc::utf8_length(to_cpp(text));
}
#define HYLE_PATH_VIEW_IMPL(name, method)                                      \
  hyle_container_string_view name(hyle_container_string_view path) {           \
    if (!valid_range(path.data, path.size, "invalid path view")) return {};    \
    return from_cpp(hc::method(to_cpp(path)));                                 \
  }
HYLE_PATH_VIEW_IMPL(hyle_container_path_extension, extension)
HYLE_PATH_VIEW_IMPL(hyle_container_path_strip_extension, strip_extension)
HYLE_PATH_VIEW_IMPL(hyle_container_path_base, base)
HYLE_PATH_VIEW_IMPL(hyle_container_path_directory, directory)
#undef HYLE_PATH_VIEW_IMPL

int hyle_container_path_is_absolute(hyle_container_string_view path) {
  if (!valid_range(path.data, path.size, "invalid path view")) return 0;
  return hc::is_absolute(to_cpp(path));
}
int hyle_container_path_is_unc(hyle_container_string_view path) {
  if (!valid_range(path.data, path.size, "invalid path view")) return 0;
  return hc::is_unc(to_cpp(path));
}
hyle_container_string_view hyle_container_path_pop_first(
    hyle_container_string_view* path) {
  if (path == nullptr ||
      !valid_range(path->data, path->size, "invalid path view")) {
    return {};
  }
  hc::string_ref value = to_cpp(*path);
  const hc::string_ref result = hc::pop_first(&value);
  *path = from_cpp(value);
  return from_cpp(result);
}
hyle_container_string_view hyle_container_path_pop_last(
    hyle_container_string_view* path) {
  if (path == nullptr ||
      !valid_range(path->data, path->size, "invalid path view")) {
    return {};
  }
  hc::string_ref value = to_cpp(*path);
  const hc::string_ref result = hc::pop_last(&value);
  *path = from_cpp(value);
  return from_cpp(result);
}
size_t hyle_container_path_join(hyle_container_string_view left,
                                hyle_container_string_view right, char* out,
                                size_t capacity) {
  if (!valid_range(left.data, left.size, "invalid left path view") ||
      !valid_range(right.data, right.size, "invalid right path view") ||
      !valid_output(out, capacity))
    return 0;
  return copy_concat(to_cpp(left), to_cpp(right), out, capacity, true);
}
size_t hyle_container_path_join_array(
    const hyle_container_string_view* parts, size_t count, char* out,
    size_t capacity) {
  if (!valid_range(parts, count, "non-zero count requires parts") ||
      !valid_output(out, capacity))
    return 0;
  for (size_t i = 0; i < count; ++i)
    if (!valid_range(parts[i].data, parts[i].size, "invalid path part"))
      return 0;
  // Explicit per-field conversion; use a simple streaming implementation to
  // avoid allocating a temporary C++ view array at the ABI boundary.
  size_t need = 0;
  size_t written = 0;
  bool any = false;
  const bool lead =
      count != 0 && parts[0].size != 0 &&
      hc::is_separator(parts[0].data[0]);
  auto put = [&](char value) {
    if (written + 1 < capacity) out[written++] = value;
    if (need != SIZE_MAX) ++need;
  };
  if (lead) put('/');
  for (size_t i = 0; i < count; ++i) {
    uint32_t p = 0;
    while (p < parts[i].size) {
      while (p < parts[i].size && hc::is_separator(parts[i].data[p])) ++p;
      const uint32_t start = p;
      while (p < parts[i].size && !hc::is_separator(parts[i].data[p])) ++p;
      if (p != start) {
        if (any) put('/');
        for (uint32_t j = start; j < p; ++j) put(parts[i].data[j]);
        any = true;
      }
    }
  }
  const bool trail =
      count != 0 && parts[count - 1].size != 0 &&
      hc::is_separator(parts[count - 1].data[parts[count - 1].size - 1]);
  if (trail && any) put('/');
  if (capacity != 0) out[written] = '\0';
  return need;
}
size_t hyle_container_path_split_all(
    hyle_container_string_view path, hyle_container_string_view* output,
    size_t output_capacity) {
  if (!valid_range(path.data, path.size, "invalid path view") ||
      !valid_range(output, output_capacity,
                   "non-zero output capacity requires output"))
    return 0;
  hc::string_ref rest = to_cpp(path);
  std::size_t count = 0;
  while (rest.size != 0) {
    const hc::string_ref component = hc::pop_first(&rest);
    if (count < output_capacity) output[count] = from_cpp(component);
    ++count;  // bounded by path.size + 1 (path.size is uint32)
  }
  return count;
}
size_t hyle_container_path_replace_extension(
    hyle_container_string_view path, hyle_container_string_view extension,
    char* out, size_t capacity) {
  if (!valid_range(path.data, path.size, "invalid path view") ||
      !valid_range(extension.data, extension.size, "invalid extension view") ||
      !valid_output(out, capacity))
    return 0;
  return copy_concat(hc::strip_extension(to_cpp(path)), to_cpp(extension), out,
                     capacity, false);
}
uint32_t hyle_container_path_normalize(char* path) {
  if (path == nullptr) return 0;
  return hc::normalize(path);
}

hyle_status hyle_container_string_interner_create(
    hyle_container_string_interner** out, hyle_allocator backing) {
  if (out == nullptr)
    return hyle_status_invalid_argument("string interner output is NULL");
  *out = nullptr;
  if (!valid_backing(backing)) {
    return hyle_status_invalid_argument("invalid string interner allocator");
  }
  auto* value =
      allocate_handle<hyle_container_string_interner>(backing, backing);
  if (value == nullptr || !value->value.valid()) {
    free_handle(value);
    return hyle_status_resource_exhausted("string interner allocation failed");
  }
  *out = value;
  return hyle_status_ok();
}
void hyle_container_string_interner_destroy(hyle_container_string_interner* interner) {
  free_handle(interner);
}
hyle_status hyle_container_string_interner_acquire(
    hyle_container_string_id* out, hyle_container_string_interner* interner,
    hyle_container_string_view text) {
  if (out != nullptr) *out = 0;
  if (out == nullptr || interner == nullptr ||
      !valid_range(text.data, text.size, "invalid text view"))
    return hyle_status_invalid_argument(
        "output, string interner, and valid text are required");
  const hyle_container_string_id id = interner->value.acquire(to_cpp(text)).value;
  if (text.size != 0 && id == 0)
    return hyle_status_resource_exhausted("string interner allocation failed");
  *out = id;
  return hyle_status_ok();
}
hyle_status hyle_container_string_interner_acquire_many(
    size_t* out_completed,
    hyle_container_string_interner* interner,
    const hyle_container_string_view* input,
    hyle_container_string_id* output, size_t count) {
  if (out_completed != nullptr) *out_completed = 0;
  if (out_completed == nullptr || interner == nullptr ||
      !valid_range(input, count, "non-zero count requires input") ||
      !valid_range(output, count, "non-zero count requires output"))
    return hyle_status_invalid_argument(
        "completed output, interner, input, and output are required");
  for (size_t i = 0; i < count; ++i)
    if (!valid_range(input[i].data, input[i].size, "invalid text view"))
      return hyle_status_invalid_argument("input contains an invalid view");

  constexpr size_t kChunk = 64;
  hc::string_ref cpp_input[kChunk];
  hc::StringId cpp_output[kChunk];
  size_t completed = 0;
  while (completed < count) {
    const size_t chunk = std::min(kChunk, count - completed);
    for (size_t i = 0; i < chunk; ++i)
      cpp_input[i] = to_cpp(input[completed + i]);
    const size_t done =
        interner->value.acquire_many(cpp_input, cpp_output, chunk);
    for (size_t i = 0; i < done; ++i)
      output[completed + i] = cpp_output[i].value;
    completed += done;
    if (done != chunk) {
      *out_completed = completed;
      return hyle_status_resource_exhausted("string interner allocation failed");
    }
  }
  *out_completed = completed;
  return hyle_status_ok();
}
int hyle_container_string_interner_retain(hyle_container_string_interner* interner,
                                          hyle_container_string_id id) {
  if (interner == nullptr) return 0;
  const bool retained = interner->value.retain(hc::StringId{id});
  return retained;
}
size_t hyle_container_string_interner_retain_many(
    hyle_container_string_interner* interner,
    const hyle_container_string_id* ids, size_t count) {
  if (interner == nullptr ||
      !valid_range(ids, count, "non-zero count requires ids"))
    return 0;
  constexpr size_t kChunk = 64;
  hc::StringId cpp_ids[kChunk];
  size_t completed = 0;
  while (completed < count) {
    const size_t chunk = std::min(kChunk, count - completed);
    for (size_t i = 0; i < chunk; ++i)
      cpp_ids[i] = hc::StringId{ids[completed + i]};
    const size_t done = interner->value.retain_many(cpp_ids, chunk);
    completed += done;
    if (done != chunk) return completed;
  }
  return completed;
}
int hyle_container_string_interner_release(hyle_container_string_interner* interner,
                                           hyle_container_string_id id) {
  if (interner == nullptr) return 0;
  return interner->value.release(hc::StringId{id});
}
size_t hyle_container_string_interner_release_many(
    hyle_container_string_interner* interner,
    const hyle_container_string_id* ids, size_t count) {
  if (interner == nullptr ||
      !valid_range(ids, count, "non-zero count requires ids"))
    return 0;
  constexpr size_t kChunk = 64;
  hc::StringId cpp_ids[kChunk];
  size_t completed = 0;
  while (completed < count) {
    const size_t chunk = std::min(kChunk, count - completed);
    for (size_t i = 0; i < chunk; ++i)
      cpp_ids[i] = hc::StringId{ids[completed + i]};
    const size_t done = interner->value.release_many(cpp_ids, chunk);
    completed += done;
    if (done != chunk) return completed;
  }
  return completed;
}
hyle_container_string_view hyle_container_string_interner_lookup(
    hyle_container_string_interner* interner, hyle_container_string_id id) {
  if (interner == nullptr) return {};
  return from_cpp(interner->value.lookup(hc::StringId{id}));
}
hyle_status hyle_container_string_interner_intern(
    hyle_container_string_view* out, hyle_container_string_interner* interner,
    hyle_container_string_view text) {
  if (out != nullptr) *out = {};
  if (out == nullptr || interner == nullptr ||
      !valid_range(text.data, text.size, "invalid text view"))
    return hyle_status_invalid_argument(
        "output, string interner, and valid text are required");
  const hc::string_ref result = interner->value.intern(to_cpp(text));
  if (text.size != 0 && result.data == nullptr)
    return hyle_status_resource_exhausted("string interner allocation failed");
  *out = from_cpp(result);
  return hyle_status_ok();
}
hyle_status hyle_container_string_interner_intern_many(
    size_t* out_completed,
    hyle_container_string_interner* interner,
    const hyle_container_string_view* input,
    hyle_container_string_view* output, size_t count) {
  if (out_completed != nullptr) *out_completed = 0;
  if (out_completed == nullptr || interner == nullptr ||
      !valid_range(input, count, "non-zero count requires input")) {
    return hyle_status_invalid_argument(
        "completed output, interner, and input are required");
  }
  for (size_t i = 0; i < count; ++i)
    if (!valid_range(input[i].data, input[i].size, "invalid text view"))
      return hyle_status_invalid_argument("input contains an invalid view");

  constexpr size_t kChunk = 64;
  hc::string_ref cpp_input[kChunk];
  hc::string_ref cpp_output[kChunk];
  size_t completed = 0;
  while (completed < count) {
    const size_t chunk = std::min(kChunk, count - completed);
    for (size_t i = 0; i < chunk; ++i)
      cpp_input[i] = to_cpp(input[completed + i]);
    const size_t done = interner->value.intern_many(
        cpp_input, output == nullptr ? nullptr : cpp_output, chunk);
    if (output != nullptr)
      for (size_t i = 0; i < done; ++i)
        output[completed + i] = from_cpp(cpp_output[i]);
    completed += done;
    if (done != chunk) {
      *out_completed = completed;
      return hyle_status_resource_exhausted("string interner allocation failed");
    }
  }
  *out_completed = completed;
  return hyle_status_ok();
}
uint32_t hyle_container_string_interner_size(
    const hyle_container_string_interner* interner) {
  if (interner == nullptr) return 0;
  return interner->value.size();
}

hyle_status hyle_container_dynamic_bitset_create(
    hyle_container_dynamic_bitset** out, size_t bit_count, int initial_value,
    hyle_allocator backing) {
  if (out == nullptr)
    return hyle_status_invalid_argument("dynamic bitset output is NULL");
  *out = nullptr;
  if (!valid_backing(backing))
    return hyle_status_invalid_argument("invalid dynamic bitset allocator");
  auto* set = allocate_handle<hyle_container_dynamic_bitset>(backing, backing);
  if (set == nullptr)
    return hyle_status_resource_exhausted(
        "dynamic bitset handle allocation failed");
  if (!set->value.try_resize(bit_count, initial_value != 0)) {
    free_handle(set);
    return hyle_status_resource_exhausted(
        "dynamic bitset storage allocation failed");
  }
  *out = set;
  return hyle_status_ok();
}

void hyle_container_dynamic_bitset_destroy(
    hyle_container_dynamic_bitset* set) {
  free_handle(set);
}

hyle_status hyle_container_dynamic_bitset_reserve(
    hyle_container_dynamic_bitset* set, size_t bit_count) {
  if (set == nullptr)
    return hyle_status_invalid_argument("dynamic bitset is NULL");
  if (!set->value.try_reserve(bit_count))
    return hyle_status_resource_exhausted(
        "dynamic bitset storage allocation failed");
  return hyle_status_ok();
}

hyle_status hyle_container_dynamic_bitset_resize(
    hyle_container_dynamic_bitset* set, size_t bit_count, int value) {
  if (set == nullptr)
    return hyle_status_invalid_argument("dynamic bitset is NULL");
  if (!set->value.try_resize(bit_count, value != 0))
    return hyle_status_resource_exhausted(
        "dynamic bitset storage allocation failed");
  return hyle_status_ok();
}

size_t hyle_container_dynamic_bitset_size(
    const hyle_container_dynamic_bitset* set) {
  return set == nullptr ? 0 : set->value.size();
}

size_t hyle_container_dynamic_bitset_capacity(
    const hyle_container_dynamic_bitset* set) {
  return set == nullptr ? 0 : set->value.capacity();
}

int hyle_container_dynamic_bitset_test(
    const hyle_container_dynamic_bitset* set, size_t index) {
  return set != nullptr && index < set->value.size() && set->value[index];
}

int hyle_container_dynamic_bitset_set(hyle_container_dynamic_bitset* set,
                                      size_t index, int value) {
  if (set == nullptr || index >= set->value.size()) return 0;
  set->value.set(index, value != 0);
  return 1;
}

int hyle_container_dynamic_bitset_flip(hyle_container_dynamic_bitset* set,
                                       size_t index) {
  if (set == nullptr || index >= set->value.size()) return 0;
  set->value.flip(index);
  return 1;
}

void hyle_container_dynamic_bitset_set_all(
    hyle_container_dynamic_bitset* set) {
  if (set != nullptr) set->value.set();
}

void hyle_container_dynamic_bitset_reset_all(
    hyle_container_dynamic_bitset* set) {
  if (set != nullptr) set->value.reset();
}

void hyle_container_dynamic_bitset_flip_all(
    hyle_container_dynamic_bitset* set) {
  if (set != nullptr) set->value.flip();
}

void hyle_container_dynamic_bitset_shift_left(
    hyle_container_dynamic_bitset* set, size_t shift) {
  if (set != nullptr) set->value <<= shift;
}

void hyle_container_dynamic_bitset_shift_right(
    hyle_container_dynamic_bitset* set, size_t shift) {
  if (set != nullptr) set->value >>= shift;
}

size_t hyle_container_dynamic_bitset_count(
    const hyle_container_dynamic_bitset* set) {
  return set == nullptr ? 0 : set->value.count();
}

int hyle_container_dynamic_bitset_any(
    const hyle_container_dynamic_bitset* set) {
  return set != nullptr && set->value.any();
}

int hyle_container_dynamic_bitset_all(
    const hyle_container_dynamic_bitset* set) {
  return set != nullptr && set->value.all();
}

size_t hyle_container_dynamic_bitset_find_first(
    const hyle_container_dynamic_bitset* set) {
  return set == nullptr ? HYLE_CONTAINER_BITSET_NPOS
                        : set->value.find_first();
}

size_t hyle_container_dynamic_bitset_find_next(
    const hyle_container_dynamic_bitset* set, size_t index) {
  return set == nullptr ? HYLE_CONTAINER_BITSET_NPOS
                        : set->value.find_next(index);
}

#define HYLE_BITSET_BINARY_C(name, operation)                                 \
  hyle_status name(hyle_container_dynamic_bitset* destination,               \
                   const hyle_container_dynamic_bitset* source) {            \
    if (destination == nullptr || source == nullptr)                         \
      return hyle_status_invalid_argument("dynamic bitset is NULL");        \
    if (destination->value.size() != source->value.size())                   \
      return hyle_status_invalid_argument("dynamic bitset sizes differ");   \
    destination->value operation source->value;                              \
    return hyle_status_ok();                                                  \
  }
HYLE_BITSET_BINARY_C(hyle_container_dynamic_bitset_and, &=)
HYLE_BITSET_BINARY_C(hyle_container_dynamic_bitset_or, |=)
HYLE_BITSET_BINARY_C(hyle_container_dynamic_bitset_xor, ^=)
#undef HYLE_BITSET_BINARY_C

hyle_status hyle_container_dynamic_bitset_and_not(
    hyle_container_dynamic_bitset* destination,
    const hyle_container_dynamic_bitset* source) {
  if (destination == nullptr || source == nullptr)
    return hyle_status_invalid_argument("dynamic bitset is NULL");
  if (destination->value.size() != source->value.size())
    return hyle_status_invalid_argument("dynamic bitset sizes differ");
  destination->value.and_not(source->value);
  return hyle_status_ok();
}

int hyle_container_dynamic_bitset_intersects(
    const hyle_container_dynamic_bitset* left,
    const hyle_container_dynamic_bitset* right) {
  return left != nullptr && right != nullptr &&
         left->value.size() == right->value.size() &&
         left->value.intersects(right->value);
}

int hyle_container_dynamic_bitset_is_subset(
    const hyle_container_dynamic_bitset* subset,
    const hyle_container_dynamic_bitset* superset) {
  return subset != nullptr && superset != nullptr &&
         subset->value.size() == superset->value.size() &&
         subset->value.is_subset_of(superset->value);
}

hyle_status hyle_container_sparse_u32_set_create(
    hyle_container_sparse_u32_set** out, size_t universe_size,
    hyle_allocator backing) {
  if (out == nullptr)
    return hyle_status_invalid_argument("sparse set output is NULL");
  *out = nullptr;
  if (!valid_backing(backing) ||
      universe_size > static_cast<std::uint64_t>(UINT32_MAX) + 1)
    return hyle_status_invalid_argument("invalid sparse set options");
  auto* set =
      allocate_handle<hyle_container_sparse_u32_set>(backing, backing);
  if (set == nullptr)
    return hyle_status_resource_exhausted(
        "sparse set handle allocation failed");
  if (!set->value.try_reset_universe(universe_size)) {
    free_handle(set);
    return hyle_status_resource_exhausted(
        "sparse set storage allocation failed");
  }
  *out = set;
  return hyle_status_ok();
}

void hyle_container_sparse_u32_set_destroy(
    hyle_container_sparse_u32_set* set) {
  free_handle(set);
}

hyle_status hyle_container_sparse_u32_set_reset_universe(
    hyle_container_sparse_u32_set* set, size_t universe_size) {
  if (set == nullptr ||
      universe_size > static_cast<std::uint64_t>(UINT32_MAX) + 1)
    return hyle_status_invalid_argument("invalid sparse set universe");
  if (!set->value.try_reset_universe(universe_size))
    return hyle_status_resource_exhausted(
        "sparse set storage allocation failed");
  return hyle_status_ok();
}

size_t hyle_container_sparse_u32_set_universe_size(
    const hyle_container_sparse_u32_set* set) {
  return set == nullptr ? 0 : set->value.universe_size();
}

size_t hyle_container_sparse_u32_set_size(
    const hyle_container_sparse_u32_set* set) {
  return set == nullptr ? 0 : set->value.size();
}

const uint32_t* hyle_container_sparse_u32_set_data(
    const hyle_container_sparse_u32_set* set) {
  return set == nullptr ? nullptr : set->value.data();
}

int hyle_container_sparse_u32_set_contains(
    const hyle_container_sparse_u32_set* set, uint32_t id) {
  return set != nullptr && set->value.contains(id);
}

int hyle_container_sparse_u32_set_insert(hyle_container_sparse_u32_set* set,
                                         uint32_t id) {
  return set != nullptr && id < set->value.universe_size() &&
         set->value.insert(id);
}

int hyle_container_sparse_u32_set_erase(hyle_container_sparse_u32_set* set,
                                        uint32_t id) {
  return set != nullptr && set->value.erase(id);
}

void hyle_container_sparse_u32_set_clear(hyle_container_sparse_u32_set* set) {
  if (set != nullptr) set->value.clear();
}

hyle_status hyle_container_byte_ring_create(hyle_container_byte_ring** out,
                                            size_t capacity,
                                            hyle_allocator backing) {
  if (out == nullptr)
    return hyle_status_invalid_argument("byte ring output is NULL");
  * out = nullptr;
  if (!valid_backing(backing))
    return hyle_status_invalid_argument("invalid byte ring allocator");
  if (capacity > SIZE_MAX / 2 + 1)
    return hyle_status_out_of_range(
        "byte ring capacity cannot round to a power of two");
  auto* ring = allocate_handle<hyle_container_byte_ring>(backing, backing);
  if (ring == nullptr)
    return hyle_status_resource_exhausted("byte ring handle allocation failed");
  if (!ring->value.try_reserve(capacity)) {
    free_handle(ring);
    return hyle_status_resource_exhausted(
        "byte ring storage allocation failed");
  }
  * out = ring;
  return hyle_status_ok();
}

void hyle_container_byte_ring_destroy(hyle_container_byte_ring* ring) {
  free_handle(ring);
}

hyle_status hyle_container_byte_ring_reserve(hyle_container_byte_ring* ring,
                                             size_t capacity) {
  if (ring == nullptr)
    return hyle_status_invalid_argument("byte ring is NULL");
  if (capacity > SIZE_MAX / 2 + 1)
    return hyle_status_out_of_range(
        "byte ring capacity cannot round to a power of two");
  if (!ring->value.try_reserve(capacity))
    return hyle_status_resource_exhausted(
        "byte ring storage allocation failed");
  return hyle_status_ok();
}

size_t hyle_container_byte_ring_size(const hyle_container_byte_ring* ring) {
  return ring == nullptr ? 0 : ring->value.size();
}

size_t hyle_container_byte_ring_capacity(const hyle_container_byte_ring* ring) {
  return ring == nullptr ? 0 : ring->value.capacity();
}

size_t
hyle_container_byte_ring_free_space(const hyle_container_byte_ring* ring) {
  return ring == nullptr ? 0 : ring->value.free_space();
}

size_t hyle_container_byte_ring_write(hyle_container_byte_ring* ring,
                                      const void* data, size_t size) {
  if (ring == nullptr || !valid_range(data, size, "invalid byte input"))
    return 0;
  return ring->value.write({static_cast<const std::byte*>(data), size});
}

size_t hyle_container_byte_ring_read(hyle_container_byte_ring* ring,
                                     void* output, size_t capacity) {
  if (ring == nullptr || !valid_output(output, capacity))
    return 0;
  return ring->value.read({static_cast<std::byte*>(output), capacity});
}

size_t
hyle_container_byte_ring_readable(const hyle_container_byte_ring* ring,
                                  hyle_container_const_byte_span out_spans[2]) {
  if (out_spans == nullptr)
    return 0;
  out_spans[0] = {nullptr, 0};
  out_spans[1] = {nullptr, 0};
  if (ring == nullptr)
    return 0;
  const auto spans = ring->value.readable();
  out_spans[0] = {spans.first.data(), spans.first.size()};
  out_spans[1] = {spans.second.data(), spans.second.size()};
  return static_cast<size_t>(!spans.first.empty()) +
         static_cast<size_t>(!spans.second.empty());
}

size_t hyle_container_byte_ring_writable(
    hyle_container_byte_ring* ring,
    hyle_container_mutable_byte_span out_spans[2]) {
  if (out_spans == nullptr)
    return 0;
  out_spans[0] = {nullptr, 0};
  out_spans[1] = {nullptr, 0};
  if (ring == nullptr)
    return 0;
  const auto spans = ring->value.writable();
  out_spans[0] = {spans.first.data(), spans.first.size()};
  out_spans[1] = {spans.second.data(), spans.second.size()};
  return static_cast<size_t>(!spans.first.empty()) +
         static_cast<size_t>(!spans.second.empty());
}

int hyle_container_byte_ring_commit_write(hyle_container_byte_ring* ring,
                                          size_t count) {
  if (ring == nullptr || count > ring->value.free_space())
    return 0;
  ring->value.commit_write(count);
  return 1;
}

int hyle_container_byte_ring_consume(hyle_container_byte_ring* ring,
                                     size_t count) {
  if (ring == nullptr || count > ring->value.size())
    return 0;
  ring->value.consume(count);
  return 1;
}

void hyle_container_byte_ring_clear(hyle_container_byte_ring* ring) {
  if (ring != nullptr)
    ring->value.clear();
}

hyle_status hyle_container_vector_create(
    hyle_container_vector** out,
    const hyle_container_vector_options* options) {
  if (out == nullptr)
    return hyle_status_invalid_argument("vector output is NULL");
  *out = nullptr;
  hyle_container_vector_options decoded =
      HYLE_CONTAINER_VECTOR_OPTIONS_INIT;
  if (!decode_vector_options(options, decoded) || decoded.item_size == 0 ||
      !is_power_of_two(decoded.item_alignment) ||
      !valid_backing(decoded.backing) || decoded.item_alignment > UINT32_MAX ||
      decoded.initial_capacity >
          (SIZE_MAX - sizeof(hc::CArrayHeader) - decoded.item_alignment) /
              decoded.item_size) {
    return hyle_status_invalid_argument("invalid vector options");
  }
  auto* vector = allocate_handle<hyle_container_vector>(decoded.backing);
  if (vector == nullptr) {
    return hyle_status_resource_exhausted("vector handle allocation failed");
  }
  vector->item_size = decoded.item_size;
  vector->item_alignment = decoded.item_alignment;
  vector->backing = decoded.backing;
  if (decoded.initial_capacity != 0) {
    vector->data = hc::detail::carray_set_capacity_internal(
        nullptr, decoded.initial_capacity, vector->item_size,
        vector->item_alignment, to_cpp(&vector->backing));
    if (vector->data == nullptr) {
      free_handle(vector);
      return hyle_status_resource_exhausted(
          "vector storage allocation failed");
    }
  }
  *out = vector;
  return hyle_status_ok();
}
void hyle_container_vector_destroy(hyle_container_vector* vector) {
  if (vector == nullptr) return;
  hc::detail::carray_free_internal(vector->data, vector->item_size);
  vector->data = nullptr;
  free_handle(vector);
}
size_t hyle_container_vector_size(const hyle_container_vector* vector) {
  if (vector == nullptr) return 0;
  return hc::carray_size(vector->data);
}
size_t hyle_container_vector_capacity(const hyle_container_vector* vector) {
  if (vector == nullptr) return 0;
  return hc::carray_capacity(vector->data);
}
void* hyle_container_vector_data(hyle_container_vector* vector) {
  if (vector == nullptr) return nullptr;
  return vector->data;
}
const void* hyle_container_vector_data_const(
    const hyle_container_vector* vector) {
  if (vector == nullptr) return nullptr;
  return vector->data;
}
void* hyle_container_vector_at(hyle_container_vector* vector, size_t index) {
  if (vector == nullptr || index >= hc::carray_size(vector->data))
    return nullptr;
  return static_cast<std::byte*>(vector->data) + index * vector->item_size;
}
hyle_status hyle_container_vector_reserve(hyle_container_vector* vector,
                                          size_t capacity) {
  if (vector == nullptr)
    return hyle_status_invalid_argument("vector is NULL");
  if (capacity <= hc::carray_capacity(vector->data)) return hyle_status_ok();
  if (capacity >
      (SIZE_MAX - sizeof(hc::CArrayHeader) - vector->item_alignment) /
          vector->item_size) {
    return hyle_status_out_of_range(
        "vector capacity overflows address space");
  }
  void* replacement = hc::detail::carray_set_capacity_internal(
      vector->data, capacity, vector->item_size, vector->item_alignment,
      to_cpp(&vector->backing));
  if (replacement == nullptr) {
    return hyle_status_resource_exhausted("vector storage allocation failed");
  }
  vector->data = replacement;
  return hyle_status_ok();
}
hyle_status hyle_container_vector_try_push_back(hyle_container_vector* vector,
                                                const void* item) {
  return hyle_container_vector_try_append(vector, item, 1);
}
hyle_status hyle_container_vector_try_append(hyle_container_vector* vector,
                                             const void* items,
                                             size_t count) {
  if (vector == nullptr || (count != 0 && items == nullptr))
    return hyle_status_invalid_argument(
        vector == nullptr ? "vector is NULL" : "items are NULL");
  if (count == 0) return hyle_status_ok();
  const size_t size = hc::carray_size(vector->data);
  if (count > SIZE_MAX - size || count > SIZE_MAX / vector->item_size)
    return hyle_status_out_of_range("vector size overflows address space");
  std::size_t aliased_offset = 0;
  const VectorSource source =
      classify_vector_source(vector, items, count, aliased_offset);
  if (source == VectorSource::kInvalid)
    return hyle_status_invalid_argument(
        "aliased items must be entirely inside live vector data");
  hyle_status status = vector_reserve_for(vector, size + count);
  if (!hyle_status_is_ok(status)) return status;
  hyle_status_free(&status);
  if (source == VectorSource::kLiveAlias)
    items = static_cast<const std::byte*>(vector->data) + aliased_offset;
  std::memmove(static_cast<std::byte*>(vector->data) +
                   size * vector->item_size,
               items, count * vector->item_size);
  hc::carray_header(vector->data)->size = size + count;
  return hyle_status_ok();
}
hyle_status hyle_container_vector_try_resize(hyle_container_vector* vector,
                                             size_t requested,
                                             const void* fill_item) {
  if (vector == nullptr)
    return hyle_status_invalid_argument("vector is NULL");
  const size_t size = hc::carray_size(vector->data);
  if (requested <= size) {
    if (vector->data != nullptr)
      hc::carray_header(vector->data)->size = requested;
    return hyle_status_ok();
  }
  if (fill_item == nullptr)
    return hyle_status_invalid_argument(
        "growing a vector requires fill_item");
  std::size_t aliased_offset = 0;
  const VectorSource source =
      classify_vector_source(vector, fill_item, 1, aliased_offset);
  if (source == VectorSource::kInvalid)
    return hyle_status_invalid_argument(
        "aliased fill_item must be entirely inside live vector data");
  hyle_status status = vector_reserve_for(vector, requested);
  if (!hyle_status_is_ok(status)) return status;
  hyle_status_free(&status);
  if (source == VectorSource::kLiveAlias)
    fill_item = static_cast<const std::byte*>(vector->data) + aliased_offset;
  fill_vector_items(static_cast<std::byte*>(vector->data) +
                        size * vector->item_size,
                    requested - size, fill_item, vector->item_size);
  hc::carray_header(vector->data)->size = requested;
  return hyle_status_ok();
}
hyle_status hyle_container_vector_try_insert(hyle_container_vector* vector,
                                             size_t index,
                                             const void* items,
                                             size_t count) {
  if (vector == nullptr || (count != 0 && items == nullptr))
    return hyle_status_invalid_argument(
        vector == nullptr ? "vector is NULL" : "items are NULL");
  const size_t size = hc::carray_size(vector->data);
  if (index > size)
    return hyle_status_out_of_range("vector insert index is out of range");
  if (count == 0) return hyle_status_ok();
  if (count > SIZE_MAX - size || count > SIZE_MAX / vector->item_size)
    return hyle_status_out_of_range("vector size overflows address space");

  std::size_t aliased_offset = 0;
  const VectorSource source =
      classify_vector_source(vector, items, count, aliased_offset);
  if (source == VectorSource::kInvalid)
    return hyle_status_invalid_argument(
        "aliased items must be entirely inside live vector data");
  const size_t input_bytes = count * vector->item_size;
  void* scratch = nullptr;
  if (source == VectorSource::kLiveAlias) {
    scratch = vector->backing.alloc(vector->backing.state, input_bytes,
                                    vector->item_alignment);
    if (scratch == nullptr)
      return hyle_status_resource_exhausted(
          "vector alias scratch allocation failed");
    std::memcpy(scratch,
                static_cast<const std::byte*>(vector->data) + aliased_offset,
                input_bytes);
    items = scratch;
  }

  hyle_status status = vector_reserve_for(vector, size + count);
  if (!hyle_status_is_ok(status)) {
    if (scratch != nullptr)
      vector->backing.free(vector->backing.state, scratch, input_bytes,
                           vector->item_alignment);
    return status;
  }
  hyle_status_free(&status);
  auto* data = static_cast<std::byte*>(vector->data);
  std::memmove(data + (index + count) * vector->item_size,
               data + index * vector->item_size,
               (size - index) * vector->item_size);
  std::memmove(data + index * vector->item_size, items, input_bytes);
  hc::carray_header(vector->data)->size = size + count;
  if (scratch != nullptr)
    vector->backing.free(vector->backing.state, scratch, input_bytes,
                         vector->item_alignment);
  return hyle_status_ok();
}
int hyle_container_vector_try_pop_back(hyle_container_vector* vector,
                                       void* out_item) {
  if (vector == nullptr || out_item == nullptr) return 0;
  if (hc::carray_size(vector->data) == 0) return 0;
  const size_t index = hc::carray_size(vector->data) - 1;
  std::memmove(out_item, static_cast<std::byte*>(vector->data) +
                             index * vector->item_size,
               vector->item_size);
  hc::carray_header(vector->data)->size = index;
  return 1;
}
hyle_status hyle_container_vector_erase(hyle_container_vector* vector,
                                        size_t index, size_t count) {
  if (vector == nullptr)
    return hyle_status_invalid_argument("vector is NULL");
  const size_t size = hc::carray_size(vector->data);
  if (index > size || count > size - index)
    return hyle_status_out_of_range("vector erase range is out of bounds");
  if (count == 0) return hyle_status_ok();
  auto* data = static_cast<std::byte*>(vector->data);
  std::memmove(data + index * vector->item_size,
               data + (index + count) * vector->item_size,
               (size - index - count) * vector->item_size);
  hc::carray_header(vector->data)->size = size - count;
  return hyle_status_ok();
}
hyle_status hyle_container_vector_try_shrink_to_fit(
    hyle_container_vector* vector) {
  if (vector == nullptr)
    return hyle_status_invalid_argument("vector is NULL");
  const size_t size = hc::carray_size(vector->data);
  if (vector->data == nullptr || hc::carray_capacity(vector->data) == size)
    return hyle_status_ok();
  if (size == 0) {
    hc::detail::carray_free_internal(vector->data, vector->item_size);
    vector->data = nullptr;
    return hyle_status_ok();
  }
  void* replacement = hc::detail::carray_set_capacity_internal(
      vector->data, size, vector->item_size, vector->item_alignment,
      to_cpp(&vector->backing));
  if (replacement == nullptr)
    return hyle_status_resource_exhausted(
        "vector storage shrink allocation failed");
  vector->data = replacement;
  return hyle_status_ok();
}
void hyle_container_vector_clear(hyle_container_vector* vector) {
  if (vector == nullptr) return;
  if (vector->data != nullptr)
    hc::carray_header(vector->data)->size = 0;
}

hyle_status hyle_container_hash_map_create(
    hyle_container_hash_map** out,
    const hyle_container_hash_map_options* options) {
  if (out == nullptr)
    return hyle_status_invalid_argument("hash map output is NULL");
  *out = nullptr;
  hyle_container_hash_map_options decoded =
      HYLE_CONTAINER_HASH_MAP_OPTIONS_INIT;
  if (!decode_hash_map_options(options, decoded) || decoded.value_size == 0 ||
      !valid_backing(decoded.backing) ||
      !valid_hash_callbacks(decoded.hash, decoded.equal) ||
      !valid_hash_slot_layout(decoded.key_size, decoded.key_alignment,
                              decoded.value_size, decoded.value_alignment)) {
    return hyle_status_invalid_argument("invalid hash map options");
  }
  auto* map =
      allocate_handle<hyle_container_hash_map>(decoded.backing, decoded);
  if (map == nullptr)
    return hyle_status_resource_exhausted("hash map allocation failed");
  *out = map;
  return hyle_status_ok();
}
void hyle_container_hash_map_destroy(hyle_container_hash_map* map) {
  free_handle(map);
}
size_t hyle_container_hash_map_size(const hyle_container_hash_map* map) {
  if (map == nullptr) return 0;
  return map->value.size;
}
size_t hyle_container_hash_map_capacity(const hyle_container_hash_map* map) {
  if (map == nullptr) return 0;
  return map->value.cap;
}
hyle_status hyle_container_hash_map_reserve(hyle_container_hash_map* map,
                                            size_t count) {
  if (map == nullptr)
    return hyle_status_invalid_argument("map is NULL");
  if (count > SIZE_MAX / 8)
    return hyle_status_out_of_range(
        "hash map capacity overflows address space");
  if (!map->value.reserve(count))
    return hyle_status_resource_exhausted(
        "hash map bucket allocation failed");
  return hyle_status_ok();
}
void hyle_container_hash_map_clear(hyle_container_hash_map* map) {
  if (map == nullptr) return;
  map->value.clear();
}
static hyle_status raw_hash_insert_prehashed_impl(
    hyle_container_insert_result* out_result,
    RawMap& raw, const void* key, const void* value,
    bool assign_existing, uint64_t hash) {
  *out_result = HYLE_CONTAINER_ALREADY_PRESENT;
  const std::size_t existing = raw.find_slot(key, hash);
  if (existing != RawMap::kNoSlot) {
    if (assign_existing && raw.options.value_size != 0) {
      std::memmove(raw.value(existing), value, raw.options.value_size);
      *out_result = HYLE_CONTAINER_ASSIGNED;
    }
    return hyle_status_ok();
  }

  if (raw.cap == 0 && !raw.resize(hc::detail::kGroupWidth)) {
    return hyle_status_resource_exhausted(
        "hash table storage allocation failed");
  }
  std::size_t target = raw.find_insert_slot(hash);
  if (raw.growth_left == 0 &&
      raw.ctrl[target] == hc::detail::kEmpty) {
    // The C facade permits insertion from pointers returned by find(). Preserve
    // such inputs across the flat-table reallocation, but keep the normal path
    // allocation-free.
    const bool key_alias =
        raw.aliases_slots(key, raw.options.key_size);
    const bool value_alias = raw.options.value_size != 0 &&
                             raw.aliases_slots(value, raw.options.value_size);
    void* scratch = nullptr;
    const void* stable_key = key;
    const void* stable_value = value;
    if (key_alias || value_alias) {
      const std::size_t scratch_size =
          raw.options.key_size + raw.options.value_size;
      scratch = raw.alloc.allocate(scratch_size, raw.block_alignment);
      if (scratch == nullptr) {
        return hyle_status_resource_exhausted(
            "hash table alias scratch allocation failed");
      }
      auto* bytes = static_cast<std::byte*>(scratch);
      std::memmove(bytes, key, raw.options.key_size);
      if (raw.options.value_size != 0)
        std::memmove(bytes + raw.options.key_size, value,
                     raw.options.value_size);
      stable_key = bytes;
      if (raw.options.value_size != 0)
        stable_value = bytes + raw.options.key_size;
    }
    if (raw.cap > SIZE_MAX / 2 ||
        !raw.resize(raw.size >= raw.cap - raw.cap / 8 ? raw.cap * 2
                                                       : raw.cap)) {
      if (scratch != nullptr)
        raw.alloc.deallocate(scratch,
                             raw.options.key_size + raw.options.value_size,
                             raw.block_alignment);
      return hyle_status_resource_exhausted(
          "hash table storage allocation failed");
    }
    target = raw.find_insert_slot(hash);
    raw.set_hash(target, hash);
    std::memcpy(raw.key(target), stable_key, raw.options.key_size);
    if (raw.options.value_size != 0)
      std::memcpy(raw.value(target), stable_value, raw.options.value_size);
    if (scratch != nullptr)
      raw.alloc.deallocate(scratch,
                           raw.options.key_size + raw.options.value_size,
                           raw.block_alignment);
  } else {
    raw.set_hash(target, hash);
    std::memcpy(raw.key(target), key, raw.options.key_size);
    if (raw.options.value_size != 0)
      std::memcpy(raw.value(target), value, raw.options.value_size);
  }
  const bool was_empty = raw.ctrl[target] == hc::detail::kEmpty;
  raw.set_ctrl(target, hc::detail::h2(hash));
  ++raw.size;
  if (was_empty) --raw.growth_left;
  *out_result = HYLE_CONTAINER_INSERTED;
  return hyle_status_ok();
}
static hyle_status raw_hash_insert_impl(
    hyle_container_insert_result* out_result, RawMap& raw, const void* key,
    const void* value, bool assign_existing) {
  return raw_hash_insert_prehashed_impl(out_result, raw, key, value,
                                        assign_existing, raw.hash(key));
}
hyle_status hyle_container_hash_map_insert(
    hyle_container_insert_result* out_result,
    hyle_container_hash_map* map, const void* key, const void* value) {
  if (out_result != nullptr)
    *out_result = HYLE_CONTAINER_ALREADY_PRESENT;
  if (out_result == nullptr || map == nullptr || key == nullptr ||
      value == nullptr)
    return hyle_status_invalid_argument(
        "result, map, key, and value are required");
  return raw_hash_insert_impl(out_result, map->value, key, value, false);
}
hyle_status hyle_container_hash_map_insert_or_assign(
    hyle_container_insert_result* out_result,
    hyle_container_hash_map* map, const void* key, const void* value) {
  if (out_result != nullptr)
    *out_result = HYLE_CONTAINER_ALREADY_PRESENT;
  if (out_result == nullptr || map == nullptr || key == nullptr ||
      value == nullptr)
    return hyle_status_invalid_argument(
        "result, map, key, and value are required");
  return raw_hash_insert_impl(out_result, map->value, key, value, true);
}
void* hyle_container_hash_map_find(hyle_container_hash_map* map,
                                   const void* key) {
  if (map == nullptr || key == nullptr) return nullptr;
  const uint64_t hash = map->value.hash(key);
  const std::size_t slot = map->value.find_slot(key, hash);
  return slot == RawMap::kNoSlot ? nullptr : map->value.value(slot);
}
const void* hyle_container_hash_map_find_const(
    const hyle_container_hash_map* map, const void* key) {
  if (map == nullptr || key == nullptr) return nullptr;
  const uint64_t hash = map->value.hash(key);
  const std::size_t slot = map->value.find_slot(key, hash);
  return slot == RawMap::kNoSlot ? nullptr : map->value.value(slot);
}
int hyle_container_hash_map_contains(const hyle_container_hash_map* map,
                                     const void* key) {
  if (map == nullptr || key == nullptr) return 0;
  const uint64_t hash = map->value.hash(key);
  const bool present =
      map->value.find_slot(key, hash) != RawMap::kNoSlot;
  return present;
}
int hyle_container_hash_map_erase(hyle_container_hash_map* map,
                                  const void* key) {
  if (map == nullptr || key == nullptr) return 0;
  RawMap& raw = map->value;
  const uint64_t hash = raw.hash(key);
  const std::size_t slot = raw.find_slot(key, hash);
  if (slot == RawMap::kNoSlot) return 0;
  raw.erase_slot(slot);
  return 1;
}
int hyle_container_hash_map_visit(
    hyle_container_hash_map* map, hyle_container_hash_map_visit_fn visitor,
    void* context) {
  if (map == nullptr || visitor == nullptr) return 0;
  bool keep_going = true;
  RawMap& raw = map->value;
  for (std::size_t group = 0; keep_going && group < raw.cap;
       group += hc::detail::kGroupWidth)
    for (std::uint32_t lane :
         hc::detail::Group(raw.ctrl + group).match_full()) {
      const std::size_t index = group + lane;
      keep_going =
          visitor(raw.key(index), raw.value(index), context) != 0;
      if (!keep_going) break;
    }
  return keep_going ? 1 : 0;
}

hyle_status hyle_container_concurrent_hash_map_create(
    hyle_container_concurrent_hash_map** out,
    const hyle_container_hash_map_options* options, size_t shard_count) {
  if (out == nullptr)
    return hyle_status_invalid_argument(
        "concurrent hash map output is NULL");
  *out = nullptr;
  hyle_container_hash_map_options decoded =
      HYLE_CONTAINER_HASH_MAP_OPTIONS_INIT;
  if (!decode_hash_map_options(options, decoded) || decoded.value_size == 0 ||
      !valid_backing(decoded.backing) ||
      !valid_hash_callbacks(decoded.hash, decoded.equal) ||
      shard_count == 0 || !is_power_of_two(shard_count) ||
      !valid_hash_slot_layout(decoded.key_size, decoded.key_alignment,
                              decoded.value_size, decoded.value_alignment)) {
    return hyle_status_invalid_argument(
        "invalid concurrent hash map options or shard count");
  }
  auto* map = allocate_handle<hyle_container_concurrent_hash_map>(
      decoded.backing, decoded, shard_count);
  if (map == nullptr || map->shard_count != shard_count) {
    free_handle(map);
    return hyle_status_resource_exhausted(
        "concurrent hash map allocation failed");
  }
  *out = map;
  return hyle_status_ok();
}

void hyle_container_concurrent_hash_map_destroy(
    hyle_container_concurrent_hash_map* map) {
  free_handle(map);
}

size_t hyle_container_concurrent_hash_map_shard_count(
    const hyle_container_concurrent_hash_map* map) {
  return map == nullptr ? 0 : map->shard_count;
}

size_t hyle_container_concurrent_hash_map_size(
    const hyle_container_concurrent_hash_map* map) {
  if (map == nullptr) return 0;
  std::size_t result = 0;
  for (std::size_t i = 0; i < map->shard_count; ++i) {
    ConcurrentRawShard& shard = map->shards[i];
    shard.lock.lock_shared();
    result += shard.value.size;
    shard.lock.unlock_shared();
  }
  return result;
}

size_t hyle_container_concurrent_hash_map_capacity(
    const hyle_container_concurrent_hash_map* map) {
  if (map == nullptr) return 0;
  std::size_t result = 0;
  for (std::size_t i = 0; i < map->shard_count; ++i) {
    ConcurrentRawShard& shard = map->shards[i];
    shard.lock.lock_shared();
    result += shard.value.cap;
    shard.lock.unlock_shared();
  }
  return result;
}

hyle_status hyle_container_concurrent_hash_map_reserve(
    hyle_container_concurrent_hash_map* map, size_t count) {
  if (map == nullptr)
    return hyle_status_invalid_argument("concurrent map is NULL");
  if (count > SIZE_MAX - (map->shard_count - 1))
    return hyle_status_out_of_range(
        "concurrent hash map capacity overflows address space");
  const std::size_t per_shard =
      (count + map->shard_count - 1) / map->shard_count;
  for (std::size_t i = 0; i < map->shard_count; ++i) {
    ConcurrentRawShard& shard = map->shards[i];
    shard.lock.lock();
    const bool reserved = shard.value.reserve(per_shard);
    shard.lock.unlock();
    if (!reserved)
      return hyle_status_resource_exhausted(
          "concurrent hash map bucket allocation failed");
  }
  return hyle_status_ok();
}

void hyle_container_concurrent_hash_map_clear(
    hyle_container_concurrent_hash_map* map) {
  if (map == nullptr) return;
  for (std::size_t i = 0; i < map->shard_count; ++i) {
    ConcurrentRawShard& shard = map->shards[i];
    shard.lock.lock();
    shard.value.clear();
    shard.lock.unlock();
  }
}

static hyle_status concurrent_raw_hash_insert_impl(
    hyle_container_insert_result* out_result,
    hyle_container_concurrent_hash_map* map, const void* key,
    const void* value, bool assign_existing) {
  if (out_result != nullptr)
    *out_result = HYLE_CONTAINER_ALREADY_PRESENT;
  if (out_result == nullptr || map == nullptr || key == nullptr ||
      value == nullptr)
    return hyle_status_invalid_argument(
        "result, concurrent map, key, and value are required");
  const std::uint64_t hash = map->shards[0].value.hash(key);
  ConcurrentRawShard& shard = map->shard(hash);
  shard.lock.lock();
  hyle_status status = raw_hash_insert_prehashed_impl(
      out_result, shard.value, key, value, assign_existing, hash);
  shard.lock.unlock();
  return status;
}

hyle_status hyle_container_concurrent_hash_map_insert(
    hyle_container_insert_result* out_result,
    hyle_container_concurrent_hash_map* map, const void* key,
    const void* value) {
  return concurrent_raw_hash_insert_impl(out_result, map, key, value, false);
}

hyle_status hyle_container_concurrent_hash_map_insert_or_assign(
    hyle_container_insert_result* out_result,
    hyle_container_concurrent_hash_map* map, const void* key,
    const void* value) {
  return concurrent_raw_hash_insert_impl(out_result, map, key, value, true);
}

int hyle_container_concurrent_hash_map_find_copy(
    void* out_value, const hyle_container_concurrent_hash_map* map,
    const void* key) {
  if (out_value == nullptr || map == nullptr || key == nullptr) return 0;
  const std::uint64_t hash = map->shards[0].value.hash(key);
  ConcurrentRawShard& shard = map->shard(hash);
  shard.lock.lock_shared();
  const std::size_t slot = shard.value.find_slot(key, hash);
  if (slot != RawMap::kNoSlot)
    std::memcpy(out_value, shard.value.value(slot),
                shard.value.options.value_size);
  shard.lock.unlock_shared();
  return slot != RawMap::kNoSlot ? 1 : 0;
}

int hyle_container_concurrent_hash_map_contains(
    const hyle_container_concurrent_hash_map* map, const void* key) {
  if (map == nullptr || key == nullptr) return 0;
  const std::uint64_t hash = map->shards[0].value.hash(key);
  ConcurrentRawShard& shard = map->shard(hash);
  shard.lock.lock_shared();
  const bool present =
      shard.value.find_slot(key, hash) != RawMap::kNoSlot;
  shard.lock.unlock_shared();
  return present ? 1 : 0;
}

int hyle_container_concurrent_hash_map_update(
    hyle_container_concurrent_hash_map* map, const void* key,
    hyle_container_concurrent_hash_map_update_fn updater, void* context) {
  if (map == nullptr || key == nullptr || updater == nullptr) return 0;
  const std::uint64_t hash = map->shards[0].value.hash(key);
  ConcurrentRawShard& shard = map->shard(hash);
  shard.lock.lock();
  const std::size_t slot = shard.value.find_slot(key, hash);
  if (slot != RawMap::kNoSlot)
    updater(shard.value.value(slot), context);
  shard.lock.unlock();
  return slot != RawMap::kNoSlot ? 1 : 0;
}

int hyle_container_concurrent_hash_map_erase(
    hyle_container_concurrent_hash_map* map, const void* key) {
  if (map == nullptr || key == nullptr) return 0;
  const std::uint64_t hash = map->shards[0].value.hash(key);
  ConcurrentRawShard& shard = map->shard(hash);
  shard.lock.lock();
  const std::size_t slot = shard.value.find_slot(key, hash);
  if (slot != RawMap::kNoSlot) shard.value.erase_slot(slot);
  shard.lock.unlock();
  return slot != RawMap::kNoSlot ? 1 : 0;
}

int hyle_container_concurrent_hash_map_visit(
    const hyle_container_concurrent_hash_map* map,
    hyle_container_concurrent_hash_map_visit_fn visitor, void* context) {
  if (map == nullptr || visitor == nullptr) return 0;
  for (std::size_t i = 0; i < map->shard_count; ++i) {
    ConcurrentRawShard& shard = map->shards[i];
    shard.lock.lock_shared();
    const RawMap& raw = shard.value;
    bool keep_going = true;
    for (std::size_t group = 0; keep_going && group < raw.cap;
         group += hc::detail::kGroupWidth) {
      for (std::uint32_t lane :
           hc::detail::Group(raw.ctrl + group).match_full()) {
        const std::size_t index = group + lane;
        keep_going =
            visitor(raw.key(index), raw.value(index), context) != 0;
        if (!keep_going) break;
      }
    }
    shard.lock.unlock_shared();
    if (!keep_going) return 0;
  }
  return 1;
}

hyle_status hyle_container_hash_set_create(
    hyle_container_hash_set** out,
    const hyle_container_hash_set_options* options) {
  if (out == nullptr)
    return hyle_status_invalid_argument("hash set output is NULL");
  *out = nullptr;
  hyle_container_hash_set_options decoded =
      HYLE_CONTAINER_HASH_SET_OPTIONS_INIT;
  if (!decode_hash_set_options(options, decoded) ||
      !valid_backing(decoded.backing) ||
      !valid_hash_callbacks(decoded.hash, decoded.equal) ||
      !valid_hash_slot_layout(decoded.key_size, decoded.key_alignment, 0, 1)) {
    return hyle_status_invalid_argument("invalid hash set options");
  }
  auto* set =
      allocate_handle<hyle_container_hash_set>(decoded.backing, decoded);
  if (set == nullptr)
    return hyle_status_resource_exhausted("hash set allocation failed");
  *out = set;
  return hyle_status_ok();
}
void hyle_container_hash_set_destroy(hyle_container_hash_set* set) {
  free_handle(set);
}
size_t hyle_container_hash_set_size(const hyle_container_hash_set* set) {
  if (set == nullptr) return 0;
  return set->value.size;
}
size_t hyle_container_hash_set_capacity(const hyle_container_hash_set* set) {
  if (set == nullptr) return 0;
  return set->value.cap;
}
hyle_status hyle_container_hash_set_reserve(hyle_container_hash_set* set,
                                            size_t count) {
  if (set == nullptr)
    return hyle_status_invalid_argument("set is NULL");
  if (count > SIZE_MAX / 8)
    return hyle_status_out_of_range(
        "hash set capacity overflows address space");
  if (!set->value.reserve(count))
    return hyle_status_resource_exhausted(
        "hash set bucket allocation failed");
  return hyle_status_ok();
}
void hyle_container_hash_set_clear(hyle_container_hash_set* set) {
  if (set == nullptr) return;
  set->value.clear();
}
hyle_status hyle_container_hash_set_insert(
    hyle_container_insert_result* out_result,
    hyle_container_hash_set* set, const void* key) {
  if (out_result != nullptr)
    *out_result = HYLE_CONTAINER_ALREADY_PRESENT;
  if (out_result == nullptr || set == nullptr || key == nullptr)
    return hyle_status_invalid_argument(
        "result, set, and key are required");
  return raw_hash_insert_impl(out_result, set->value, key, nullptr, false);
}
const void* hyle_container_hash_set_find(
    const hyle_container_hash_set* set, const void* key) {
  if (set == nullptr || key == nullptr) return nullptr;
  const uint64_t hash = set->value.hash(key);
  const std::size_t slot = set->value.find_slot(key, hash);
  return slot == RawMap::kNoSlot ? nullptr : set->value.key(slot);
}
int hyle_container_hash_set_contains(const hyle_container_hash_set* set,
                                     const void* key) {
  return hyle_container_hash_set_find(set, key) != nullptr;
}
int hyle_container_hash_set_erase(hyle_container_hash_set* set,
                                  const void* key) {
  if (set == nullptr || key == nullptr) return 0;
  RawMap& raw = set->value;
  const uint64_t hash = raw.hash(key);
  const std::size_t slot = raw.find_slot(key, hash);
  if (slot == RawMap::kNoSlot) return 0;
  raw.erase_slot(slot);
  return 1;
}
int hyle_container_hash_set_visit(
    const hyle_container_hash_set* set,
    hyle_container_hash_set_visit_fn visitor, void* context) {
  if (set == nullptr || visitor == nullptr) return 0;
  const RawMap& raw = set->value;
  bool keep_going = true;
  for (std::size_t group = 0; keep_going && group < raw.cap;
       group += hc::detail::kGroupWidth)
    for (std::uint32_t lane :
         hc::detail::Group(raw.ctrl + group).match_full()) {
      const std::size_t index = group + lane;
      keep_going = visitor(raw.key(index), context) != 0;
      if (!keep_going) break;
    }
  return keep_going ? 1 : 0;
}

static bool valid_queue_capacity(size_t capacity) {
  return capacity >= 2 && is_power_of_two(capacity) &&
         capacity <= (UINT32_C(1) << 31);
}

#define HYLE_QUEUE_CREATE(name)                                                \
  hyle_status name##_create(name** out, hyle_allocator backing,                \
                            size_t capacity) {                                  \
    if (out == nullptr)                                                        \
      return hyle_status_invalid_argument("queue output is NULL");            \
    *out = nullptr;                                                            \
    if (!valid_backing(backing) || !valid_queue_capacity(capacity))            \
      return hyle_status_invalid_argument(                                     \
          "invalid allocator or queue capacity");                             \
    auto* queue = allocate_handle<name>(backing, backing, capacity);           \
    if (queue == nullptr || !queue->value.valid()) {                           \
      free_handle(queue);                                                      \
      return hyle_status_resource_exhausted("queue allocation failed");       \
    }                                                                          \
    *out = queue;                                                              \
    return hyle_status_ok();                                                   \
  }                                                                            \
  void name##_destroy(name* queue) { free_handle(queue); }                     \
  size_t name##_capacity(const name* queue) {                                  \
    if (queue == nullptr) return 0;                                            \
    return queue->value.capacity();                                             \
  }

HYLE_QUEUE_CREATE(hyle_container_bounded_queue)
HYLE_QUEUE_CREATE(hyle_container_work_stealing_deque)
HYLE_QUEUE_CREATE(hyle_container_ws_fifo_queue)
#undef HYLE_QUEUE_CREATE

int hyle_container_bounded_queue_try_push(hyle_container_bounded_queue* queue,
                                          void* item) {
  if (queue == nullptr) return 0;
  return queue->value.try_push(item);
}
int hyle_container_bounded_queue_try_pop(hyle_container_bounded_queue* queue,
                                         void** out_item) {
  if (queue == nullptr || out_item == nullptr) return 0;
  return queue->value.try_pop(*out_item);
}
hyle_status hyle_container_spsc_queue_create(
    hyle_container_spsc_queue** out, hyle_allocator backing,
    size_t capacity) {
  if (out == nullptr)
    return hyle_status_invalid_argument("SPSC queue output is NULL");
  *out = nullptr;
  if (!valid_backing(backing) || capacity == 0 ||
      capacity >= SIZE_MAX / sizeof(void*))
    return hyle_status_invalid_argument(
        "invalid allocator or SPSC queue capacity");
  auto* queue = allocate_handle<hyle_container_spsc_queue>(
      backing, backing, capacity);
  if (queue == nullptr || !queue->value.valid()) {
    free_handle(queue);
    return hyle_status_resource_exhausted("SPSC queue allocation failed");
  }
  *out = queue;
  return hyle_status_ok();
}
void hyle_container_spsc_queue_destroy(hyle_container_spsc_queue* queue) {
  free_handle(queue);
}
size_t hyle_container_spsc_queue_capacity(
    const hyle_container_spsc_queue* queue) {
  if (queue == nullptr) return 0;
  return queue->value.capacity();
}
int hyle_container_spsc_queue_try_push(hyle_container_spsc_queue* queue,
                                       void* item) {
  if (queue == nullptr) return 0;
  return queue->value.try_push(item);
}
int hyle_container_spsc_queue_try_pop(hyle_container_spsc_queue* queue,
                                      void** out_item) {
  if (queue == nullptr || out_item == nullptr) return 0;
  return queue->value.try_pop(*out_item);
}
size_t hyle_container_spsc_queue_size_approx(
    const hyle_container_spsc_queue* queue) {
  if (queue == nullptr) return 0;
  return queue->value.size_approx();
}
hyle_status hyle_container_sharded_queue_create(
    hyle_container_sharded_queue** out, hyle_allocator backing,
    size_t shard_count, size_t capacity_per_shard) {
  if (out == nullptr)
    return hyle_status_invalid_argument("sharded queue output is NULL");
  *out = nullptr;
  if (!valid_backing(backing) || shard_count == 0 ||
      !valid_queue_capacity(capacity_per_shard)) {
    return hyle_status_invalid_argument(
        "invalid allocator or sharded queue dimensions");
  }
  if (shard_count >
          SIZE_MAX /
              sizeof(hc::BoundedQueue<void*, hyle::mem::Allocator>) ||
      capacity_per_shard > SIZE_MAX / shard_count) {
    return hyle_status_out_of_range(
        "sharded queue dimensions overflow address space");
  }
  auto* queue = allocate_handle<hyle_container_sharded_queue>(
      backing, backing, shard_count, capacity_per_shard);
  if (queue == nullptr || !queue->value.valid()) {
    free_handle(queue);
    return hyle_status_resource_exhausted(
        "sharded queue allocation failed");
  }
  *out = queue;
  return hyle_status_ok();
}
void hyle_container_sharded_queue_destroy(
    hyle_container_sharded_queue* queue) {
  free_handle(queue);
}
int hyle_container_sharded_queue_try_push(hyle_container_sharded_queue* queue,
                                          size_t producer_hint, void* item) {
  if (queue == nullptr) return 0;
  return queue->value.try_push(producer_hint, item);
}
int hyle_container_sharded_queue_try_pop(hyle_container_sharded_queue* queue,
                                         size_t* cursor, void** out_item) {
  if (queue == nullptr || cursor == nullptr || out_item == nullptr) return 0;
  return queue->value.try_pop(*cursor, *out_item);
}
size_t hyle_container_sharded_queue_capacity(
    const hyle_container_sharded_queue* queue) {
  if (queue == nullptr) return 0;
  return queue->value.capacity();
}
size_t hyle_container_sharded_queue_shard_count(
    const hyle_container_sharded_queue* queue) {
  if (queue == nullptr) return 0;
  return queue->value.num_shards();
}
int hyle_container_work_stealing_deque_try_push_bottom(
    hyle_container_work_stealing_deque* deque, void* item) {
  if (deque == nullptr) return 0;
  return deque->value.try_push_bottom(item);
}
int hyle_container_work_stealing_deque_try_pop_bottom(
    hyle_container_work_stealing_deque* deque, void** out_item) {
  if (deque == nullptr || out_item == nullptr) return 0;
  return deque->value.try_pop_bottom(*out_item);
}
hyle_container_steal_status hyle_container_work_stealing_deque_try_steal_top(
    hyle_container_work_stealing_deque* deque, void** out_item) {
  if (deque == nullptr || out_item == nullptr)
    return HYLE_CONTAINER_STEAL_EMPTY;
  return static_cast<hyle_container_steal_status>(
      deque->value.try_steal_top(*out_item));
}
int hyle_container_ws_fifo_queue_try_push_back(
    hyle_container_ws_fifo_queue* queue, void* item) {
  if (queue == nullptr) return 0;
  return queue->value.try_push_back(item);
}
int hyle_container_ws_fifo_queue_try_pop_front(
    hyle_container_ws_fifo_queue* queue, void** out_item) {
  if (queue == nullptr || out_item == nullptr) return 0;
  return queue->value.try_pop_front(*out_item);
}
int hyle_container_ws_fifo_queue_try_steal_into(
    hyle_container_ws_fifo_queue* source, hyle_container_ws_fifo_queue* target,
    void** out_item) {
  if (source == nullptr || target == nullptr || out_item == nullptr ||
      source == target || source->value.capacity() != target->value.capacity()) {
    return 0;
  }
  return source->value.try_steal_into(target->value, *out_item);
}

hyle_status hyle_container_concurrent_ptr_vector_create(
    hyle_container_concurrent_ptr_vector** out, hyle_allocator backing,
    size_t first_segment_capacity) {
  if (out == nullptr)
    return hyle_status_invalid_argument(
        "concurrent pointer vector output is NULL");
  *out = nullptr;
  if (!valid_backing(backing)) {
    return hyle_status_invalid_argument(
        "invalid concurrent pointer vector allocator");
  }
  const size_t max_first_segment =
      (SIZE_MAX / sizeof(void*) + 1) / 2;
  if (first_segment_capacity > max_first_segment)
    return hyle_status_out_of_range(
        "concurrent pointer vector segment capacity overflows address space");
  auto* vector = allocate_handle<hyle_container_concurrent_ptr_vector>(
      backing, backing, first_segment_capacity);
  if (vector == nullptr)
    return hyle_status_resource_exhausted(
        "concurrent pointer vector allocation failed");
  *out = vector;
  return hyle_status_ok();
}
void hyle_container_concurrent_ptr_vector_destroy(
    hyle_container_concurrent_ptr_vector* vector) {
  free_handle(vector);
}
hyle_status hyle_container_concurrent_ptr_vector_reserve(
    hyle_container_concurrent_ptr_vector* vector, size_t count) {
  if (vector == nullptr)
    return hyle_status_invalid_argument(
        "concurrent pointer vector is NULL");
  if (count > SIZE_MAX / sizeof(void*))
    return hyle_status_out_of_range(
        "concurrent pointer vector capacity overflows address space");
  if (!vector->value.try_reserve(count))
    return hyle_status_resource_exhausted(
        "concurrent pointer vector storage allocation failed");
  return hyle_status_ok();
}
hyle_status hyle_container_concurrent_ptr_vector_try_push_back(
    size_t* out_index, hyle_container_concurrent_ptr_vector* vector,
    void* item) {
  if (out_index != nullptr) *out_index = SIZE_MAX;
  if (out_index == nullptr || vector == nullptr)
    return hyle_status_invalid_argument(
        "index output and concurrent pointer vector are required");
  const size_t index = vector->value.try_push_back(item);
  if (index == SIZE_MAX)
    return hyle_status_resource_exhausted(
        "concurrent pointer vector storage allocation failed");
  *out_index = index;
  return hyle_status_ok();
}
size_t hyle_container_concurrent_ptr_vector_size(
    const hyle_container_concurrent_ptr_vector* vector) {
  if (vector == nullptr) return 0;
  return vector->value.size();
}
size_t hyle_container_concurrent_ptr_vector_capacity(
    const hyle_container_concurrent_ptr_vector* vector) {
  if (vector == nullptr) return 0;
  return vector->value.capacity();
}
int hyle_container_concurrent_ptr_vector_try_get(
    const hyle_container_concurrent_ptr_vector* vector, size_t index,
    void** out_item) {
  if (vector == nullptr || out_item == nullptr || index >= vector->value.size())
    return 0;
  *out_item = vector->value[index];
  return 1;
}

}  // extern "C"

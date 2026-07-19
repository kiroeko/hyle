#include "hyle/c/container.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/container/bounded_queue.h"
#include "hyle/container/carray.h"
#include "hyle/container/concurrent_vector.h"
#include "hyle/container/path.h"
#include "hyle/container/probe.h"
#include "hyle/container/sharded_queue.h"
#include "hyle/container/string_algo.h"
#include "hyle/container/string_buf.h"
#include "hyle/container/swiss_table.h"
#include "hyle/container/work_stealing_deque.h"
#include "hyle/container/ws_fifo_queue.h"
#include "hyle/mem/allocator.h"

namespace hc = hyle::container;

namespace {

thread_local const char* g_last_error = "";

void clear_error() noexcept { g_last_error = ""; }
void set_error(const char* message) noexcept { g_last_error = message; }

bool is_power_of_two(std::size_t value) noexcept {
  return value != 0 && (value & (value - 1)) == 0;
}

bool valid_backing(const hyle_allocator& backing) noexcept {
  return backing.alloc != nullptr && backing.free != nullptr &&
         backing.realloc != nullptr;
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
                 const char* message) noexcept {
  if (size != 0 && pointer == nullptr) {
    set_error(message);
    return false;
  }
  return true;
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
      : options(options), alloc(alloc) {
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
    hc::detail::swiss_free(storage, block_size, block_alignment, alloc);
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
               : hc::hash_bytes(input, options.key_size);
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
      if (hc::detail::BitMask reusable = group.match_empty_or_deleted())
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
    const hc::detail::BitMask empty_after =
        hc::detail::Group(ctrl + index).match_empty();
    const hc::detail::BitMask empty_before =
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
  bool resize(std::size_t new_capacity) noexcept {
    hc::detail::SwissStorage storage{ctrl, slots, cap, mask, size,
                                     growth_left};
    if (!hc::detail::swiss_resize(storage, new_capacity, block_size,
                                  block_alignment, alloc, &slot_hash, nullptr))
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
    if (count <= cap - cap / 8) return true;
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

struct hyle_container_string_buf_o {
  hyle_allocator backing;
  hc::StringBuf value;
  explicit hyle_container_string_buf_o(hyle_allocator allocator)
      : backing(allocator), value(to_cpp(&backing)) {}
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
struct hyle_container_bounded_queue_o {
  hyle_allocator backing;
  hyle_container_bounded_queue_o(hyle_allocator allocator,
                                 std::size_t capacity)
      : backing(allocator), value(to_cpp(&backing), capacity) {}
  hc::BoundedQueue<void*, hyle::mem::Allocator> value;
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
struct hyle_container_concurrent_vector_o {
  hyle_allocator backing;
  hyle_container_concurrent_vector_o(hyle_allocator allocator,
                                     std::size_t capacity)
      : backing(allocator), value(to_cpp(&backing), capacity) {}
  hc::ConcurrentVector<void*, hyle::mem::Allocator> value;
};

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
static_assert(static_cast<int>(hc::StealStatus::kOk) ==
              HYLE_CONTAINER_STEAL_OK);
static_assert(static_cast<int>(hc::StealStatus::kEmpty) ==
              HYLE_CONTAINER_STEAL_EMPTY);
static_assert(static_cast<int>(hc::StealStatus::kAbort) ==
              HYLE_CONTAINER_STEAL_ABORT);

extern "C" {

const char* hyle_container_last_error(void) { return g_last_error; }

uint64_t hyle_container_hash_bytes(const void* data, size_t size,
                                   uint64_t seed) {
  if (!valid_range(data, size, "non-zero hash size requires data")) return 0;
  clear_error();
  return hc::hash_bytes(data, size, seed);
}

int hyle_container_string_compare(hyle_container_string_view left,
                                  hyle_container_string_view right) {
  if (!valid_range(left.data, left.size, "invalid left string view") ||
      !valid_range(right.data, right.size, "invalid right string view"))
    return 0;
  clear_error();
  const auto result = to_cpp(left) <=> to_cpp(right);
  return result < 0 ? -1 : result > 0 ? 1 : 0;
}

int hyle_container_string_equal_ignore_case(
    hyle_container_string_view left, hyle_container_string_view right) {
  if (!valid_range(left.data, left.size, "invalid left string view") ||
      !valid_range(right.data, right.size, "invalid right string view"))
    return 0;
  clear_error();
  return hc::equal_ignore_case(to_cpp(left), to_cpp(right));
}

#define HYLE_STRING_SEARCH_IMPL(name, method)                                  \
  uint32_t name(hyle_container_string_view text,                               \
                hyle_container_string_view needle, uint32_t position) {        \
    if (!valid_range(text.data, text.size, "invalid text view") ||             \
        !valid_range(needle.data, needle.size, "invalid needle view"))         \
      return HYLE_CONTAINER_NPOS;                                              \
    clear_error();                                                             \
    return to_cpp(text).method(to_cpp(needle), position);                      \
  }
HYLE_STRING_SEARCH_IMPL(hyle_container_string_find, find)
HYLE_STRING_SEARCH_IMPL(hyle_container_string_rfind, rfind)
#undef HYLE_STRING_SEARCH_IMPL

uint32_t hyle_container_string_find_char(hyle_container_string_view text,
                                         char needle, uint32_t position) {
  if (!valid_range(text.data, text.size, "invalid text view"))
    return HYLE_CONTAINER_NPOS;
  clear_error();
  return to_cpp(text).find(needle, position);
}
uint32_t hyle_container_string_rfind_char(hyle_container_string_view text,
                                          char needle, uint32_t position) {
  if (!valid_range(text.data, text.size, "invalid text view"))
    return HYLE_CONTAINER_NPOS;
  clear_error();
  return to_cpp(text).rfind(needle, position);
}
uint32_t hyle_container_string_count_char(hyle_container_string_view text,
                                          char needle) {
  if (!valid_range(text.data, text.size, "invalid text view")) return 0;
  clear_error();
  return to_cpp(text).count(needle);
}

#define HYLE_STRING_SET_SEARCH_IMPL(name, method)                              \
  uint32_t name(hyle_container_string_view text,                              \
                hyle_container_string_view set, uint32_t position) {          \
    if (!valid_range(text.data, text.size, "invalid text view") ||            \
        !valid_range(set.data, set.size, "invalid character set view"))       \
      return HYLE_CONTAINER_NPOS;                                             \
    clear_error();                                                            \
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
    clear_error();                                                            \
    return to_cpp(text).method(to_cpp(part));                                 \
  }
HYLE_STRING_PREDICATE_IMPL(hyle_container_string_starts_with, starts_with)
HYLE_STRING_PREDICATE_IMPL(hyle_container_string_ends_with, ends_with)
HYLE_STRING_PREDICATE_IMPL(hyle_container_string_contains, contains)
#undef HYLE_STRING_PREDICATE_IMPL

hyle_container_string_view hyle_container_string_slice(
    hyle_container_string_view text, uint32_t start, uint32_t length) {
  if (!valid_range(text.data, text.size, "invalid text view")) return {};
  clear_error();
  return from_cpp(to_cpp(text).substr(start, length));
}
hyle_container_string_view hyle_container_string_split(
    hyle_container_string_view* text, char delimiter) {
  if (text == nullptr ||
      !valid_range(text->data, text->size, "invalid text view")) {
    if (text == nullptr) set_error("text is NULL");
    return {};
  }
  hc::string_ref remainder = to_cpp(*text);
  const hc::string_ref result = hc::split(&remainder, delimiter);
  *text = from_cpp(remainder);
  clear_error();
  return from_cpp(result);
}
hyle_container_string_view hyle_container_string_rsplit(
    hyle_container_string_view* text, char delimiter) {
  if (text == nullptr ||
      !valid_range(text->data, text->size, "invalid text view")) {
    if (text == nullptr) set_error("text is NULL");
    return {};
  }
  hc::string_ref remainder = to_cpp(*text);
  const hc::string_ref result = hc::rsplit(&remainder, delimiter);
  *text = from_cpp(remainder);
  clear_error();
  return from_cpp(result);
}
hyle_container_string_view hyle_container_string_trim(
    hyle_container_string_view text, hyle_container_string_view cut) {
  if (!valid_range(text.data, text.size, "invalid text view") ||
      !valid_range(cut.data, cut.size, "invalid cut view"))
    return {};
  clear_error();
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
  clear_error();
  return text.size;
}
size_t hyle_container_string_concat(hyle_container_string_view left,
                                    hyle_container_string_view right,
                                    char* out, size_t capacity) {
  if (!valid_range(left.data, left.size, "invalid left string view") ||
      !valid_range(right.data, right.size, "invalid right string view") ||
      !valid_output(out, capacity))
    return 0;
  clear_error();
  return copy_concat(to_cpp(left), to_cpp(right), out, capacity, false);
}

size_t hyle_container_string_lower(hyle_container_string_view text, char* out,
                                   size_t capacity) {
  if (!valid_range(text.data, text.size, "invalid text view") ||
      !valid_output(out, capacity))
    return 0;
  clear_error();
  if (capacity > UINT32_MAX) capacity = UINT32_MAX;
  return hc::to_lower(to_cpp(text), out, static_cast<uint32_t>(capacity));
}
size_t hyle_container_string_upper(hyle_container_string_view text, char* out,
                                   size_t capacity) {
  if (!valid_range(text.data, text.size, "invalid text view") ||
      !valid_output(out, capacity))
    return 0;
  clear_error();
  if (capacity > UINT32_MAX) capacity = UINT32_MAX;
  return hc::to_upper(to_cpp(text), out, static_cast<uint32_t>(capacity));
}
size_t hyle_container_hex_encode(const void* data, size_t size, char* out,
                                 size_t capacity) {
  if (!valid_range(data, size, "non-zero input size requires data") ||
      !valid_output(out, capacity))
    return 0;
  clear_error();
  return hc::hex_encode(data, size, out, capacity);
}
size_t hyle_container_hex_decode(hyle_container_string_view hex, void* out,
                                 size_t capacity) {
  if (!valid_range(hex.data, hex.size, "invalid hexadecimal view") ||
      !valid_output(out, capacity))
    return SIZE_MAX;
  const size_t result = hc::hex_decode(to_cpp(hex), out, capacity);
  if (result == SIZE_MAX) set_error("malformed hexadecimal input");
  else clear_error();
  return result;
}
size_t hyle_container_string_escape(hyle_container_string_view text, char* out,
                                    size_t capacity) {
  if (!valid_range(text.data, text.size, "invalid text view") ||
      !valid_output(out, capacity))
    return 0;
  clear_error();
  return hc::escape(to_cpp(text), out, capacity);
}
void hyle_container_string_capitalize(char* text) {
  if (text == nullptr) {
    set_error("text is NULL");
    return;
  }
  hc::capitalize(text);
  clear_error();
}
int64_t hyle_container_utf8_length(hyle_container_string_view text) {
  if (!valid_range(text.data, text.size, "invalid text view")) return -1;
  const int64_t result = hc::utf8_length(to_cpp(text));
  if (result < 0)
    set_error("malformed UTF-8 input");
  else
    clear_error();
  return result;
}
int hyle_container_utf8_icmp(hyle_container_string_view left,
                             hyle_container_string_view right,
                             int codepoint_limit) {
  if (!valid_range(left.data, left.size, "invalid left string view") ||
      !valid_range(right.data, right.size, "invalid right string view") ||
      codepoint_limit < 0) {
    if (codepoint_limit < 0) set_error("codepoint_limit must be non-negative");
    return 0;
  }
  if (hc::utf8_length(to_cpp(left)) < 0 ||
      hc::utf8_length(to_cpp(right)) < 0) {
    set_error("malformed UTF-8 input");
    return 0;
  }
  clear_error();
  return hc::utf8_icmp(to_cpp(left), to_cpp(right), codepoint_limit);
}
int hyle_container_fuzzy_score(hyle_container_string_view candidate,
                               hyle_container_string_view pattern,
                               int path_aware) {
  if (!valid_range(candidate.data, candidate.size, "invalid candidate view") ||
      !valid_range(pattern.data, pattern.size, "invalid pattern view"))
    return 0;
  clear_error();
  return path_aware ? hc::path_match_score(to_cpp(candidate), to_cpp(pattern))
                    : hc::fuzzy_match_score(to_cpp(candidate), to_cpp(pattern));
}
uint32_t hyle_container_find_path_matches(
    const hyle_container_string_view* candidates, uint32_t count,
    hyle_container_string_view pattern, uint32_t* out_indices, int* out_scores,
    uint32_t output_capacity, int path_aware) {
  if (!valid_range(candidates, count, "non-zero count requires candidates") ||
      !valid_range(pattern.data, pattern.size, "invalid pattern view") ||
      !valid_range(out_indices, output_capacity,
                   "non-zero output capacity requires indices") ||
      !valid_range(out_scores, output_capacity,
                   "non-zero output capacity requires scores"))
    return 0;
  for (uint32_t i = 0; i < count; ++i)
    if (!valid_range(candidates[i].data, candidates[i].size,
                     "invalid candidate view"))
      return 0;
  clear_error();
  if (output_capacity == 0 || count == 0) return 0;
  // Keep the same O(n log k), caller-buffer min-heap as the C++ implementation.
  // Candidate views are converted one at a time: no cross-ABI reinterpret cast
  // and no hidden temporary allocation.
  auto worse = [](int left_score, uint32_t left_index, int right_score,
                  uint32_t right_index) noexcept {
    return left_score < right_score ||
           (left_score == right_score && left_index > right_index);
  };
  uint32_t written = 0;
  for (uint32_t i = 0; i < count; ++i) {
    const int score = path_aware
                          ? hc::path_match_score(to_cpp(candidates[i]),
                                                 to_cpp(pattern))
                          : hc::fuzzy_match_score(to_cpp(candidates[i]),
                                                  to_cpp(pattern));
    if (score <= 0) continue;
    if (written < output_capacity) {
      uint32_t child = written++;
      out_indices[child] = i;
      out_scores[child] = score;
      while (child != 0) {
        const uint32_t parent = (child - 1) / 2;
        if (!worse(out_scores[child], out_indices[child],
                   out_scores[parent], out_indices[parent]))
          break;
        std::swap(out_scores[child], out_scores[parent]);
        std::swap(out_indices[child], out_indices[parent]);
        child = parent;
      }
    } else if (worse(out_scores[0], out_indices[0], score, i)) {
      out_indices[0] = i;
      out_scores[0] = score;
      uint32_t child = 0;
      for (;;) {
        const uint64_t left_wide = static_cast<uint64_t>(child) * 2 + 1;
        if (left_wide >= written) break;
        const uint32_t left = static_cast<uint32_t>(left_wide);
        const uint32_t right = left + 1;
        uint32_t smallest = child;
        if (left < written &&
            worse(out_scores[left], out_indices[left],
                  out_scores[smallest], out_indices[smallest]))
          smallest = left;
        if (right < written &&
            worse(out_scores[right], out_indices[right],
                  out_scores[smallest], out_indices[smallest]))
          smallest = right;
        if (smallest == child) break;
        std::swap(out_scores[child], out_scores[smallest]);
        std::swap(out_indices[child], out_indices[smallest]);
        child = smallest;
      }
    }
  }
  for (uint32_t heap_size = written; heap_size > 1;) {
    --heap_size;
    std::swap(out_scores[0], out_scores[heap_size]);
    std::swap(out_indices[0], out_indices[heap_size]);
    uint32_t child = 0;
    for (;;) {
      const uint64_t left_wide = static_cast<uint64_t>(child) * 2 + 1;
      if (left_wide >= heap_size) break;
      const uint32_t left = static_cast<uint32_t>(left_wide);
      const uint32_t right = left + 1;
      uint32_t smallest = child;
      if (left < heap_size &&
          worse(out_scores[left], out_indices[left],
                out_scores[smallest], out_indices[smallest]))
        smallest = left;
      if (right < heap_size &&
          worse(out_scores[right], out_indices[right],
                out_scores[smallest], out_indices[smallest]))
        smallest = right;
      if (smallest == child) break;
      std::swap(out_scores[child], out_scores[smallest]);
      std::swap(out_indices[child], out_indices[smallest]);
      child = smallest;
    }
  }
  clear_error();
  return written;
}

#define HYLE_PATH_VIEW_IMPL(name, method)                                      \
  hyle_container_string_view name(hyle_container_string_view path) {           \
    if (!valid_range(path.data, path.size, "invalid path view")) return {};    \
    clear_error();                                                             \
    return from_cpp(hc::method(to_cpp(path)));                                 \
  }
HYLE_PATH_VIEW_IMPL(hyle_container_path_extension, extension)
HYLE_PATH_VIEW_IMPL(hyle_container_path_strip_extension, strip_extension)
HYLE_PATH_VIEW_IMPL(hyle_container_path_base, base)
HYLE_PATH_VIEW_IMPL(hyle_container_path_directory, directory)
#undef HYLE_PATH_VIEW_IMPL

int hyle_container_path_is_absolute(hyle_container_string_view path) {
  if (!valid_range(path.data, path.size, "invalid path view")) return 0;
  clear_error();
  return hc::is_absolute(to_cpp(path));
}
int hyle_container_path_is_unc(hyle_container_string_view path) {
  if (!valid_range(path.data, path.size, "invalid path view")) return 0;
  clear_error();
  return hc::is_unc(to_cpp(path));
}
hyle_container_string_view hyle_container_path_pop_first(
    hyle_container_string_view* path) {
  if (path == nullptr ||
      !valid_range(path->data, path->size, "invalid path view")) {
    if (path == nullptr) set_error("path is NULL");
    return {};
  }
  hc::string_ref value = to_cpp(*path);
  const hc::string_ref result = hc::pop_first(&value);
  *path = from_cpp(value);
  clear_error();
  return from_cpp(result);
}
hyle_container_string_view hyle_container_path_pop_last(
    hyle_container_string_view* path) {
  if (path == nullptr ||
      !valid_range(path->data, path->size, "invalid path view")) {
    if (path == nullptr) set_error("path is NULL");
    return {};
  }
  hc::string_ref value = to_cpp(*path);
  const hc::string_ref result = hc::pop_last(&value);
  *path = from_cpp(value);
  clear_error();
  return from_cpp(result);
}
size_t hyle_container_path_join(hyle_container_string_view left,
                                hyle_container_string_view right, char* out,
                                size_t capacity) {
  if (!valid_range(left.data, left.size, "invalid left path view") ||
      !valid_range(right.data, right.size, "invalid right path view") ||
      !valid_output(out, capacity))
    return 0;
  clear_error();
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
  clear_error();
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
  clear_error();
  return count;
}
size_t hyle_container_path_replace_extension(
    hyle_container_string_view path, hyle_container_string_view extension,
    char* out, size_t capacity) {
  if (!valid_range(path.data, path.size, "invalid path view") ||
      !valid_range(extension.data, extension.size, "invalid extension view") ||
      !valid_output(out, capacity))
    return 0;
  clear_error();
  return copy_concat(hc::strip_extension(to_cpp(path)), to_cpp(extension), out,
                     capacity, false);
}
uint32_t hyle_container_path_normalize(char* path) {
  if (path == nullptr) {
    set_error("path is NULL");
    return 0;
  }
  clear_error();
  return hc::normalize(path);
}

hyle_container_string_buf* hyle_container_string_buf_create(
    hyle_allocator backing) {
  if (!valid_backing(backing)) {
    set_error("invalid string buffer allocator");
    return nullptr;
  }
  auto* value =
      allocate_handle<hyle_container_string_buf>(backing, backing);
  if (value == nullptr || !value->value.valid()) {
    free_handle(value);
    value = nullptr;
    set_error("string buffer allocation failed");
  } else {
    clear_error();
  }
  return value;
}
void hyle_container_string_buf_destroy(hyle_container_string_buf* buffer) {
  free_handle(buffer);
}
hyle_container_intern_id hyle_container_string_buf_add(
    hyle_container_string_buf* buffer, hyle_container_string_view text) {
  if (buffer == nullptr ||
      !valid_range(text.data, text.size, "invalid text view")) {
    if (buffer == nullptr) set_error("string buffer is NULL");
    return 0;
  }
  const hyle_container_intern_id id = buffer->value.add(to_cpp(text)).value;
  if (text.size != 0 && id == 0)
    set_error("string buffer allocation failed");
  else
    clear_error();
  return id;
}
size_t hyle_container_string_buf_add_many(
    hyle_container_string_buf* buffer,
    const hyle_container_string_view* input,
    hyle_container_intern_id* output, size_t count) {
  if (buffer == nullptr ||
      !valid_range(input, count, "non-zero count requires input") ||
      !valid_range(output, count, "non-zero count requires output")) {
    if (buffer == nullptr) set_error("string buffer is NULL");
    return 0;
  }
  for (size_t i = 0; i < count; ++i)
    if (!valid_range(input[i].data, input[i].size, "invalid text view"))
      return 0;

  constexpr size_t kChunk = 64;
  hc::string_ref cpp_input[kChunk];
  hc::InternId cpp_output[kChunk];
  size_t completed = 0;
  while (completed < count) {
    const size_t chunk = std::min(kChunk, count - completed);
    for (size_t i = 0; i < chunk; ++i)
      cpp_input[i] = to_cpp(input[completed + i]);
    const size_t done =
        buffer->value.add_many(cpp_input, cpp_output, chunk);
    for (size_t i = 0; i < done; ++i)
      output[completed + i] = cpp_output[i].value;
    completed += done;
    if (done != chunk) {
      set_error("string buffer allocation failed");
      return completed;
    }
  }
  clear_error();
  return completed;
}
int hyle_container_string_buf_retain(hyle_container_string_buf* buffer,
                                     hyle_container_intern_id id) {
  if (buffer == nullptr) {
    set_error("string buffer is NULL");
    return 0;
  }
  const bool retained = buffer->value.retain(hc::InternId{id});
  clear_error();
  return retained;
}
size_t hyle_container_string_buf_retain_many(
    hyle_container_string_buf* buffer,
    const hyle_container_intern_id* ids, size_t count) {
  if (buffer == nullptr ||
      !valid_range(ids, count, "non-zero count requires ids")) {
    if (buffer == nullptr) set_error("string buffer is NULL");
    return 0;
  }
  constexpr size_t kChunk = 64;
  hc::InternId cpp_ids[kChunk];
  size_t completed = 0;
  while (completed < count) {
    const size_t chunk = std::min(kChunk, count - completed);
    for (size_t i = 0; i < chunk; ++i)
      cpp_ids[i] = hc::InternId{ids[completed + i]};
    const size_t done = buffer->value.retain_many(cpp_ids, chunk);
    completed += done;
    if (done != chunk) {
      clear_error();
      return completed;
    }
  }
  clear_error();
  return completed;
}
int hyle_container_string_buf_remove(hyle_container_string_buf* buffer,
                                     hyle_container_intern_id id) {
  if (buffer == nullptr) {
    set_error("string buffer is NULL");
    return 0;
  }
  const bool removed = buffer->value.remove(hc::InternId{id});
  clear_error();
  return removed;
}
size_t hyle_container_string_buf_remove_many(
    hyle_container_string_buf* buffer,
    const hyle_container_intern_id* ids, size_t count) {
  if (buffer == nullptr ||
      !valid_range(ids, count, "non-zero count requires ids")) {
    if (buffer == nullptr) set_error("string buffer is NULL");
    return 0;
  }
  constexpr size_t kChunk = 64;
  hc::InternId cpp_ids[kChunk];
  size_t completed = 0;
  while (completed < count) {
    const size_t chunk = std::min(kChunk, count - completed);
    for (size_t i = 0; i < chunk; ++i)
      cpp_ids[i] = hc::InternId{ids[completed + i]};
    const size_t done = buffer->value.remove_many(cpp_ids, chunk);
    completed += done;
    if (done != chunk) {
      clear_error();
      return completed;
    }
  }
  clear_error();
  return completed;
}
hyle_container_string_view hyle_container_string_buf_lookup(
    hyle_container_string_buf* buffer, hyle_container_intern_id id) {
  if (buffer == nullptr) {
    set_error("string buffer is NULL");
    return {};
  }
  clear_error();
  return from_cpp(buffer->value.lookup(hc::InternId{id}));
}
hyle_container_string_view hyle_container_string_buf_intern(
    hyle_container_string_buf* buffer, hyle_container_string_view text) {
  if (buffer == nullptr ||
      !valid_range(text.data, text.size, "invalid text view")) {
    if (buffer == nullptr) set_error("string buffer is NULL");
    return {};
  }
  const hc::string_ref result = buffer->value.intern(to_cpp(text));
  if (text.size != 0 && result.data == nullptr)
    set_error("string buffer allocation failed");
  else
    clear_error();
  return from_cpp(result);
}
size_t hyle_container_string_buf_intern_many(
    hyle_container_string_buf* buffer,
    const hyle_container_string_view* input,
    hyle_container_string_view* output, size_t count) {
  if (buffer == nullptr ||
      !valid_range(input, count, "non-zero count requires input")) {
    if (buffer == nullptr) set_error("string buffer is NULL");
    return 0;
  }
  for (size_t i = 0; i < count; ++i)
    if (!valid_range(input[i].data, input[i].size, "invalid text view"))
      return 0;

  constexpr size_t kChunk = 64;
  hc::string_ref cpp_input[kChunk];
  hc::string_ref cpp_output[kChunk];
  size_t completed = 0;
  while (completed < count) {
    const size_t chunk = std::min(kChunk, count - completed);
    for (size_t i = 0; i < chunk; ++i)
      cpp_input[i] = to_cpp(input[completed + i]);
    const size_t done = buffer->value.intern_many(
        cpp_input, output == nullptr ? nullptr : cpp_output, chunk);
    if (output != nullptr)
      for (size_t i = 0; i < done; ++i)
        output[completed + i] = from_cpp(cpp_output[i]);
    completed += done;
    if (done != chunk) {
      set_error("string buffer allocation failed");
      return completed;
    }
  }
  clear_error();
  return completed;
}
uint32_t hyle_container_string_buf_size(
    const hyle_container_string_buf* buffer) {
  if (buffer == nullptr) {
    set_error("string buffer is NULL");
    return 0;
  }
  clear_error();
  return buffer->value.size();
}

hyle_container_vector* hyle_container_vector_create(
    const hyle_container_vector_options* options) {
  if (options == nullptr ||
      options->struct_size < sizeof(hyle_container_vector_options) ||
      options->item_size == 0 ||
      !is_power_of_two(options->item_alignment) ||
      !valid_backing(options->backing) ||
      options->item_alignment > UINT32_MAX ||
      options->initial_capacity >
          (SIZE_MAX - sizeof(hc::CArrayHeader) - options->item_alignment) /
              options->item_size) {
    set_error("invalid vector options");
    return nullptr;
  }
  auto* vector =
      allocate_handle<hyle_container_vector>(options->backing);
  if (vector == nullptr) {
    set_error("vector handle allocation failed");
    return nullptr;
  }
  vector->item_size = options->item_size;
  vector->item_alignment = options->item_alignment;
  vector->backing = options->backing;
  if (options->initial_capacity != 0) {
    vector->data = hc::detail::carray_set_capacity_internal(
        nullptr, options->initial_capacity, vector->item_size,
        vector->item_alignment, to_cpp(&vector->backing));
    if (vector->data == nullptr) {
      free_handle(vector);
      set_error("vector storage allocation failed");
      return nullptr;
    }
  }
  clear_error();
  return vector;
}
void hyle_container_vector_destroy(hyle_container_vector* vector) {
  if (vector == nullptr) return;
  hc::detail::carray_free_internal(vector->data, vector->item_size);
  vector->data = nullptr;
  free_handle(vector);
}
size_t hyle_container_vector_size(const hyle_container_vector* vector) {
  if (vector == nullptr) {
    set_error("vector is NULL");
    return 0;
  }
  clear_error();
  return hc::carray_size(vector->data);
}
size_t hyle_container_vector_capacity(const hyle_container_vector* vector) {
  if (vector == nullptr) {
    set_error("vector is NULL");
    return 0;
  }
  clear_error();
  return hc::carray_capacity(vector->data);
}
void* hyle_container_vector_data(hyle_container_vector* vector) {
  if (vector == nullptr) {
    set_error("vector is NULL");
    return nullptr;
  }
  clear_error();
  return vector->data;
}
const void* hyle_container_vector_data_const(
    const hyle_container_vector* vector) {
  if (vector == nullptr) {
    set_error("vector is NULL");
    return nullptr;
  }
  clear_error();
  return vector->data;
}
void* hyle_container_vector_at(hyle_container_vector* vector, size_t index) {
  if (vector == nullptr || index >= hc::carray_size(vector->data)) {
    set_error("vector index out of range");
    return nullptr;
  }
  clear_error();
  return static_cast<std::byte*>(vector->data) + index * vector->item_size;
}
int hyle_container_vector_reserve(hyle_container_vector* vector,
                                  size_t capacity) {
  if (vector == nullptr) {
    set_error("vector is NULL");
    return 0;
  }
  if (capacity <= hc::carray_capacity(vector->data)) {
    clear_error();
    return 1;
  }
  if (capacity >
      (SIZE_MAX - sizeof(hc::CArrayHeader) - vector->item_alignment) /
          vector->item_size) {
    set_error("vector capacity overflows address space");
    return 0;
  }
  void* replacement = hc::detail::carray_set_capacity_internal(
      vector->data, capacity, vector->item_size, vector->item_alignment,
      to_cpp(&vector->backing));
  if (replacement == nullptr) {
    set_error("vector storage allocation failed");
    return 0;
  }
  vector->data = replacement;
  clear_error();
  return 1;
}
int hyle_container_vector_push(hyle_container_vector* vector,
                               const void* item) {
  if (vector == nullptr || item == nullptr) {
    set_error(vector == nullptr ? "vector is NULL" : "item is NULL");
    return 0;
  }
  const size_t size = hc::carray_size(vector->data);
  std::size_t aliased_offset = SIZE_MAX;
  if (vector->data != nullptr) {
    const auto begin = reinterpret_cast<std::uintptr_t>(vector->data);
    const auto end = begin + size * vector->item_size;
    const auto address = reinterpret_cast<std::uintptr_t>(item);
    if (address >= begin && address < end) {
      if (address > end - vector->item_size) {
        set_error("aliased item does not contain a complete element");
        return 0;
      }
      aliased_offset = address - begin;
    }
  }
  if (size == hc::carray_capacity(vector->data)) {
    const size_t next = size == 0 ? 4 : size * 2;
    if (next < size || !hyle_container_vector_reserve(vector, next)) return 0;
  }
  if (aliased_offset != SIZE_MAX)
    item = static_cast<const std::byte*>(vector->data) + aliased_offset;
  std::memcpy(static_cast<std::byte*>(vector->data) +
                  size * vector->item_size,
              item, vector->item_size);
  hc::carray_header(vector->data)->size = size + 1;
  clear_error();
  return 1;
}
int hyle_container_vector_pop(hyle_container_vector* vector, void* out_item) {
  if (vector == nullptr || out_item == nullptr) {
    set_error("vector and output are required");
    return 0;
  }
  if (hc::carray_size(vector->data) == 0) {
    clear_error();
    return 0;
  }
  const size_t index = hc::carray_size(vector->data) - 1;
  std::memcpy(out_item, static_cast<std::byte*>(vector->data) +
                            index * vector->item_size,
              vector->item_size);
  hc::carray_header(vector->data)->size = index;
  clear_error();
  return 1;
}
void hyle_container_vector_clear(hyle_container_vector* vector) {
  if (vector == nullptr) {
    set_error("vector is NULL");
    return;
  }
  if (vector->data != nullptr)
    hc::carray_header(vector->data)->size = 0;
  clear_error();
}

hyle_container_hash_map* hyle_container_hash_map_create(
    const hyle_container_hash_map_options* options) {
  if (options == nullptr ||
      options->struct_size < sizeof(hyle_container_hash_map_options) ||
      options->key_size == 0 || options->value_size == 0 ||
      !is_power_of_two(options->key_alignment) ||
      !is_power_of_two(options->value_alignment) ||
      !valid_backing(options->backing) ||
      options->key_alignment > SIZE_MAX / 4 ||
      options->value_alignment > SIZE_MAX / 4 ||
      options->key_size >
          SIZE_MAX - sizeof(std::uint64_t) - options->key_alignment ||
      options->value_size >
          SIZE_MAX - sizeof(std::uint64_t) - options->key_alignment -
              options->key_size - options->value_alignment) {
    set_error("invalid hash map options");
    return nullptr;
  }
  auto* map = allocate_handle<hyle_container_hash_map>(
      options->backing, *options);
  if (map == nullptr) set_error("hash map allocation failed");
  else clear_error();
  return map;
}
void hyle_container_hash_map_destroy(hyle_container_hash_map* map) {
  free_handle(map);
}
size_t hyle_container_hash_map_size(const hyle_container_hash_map* map) {
  if (map == nullptr) {
    set_error("map is NULL");
    return 0;
  }
  clear_error();
  return map->value.size;
}
size_t hyle_container_hash_map_capacity(const hyle_container_hash_map* map) {
  if (map == nullptr) {
    set_error("map is NULL");
    return 0;
  }
  clear_error();
  return map->value.cap;
}
int hyle_container_hash_map_reserve(hyle_container_hash_map* map,
                                    size_t count) {
  if (map == nullptr) {
    set_error("map is NULL");
    return 0;
  }
  if (count > SIZE_MAX / 8) {
    set_error("hash map capacity overflows address space");
    return 0;
  }
  if (!map->value.reserve(count)) {
    set_error("hash map bucket allocation failed");
    return 0;
  }
  clear_error();
  return 1;
}
void hyle_container_hash_map_clear(hyle_container_hash_map* map) {
  if (map == nullptr) {
    set_error("map is NULL");
    return;
  }
  map->value.clear();
  clear_error();
}
int hyle_container_hash_map_insert(hyle_container_hash_map* map,
                                   const void* key, const void* value) {
  if (map == nullptr || key == nullptr || value == nullptr) {
    set_error("map, key, and value are required");
    return -1;
  }
  RawMap& raw = map->value;
  const uint64_t hash = raw.hash(key);
  const std::size_t existing = raw.find_slot(key, hash);
  if (existing != RawMap::kNoSlot) {
    std::memmove(raw.value(existing), value, raw.options.value_size);
    clear_error();
    return 0;  // updated
  }

  if (raw.cap == 0 && !raw.resize(hc::detail::kGroupWidth)) {
    set_error("hash map storage allocation failed");
    return -1;
  }
  std::size_t target = raw.find_insert_slot(hash);
  if (raw.growth_left == 0 &&
      raw.ctrl[target] == hc::detail::kEmpty) {
    // The C facade permits insertion from pointers returned by find(). Preserve
    // such inputs across the flat-table reallocation, but keep the normal path
    // allocation-free.
    const bool key_alias =
        raw.aliases_slots(key, raw.options.key_size);
    const bool value_alias =
        raw.aliases_slots(value, raw.options.value_size);
    void* scratch = nullptr;
    const void* stable_key = key;
    const void* stable_value = value;
    if (key_alias || value_alias) {
      const std::size_t scratch_size =
          raw.options.key_size + raw.options.value_size;
      scratch = raw.alloc.allocate(scratch_size, raw.block_alignment);
      if (scratch == nullptr) {
        set_error("hash map alias scratch allocation failed");
        return -1;
      }
      auto* bytes = static_cast<std::byte*>(scratch);
      std::memmove(bytes, key, raw.options.key_size);
      std::memmove(bytes + raw.options.key_size, value,
                   raw.options.value_size);
      stable_key = bytes;
      stable_value = bytes + raw.options.key_size;
    }
    if (raw.cap > SIZE_MAX / 2 ||
        !raw.resize(raw.size >= raw.cap - raw.cap / 8 ? raw.cap * 2
                                                       : raw.cap)) {
      if (scratch != nullptr)
        raw.alloc.deallocate(scratch,
                             raw.options.key_size + raw.options.value_size,
                             raw.block_alignment);
      set_error("hash map storage allocation failed");
      return -1;
    }
    target = raw.find_insert_slot(hash);
    raw.set_hash(target, hash);
    std::memcpy(raw.key(target), stable_key, raw.options.key_size);
    std::memcpy(raw.value(target), stable_value, raw.options.value_size);
    if (scratch != nullptr)
      raw.alloc.deallocate(scratch,
                           raw.options.key_size + raw.options.value_size,
                           raw.block_alignment);
  } else {
    raw.set_hash(target, hash);
    std::memcpy(raw.key(target), key, raw.options.key_size);
    std::memcpy(raw.value(target), value, raw.options.value_size);
  }
  const bool was_empty = raw.ctrl[target] == hc::detail::kEmpty;
  raw.set_ctrl(target, hc::detail::h2(hash));
  ++raw.size;
  if (was_empty) --raw.growth_left;
  clear_error();
  return 1;  // inserted
}
void* hyle_container_hash_map_find(hyle_container_hash_map* map,
                                   const void* key) {
  if (map == nullptr || key == nullptr) {
    set_error("map and key are required");
    return nullptr;
  }
  const uint64_t hash = map->value.hash(key);
  const std::size_t slot = map->value.find_slot(key, hash);
  clear_error();
  return slot == RawMap::kNoSlot ? nullptr : map->value.value(slot);
}
const void* hyle_container_hash_map_find_const(
    const hyle_container_hash_map* map, const void* key) {
  if (map == nullptr || key == nullptr) {
    set_error("map and key are required");
    return nullptr;
  }
  const uint64_t hash = map->value.hash(key);
  const std::size_t slot = map->value.find_slot(key, hash);
  clear_error();
  return slot == RawMap::kNoSlot ? nullptr : map->value.value(slot);
}
int hyle_container_hash_map_contains(const hyle_container_hash_map* map,
                                     const void* key) {
  if (map == nullptr || key == nullptr) {
    set_error("map and key are required");
    return 0;
  }
  const uint64_t hash = map->value.hash(key);
  const bool present =
      map->value.find_slot(key, hash) != RawMap::kNoSlot;
  clear_error();
  return present;
}
int hyle_container_hash_map_erase(hyle_container_hash_map* map,
                                  const void* key) {
  if (map == nullptr || key == nullptr) {
    set_error("map and key are required");
    return 0;
  }
  RawMap& raw = map->value;
  const uint64_t hash = raw.hash(key);
  const std::size_t slot = raw.find_slot(key, hash);
  if (slot == RawMap::kNoSlot) {
    clear_error();
    return 0;
  }
  raw.erase_slot(slot);
  clear_error();
  return 1;
}
int hyle_container_hash_map_for_each(
    hyle_container_hash_map* map, hyle_container_hash_map_visit_fn visitor,
    void* context) {
  if (map == nullptr || visitor == nullptr) {
    set_error("map and visitor are required");
    return 0;
  }
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
  clear_error();
  return 1;
}

static bool valid_queue_capacity(size_t capacity) {
  return capacity >= 2 && is_power_of_two(capacity) &&
         capacity <= (UINT32_C(1) << 31);
}

#define HYLE_QUEUE_CREATE(name)                                                \
  name* name##_create(hyle_allocator backing, size_t capacity) {               \
    if (!valid_backing(backing) || !valid_queue_capacity(capacity)) {           \
      set_error("invalid allocator or queue capacity");                        \
      return nullptr;                                                          \
    }                                                                          \
    auto* queue = allocate_handle<name>(backing, backing, capacity);           \
    if (queue == nullptr || !queue->value.valid()) {                           \
      free_handle(queue);                                                      \
      queue = nullptr;                                                         \
      set_error("queue allocation failed");                                   \
    } else {                                                                   \
      clear_error();                                                           \
    }                                                                          \
    return queue;                                                              \
  }                                                                            \
  void name##_destroy(name* queue) { free_handle(queue); }                     \
  size_t name##_capacity(const name* queue) {                                  \
    if (queue == nullptr) {                                                     \
      set_error("queue is NULL");                                              \
      return 0;                                                                \
    }                                                                          \
    clear_error();                                                             \
    return queue->value.capacity();                                             \
  }

HYLE_QUEUE_CREATE(hyle_container_bounded_queue)
HYLE_QUEUE_CREATE(hyle_container_work_stealing_deque)
HYLE_QUEUE_CREATE(hyle_container_ws_fifo_queue)
#undef HYLE_QUEUE_CREATE

int hyle_container_bounded_queue_push(hyle_container_bounded_queue* queue,
                                      void* item) {
  if (queue == nullptr) {
    set_error("bounded queue is NULL");
    return 0;
  }
  clear_error();
  return queue->value.try_push(item);
}
int hyle_container_bounded_queue_pop(hyle_container_bounded_queue* queue,
                                     void** out_item) {
  if (queue == nullptr || out_item == nullptr) {
    set_error("bounded queue and output are required");
    return 0;
  }
  clear_error();
  return queue->value.try_pop(*out_item);
}
hyle_container_sharded_queue* hyle_container_sharded_queue_create(
    hyle_allocator backing, size_t shard_count, size_t capacity_per_shard) {
  if (!valid_backing(backing) || shard_count == 0 ||
      !valid_queue_capacity(capacity_per_shard)) {
    set_error("invalid allocator or sharded queue dimensions");
    return nullptr;
  }
  if (shard_count >
          SIZE_MAX /
              sizeof(hc::BoundedQueue<void*, hyle::mem::Allocator>) ||
      capacity_per_shard > SIZE_MAX / shard_count) {
    set_error("sharded queue dimensions overflow address space");
    return nullptr;
  }
  auto* queue = allocate_handle<hyle_container_sharded_queue>(
      backing, backing, shard_count, capacity_per_shard);
  if (queue == nullptr || !queue->value.valid()) {
    free_handle(queue);
    queue = nullptr;
    set_error("sharded queue allocation failed");
  } else {
    clear_error();
  }
  return queue;
}
void hyle_container_sharded_queue_destroy(
    hyle_container_sharded_queue* queue) {
  free_handle(queue);
}
int hyle_container_sharded_queue_push(hyle_container_sharded_queue* queue,
                                      size_t producer_hint, void* item) {
  if (queue == nullptr) {
    set_error("sharded queue is NULL");
    return 0;
  }
  clear_error();
  return queue->value.try_push(producer_hint, item);
}
int hyle_container_sharded_queue_pop(hyle_container_sharded_queue* queue,
                                     size_t* cursor, void** out_item) {
  if (queue == nullptr || cursor == nullptr || out_item == nullptr) {
    set_error("sharded queue, cursor, and output are required");
    return 0;
  }
  clear_error();
  return queue->value.try_pop(*cursor, *out_item);
}
size_t hyle_container_sharded_queue_capacity(
    const hyle_container_sharded_queue* queue) {
  if (queue == nullptr) {
    set_error("sharded queue is NULL");
    return 0;
  }
  clear_error();
  return queue->value.capacity();
}
size_t hyle_container_sharded_queue_shard_count(
    const hyle_container_sharded_queue* queue) {
  if (queue == nullptr) {
    set_error("sharded queue is NULL");
    return 0;
  }
  clear_error();
  return queue->value.num_shards();
}
int hyle_container_work_stealing_deque_push(
    hyle_container_work_stealing_deque* deque, void* item) {
  if (deque == nullptr) {
    set_error("work-stealing deque is NULL");
    return 0;
  }
  clear_error();
  return deque->value.push_bottom(item);
}
int hyle_container_work_stealing_deque_pop(
    hyle_container_work_stealing_deque* deque, void** out_item) {
  if (deque == nullptr || out_item == nullptr) {
    set_error("work-stealing deque and output are required");
    return 0;
  }
  const auto result = deque->value.pop_bottom();
  if (result.ok) *out_item = result.value;
  clear_error();
  return result.ok;
}
hyle_container_steal_status hyle_container_work_stealing_deque_steal(
    hyle_container_work_stealing_deque* deque, void** out_item) {
  if (deque == nullptr || out_item == nullptr) {
    set_error("work-stealing deque and output are required");
    return HYLE_CONTAINER_STEAL_EMPTY;
  }
  const auto result = deque->value.steal();
  if (result.status == hc::StealStatus::kOk) *out_item = result.value;
  clear_error();
  return static_cast<hyle_container_steal_status>(result.status);
}
int hyle_container_ws_fifo_queue_push(hyle_container_ws_fifo_queue* queue,
                                      void* item) {
  if (queue == nullptr) {
    set_error("work-sharing FIFO queue is NULL");
    return 0;
  }
  clear_error();
  return queue->value.push_back(item);
}
int hyle_container_ws_fifo_queue_pop(hyle_container_ws_fifo_queue* queue,
                                     void** out_item) {
  if (queue == nullptr || out_item == nullptr) {
    set_error("work-sharing FIFO queue and output are required");
    return 0;
  }
  const auto result = queue->value.pop();
  if (result.ok) *out_item = result.value;
  clear_error();
  return result.ok;
}
int hyle_container_ws_fifo_queue_steal_into(
    hyle_container_ws_fifo_queue* source, hyle_container_ws_fifo_queue* target,
    void** out_item) {
  if (source == nullptr || target == nullptr || out_item == nullptr ||
      source == target || source->value.capacity() != target->value.capacity()) {
    set_error("steal requires distinct equal-capacity queues and output");
    return 0;
  }
  const auto result = source->value.steal_into(target->value);
  if (result.ok) *out_item = result.value;
  clear_error();
  return result.ok;
}

hyle_container_concurrent_vector* hyle_container_concurrent_vector_create(
    hyle_allocator backing, size_t initial_capacity) {
  if (!valid_backing(backing) ||
      initial_capacity >
      (size_t(1) << (sizeof(size_t) * 8 - 1))) {
    set_error("invalid allocator or initial capacity");
    return nullptr;
  }
  auto* vector = allocate_handle<hyle_container_concurrent_vector>(
      backing, backing, initial_capacity);
  if (vector == nullptr) set_error("concurrent vector allocation failed");
  else clear_error();
  return vector;
}
void hyle_container_concurrent_vector_destroy(
    hyle_container_concurrent_vector* vector) {
  free_handle(vector);
}
size_t hyle_container_concurrent_vector_push(
    hyle_container_concurrent_vector* vector, void* item) {
  if (vector == nullptr) {
    set_error("concurrent vector is NULL");
    return SIZE_MAX;
  }
  const size_t index = vector->value.push_back(item);
  if (index == SIZE_MAX)
    set_error("concurrent vector storage allocation failed");
  else
    clear_error();
  return index;
}
size_t hyle_container_concurrent_vector_size(
    const hyle_container_concurrent_vector* vector) {
  if (vector == nullptr) {
    set_error("concurrent vector is NULL");
    return 0;
  }
  clear_error();
  return vector->value.size();
}
size_t hyle_container_concurrent_vector_capacity(
    const hyle_container_concurrent_vector* vector) {
  if (vector == nullptr) {
    set_error("concurrent vector is NULL");
    return 0;
  }
  clear_error();
  return vector->value.capacity();
}
void* hyle_container_concurrent_vector_at(
    const hyle_container_concurrent_vector* vector, size_t index) {
  if (vector == nullptr || index >= vector->value.size()) {
    set_error("concurrent vector index out of range");
    return nullptr;
  }
  clear_error();
  return vector->value[index];
}

}  // extern "C"

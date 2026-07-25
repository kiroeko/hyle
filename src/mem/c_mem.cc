// C-ABI bridge for mem (include/hyle/c/mem.h).
//
// Two directions, both WITHOUT reinterpret_cast on the vtable:
//   C -> C++ : a caller's hyle_allocator is stored by value in the opaque
//   control handle;
//              the C++ object gets a mem::Allocator whose state points at that
//              stored struct and whose fn pointers are noexcept trampolines
//              forwarding to it. (The C fn pointers are plain `void*(*)(...)`;
//              mem::Allocator wants `noexcept` fn types, which a plain pointer
//              is NOT convertible to -- so a trampoline is required, not a
//              cast.)
//   C++ -> C : filling a hyle_allocator from a mem::Allocator is a
//   field-by-field copy;
//              a noexcept fn pointer converts implicitly to the plain C field
//              type.

#include "hyle/c/mem.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>

#include "hyle/mem/allocator.h"
#include "hyle/mem/fixed_pool.h"
#include "hyle/mem/growable_pool.h"
#include "hyle/mem/hugepage.h"
#include "hyle/mem/pool_cache.h"
#include "hyle/mem/quota_allocator.h"
#include "hyle/mem/region.h"
#include "hyle/mem/shared_pool.h"
#include "hyle/mem/stack_allocator.h"
#include "hyle/mem/stats.h"
#include "../error/status_access.h"

namespace {

using hyle::mem::Allocator;

hyle_status steal_status(hyle::Status &&status) noexcept {
  return hyle_status{hyle::detail::StatusAccess::release(status)};
}

static_assert(std::is_standard_layout_v<hyle_allocator>);
static_assert(std::is_trivially_copyable_v<hyle_allocator>);
static_assert(sizeof(hyle_allocator) == sizeof(void *) * 4);

// ---- C hyle_allocator -> C++ mem::Allocator (via noexcept trampolines)
// ------- `state` points at a live hyle_allocator (stored inside the owning
// control handle, so it outlives every C++ object built from it).
void *c_backing_alloc(void *state, std::size_t size,
                      std::size_t align) noexcept {
  auto *a = static_cast<const hyle_allocator *>(state);
  return a->alloc(a->state, size, align);
}
void c_backing_free(void *state, void *ptr, std::size_t size,
                    std::size_t align) noexcept {
  auto *a = static_cast<const hyle_allocator *>(state);
  a->free(a->state, ptr, size, align);
}
void *c_backing_realloc(void *state, void *ptr, std::size_t old_size,
                        std::size_t new_size, std::size_t align) noexcept {
  auto *a = static_cast<const hyle_allocator *>(state);
  return a->realloc(a->state, ptr, old_size, new_size, align);
}

// Build a C++ Allocator that forwards to the stored C handle at `*backing`.
Allocator to_cpp(const hyle_allocator *backing) noexcept {
  // Preserve the process system allocator's identity instead of wrapping it in
  // C trampolines. This keeps allocator equality/PMR identity stable; every
  // genuinely injected C backing remains authoritative for its storage.
  const Allocator sys = hyle::mem::system_allocator();
  if (backing->alloc == sys.alloc_fn() && backing->free == sys.free_fn() &&
      backing->realloc == sys.realloc_fn() && backing->state == sys.state()) {
    return sys;
  }
  return Allocator(&c_backing_alloc, &c_backing_free, &c_backing_realloc,
                   const_cast<hyle_allocator *>(backing));
}

// ---- C++ mem::Allocator -> C hyle_allocator (field-by-field, no cast)
// -------- AllocFn/FreeFn/ReallocFn are `noexcept` fn pointers; assigning them
// to the plain C field type is a standard implicit (noexcept -> non-noexcept)
// pointer conversion.
hyle_allocator from_cpp(const Allocator &a) noexcept {
  hyle_allocator out;
  out.alloc = a.alloc_fn();
  out.free = a.free_fn();
  out.realloc = a.realloc_fn();
  out.state = a.state();
  return out;
}

// A caller-supplied handle is only usable if all three fn pointers are present
// (system-boundary validation: an FFI caller can pass a zeroed struct).
bool valid_backing(const hyle_allocator &a) noexcept {
  return a.alloc != nullptr && a.free != nullptr && a.realloc != nullptr;
}

std::size_t norm_align(std::size_t a) noexcept {
  return a != 0 ? a : alignof(std::max_align_t);
}

bool normalize_alignment(std::size_t input, std::size_t *out) noexcept {
  const std::size_t alignment = norm_align(input);
  if ((alignment & (alignment - 1)) != 0) return false;
  *out = alignment;
  return true;
}

bool valid_shared_pool_layout(std::size_t block_size,
                              std::uint32_t block_count,
                              std::size_t alignment) noexcept {
  if (block_size == 0 || block_count == 0 || block_count == UINT32_MAX)
    return false;
  if (block_size > SIZE_MAX - (alignment - 1))
    return false;
  const std::size_t stride =
      (block_size + alignment - 1) & ~(alignment - 1);
  return static_cast<std::size_t>(block_count) <= SIZE_MAX / stride;
}

hyle_allocator system_backing() noexcept {
  return from_cpp(hyle::mem::system_allocator());
}

template <class T, class... Args>
T *allocate_handle(hyle_allocator backing, Args &&...args) noexcept {
  static_assert(std::is_nothrow_constructible_v<T, hyle_allocator, Args...>);
  void *memory = backing.alloc(backing.state, sizeof(T), alignof(T));
  if (memory == nullptr)
    return nullptr;
  return ::new (memory) T(backing, std::forward<Args>(args)...);
}

template <class T>
void free_handle(T *handle) noexcept {
  if (handle == nullptr)
    return;
  const hyle_allocator backing = handle->backing;
  handle->~T();
  backing.free(backing.state, handle, sizeof(T), alignof(T));
}

hyle::mem::HugeFlags to_huge_flags(std::uint32_t f) noexcept {
  using hyle::mem::HugeFlags;
  auto out = HugeFlags::kNone;
  if (f & HYLE_HUGE_THP_HINT)
    out = out | HugeFlags::kThpHint;
  if (f & HYLE_HUGE_PREFAULT)
    out = out | HugeFlags::kPrefault;
  if (f & HYLE_HUGE_NUMA_BIND)
    out = out | HugeFlags::kNumaBind;
  if (f & HYLE_HUGE_NUMA_INTERLEAVE)
    out = out | HugeFlags::kNumaInterleave;
  return out;
}

} // namespace

// ---- opaque control handles ------------------------------------------------
// Each stores the caller's hyle_allocator FIRST (so it is destroyed LAST) and
// the C++ object SECOND (destroyed first -- it may free through `backing` in
// its dtor).

struct hyle_shared_pool_o {
  hyle_allocator backing;
  hyle::mem::SharedPool pool;
  hyle_shared_pool_o(hyle_allocator b, std::size_t block_size,
                     std::uint32_t block_count,
                     std::size_t alignment) noexcept
      : backing(b), pool(to_cpp(&backing), block_size, block_count, alignment) {
  }
};

struct hyle_quota_allocator_o {
  hyle_allocator backing;
  hyle::mem::QuotaAllocator quota;
  hyle_quota_allocator_o(hyle_allocator b, std::size_t limit) noexcept
      : backing(b), quota(to_cpp(&backing), limit) {}
};

struct hyle_stats_allocator_o {
  // `backing` is the control/metadata allocator so generic free_handle returns
  // this opaque object through the same domain that created it.
  hyle_allocator backing;
  hyle_allocator measured;
  std::optional<hyle::mem::StatsAllocator> stats;
  hyle_stats_allocator_o(hyle_allocator metadata,
                         hyle_allocator measured_backing) noexcept
      : backing(metadata), measured(measured_backing) {}
};

struct hyle_region_o {
  hyle_allocator backing;
  hyle::mem::Region region;
  hyle_region_o(hyle_allocator b, hyle::mem::RegionOptions options) noexcept
      : backing(b), region(to_cpp(&backing), options) {}
};

struct hyle_fixed_pool_o {
  hyle_allocator backing;
  bool owns_control;
  hyle::mem::FixedPool pool;
  hyle_fixed_pool_o(hyle_allocator b, void *storage,
                    std::size_t size, bool owns = true) noexcept
      : backing(b), owns_control(owns), pool(storage, size) {}
};

struct hyle_shared_pool_cache_o {
  hyle_allocator backing;
  hyle::mem::SharedPoolCache cache;
  hyle_shared_pool_cache_o(hyle_allocator m, hyle::mem::SharedPool &pool,
                           std::uint32_t capacity,
                           std::uint32_t transfer_batch) noexcept
      : backing(m), cache(pool, to_cpp(&backing), capacity, transfer_batch) {}
};

struct hyle_stack_allocator_o {
  hyle_allocator backing;
  hyle::mem::StackAllocator stack;
  hyle_stack_allocator_o(hyle_allocator b, std::size_t size) noexcept
      : backing(b), stack(to_cpp(&backing), size) {}
};

struct hyle_pooled_region_o {
  hyle_allocator backing;
  hyle::mem::PooledRegion region; // borrows one block; owns no backing
  hyle_pooled_region_o(hyle_allocator b,
                       hyle::mem::SharedPool *pool) noexcept
      : backing(b), region(pool) {}
};

struct hyle_growable_pool_o {
  hyle_allocator backing;
  hyle::mem::detail::GrowablePoolCore core;
  hyle_growable_pool_o(hyle_allocator b, std::size_t slot_size,
                       std::size_t slot_align,
                       std::uint32_t initial_slots) noexcept
      : backing(b),
        core(to_cpp(&backing), slot_size, slot_align, initial_slots) {}
};

struct hyle_huge_region_o {
  hyle_allocator backing;
  hyle::mem::HugeRegion region;
  hyle_huge_region_o(hyle_allocator b, hyle::mem::HugeRegion &&r) noexcept
      : backing(b), region(std::move(r)) {}
};

static_assert(sizeof(hyle_fixed_pool_o) <= HYLE_FIXED_POOL_CONTROL_BYTES);
static_assert(alignof(hyle_fixed_pool_o) <=
              alignof(hyle_fixed_pool_storage));

template <class T>
void destroy_external_control(T *handle) noexcept {
  if (handle == nullptr)
    return;
  const hyle_allocator backing = handle->backing;
  const bool owns_control = handle->owns_control;
  handle->~T();
  if (owns_control)
    backing.free(backing.state, handle, sizeof(T), alignof(T));
}

extern "C" {

/* ---- allocator vtable ------------------------------------------------------
 */

hyle_allocator hyle_system_allocator(void) {
  return from_cpp(hyle::mem::system_allocator());
}

size_t hyle_system_usable_size(const void *ptr) {
  return hyle::mem::system_usable_size(ptr);
}

size_t hyle_system_good_size(size_t size) {
  return hyle::mem::system_good_size(size);
}

void hyle_system_collect(int force) { hyle::mem::system_collect(force != 0); }

/* ---- allocator adapters ---------------------------------------------------
 */

hyle_status hyle_quota_allocator_create(hyle_quota_allocator_o **out,
                                        hyle_allocator backing, size_t limit) {
  if (out == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_quota_allocator_create: out is NULL"));
  }
  *out = nullptr;
  if (!valid_backing(backing)) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_quota_allocator_create: invalid backing"));
  }
  auto *q = allocate_handle<hyle_quota_allocator_o>(backing, limit);
  if (q == nullptr) {
    return steal_status(hyle::Status::ResourceExhausted(
        "hyle_quota_allocator_create: control allocation failed"));
  }
  *out = q;
  return steal_status(hyle::Status());
}

void hyle_quota_allocator_destroy(hyle_quota_allocator_o *q) { free_handle(q); }
hyle_allocator hyle_quota_allocator_as_allocator(hyle_quota_allocator_o *q) {
  return q ? from_cpp(q->quota.get())
           : hyle_allocator{nullptr, nullptr, nullptr, nullptr};
}
size_t hyle_quota_allocator_used(const hyle_quota_allocator_o *q) {
  return q ? q->quota.used() : 0;
}
size_t hyle_quota_allocator_peak(const hyle_quota_allocator_o *q) {
  return q ? q->quota.peak() : 0;
}
size_t hyle_quota_allocator_limit(const hyle_quota_allocator_o *q) {
  return q ? q->quota.limit() : 0;
}
size_t hyle_quota_allocator_remaining(const hyle_quota_allocator_o *q) {
  return q ? q->quota.remaining() : 0;
}

hyle_status hyle_stats_allocator_create(hyle_stats_allocator_o **out,
                                        hyle_allocator backing,
                                        hyle_allocator metadata,
                                        int track_leaks) {
  if (out == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_stats_allocator_create: out is NULL"));
  }
  *out = nullptr;
  if (!valid_backing(backing) || !valid_backing(metadata)) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_stats_allocator_create: invalid backing or metadata"));
  }
  auto *handle =
      allocate_handle<hyle_stats_allocator_o>(metadata, backing);
  if (handle == nullptr) {
    return steal_status(hyle::Status::ResourceExhausted(
        "hyle_stats_allocator_create: control allocation failed"));
  }
  auto created = hyle::mem::StatsAllocator::create(
      to_cpp(&handle->measured), track_leaks != 0,
      to_cpp(&handle->backing));
  if (!created) {
    hyle::Status status = std::move(created).error();
    free_handle(handle);
    return steal_status(std::move(status));
  }
  handle->stats.emplace(*std::move(created));
  *out = handle;
  return steal_status(hyle::Status());
}

void hyle_stats_allocator_destroy(hyle_stats_allocator_o *s) {
  free_handle(s);
}

hyle_allocator hyle_stats_allocator_as_allocator(hyle_stats_allocator_o *s) {
  return s ? from_cpp(s->stats->get())
           : hyle_allocator{nullptr, nullptr, nullptr, nullptr};
}

static hyle_alloc_stats stats_snapshot(const hyle_stats_allocator_o *s) {
  hyle_alloc_stats out{};
  if (s == nullptr)
    return out;
  const hyle::mem::AllocStatsSnapshot in = s->stats->snapshot();
  out.live_count = in.live_count;
  out.live_bytes = in.live_bytes;
  out.total_allocs = in.total_allocs;
  out.total_frees = in.total_frees;
  out.total_bytes_allocated = in.total_bytes_allocated;
  out.peak_live_bytes = in.peak_live_bytes;
  out.reallocs = in.reallocs;
  return out;
}

void hyle_stats_allocator_snapshot(const hyle_stats_allocator_o *s,
                                   hyle_alloc_stats *out) {
  if (out != nullptr)
    *out = stats_snapshot(s);
}

void hyle_stats_allocator_reset(hyle_stats_allocator_o *s) {
  if (s != nullptr) {
    s->stats->stats().reset();
    if (s->stats->leaks() != nullptr)
      s->stats->leaks()->clear();
  }
}

int hyle_stats_allocator_balanced(const hyle_stats_allocator_o *s) {
  return s != nullptr && s->stats->balanced();
}

uint64_t hyle_stats_allocator_leak_records_dropped(
    const hyle_stats_allocator_o *s) {
  if (s == nullptr || s->stats->leaks() == nullptr)
    return 0;
  return s->stats->leaks()->dropped_records();
}

int hyle_stats_allocator_leaks_complete(const hyle_stats_allocator_o *s) {
  return hyle_stats_allocator_leak_records_dropped(s) == 0;
}

hyle_status hyle_stats_allocator_copy_leaks(
    size_t *out_total, const hyle_stats_allocator_o *s, hyle_mem_leak *out,
    size_t capacity) {
  static_assert(sizeof(hyle_mem_leak) == sizeof(hyle::mem::LeakEntry));
  static_assert(alignof(hyle_mem_leak) == alignof(hyle::mem::LeakEntry));
  static_assert(offsetof(hyle_mem_leak, ptr) ==
                offsetof(hyle::mem::LeakEntry, ptr));
  static_assert(offsetof(hyle_mem_leak, size) ==
                offsetof(hyle::mem::LeakEntry, size));
  if (out_total == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_stats_allocator_copy_leaks: out_total is NULL"));
  }
  *out_total = 0;
  if (s == nullptr || s->stats->leaks() == nullptr)
    return steal_status(hyle::Status());
  if (out == nullptr || capacity == 0) {
    *out_total = s->stats->leaks()->outstanding();
    return s->stats->leaks()->complete()
               ? steal_status(hyle::Status())
               : steal_status(hyle::Status::DataLoss(
                     "hyle_stats_allocator_copy_leaks: metadata OOM made "
                     "the leak inventory incomplete"));
  }
  if (capacity > SIZE_MAX / sizeof(hyle::mem::LeakEntry)) {
    return steal_status(hyle::Status::OutOfRange(
        "hyle_stats_allocator_copy_leaks: capacity overflow"));
  }
  const size_t bytes = capacity * sizeof(hyle::mem::LeakEntry);
  void *scratch = s->backing.alloc(s->backing.state, bytes,
                                   alignof(hyle::mem::LeakEntry));
  if (scratch == nullptr) {
    return steal_status(hyle::Status::ResourceExhausted(
        "hyle_stats_allocator_copy_leaks: scratch allocation failed"));
  }
  auto *entries = static_cast<hyle::mem::LeakEntry *>(scratch);
  const size_t total = s->stats->leaks()->copy_leaks(entries, capacity);
  const size_t copied = total < capacity ? total : capacity;
  std::memcpy(out, entries, copied * sizeof(*entries));
  s->backing.free(s->backing.state, scratch, bytes,
                  alignof(hyle::mem::LeakEntry));
  *out_total = total;
  return s->stats->leaks()->complete()
             ? steal_status(hyle::Status())
             : steal_status(hyle::Status::DataLoss(
                   "hyle_stats_allocator_copy_leaks: metadata OOM made the "
                   "leak inventory incomplete"));
}

/* ---- Region family --------------------------------------------------------
 */

hyle_status hyle_region_create(hyle_region_o **out, hyle_allocator backing,
                               const hyle_region_options *options) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_region_create: out is NULL"));
  }
  *out = nullptr;
  if (!valid_backing(backing)) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_region_create: invalid backing"));
  }
  hyle_region_options defaults = HYLE_REGION_OPTIONS_INIT;
  const hyle_region_options *in = options != nullptr ? options : &defaults;
  if (in->struct_size < sizeof(hyle_region_options) ||
      in->initial_block_size == 0 ||
      in->max_block_size < in->initial_block_size ||
      in->large_allocation_threshold == 0) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_region_create: invalid options"));
  }
  hyle::mem::RegionOptions cpp;
  cpp.initial_block_size = in->initial_block_size;
  cpp.max_block_size = in->max_block_size;
  cpp.large_allocation_threshold = in->large_allocation_threshold;
  cpp.size_limit = in->size_limit;
  auto *r = allocate_handle<hyle_region_o>(backing, cpp);
  if (r == nullptr) {
    return steal_status(hyle::Status::ResourceExhausted(
        "hyle_region_create: control allocation failed"));
  }
  *out = r;
  return steal_status(hyle::Status());
}

void hyle_region_destroy(hyle_region_o *r) { free_handle(r); }
void *hyle_region_allocate(hyle_region_o *r, size_t bytes, size_t alignment) {
  std::size_t normalized = 0;
  return r && normalize_alignment(alignment, &normalized)
             ? r->region.allocate(bytes, normalized)
             : nullptr;
}
void hyle_region_clear(hyle_region_o *r) {
  if (r)
    r->region.clear();
}
void hyle_region_release(hyle_region_o *r) {
  if (r)
    r->region.release();
}
size_t hyle_region_used(const hyle_region_o *r) {
  return r ? r->region.bytes_used() : 0;
}
size_t hyle_region_reserved(const hyle_region_o *r) {
  return r ? r->region.reserved() : 0;
}
size_t hyle_region_peak_used(const hyle_region_o *r) {
  return r ? r->region.peak_used() : 0;
}
size_t hyle_region_size_limit(const hyle_region_o *r) {
  return r ? r->region.size_limit() : 0;
}
hyle_allocator hyle_region_as_allocator(hyle_region_o *r) {
  return r ? from_cpp(r->region.as_allocator())
           : hyle_allocator{nullptr, nullptr, nullptr, nullptr};
}

hyle_status hyle_fixed_region_init(hyle_fixed_region *r, void *storage,
                                   size_t size) {
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(storage);
  if (r != nullptr)
    r->state[0] = r->state[1] = r->state[2] = 0;
  if (r == nullptr || storage == nullptr || size == 0 ||
      size > UINTPTR_MAX - begin) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_fixed_region_init: invalid storage"));
  }
  r->state[0] = begin;
  r->state[1] = begin;
  r->state[2] = begin + size;
  return steal_status(hyle::Status());
}

static bool valid_fixed_region(const hyle_fixed_region *r) {
  return r != nullptr && r->state[0] != 0 && r->state[1] >= r->state[0] &&
         r->state[1] <= r->state[2];
}

void *hyle_fixed_region_allocate(hyle_fixed_region *r, size_t bytes,
                                 size_t alignment) {
  std::size_t normalized = 0;
  if (!valid_fixed_region(r) || bytes == 0 ||
      !normalize_alignment(alignment, &normalized))
    return nullptr;
  const std::uintptr_t current = r->state[1];
  if (current > UINTPTR_MAX - (normalized - 1))
    return nullptr;
  const std::uintptr_t aligned =
      (current + normalized - 1) & ~(std::uintptr_t(normalized) - 1);
  if (aligned > r->state[2] || bytes > r->state[2] - aligned)
    return nullptr;
  r->state[1] = aligned + bytes;
  return reinterpret_cast<void *>(aligned);
}
void hyle_fixed_region_clear(hyle_fixed_region *r) {
  if (r)
    r->state[1] = r->state[0];
}
size_t hyle_fixed_region_used(const hyle_fixed_region *r) {
  return r && r->state[1] >= r->state[0] ? r->state[1] - r->state[0] : 0;
}
size_t hyle_fixed_region_capacity(const hyle_fixed_region *r) {
  return r && r->state[2] >= r->state[0] ? r->state[2] - r->state[0] : 0;
}
size_t hyle_fixed_region_remaining(const hyle_fixed_region *r) {
  return r && r->state[2] >= r->state[1] ? r->state[2] - r->state[1] : 0;
}

namespace {
void *c_fixed_region_alloc(void *state, size_t size, size_t alignment) {
  return hyle_fixed_region_allocate(static_cast<hyle_fixed_region *>(state),
                                    size, alignment);
}
void c_fixed_region_free(void *, void *, size_t, size_t) {}
void *c_fixed_region_realloc(void *state, void *ptr, size_t old_size,
                             size_t new_size, size_t alignment) {
  if (new_size == 0)
    return nullptr;
  void *out = c_fixed_region_alloc(state, new_size, alignment);
  if (out != nullptr && ptr != nullptr)
    std::memcpy(out, ptr, old_size < new_size ? old_size : new_size);
  return out;
}
} // namespace

hyle_allocator hyle_fixed_region_as_allocator(hyle_fixed_region *r) {
  return valid_fixed_region(r)
             ? hyle_allocator{&c_fixed_region_alloc, &c_fixed_region_free,
                              &c_fixed_region_realloc, r}
             : hyle_allocator{nullptr, nullptr, nullptr, nullptr};
}

size_t hyle_fixed_pool_control_size(void) { return sizeof(hyle_fixed_pool_o); }

size_t hyle_fixed_pool_control_alignment(void) {
  return alignof(hyle_fixed_pool_o);
}

hyle_status hyle_fixed_pool_init(hyle_fixed_pool_o **out, void *control,
                                 size_t control_bytes, void *storage,
                                 size_t size) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_fixed_pool_init: out is NULL"));
  }
  *out = nullptr;
  if (control == nullptr || control_bytes < sizeof(hyle_fixed_pool_o) ||
      reinterpret_cast<std::uintptr_t>(control) % alignof(hyle_fixed_pool_o) !=
          0) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_fixed_pool_init: control is null, undersized, or misaligned"));
  }
  const hyle_allocator backing = system_backing();
  auto *p = ::new (control)
      hyle_fixed_pool_o(backing, storage, size, false);
  if (!p->pool.valid()) {
    destroy_external_control(p);
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_fixed_pool_init: invalid storage"));
  }
  *out = p;
  return steal_status(hyle::Status());
}

hyle_status hyle_fixed_pool_create(hyle_fixed_pool_o **out,
                                   hyle_allocator backing, void *storage,
                                   size_t size) {
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_fixed_pool_create: out is NULL"));
  }
  *out = nullptr;
  if (!valid_backing(backing)) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_fixed_pool_create: invalid control allocator"));
  }
  auto *p = allocate_handle<hyle_fixed_pool_o>(backing, storage, size);
  if (p == nullptr) {
    return steal_status(hyle::Status::ResourceExhausted(
        "hyle_fixed_pool_create: control allocation failed"));
  }
  if (!p->pool.valid()) {
    free_handle(p);
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_fixed_pool_create: invalid storage"));
  }
  *out = p;
  return steal_status(hyle::Status());
}
void hyle_fixed_pool_destroy(hyle_fixed_pool_o *p) {
  destroy_external_control(p);
}
void *hyle_fixed_pool_allocate(hyle_fixed_pool_o *p, size_t bytes,
                               size_t alignment) {
  std::size_t normalized = 0;
  return p && normalize_alignment(alignment, &normalized)
             ? p->pool.allocate(bytes, normalized)
             : nullptr;
}
void hyle_fixed_pool_deallocate(hyle_fixed_pool_o *p, void *ptr) {
  if (p)
    p->pool.deallocate(ptr);
}
void *hyle_fixed_pool_reallocate(hyle_fixed_pool_o *p, void *ptr,
                                 size_t old_size, size_t new_size,
                                 size_t alignment) {
  std::size_t normalized = 0;
  return p && normalize_alignment(alignment, &normalized)
             ? p->pool.reallocate(ptr, old_size, new_size, normalized)
             : nullptr;
}
void hyle_fixed_pool_reset(hyle_fixed_pool_o *p) {
  if (p)
    p->pool.reset();
}
size_t hyle_fixed_pool_used(const hyle_fixed_pool_o *p) {
  return p ? p->pool.used() : 0;
}
size_t hyle_fixed_pool_peak(const hyle_fixed_pool_o *p) {
  return p ? p->pool.peak() : 0;
}
size_t hyle_fixed_pool_capacity(const hyle_fixed_pool_o *p) {
  return p ? p->pool.capacity() : 0;
}
hyle_allocator hyle_fixed_pool_as_allocator(hyle_fixed_pool_o *p) {
  return p ? from_cpp(p->pool.as_allocator())
           : hyle_allocator{nullptr, nullptr, nullptr, nullptr};
}

/* ---- SharedPool (hard capacity) -------------------------------------------
 */

hyle_status hyle_shared_pool_create(hyle_shared_pool_o **out,
                                    hyle_allocator backing, size_t block_size,
                                    uint32_t block_count, size_t alignment) {
  if (out == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_shared_pool_create: out is NULL"));
  }
  *out = nullptr;
  if (!valid_backing(backing)) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_shared_pool_create: invalid backing (a fn pointer is null)"));
  }
  std::size_t normalized_alignment = 0;
  if (!normalize_alignment(alignment, &normalized_alignment) ||
      !valid_shared_pool_layout(block_size, block_count,
                                normalized_alignment)) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_shared_pool_create: invalid or overflowing pool layout"));
  }
  auto *p = allocate_handle<hyle_shared_pool_o>(
      backing, block_size, block_count, normalized_alignment);
  if (p == nullptr) {
    return steal_status(hyle::Status::ResourceExhausted(
        "hyle_shared_pool_create: control allocation failed (out of memory)"));
  }
  if (!p->pool.valid()) {
    free_handle(p);
    return steal_status(hyle::Status::ResourceExhausted(
        "hyle_shared_pool_create: slab allocation failed (out of memory)"));
  }
  *out = p;
  return steal_status(hyle::Status());
}

void hyle_shared_pool_destroy(hyle_shared_pool_o *p) { free_handle(p); }

void *hyle_shared_pool_allocate(hyle_shared_pool_o *p) {
  return p ? p->pool.allocate() : nullptr;
}

void hyle_shared_pool_deallocate(hyle_shared_pool_o *p, void *ptr) {
  if (p)
    p->pool.deallocate(ptr);
}

void hyle_shared_pool_reset(hyle_shared_pool_o *p) {
  if (p)
    p->pool.reset();
}

size_t hyle_shared_pool_block_size(const hyle_shared_pool_o *p) {
  return p ? p->pool.block_size() : 0;
}

uint32_t hyle_shared_pool_capacity(const hyle_shared_pool_o *p) {
  return p ? p->pool.capacity() : 0;
}

hyle_allocator hyle_shared_pool_as_allocator(hyle_shared_pool_o *p) {
  if (!p)
    return hyle_allocator{nullptr, nullptr, nullptr, nullptr};
  return from_cpp(p->pool.as_allocator());
}

/* ---- SharedPoolCache (explicit owner-local soft capacity) -----------------
 */

hyle_status hyle_shared_pool_cache_create(
    hyle_shared_pool_cache_o **out, hyle_shared_pool_o *pool,
    hyle_allocator metadata, uint32_t capacity, uint32_t transfer_batch) {
  if (out == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_shared_pool_cache_create: out is NULL"));
  }
  *out = nullptr;
  if (pool == nullptr || !valid_backing(metadata) || capacity == 0) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_shared_pool_cache_create: pool, metadata allocator and nonzero "
        "capacity are required"));
  }
  auto *p = allocate_handle<hyle_shared_pool_cache_o>(
      metadata, pool->pool, capacity, transfer_batch);
  if (p == nullptr) {
    return steal_status(hyle::Status::ResourceExhausted(
        "hyle_shared_pool_cache_create: control allocation failed (out of "
        "memory)"));
  }
  if (!p->cache.valid()) {
    free_handle(p);
    return steal_status(hyle::Status::ResourceExhausted(
        "hyle_shared_pool_cache_create: index metadata allocation failed "
        "(out of memory)"));
  }
  *out = p;
  return steal_status(hyle::Status());
}

void hyle_shared_pool_cache_destroy(hyle_shared_pool_cache_o *p) {
  free_handle(p);
}

void *hyle_shared_pool_cache_allocate(hyle_shared_pool_cache_o *p) {
  return p ? p->cache.allocate() : nullptr;
}

void hyle_shared_pool_cache_deallocate(hyle_shared_pool_cache_o *p,
                                       void *ptr) {
  if (p)
    p->cache.deallocate(ptr);
}

void hyle_shared_pool_cache_publish(hyle_shared_pool_cache_o *p) {
  if (p)
    p->cache.publish();
}

uint32_t hyle_shared_pool_cache_capacity(const hyle_shared_pool_cache_o *p) {
  return p ? p->cache.capacity() : 0;
}

uint32_t hyle_shared_pool_cache_cached(const hyle_shared_pool_cache_o *p) {
  return p ? p->cache.cached() : 0;
}

uint32_t
hyle_shared_pool_cache_transfer_batch(const hyle_shared_pool_cache_o *p) {
  return p ? p->cache.transfer_batch() : 0;
}

size_t hyle_shared_pool_cache_block_size(const hyle_shared_pool_cache_o *p) {
  return p ? p->cache.block_size() : 0;
}

hyle_allocator hyle_shared_pool_cache_as_allocator(hyle_shared_pool_cache_o *p) {
  if (!p)
    return hyle_allocator{nullptr, nullptr, nullptr, nullptr};
  return from_cpp(p->cache.as_allocator());
}

/* ---- StackAllocator --------------------------------------------------------
 */

hyle_status hyle_stack_allocator_create(hyle_stack_allocator_o **out,
                                        hyle_allocator backing, size_t size) {
  if (out == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_stack_allocator_create: out is NULL"));
  }
  *out = nullptr;
  // Validate C data before entering the C++ constructor; malformed input does
  // not expose a C++ assertion across the language boundary.
  if (!valid_backing(backing)) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_stack_allocator_create: invalid backing (a fn pointer is "
        "null)"));
  }
  if (size == 0) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_stack_allocator_create: size must be nonzero"));
  }
  auto *s = allocate_handle<hyle_stack_allocator_o>(backing, size);
  if (s == nullptr) {
    return steal_status(hyle::Status::ResourceExhausted(
        "hyle_stack_allocator_create: control allocation failed (out of "
        "memory)"));
  }
  // slab-OOM in the C++ ctor -> invalid (graceful, not abort; followup #2 arch
  // B).
  if (!s->stack.valid()) {
    free_handle(s);
    return steal_status(hyle::Status::ResourceExhausted(
        "hyle_stack_allocator_create: slab allocation failed (out of memory)"));
  }
  *out = s;
  return steal_status(hyle::Status());
}

void hyle_stack_allocator_destroy(hyle_stack_allocator_o *s) { free_handle(s); }

void *hyle_stack_allocator_allocate(hyle_stack_allocator_o *s, size_t bytes,
                                    size_t alignment) {
  std::size_t normalized = 0;
  return s && normalize_alignment(alignment, &normalized)
             ? s->stack.allocate(bytes, normalized)
             : nullptr;
}

void hyle_stack_allocator_reset(hyle_stack_allocator_o *s) {
  if (s)
    s->stack.reset();
}

void *hyle_stack_allocator_mark(const hyle_stack_allocator_o *s) {
  return s ? s->stack.mark() : nullptr;
}

void hyle_stack_allocator_rewind(hyle_stack_allocator_o *s, void *marker) {
  if (s)
    s->stack.rewind(static_cast<hyle::mem::StackAllocator::Marker>(marker));
}

size_t hyle_stack_allocator_used(const hyle_stack_allocator_o *s) {
  return s ? s->stack.used() : 0;
}

size_t hyle_stack_allocator_capacity(const hyle_stack_allocator_o *s) {
  return s ? s->stack.capacity() : 0;
}

size_t hyle_stack_allocator_remaining(const hyle_stack_allocator_o *s) {
  return s ? s->stack.remaining() : 0;
}

hyle_allocator hyle_stack_allocator_as_allocator(hyle_stack_allocator_o *s) {
  if (!s)
    return hyle_allocator{nullptr, nullptr, nullptr, nullptr};
  return from_cpp(s->stack.as_allocator());
}

/* ---- PooledRegion --------------------------------------------------------- */

hyle_status hyle_pooled_region_create(hyle_pooled_region_o **out,
                                      hyle_shared_pool_o *pool) {
  if (out == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_pooled_region_create: out is NULL"));
  }
  *out = nullptr;
  if (!pool) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_pooled_region_create: pool is NULL"));
  }
  auto *r = allocate_handle<hyle_pooled_region_o>(pool->backing, &pool->pool);
  if (r == nullptr) { // control OOM
    return steal_status(hyle::Status::ResourceExhausted(
        "hyle_pooled_region_create: control allocation failed (out of "
        "memory)"));
  }
  // Pool exhausted (no free block) is a graceful resource failure, not a
  // contract violation: return RESOURCE_EXHAUSTED so callers racing for a
  // bounded pool block can distinguish it from bad input.
  if (!r->region.valid()) {
    free_handle(r);
    return steal_status(hyle::Status::ResourceExhausted(
        "hyle_pooled_region_create: pool exhausted (no free block)"));
  }
  *out = r;
  return steal_status(hyle::Status());
}

void hyle_pooled_region_destroy(hyle_pooled_region_o *r) { free_handle(r); }

void *hyle_pooled_region_allocate(hyle_pooled_region_o *r, size_t bytes,
                                  size_t alignment) {
  std::size_t normalized = 0;
  return r && normalize_alignment(alignment, &normalized)
             ? r->region.allocate(bytes, normalized)
             : nullptr;
}

void hyle_pooled_region_clear(hyle_pooled_region_o *r) {
  if (r)
    r->region.clear();
}

size_t hyle_pooled_region_used(const hyle_pooled_region_o *r) {
  return r ? r->region.used() : 0;
}

size_t hyle_pooled_region_remaining(const hyle_pooled_region_o *r) {
  return r ? r->region.remaining() : 0;
}

size_t hyle_pooled_region_block_size(const hyle_pooled_region_o *r) {
  return r ? r->region.block_size() : 0;
}

hyle_allocator hyle_pooled_region_as_allocator(hyle_pooled_region_o *r) {
  if (!r)
    return hyle_allocator{nullptr, nullptr, nullptr, nullptr};
  return from_cpp(r->region.as_allocator());
}

/* ---- GrowablePool ----------------------------------------------------------
 */

hyle_status hyle_growable_pool_create(hyle_growable_pool_o **out,
                                      hyle_allocator backing,
                                      size_t slot_size, size_t slot_align,
                                      uint32_t initial_slots) {
  if (out == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_growable_pool_create: out is NULL"));
  }
  *out = nullptr;
  if (!valid_backing(backing)) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_growable_pool_create: invalid backing (a fn pointer is null)"));
  }
  const size_t normalized_align = norm_align(slot_align);
  if (!hyle::mem::detail::GrowablePoolCore::valid_layout(slot_size,
                                                         normalized_align)) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_growable_pool_create: invalid or overflowing slot layout"));
  }
  auto *p = allocate_handle<hyle_growable_pool_o>(
      backing, slot_size, normalized_align, initial_slots);
  if (p == nullptr) {
    return steal_status(hyle::Status::ResourceExhausted(
        "hyle_growable_pool_create: control allocation failed (out of "
        "memory)"));
  }
  *out = p;
  return steal_status(hyle::Status());
}

void hyle_growable_pool_destroy(hyle_growable_pool_o *p) { free_handle(p); }

void *hyle_growable_pool_acquire(hyle_growable_pool_o *p) {
  return p != nullptr ? p->core.acquire() : nullptr;
}

void hyle_growable_pool_release(hyle_growable_pool_o *p, void *slot) {
  if (p)
    p->core.release(slot);
}

void hyle_growable_pool_reset(hyle_growable_pool_o *p) {
  if (p)
    p->core.clear();
}

size_t hyle_growable_pool_capacity(const hyle_growable_pool_o *p) {
  return p ? p->core.capacity() : 0;
}

size_t hyle_growable_pool_live(const hyle_growable_pool_o *p) {
  return p ? p->core.live() : 0;
}

size_t hyle_growable_pool_slot_stride(const hyle_growable_pool_o *p) {
  return p ? p->core.slot_stride() : 0;
}

/* ---- HugeRegion ------------------------------------------------------------
 */

hyle_status hyle_huge_region_open(hyle_huge_region_o **out, size_t bytes,
                                  uint32_t flags, int numa_node) {
  constexpr uint32_t kKnownFlags = HYLE_HUGE_THP_HINT | HYLE_HUGE_PREFAULT |
                                   HYLE_HUGE_NUMA_BIND |
                                   HYLE_HUGE_NUMA_INTERLEAVE;
  if (out == nullptr) {
    return steal_status(
        hyle::Status::InvalidArgument("hyle_huge_region_open: out is NULL"));
  }
  *out = nullptr;
  if (bytes == 0 || (flags & ~kKnownFlags) != 0 ||
      ((flags & HYLE_HUGE_NUMA_BIND) != 0 &&
       (flags & HYLE_HUGE_NUMA_INTERLEAVE) != 0)) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_huge_region_open: invalid size or flag combination"));
  }
  auto r =
      hyle::mem::HugeRegion::allocate(bytes, to_huge_flags(flags), numa_node);
  if (!r) return steal_status(std::move(r).error());
  const hyle_allocator backing = system_backing();
  auto *handle =
      allocate_handle<hyle_huge_region_o>(backing, *std::move(r));
  if (handle == nullptr) {
    return steal_status(hyle::Status::ResourceExhausted(
        "hyle_huge_region_open: control allocation failed (out of memory)"));
  }
  *out = handle;
  return steal_status(hyle::Status());
}

hyle_status hyle_huge_region_open_local(hyle_huge_region_o **out, size_t bytes,
                                        uint32_t flags) {
  constexpr uint32_t kKnownFlags = HYLE_HUGE_THP_HINT | HYLE_HUGE_PREFAULT |
                                   HYLE_HUGE_NUMA_BIND |
                                   HYLE_HUGE_NUMA_INTERLEAVE;
  if (out == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_huge_region_open_local: out is NULL"));
  }
  *out = nullptr;
  if (bytes == 0 || (flags & ~kKnownFlags) != 0 ||
      ((flags & HYLE_HUGE_NUMA_BIND) != 0 &&
       (flags & HYLE_HUGE_NUMA_INTERLEAVE) != 0)) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_huge_region_open_local: invalid size or flag combination"));
  }
  auto r = hyle::mem::HugeRegion::allocate_local(bytes, to_huge_flags(flags));
  if (!r) return steal_status(std::move(r).error());
  const hyle_allocator backing = system_backing();
  auto *handle =
      allocate_handle<hyle_huge_region_o>(backing, *std::move(r));
  if (handle == nullptr) {
    return steal_status(hyle::Status::ResourceExhausted(
        "hyle_huge_region_open_local: control allocation failed (out of "
        "memory)"));
  }
  *out = handle;
  return steal_status(hyle::Status());
}

hyle_status hyle_huge_region_open_interleaved(hyle_huge_region_o **out,
                                              size_t bytes) {
  if (out == nullptr) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_huge_region_open_interleaved: out is NULL"));
  }
  *out = nullptr;
  if (bytes == 0) {
    return steal_status(hyle::Status::InvalidArgument(
        "hyle_huge_region_open_interleaved: size must be nonzero"));
  }
  auto r = hyle::mem::HugeRegion::allocate_interleaved(bytes);
  if (!r) return steal_status(std::move(r).error());
  const hyle_allocator backing = system_backing();
  auto *handle =
      allocate_handle<hyle_huge_region_o>(backing, *std::move(r));
  if (handle == nullptr) {
    return steal_status(hyle::Status::ResourceExhausted(
        "hyle_huge_region_open_interleaved: control allocation failed (out "
        "of memory)"));
  }
  *out = handle;
  return steal_status(hyle::Status());
}

void hyle_huge_region_destroy(hyle_huge_region_o *r) { free_handle(r); }

void *hyle_huge_region_data(hyle_huge_region_o *r) {
  return r ? r->region.data() : nullptr;
}

size_t hyle_huge_region_size(const hyle_huge_region_o *r) {
  return r ? r->region.size() : 0;
}

uint32_t hyle_huge_region_requested_flags(const hyle_huge_region_o *r) {
  return r ? static_cast<uint32_t>(r->region.requested_flags()) : 0;
}

uint32_t hyle_huge_region_accepted_flags(const hyle_huge_region_o *r) {
  return r ? static_cast<uint32_t>(r->region.accepted_flags()) : 0;
}

int hyle_huge_region_requested_numa_node(const hyle_huge_region_o *r) {
  return r ? r->region.requested_numa_node() : HYLE_MEM_NUMA_ANY;
}

int hyle_huge_region_numa_placement_applied(const hyle_huge_region_o *r) {
  return r && r->region.numa_placement_applied() ? 1 : 0;
}

/* ---- process-wide policy ---------------------------------------------------
 */

void hyle_enable_allocator_hugepages(int allow_thp, int allow_large_os_pages) {
  hyle::mem::enable_allocator_hugepages(
      {allow_thp != 0, allow_large_os_pages != 0});
}

void hyle_enable_allocator_numa(unsigned max_nodes) {
  hyle::mem::enable_allocator_numa(max_nodes);
}

hyle_status hyle_reserve_huge_1g_pages(size_t pages, int numa_node,
                                       unsigned timeout_ms) {
  return steal_status(
      hyle::mem::reserve_huge_1g_pages(pages, numa_node, timeout_ms));
}

} // extern "C"

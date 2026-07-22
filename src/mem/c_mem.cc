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
#include <cstdio>
#include <new>
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
#include "hyle/platform/virtual_memory.h" // vm_last_error() for hyle_mem_last_error

namespace {

using hyle::mem::Allocator;

// The C mem surface owns its diagnostic channel. In particular, a failed mem
// factory must not overwrite a platform VM error observed by an unrelated
// caller on the same thread.
thread_local char g_mem_last_error[512] = {};
thread_local hyle_mem_error g_mem_last_error_code = HYLE_MEM_ERROR_NONE;

void set_mem_error(hyle_mem_error code, const char *message) noexcept {
  g_mem_last_error_code = code;
  if (code == HYLE_MEM_ERROR_NONE || message == nullptr ||
      message[0] == '\0') {
    g_mem_last_error[0] = '\0';
    return;
  }
  std::snprintf(g_mem_last_error, sizeof(g_mem_last_error), "%s", message);
}

void set_mem_error_from_platform(const char *fallback) noexcept {
  const char *message = hyle::platform::vm_last_error();
  set_mem_error(HYLE_MEM_ERROR_PLATFORM,
                message != nullptr && message[0] != '\0' ? message : fallback);
}

class VmErrorCapture {
public:
  VmErrorCapture() noexcept {
    const char *previous = hyle::platform::vm_last_error();
    std::snprintf(previous_, sizeof(previous_), "%s",
                  previous != nullptr ? previous : "");
    hyle::platform::vm_set_last_error(nullptr);
  }
  ~VmErrorCapture() {
    hyle::platform::vm_set_last_error(previous_[0] != '\0' ? previous_
                                                               : nullptr);
  }
  VmErrorCapture(const VmErrorCapture &) = delete;
  VmErrorCapture &operator=(const VmErrorCapture &) = delete;

private:
  char previous_[512] = {};
};

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
  if ((alignment & (alignment - 1)) != 0) {
    set_mem_error(HYLE_MEM_ERROR_INVALID_ARGUMENT,
        "mem: alignment must be zero or a power of two");
    return false;
  }
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

struct hyle_region_o {
  hyle_allocator backing;
  hyle::mem::Region region;
  hyle_region_o(hyle_allocator b, hyle::mem::RegionOptions options) noexcept
      : backing(b), region(to_cpp(&backing), options) {}
};

struct hyle_fixed_region_o {
  hyle_allocator backing;
  bool owns_control;
  hyle::mem::FixedRegion region;
  hyle_fixed_region_o(hyle_allocator b, void *storage,
                      std::size_t size, bool owns = true) noexcept
      : backing(b), owns_control(owns), region(storage, size) {}
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

static_assert(sizeof(hyle_fixed_region_o) <=
              HYLE_FIXED_REGION_CONTROL_BYTES);
static_assert(alignof(hyle_fixed_region_o) <=
              alignof(hyle_fixed_region_storage));
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

/* ---- allocator adapters ---------------------------------------------------
 */

hyle_quota_allocator_o *hyle_quota_allocator_create(hyle_allocator backing,
                                                    size_t limit) {
  set_mem_error(HYLE_MEM_ERROR_NONE, nullptr);
  if (!valid_backing(backing)) {
    set_mem_error(HYLE_MEM_ERROR_INVALID_ARGUMENT,
        "hyle_quota_allocator_create: invalid backing");
    return nullptr;
  }
  auto *q = allocate_handle<hyle_quota_allocator_o>(backing, limit);
  if (q == nullptr)
    set_mem_error(HYLE_MEM_ERROR_OUT_OF_MEMORY,
        "hyle_quota_allocator_create: control allocation failed");
  return q;
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

/* ---- Region family --------------------------------------------------------
 */

hyle_region_o *hyle_region_create(hyle_allocator backing,
                                  const hyle_region_options *options) {
  set_mem_error(HYLE_MEM_ERROR_NONE, nullptr);
  if (!valid_backing(backing)) {
    set_mem_error(HYLE_MEM_ERROR_INVALID_ARGUMENT,
                  "hyle_region_create: invalid backing");
    return nullptr;
  }
  hyle_region_options defaults = HYLE_REGION_OPTIONS_INIT;
  const hyle_region_options *in = options != nullptr ? options : &defaults;
  if (in->struct_size < sizeof(hyle_region_options) ||
      in->initial_block_size == 0 ||
      in->max_block_size < in->initial_block_size ||
      in->large_allocation_threshold == 0) {
    set_mem_error(HYLE_MEM_ERROR_INVALID_ARGUMENT,
                  "hyle_region_create: invalid options");
    return nullptr;
  }
  hyle::mem::RegionOptions cpp;
  cpp.initial_block_size = in->initial_block_size;
  cpp.max_block_size = in->max_block_size;
  cpp.large_allocation_threshold = in->large_allocation_threshold;
  cpp.size_limit = in->size_limit;
  auto *r = allocate_handle<hyle_region_o>(backing, cpp);
  if (r == nullptr)
    set_mem_error(HYLE_MEM_ERROR_OUT_OF_MEMORY,
        "hyle_region_create: control allocation failed");
  return r;
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

hyle_fixed_region_o *
hyle_fixed_region_init(hyle_fixed_region_storage *control, void *storage,
                       size_t size) {
  set_mem_error(HYLE_MEM_ERROR_NONE, nullptr);
  if (control == nullptr || storage == nullptr || size == 0) {
    set_mem_error(HYLE_MEM_ERROR_INVALID_ARGUMENT,
                  "hyle_fixed_region_init: invalid control or storage");
    return nullptr;
  }
  const hyle_allocator backing = system_backing();
  auto *r = ::new (static_cast<void *>(control->bytes))
      hyle_fixed_region_o(backing, storage, size, false);
  if (!r->region.valid()) {
    destroy_external_control(r);
    set_mem_error(HYLE_MEM_ERROR_INVALID_ARGUMENT,
                  "hyle_fixed_region_init: invalid storage");
    return nullptr;
  }
  return r;
}

hyle_fixed_region_o *hyle_fixed_region_create(void *storage, size_t size) {
  set_mem_error(HYLE_MEM_ERROR_NONE, nullptr);
  if (storage == nullptr || size == 0) {
    set_mem_error(HYLE_MEM_ERROR_INVALID_ARGUMENT,
        "hyle_fixed_region_create: invalid storage");
    return nullptr;
  }
  const hyle_allocator backing = system_backing();
  auto *r = allocate_handle<hyle_fixed_region_o>(backing, storage, size);
  if (r == nullptr) {
    set_mem_error(HYLE_MEM_ERROR_OUT_OF_MEMORY,
                  "hyle_fixed_region_create: control allocation failed");
    return nullptr;
  }
  if (!r->region.valid()) {
    free_handle(r);
    set_mem_error(HYLE_MEM_ERROR_INVALID_ARGUMENT,
                  "hyle_fixed_region_create: invalid storage");
    return nullptr;
  }
  return r;
}
void hyle_fixed_region_destroy(hyle_fixed_region_o *r) {
  destroy_external_control(r);
}
void *hyle_fixed_region_allocate(hyle_fixed_region_o *r, size_t bytes,
                                 size_t alignment) {
  std::size_t normalized = 0;
  return r && normalize_alignment(alignment, &normalized)
             ? r->region.allocate(bytes, normalized)
             : nullptr;
}
void hyle_fixed_region_clear(hyle_fixed_region_o *r) {
  if (r)
    r->region.clear();
}
size_t hyle_fixed_region_used(const hyle_fixed_region_o *r) {
  return r ? r->region.used() : 0;
}
size_t hyle_fixed_region_capacity(const hyle_fixed_region_o *r) {
  return r ? r->region.capacity() : 0;
}
hyle_allocator hyle_fixed_region_as_allocator(hyle_fixed_region_o *r) {
  return r ? from_cpp(r->region.as_allocator())
           : hyle_allocator{nullptr, nullptr, nullptr, nullptr};
}

hyle_fixed_pool_o *
hyle_fixed_pool_init(hyle_fixed_pool_storage *control, void *storage,
                     size_t size) {
  set_mem_error(HYLE_MEM_ERROR_NONE, nullptr);
  if (control == nullptr) {
    set_mem_error(HYLE_MEM_ERROR_INVALID_ARGUMENT,
                  "hyle_fixed_pool_init: invalid control");
    return nullptr;
  }
  const hyle_allocator backing = system_backing();
  auto *p = ::new (static_cast<void *>(control->bytes))
      hyle_fixed_pool_o(backing, storage, size, false);
  if (!p->pool.valid()) {
    destroy_external_control(p);
    set_mem_error(HYLE_MEM_ERROR_INVALID_ARGUMENT,
                  "hyle_fixed_pool_init: invalid storage");
    return nullptr;
  }
  return p;
}

hyle_fixed_pool_o *hyle_fixed_pool_create(void *storage, size_t size) {
  set_mem_error(HYLE_MEM_ERROR_NONE, nullptr);
  const hyle_allocator backing = system_backing();
  auto *p = allocate_handle<hyle_fixed_pool_o>(backing, storage, size);
  if (p == nullptr) {
    set_mem_error(HYLE_MEM_ERROR_OUT_OF_MEMORY,
                  "hyle_fixed_pool_create: control allocation failed");
    return nullptr;
  }
  if (!p->pool.valid()) {
    free_handle(p);
    set_mem_error(HYLE_MEM_ERROR_INVALID_ARGUMENT,
                  "hyle_fixed_pool_create: invalid storage");
    return nullptr;
  }
  return p;
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
size_t hyle_fixed_pool_capacity(const hyle_fixed_pool_o *p) {
  return p ? p->pool.capacity() : 0;
}
hyle_allocator hyle_fixed_pool_as_allocator(hyle_fixed_pool_o *p) {
  return p ? from_cpp(p->pool.as_allocator())
           : hyle_allocator{nullptr, nullptr, nullptr, nullptr};
}

/* ---- SharedPool (hard capacity) -------------------------------------------
 */

hyle_shared_pool_o *hyle_shared_pool_create(hyle_allocator backing,
                                            size_t block_size,
                                            uint32_t block_count,
                                            size_t alignment) {
  set_mem_error(HYLE_MEM_ERROR_NONE, nullptr);
  if (!valid_backing(backing)) {
    set_mem_error(HYLE_MEM_ERROR_INVALID_ARGUMENT,
        "hyle_shared_pool_create: invalid backing (a fn pointer is null)");
    return nullptr;
  }
  std::size_t normalized_alignment = 0;
  if (!normalize_alignment(alignment, &normalized_alignment) ||
      !valid_shared_pool_layout(block_size, block_count,
                                normalized_alignment)) {
    set_mem_error(HYLE_MEM_ERROR_INVALID_ARGUMENT,
        "hyle_shared_pool_create: invalid or overflowing pool layout");
    return nullptr;
  }
  auto *p = allocate_handle<hyle_shared_pool_o>(
      backing, block_size, block_count, normalized_alignment);
  if (p == nullptr) {
    set_mem_error(HYLE_MEM_ERROR_OUT_OF_MEMORY,
        "hyle_shared_pool_create: control allocation failed (out of memory)");
    return nullptr;
  }
  if (!p->pool.valid()) {
    free_handle(p);
    set_mem_error(HYLE_MEM_ERROR_OUT_OF_MEMORY,
        "hyle_shared_pool_create: slab allocation failed (out of memory)");
    return nullptr;
  }
  set_mem_error(HYLE_MEM_ERROR_NONE, nullptr);
  return p;
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

hyle_shared_pool_cache_o *
hyle_shared_pool_cache_create(hyle_shared_pool_o *pool,
                              hyle_allocator metadata, uint32_t capacity,
                              uint32_t transfer_batch) {
  set_mem_error(HYLE_MEM_ERROR_NONE, nullptr);
  if (pool == nullptr || !valid_backing(metadata) || capacity == 0) {
    set_mem_error(HYLE_MEM_ERROR_INVALID_ARGUMENT,
                  "hyle_shared_pool_cache_create: pool, metadata allocator "
                  "and nonzero capacity are required");
    return nullptr;
  }
  auto *p = allocate_handle<hyle_shared_pool_cache_o>(
      metadata, pool->pool, capacity, transfer_batch);
  if (p == nullptr) {
    set_mem_error(HYLE_MEM_ERROR_OUT_OF_MEMORY,
                  "hyle_shared_pool_cache_create: control "
                  "allocation failed (out of memory)");
    return nullptr;
  }
  if (!p->cache.valid()) {
    free_handle(p);
    set_mem_error(HYLE_MEM_ERROR_OUT_OF_MEMORY,
                  "hyle_shared_pool_cache_create: index metadata "
                  "allocation failed (out of memory)");
    return nullptr;
  }
  set_mem_error(HYLE_MEM_ERROR_NONE, nullptr);
  return p;
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

hyle_stack_allocator_o *hyle_stack_allocator_create(hyle_allocator backing,
                                                    size_t size) {
  set_mem_error(HYLE_MEM_ERROR_NONE, nullptr);
  // Validate C data before entering the C++ constructor; malformed input does
  // not expose a C++ assertion across the language boundary.
  if (!valid_backing(backing)) {
    set_mem_error(HYLE_MEM_ERROR_INVALID_ARGUMENT,
        "hyle_stack_allocator_create: invalid backing (a fn pointer is null)");
    return nullptr;
  }
  if (size == 0) {
    set_mem_error(HYLE_MEM_ERROR_INVALID_ARGUMENT,
        "hyle_stack_allocator_create: size must be nonzero");
    return nullptr;
  }
  auto *s = allocate_handle<hyle_stack_allocator_o>(backing, size);
  if (s == nullptr) {
    set_mem_error(HYLE_MEM_ERROR_OUT_OF_MEMORY,
                  "hyle_stack_allocator_create: control "
                  "allocation failed (out of memory)");
    return nullptr;
  }
  // slab-OOM in the C++ ctor -> invalid (graceful, not abort; followup #2 arch
  // B).
  if (!s->stack.valid()) {
    free_handle(s);
    set_mem_error(HYLE_MEM_ERROR_OUT_OF_MEMORY,
        "hyle_stack_allocator_create: slab allocation failed (out of memory)");
    return nullptr;
  }
  set_mem_error(HYLE_MEM_ERROR_NONE, nullptr);
  return s;
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

hyle_pooled_region_o *hyle_pooled_region_create(hyle_shared_pool_o *pool) {
  set_mem_error(HYLE_MEM_ERROR_NONE, nullptr);
  if (!pool) {
    set_mem_error(HYLE_MEM_ERROR_INVALID_ARGUMENT,
        "hyle_pooled_region_create: pool is NULL");
    return nullptr;
  }
  auto *r = allocate_handle<hyle_pooled_region_o>(pool->backing, &pool->pool);
  if (r == nullptr) { // control OOM
    set_mem_error(HYLE_MEM_ERROR_OUT_OF_MEMORY,
        "hyle_pooled_region_create: control allocation failed (out of memory)");
    return nullptr;
  }
  // Pool exhausted (no free block) is a graceful resource failure, not a
  // contract violation: return NULL + set last-error so an FFI caller has a
  // defensive path for callers that race for a bounded pool block.
  if (!r->region.valid()) {
    free_handle(r);
    set_mem_error(HYLE_MEM_ERROR_EXHAUSTED,
        "hyle_pooled_region_create: pool exhausted (no free block)");
    return nullptr;
  }
  set_mem_error(HYLE_MEM_ERROR_NONE, nullptr);
  return r;
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

hyle_growable_pool_o *hyle_growable_pool_create(hyle_allocator backing,
                                                size_t slot_size,
                                                size_t slot_align,
                                                uint32_t initial_slots) {
  set_mem_error(HYLE_MEM_ERROR_NONE, nullptr);
  if (!valid_backing(backing)) {
    set_mem_error(HYLE_MEM_ERROR_INVALID_ARGUMENT,
        "hyle_growable_pool_create: invalid backing (a fn pointer is null)");
    return nullptr;
  }
  const size_t normalized_align = norm_align(slot_align);
  if (!hyle::mem::detail::GrowablePoolCore::valid_layout(slot_size,
                                                         normalized_align)) {
    set_mem_error(HYLE_MEM_ERROR_INVALID_ARGUMENT,
        "hyle_growable_pool_create: invalid or overflowing slot layout");
    return nullptr;
  }
  auto *p = allocate_handle<hyle_growable_pool_o>(
      backing, slot_size, normalized_align, initial_slots);
  if (p == nullptr)
    set_mem_error(HYLE_MEM_ERROR_OUT_OF_MEMORY,
        "hyle_growable_pool_create: control allocation failed (out of memory)");
  else
    set_mem_error(HYLE_MEM_ERROR_NONE, nullptr);
  return p;
}

void hyle_growable_pool_destroy(hyle_growable_pool_o *p) { free_handle(p); }

void *hyle_growable_pool_acquire(hyle_growable_pool_o *p) {
  if (p == nullptr) {
    set_mem_error(HYLE_MEM_ERROR_INVALID_ARGUMENT,
        "hyle_growable_pool_acquire: pool is NULL");
    return nullptr;
  }
  void *slot = p->core.acquire();
  set_mem_error(slot == nullptr ? HYLE_MEM_ERROR_OUT_OF_MEMORY
                                : HYLE_MEM_ERROR_NONE,
                slot == nullptr
                    ? "hyle_growable_pool_acquire: growth allocation failed "
                      "(out of memory)"
                    : nullptr);
  return slot;
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

/* ---- HugeRegion ------------------------------------------------------------
 */

hyle_huge_region_o *hyle_huge_region_open(size_t bytes, uint32_t flags,
                                          int numa_node) {
  constexpr uint32_t kKnownFlags = HYLE_HUGE_THP_HINT | HYLE_HUGE_PREFAULT |
                                   HYLE_HUGE_NUMA_BIND |
                                   HYLE_HUGE_NUMA_INTERLEAVE;
  set_mem_error(HYLE_MEM_ERROR_NONE, nullptr);
  if (bytes == 0 || (flags & ~kKnownFlags) != 0 ||
      ((flags & HYLE_HUGE_NUMA_BIND) != 0 &&
       (flags & HYLE_HUGE_NUMA_INTERLEAVE) != 0)) {
    set_mem_error(HYLE_MEM_ERROR_INVALID_ARGUMENT,
                  "hyle_huge_region_open: invalid size or flag combination");
    return nullptr;
  }
  VmErrorCapture vm_error;
  auto r =
      hyle::mem::HugeRegion::allocate(bytes, to_huge_flags(flags), numa_node);
  if (!r.valid()) {
    set_mem_error_from_platform("hyle_huge_region_open: allocation failed");
    return nullptr;
  }
  const hyle_allocator backing = system_backing();
  auto *out = allocate_handle<hyle_huge_region_o>(backing, std::move(r));
  if (out == nullptr) {
    set_mem_error(HYLE_MEM_ERROR_OUT_OF_MEMORY,
        "hyle_huge_region_open: control allocation failed (out of memory)");
  } else {
    set_mem_error(HYLE_MEM_ERROR_NONE, nullptr);
  }
  return out;
}

hyle_huge_region_o *hyle_huge_region_open_local(size_t bytes, uint32_t flags) {
  constexpr uint32_t kKnownFlags = HYLE_HUGE_THP_HINT | HYLE_HUGE_PREFAULT |
                                   HYLE_HUGE_NUMA_BIND |
                                   HYLE_HUGE_NUMA_INTERLEAVE;
  set_mem_error(HYLE_MEM_ERROR_NONE, nullptr);
  if (bytes == 0 || (flags & ~kKnownFlags) != 0 ||
      ((flags & HYLE_HUGE_NUMA_BIND) != 0 &&
       (flags & HYLE_HUGE_NUMA_INTERLEAVE) != 0)) {
    set_mem_error(
        HYLE_MEM_ERROR_INVALID_ARGUMENT,
        "hyle_huge_region_open_local: invalid size or flag combination");
    return nullptr;
  }
  VmErrorCapture vm_error;
  auto r = hyle::mem::HugeRegion::allocate_local(bytes, to_huge_flags(flags));
  if (!r.valid()) {
    set_mem_error_from_platform(
        "hyle_huge_region_open_local: allocation failed");
    return nullptr;
  }
  const hyle_allocator backing = system_backing();
  auto *out = allocate_handle<hyle_huge_region_o>(backing, std::move(r));
  if (out == nullptr) {
    set_mem_error(HYLE_MEM_ERROR_OUT_OF_MEMORY,
                  "hyle_huge_region_open_local: control allocation failed "
                  "(out of memory)");
  } else {
    set_mem_error(HYLE_MEM_ERROR_NONE, nullptr);
  }
  return out;
}

hyle_huge_region_o *hyle_huge_region_open_interleaved(size_t bytes) {
  set_mem_error(HYLE_MEM_ERROR_NONE, nullptr);
  if (bytes == 0) {
    set_mem_error(HYLE_MEM_ERROR_INVALID_ARGUMENT,
                  "hyle_huge_region_open_interleaved: size must be nonzero");
    return nullptr;
  }
  VmErrorCapture vm_error;
  auto r = hyle::mem::HugeRegion::allocate_interleaved(bytes);
  if (!r.valid()) {
    set_mem_error_from_platform(
        "hyle_huge_region_open_interleaved: allocation failed");
    return nullptr;
  }
  const hyle_allocator backing = system_backing();
  auto *out = allocate_handle<hyle_huge_region_o>(backing, std::move(r));
  if (out == nullptr) {
    set_mem_error(HYLE_MEM_ERROR_OUT_OF_MEMORY,
        "hyle_huge_region_open_interleaved: control allocation failed (out of "
        "memory)");
  } else {
    set_mem_error(HYLE_MEM_ERROR_NONE, nullptr);
  }
  return out;
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

/* ---- diagnostics -----------------------------------------------------------
 */

const char *hyle_mem_last_error(void) {
  return g_mem_last_error;
}

hyle_mem_error hyle_mem_last_error_code(void) {
  return g_mem_last_error_code;
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

int hyle_reserve_huge_1g_pages(size_t pages, int numa_node,
                               unsigned timeout_ms) {
  return hyle::mem::reserve_huge_1g_pages(pages, numa_node, timeout_ms) ? 1 : 0;
}

} // extern "C"

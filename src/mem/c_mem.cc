// C-ABI bridge for mem (include/hyle/c/mem.h).
//
// Two directions, both WITHOUT reinterpret_cast on the vtable:
//   C -> C++ : a caller's hyle_allocator is stored by value in the opaque
//   wrapper;
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
#include <new>
#include <type_traits>
#include <utility>

#include "hyle/mem/allocator.h"
#include "hyle/mem/fixed_pool.h"
#include "hyle/mem/growable_pool.h"
#include "hyle/mem/hugepage.h"
#include "hyle/mem/quota_allocator.h"
#include "hyle/mem/region.h"
#include "hyle/mem/shared_pool.h"
#include "hyle/mem/stack_allocator.h"
#include "hyle/platform/virtual_memory.h" // vm_last_error() for hyle_mem_last_error

namespace {

using hyle::mem::Allocator;

static_assert(std::is_standard_layout_v<hyle_allocator>);
static_assert(std::is_trivially_copyable_v<hyle_allocator>);
static_assert(sizeof(hyle_allocator) == sizeof(void *) * 4);

// ---- C hyle_allocator -> C++ mem::Allocator (via noexcept trampolines)
// ------- `state` points at a live hyle_allocator (stored inside the owning
// wrapper, so it outlives every C++ object built from it).
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

// ---- opaque wrappers --------------------------------------------------------
// Each stores the caller's hyle_allocator FIRST (so it is destroyed LAST) and
// the C++ object SECOND (destroyed first -- it may free through `backing` in
// its dtor).

struct hyle_shared_pool_o {
  hyle_allocator backing;
  hyle::mem::SharedPool pool;
  hyle_shared_pool_o(hyle_allocator b, std::size_t block_size,
                     std::uint32_t block_count, std::size_t alignment)
      : backing(b), pool(to_cpp(&backing), block_size, block_count, alignment) {
  }
};

struct hyle_quota_allocator_o {
  hyle_allocator backing;
  hyle::mem::QuotaAllocator quota;
  hyle_quota_allocator_o(hyle_allocator b, std::size_t limit)
      : backing(b), quota(to_cpp(&backing), limit) {}
};

struct hyle_region_o {
  hyle_allocator backing;
  hyle::mem::Region region;
  hyle_region_o(hyle_allocator b, hyle::mem::RegionOptions options)
      : backing(b), region(to_cpp(&backing), options) {}
};

struct hyle_fixed_region_o {
  hyle::mem::FixedRegion region;
  hyle_fixed_region_o(void *storage, std::size_t size)
      : region(storage, size) {}
};

struct hyle_fixed_pool_o {
  hyle::mem::FixedPool pool;
  hyle_fixed_pool_o(void *storage, std::size_t size) : pool(storage, size) {}
};

struct hyle_shared_pool_cached_o {
  hyle_allocator backing;
  hyle::mem::SharedPoolCached pool;
  hyle_shared_pool_cached_o(hyle_allocator b, std::size_t block_size,
                            std::uint32_t block_count, std::size_t alignment)
      : backing(b), pool(to_cpp(&backing), block_size, block_count, alignment) {
  }
};

struct hyle_stack_allocator_o {
  hyle_allocator backing;
  hyle::mem::StackAllocator stack;
  hyle_stack_allocator_o(hyle_allocator b, std::size_t size)
      : backing(b), stack(to_cpp(&backing), size) {}
};

struct hyle_arena_o {
  hyle::mem::Arena arena; // borrows a block from a pool (no owned backing)
  explicit hyle_arena_o(hyle::mem::SharedPool *pool) : arena(pool) {}
};

struct hyle_growable_pool_o {
  hyle_allocator backing;
  hyle::mem::detail::GrowablePoolCore core;
  hyle_growable_pool_o(hyle_allocator b, std::size_t slot_size,
                       std::size_t slot_align, std::uint32_t initial_slots)
      : backing(b),
        core(to_cpp(&backing), slot_size, slot_align, initial_slots) {}
};

struct hyle_huge_region_o {
  hyle::mem::HugeRegion region;
  explicit hyle_huge_region_o(hyle::mem::HugeRegion &&r)
      : region(std::move(r)) {}
};

extern "C" {

/* ---- allocator vtable ------------------------------------------------------
 */

size_t hyle_default_alignment(void) { return alignof(std::max_align_t); }

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
  hyle::platform::vm_set_last_error(nullptr);
  if (!valid_backing(backing)) {
    hyle::platform::vm_set_last_error(
        "hyle_quota_allocator_create: invalid backing");
    return nullptr;
  }
  auto *q = new (std::nothrow) hyle_quota_allocator_o(backing, limit);
  if (q == nullptr)
    hyle::platform::vm_set_last_error(
        "hyle_quota_allocator_create: wrapper allocation failed");
  return q;
}

void hyle_quota_allocator_destroy(hyle_quota_allocator_o *q) { delete q; }
hyle_allocator hyle_quota_allocator_get(hyle_quota_allocator_o *q) {
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
  hyle::platform::vm_set_last_error(nullptr);
  if (!valid_backing(backing)) {
    hyle::platform::vm_set_last_error("hyle_region_create: invalid backing");
    return nullptr;
  }
  hyle_region_options defaults = HYLE_REGION_OPTIONS_INIT;
  const hyle_region_options *in = options != nullptr ? options : &defaults;
  if (in->struct_size < sizeof(hyle_region_options) ||
      in->initial_block_size == 0 ||
      in->max_block_size < in->initial_block_size ||
      in->large_allocation_threshold == 0) {
    hyle::platform::vm_set_last_error("hyle_region_create: invalid options");
    return nullptr;
  }
  hyle::mem::RegionOptions cpp;
  cpp.initial_block_size = in->initial_block_size;
  cpp.max_block_size = in->max_block_size;
  cpp.large_allocation_threshold = in->large_allocation_threshold;
  cpp.size_limit = in->size_limit;
  auto *r = new (std::nothrow) hyle_region_o(backing, cpp);
  if (r == nullptr)
    hyle::platform::vm_set_last_error(
        "hyle_region_create: wrapper allocation failed");
  return r;
}

void hyle_region_destroy(hyle_region_o *r) { delete r; }
void *hyle_region_allocate(hyle_region_o *r, size_t bytes, size_t alignment) {
  return r ? r->region.allocate(bytes, norm_align(alignment)) : nullptr;
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

hyle_fixed_region_o *hyle_fixed_region_create(void *storage, size_t size) {
  hyle::platform::vm_set_last_error(nullptr);
  if (storage == nullptr || size == 0) {
    hyle::platform::vm_set_last_error(
        "hyle_fixed_region_create: invalid storage");
    return nullptr;
  }
  auto *r = new (std::nothrow) hyle_fixed_region_o(storage, size);
  if (r == nullptr)
    hyle::platform::vm_set_last_error(
        "hyle_fixed_region_create: wrapper allocation failed");
  return r;
}
void hyle_fixed_region_destroy(hyle_fixed_region_o *r) { delete r; }
void *hyle_fixed_region_allocate(hyle_fixed_region_o *r, size_t bytes,
                                 size_t alignment) {
  return r ? r->region.allocate(bytes, norm_align(alignment)) : nullptr;
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

hyle_fixed_pool_o *hyle_fixed_pool_create(void *storage, size_t size) {
  hyle::platform::vm_set_last_error(nullptr);
  auto *p = new (std::nothrow) hyle_fixed_pool_o(storage, size);
  if (p == nullptr || !p->pool.valid()) {
    delete p;
    hyle::platform::vm_set_last_error(
        "hyle_fixed_pool_create: invalid storage or wrapper OOM");
    return nullptr;
  }
  return p;
}
void hyle_fixed_pool_destroy(hyle_fixed_pool_o *p) { delete p; }
void *hyle_fixed_pool_allocate(hyle_fixed_pool_o *p, size_t bytes,
                               size_t alignment) {
  return p ? p->pool.allocate(bytes, norm_align(alignment)) : nullptr;
}
void hyle_fixed_pool_deallocate(hyle_fixed_pool_o *p, void *ptr) {
  if (p)
    p->pool.deallocate(ptr);
}
void *hyle_fixed_pool_reallocate(hyle_fixed_pool_o *p, void *ptr,
                                 size_t old_size, size_t new_size,
                                 size_t alignment) {
  return p ? p->pool.reallocate(ptr, old_size, new_size, norm_align(alignment))
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
  hyle::platform::vm_set_last_error(nullptr);
  if (!valid_backing(backing)) {
    hyle::platform::vm_set_last_error(
        "hyle_shared_pool_create: invalid backing (a fn pointer is null)");
    return nullptr;
  }
  auto *p = new (std::nothrow) hyle_shared_pool_o(
      backing, block_size, block_count, norm_align(alignment));
  if (p == nullptr) {
    hyle::platform::vm_set_last_error(
        "hyle_shared_pool_create: wrapper allocation failed (out of memory)");
    return nullptr;
  }
  if (!p->pool.valid()) {
    delete p;
    hyle::platform::vm_set_last_error(
        "hyle_shared_pool_create: slab allocation failed (out of memory)");
    return nullptr;
  }
  hyle::platform::vm_set_last_error(nullptr);
  return p;
}

void hyle_shared_pool_destroy(hyle_shared_pool_o *p) { delete p; }

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

/* ---- SharedPoolCached (soft capacity) -------------------------------------
 */

hyle_shared_pool_cached_o *
hyle_shared_pool_cached_create(hyle_allocator backing, size_t block_size,
                               uint32_t block_count, size_t alignment) {
  hyle::platform::vm_set_last_error(nullptr);
  if (!valid_backing(backing)) {
    hyle::platform::vm_set_last_error("hyle_shared_pool_cached_create: invalid "
                                      "backing (a fn pointer is null)");
    return nullptr;
  }
  auto *p = new (std::nothrow) hyle_shared_pool_cached_o(
      backing, block_size, block_count, norm_align(alignment));
  if (p == nullptr) {
    hyle::platform::vm_set_last_error("hyle_shared_pool_cached_create: wrapper "
                                      "allocation failed (out of memory)");
    return nullptr;
  }
  if (!p->pool.valid()) {
    delete p;
    hyle::platform::vm_set_last_error("hyle_shared_pool_cached_create: slab "
                                      "allocation failed (out of memory)");
    return nullptr;
  }
  hyle::platform::vm_set_last_error(nullptr);
  return p;
}

void hyle_shared_pool_cached_destroy(hyle_shared_pool_cached_o *p) { delete p; }

void *hyle_shared_pool_cached_allocate(hyle_shared_pool_cached_o *p) {
  return p ? p->pool.allocate() : nullptr;
}

void hyle_shared_pool_cached_deallocate(hyle_shared_pool_cached_o *p,
                                        void *ptr) {
  if (p)
    p->pool.deallocate(ptr);
}

void hyle_shared_pool_cached_publish(hyle_shared_pool_cached_o *p) {
  if (p)
    p->pool.publish_thread_cache();
}

void hyle_shared_pool_cached_reset(hyle_shared_pool_cached_o *p) {
  if (p)
    p->pool.reset();
}

size_t hyle_shared_pool_cached_block_size(const hyle_shared_pool_cached_o *p) {
  return p ? p->pool.block_size() : 0;
}

uint32_t hyle_shared_pool_cached_capacity(const hyle_shared_pool_cached_o *p) {
  return p ? p->pool.capacity() : 0;
}

hyle_allocator
hyle_shared_pool_cached_as_allocator(hyle_shared_pool_cached_o *p) {
  if (!p)
    return hyle_allocator{nullptr, nullptr, nullptr, nullptr};
  return from_cpp(p->pool.as_allocator());
}

/* ---- StackAllocator --------------------------------------------------------
 */

hyle_stack_allocator_o *hyle_stack_allocator_create(hyle_allocator backing,
                                                    size_t size) {
  hyle::platform::vm_set_last_error(nullptr);
  // valid_backing guards a null fn pointer (a raw segfault in the trampoline,
  // no clean assert). size==0 is a caller-contract violation -> the C++ ctor's
  // HYLE_FATAL_ASSERT fires (fail-fast, same as a C++ caller); not masked here.
  if (!valid_backing(backing)) {
    hyle::platform::vm_set_last_error(
        "hyle_stack_allocator_create: invalid backing (a fn pointer is null)");
    return nullptr;
  }
  auto *s = new (std::nothrow) hyle_stack_allocator_o(backing, size);
  if (s == nullptr) {
    hyle::platform::vm_set_last_error("hyle_stack_allocator_create: wrapper "
                                      "allocation failed (out of memory)");
    return nullptr;
  }
  // slab-OOM in the C++ ctor -> invalid (graceful, not abort; followup #2 arch
  // B).
  if (!s->stack.valid()) {
    delete s;
    hyle::platform::vm_set_last_error(
        "hyle_stack_allocator_create: slab allocation failed (out of memory)");
    return nullptr;
  }
  hyle::platform::vm_set_last_error(nullptr);
  return s;
}

void hyle_stack_allocator_destroy(hyle_stack_allocator_o *s) { delete s; }

void *hyle_stack_allocator_allocate(hyle_stack_allocator_o *s, size_t bytes,
                                    size_t alignment) {
  return s ? s->stack.allocate(bytes, norm_align(alignment)) : nullptr;
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

/* ---- Arena -----------------------------------------------------------------
 */

hyle_arena_o *hyle_arena_create(hyle_shared_pool_o *pool) {
  hyle::platform::vm_set_last_error(nullptr);
  if (!pool) {
    hyle::platform::vm_set_last_error("hyle_arena_create: pool is NULL");
    return nullptr;
  }
  auto *a = new (std::nothrow) hyle_arena_o(&pool->pool);
  if (a == nullptr) { // wrapper OOM
    hyle::platform::vm_set_last_error(
        "hyle_arena_create: wrapper allocation failed (out of memory)");
    return nullptr;
  }
  // Pool exhausted (no free block) is a graceful resource failure, not a
  // contract violation: return NULL + set last-error so an FFI caller has a
  // defensive path (was a fatal abort in the C++ Arena ctor; see
  // mem-audit-followup #2).
  if (!a->arena.valid()) {
    delete a;
    hyle::platform::vm_set_last_error(
        "hyle_arena_create: pool exhausted (no free block)");
    return nullptr;
  }
  hyle::platform::vm_set_last_error(nullptr);
  return a;
}

void hyle_arena_destroy(hyle_arena_o *a) { delete a; }

void *hyle_arena_allocate(hyle_arena_o *a, size_t bytes, size_t alignment) {
  return a ? a->arena.allocate(bytes, norm_align(alignment)) : nullptr;
}

void hyle_arena_reset(hyle_arena_o *a) {
  if (a)
    a->arena.reset();
}

size_t hyle_arena_used(const hyle_arena_o *a) {
  return a ? a->arena.used() : 0;
}

size_t hyle_arena_remaining(const hyle_arena_o *a) {
  return a ? a->arena.remaining() : 0;
}

size_t hyle_arena_block_size(const hyle_arena_o *a) {
  return a ? a->arena.block_size() : 0;
}

hyle_allocator hyle_arena_as_allocator(hyle_arena_o *a) {
  if (!a)
    return hyle_allocator{nullptr, nullptr, nullptr, nullptr};
  return from_cpp(a->arena.as_allocator());
}

/* ---- GrowablePool ----------------------------------------------------------
 */

hyle_growable_pool_o *hyle_growable_pool_create(hyle_allocator backing,
                                                size_t slot_size,
                                                size_t slot_align,
                                                uint32_t initial_slots) {
  hyle::platform::vm_set_last_error(nullptr);
  if (!valid_backing(backing)) {
    hyle::platform::vm_set_last_error(
        "hyle_growable_pool_create: invalid backing (a fn pointer is null)");
    return nullptr;
  }
  const size_t normalized_align = norm_align(slot_align);
  if (!hyle::mem::detail::GrowablePoolCore::valid_layout(slot_size,
                                                         normalized_align)) {
    hyle::platform::vm_set_last_error(
        "hyle_growable_pool_create: invalid or overflowing slot layout");
    return nullptr;
  }
  auto *p = new (std::nothrow)
      hyle_growable_pool_o(backing, slot_size, normalized_align, initial_slots);
  if (p == nullptr)
    hyle::platform::vm_set_last_error(
        "hyle_growable_pool_create: wrapper allocation failed (out of memory)");
  else
    hyle::platform::vm_set_last_error(nullptr);
  return p;
}

void hyle_growable_pool_destroy(hyle_growable_pool_o *p) { delete p; }

void *hyle_growable_pool_acquire(hyle_growable_pool_o *p) {
  if (p == nullptr) {
    hyle::platform::vm_set_last_error(
        "hyle_growable_pool_acquire: pool is NULL");
    return nullptr;
  }
  void *slot = p->core.acquire();
  hyle::platform::vm_set_last_error(slot == nullptr
                                        ? "hyle_growable_pool_acquire: growth "
                                          "allocation failed (out of memory)"
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
  hyle::platform::vm_set_last_error(nullptr);
  if ((flags & ~kKnownFlags) != 0 ||
      ((flags & HYLE_HUGE_NUMA_BIND) != 0 &&
       (flags & HYLE_HUGE_NUMA_INTERLEAVE) != 0)) {
    hyle::platform::vm_set_last_error(
        "hyle_huge_region_open: invalid flag combination");
    return nullptr;
  }
  auto r =
      hyle::mem::HugeRegion::allocate(bytes, to_huge_flags(flags), numa_node);
  if (!r.valid())
    return nullptr;
  auto *out = new (std::nothrow) hyle_huge_region_o(std::move(r));
  if (out == nullptr) {
    hyle::platform::vm_set_last_error(
        "hyle_huge_region_open: wrapper allocation failed (out of memory)");
  } else {
    hyle::platform::vm_set_last_error(nullptr);
  }
  return out;
}

hyle_huge_region_o *hyle_huge_region_open_local(size_t bytes, uint32_t flags) {
  constexpr uint32_t kKnownFlags = HYLE_HUGE_THP_HINT | HYLE_HUGE_PREFAULT |
                                   HYLE_HUGE_NUMA_BIND |
                                   HYLE_HUGE_NUMA_INTERLEAVE;
  hyle::platform::vm_set_last_error(nullptr);
  if ((flags & ~kKnownFlags) != 0 ||
      ((flags & HYLE_HUGE_NUMA_BIND) != 0 &&
       (flags & HYLE_HUGE_NUMA_INTERLEAVE) != 0)) {
    hyle::platform::vm_set_last_error(
        "hyle_huge_region_open_local: invalid flag combination");
    return nullptr;
  }
  auto r = hyle::mem::HugeRegion::allocate_local(bytes, to_huge_flags(flags));
  if (!r.valid())
    return nullptr;
  auto *out = new (std::nothrow) hyle_huge_region_o(std::move(r));
  if (out == nullptr) {
    hyle::platform::vm_set_last_error("hyle_huge_region_open_local: wrapper "
                                      "allocation failed (out of memory)");
  } else {
    hyle::platform::vm_set_last_error(nullptr);
  }
  return out;
}

hyle_huge_region_o *hyle_huge_region_open_interleaved(size_t bytes) {
  hyle::platform::vm_set_last_error(nullptr);
  auto r = hyle::mem::HugeRegion::allocate_interleaved(bytes);
  if (!r.valid())
    return nullptr;
  auto *out = new (std::nothrow) hyle_huge_region_o(std::move(r));
  if (out == nullptr) {
    hyle::platform::vm_set_last_error(
        "hyle_huge_region_open_interleaved: wrapper allocation failed (out of "
        "memory)");
  } else {
    hyle::platform::vm_set_last_error(nullptr);
  }
  return out;
}

void hyle_huge_region_destroy(hyle_huge_region_o *r) { delete r; }

int hyle_huge_region_valid(const hyle_huge_region_o *r) {
  return r && r->region.valid() ? 1 : 0;
}

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
  return hyle::platform::vm_last_error();
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

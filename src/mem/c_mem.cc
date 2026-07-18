// C-ABI bridge for mem (include/hyle/c/mem.h).
//
// Two directions, both WITHOUT reinterpret_cast on the vtable:
//   C -> C++ : a caller's hyle_allocator is stored by value in the opaque wrapper;
//              the C++ object gets a mem::Allocator whose state points at that stored
//              struct and whose fn pointers are noexcept trampolines forwarding to it.
//              (The C fn pointers are plain `void*(*)(...)`; mem::Allocator wants
//              `noexcept` fn types, which a plain pointer is NOT convertible to -- so a
//              trampoline is required, not a cast.)
//   C++ -> C : filling a hyle_allocator from a mem::Allocator is a field-by-field copy;
//              a noexcept fn pointer converts implicitly to the plain C field type.

#include "hyle/c/mem.h"

#include <cstddef>
#include <new>
#include <utility>

#include "hyle/mem/allocator.h"
#include "hyle/mem/hugepage.h"
#include "hyle/mem/memory_arena.h"
#include "hyle/mem/stack_allocator.h"
#include "hyle/platform/virtual_memory.h"  // vm_last_error() for hyle_mem_last_error

namespace {

using hyle::mem::Allocator;

// ---- C hyle_allocator -> C++ mem::Allocator (via noexcept trampolines) -------
// `state` points at a live hyle_allocator (stored inside the owning wrapper, so it
// outlives every C++ object built from it).
void* c_backing_alloc(void* state, std::size_t size, std::size_t align) noexcept {
  auto* a = static_cast<const hyle_allocator*>(state);
  return a->alloc(a->state, size, align);
}
void c_backing_free(void* state, void* ptr, std::size_t size,
                    std::size_t align) noexcept {
  auto* a = static_cast<const hyle_allocator*>(state);
  a->free(a->state, ptr, size, align);
}
void* c_backing_realloc(void* state, void* ptr, std::size_t old_size,
                        std::size_t new_size, std::size_t align) noexcept {
  auto* a = static_cast<const hyle_allocator*>(state);
  return a->realloc(a->state, ptr, old_size, new_size, align);
}

// Build a C++ Allocator that forwards to the stored C handle at `*backing`.
Allocator to_cpp(const hyle_allocator* backing) noexcept {
  return Allocator(&c_backing_alloc, &c_backing_free, &c_backing_realloc,
                   const_cast<hyle_allocator*>(backing));
}

// ---- C++ mem::Allocator -> C hyle_allocator (field-by-field, no cast) --------
// AllocFn/FreeFn/ReallocFn are `noexcept` fn pointers; assigning them to the plain
// C field type is a standard implicit (noexcept -> non-noexcept) pointer conversion.
hyle_allocator from_cpp(const Allocator& a) noexcept {
  hyle_allocator out;
  out.alloc = a.alloc_fn();
  out.free = a.free_fn();
  out.realloc = a.realloc_fn();
  out.state = a.state();
  return out;
}

// A caller-supplied handle is only usable if all three fn pointers are present
// (system-boundary validation: an FFI caller can pass a zeroed struct).
bool valid_backing(const hyle_allocator& a) noexcept {
  return a.alloc != nullptr && a.free != nullptr && a.realloc != nullptr;
}

std::size_t norm_align(std::size_t a) noexcept {
  return a != 0 ? a : alignof(std::max_align_t);
}

hyle::mem::HugeFlags to_huge_flags(std::uint32_t f) noexcept {
  using hyle::mem::HugeFlags;
  auto out = HugeFlags::kNone;
  if (f & HYLE_HUGE_THP_HINT) out = out | HugeFlags::kThpHint;
  if (f & HYLE_HUGE_PREFAULT) out = out | HugeFlags::kPrefault;
  if (f & HYLE_HUGE_NUMA_BIND) out = out | HugeFlags::kNumaBind;
  if (f & HYLE_HUGE_NUMA_INTERLEAVE) out = out | HugeFlags::kNumaInterleave;
  return out;
}

}  // namespace

// ---- opaque wrappers --------------------------------------------------------
// Each stores the caller's hyle_allocator FIRST (so it is destroyed LAST) and the
// C++ object SECOND (destroyed first -- it may free through `backing` in its dtor).

struct hyle_memory_arena_o {
  hyle_allocator backing;
  hyle::mem::MemoryArena arena;
  hyle_memory_arena_o(hyle_allocator b, std::size_t block_size,
                      std::uint32_t block_count, std::size_t alignment)
      : backing(b),
        arena(to_cpp(&backing), block_size, block_count, alignment) {}
};

struct hyle_stack_allocator_o {
  hyle_allocator backing;
  hyle::mem::StackAllocator stack;
  hyle_stack_allocator_o(hyle_allocator b, std::size_t size)
      : backing(b), stack(to_cpp(&backing), size) {}
};

struct hyle_arena_o {
  hyle::mem::Arena arena;  // borrows a block from a pool (no owned backing)
  explicit hyle_arena_o(hyle::mem::MemoryArena* pool) : arena(pool) {}
};

struct hyle_huge_region_o {
  hyle::mem::HugeRegion region;
  explicit hyle_huge_region_o(hyle::mem::HugeRegion&& r)
      : region(std::move(r)) {}
};

extern "C" {

/* ---- allocator vtable ------------------------------------------------------ */

size_t hyle_default_alignment(void) { return alignof(std::max_align_t); }

hyle_allocator hyle_system_allocator(void) {
  return from_cpp(hyle::mem::system_allocator());
}

size_t hyle_system_usable_size(const void* ptr) {
  return hyle::mem::system_usable_size(ptr);
}

size_t hyle_system_good_size(size_t size) {
  return hyle::mem::system_good_size(size);
}

/* ---- MemoryArena ----------------------------------------------------------- */

hyle_memory_arena_o* hyle_memory_arena_create(hyle_allocator backing,
                                              size_t block_size,
                                              uint32_t block_count,
                                              size_t alignment) {
  if (!valid_backing(backing)) return nullptr;
  return new (std::nothrow)
      hyle_memory_arena_o(backing, block_size, block_count, norm_align(alignment));
}

void hyle_memory_arena_destroy(hyle_memory_arena_o* a) { delete a; }

void* hyle_memory_arena_allocate(hyle_memory_arena_o* a) {
  return a ? a->arena.allocate() : nullptr;
}

void hyle_memory_arena_deallocate(hyle_memory_arena_o* a, void* p) {
  if (a) a->arena.deallocate(p);
}

void hyle_memory_arena_reset(hyle_memory_arena_o* a) {
  if (a) a->arena.reset();
}

size_t hyle_memory_arena_block_size(const hyle_memory_arena_o* a) {
  return a ? a->arena.block_size() : 0;
}

uint32_t hyle_memory_arena_capacity(const hyle_memory_arena_o* a) {
  return a ? a->arena.capacity() : 0;
}

hyle_allocator hyle_memory_arena_as_allocator(hyle_memory_arena_o* a) {
  if (!a) return hyle_allocator{nullptr, nullptr, nullptr, nullptr};
  return from_cpp(a->arena.as_allocator());
}

/* ---- StackAllocator -------------------------------------------------------- */

hyle_stack_allocator_o* hyle_stack_allocator_create(hyle_allocator backing,
                                                    size_t size) {
  // valid_backing guards a null fn pointer (a raw segfault in the trampoline, no
  // clean assert). size==0 is a caller-contract violation -> the C++ ctor's
  // HYLE_FATAL_ASSERT fires (fail-fast, same as a C++ caller); not masked here.
  if (!valid_backing(backing)) return nullptr;
  return new (std::nothrow) hyle_stack_allocator_o(backing, size);
}

void hyle_stack_allocator_destroy(hyle_stack_allocator_o* s) { delete s; }

void* hyle_stack_allocator_allocate(hyle_stack_allocator_o* s, size_t bytes,
                                    size_t alignment) {
  return s ? s->stack.allocate(bytes, norm_align(alignment)) : nullptr;
}

void hyle_stack_allocator_reset(hyle_stack_allocator_o* s) {
  if (s) s->stack.reset();
}

void* hyle_stack_allocator_mark(const hyle_stack_allocator_o* s) {
  return s ? s->stack.mark() : nullptr;
}

void hyle_stack_allocator_rewind(hyle_stack_allocator_o* s, void* marker) {
  if (s) s->stack.rewind(static_cast<hyle::mem::StackAllocator::Marker>(marker));
}

size_t hyle_stack_allocator_used(const hyle_stack_allocator_o* s) {
  return s ? s->stack.used() : 0;
}

size_t hyle_stack_allocator_capacity(const hyle_stack_allocator_o* s) {
  return s ? s->stack.capacity() : 0;
}

size_t hyle_stack_allocator_remaining(const hyle_stack_allocator_o* s) {
  return s ? s->stack.remaining() : 0;
}

hyle_allocator hyle_stack_allocator_as_allocator(hyle_stack_allocator_o* s) {
  if (!s) return hyle_allocator{nullptr, nullptr, nullptr, nullptr};
  return from_cpp(s->stack.as_allocator());
}

/* ---- Arena ----------------------------------------------------------------- */

hyle_arena_o* hyle_arena_create(hyle_memory_arena_o* pool) {
  if (!pool) return nullptr;
  return new (std::nothrow) hyle_arena_o(&pool->arena);
}

void hyle_arena_destroy(hyle_arena_o* a) { delete a; }

void* hyle_arena_allocate(hyle_arena_o* a, size_t bytes, size_t alignment) {
  return a ? a->arena.allocate(bytes, norm_align(alignment)) : nullptr;
}

void hyle_arena_reset(hyle_arena_o* a) {
  if (a) a->arena.reset();
}

size_t hyle_arena_used(const hyle_arena_o* a) { return a ? a->arena.used() : 0; }

size_t hyle_arena_remaining(const hyle_arena_o* a) {
  return a ? a->arena.remaining() : 0;
}

size_t hyle_arena_block_size(const hyle_arena_o* a) {
  return a ? a->arena.block_size() : 0;
}

hyle_allocator hyle_arena_as_allocator(hyle_arena_o* a) {
  if (!a) return hyle_allocator{nullptr, nullptr, nullptr, nullptr};
  return from_cpp(a->arena.as_allocator());
}

/* ---- HugeRegion ------------------------------------------------------------ */

hyle_huge_region_o* hyle_huge_region_open(size_t bytes, uint32_t flags,
                                          int numa_node) {
  auto r = hyle::mem::HugeRegion::allocate(bytes, to_huge_flags(flags), numa_node);
  if (!r.valid()) return nullptr;
  return new (std::nothrow) hyle_huge_region_o(std::move(r));
}

hyle_huge_region_o* hyle_huge_region_open_local(size_t bytes, uint32_t flags) {
  auto r = hyle::mem::HugeRegion::allocate_local(bytes, to_huge_flags(flags));
  if (!r.valid()) return nullptr;
  return new (std::nothrow) hyle_huge_region_o(std::move(r));
}

hyle_huge_region_o* hyle_huge_region_open_interleaved(size_t bytes) {
  auto r = hyle::mem::HugeRegion::allocate_interleaved(bytes);
  if (!r.valid()) return nullptr;
  return new (std::nothrow) hyle_huge_region_o(std::move(r));
}

void hyle_huge_region_destroy(hyle_huge_region_o* r) { delete r; }

int hyle_huge_region_valid(const hyle_huge_region_o* r) {
  return r && r->region.valid() ? 1 : 0;
}

void* hyle_huge_region_data(hyle_huge_region_o* r) {
  return r ? r->region.data() : nullptr;
}

size_t hyle_huge_region_size(const hyle_huge_region_o* r) {
  return r ? r->region.size() : 0;
}

/* ---- diagnostics ----------------------------------------------------------- */

const char* hyle_mem_last_error(void) { return hyle::platform::vm_last_error(); }

/* ---- process-wide policy --------------------------------------------------- */

void hyle_enable_allocator_hugepages(int allow_thp, int allow_large_os_pages) {
  hyle::mem::enable_allocator_hugepages(
      {allow_thp != 0, allow_large_os_pages != 0});
}

void hyle_enable_allocator_numa(unsigned max_nodes) {
  hyle::mem::enable_allocator_numa(max_nodes);
}

int hyle_reserve_huge_1g_pages(size_t pages, int numa_node, unsigned timeout_ms) {
  return hyle::mem::reserve_huge_1g_pages(pages, numa_node, timeout_ms) ? 1 : 0;
}

}  // extern "C"

#include "hyle/mem/allocator.h"
#include "hyle/mem/hugepage.h"

#include "hyle/macros.h"

#include <mimalloc.h>  // PRIVATE include -- mi_* types never leak to a header

#if defined(_ADDRESS_SANITIZER) || defined(__SANITIZE_ADDRESS__)
#  define HYLE_MEM_ASAN 1
#elif defined(__has_feature)
#  if __has_feature(address_sanitizer)
#    define HYLE_MEM_ASAN 1
#  endif
#endif

namespace hyle::mem {
namespace {

// Round an alignment request up to something the backing accepts: a power of
// two, at least kDefaultAlignment. A non-power-of-two request is a caller bug.
std::size_t normalize_alignment(std::size_t a) noexcept {
  if (a < kDefaultAlignment) return kDefaultAlignment;
  HYLE_FATAL_ASSERT((a & (a - 1)) == 0);
  return a;
}

// mimalloc backing (mem-design decision: borrow a battle-tested general malloc;
// self-rolling one is wasted effort and necessarily worse). Wrapped behind the
// Allocator vtable so mi_* types never appear in any HYLE header -- swapping the
// backing later touches only this file (same discipline as the ggml bridge).
// mimalloc tracks each block's size, so realloc needs no manual copy/old_size.
//
// This is the ONLY backing: the former OS aligned-malloc fallback was removed on
// purpose (user directive). A disable/fallback path is a trap -- it lets a build
// silently sidestep a mimalloc-dependent problem and be mis-judged as fine. If
// something breaks with mimalloc, we face it, not hide behind glibc.
//
// THREE split entries (alloc/free/realloc) instead of one 5-arg realloc_func: the hot
// alloc path is 3 args (no Win64 stack spill) with NO dispatch branch -- the seam costs
// only the unavoidable indirect call now. mi_malloc already guarantees >= 16-byte
// (alignof(max_align_t)) alignment, so the common default-aligned case takes the plain
// (faster) path; only over-aligned (>16, e.g. AVX512 64B) needs mi_*_aligned.
void* system_alloc(void* /*state*/, std::size_t size, std::size_t alignment) noexcept {
  if (size == 0) return nullptr;
  // Common-case fast path: default (or weaker) alignment needs no normalization at all --
  // mi_malloc already guarantees >= 16-byte. Skip normalize_alignment (its assert + bit
  // test) for the hot path; only over-aligned requests pay it.
  if (alignment <= kDefaultAlignment) return mi_malloc(size);
  return mi_malloc_aligned(size, normalize_alignment(alignment));
}

void system_free(void* /*state*/, void* ptr, std::size_t /*size*/,
                 std::size_t /*alignment*/) noexcept {
  if (ptr != nullptr) mi_free(ptr);  // mi_free recovers size/alignment from the block
}

void* system_realloc(void* /*state*/, void* ptr, std::size_t /*old_size*/,
                     std::size_t new_size, std::size_t alignment) noexcept {
  if (new_size == 0) {  // realloc-to-0 == free
    if (ptr != nullptr) mi_free(ptr);
    return nullptr;
  }
  std::size_t align = normalize_alignment(alignment);
  // mi_realloc_aligned preserves alignment and returns nullptr on OOM with the
  // original block left intact (C realloc contract) -- matches our vtable.
  if (align <= kDefaultAlignment) {
    if (ptr == nullptr) return mi_malloc(new_size);
    return mi_realloc(ptr, new_size);
  }
  if (ptr == nullptr) return mi_malloc_aligned(new_size, align);
  return mi_realloc_aligned(ptr, new_size, align);
}

}  // namespace

Allocator system_allocator() noexcept {
  return Allocator(&system_alloc, &system_free, &system_realloc, nullptr);
}

std::size_t system_usable_size(const void* ptr) noexcept {
  return ptr != nullptr ? mi_usable_size(ptr) : 0;
}

std::size_t system_good_size(std::size_t size) noexcept {
#if defined(HYLE_MEM_ASAN)
  // ASan deliberately reports the requested payload size from mi_usable_size
  // (redzones are not caller-usable), whereas mi_good_size still exposes
  // mimalloc's uninstrumented size class. Keep the public preview==actual
  // contract under instrumentation by previewing the usable payload.
  return size;
#else
  return size != 0 ? mi_good_size(size) : 0;
#endif
}

void enable_allocator_hugepages(AllocatorHugepagePolicy policy) noexcept {
  // Programmatic stand-in for MIMALLOC_ALLOW_THP / MIMALLOC_ALLOW_LARGE_OS_PAGES.
  // Call early; mimalloc reads options on first OS alloc. Safe if called later
  // (affects subsequent segments).
  if (policy.allow_thp) {
    mi_option_enable(mi_option_allow_thp);
  } else {
    mi_option_disable(mi_option_allow_thp);
  }
  if (policy.allow_large_os_pages) {
    mi_option_enable(mi_option_allow_large_os_pages);
  } else {
    mi_option_disable(mi_option_allow_large_os_pages);
  }
}

void enable_allocator_numa(unsigned max_nodes) noexcept {
  // 0 = use all NUMA nodes (mimalloc default meaning).
  mi_option_set(mi_option_use_numa_nodes, static_cast<long>(max_nodes));
}

bool reserve_huge_1g_pages(std::size_t pages, int numa_node,
                           unsigned timeout_ms) noexcept {
  if (pages == 0) return true;
  // mimalloc: 0 success, non-zero error. timeout 0 → use a short default so a
  // fragmented box does not hang the process for minutes.
  const std::size_t t =
      timeout_ms == 0 ? static_cast<std::size_t>(100) : timeout_ms;
  int rc;
  if (numa_node == platform::kNumaAny) {
    // Interleave 1GiB pages across nodes (0 = all nodes in mi API).
    rc = mi_reserve_huge_os_pages_interleave(pages, /*numa_nodes=*/0, t);
  } else {
    rc = mi_reserve_huge_os_pages_at(pages, numa_node, t);
  }
  return rc == 0;
}

}  // namespace hyle::mem

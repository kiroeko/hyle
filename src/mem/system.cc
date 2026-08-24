#include "hyle/mem/allocator.h"
#include "hyle/mem/hugepage.h"
#include "hyle/mem/pmr.h"

#include "system_internal.h"

#include <mimalloc.h> // PRIVATE include -- mi_* types never leak to a header

#include "hyle/mem/system_inline.h"

#include "hyle/macros.h"
#include "hyle/obs/trace.h"

#include <cerrno>
#include <cstdio>
#include <limits>
#include <memory_resource>
#include <new>

namespace hyle::mem {

static_assert(MI_MALLOC_VERSION == 30500,
              "HYLE's reviewed mimalloc contract requires v3.5.0");

namespace {

// Calls originating inside hyle_mem must not bounce through the exported
// shared-consumer seam.  On ELF that symbol is preemptible and therefore uses
// the PLT even from this DSO, adding a second library boundary to runtime and
// PMR adapters.  External SystemAllocator callers still use detail::* below;
// both routes reach these same private backend helpers.
void *system_backend_allocate_local(std::size_t size,
                                    std::size_t alignment) noexcept {
  alignment = detail::system_backend_checked_alignment(alignment);
  if (alignment <= detail::kMimallocV3NaturalAlignment) {
    return mi_malloc(size < detail::kMimallocV3NaturalAlignment
                         ? detail::kMimallocV3NaturalAlignment
                         : size);
  }
  return mi_malloc_aligned(size, alignment);
}

void system_backend_deallocate_local(void *ptr) noexcept { mi_free(ptr); }

void *system_backend_reallocate_local(void *ptr, std::size_t new_size,
                                      std::size_t alignment) noexcept {
  alignment = detail::system_backend_checked_alignment(alignment);
  if (alignment <= detail::kMimallocV3NaturalAlignment) {
    const std::size_t backing_size =
        new_size < detail::kMimallocV3NaturalAlignment
            ? detail::kMimallocV3NaturalAlignment
            : new_size;
    return mi_realloc(ptr, backing_size);
  }
  return mi_realloc_aligned(ptr, new_size, alignment);
}

} // namespace

#if !defined(HYLE_STATIC)
namespace detail {

void *system_backend_allocate(std::size_t size,
                              std::size_t alignment) noexcept {
  return system_backend_allocate_local(size, alignment);
}

void system_backend_deallocate(void *ptr) noexcept {
  system_backend_deallocate_local(ptr);
}

void *system_backend_reallocate(void *ptr, std::size_t new_size,
                                std::size_t alignment) noexcept {
  return system_backend_reallocate_local(ptr, new_size, alignment);
}

} // namespace detail
#endif

namespace {

// mimalloc backing (mem-design decision: borrow a battle-tested general malloc;
// self-rolling one is wasted effort and necessarily worse). Wrapped behind the
// Allocator vtable so mi_* types never appear in any HYLE header -- swapping
// the backing later touches only this file (same discipline as the ggml
// bridge). mimalloc tracks each block's size, so realloc needs no manual
// copy/old_size.
//
// This is the ONLY backing: the former OS aligned-malloc fallback was removed
// on purpose (user directive). A disable/fallback path is a trap -- it lets a
// build silently sidestep a mimalloc-dependent problem and be mis-judged as
// fine. If something breaks with mimalloc, we face it, not hide behind glibc.
//
// THREE split entries (alloc/free/realloc) instead of one 5-arg realloc_func:
// the hot alloc path is 3 args (no Win64 stack spill) with NO dispatch branch
// -- the seam costs only the unavoidable indirect call now. HYLE guarantees
// at least alignof(max_align_t), including for weaker explicit requests.
// mimalloc v3 naturally aligns blocks at that boundary once their size reaches
// it, so sub-default sizes are padded while the common path stays on plain
// mi_malloc/realloc. Only over-alignment uses mi_*_aligned.
void *system_alloc(void * /*state*/, std::size_t size,
                   std::size_t alignment) noexcept {
  if (size == 0)
    return nullptr;
  void *pointer = system_backend_allocate_local(size, alignment);
  if (pointer != nullptr) {
    HYLE_TRACE_ALLOC_NAMED(pointer, size, "hyle.system");
  }
  return pointer;
}

void system_free(void * /*state*/, void *ptr, std::size_t /*size*/,
                 std::size_t /*alignment*/) noexcept {
  if (ptr != nullptr) {
    HYLE_TRACE_FREE_NAMED(ptr, "hyle.system");
    system_backend_deallocate_local(ptr);
  }
}

void *system_realloc(void * /*state*/, void *ptr, std::size_t /*old_size*/,
                     std::size_t new_size, std::size_t alignment) noexcept {
  if (new_size == 0) { // realloc-to-0 == free
    if (ptr != nullptr) {
      HYLE_TRACE_FREE_NAMED(ptr, "hyle.system");
      system_backend_deallocate_local(ptr);
    }
    return nullptr;
  }
  // The backend selector preserves the normalized alignment and leaves the
  // original block intact on OOM (C realloc contract). Plain mi_realloc handles
  // the <=16B-alignment fast path; only over-alignment uses the aligned entry.
  // Both forms accept nullptr as allocate, so no separate pointer branch is
  // needed.
  void *result =
      system_backend_reallocate_local(ptr, new_size, alignment);
  if (result != nullptr) {
    if (ptr != nullptr)
      HYLE_TRACE_FREE_NAMED(ptr, "hyle.system");
    HYLE_TRACE_ALLOC_NAMED(result, new_size, "hyle.system");
  }
  return result;
}

class SystemMemoryResource final : public std::pmr::memory_resource {
  void *do_allocate(std::size_t bytes, std::size_t alignment) override {
    const std::size_t backing_size = detail::pmr_backing_size(bytes);
    void *pointer = system_backend_allocate_local(backing_size, alignment);
    if (pointer == nullptr)
      throw std::bad_alloc{};
    HYLE_TRACE_ALLOC_NAMED(pointer, backing_size, "hyle.system");
    return pointer;
  }

  void do_deallocate(void *pointer, std::size_t /*bytes*/,
                     std::size_t /*alignment*/) override {
    // memory_resource::deallocate requires a pointer/alignment pair previously
    // returned by this resource.  Rechecking those preconditions here duplicates
    // the PMR caller's contract on every normal container deallocation.
    HYLE_TRACE_FREE_NAMED(pointer, "hyle.system");
    system_backend_deallocate_local(pointer);
  }

  bool do_is_equal(
      const std::pmr::memory_resource &other) const noexcept override {
    // There is one process-wide instance.  Identity is therefore both the
    // cheapest and the complete deallocation-compatibility test.
    return this == &other;
  }
};

} // namespace

Allocator system_allocator() noexcept {
  return Allocator(&system_alloc, &system_free, &system_realloc, nullptr);
}

std::pmr::memory_resource *system_memory_resource() noexcept {
  static SystemMemoryResource resource;
  return &resource;
}

std::size_t system_usable_size(const void *ptr) noexcept {
  return ptr != nullptr ? mi_usable_size(ptr) : 0;
}

std::size_t system_good_size(std::size_t size) noexcept {
  return size != 0 ? mi_good_size(size) : 0;
}

void system_collect(bool force) noexcept { mi_collect(force); }

#if defined(HYLE_TESTING)
int system_backend_version_for_testing() noexcept { return mi_version(); }
#endif

void enable_allocator_hugepages(AllocatorHugepagePolicy policy) noexcept {
  // Programmatic stand-in for MIMALLOC_ALLOW_THP /
  // MIMALLOC_ALLOW_LARGE_OS_PAGES. Call early; mimalloc reads options on first
  // OS alloc. Safe if called later (affects subsequent segments).
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

Status reserve_huge_1g_pages(std::size_t pages, int numa_node,
                             unsigned timeout_ms) noexcept {
  if (pages == 0)
    return Status::Ok();
  constexpr std::size_t kHuge1GiB = std::size_t{1} << 30;
  if (pages > (std::numeric_limits<std::size_t>::max)() / kHuge1GiB) {
    return Status::OutOfRange(
        "reserve_huge_1g_pages: requested byte size overflows size_t");
  }
  if (numa_node != platform::kNumaAny &&
      (numa_node < 0 ||
       !platform::numa_node_available(static_cast<unsigned>(numa_node)))) {
    return Status::OutOfRange(
        "reserve_huge_1g_pages: NUMA node ID is unavailable");
  }
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
  if (rc == 0)
    return Status::Ok();

  char context[160];
  std::snprintf(context, sizeof(context),
                "mimalloc reserve huge 1GiB pages (pages=%zu, numa_node=%d)",
                pages, numa_node);
  // The pinned mimalloc implementation returns ENOMEM for privilege,
  // fragmentation and arena-management failures. It does not expose enough
  // detail to claim PermissionDenied or DeadlineExceeded, so preserve its
  // native errno and use the error module's O(1) canonical taxonomy. A future
  // backend/native code outside that documented contract remains Unknown.
  return status_from_errno(rc, context);
}

} // namespace hyle::mem

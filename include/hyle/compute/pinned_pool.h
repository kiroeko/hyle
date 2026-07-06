#pragma once

#include <cstddef>

#include "hyle/compute/buffer.h"
#include "hyle/compute/device.h"
#include "hyle/error/result.h"
#include "hyle/macros.h"
#include "hyle/mem/allocator.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_COMPUTE_API
#elif defined(HYLE_COMPUTE_BUILD)
#  define HYLE_COMPUTE_API HYLE_EXPORT
#else
#  define HYLE_COMPUTE_API HYLE_IMPORT
#endif

// compute-02b: pinned host staging built by REUSING mem/, not reinventing it.
//
// THE INSIGHT (user, 2026-07-06): prepare upload data CONTIGUOUSLY in host memory
// (a bump arena), then submit the whole contiguous run to the device in ONE copy --
// the arena's contiguity IS the batching ("合批 for perf"). And the pinned block
// itself is pooled (预分配大块换性能), amortizing cudaMallocHost (a heavy
// page-locking call) + reusing blocks. ggml does NOT pool pinned host memory
// (cudaMallocHost every time, ggml-cuda.cu:1503) -- this is where HYLE beats it.
//
// KEY: mem/ ALREADY provides the pool + the bump arena. compute only needs to hand
// mem a PINNED backing allocator; then a pinned block pool is just
//   mem::MemoryArena(pinned_allocator(dev), block_size, block_count)   // pooled 64MB blocks
// and a contiguous staging region is just
//   mem::Arena staging(&pool);  staging.allocate(...);  // bump, contiguous
// then ONE batched submit:
//   dev_buffer.set(dst_off, staging_base, staging.used());
// So compute-02b is ~one adapter (pinned_allocator) + a thin PinnedStaging helper --
// NOT a reimplemented pool. This is the correct "结合 mem 加速" (pinned = host RAM,
// fits mem; device VRAM does NOT go through mem -- see mem-design "device 内存").

namespace hyle::compute {

// A mem::Allocator whose backing is PINNED host memory (cudaMallocHost/cudaFreeHost
// on the CUDA backend; plain malloc on the host fallback). Feed it to a
// mem::MemoryArena to get a pooled fixed-block pinned staging pool, or use it
// anywhere a mem::Allocator is taken. `alignment` requests are honored by the
// backend (pinned host is max_align-capable). The Device must outlive any pool
// built on the returned allocator (the allocator captures the backend + ordinal).
//
// This is the ONE piece compute must provide; everything else (pooling, bump,
// block reuse) is mem/ reused verbatim.
HYLE_COMPUTE_API mem::Allocator pinned_allocator(const Device& device) noexcept;

// Thin convenience over "borrow a pinned block + bump-prepare contiguous data +
// submit it in one copy". Holds a mem::Arena over one pooled pinned block and
// remembers the block base so a whole contiguous run commits in a single H2D set().
// For callers who don't want to wire mem::Arena + Buffer::set by hand; the manual
// path (pinned_allocator + mem::MemoryArena + mem::Arena + Buffer::set) is equally
// valid and gives full control.
//
// Not thread-safe (a bump arena is single-owner -- one PinnedStaging per fiber/
// thread, mirroring mem::Arena's per-core-scratch model). Move-only.
class PinnedStaging {
 public:
  PinnedStaging() noexcept = default;
  HYLE_DISABLE_COPY(PinnedStaging);
  HYLE_COMPUTE_API PinnedStaging(PinnedStaging&& o) noexcept;
  HYLE_COMPUTE_API PinnedStaging& operator=(PinnedStaging&& o) noexcept;
  HYLE_COMPUTE_API ~PinnedStaging();

  // Allocate a standalone pinned staging region of `bytes` (its own pinned block,
  // not from a shared pool -- for a one-off; use a mem::MemoryArena + acquire()
  // pattern for a reused pool). bump from it with allocate(), then commit_to().
  HYLE_COMPUTE_API static Result<PinnedStaging> create(const Device& device,
                                                       std::size_t bytes) noexcept;

  bool valid() const noexcept { return base_ != nullptr; }

  // Bump `bytes` (aligned) out of the staging region; nullptr when full. The
  // returned host pointer is PINNED and writable -- fill it with upload data. Bumps
  // are contiguous, which is what lets commit_to() batch them into one copy.
  HYLE_COMPUTE_API void* allocate(std::size_t bytes,
                                  std::size_t alignment = 16) noexcept;

  // Bytes bump-allocated so far (the length of the contiguous prepared run).
  std::size_t used() const noexcept { return used_; }
  // Base of the contiguous prepared run (pinned host pointer).
  void* base() const noexcept { return base_; }

  // Submit the whole contiguous prepared run [base, base+used) to `dst` at
  // `dst_offset` in ONE Buffer::set (one H2D copy = the batched submit). Returns the
  // copy Status. Does NOT reset -- call reset() to reuse the region for the next batch.
  HYLE_COMPUTE_API Status commit_to(Buffer& dst, std::size_t dst_offset) noexcept;

  // Rewind the bump cursor to reuse the region for the next batch (no free).
  void reset() noexcept { used_ = 0; }

 private:
  const ComputeBackend* vt_ = nullptr;  // backend owning the pinned block (for free)
  int ordinal_ = 0;
  void* base_ = nullptr;   // pinned block base (cudaMallocHost)
  std::size_t cap_ = 0;    // block capacity
  std::size_t used_ = 0;   // bump cursor
};

}  // namespace hyle::compute

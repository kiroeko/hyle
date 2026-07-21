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
// THE INSIGHT (user, 2026-07-06): prepare upload data CONTIGUOUSLY in page-locked
// host memory (a bump arena), then submit the whole contiguous run to the device in
// ONE H2D copy -- the arena's contiguity IS the batching ("合批 for perf"). pinned
// (cudaMallocHost) host RAM makes that copy a fast DMA and lets it overlap async.
//
// WHAT THIS FILE SHIPS: two pieces, both single-block.
//   1. pinned_allocator(device) -- a mem::Allocator whose backing is cudaMallocHost/
//      cudaFreeHost. The one adapter compute must provide.
//   2. PinnedStaging -- a bump arena over ONE pinned block + a batched commit_to().
//      Single-owner (one per fiber/thread), not a pool.
//
// NOT SHIPPED (available composition, no consumer yet): a MULTI-block pinned pool.
// Because pinned_allocator is just a mem::Allocator, you *can* feed it to a
// mem::SharedPool(pinned_allocator(dev), block_size, block_count) to amortize the
// heavy cudaMallocHost across a fixed set of reused blocks (ggml re-calls
// cudaMallocHost every time, ggml-cuda.cu:1503 -- so this would beat it). That is a
// real, correct use of mem/, but it is a composition the caller wires when a pooled
// staging ring actually lands (the event-driven ring is L1 / post-compute-03). The
// shipped path here is the single-block PinnedStaging; no production code pools
// pinned blocks today.

namespace hyle::compute {

// A mem::Allocator whose backing is PINNED host memory (cudaMallocHost/cudaFreeHost
// on the CUDA backend; plain malloc on the host fallback). Use it directly, or feed
// it to a mem::SharedPool for a fixed-block pinned pool if/when a pooled staging ring
// is built. `alignment` requests are honored by the backend (pinned host is
// max_align-capable). The Device must outlive any pool built on the returned
// allocator (the allocator captures the backend + ordinal).
HYLE_COMPUTE_API mem::Allocator pinned_allocator(const Device& device) noexcept;

// Thin convenience over "one pinned block + bump-prepare contiguous data + submit it
// in one copy". Owns a single pinned block (via pinned_allocator's backend) and bumps
// from it, so a whole contiguous run commits in a single H2D set(). For callers who
// don't want to wire the bump + Buffer::set by hand.
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
  // not from a shared pool). For a reusable multi-block ring, compose
  // mem::SharedPool with pinned_allocator(). Bump here, then commit_to().
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

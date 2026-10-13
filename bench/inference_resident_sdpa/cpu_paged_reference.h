#pragma once
#include "hyle/compute/provider.h"
#include "hyle/compute/paged_arena.h"
#include "hyle/mem/buffer.h"
#include <memory>

namespace hyle::bench::resident {
// Immutable original ready pages remain owned throughout the benchmark. The
// peer's alternative contiguous cache is constructed cold and billed explicitly.
struct CpuPagedReferenceData final {
  compute::BatchLayoutVersion layout;
  compute::PagedArenaRead keys, values;
  mem::Buffer key_cache, value_cache;
};
Result<std::shared_ptr<const CpuPagedReferenceData>>
cpu_paged_reference_data(compute::BatchLayoutVersion,
                         compute::PagedArenaRead, compute::PagedArenaRead) noexcept;
compute::Provider cpu_paged_reference_provider(
    const CpuPagedReferenceData *) noexcept;
}

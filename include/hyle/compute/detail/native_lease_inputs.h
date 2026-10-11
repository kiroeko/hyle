#pragma once

#include <algorithm>
#include <vector>

#include "hyle/compute/resource_leases.h"
#include "hyle/mem/std_allocator.h"

namespace hyle::compute::detail {
// Common bounded permission metadata only. Native routes own their own
// invocation, view construction and submit; no queue/execution interface here.
class NativeLeaseInputs final {
public:
  explicit NativeLeaseInputs(mem::AllocatorRef allocator) : reads_(mem::StdAllocator<ReadLease>(allocator)) {}
  Status prepare(std::size_t count, LeaseImportOptions options) noexcept {
    if (count >= UINT32_MAX) return Status::ResourceExhausted("native lease input capacity");
    HYLE_ASSIGN_OR_RETURN(uses_, LeaseUsePool::create({.max_batches = 1,
        .max_resources_per_batch = static_cast<std::uint32_t>((std::max)(count, std::size_t{1})),
        .allocator = options.allocator, .resource_budget = options.resource_budget}));
    try { reads_.resize(count); }
    catch (...) { return Status::ResourceExhausted("native lease input metadata"); }
    return Status::Ok();
  }
  std::size_t size() const noexcept { return reads_.size(); }
  bool active() const noexcept { return uses_.active_batches() != 0; }
  bool closed() const noexcept { return closed_; }
  void retain(std::size_t index, const ReadLease &source) noexcept { reads_[index] = source; }
  Result<LeaseUseBatch> admit() noexcept {
    HYLE_ASSIGN_OR_RETURN(auto batch, uses_.begin());
    for (const auto &source : reads_) HYLE_RETURN_IF_ERROR(batch.read(source));
    return batch;
  }
  Status close() noexcept { closed_ = true; return uses_.close(); }
  Status drain() noexcept { closed_ = true; return uses_.drain(); }
private:
  std::vector<ReadLease, mem::StdAllocator<ReadLease>> reads_;
  LeaseUsePool uses_;
  bool closed_ = false;
};
} // namespace hyle::compute::detail

#pragma once
#include "hyle/compute/detail/paged_append_backend.h"
#include "hyle/compute/resource_leases.h"

namespace hyle::compute::detail {
class PagedAppendProgram final {
public:
  PagedAppendProgram() noexcept=default;
  PagedAppendProgram(const PagedAppendProgram&)=delete;
  PagedAppendProgram&operator=(const PagedAppendProgram&)=delete;
  HYLE_COMPUTE_BATCH_API PagedAppendProgram(PagedAppendProgram&&) noexcept;
  HYLE_COMPUTE_BATCH_API PagedAppendProgram&operator=(PagedAppendProgram&&) noexcept;
  HYLE_COMPUTE_BATCH_API ~PagedAppendProgram();
  HYLE_COMPUTE_BATCH_API static Result<PagedAppendProgram> prepare(
      BackendExtensionRef,PagedAppendNativeSpec,mem::AllocatorRef) noexcept;
  bool valid()const noexcept{return program_!=nullptr;}
  HYLE_COMPUTE_BATCH_API EnqueueResult enqueue(QueueSequence&,PagedAppendNativeBindings)const noexcept;
private:
  void reset()noexcept;
  BackendExtensionRef extension_;
  const PagedAppendBackendVTable*table_=nullptr;
  void*program_=nullptr;
  mem::AllocatorRef allocator_{};
};

// One fixed carrier shared only within the original retired Graph Invocation.
// Upload uses an actual F1 episode/Completion; Graph separately admits ready()
// as a read control through every actual captured execution.
class PagedAppendMirror final {
public:
  PagedAppendMirror()noexcept=default;
  PagedAppendMirror(const PagedAppendMirror&)=delete;
  PagedAppendMirror&operator=(const PagedAppendMirror&)=delete;
  PagedAppendMirror(PagedAppendMirror&&)noexcept=default;
  PagedAppendMirror&operator=(PagedAppendMirror&&)noexcept=default;
  HYLE_COMPUTE_BATCH_API static Result<PagedAppendMirror> create(
      mem::MemoryResourceRef,mem::MemoryResourceRef host,std::uint32_t capacity,
      mem::AllocatorRef,ResourceBudget)noexcept;
  HYLE_COMPUTE_BATCH_API Status upload(const QueueRef&,std::span<const PagedAppendInstruction>)noexcept;
  ReadLease ready()const noexcept{return ready_;}
private:
  LeasePool device_pool_,host_pool_;
  LeaseUsePool uses_;
  WriteLease writer_,staging_;
  ReadLease ready_,source_;
  Completion pending_;
  std::uint32_t capacity_=0;
  bool failed_=false;
};
} // namespace hyle::compute::detail

#pragma once
#include "hyle/compute/batch_ops.h"
#include "hyle/compute/batch_transfer.h"
#include "hyle/compute/provider.h"
#include <optional>

namespace hyle::compute {
struct BatchProgramOptions final {
  mem::MemoryResourceRef resource;
  mem::MemoryResourceRef host_workspace = mem::system_resource();
  std::size_t max_float_workspace_bytes = 64u*1024u*1024u;
  std::size_t max_int_workspace_bytes = 8u*1024u*1024u;
  std::uint32_t max_invocations = 2;
  float scale = 1.0F;
  bool causal = true;
  mem::AllocatorRef allocator = mem::system_allocator();
  ResourceBudget resource_budget{};
};
struct BatchPlanReport final {
  std::uint64_t layout_generation = 0;
  std::size_t metadata_staging_bytes = 0;
  std::size_t float_workspace_bytes = 0, int_workspace_bytes = 0;
  std::uint32_t cta_tile_q = 0, padded_batch_size = 0;
  bool split_kv = false, native_metadata_unknown = true, capture_compatible = false;
  bool reused_native_plan = false;
  std::uint32_t host_waits = 0;
  CopyBatchReport metadata_upload{};
};
namespace detail { struct BatchProgramData; struct BatchInvocationData; struct BatchGraphAccess; }
class BatchInvocation;
class PreparedBatch;
class PagedArenaRead;
// Cold, immutable native domain. Only explicit batch users acquire buffers,
// slots or planning; ordinary operation and Graph v19 contracts are unchanged.
class BatchProgram final {
public:
  BatchProgram() noexcept = default;
  HYLE_COMPUTE_BATCH_API BatchProgram(const BatchProgram &) noexcept;
  HYLE_COMPUTE_BATCH_API BatchProgram &operator=(const BatchProgram &) noexcept;
  HYLE_COMPUTE_BATCH_API BatchProgram(BatchProgram &&) noexcept;
  HYLE_COMPUTE_BATCH_API BatchProgram &operator=(BatchProgram &&) noexcept;
  HYLE_COMPUTE_BATCH_API ~BatchProgram();
  HYLE_COMPUTE_BATCH_API static Result<BatchProgram> prepare(const Context &,BatchSchema,BatchProgramOptions) noexcept;
  bool valid() const noexcept{return data_!=nullptr;}
  HYLE_COMPUTE_BATCH_API BatchSchema schema() const noexcept;
private:
  friend class BatchInvocation;
  friend struct detail::BatchGraphAccess;
  static Result<BatchProgram> prepare(const Context&,BatchSchema,BatchProgramOptions,ProviderInstanceId) noexcept;
  explicit BatchProgram(detail::BatchProgramData *p) noexcept:data_(p){}
  detail::BatchProgramData *data_=nullptr;
};
// Thread-compatible control, one accepted episode per slot. Queue/resource/
// injected allocator remain borrowed through all controls/readers/native uses.
// Destruction has no hidden wait. Keep a token only for this layout episode;
// retained old output readers prevent output reuse on the next prepare.
class BatchInvocation final {
public:
  BatchInvocation() noexcept=default;
  BatchInvocation(const BatchInvocation&)=delete;
  BatchInvocation&operator=(const BatchInvocation&)=delete;
  HYLE_COMPUTE_BATCH_API BatchInvocation(BatchInvocation&&) noexcept;
  HYLE_COMPUTE_BATCH_API BatchInvocation&operator=(BatchInvocation&&) noexcept;
  HYLE_COMPUTE_BATCH_API ~BatchInvocation();
  HYLE_COMPUTE_BATCH_API static Result<BatchInvocation> create(const BatchProgram &,Queue &) noexcept;
  bool valid() const noexcept{return data_!=nullptr;}
  // Explicit host waits for original F2 metadata upload and native planning.
  // Failed accepted prefixes invalidate the old plan and retain their uses.
  HYLE_COMPUTE_BATCH_API Result<PreparedBatch> prepare_batch(const BatchLayoutVersion &) noexcept;
  // Explicit update: reuse only when the provider proves its complete native
  // topology unchanged; otherwise prepare the new plan here, never in run().
  HYLE_COMPUTE_BATCH_API Result<PreparedBatch> update_batch(const BatchLayoutVersion &) noexcept;
  HYLE_COMPUTE_BATCH_API Result<ReadLease> output() noexcept;
  HYLE_COMPUTE_BATCH_API BatchPlanReport report() const noexcept;
  HYLE_COMPUTE_BATCH_API bool in_flight() const noexcept;
  HYLE_COMPUTE_BATCH_API Status request_cancel() noexcept;
  HYLE_COMPUTE_BATCH_API Status close() noexcept;
  HYLE_COMPUTE_BATCH_API Status drain() noexcept;
private:
  friend class PreparedBatch;
  friend struct detail::BatchGraphAccess;
  static Result<BatchInvocation> create(const BatchProgram&,Queue&,bool external_output) noexcept;
  explicit BatchInvocation(detail::BatchInvocationData *p) noexcept:data_(p){}
  detail::BatchInvocationData *data_=nullptr;
  Result<PreparedBatch> prepare(const BatchLayoutVersion &,bool update) noexcept;
};
class PreparedBatch final {
public:
  PreparedBatch() noexcept=default;
  PreparedBatch(const PreparedBatch&)=delete;
  PreparedBatch&operator=(const PreparedBatch&)=delete;
  HYLE_COMPUTE_BATCH_API PreparedBatch(PreparedBatch&&) noexcept;
  HYLE_COMPUTE_BATCH_API PreparedBatch&operator=(PreparedBatch&&) noexcept;
  HYLE_COMPUTE_BATCH_API ~PreparedBatch();
  bool valid() const noexcept{return data_!=nullptr;}
  HYLE_COMPUTE_BATCH_API std::uint64_t generation() const noexcept;
  // Q/K/V exact capacity descriptions; actual rows are from this layout.
  // Runs this prepared plan once. No plan/JIT/metadata reread occurs here.
  HYLE_COMPUTE_BATCH_API Result<Completion> run(std::span<const ReadLease> inputs) noexcept;
  // Explicit paged arena footprint: Q is an ordinary lease, K/V retain only
  // this prepared layout's actual physical pages through native retirement.
  HYLE_COMPUTE_BATCH_API Result<Completion> run(const ReadLease&query,
      const PagedArenaRead&keys,const PagedArenaRead&values) noexcept;
private:
  friend class BatchInvocation;
  friend struct detail::BatchGraphAccess;
  PreparedBatch(detail::BatchInvocationData *p,std::uint64_t epoch) noexcept:data_(p),epoch_(epoch){}
  detail::BatchInvocationData *data_=nullptr;
  std::uint64_t epoch_=0;
};
} // namespace hyle::compute

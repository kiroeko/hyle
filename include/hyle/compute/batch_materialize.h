#pragma once

#include <array>
#include "hyle/compute/batch_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/paged_arena.h"

namespace hyle::compute {
struct MaterializedBatchOptions final {
  std::uint32_t max_kv_per_request = 0;
  std::size_t max_host_metadata_bytes = 0;
  std::size_t max_logical_intermediate_bytes = 0;
  float scale = 1.0F;
  bool causal = true;
  mem::AllocatorRef allocator = mem::system_allocator();
  ResourceBudget resource_budget{};
};
struct MaterializedBatchCosts final {
  std::size_t index_bytes = 0, mask_bytes = 0;
  std::size_t gathered_kv_bytes = 0;
  // Sum of every generated logical view, including aliases and copies. This
  // conservative admission bound is not native resident memory or scratch.
  std::size_t logical_intermediate_bytes = 0;
  bool native_private_bytes_unknown = true;
};
struct MaterializedBatchStep final {
  Operation operation;
  std::array<std::uint16_t,2> inputs{};
  std::uint16_t output = 0;
  std::uint8_t input_count = 0;
};
namespace detail {struct MaterializedBatchProgramData;struct MaterializedBatchVersionData;}
class MaterializedBatchVersion;
// Explicit reference algorithm, never an automatic native-paged fallback.
// Cold immutable typed recipe; native preparation still qualifies every step.
// Values 0..4 are packed Q, K, V, I32 indices and F32 mask. All math runs on
// the selected original route. No payload materialization occurs on the host.
class MaterializedBatchProgram final {
public:
  MaterializedBatchProgram() noexcept = default;
  HYLE_COMPUTE_BATCH_API MaterializedBatchProgram(const MaterializedBatchProgram&) noexcept;
  HYLE_COMPUTE_BATCH_API MaterializedBatchProgram&operator=(const MaterializedBatchProgram&) noexcept;
  HYLE_COMPUTE_BATCH_API MaterializedBatchProgram(MaterializedBatchProgram&&) noexcept;
  HYLE_COMPUTE_BATCH_API MaterializedBatchProgram&operator=(MaterializedBatchProgram&&) noexcept;
  HYLE_COMPUTE_BATCH_API ~MaterializedBatchProgram();
  HYLE_COMPUTE_BATCH_API static Result<MaterializedBatchProgram> prepare(BatchSchema,MaterializedBatchOptions) noexcept;
  bool valid()const noexcept{return data_!=nullptr;}
  HYLE_COMPUTE_BATCH_API BatchSchema schema()const noexcept;
  HYLE_COMPUTE_BATCH_API std::span<const TensorDesc> descriptions()const noexcept;
  HYLE_COMPUTE_BATCH_API std::span<const MaterializedBatchStep> steps()const noexcept;
  HYLE_COMPUTE_BATCH_API std::uint16_t output_value()const noexcept;
  HYLE_COMPUTE_BATCH_API MaterializedBatchCosts costs()const noexcept;
  // Owned readonly metadata for one same-schema layout. Thread-safe cold
  // creation if the injected allocator is; spans are never retained.
  HYLE_COMPUTE_BATCH_API Result<MaterializedBatchVersion> version(BatchLayoutVersion)const noexcept;
private:
  friend class MaterializedBatchVersion;
  explicit MaterializedBatchProgram(detail::MaterializedBatchProgramData*p)noexcept:data_(p){}
  detail::MaterializedBatchProgramData*data_=nullptr;
};
// Shares the program, exact layout and host metadata backing. Borrowed allocator
// and resource budget outlive versions and exported leases/native uses. F2
// upload is explicit; route controllers keep these inputs until actual retirement.
class MaterializedBatchVersion final {
public:
  MaterializedBatchVersion() noexcept = default;
  HYLE_COMPUTE_BATCH_API MaterializedBatchVersion(const MaterializedBatchVersion&) noexcept;
  HYLE_COMPUTE_BATCH_API MaterializedBatchVersion&operator=(const MaterializedBatchVersion&) noexcept;
  HYLE_COMPUTE_BATCH_API MaterializedBatchVersion(MaterializedBatchVersion&&) noexcept;
  HYLE_COMPUTE_BATCH_API MaterializedBatchVersion&operator=(MaterializedBatchVersion&&) noexcept;
  HYLE_COMPUTE_BATCH_API ~MaterializedBatchVersion();
  bool valid()const noexcept{return data_!=nullptr;}
  HYLE_COMPUTE_BATCH_API BatchLayoutVersion layout()const noexcept;
  HYLE_COMPUTE_BATCH_API MaterializedBatchProgram program()const noexcept;
  HYLE_COMPUTE_BATCH_API Result<ReadLease> indices()const noexcept;
  HYLE_COMPUTE_BATCH_API Result<ReadLease> mask()const noexcept;
private:
  friend class MaterializedBatchProgram;
  explicit MaterializedBatchVersion(detail::MaterializedBatchVersionData*p)noexcept:data_(p){}
  detail::MaterializedBatchVersionData*data_=nullptr;
};
struct MaterializedBatchInvocationOptions final {
  std::size_t max_payload_bytes=0,max_workspace_bytes=0;
  mem::AllocatorRef allocator=mem::system_allocator();
  ResourceBudget resource_budget{};
};
struct MaterializedBatchRunReport final {
  std::uint64_t layout_generation=0;
  std::uint32_t submitted_stages=0,host_waits=0;
};
namespace detail {struct MaterializedBatchInvocationData;}
// Explicit host/system-memory reference over original prepared Direct math.
// Cold fixed buffers/operations; submit waits each accepted stage and reports
// that cost. Native GPU routes retain their own typed resources/controllers.
// Control is thread-compatible; Context/Queue and injected allocators outlive
// all native work/results. One successful output is readonly and may escape.
class MaterializedBatchInvocation final {
public:
  MaterializedBatchInvocation()noexcept=default;
  MaterializedBatchInvocation(const MaterializedBatchInvocation&)=delete;
  MaterializedBatchInvocation&operator=(const MaterializedBatchInvocation&)=delete;
  HYLE_COMPUTE_BATCH_API MaterializedBatchInvocation(MaterializedBatchInvocation&&)noexcept;
  HYLE_COMPUTE_BATCH_API MaterializedBatchInvocation&operator=(MaterializedBatchInvocation&&)noexcept;
  HYLE_COMPUTE_BATCH_API ~MaterializedBatchInvocation();
  HYLE_COMPUTE_BATCH_API static Result<MaterializedBatchInvocation> create(const Context&,
      MaterializedBatchProgram,Queue&,MaterializedBatchInvocationOptions)noexcept;
  bool valid()const noexcept{return data_!=nullptr;}
  HYLE_COMPUTE_BATCH_API Status prepare_batch(BatchLayoutVersion)noexcept;
  HYLE_COMPUTE_BATCH_API Status bind(ReadLease,PagedArenaRead,PagedArenaRead)noexcept;
  HYLE_COMPUTE_BATCH_API Status bind(ReadLease,ReadLease,ReadLease)noexcept;
  // Synchronous stage submission/waits. Returned success is already terminal;
  // cancelling it cannot preempt earlier stages. Unknown/failure never publishes
  // an output or reuses its slot; drain waits original actual native retirement.
  HYLE_COMPUTE_BATCH_API Result<Completion> submit()noexcept;
  HYLE_COMPUTE_BATCH_API Result<ReadLease> output()const noexcept;
  HYLE_COMPUTE_BATCH_API MaterializedBatchRunReport report()const noexcept;
  HYLE_COMPUTE_BATCH_API Status close()noexcept;
  HYLE_COMPUTE_BATCH_API Status drain()noexcept;
private:
  explicit MaterializedBatchInvocation(detail::MaterializedBatchInvocationData*p)noexcept:data_(p){}
  detail::MaterializedBatchInvocationData*data_=nullptr;
};
} // namespace hyle::compute

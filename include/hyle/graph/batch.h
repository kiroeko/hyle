#pragma once
#include "hyle/graph/leased_invocation.h"
#include "hyle/compute/batch_execution.h"
#include "hyle/compute/batch_materialize.h"
#include "hyle/compute/batch_paged_native.h"
#include "hyle/compute/paged_arena.h"
#include <array>
#include <cstdint>
#include <string_view>

namespace hyle::graph::batch {
inline constexpr compute::ProviderId kProviderId{0x48594c4547524241ULL,0x5443485245470001ULL};
// Original native BatchPlanning capability + original Graph region v19. The
// wrapper owns configuration; resources/allocator retain their borrow contracts.
HYLE_GRAPH_BATCH_API Result<compute::Provider> provider(compute::Provider base,
    compute::BatchSchema,compute::BatchProgramOptions) noexcept;
struct NodeLayout final {NodeId node;compute::BatchLayoutVersion layout;};
// Native page math schema for a coarse joint graph. The three inputs are the
// truthful packed-Q/physical-K/physical-V descriptions of the common recipe;
// the output has exactly active_query_rows. Owned recipe/layout and page-table
// uploads belong to the explicit cold NativePagedNodeBinding, not raw ports.
// This schema alone does not qualify a native route or authorize arena reads.
struct NativePagedAttention final {
  compute::BatchCapacity capacity{};
  std::uint32_t active_query_rows = 0;
  float scale = 1.0F;
  bool causal = true;
  std::int32_t window_left = -1;
  static constexpr compute::OperationId operation_id{
      0x48594c4547524241ULL, 0x4e41544956455001ULL};
  static constexpr std::string_view operation_name = "native-paged-graph-attention";
  constexpr compute::OperationTraits operation_traits() const noexcept {
    return {{3,3},1,compute::AliasPolicy::kNone,compute::EffectKind::kPure,
            compute::PrecisionPolicy::kExact,compute::DataType::kFloat32};
  }
  HYLE_GRAPH_BATCH_API Status infer_output_descriptions(
      std::span<const compute::TensorDesc>,std::span<compute::TensorDesc>) const noexcept;
};
static_assert(compute::detail::InlineCustomOperationDescriptor<NativePagedAttention>);
HYLE_GRAPH_BATCH_API Result<NativePagedAttention> native_paged_attention(
    const compute::NativePagedBatchProgram&,compute::BatchLayoutVersion) noexcept;
// Borrowed only by compile; the immutable joint plan copies these owners.
// Allocator/budget and native resources retain their original lifetime rules.
// Another active-Q output shape requires the original exact variant/compile.
struct NativePagedNodeBinding final {
  NodeId node;
  compute::NativePagedBatchProgram program;
  compute::BatchLayoutVersion layout;
  compute::NativePagedInvocationOptions options;
};
struct MaterializedGraph final {
  Graph graph;
  std::array<ValueId,5> inputs{};
  ValueId output;
};
// Explicitly selected reference recipe over the original ordinary Graph.
// Native route compile/prepare still qualifies each operation and its cost.
// Bind owned same-version indices/mask; upload them with original F2 first.
HYLE_GRAPH_BATCH_API Result<MaterializedGraph> materialized_graph(
    const compute::MaterializedBatchProgram&,GraphBuilderOptions = {}) noexcept;
namespace detail {struct InvocationData;struct MaterializedInvocationData;struct PendingInvocationData;}
struct MaterializedInvocationOptions final {
  // One cold output backing, including alignment padding; explicit byte cap.
  std::size_t max_output_bytes=0;
  CompileOptions compile{};
  LeasedInvocationOptions leases{};
};
// Host/system-memory reference over the original shared-Queue Graph. Owns the
// fixed recipe/compiled graph, exact metadata generation and one output slot.
// Native GPU reference uses its own typed resources and preparation contract.
// Control is thread-compatible. Borrowed Context/Queue and injected allocators
// outlive native work/results; readonly metadata and output leases retain their
// original owners. Independent controllers have separate compiled slots.
class MaterializedInvocation final {
public:
  MaterializedInvocation()noexcept=default;
  MaterializedInvocation(const MaterializedInvocation&)=delete;
  MaterializedInvocation&operator=(const MaterializedInvocation&)=delete;
  HYLE_GRAPH_BATCH_API MaterializedInvocation(MaterializedInvocation&&)noexcept;
  HYLE_GRAPH_BATCH_API MaterializedInvocation&operator=(MaterializedInvocation&&)noexcept;
  HYLE_GRAPH_BATCH_API ~MaterializedInvocation();
  HYLE_GRAPH_BATCH_API static Result<MaterializedInvocation> create(
      const compute::Context&,compute::MaterializedBatchProgram,compute::Queue&,
      MaterializedInvocationOptions)noexcept;
  bool valid()const noexcept{return data_!=nullptr;}
  // Explicit update boundary. Old results/completions/storage/native reads
  // prevent output reuse. Repeated prepare before execution is supported.
  HYLE_GRAPH_BATCH_API Status prepare_batch(compute::BatchLayoutVersion)noexcept;
  HYLE_GRAPH_BATCH_API Status bind(compute::ReadLease query,compute::PagedArenaRead keys,
      compute::PagedArenaRead values)noexcept;
  HYLE_GRAPH_BATCH_API Status bind(compute::ReadLease query,compute::ReadLease keys,
      compute::ReadLease values)noexcept;
  // Consumes this prepared episode; each subsequent run needs prepare/bind.
  HYLE_GRAPH_BATCH_API Result<compute::Completion> submit()noexcept;
  HYLE_GRAPH_BATCH_API Result<LeasedGraphOutput> output()noexcept;
  HYLE_GRAPH_BATCH_API compute::MaterializedBatchVersion version()const noexcept;
  HYLE_GRAPH_BATCH_API Status close()noexcept;
  HYLE_GRAPH_BATCH_API Status drain()noexcept;
private:
  explicit MaterializedInvocation(detail::MaterializedInvocationData*p)noexcept:data_(p){}
  detail::MaterializedInvocationData*data_=nullptr;
};
// Thread-compatible control over the original leased Graph invocation.
// Borrowed Context/resources/Queue/allocator outlive native work and results.
// Independent invocations have separate plans/workspaces and may run separately.
struct LaneBinding final {
  ExecutionLaneId lane{};
  const compute::Context* context=nullptr;
  compute::Queue* queue=nullptr;
  // Required for a native batch lane. It must be a host carrier admitted by
  // this Context's original transfer/planner, rather than another lane's pool.
  mem::MemoryResourceRef host_workspace{};
};
class Invocation final {
public:
  Invocation() noexcept=default;
  Invocation(const Invocation&)=delete;
  Invocation&operator=(const Invocation&)=delete;
  HYLE_GRAPH_BATCH_API Invocation(Invocation&&) noexcept;
  HYLE_GRAPH_BATCH_API Invocation&operator=(Invocation&&) noexcept;
  HYLE_GRAPH_BATCH_API ~Invocation();
  // Exactly the active shared lanes of the compiled plan. The array and
  // Context pointers are borrowed only during create(); Queue bindings are
  // copied, and each batch slot plans and runs on its own lane's Queue.
  // Contexts/Queues/resources retain the original native lifetime contract.
  HYLE_GRAPH_BATCH_API static Result<Invocation> create(const Executable&,
      std::span<const LaneBinding>,
      LeasedInvocationOptions = {},ReplayOptions = {}) noexcept;
  bool valid() const noexcept{return data_!=nullptr;}
  // Complete layout set for selected batch nodes. Resets old bindings only
  // after original Graph's idle/result/completion checks. Bind Q/K/V and output
  // afterwards; metadata is bound from the same original region's mirror.
  // Nodes sharing one metadata input must use the identical layout version.
  // All regions still explicitly upload/plan and report their own cost. A
  // preparation error clears every token, so submit cannot mix old/new plans.
  HYLE_GRAPH_BATCH_API Status prepare_batch(std::span<const NodeLayout>,bool update=true) noexcept;
  HYLE_GRAPH_BATCH_API Status bind(ValueId,compute::TensorDesc,compute::ReadLease) noexcept;
  // Exact same-layout paged K/V permissions. Only qualified batch readers;
  // capacity cannot be published or consumed by ordinary dense operations.
  HYLE_GRAPH_BATCH_API Status bind(ValueId,compute::PagedArenaRead) noexcept;
  HYLE_GRAPH_BATCH_API Status bind(ValueId,compute::TensorDesc,compute::WriteLease&&) noexcept;
  // Explicit cold capture of selected native batch regions using current bound
  // storage. Outer Graph/CPU stages keep their original command replay route.
  // After native-plan/storage changes call again; submit never records capture.
  HYLE_GRAPH_BATCH_API Status prepare_replay() noexcept;
  HYLE_GRAPH_BATCH_API Result<compute::Completion> submit() noexcept;
  HYLE_GRAPH_BATCH_API Result<LeasedGraphOutput> output(ValueId) noexcept;
  HYLE_GRAPH_BATCH_API Result<compute::BatchPlanReport> report(NodeId) const noexcept;
  HYLE_GRAPH_BATCH_API Status close() noexcept;
  HYLE_GRAPH_BATCH_API Status drain() noexcept;
private:
  friend class PendingInvocation;
  explicit Invocation(detail::InvocationData*p)noexcept:data_(p){}
  detail::InvocationData*data_=nullptr;
};
// Explicit cold composition owner over the same original Graph invocation,
// region slots and public completion. Pending page capacity is accepted only
// by its exact prepared paged readers. It never grants a public ready view.
// A single original Update may supply several inputs/nodes/lanes; unsupported
// asynchronous CPU ordering rejects without introducing a hidden host wait.
class PendingInvocation final {
public:
  PendingInvocation()noexcept=default;
  PendingInvocation(const PendingInvocation&)=delete;
  PendingInvocation&operator=(const PendingInvocation&)=delete;
  HYLE_GRAPH_BATCH_API PendingInvocation(PendingInvocation&&)noexcept;
  HYLE_GRAPH_BATCH_API PendingInvocation&operator=(PendingInvocation&&)noexcept;
  HYLE_GRAPH_BATCH_API ~PendingInvocation();
  // Failure preserves owner. Allocator/budget and original Context/Queue/
  // resources remain alive through outcomes and actual native retirement.
  HYLE_GRAPH_BATCH_API static Result<PendingInvocation> create(Invocation&& owner,
      std::size_t max_source_completions,LeasedInvocationOptions options={})noexcept;
  bool valid()const noexcept{return data_!=nullptr;}
  HYLE_GRAPH_BATCH_API Status prepare_batch(std::span<const NodeLayout>,bool update=true)noexcept;
  HYLE_GRAPH_BATCH_API Status bind(ValueId,compute::TensorDesc,compute::ReadLease)noexcept;
  HYLE_GRAPH_BATCH_API Status bind(ValueId,compute::PagedArenaRead)noexcept;
  HYLE_GRAPH_BATCH_API Status bind(ValueId,compute::PagedArenaPendingRead)noexcept;
  HYLE_GRAPH_BATCH_API Status bind(ValueId,compute::TensorDesc,compute::WriteLease&&)noexcept;
  // Before acceptance rejection preserves Update. After any accepted prefix,
  // return its actual aggregate, including errors; do not drop native owners.
  HYLE_GRAPH_BATCH_API Result<compute::Completion> submit(compute::CopyBatchUpdate&&)noexcept;
  HYLE_GRAPH_BATCH_API Result<LeasedGraphOutput> output(ValueId)noexcept;
  // Observe/release old completion/output first. Returning controls does not
  // convert a failed update into successful mutable/read permission.
  HYLE_GRAPH_BATCH_API Result<compute::CopyBatchUpdate> take_update()noexcept;
  HYLE_GRAPH_BATCH_API Result<compute::BatchPlanReport> report(NodeId)const noexcept;
  HYLE_GRAPH_BATCH_API Status close()noexcept;
  HYLE_GRAPH_BATCH_API Status drain()noexcept;
private:
  explicit PendingInvocation(detail::PendingInvocationData*p)noexcept:data_(p){}
  detail::PendingInvocationData*data_=nullptr;
};
} // namespace hyle::graph::batch

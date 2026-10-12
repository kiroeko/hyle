#pragma once
#include "hyle/graph/leased_invocation.h"
#include "hyle/compute/batch_execution.h"
#include "hyle/compute/batch_materialize.h"

#if defined(HYLE_STATIC)
#define HYLE_GRAPH_BATCH_API
#elif defined(HYLE_GRAPH_BATCH_BUILD)
#define HYLE_GRAPH_BATCH_API HYLE_EXPORT
#else
#define HYLE_GRAPH_BATCH_API HYLE_IMPORT
#endif

namespace hyle::graph::batch {
inline constexpr compute::ProviderId kProviderId{0x48594c4547524241ULL,0x5443485245470001ULL};
// Original native BatchPlanning capability + original Graph region v19. The
// wrapper owns configuration; resources/allocator retain their borrow contracts.
HYLE_GRAPH_BATCH_API Result<compute::Provider> provider(compute::Provider base,
    compute::BatchSchema,compute::BatchProgramOptions) noexcept;
struct NodeLayout final {NodeId node;compute::BatchLayoutVersion layout;};
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
namespace detail {struct InvocationData;}
// Thread-compatible control over the original leased Graph invocation.
// Borrowed Context/resources/Queue/allocator outlive native work and results.
// Independent invocations have separate plans/workspaces and may run separately.
class Invocation final {
public:
  Invocation() noexcept=default;
  Invocation(const Invocation&)=delete;
  Invocation&operator=(const Invocation&)=delete;
  HYLE_GRAPH_BATCH_API Invocation(Invocation&&) noexcept;
  HYLE_GRAPH_BATCH_API Invocation&operator=(Invocation&&) noexcept;
  HYLE_GRAPH_BATCH_API ~Invocation();
  HYLE_GRAPH_BATCH_API static Result<Invocation> create(const Executable&,
      const compute::Context&,compute::Queue&,mem::MemoryResourceRef host_workspace,LeasedInvocationOptions = {}) noexcept;
  bool valid() const noexcept{return data_!=nullptr;}
  // Complete layout set for selected batch nodes. Resets old bindings only
  // after original Graph's idle/result/completion checks. Bind Q/K/V and output
  // afterwards; metadata is bound from the same original region's mirror.
  // Nodes sharing one metadata input must use the identical layout version.
  // All regions still explicitly upload/plan and report their own cost. A
  // preparation error clears every token, so submit cannot mix old/new plans.
  HYLE_GRAPH_BATCH_API Status prepare_batch(std::span<const NodeLayout>,bool update=true) noexcept;
  HYLE_GRAPH_BATCH_API Status bind(ValueId,compute::TensorDesc,compute::ReadLease) noexcept;
  HYLE_GRAPH_BATCH_API Status bind(ValueId,compute::TensorDesc,compute::WriteLease&&) noexcept;
  HYLE_GRAPH_BATCH_API Result<compute::Completion> submit() noexcept;
  HYLE_GRAPH_BATCH_API Result<LeasedGraphOutput> output(ValueId) noexcept;
  HYLE_GRAPH_BATCH_API Result<compute::BatchPlanReport> report(NodeId) const noexcept;
  HYLE_GRAPH_BATCH_API Status close() noexcept;
  HYLE_GRAPH_BATCH_API Status drain() noexcept;
private:
  explicit Invocation(detail::InvocationData*p)noexcept:data_(p){}
  detail::InvocationData*data_=nullptr;
};
} // namespace hyle::graph::batch

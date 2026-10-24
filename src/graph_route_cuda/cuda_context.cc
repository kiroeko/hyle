#include "hyle/graph/cuda.h"

#include <array>

#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/providers/ggml_cuda.h"
#include "default_context.h"
#include "hyle/graph/batch.h"
#include "dense_bfloat_scores_region.h"
#include "ct_packed_linear_region.h"
#include "fp8_shared_prologue_region.h"

namespace hyle::graph::cuda {

Result<compute::Context>
create_context(compute::cuda::DefaultContextOptions options) noexcept {
  return compute::cuda::internal::create_context(options, ggml_cuda::provider(),
      compute::cuda::internal::DefaultScope::kGraph,
      internal::dense_bfloat_scores_provider(), internal::ct_packed_linear_provider(),
      internal::fp8_shared_prologue_provider());
}

Result<CompiledGraph> compile(const Graph &graph,
                              const compute::Context &context,
                              mem::MemoryResourceRef device_resource,
                              CompileOptions options) noexcept {
  if (context.device().backend_id() != compute::cuda::kBackendId)
    return std::unexpected(Status::InvalidArgument(
        "CUDA graph compilation requires a CUDA Context"));
  if (!device_resource.valid())
    return std::unexpected(Status::InvalidArgument(
        "CUDA graph compilation requires an explicit memory resource"));
  const std::array storage{StorageDomain{.resource = device_resource}};
  HYLE_ASSIGN_OR_RETURN(auto source, ExecutionSource::acquire(context));
  const std::array targets{ExecutionTarget{
      .source = std::move(source), .default_storage = StorageDomainId{0}}};
  const std::array lanes{ExecutionLane{.target = ExecutionTargetId{0}}};
  return cuda::compile(
      graph, {.storage_domains = storage, .targets = targets, .lanes = lanes},
      options);
}

Result<CompiledGraph> compile(const Graph &graph, CompileEnvironment environment,
                              CompileOptions options) noexcept {
  for (const auto &target : environment.targets)
    if (!target.source ||
        target.source.backend_id() != compute::cuda::kBackendId)
      return std::unexpected(Status::InvalidArgument(
          "CUDA graph compilation requires CUDA execution sources"));
  return batch::compile(graph, environment, std::move(options));
}

Result<CompiledGraph> compile(const Graph &graph,
                              const compute::Context &context,
                              const compute::StorageOwner &owner,
                              CompileOptions options) noexcept {
  if (!owner)
    return std::unexpected(Status::InvalidArgument("CUDA graph storage owner is empty"));
  HYLE_ASSIGN_OR_RETURN(auto source, ExecutionSource::acquire(context));
  const std::array storage{StorageDomain{.resource = owner.ref(), .owner = owner}};
  const std::array targets{ExecutionTarget{
      .source = std::move(source), .default_storage = StorageDomainId{0}}};
  return cuda::compile(graph,
      {.storage_domains = storage, .targets = targets}, std::move(options));
}

Status prepare_replay(Invocation &invocation, compute::Queue &queue,
                      ReplayOptions options) noexcept {
  if (queue.device().backend_id() != compute::cuda::kBackendId)
    return Status::InvalidArgument("CUDA graph replay requires a CUDA Queue");
  const std::array queues{
      QueueBinding{.lane = ExecutionLaneId{0}, .queue = &queue}};
  return invocation.prepare_replay(queues, options);
}

} // namespace hyle::graph::cuda

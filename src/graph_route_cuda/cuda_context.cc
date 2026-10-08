#include "hyle/graph/cuda.h"

#include <array>

#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/providers/ggml_cuda.h"
#include "default_context.h"

namespace hyle::graph::cuda {

Result<compute::Context>
create_context(compute::cuda::DefaultContextOptions options) noexcept {
  return compute::cuda::internal::create_context(options, ggml_cuda::provider(),
      compute::cuda::internal::DefaultScope::kGraph);
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
  const std::array targets{ExecutionTarget{
      .context = &context, .default_storage = StorageDomainId{0}}};
  const std::array lanes{ExecutionLane{.target = ExecutionTargetId{0}}};
  return graph::compile(
      graph, {.storage_domains = storage, .targets = targets, .lanes = lanes},
      options);
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

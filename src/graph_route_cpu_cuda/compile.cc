#include "hyle/graph/cpu_cuda.h"

#include <array>

#include "../compute_backend_cuda/resource_admission.h"
#include "../graph/compile_internal.h"
#include "hyle/compute/backends/cpu.h"

namespace hyle::graph::cpu_cuda {

Result<CompiledGraph> compile(const Graph &graph, Environment environment,
                              CompileOptions options) noexcept {
  if (environment.cpu == nullptr || environment.cuda == nullptr ||
      environment.host == nullptr || environment.device == nullptr) {
    return std::unexpected(Status::InvalidArgument(
        "CPU+CUDA graph requires two Contexts and host/device resources"));
  }
  if (environment.cpu->device().backend_id() != compute::cpu::kBackendId ||
      environment.cuda->device().backend_id() != compute::cuda::kBackendId) {
    return std::unexpected(Status::InvalidArgument(
        "CPU+CUDA graph endpoints have different backend identities"));
  }
  HYLE_RETURN_IF_ERROR(compute::cuda::detail::validate_composition_resources(
      *environment.cuda, *environment.host, *environment.device));
  const std::array storage{
      StorageDomain{.resource = environment.device->ref(),
                    .workspace_budget_bytes =
                        environment.device_workspace_budget_bytes},
      StorageDomain{.resource = environment.host->ref(),
                    .workspace_budget_bytes =
                        environment.host_workspace_budget_bytes}};
  const std::array targets{
      ExecutionTarget{.context = environment.cuda,
                      .default_storage = kDeviceStorage},
      ExecutionTarget{.context = environment.cpu,
                      .default_storage = kHostStorage}};
  const std::array lanes{ExecutionLane{.target = kCudaTarget},
                        ExecutionLane{.target = kCpuTarget}};
  return graph::detail::compile_cpu_gpu(
      graph, {.storage_domains = storage, .targets = targets, .lanes = lanes},
      options, compute::cpu::kBackendId, compute::cuda::kBackendId);
}

} // namespace hyle::graph::cpu_cuda

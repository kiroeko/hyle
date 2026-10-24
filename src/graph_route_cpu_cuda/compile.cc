#include "hyle/graph/cpu_cuda.h"

#include <array>

#include "../compute_backend_cuda/resource_admission.h"
#include "hyle/graph/batch.h"
#include "hyle/compute/backends/cpu.h"

namespace hyle::graph::cpu_cuda {

Result<CompiledGraph> compile(const Graph &graph, Environment environment,
                              CompileOptions options) noexcept {
  if (environment.cpu == nullptr || environment.cuda == nullptr ||
      !environment.host.valid() || environment.staging == nullptr ||
      environment.device == nullptr) {
    return std::unexpected(Status::InvalidArgument(
        "CPU+CUDA graph requires two Contexts and host/staging/device resources"));
  }
  if (environment.cpu->device().backend_id() != compute::cpu::kBackendId ||
      environment.cuda->device().backend_id() != compute::cuda::kBackendId) {
    return std::unexpected(Status::InvalidArgument(
        "CPU+CUDA graph endpoints have different backend identities"));
  }
  HYLE_RETURN_IF_ERROR(compute::cuda::detail::validate_composition_resources(
      *environment.cuda, *environment.staging, *environment.device));
  if (!environment.host.properties().has(mem::MemoryProperty::kHostAccessible))
    return std::unexpected(Status::InvalidArgument(
        "CPU+CUDA host storage must be host accessible"));
  const std::array storage{
      StorageDomain{.resource = environment.device->ref(),
                    .workspace_budget_bytes =
                        environment.device_workspace_budget_bytes},
      StorageDomain{.resource = environment.host,
                    .workspace_budget_bytes =
                        environment.host_workspace_budget_bytes},
      StorageDomain{.resource = environment.staging->ref(),
                    .workspace_budget_bytes =
                        environment.staging_workspace_budget_bytes}};
  HYLE_ASSIGN_OR_RETURN(auto cuda_source,
                        ExecutionSource::acquire(*environment.cuda));
  HYLE_ASSIGN_OR_RETURN(auto cpu_source,
                        ExecutionSource::acquire(*environment.cpu));
  const std::array targets{
      ExecutionTarget{.source = std::move(cuda_source),
                      .default_storage = kDeviceStorage},
      ExecutionTarget{.source = std::move(cpu_source),
                      .default_storage = kHostStorage}};
  const std::array lanes{ExecutionLane{.target = kCudaTarget},
                        ExecutionLane{.target = kCpuTarget}};
  if (options.resource_budget) {
    for (const auto &domain : storage) {
      auto usage = options.resource_budget.usage(domain.resource);
      if (!usage) return std::unexpected(std::move(usage).error());
    }
  }
  return graph::batch::compile(
      graph, {.storage_domains = storage, .targets = targets, .lanes = lanes},
      options);
}

} // namespace hyle::graph::cpu_cuda

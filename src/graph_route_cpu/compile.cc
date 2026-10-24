#include "hyle/graph/cpu.h"

#include "hyle/graph/batch.h"
#include <array>
#include <utility>

namespace hyle::graph::cpu {

Result<CompiledGraph> compile(const Graph &graph,
                              const compute::Context &context,
                              CompileOptions options) noexcept {
  const std::array storage{StorageDomain{.resource = mem::system_resource()}};
  HYLE_ASSIGN_OR_RETURN(auto source, ExecutionSource::acquire(context));
  const std::array targets{ExecutionTarget{
      .source = std::move(source), .default_storage = StorageDomainId{0}}};
  const std::array lanes{ExecutionLane{.target = ExecutionTargetId{0}}};
  return cpu::compile(graph, {.storage_domains = storage, .targets = targets,
                              .lanes = lanes}, std::move(options));
}

Result<CompiledGraph> compile(const Graph &graph, CompileEnvironment environment,
                              CompileOptions options) noexcept {
  for (const auto &target : environment.targets)
    if (!target.source ||
        target.source.backend_id() != compute::cpu::kBackendId)
      return std::unexpected(Status::InvalidArgument(
          "CPU graph compilation requires CPU execution sources"));
  return batch::compile(graph, environment, std::move(options));
}

} // namespace hyle::graph::cpu

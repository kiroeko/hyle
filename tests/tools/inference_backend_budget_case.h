#pragma once

#include "../../engine/inference/backend.h"
#include "../../engine/inference/execution_resources.h"
#include "../../engine/inference/readback.h"

#include <array>
#include <cmath>
#include <string_view>

namespace hyle::inference::test {

struct BackendBudgetFacts final {
  std::size_t host_submissions = 0, cuda_submissions = 0, compared_values = 0;
  std::size_t missing_resource_rejections = 0, retired_scopes = 0;
  std::size_t cuda_workspace_bytes = 0;
};

// Both real endpoints exist, but each graph uses exactly one. An omitted
// idle resource is different from omitting its actual private workspace.
// Provider-private unknown bounds remain explicitly unattributed throughout.
inline Result<BackendBudgetFacts> backend_budget_case() {
  namespace c = compute;
  namespace g = graph;
  HYLE_ASSIGN_OR_RETURN(auto backend,
      Backend::create({.primary = BackendKind::kCuda, .cpu_assist = true}));
  HYLE_ASSIGN_OR_RETURN(auto description, c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{17}));
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto input, builder.input(description));
  HYLE_ASSIGN_OR_RETURN(auto absolute, builder.add(c::Unary{c::UnaryKind::kAbs}, input));
  HYLE_ASSIGN_OR_RETURN(auto output, builder.add(c::Unary{c::UnaryKind::kNeg}, absolute));
  HYLE_RETURN_IF_ERROR(builder.publish(output));
  HYLE_ASSIGN_OR_RETURN(auto logical, std::move(builder).finish());
  HYLE_ASSIGN_OR_RETURN(auto first_node, logical.node_id(0));
  HYLE_ASSIGN_OR_RETURN(auto second_node, logical.node_id(1));
  BackendBudgetFacts facts;
  const auto check_idle = [](const c::ResourceBudget &budget,
      std::span<const c::ResourceBudgetLimit> limits,
      std::span<const c::ResourceBudgetUsage> before,
      c::UnattributedStorageUsage opaque_before) -> Status {
    for (std::size_t index = 0; index < limits.size(); ++index) {
      HYLE_ASSIGN_OR_RETURN(auto after, budget.usage(limits[index].resource));
      if (after.live_bytes != before[index].live_bytes ||
          after.live_by_category != before[index].live_by_category)
        return Status::Internal("retired endpoint graph retained an actual resource charge");
    }
    const auto opaque = budget.unattributed_usage();
    if (opaque.private_upper_bound_bytes != opaque_before.private_upper_bound_bytes ||
        opaque.unknown_private_owners != opaque_before.unknown_private_owners ||
        opaque.control_bytes != opaque_before.control_bytes)
      return Status::Internal("retired endpoint graph retained an unattributed provider charge");
    return Status::Ok();
  };
  const auto run_selected = [&](bool cuda) -> Status {
    const auto resource = cuda ? backend.device() : backend.host_device();
    const g::StorageDomainId domain{cuda ? 0u : 1u};
    const g::ExecutionTargetId target{cuda ? 0u : 1u};
    const std::array limits{c::ResourceBudgetLimit{mem::system_resource(), 64 * 1024 * 1024},
        c::ResourceBudgetLimit{backend.device(), 64 * 1024 * 1024}};
    const auto active_limits = std::span(limits).first(cuda ? 2u : 1u);
    HYLE_ASSIGN_OR_RETURN(auto budget,
        c::ResourceBudget::create({.limits = active_limits, .allow_unattributed = true}));
    std::array<c::ResourceBudgetUsage, 2> before{};
    for (std::size_t index = 0; index < active_limits.size(); ++index) {
      HYLE_ASSIGN_OR_RETURN(before[index], budget.usage(active_limits[index].resource));
    }
    const auto opaque_before = budget.unattributed_usage();
    {
      const std::array placements{
          g::NodePlacement{first_node, target, g::PlacementStrength::kRequired},
          g::NodePlacement{second_node, target, g::PlacementStrength::kRequired}};
      const std::array external{g::ExternalResourceContract{input, domain},
          g::ExternalResourceContract{output, domain}};
      HYLE_ASSIGN_OR_RETURN(auto compiled, backend.compile(logical, {}, {
          .resource_budget = budget, .node_placements = placements, .external_resources = external,
          .region_selection = g::RegionSelectionPolicy::kDisabled}));
      if (compiled.report.copy_bytes() != 0)
        return Status::Internal("single-endpoint graph unexpectedly acquired a cross-domain copy");
      HYLE_ASSIGN_OR_RETURN(auto workspace, compiled.report.storage_domain(domain.ordinal()));
      if (!workspace.workspace_bytes)
        return Status::Internal("endpoint budget graph did not retain its real intermediate workspace");
      if (cuda) facts.cuda_workspace_bytes = workspace.workspace_bytes;
      HYLE_ASSIGN_OR_RETURN(auto queues, prepare_queue_bindings(backend.queue_bindings(), compiled.report));
      if (queues.size() != 1 || queues[0].lane != g::ExecutionLaneId{cuda ? 0u : 1u})
        return Status::Internal("single-endpoint graph did not select its original required Queue");
      HYLE_ASSIGN_OR_RETURN(auto readback, PreparedReadback::prepare(compiled, output, backend, budget));
      // The caller owns this input before admission; Graph's private workspace
      // and complete caller-output writer use the actual budget below.
      HYLE_ASSIGN_OR_RETURN(auto storage, mem::Buffer::create(resource, 17 * sizeof(float), 256));
      HYLE_ASSIGN_OR_RETURN(auto run, compiled.executable.create_invocation());
      HYLE_RETURN_IF_ERROR(run.prepare_resources());
      HYLE_ASSIGN_OR_RETURN(auto tensor, c::ConstTensorView::create(description, storage.view()));
      HYLE_RETURN_IF_ERROR(run.bind(input, tensor));
      HYLE_RETURN_IF_ERROR(readback.bind(run));
      for (std::size_t replay = 0; replay < 2; ++replay) {
        std::array<float, 17> values{}, expected{}, actual{};
        for (std::size_t index = 0; index < values.size(); ++index) {
          values[index] = static_cast<float>(static_cast<int>(index) - 8) / 8.F +
              static_cast<float>(replay + 1) / 32.F;
          expected[index] = -std::abs(values[index]);
        }
        HYLE_RETURN_IF_ERROR(backend.write(storage.view(), std::as_bytes(std::span(values))));
        HYLE_RETURN_IF_ERROR(run.prepare_replay(queue_span(queues), {.mode = g::ReplayMode::kCommands}));
        HYLE_ASSIGN_OR_RETURN(auto done, run.submit(queue_span(queues)));
        if (cuda) ++facts.cuda_submissions; else ++facts.host_submissions;
        HYLE_RETURN_IF_ERROR(done.wait());
        HYLE_RETURN_IF_ERROR(readback.read(backend, run, std::as_writable_bytes(std::span(actual))));
        if (actual != expected)
          return Status::Internal("single-endpoint budget graph changed exact Abs/Neg values");
        facts.compared_values += actual.size();
      }
      HYLE_RETURN_IF_ERROR(run.drain());
    }
    HYLE_RETURN_IF_ERROR(backend.collect_retired());
    HYLE_RETURN_IF_ERROR(check_idle(budget, active_limits,
        std::span(before).first(active_limits.size()), opaque_before));
    ++facts.retired_scopes;
    return Status::Ok();
  };
  HYLE_RETURN_IF_ERROR(run_selected(false));
  HYLE_RETURN_IF_ERROR(run_selected(true));
  const std::array limits{c::ResourceBudgetLimit{mem::system_resource(), 64 * 1024 * 1024}};
  HYLE_ASSIGN_OR_RETURN(auto missing_device,
      c::ResourceBudget::create({.limits = limits, .allow_unattributed = true}));
  HYLE_ASSIGN_OR_RETURN(auto before, missing_device.usage(limits[0].resource));
  const auto opaque_before = missing_device.unattributed_usage();
  {
    const std::array placements{
        g::NodePlacement{first_node, g::ExecutionTargetId{0}, g::PlacementStrength::kRequired},
        g::NodePlacement{second_node, g::ExecutionTargetId{0}, g::PlacementStrength::kRequired}};
    const std::array external{g::ExternalResourceContract{input, g::StorageDomainId{0}},
        g::ExternalResourceContract{output, g::StorageDomainId{0}}};
    auto compiled = backend.compile(logical, {}, {
        .resource_budget = missing_device, .node_placements = placements, .external_resources = external,
        .region_selection = g::RegionSelectionPolicy::kDisabled});
    Status refusal;
    if (!compiled) refusal = compiled.error();
    else {
      HYLE_ASSIGN_OR_RETURN(auto workspace, compiled->report.storage_domain(0));
      if (!workspace.workspace_bytes || workspace.workspace_bytes != facts.cuda_workspace_bytes)
        return std::unexpected(Status::Internal("missing-device graph lost its original actual GPU workspace"));
      // Graph reserves this known workspace before native allocation. Never
      // bind outputs or submit a graph which accepted the missing resource.
      auto run = compiled->executable.create_invocation();
      if (run)
        return std::unexpected(Status::Internal("missing actual GPU workspace budget admitted an Invocation"));
      refusal = run.error();
    }
    if (!refusal.IsInvalidArgument() || refusal.has_native_error() ||
        std::string_view(refusal.message()).find("resource is absent from budget") == std::string_view::npos)
      return std::unexpected(Status::Internal("missing GPU resource did not preserve its typed budget rejection"));
    ++facts.missing_resource_rejections;
  }
  HYLE_RETURN_IF_ERROR(backend.collect_retired());
  HYLE_RETURN_IF_ERROR(check_idle(missing_device, limits, std::span(&before, 1), opaque_before));
  ++facts.retired_scopes;
  return facts;
}

} // namespace hyle::inference::test

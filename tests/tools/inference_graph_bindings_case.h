#pragma once

#include "../../engine/inference/backend.h"
#include "../../engine/inference/execution_resources.h"
#include "../../engine/inference/graph_bindings.h"
#include "../../engine/inference/program_io.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

namespace hyle::inference::test {

struct GraphBindingsFacts final {
  std::size_t native_submissions = 0, compared_values = 0, retired_scopes = 0;
  std::size_t conflicting_external_rejections = 0, materialized_bytes = 0;
  std::size_t readback_values = 0, readback_domain_rejections = 0;
  std::size_t readback_owner_rejections = 0, readback_missing_domain_rejections = 0;
};

// Original CPU/CUDA endpoints, with CPU at storage zero and CUDA at one.
// Neither the assembler nor this consumer infers a device from its ordinal.
// Four real Graph submissions; no model, replacement Context or test kernel.
inline Result<GraphBindingsFacts> graph_bindings_case() {
  namespace c = compute;
  namespace g = graph;
  HYLE_ASSIGN_OR_RETURN(auto backend,
      Backend::create({.primary = BackendKind::kCuda, .cpu_assist = true}));
  HYLE_ASSIGN_OR_RETURN(auto cpu_context, backend.cpu_context());
  HYLE_ASSIGN_OR_RETURN(auto cuda_context, backend.common_context());
  HYLE_ASSIGN_OR_RETURN(auto cpu_source, g::ExecutionSource::acquire(*cpu_context));
  HYLE_ASSIGN_OR_RETURN(auto cuda_source, g::ExecutionSource::acquire(*cuda_context));
  std::array<g::StorageDomain, 2> domains{};
  bool found_cpu = false, found_cuda = false;
  for (const auto &domain : backend.storage_domains()) {
    if (domain.resource == backend.host_device()) { domains[0] = domain; found_cpu = true; }
    if (domain.resource == backend.device()) { domains[1] = domain; found_cuda = true; }
  }
  if (!found_cpu || !found_cuda || domains[0].resource == domains[1].resource || !domains[1].owner)
    return std::unexpected(Status::Internal("Graph binding case needs its original distinct CPU/CUDA domains"));
  const g::StorageDomainId cpu_storage{0}, cuda_storage{1};
  const g::ExecutionTargetId cuda_target{0}, cpu_target{1};
  const std::array targets{g::ExecutionTarget{cuda_source, cuda_storage},
      g::ExecutionTarget{cpu_source, cpu_storage}};
  const std::array lanes{g::ExecutionLane{cuda_target}, g::ExecutionLane{cpu_target}};
  const g::CompileEnvironment environment{.storage_domains = domains, .targets = targets, .lanes = lanes};
  constexpr std::size_t count = 17, bytes = count * sizeof(float);
  HYLE_ASSIGN_OR_RETURN(auto description, c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{count}));
  std::array<float, count> weight{};
  for (std::size_t index = 0; index < count; ++index) weight[index] = static_cast<float>(index + 1) / 8.F;
  GraphBindingsFacts facts;
  const auto read_result = [&](ProgramIo &io, ProgramIo::Readback port, g::Invocation &run,
      mem::MemoryResourceRef resource, std::span<float> values) -> Status {
    HYLE_ASSIGN_OR_RETURN(auto prepared, io.readback(port));
    HYLE_ASSIGN_OR_RETURN(auto result, run.result(prepared->facts.value));
    HYLE_ASSIGN_OR_RETURN(auto tensor, result.tensor());
    if (prepared->facts.storage.ordinal() >= domains.size() ||
        domains[prepared->facts.storage.ordinal()].resource != resource ||
        tensor.storage().resource() != resource)
      return Status::Internal("Graph binding output used the wrong actual storage resource");
    HYLE_RETURN_IF_ERROR(io.read(run, port, std::as_writable_bytes(values)));
    facts.readback_values += values.size();
    return Status::Ok();
  };
  {
    HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
    HYLE_ASSIGN_OR_RETURN(auto host_input, builder.input(description));
    HYLE_ASSIGN_OR_RETURN(auto device_input, builder.input(description));
    HYLE_ASSIGN_OR_RETURN(auto constant, builder.constant(description));
    HYLE_ASSIGN_OR_RETURN(auto host_output, builder.add(c::Unary{c::UnaryKind::kNeg}, host_input));
    HYLE_ASSIGN_OR_RETURN(auto device_output, builder.add(c::Binary{c::BinaryKind::kAdd}, device_input, constant));
    HYLE_RETURN_IF_ERROR(builder.publish(host_output));
    HYLE_RETURN_IF_ERROR(builder.publish(device_output));
    HYLE_ASSIGN_OR_RETURN(auto logical, std::move(builder).finish());
    HYLE_ASSIGN_OR_RETURN(auto host_node, logical.producer(host_output));
    HYLE_ASSIGN_OR_RETURN(auto device_node, logical.producer(device_output));
    g::CompiledGraph compiled;
    {
      HYLE_ASSIGN_OR_RETURN(auto buffer, mem::Buffer::create(domains[1].resource, bytes, 256));
      HYLE_RETURN_IF_ERROR(backend.write(buffer.view(), std::as_bytes(std::span(weight))));
      HYLE_ASSIGN_OR_RETURN(auto ready, domains[1].owner.adopt_buffer(std::move(buffer)));
      HYLE_ASSIGN_OR_RETURN(auto tensor, ready.tensor(description));
      GraphBindings bindings(domains);
      HYLE_ASSIGN_OR_RETURN(auto mapped_cpu, bindings.storage_for(domains[0].resource));
      HYLE_ASSIGN_OR_RETURN(auto mapped_cuda, bindings.storage_for(domains[1].resource));
      if (mapped_cpu != cpu_storage || mapped_cuda != cuda_storage)
        return std::unexpected(Status::Internal("Graph bindings inferred storage ordinals from backend names"));
      HYLE_RETURN_IF_ERROR(bindings.bind_constant({.value = constant, .tensor = tensor,
          .owner = domains[1].owner, .lease = ready}));
      const auto before = bindings.options();
      const auto conflict = bindings.external(constant, cpu_storage);
      const auto repeated = bindings.external(constant, cuda_storage);
      const auto after = bindings.options();
      if (!conflict.IsInvalidArgument() || !repeated.ok() || before.external_resources.size() != 1 ||
          after.external_resources.size() != 1 || after.external_resources[0].value != constant ||
          after.external_resources[0].storage != cuda_storage || after.constants.size() != 1 ||
          after.constants[0].policy != g::ConstantPolicy::kBorrow ||
          after.constants[0].owner.context_identity() != domains[1].owner.context_identity() ||
          after.constants[0].owner.ref() != domains[1].resource ||
          after.constants[0].lease.generation() != ready.generation() ||
          after.constants[0].tensor.storage().native_handle() != tensor.storage().native_handle())
        return std::unexpected(Status::Internal("conflicting external contract changed the original borrowed constant facts"));
      ++facts.conflicting_external_rejections;
      HYLE_RETURN_IF_ERROR(bindings.external(host_input, cpu_storage));
      HYLE_RETURN_IF_ERROR(bindings.external(host_output, cpu_storage));
      HYLE_RETURN_IF_ERROR(bindings.external(device_input, cuda_storage));
      HYLE_RETURN_IF_ERROR(bindings.external(device_output, cuda_storage));
      HYLE_RETURN_IF_ERROR(bindings.node(host_node, {cpu_target, cpu_storage}));
      HYLE_RETURN_IF_ERROR(bindings.node(device_node, {cuda_target, cuda_storage}));
      const auto options = bindings.options({.max_placement_attempts = 512,
          .region_selection = g::RegionSelectionPolicy::kDisabled, .max_route_search_states = 2048});
      if (options.max_placement_attempts != 512 || options.max_route_search_states != 2048 ||
          options.region_selection != g::RegionSelectionPolicy::kDisabled || options.node_placements.size() != 2 ||
          options.node_placements[0].node != host_node || options.node_placements[0].target != cpu_target ||
          options.node_placements[1].node != device_node || options.node_placements[1].target != cuda_target ||
          options.node_placements[0].strength != g::PlacementStrength::kRequired ||
          options.node_placements[1].strength != g::PlacementStrength::kRequired)
        return std::unexpected(Status::Internal("Graph bindings changed the caller's compile policy or exact node placement"));
      HYLE_ASSIGN_OR_RETURN(compiled, g::batch::compile(logical, environment, options));
    } // The compiled Graph must retain the actual borrowed owner and lease.
    HYLE_ASSIGN_OR_RETURN(auto borrowed, compiled.report.constant());
    HYLE_ASSIGN_OR_RETURN(auto cuda_report, compiled.report.target(cuda_target.ordinal()));
    HYLE_ASSIGN_OR_RETURN(auto cpu_report, compiled.report.target(cpu_target.ordinal()));
    if (compiled.report.borrowed_constant_count() != 1 || borrowed.storage != cuda_storage ||
        borrowed.policy != g::ConstantPolicy::kBorrow || compiled.report.materialized_constant_bytes() ||
        compiled.report.copy_bytes() || cuda_report.default_storage != cuda_storage ||
        cpu_report.default_storage != cpu_storage)
      return std::unexpected(Status::Internal("Graph did not preserve the reversed-domain borrowed route"));
    HYLE_ASSIGN_OR_RETURN(auto queues, prepare_queue_bindings(backend.queue_bindings(), compiled.report));
    if (queues.size() != 2)
      return std::unexpected(Status::Internal("CPU/CUDA Graph did not retain both original execution lanes"));
    HYLE_ASSIGN_OR_RETURN(auto host_in, mem::Buffer::create(domains[0].resource, bytes, 256));
    HYLE_ASSIGN_OR_RETURN(auto device_in, mem::Buffer::create(domains[1].resource, bytes, 256));
    {
      ProgramIo refused;
      HYLE_RETURN_IF_ERROR(refused.prepare(compiled, 0, 0, 1, 0));
      const auto missing = refused.add_readback(host_output, {});
      if (missing || !missing.error().IsOutOfRange())
        return std::unexpected(Status::Internal("caller readback accepted a missing actual storage domain"));
      ++facts.readback_missing_domain_rejections;
      for (const auto output : {host_output, device_output}) {
        const auto wrong_order = refused.add_readback(output, backend.storage_domains());
        if (wrong_order || !wrong_order.error().IsInvalidArgument() ||
            wrong_order.error().message() != "readback storage differs from its compiled domain")
          return std::unexpected(Status::Internal("readback accepted the Backend's unrelated domain ordinals"));
        ++facts.readback_domain_rejections;
      }
      auto wrong_owner = domains;
      wrong_owner[0].owner = domains[1].owner;
      const auto mismatched = refused.add_readback(host_output, wrong_owner);
      if (mismatched || !mismatched.error().IsInvalidArgument() ||
          mismatched.error().message() != "readback storage resource and owner differ")
        return std::unexpected(Status::Internal("readback accepted an owner of a different actual resource"));
      ++facts.readback_owner_rejections;
      // All cold refusals leave the same inventory available for its actual
      // domain. No Invocation, payload writer or partial IO entry was installed.
      HYLE_ASSIGN_OR_RETURN(auto recovered, refused.add_readback(host_output, domains));
      HYLE_RETURN_IF_ERROR(refused.seal());
      HYLE_ASSIGN_OR_RETURN(auto recovered_port, refused.readback(recovered));
      if (recovered_port->facts.storage != cpu_storage)
        return std::unexpected(Status::Internal("refused readback registration changed its original output facts"));
    }
    ProgramIo io;
    HYLE_RETURN_IF_ERROR(io.prepare(compiled, 0, 0, 2, 0));
    HYLE_ASSIGN_OR_RETURN(auto host_readback, io.add_readback(host_output, domains));
    HYLE_ASSIGN_OR_RETURN(auto device_readback, io.add_readback(device_output, domains));
    HYLE_RETURN_IF_ERROR(io.seal());
    HYLE_ASSIGN_OR_RETURN(auto run, compiled.executable.create_invocation());
    HYLE_RETURN_IF_ERROR(run.prepare_resources());
    HYLE_ASSIGN_OR_RETURN(auto host_in_view, c::ConstTensorView::create(description, host_in.view()));
    HYLE_ASSIGN_OR_RETURN(auto device_in_view, c::ConstTensorView::create(description, device_in.view()));
    HYLE_RETURN_IF_ERROR(run.bind(host_input, host_in_view));
    HYLE_RETURN_IF_ERROR(run.bind(device_input, device_in_view));
    HYLE_RETURN_IF_ERROR(io.bind_fixed(run));
    for (std::size_t replay = 0; replay < 2; ++replay) {
      std::array<float, count> values{}, expected_host{}, expected_device{}, actual_host{}, actual_device{};
      for (std::size_t index = 0; index < count; ++index) {
        values[index] = static_cast<float>(static_cast<int>(index) - 8) / 8.F + static_cast<float>(replay) / 32.F;
        expected_host[index] = -values[index]; expected_device[index] = values[index] + weight[index];
      }
      HYLE_RETURN_IF_ERROR(backend.write(host_in.view(), std::as_bytes(std::span(values))));
      HYLE_RETURN_IF_ERROR(backend.write(device_in.view(), std::as_bytes(std::span(values))));
      HYLE_RETURN_IF_ERROR(run.prepare_replay(queue_span(queues), {.mode = g::ReplayMode::kCommands}));
      HYLE_ASSIGN_OR_RETURN(auto done, run.submit(queue_span(queues))); ++facts.native_submissions;
      HYLE_RETURN_IF_ERROR(done.wait());
      HYLE_RETURN_IF_ERROR(read_result(io, host_readback, run, domains[0].resource, actual_host));
      HYLE_RETURN_IF_ERROR(read_result(io, device_readback, run, domains[1].resource, actual_device));
      if (actual_host != expected_host || actual_device != expected_device)
        return std::unexpected(Status::Internal("reversed-domain borrowed replay changed exact CPU/CUDA outputs"));
      facts.compared_values += 2 * count;
    }
    HYLE_RETURN_IF_ERROR(run.drain());
  }
  HYLE_RETURN_IF_ERROR(backend.collect_retired()); ++facts.retired_scopes;
  {
    HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
    HYLE_ASSIGN_OR_RETURN(auto input, builder.input(description));
    HYLE_ASSIGN_OR_RETURN(auto constant, builder.constant(description));
    HYLE_ASSIGN_OR_RETURN(auto output, builder.add(c::Binary{c::BinaryKind::kAdd}, input, constant));
    HYLE_RETURN_IF_ERROR(builder.publish(output));
    HYLE_ASSIGN_OR_RETURN(auto logical, std::move(builder).finish());
    HYLE_ASSIGN_OR_RETURN(auto node, logical.producer(output));
    g::CompiledGraph compiled;
    {
      HYLE_ASSIGN_OR_RETURN(auto source_owner, backend.storage_owner_for(backend.staging()));
      HYLE_ASSIGN_OR_RETURN(auto buffer, mem::Buffer::create(source_owner.ref(), bytes, 256));
      HYLE_RETURN_IF_ERROR(backend.write(buffer.view(), std::as_bytes(std::span(weight))));
      // CUDA's actual pinned-host source and owner are intentionally outside
      // these two compile domains. Its native copy eligibility is unchanged;
      // materialization's explicit destination is still local CUDA domain one.
      HYLE_ASSIGN_OR_RETURN(auto ready, source_owner.adopt_buffer(std::move(buffer)));
      HYLE_ASSIGN_OR_RETURN(auto tensor, ready.tensor(description));
      GraphBindings bindings(domains);
      HYLE_RETURN_IF_ERROR(bindings.bind_constant({.value = constant, .tensor = tensor,
          .policy = g::ConstantPolicy::kMaterialize, .owner = source_owner, .lease = ready}));
      if (!bindings.options().external_resources.empty())
        return std::unexpected(Status::Internal("materialized constant was automatically placed in its source domain"));
      HYLE_RETURN_IF_ERROR(bindings.external(constant, cuda_storage));
      HYLE_RETURN_IF_ERROR(bindings.external(input, cuda_storage));
      HYLE_RETURN_IF_ERROR(bindings.external(output, cuda_storage));
      HYLE_RETURN_IF_ERROR(bindings.node(node, {cuda_target, cuda_storage}));
      HYLE_ASSIGN_OR_RETURN(compiled, g::batch::compile(logical, environment,
          bindings.options({.region_selection = g::RegionSelectionPolicy::kDisabled})));
    } // The pinned-host source, owner, lease and assembly end before replay.
    HYLE_ASSIGN_OR_RETURN(auto materialized, compiled.report.constant());
    if (compiled.report.materialized_constant_count() != 1 || materialized.storage != cuda_storage ||
        materialized.policy != g::ConstantPolicy::kMaterialize || materialized.materialized_bytes != bytes)
      return std::unexpected(Status::Internal("Graph materialized the pinned-host source into the wrong destination domain"));
    facts.materialized_bytes = materialized.materialized_bytes;
    HYLE_ASSIGN_OR_RETURN(auto queues, prepare_queue_bindings(backend.queue_bindings(), compiled.report));
    if (queues.size() != 1 || queues[0].lane != g::ExecutionLaneId{0})
      return std::unexpected(Status::Internal("materialized CUDA route changed its original execution lane"));
    HYLE_ASSIGN_OR_RETURN(auto storage, mem::Buffer::create(domains[1].resource, bytes, 256));
    ProgramIo io;
    HYLE_RETURN_IF_ERROR(io.prepare(compiled, 0, 0, 1, 0));
    HYLE_ASSIGN_OR_RETURN(auto readback, io.add_readback(output, domains));
    HYLE_RETURN_IF_ERROR(io.seal());
    HYLE_ASSIGN_OR_RETURN(auto run, compiled.executable.create_invocation());
    HYLE_RETURN_IF_ERROR(run.prepare_resources());
    HYLE_ASSIGN_OR_RETURN(auto in_view, c::ConstTensorView::create(description, storage.view()));
    HYLE_RETURN_IF_ERROR(run.bind(input, in_view)); HYLE_RETURN_IF_ERROR(io.bind_fixed(run));
    for (std::size_t replay = 0; replay < 2; ++replay) {
      std::array<float, count> values{}, expected{}, actual{};
      for (std::size_t index = 0; index < count; ++index) {
        values[index] = static_cast<float>(replay + 1) / 4.F - static_cast<float>(index) / 16.F;
        expected[index] = values[index] + weight[index];
      }
      HYLE_RETURN_IF_ERROR(backend.write(storage.view(), std::as_bytes(std::span(values))));
      HYLE_RETURN_IF_ERROR(run.prepare_replay(queue_span(queues), {.mode = g::ReplayMode::kCommands}));
      HYLE_ASSIGN_OR_RETURN(auto done, run.submit(queue_span(queues))); ++facts.native_submissions;
      HYLE_RETURN_IF_ERROR(done.wait());
      HYLE_RETURN_IF_ERROR(read_result(io, readback, run, domains[1].resource, actual));
      if (actual != expected)
        return std::unexpected(Status::Internal("materialized pinned-host-to-CUDA replay changed exact output"));
      facts.compared_values += count;
    }
    HYLE_RETURN_IF_ERROR(run.drain());
  }
  HYLE_RETURN_IF_ERROR(backend.collect_retired()); ++facts.retired_scopes;
  return facts;
}

} // namespace hyle::inference::test

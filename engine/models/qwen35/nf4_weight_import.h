#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <new>
#include <span>
#include <string_view>
#include <utility>

#include "nf4_state.h"
#include "../../inference/artifact_digest.h"
#include "../../inference/backend.h"
#include "../../inference/execution_resources.h"
#include "../../inference/prepared_output.h"
#include "hyle/compute/model_ops.h"

namespace qwen35 {

// Owned ready inputs survive both rejected work and native retirement. The
// caller authenticates their original file ranges; the raw JSON is checked
// against its separately prepared SHA and values below. No files are opened.
struct Nf4DecodeInputs final {
  std::array<hyle::compute::TensorDesc, 6> descriptions;
  std::array<hyle::compute::ReadLease, 6> ready;
  std::span<const std::byte> original_json;
};

struct ImportedNf4Weight final {
  // Destroy the payload before its native resource/Context controller.
  hyle::compute::StorageOwner owner;
  hyle::compute::TensorDesc description;
  hyle::compute::ReadLease ready;
  hyle::graph::StorageDomainId storage;
  std::size_t canonical_bytes = 0, graph_workspace_bytes = 0;
};

// The original metadata, typed operands and selected resource are validated
// once during cold preparation. Both source-retaining projections and explicit
// dense import use this same contract; this function performs no Decode.
inline hyle::Result<hyle::compute::TensorDesc> validate_nf4_decode_inputs(
    const hyle::inference::Backend &backend, const prepared::Nf4State &state,
    const Nf4DecodeInputs &inputs,
    hyle::graph::StorageDomainId storage = hyle::graph::StorageDomainId{0}) noexcept {
  namespace c = hyle::compute;
  namespace inf = hyle::inference;
  try {
    if (inputs.original_json.empty() || inputs.original_json.size() > prepared::maximum_nf4_state_bytes ||
        inf::artifact::sha256(inputs.original_json) != state.original_json_sha256)
      return std::unexpected(hyle::Status::InvalidArgument("original NF4 quant_state SHA256 differs"));
    prepared::validate_nf4_state_values(std::string_view(
        reinterpret_cast<const char *>(inputs.original_json.data()), inputs.original_json.size()), state);
    HYLE_ASSIGN_OR_RETURN(auto resource, backend.resource_for(storage));
    const c::model_ops::ArtifactWeightDecode decode{
        c::model_ops::WeightArtifactFormat::kNf4DoubleQuant, 64, c::DataType::kFloat16};
    std::array<c::TensorDesc, 1> inferred;
    HYLE_RETURN_IF_ERROR(decode.infer_output_descriptions(inputs.descriptions, inferred));
    if (inferred[0].dimensions()[0] != state.logical_shape[0] ||
        inferred[0].dimensions()[1] != state.logical_shape[1])
      return std::unexpected(hyle::Status::InvalidArgument("NF4 decode inputs differ from original logical shape"));
    std::array<c::ConstTensorView, 6> views;
    for (std::size_t i = 0; i < views.size(); ++i) {
      HYLE_ASSIGN_OR_RETURN(views[i], inputs.ready[i].tensor(inputs.descriptions[i]));
      if (views[i].storage().resource() != resource)
        return std::unexpected(hyle::Status::InvalidArgument("NF4 decode input resource differs from import storage"));
    }
    std::array<std::byte, 4> offset;
    HYLE_RETURN_IF_ERROR(backend.read(views[4].storage(), offset));
    if (std::bit_cast<std::uint32_t>(offset) != state.offset_f32_bits)
      return std::unexpected(hyle::Status::InvalidArgument("NF4 offset operand differs from original F32 bits"));
    return inferred[0];
  } catch (const std::bad_alloc &) {
    return std::unexpected(hyle::Status::ResourceExhausted("NF4 cold validation metadata allocation failed"));
  } catch (const std::exception &) {
    return std::unexpected(hyle::Status::InvalidArgument("invalid original NF4 decode metadata"));
  }
}

// Exactly one explicit F16 decode on the selected original Backend. The IO
// caller keeps its Backend until the import finishes and retains the returned
// owner (or the same Backend) while using ready. There is no hidden CPU route,
// hot decode, dense full-bank cache or permission promotion of source readers.
inline hyle::Result<ImportedNf4Weight> import_nf4_weight(
    hyle::inference::Backend &backend, const prepared::Nf4State &state,
    const Nf4DecodeInputs &inputs,
    hyle::graph::StorageDomainId storage = hyle::graph::StorageDomainId{0},
    const hyle::compute::ResourceBudget &budget = {}) noexcept {
  namespace c = hyle::compute;
  namespace g = hyle::graph;
  namespace inf = hyle::inference;
  try {
    HYLE_ASSIGN_OR_RETURN(auto description, validate_nf4_decode_inputs(backend, state, inputs, storage));
    HYLE_ASSIGN_OR_RETURN(auto resource, backend.resource_for(storage));
    HYLE_ASSIGN_OR_RETURN(auto owner, backend.storage_owner_for(resource));
    const c::model_ops::ArtifactWeightDecode decode{
        c::model_ops::WeightArtifactFormat::kNf4DoubleQuant, 64, c::DataType::kFloat16};
    const std::array inferred{description};
    HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
    std::array<g::ValueId, 6> ports;
    std::array<g::ExternalResourceContract, 7> external;
    for (std::size_t i = 0; i < ports.size(); ++i) {
      HYLE_ASSIGN_OR_RETURN(ports[i], builder.input(inputs.descriptions[i]));
      external[i] = {ports[i], storage};
    }
    HYLE_ASSIGN_OR_RETURN(auto node, builder.add_node(c::Operation::custom(decode), ports));
    HYLE_ASSIGN_OR_RETURN(auto output, builder.result(node));
    HYLE_RETURN_IF_ERROR(builder.publish(output));
    external.back() = {output, storage};
    HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
    HYLE_ASSIGN_OR_RETURN(auto compiled, backend.compile(graph, {},
        {.resource_budget = budget, .external_resources = external}));
    HYLE_ASSIGN_OR_RETURN(auto facts, inf::prepare_output(compiled, output));
    if (facts.description != inferred[0] || facts.storage != storage)
      return std::unexpected(hyle::Status::Internal("NF4 prepared output differs from exact import contract"));
    HYLE_ASSIGN_OR_RETURN(auto bytes, facts.description.required_bytes());
    // Caller output is admitted as an owned writer, never a raw borrowed
    // Buffer. The original Graph retains it through accepted failed work.
    c::LeasePool destination;
    c::WriteLease writer;
    if (facts.caller_owned()) {
      HYLE_ASSIGN_OR_RETURN(destination, c::LeasePool::create(resource, bytes,
          {.max_blocks = 1, .resource_budget = budget}));
      HYLE_ASSIGN_OR_RETURN(writer, destination.allocate(bytes));
    }
    HYLE_ASSIGN_OR_RETURN(auto run, compiled.executable.create_invocation());
    struct Drain final {
      inf::Backend &backend; g::Invocation &run;
      ~Drain() { run.close().IgnoreError(); backend.drain_retired(run).IgnoreError(); }
    } drain{backend, run};
    HYLE_RETURN_IF_ERROR(run.prepare_resources({.resource_budget = budget}));
    for (std::size_t i = 0; i < ports.size(); ++i)
      HYLE_RETURN_IF_ERROR(run.bind(ports[i], inputs.descriptions[i], inputs.ready[i]));
    if (facts.caller_owned()) HYLE_RETURN_IF_ERROR(run.bind(output, facts.description, std::move(writer)));
    HYLE_ASSIGN_OR_RETURN(auto queues, inf::prepare_queue_bindings(backend.queue_bindings(), compiled.report));
    const auto bindings = inf::queue_span(queues);
    HYLE_RETURN_IF_ERROR(run.prepare_replay(bindings,
        {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kAllow}));
    HYLE_ASSIGN_OR_RETURN(auto done, run.submit(bindings));
    HYLE_RETURN_IF_ERROR(done.wait());
    HYLE_ASSIGN_OR_RETURN(auto ready, inf::retain_output(run, facts, budget));
    return ImportedNf4Weight{std::move(owner), facts.description, std::move(ready),
        storage, bytes, compiled.executable.workspace_bytes()};
  } catch (const std::bad_alloc &) {
    return std::unexpected(hyle::Status::ResourceExhausted("NF4 cold import metadata allocation failed"));
  } catch (const std::exception &) {
    return std::unexpected(hyle::Status::InvalidArgument("invalid original NF4 import metadata"));
  }
}
} // namespace qwen35

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hyle/graph/region_provider.h"
#include "../graph_ggml_shared/constant_packing.h"
#include "../compute_provider_ggml_cuda/program.h"

namespace hyle::graph::ggml_cuda::constant_expressions {
namespace native = compute::ggml_cuda::internal;

// A recipe is private native metadata, so unused values can be removed without
// changing Graph boundary port order. Ordinary inputs retain that exact order.
inline void compact(native::Recipe &recipe, std::vector<ValueId> *graph_values = nullptr) {
  std::vector<std::uint8_t> live(recipe.values.size(), 0);
  for (const auto id : recipe.inputs) live[id] = 1;
  for (const auto id : recipe.outputs) live[id] = 1;
  for (const auto &node : recipe.nodes) {
    for (const auto id : node.inputs) live[id] = 1;
    for (const auto id : node.outputs) live[id] = 1;
  }
  std::vector<std::uint32_t> remap(live.size(), UINT32_MAX);
  std::vector<compute::TensorDesc> descriptions;
  std::vector<ValueId> identities;
  for (std::size_t id = 0; id < live.size(); ++id) {
    if (!live[id]) continue;
    remap[id] = static_cast<std::uint32_t>(descriptions.size());
    descriptions.push_back(recipe.values[id]);
    if (graph_values) identities.push_back((*graph_values)[id]);
  }
  std::erase_if(recipe.constants, [&](const auto &entry) { return !live[entry.value]; });
  for (auto &entry : recipe.constants) entry.value = remap[entry.value];
  for (auto &id : recipe.inputs) id = remap[id];
  for (auto &id : recipe.outputs) id = remap[id];
  for (auto &node : recipe.nodes) {
    for (auto &id : node.inputs) id = remap[id];
    for (auto &id : node.outputs) id = remap[id];
  }
  recipe.values = std::move(descriptions);
  if (graph_values) *graph_values = std::move(identities);
}

struct PreparedExpression final {
  std::vector<mem::Buffer> storage;
  std::vector<compute::ConstTensorView> outputs;
  compute::detail::CompletionLifetime native{};
  void retire() noexcept {
    const auto retained = std::exchange(native, {});
    if (retained.valid()) retained.release(retained.state);
  }
  ~PreparedExpression() { retire(); }
};
struct Preparation final {
  void *context;
  native::Recipe recipe;
};
inline Status make(void *opaque, const ConstantArtifactPrepareRequest &request,
                   ConstantArtifactPayload *output) noexcept {
  if (!request.sequence)
    return Status::FailedPrecondition("ggml CUDA constant expressions require a preparation sequence");
  const auto &preparation = *static_cast<const Preparation *>(opaque);
  const auto allocator = request.allocator;
  auto *memory = allocator.allocate(sizeof(PreparedExpression), alignof(PreparedExpression));
  if (!memory)
    return Status::ResourceExhausted("ggml CUDA constant expression metadata failed");
  auto *result = ::new (memory) PreparedExpression;
  *output = {result, +[](void *value, mem::AllocatorRef owner) noexcept {
    auto *expression = static_cast<PreparedExpression *>(value);
    expression->~PreparedExpression();
    owner.deallocate(expression, sizeof(PreparedExpression), alignof(PreparedExpression));
  }, +[](void *value) noexcept { static_cast<PreparedExpression *>(value)->retire(); }};
  try {
    std::vector<compute::TensorView> destinations;
    const auto count = preparation.recipe.outputs.size();
    result->storage.reserve(count);
    result->outputs.reserve(count);
    destinations.reserve(count);
    for (const auto id : preparation.recipe.outputs) {
      const auto &description = preparation.recipe.values[id];
      HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
      HYLE_ASSIGN_OR_RETURN(auto buffer, mem::Buffer::create(request.destination, bytes, 256));
      HYLE_ASSIGN_OR_RETURN(auto destination, compute::TensorView::create(description, buffer.view()));
      result->storage.push_back(std::move(buffer));
      destinations.push_back(destination);
      result->outputs.push_back(destination.as_const());
    }
    return native::evaluate_constant_recipe(preparation.context, preparation.recipe,
        destinations, allocator, *request.sequence, &result->native);
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("ggml CUDA constant expression allocation failed");
  } catch (...) {
    return Status::Internal("ggml CUDA constant expression preparation failed");
  }
}

inline void append(std::vector<std::byte> &bytes, std::uint64_t word) {
  for (unsigned shift = 0; shift < 64; shift += 8)
    bytes.push_back(static_cast<std::byte>((word >> shift) & 255));
}
inline void append(std::vector<std::byte> &bytes, const compute::TensorDesc &description) {
  append(bytes, static_cast<std::uint64_t>(description.data_type()));
  append(bytes, description.rank());
  for (const auto value : description.dimensions()) append(bytes, static_cast<std::uint64_t>(value));
  for (const auto value : description.byte_strides()) append(bytes, static_cast<std::uint64_t>(value));
}

// The selected provider owns constant evaluation. Only actual immutable, owned
// Graph roots and their checked metadata aliases qualify; dynamic inputs,
// effects and writable published values
// keep their original runtime operations. Native lowering retains all dtype
// boundaries and math, including Exp and BF16 per-node rounding.
inline compute::ProviderPrepareOutcome prepare(void *context,
    const RegionCompileRequest &request, std::vector<ValueId> &values,
    native::Recipe &recipe) {
  using Outcome = compute::ProviderPrepareOutcome;
  const auto access = request.constant_artifacts;
  if (!access.available()) return Outcome::prepared();
  std::vector<std::uint8_t> immutable(recipe.values.size(), 0);
  std::vector<RegionConstantOrigin> origins(recipe.values.size());
  std::vector<std::uint8_t> folded(recipe.nodes.size(), 0);
  native::Recipe cold;
  cold.values = recipe.values;
  mem::MemoryResourceRef destination;
  ValueId anchor;
  for (const auto &constant : request.constants) {
    const auto local = std::ranges::find(values, constant.value);
    if (local == values.end()) continue;
    const auto input = std::ranges::find(request.inputs, constant.value);
    if (input == request.inputs.end()) continue;
    const auto resource = request.input_resources[static_cast<std::size_t>(input - request.inputs.begin())];
    if (!destination.valid()) destination = resource;
    if (!access.can_prepare_queued(constant.value, destination)) continue;
    const auto id = static_cast<std::uint32_t>(local - values.begin());
    if (!recipe.values[id].is_plain()) continue;
    immutable[id] = 1;
    const auto origin = std::ranges::find_if(request.constant_origins,
        [&](const auto &entry) { return entry.value == constant.value; });
    origins[id] = origin == request.constant_origins.end()
        ? RegionConstantOrigin{constant.value, constant.value, 0} : *origin;
    cold.constants.push_back({id, constant.tensor});
    if (!anchor.valid()) anchor = constant.value;
  }
  if (!anchor.valid()) return Outcome::prepared();
  for (std::size_t index = 0; index < recipe.nodes.size(); ++index) {
    const auto &node = recipe.nodes[index];
    const auto traits = compute::operation_traits(node.operation);
    if (node.inputs.empty() || traits.effect != compute::EffectKind::kPure ||
        traits.resource_count != 0 || traits.alias == compute::AliasPolicy::kMustAliasInput0 ||
        !std::ranges::all_of(node.inputs, [&](auto id) { return immutable[id] != 0; }) ||
        !std::ranges::all_of(node.outputs, [&](auto id) {
          const auto &description = recipe.values[id];
          return description.is_plain() && description.is_dense() &&
              !ggml_shared::published_view_of(*request.graph, values[id]);
        }))
      continue;
    folded[index] = 1;
    for (const auto id : node.outputs) immutable[id] = 2;
    cold.nodes.push_back(node);
  }
  if (cold.nodes.empty()) return Outcome::prepared();
  std::vector<std::uint8_t> frontier(recipe.values.size(), 0);
  for (const auto id : recipe.outputs) if (immutable[id] == 2) frontier[id] = 1;
  for (std::size_t index = 0; index < recipe.nodes.size(); ++index)
    if (!folded[index])
      for (const auto id : recipe.nodes[index].inputs)
        if (immutable[id] == 2) frontier[id] = 1;
  std::vector<std::uint32_t> prepared_values;
  std::vector<std::uint32_t> canonical(recipe.values.size());
  for (std::uint32_t id = 0; id < canonical.size(); ++id) canonical[id] = id;
  // Cast is a complete conversion schema (its sole parameter is output_type).
  // Preserve the existing sharing of repeated immutable conversions; distinct
  // expressions still have distinct identities and independent outputs.
  std::unordered_map<std::uint64_t, std::uint32_t> conversions;
  for (const auto &node : cold.nodes) {
    const auto *cast = node.operation.get_if<compute::Cast>();
    if (!cast || node.inputs.size() != 1 || node.outputs.size() != 1) continue;
    const auto key = (static_cast<std::uint64_t>(cast->output_type) << 32) |
                     canonical[node.inputs[0]];
    const auto [prior, inserted] = conversions.emplace(key, node.outputs[0]);
    if (!inserted && recipe.values[prior->second] == recipe.values[node.outputs[0]])
      canonical[node.outputs[0]] = canonical[prior->second];
  }
  std::vector<std::size_t> unique_ports(recipe.values.size(), SIZE_MAX);
  std::vector<std::size_t> output_ports;
  std::vector<std::byte> identity;
  std::size_t bytes = 0;
  // The cache is family-local to one immutable Graph: graph node/value IDs
  // fix descriptors' operations and every option. Concrete value descriptions
  // distinguish finite-shape variants; no pointers, padding or lossy DAG hash.
  // Each used boundary also names its true root and physical View offset.
  // This includes non-anchor roots: equal-shape variants can slice different
  // parts of one root without changing a node or boundary value ID.
  std::vector<std::uint8_t> used_roots(recipe.values.size(), 0);
  for (const auto &node : cold.nodes)
    for (const auto id : node.inputs)
      if (immutable[id] == 1) used_roots[id] = 1;
  append(identity, static_cast<std::uint64_t>(
      std::ranges::count(used_roots, std::uint8_t{1})));
  for (std::size_t id = 0; id < used_roots.size(); ++id) {
    if (!used_roots[id]) continue;
    append(identity, origins[id].value.ordinal());
    append(identity, origins[id].root.ordinal());
    append(identity, origins[id].byte_offset);
    append(identity, recipe.values[id]);
  }
  append(identity, cold.nodes.size());
  for (std::size_t index = 0; index < recipe.nodes.size(); ++index) {
    if (!folded[index]) continue;
    append(identity, request.nodes[index].ordinal());
    const auto &node = recipe.nodes[index];
    append(identity, node.inputs.size());
    for (const auto id : node.inputs) {
      append(identity, values[id].ordinal()); append(identity, recipe.values[id]);
    }
    append(identity, node.outputs.size());
    for (const auto id : node.outputs) {
      append(identity, values[id].ordinal()); append(identity, recipe.values[id]);
    }
  }
  for (std::uint32_t id = 0; id < frontier.size(); ++id) {
    if (!frontier[id]) continue;
    const auto boundary = std::ranges::find(request.outputs, values[id]);
    if (boundary != request.outputs.end() &&
        request.output_resources[static_cast<std::size_t>(boundary - request.outputs.begin())] != destination)
      return Outcome::prepared();
    prepared_values.push_back(id);
    append(identity, values[id].ordinal());
    auto &port = unique_ports[canonical[id]];
    if (port != SIZE_MAX) { output_ports.push_back(port); continue; }
    port = cold.outputs.size();
    output_ports.push_back(port);
    const auto size = recipe.values[id].required_bytes().value();
    if (size > SIZE_MAX - bytes)
      return Outcome::error(Status::OutOfRange("ggml CUDA constant expression size overflows"));
    bytes += size;
    // Every frontier gets independent storage, even when a native no-op Cast
    // would alias an input or another output. Copy preserves its storage bits.
    const auto output = static_cast<std::uint32_t>(cold.values.size());
    cold.values.push_back(recipe.values[id]);
    cold.nodes.push_back({compute::Operation(compute::Copy{}), {canonical[id]}, {output}});
    cold.outputs.push_back(output);
  }
  if (prepared_values.empty()) return Outcome::prepared();
  compact(cold);
  Preparation preparation{context, std::move(cold)};
  ConstantArtifactView owner;
  const auto outcome = access.get_or_create(anchor,
      {.key = {0x4747435545585052ULL, 2, identity}, .storage = destination,
       .packed_bytes = bytes, .persistent_bytes = std::nullopt,
       .private_storage = {.resident_bytes = bytes, .peak_bytes = std::nullopt},
       .queue_preparation = true}, &make, &preparation, &owner);
  if (outcome.kind != compute::ProviderPrepareOutcomeKind::kPrepared) return outcome;
  const auto &expression = *static_cast<const PreparedExpression *>(owner.state);
  for (std::size_t port = 0; port < prepared_values.size(); ++port) {
    const auto id = prepared_values[port];
    const auto binding = expression.outputs[output_ports[port]];
    recipe.constants.push_back({id, binding});
    if (std::ranges::find(request.outputs, values[id]) != request.outputs.end()) {
      const auto status = access.bind_prepared_output(values[id], owner, binding);
      if (!status.ok()) return Outcome::error(status);
      std::erase(recipe.outputs, id);
    }
  }
  std::size_t index = 0;
  std::erase_if(recipe.nodes, [&](const auto &) { return folded[index++] != 0; });
  compact(recipe, &values);
  return Outcome::prepared();
}
} // namespace hyle::graph::ggml_cuda::constant_expressions

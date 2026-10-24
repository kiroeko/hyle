#pragma once

#include <algorithm>
#include <array>
#include <new>
#include <utility>
#include <vector>
#include "hyle/compute/model_ops.h"
#include "hyle/graph/region_provider.h"
#include "hyle/mem/buffer.h"

// Header-only ownership glue for ggml region adapters. Native execution stays
// inside each existing provider; Graph core has no ggml/context dependency.
namespace hyle::graph::ggml_shared {
struct Pack final {
  using Synchronous = Status (*)(void *, compute::ConstTensorView,
      compute::TensorView, mem::AllocatorRef) noexcept;
  using Queued = Status (*)(void *, compute::ConstTensorView,
      compute::TensorView, mem::AllocatorRef, compute::detail::QueueSequence &,
      compute::detail::CompletionLifetime *) noexcept;
  Synchronous synchronous = nullptr;
  Queued queued = nullptr;
  Pack(Synchronous value = nullptr) noexcept : synchronous(value) {}
  Pack(Queued value) noexcept : queued(value) {}
  Pack(std::nullptr_t) noexcept {}
  explicit operator bool() const noexcept { return synchronous || queued; }
};
inline constexpr std::uint64_t kBfloatToFloat = 0x4747435542463332ULL;
inline constexpr std::uint64_t kMatrixTranspose = 0x47474d5452414e53ULL;
struct PackedConstant final {
  mem::Buffer storage;
  compute::TensorDesc description;
  compute::detail::CompletionLifetime native{};
  void retire() noexcept {
    const auto retained = std::exchange(native, {});
    if (retained.valid()) retained.release(retained.state);
  }
  ~PackedConstant() { retire(); }
};
struct PackingRequest final {
  void *native_context;
  Pack pack;
  compute::TensorDesc description;
};
inline Status make_constant(void *opaque,
                            const ConstantArtifactPrepareRequest &preparation,
                            ConstantArtifactPayload *output) noexcept {
  const auto &request = *static_cast<const PackingRequest *>(opaque);
  const auto allocator = preparation.allocator;
  void *storage = allocator.allocate(sizeof(PackedConstant), alignof(PackedConstant));
  if (!storage)
    return Status::ResourceExhausted("ggml packed constant metadata");
  auto *packed = ::new (storage) PackedConstant{{}, request.description};
  *output = {packed, +[](void *value, mem::AllocatorRef owner) noexcept {
    auto *constant = static_cast<PackedConstant *>(value);
    constant->~PackedConstant();
    owner.deallocate(constant, sizeof(PackedConstant), alignof(PackedConstant));
  }, +[](void *value) noexcept {
    static_cast<PackedConstant *>(value)->retire();
  }};
  HYLE_ASSIGN_OR_RETURN(auto bytes, packed->description.required_bytes());
  HYLE_ASSIGN_OR_RETURN(packed->storage, mem::Buffer::create(preparation.destination, bytes, 256));
  HYLE_ASSIGN_OR_RETURN(auto target, compute::TensorView::create(
      packed->description, packed->storage.view()));
  if (request.pack.queued) {
    if (!preparation.sequence)
      return Status::FailedPrecondition("ggml packing requires an admitted preparation sequence");
    return request.pack.queued(request.native_context, preparation.source,
        target, allocator, *preparation.sequence, &packed->native);
  }
  return request.pack.synchronous(request.native_context, preparation.source,
      target, allocator);
}
inline compute::ProviderPrepareOutcome
acquire(ConstantArtifactAccess access, ValueId source,
        compute::TensorDesc description, mem::MemoryResourceRef resource,
        std::uint64_t format, Pack pack, void *native_context,
        compute::ConstTensorView *binding,
        ConstantArtifactView *owner = nullptr) noexcept {
  PackingRequest request{native_context, pack, description};
  ConstantArtifactView result;
  const auto bytes = description.required_bytes().value();
  const auto outcome = access.get_or_create(source,
      {.key = {format, 1}, .storage = resource, .packed_bytes = bytes,
       .persistent_bytes = sizeof(PackedConstant),
       // Native cold execution can allocate handles/workspace. The immutable
       // payload is exact; the native preparation peak remains unknown.
       .private_storage = {.resident_bytes = bytes, .peak_bytes = std::nullopt},
       .queue_preparation = pack.queued != nullptr},
      &make_constant, &request, &result);
  if (outcome.kind != compute::ProviderPrepareOutcomeKind::kPrepared)
    return outcome;
  const auto &packed = *static_cast<const PackedConstant *>(result.state);
  auto view = compute::ConstTensorView::create(packed.description, packed.storage.view());
  if (!view)
    return compute::ProviderPrepareOutcome::error(std::move(view).error());
  *binding = std::move(*view);
  if (owner) *owner = result;
  return compute::ProviderPrepareOutcome::prepared();
}
inline Result<compute::TensorDesc>
matrix_description(const compute::TensorDesc &source) noexcept {
  HYLE_ASSIGN_OR_RETURN(auto transposed, source.transposed());
  HYLE_ASSIGN_OR_RETURN(auto physical, compute::TensorDesc::dense(
      source.data_type(), transposed.dimensions()));
  return physical.transposed(); // Same logical [...,K,N], packed [...,N,K].
}

inline bool published_view_of(const Graph &graph, ValueId source) noexcept {
  for (std::size_t port = 0; port < graph.output_count(); ++port) {
    auto value = graph.output_id(port).value();
    for (;;) {
      if (value == source) return true;
      if (value.kind() != ValueKind::kNodeOutput) break;
      const auto producer = graph.producer(value).value();
      if (graph.operation_kind(producer).value() != compute::OperationKind::kView) break;
      value = graph.node_input(producer, 0).value();
    }
  }
  return false;
}

// The exact GLM mHC weight boundary: immutable plain BF16 -> F32. Do not
// interpret a general constant expression or change the following GEMM route.
// Published Cast/View values retain their original writable output contract.
template <class Recipe>
compute::ProviderPrepareOutcome prepare_constant_casts(
    void *context, const RegionCompileRequest &request,
    std::span<const ValueId> values, Recipe &recipe, Pack widen) {
  using Outcome = compute::ProviderPrepareOutcome;
  if (!request.constant_artifacts.available() || !widen) return Outcome::prepared();
  for (auto node = recipe.nodes.begin(); node != recipe.nodes.end();) {
    const auto *cast = node->operation.template get_if<compute::Cast>();
    // Keep one real native operation. Prepared-only singleton regions are a
    // separate execution domain and are deliberately not introduced here.
    if (!cast || recipe.nodes.size() < 2 || node->inputs.size() != 1 ||
        node->outputs.size() != 1 || cast->output_type != compute::DataType::kFloat32) {
      ++node; continue;
    }
    const auto source = node->inputs[0], output = node->outputs[0];
    const auto &from = recipe.values[source];
    const auto &to = recipe.values[output];
    const auto constant = std::ranges::find_if(request.constants,
        [&](const auto &entry) { return entry.value == values[source]; });
    if (constant == request.constants.end() || !from.is_plain() || !from.is_dense() ||
        from.rank() != 2 || from.data_type() != compute::DataType::kBFloat16 ||
        !to.is_plain() || !to.is_dense() ||
        to.data_type() != compute::DataType::kFloat32 ||
        !std::ranges::equal(from.dimensions(), to.dimensions()) ||
        published_view_of(*request.graph, values[output])) {
      ++node; continue;
    }
    const auto input = std::ranges::find(request.inputs, values[source]);
    if (input == request.inputs.end()) { ++node; continue; }
    const auto boundary = std::ranges::find(request.outputs, values[output]);
    const auto resource = request.input_resources[static_cast<std::size_t>(input - request.inputs.begin())];
    if (widen.queued && !request.constant_artifacts.can_prepare_queued(values[source], resource)) {
      ++node; continue;
    }
    if (boundary != request.outputs.end() &&
        request.output_resources[static_cast<std::size_t>(boundary - request.outputs.begin())] != resource) {
      ++node; continue;
    }
    compute::ConstTensorView binding;
    ConstantArtifactView owner;
    const auto outcome = acquire(request.constant_artifacts, values[source], to,
        resource, kBfloatToFloat, widen, context, &binding, &owner);
    if (outcome.kind != compute::ProviderPrepareOutcomeKind::kPrepared) return outcome;
    if (boundary != request.outputs.end()) {
      const auto status = request.constant_artifacts.bind_prepared_output(values[output], owner, binding);
      if (!status.ok()) return Outcome::error(status);
      std::erase(recipe.outputs, output);
    }
    recipe.constants.push_back({output, binding});
    node = recipe.nodes.erase(node);
  }
  return Outcome::prepared();
}
// Both native adapters use the same immutable-only selection/key/ownership
// protocol. A null widening callback leaves that backend's expert path alone.
template <class Recipe>
compute::ProviderPrepareOutcome
pack_constants(void *context, const RegionCompileRequest &request,
               std::span<const ValueId> values, Recipe &recipe,
               Pack widen_bfloat, Pack transpose_matrix,
               std::vector<ValueId> *packed_matrices = nullptr) {
  using Outcome = compute::ProviderPrepareOutcome;
  if (!request.constant_artifacts.available()) return Outcome::prepared();
  std::vector<std::array<std::uint32_t, 2>> replacements(values.size(), {UINT32_MAX, UINT32_MAX});
  for (auto &node : recipe.nodes) {
    const bool expert = widen_bfloat && node.operation.template custom_as<compute::model_ops::GroupedMatMul>().has_value();
    const bool matrix = node.operation.template get_if<compute::MatMul>() != nullptr;
    if (!expert && !matrix) continue;
    const auto port = expert ? 0u : 1u;
    const auto original = node.inputs[port];
    const auto &description = recipe.values[original];
    if (!description.is_plain() || !description.is_dense() ||
        (expert && description.data_type() != compute::DataType::kBFloat16)) continue;
    const auto source = std::ranges::find_if(request.constants,
        [&](const auto &constant) { return constant.value == values[original]; });
    if (source == request.constants.end()) continue; // Dynamic inputs never enter the cache.
    const auto physical = description.transposed();
    if (matrix && (!physical || physical->is_dense())) continue;
    const auto transform = expert ? 0u : 1u;
    if (replacements[original][transform] == UINT32_MAX) {
      const auto input = std::ranges::find(request.inputs, values[original]);
      if (input == request.inputs.end()) return Outcome::error(
          Status::Internal("ggml constant misses region boundary"));
      const auto resource = request.input_resources[
          static_cast<std::size_t>(input - request.inputs.begin())];
      const auto pack = expert ? widen_bfloat : transpose_matrix;
      if (pack.queued && !request.constant_artifacts.can_prepare_queued(
              values[original], resource))
        continue;
      auto packed = expert ? compute::TensorDesc::dense(compute::DataType::kFloat32, description.dimensions())
                           : matrix_description(description);
      if (!packed) return Outcome::error(std::move(packed).error());
      compute::ConstTensorView binding;
      const auto outcome = acquire(request.constant_artifacts, values[original], *packed,
          resource,
          expert ? kBfloatToFloat : kMatrixTranspose,
          expert ? widen_bfloat : transpose_matrix, context, &binding);
      if (outcome.kind != compute::ProviderPrepareOutcomeKind::kPrepared) return outcome;
      const auto id = static_cast<std::uint32_t>(recipe.values.size());
      recipe.values.push_back(*packed);
      recipe.constants.push_back({id, std::move(binding)});
      replacements[original][transform] = id;
      if (matrix && packed_matrices) packed_matrices->push_back(values[original]);
    }
    node.inputs[port] = replacements[original][transform];
  }
  return Outcome::prepared();
}
} // namespace hyle::graph::ggml_shared

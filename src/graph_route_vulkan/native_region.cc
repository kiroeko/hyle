#include "native_region.h"
#include "native_paged_region.h"
#include "native_paged_append.h"
#include "native_ragged_region.h"
#include "native_paged_pending.h"
#include "vulkan_source.h"

#include <algorithm>
#include <mutex>
#include <new>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../graph/native_analysis.h"
#include "../graph/region_discovery.h"
#include "../graph/artifact_packed_region.h"
#include "../graph/static_fp8_region.h"
#include "../compute_route_vulkan/vulkan_carrier.h"
#include "../compute_route_vulkan/vulkan_composition_access.h"
#include "../compute_route_vulkan/vulkan_memory_internal.h"
#include "../compute_route_vulkan/vulkan_program_internal.h"
#include "hyle/compute/detail/vulkan_batch_access.h"
#include "hyle/compute/detail/lease_access.h"
#include "hyle/compute/detail/paged_arena_access.h"
#include "hyle/graph/detail/region_access.h"
#include "../compute_batch/row_cache_planning.h"

namespace hyle::graph::vulkan::detail {
namespace cv = compute::vulkan;
namespace cd = compute::detail;
namespace vn = cv::detail;
namespace bd = batch::detail;
namespace {
template<class T> using Vector = std::vector<T, mem::StdAllocator<T>>;
const unsigned tensor_type = 0;
const unsigned ordered_endpoint = 0;
const unsigned observed_endpoint = 0;
constexpr compute::ProviderRuntimeProperties runtime_properties{
    .mode = compute::ProviderRuntimeMode::kDelegated,
    .workspace_owner = compute::ProviderWorkspaceOwner::kProvider,
    .replay_class = compute::ProviderReplayClass::kManaged};
constexpr std::uint64_t kCtPackedRoute = 4;
constexpr std::uint64_t kStaticFp8Route = 5;
using CtPattern = graph::detail::ArtifactPackedPattern;
using StaticPattern = graph::detail::StaticFloat8Pattern;

template<class Rule>
Result<Vector<typename Rule::Pattern>> packed_patterns(const Graph &graph,
    std::span<const NodeId> nodes, const typename Rule::Uses &uses,
    Vector<std::uint8_t> &allowed, mem::AllocatorRef allocator) noexcept try {
  for (const auto node : nodes) allowed[node.ordinal()] = 1;
  struct Reset final {
    Vector<std::uint8_t> &mask;
    std::span<const NodeId> nodes;
    ~Reset() { for (const auto node : nodes) mask[node.ordinal()] = 0; }
  } reset{allowed, nodes};
  Vector<typename Rule::Pattern> result{mem::StdAllocator<typename Rule::Pattern>(allocator)};
  for (const auto node : nodes) {
    typename Rule::Pattern pattern;
    if (!Rule::pattern(graph, node, pattern) || !Rule::closed(graph, pattern, uses, allowed))
      continue;
    std::array<compute::TensorDesc, Rule::kMaxInputs> inputs;
    for (std::size_t port = 0; port < pattern.input_count; ++port) {
      HYLE_ASSIGN_OR_RETURN(inputs[port], graph.value_description(pattern.inputs[port]));
    }
    std::array<compute::TensorDesc, 1> output;
    const auto status = pattern.descriptor.infer_output_descriptions(
        std::span(inputs).first(pattern.input_count), output);
    if (!status.ok()) {
      if (status.IsInvalidArgument() || status.IsUnimplemented()) continue;
      return std::unexpected(status);
    }
    bool activation_layout = inputs[0].is_dense();
    if constexpr (requires { pattern.descriptor.expert_count; })
      activation_layout = activation_layout || pattern.descriptor.expert_count != 0;
    if (!inputs[0].is_plain() || !activation_layout || !output[0].is_dense() ||
        *inputs[0].element_count() == 0) continue;
    result.push_back(pattern);
  }
  return result;
} catch (const std::bad_alloc &) {
    return std::unexpected(Status::ResourceExhausted("Vulkan packed cold pattern metadata"));
}

const std::shared_ptr<vn::Runtime> &runtime(void *context) noexcept {
  return static_cast<const SourceContext *>(context)->runtime;
}

struct RowAccess final {
  std::uint32_t node = 0;
  std::uint32_t input_port = cd::OperationPreparedAccess::kNoInput;
  vn::RowCacheAccessFacts native;
};
struct RowLayout final {
  bool configured = false;
  bool empty = false;
  std::uint32_t requests = 0, page_refs = 0;
};
struct Artifact final {
  explicit Artifact(mem::AllocatorRef allocator)
      : input_ports(mem::StdAllocator<std::size_t>(allocator)),
        output_resources(mem::StdAllocator<mem::MemoryResourceRef>(allocator)),
        output_contracts(mem::StdAllocator<RegionOutputContract>(allocator)),
        row_accesses(mem::StdAllocator<RowAccess>(allocator)),
        pending_inputs(mem::StdAllocator<ValueId>(allocator)),
        pending_ports(mem::StdAllocator<std::uint32_t>(allocator)),
        pending_accesses(mem::StdAllocator<std::size_t>(allocator)) {}
  cv::Program program;
  cv::InvocationAllocation allocation{};
  Vector<std::size_t> input_ports;
  Vector<mem::MemoryResourceRef> output_resources;
  Vector<RegionOutputContract> output_contracts;
  Vector<RowAccess> row_accesses;
  Vector<ValueId> pending_inputs;
  Vector<std::uint32_t> pending_ports;
  Vector<std::size_t> pending_accesses;
};
struct Slot final {
  explicit Slot(mem::AllocatorRef owner)
      : allocator(owner), inputs(mem::StdAllocator<cv::Tensor>(owner)),
        outputs(mem::StdAllocator<cv::Tensor>(owner)),
        backing(mem::StdAllocator<mem::Buffer>(owner)),
        row_layouts(mem::StdAllocator<RowLayout>(owner)),
        pending_frames(mem::StdAllocator<compute::PagedArenaPendingRead>(owner)),
        pending_reads(owner) {}
  mem::AllocatorRef allocator;
  cv::Invocation invocation;
  Vector<cv::Tensor> inputs, outputs;
  Vector<mem::Buffer> backing;
  Vector<RowLayout> row_layouts;
  Vector<compute::PagedArenaPendingRead> pending_frames;
  NativePagePendingSource *pending_source = nullptr;
  compute::LeaseUsePool pending_uses;
  cd::BatchPageReads pending_reads;
  std::uint32_t pending_capacity = 0;
  bool pending_consumed = false;
  cd::ResultStorageOwner result;
  bool sealed = false;
  bool needs_seal = false;
};
const std::shared_ptr<vn::Runtime> &runtime(const Artifact &artifact) noexcept {
  return vn::Access::program(artifact.program)->runtime;
}

Status qualify(const RegionCompileRequest &request) noexcept {
  if (!request.graph || request.graph->dimension_count() != 0 ||
      (request.route != 1 && request.route != kCtPackedRoute && request.route != kStaticFp8Route) || request.nodes.empty() ||
      request.has_unknown_effects || !request.effect_accesses.empty())
    return Status::Unimplemented("Vulkan native region requires a fixed pure boundary");
  for (const auto node : request.nodes) {
    HYLE_ASSIGN_OR_RETURN(auto operation, request.graph->operation(node));
    const auto traits = compute::operation_traits(operation);
    if (traits.effect != compute::EffectKind::kPure || traits.resource_count != 0 ||
        (traits.alias == compute::AliasPolicy::kMustAliasInput0 &&
         !operation.get_if<compute::View>()))
      return Status::Unimplemented("Vulkan native region has unsupported effects/aliasing");
  }
  return Status::Ok();
}

Status propose(void *context, const RegionDiscoveryRequest &request,
               mem::AllocatorRef allocator, RegionProposalWriter &writer) noexcept try {
  const auto &graph = request.graph;
  if (!request.allowed_nodes.empty() && request.allowed_nodes.size() != graph.node_count())
    return Status::InvalidArgument("Vulkan discovery mask size differs");
  if (graph.dimension_count() || !graph.node_count()) return Status::Ok();
  HYLE_ASSIGN_OR_RETURN(auto analysis,
      graph::detail::analyze_native_reachability(graph, allocator));
  HYLE_RETURN_IF_ERROR(propose_native_paged(context, request, analysis.needed, writer));
  HYLE_RETURN_IF_ERROR(propose_native_append(request, writer));
  HYLE_RETURN_IF_ERROR(propose_native_ragged(request, analysis.needed, writer));
  const auto ordinary_begin = writer.size();
  HYLE_RETURN_IF_ERROR(graph::detail::discover_regions(graph, allocator,
      [&](NodeId node) -> Result<bool> {
        if (!analysis.needed[node.ordinal()] ||
            (!request.allowed_nodes.empty() && !request.allowed_nodes[node.ordinal()]))
          return false;
        HYLE_ASSIGN_OR_RETURN(const auto &operation, graph.operation(node));
        if (native_paged_operation(operation) || native_ragged_operation(operation) ||
            native_append_operation(operation)) return false;
        // Discovery is structural, without per-node prepare/JIT. Actual typed
        // mathematics and device limits remain the final lowerer's admission.
        return graph::detail::validate_native_node(operation).ok();
      },
      [&](std::span<const NodeId> nodes) -> Result<bool> {
        if (nodes.empty()) return true;
        return writer.try_add({.nodes = nodes,
            .cost = {.persistent_bytes = std::nullopt, .copy_bytes = std::nullopt,
                     .repack_bytes = std::nullopt, .synchronization_count = 0, .dispatch_count = 1},
            .route = 1, .replay_class = compute::ProviderReplayClass::kManaged,
            .implementation = compute::ProviderImplementationId{1}, .preference = 200});
      }));
  const auto ordinary_end = writer.size();
  if (!request.allowed_nodes.empty() && !writer.proposal_limit_reached() && !writer.node_limit_reached())
    HYLE_RETURN_IF_ERROR(graph::detail::add_region_singleton_fallbacks(writer, ordinary_begin));
  // The exact same whole-region boundary remains available in its original
  // form. Never turn a contracted-only route into a singleton fallback.
  const auto ordinary = writer.proposals().subspan(ordinary_begin, ordinary_end - ordinary_begin);
  CtPattern::Uses ct_uses(allocator);
  HYLE_RETURN_IF_ERROR(ct_uses.initialize(graph));
  Vector<std::uint8_t> ct_allowed(graph.node_count(), 0,
      mem::StdAllocator<std::uint8_t>(allocator));
  for (const auto &candidate : ordinary) {
    HYLE_ASSIGN_OR_RETURN(auto packed, packed_patterns<CtPattern>(graph, candidate.nodes, ct_uses, ct_allowed, allocator));
    if (packed.empty()) continue;
    auto optimized = candidate;
    optimized.route = kCtPackedRoute;
    optimized.implementation = compute::ProviderImplementationId{2};
    optimized.policy = CtPattern::kPolicy;
    // Keep every supported packed candidate. The current row kernel repeats
    // weight decode across activation rows; paired same-Graph measurements
    // favor it only for the measured one-row BF16 storage boundaries. Other
    // shapes/types retain the mature original route's default preference.
    const bool grouped_space = std::ranges::any_of(packed, [](const auto &pattern) {
      return pattern.descriptor.expert_count != 0;
    });
    bool measured_one_row = true;
    for (const auto &pattern : packed) {
      // GPTQ has a different row/index algorithm and no paired measurement.
      // Its all-M candidate remains available below the mature default rank.
      if (pattern.descriptor.expert_count != 0 ||
          pattern.descriptor.format == compute::model_ops::WeightArtifactFormat::kGptqV1Int4) {
        measured_one_row = false;
        continue;
      }
      const auto activation = graph.value_description(pattern.inputs[0]).value();
      const auto codes = graph.value_description(pattern.inputs[1]).value();
      const auto scales = graph.value_description(pattern.inputs[pattern.input_count - 1]).value();
      measured_one_row = measured_one_row &&
          *activation.element_count() / (std::uint64_t(codes.dimensions()[1]) * 8) == 1 &&
          scales.data_type() == compute::DataType::kBFloat16 &&
          pattern.descriptor.decode_type == compute::DataType::kBFloat16 &&
          pattern.descriptor.operand_type == compute::DataType::kBFloat16;
    }
    // Grouped defaults to the selected packed route for its bounded bank
    // residency. This is a space policy, not a grouped throughput claim.
    optimized.preference = grouped_space || measured_one_row ? 199 : 201;
    HYLE_ASSIGN_OR_RETURN(auto more, writer.try_add(optimized));
    if (!more) break;
  }
  for (const auto &candidate : ordinary) {
    HYLE_ASSIGN_OR_RETURN(auto packed,
        packed_patterns<StaticPattern>(graph, candidate.nodes, ct_uses, ct_allowed, allocator));
    if (packed.empty()) continue;
    auto optimized = candidate;
    optimized.route = kStaticFp8Route;
    optimized.implementation = compute::ProviderImplementationId{3};
    optimized.policy = StaticPattern::kPolicy;
    // Capability does not establish a performance crossover. Keep the
    // original whole-region route until this distinct algorithm is measured.
    optimized.preference = 201;
    HYLE_ASSIGN_OR_RETURN(auto more, writer.try_add(optimized));
    if (!more) break;
  }
  return Status::Ok();
} catch (const std::bad_alloc &) {
  return Status::ResourceExhausted("Vulkan CT discovery metadata");
}

Status resources(void *context, const RegionCompileRequest &request,
    mem::AllocatorRef allocator, std::span<RegionInputRequirement> inputs,
    std::span<RegionOutputRequirement> outputs,
    RegionResourceProposal *proposal) noexcept {
  if (request.route == kNativeAppendRoute)
    return native_append_resources(context, request, inputs, outputs, proposal);
  if (request.route == 2)
    return native_paged_resources(context, request, inputs, outputs, proposal);
  if (request.route == kNativeRaggedRoute)
    return native_ragged_resources(context, request, inputs, outputs, proposal);
  HYLE_RETURN_IF_ERROR(qualify(request));
  if (inputs.size() != request.inputs.size() || outputs.size() != request.outputs.size())
    return Status::InvalidArgument("Vulkan resource boundary arity differs");
  const cd::ResourceBindingRequirement binding{
      .required_properties = mem::MemoryProperty::kDeviceAccessible,
      .forbidden_properties = mem::MemoryProperty::kHostAccessible,
      .minimum_alignment = 1,
      .exact_domain = {runtime(context)->device.get(),
                      static_cast<std::uintptr_t>(cv::MemoryKind::kDevice)},
      .requires_exact_domain = true};
  std::ranges::fill(inputs, RegionInputRequirement{.resource = binding});
  try {
    // The original native plan only aliases logical outputs through View.
    // Captured constants have stable storage; a View rooted in a true boundary
    // input gets its backing when that input is cold-bound.
    std::unordered_set<std::uint32_t, std::hash<std::uint32_t>,
        std::equal_to<std::uint32_t>, mem::StdAllocator<std::uint32_t>> external{
            mem::StdAllocator<std::uint32_t>(allocator)};
    external.reserve(request.inputs.size());
    for (const auto input : request.inputs)
      if (input.kind() != ValueKind::kConstant)
        external.insert(input.ordinal());
    for (const auto node : request.nodes) {
      HYLE_ASSIGN_OR_RETURN(auto operation, request.graph->operation(node));
      if (!operation.get_if<compute::View>()) continue;
      HYLE_ASSIGN_OR_RETURN(auto input, request.graph->node_input(node, 0));
      HYLE_ASSIGN_OR_RETURN(auto output, request.graph->result(node, 0));
      if (external.contains(input.ordinal()))
        external.insert(output.ordinal());
    }
    for (std::size_t port = 0; port < outputs.size(); ++port)
      outputs[port] = {.resource = binding,
          .value = {.kind = external.contains(request.outputs[port].ordinal())
                       ? RegionOutputKind::kBoundGenerated
                       : RegionOutputKind::kStableGenerated,
                    .projection = RegionOutputProjection::kTensorView}};
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("Vulkan output contract metadata");
  }
  *proposal = {.runtime = runtime_properties, .workspace_resource = binding};
  return Status::Ok();
}

Status create_slot(void *prepared, mem::AllocatorRef allocator,
                   mem::BufferView, void **output) noexcept {
  *output = nullptr;
  auto &artifact = *static_cast<Artifact *>(prepared);
  auto *slot = allocator.construct<Slot>(allocator);
  if (!slot) return Status::ResourceExhausted("Vulkan region slot metadata");
  *output = slot; // Destruction is valid on every partially prepared suffix.
  try {
    HYLE_ASSIGN_OR_RETURN(slot->invocation,
        artifact.program.create_invocation(allocator, artifact.allocation));
    bool refresh = false;
    HYLE_RETURN_IF_ERROR(vn::CompositionAccess::prepare_outputs(slot->invocation, &refresh));
    slot->inputs.resize(artifact.input_ports.size());
    slot->row_layouts.resize(artifact.row_accesses.size());
    slot->pending_frames.resize(artifact.pending_inputs.size());
    slot->outputs.reserve(artifact.output_resources.size());
    slot->backing.reserve(artifact.output_resources.size());
    for (std::size_t port = 0; port < artifact.output_resources.size(); ++port) {
      HYLE_ASSIGN_OR_RETURN(auto tensor, vn::CompositionAccess::output(slot->invocation, port));
      HYLE_ASSIGN_OR_RETURN(auto backing, vn::MemoryAccess::export_storage(
          runtime(artifact), artifact.output_resources[port], tensor, allocator));
      slot->outputs.push_back(std::move(tensor));
      slot->backing.push_back(std::move(backing));
    }
    return Status::Ok();
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("Vulkan generated port metadata");
  }
}

Status refresh_outputs(const Artifact &artifact, Slot &slot) noexcept try {
  Vector<cv::Tensor> tensors{mem::StdAllocator<cv::Tensor>(slot.allocator)};
  Vector<mem::Buffer> backing{mem::StdAllocator<mem::Buffer>(slot.allocator)};
  tensors.reserve(slot.outputs.size());
  backing.reserve(slot.backing.size());
  for (std::size_t port = 0; port < slot.outputs.size(); ++port) {
    HYLE_ASSIGN_OR_RETURN(auto tensor, vn::CompositionAccess::output(slot.invocation, port));
    HYLE_ASSIGN_OR_RETURN(auto storage, vn::MemoryAccess::export_storage(
        runtime(artifact), artifact.output_resources[port], tensor, slot.allocator));
    tensors.push_back(std::move(tensor));
    backing.push_back(std::move(storage));
  }
  // Port object addresses are part of the native projection. Replace only
  // their owners, after every fallible export has succeeded; never swap the
  // Slot vectors or mutate a Buffer named by an older readonly lease.
  for (std::size_t port = 0; port < slot.outputs.size(); ++port) {
    slot.outputs[port] = std::move(tensors[port]);
    slot.backing[port] = std::move(backing[port]);
  }
  return Status::Ok();
} catch (const std::bad_alloc &) {
  return Status::ResourceExhausted("Vulkan retained output projection metadata");
}

Status bind(void *prepared, void *invocation, std::span<const RegionValue> inputs,
            std::span<const compute::TensorView> outputs) noexcept {
  auto &artifact = *static_cast<Artifact *>(prepared);
  auto &slot = *static_cast<Slot *>(invocation);
  if (outputs.size() != artifact.output_resources.size())
    return Status::InvalidArgument("Vulkan generated output arity differs");
  if (slot.pending_consumed)
    return Status::FailedPrecondition("reset the consumed Vulkan row pending episode before binding");
  // Graph's idle/no-old-result gate precedes this callback. Independent
  // readonly storage owners keep their old whole block; a cold rearm may
  // select a distinct backing without replacing the Run or its endpoint.
  slot.result.reset();
  if (slot.sealed) {
    bool refresh = false;
    HYLE_RETURN_IF_ERROR(vn::CompositionAccess::prepare_outputs(slot.invocation, &refresh));
    if (refresh) HYLE_RETURN_IF_ERROR(refresh_outputs(artifact, slot));
  }
  for (std::size_t port = 0; port < artifact.input_ports.size(); ++port) {
    const auto index = artifact.input_ports[port];
    if (index >= inputs.size())
      return Status::InvalidArgument("Vulkan region input arity differs");
    const auto &input = inputs[index];
    for (std::size_t bank = 0; bank < artifact.pending_ports.size(); ++bank) {
      if (artifact.pending_ports[bank] != index || !slot.pending_frames[bank].valid())
        continue;
      HYLE_ASSIGN_OR_RETURN(auto frame, cd::PagedArenaAccess::tensor(slot.pending_frames[bank]));
      const auto expected = frame.storage(), actual = input.tensor.storage();
      if (!input.tensor.valid() || expected.resource() != actual.resource() ||
          expected.native_handle() != actual.native_handle() ||
          expected.offset() != actual.offset() || expected.size() != actual.size())
        return Status::InvalidArgument("Vulkan row input differs from its exact pending frame");
    }
    if (input.native_type == &tensor_type && input.native_handle != nullptr) {
      slot.inputs[port] = *static_cast<const cv::Tensor *>(input.native_handle);
    } else {
      HYLE_ASSIGN_OR_RETURN(slot.inputs[port], vn::MemoryAccess::tensor_view(
          runtime(artifact), input.tensor, slot.allocator, artifact.allocation.resource_budget));
    }
    if (slot.inputs[port].description() != input.description)
      return Status::InvalidArgument("Vulkan input physical descriptor differs");
  }
  const auto status = slot.invocation.bind(std::span<const cv::Tensor>(slot.inputs));
  if (status.ok()) slot.needs_seal = true;
  return status;
}

Status seal(void *, void *invocation) noexcept {
  auto &slot = *static_cast<Slot *>(invocation);
  if (slot.pending_consumed)
    return Status::FailedPrecondition("reset the consumed Vulkan row pending episode before replay preparation");
  if (slot.sealed && !slot.needs_seal) {
    slot.result.reset();
    return vn::PagedBatchAccess::reuse_gate(slot.invocation);
  }
  const auto status = slot.sealed
      ? vn::PagedBatchAccess::reseal_composition(slot.invocation)
      : vn::CompositionAccess::seal(slot.invocation);
  if (status.ok()) {
    slot.sealed = true;
    slot.needs_seal = false;
  }
  return status;
}

Result<RegionValue> value(void *prepared, void *invocation, std::size_t port) noexcept {
  const auto &artifact = *static_cast<Artifact *>(prepared);
  auto &slot = *static_cast<Slot *>(invocation);
  if (port >= slot.outputs.size())
    return std::unexpected(Status::OutOfRange("Vulkan generated output port"));
  const auto &tensor = slot.outputs[port];
  if (tensor.native_buffer() == VK_NULL_HANDLE)
    return std::unexpected(Status::FailedPrecondition("Vulkan alias output is not cold-bound"));
  HYLE_ASSIGN_OR_RETURN(auto binding, compute::ConstTensorView::create(
      tensor.description(), slot.backing[port].view()));
  if (binding.storage().resource() != artifact.output_resources[port])
    return std::unexpected(Status::Internal("Vulkan generated output resource changed"));
  return RegionValue{tensor.description(), binding, &tensor_type, &tensor};
}

Result<compute::ReadLease> lease(void *prepared, void *invocation, std::size_t port) noexcept {
  const auto &artifact = *static_cast<Artifact *>(prepared);
  auto &slot = *static_cast<Slot *>(invocation);
  if (!slot.result || port >= slot.outputs.size())
    return std::unexpected(Status::FailedPrecondition("Vulkan output has no successful result owner"));
  const auto &run = vn::Access::invocation_state(slot.invocation)->run;
  {
    std::lock_guard lock(runtime(artifact)->mutex);
    if (run->pending || run->retiring || !run->terminal.ok() ||
        !run->observation_error.ok() || !run->submission_failure.ok())
      return std::unexpected(Status::FailedPrecondition("Vulkan output has not retired successfully"));
  }
  HYLE_ASSIGN_OR_RETURN(auto backing, vn::MemoryAccess::export_storage(
      runtime(artifact), artifact.output_resources[port], slot.outputs[port], slot.allocator));
  HYLE_ASSIGN_OR_RETURN(auto result,
      compute::ReadLease::adopt(std::move(backing), {.allocator = slot.allocator}));
  const auto &state = *vn::Access::invocation_state(slot.invocation);
  const auto &program = *state.program;
  const auto output = program.outputs[port];
  const auto alias = program.aliases[output];
  const auto block = program.storage_blocks[alias.root];
  // Only the actual current Program/Run may nominate this storage root.
  // Constants/input imports keep their existing conservative domain identity.
  if (block != UINT32_MAX && !program.values[alias.root].constant &&
      std::ranges::find(program.inputs, alias.root) == program.inputs.end()) {
    const auto &root = state.run->retained_buffers[alias.root];
    if (block >= program.block_bytes.size() || !root || root->bytes != program.block_bytes[block])
      return std::unexpected(Status::Internal("Vulkan retained output storage block differs"));
    HYLE_ASSIGN_OR_RETURN(auto facts, vn::MemoryAccess::retained_backing(runtime(artifact),
        artifact.output_resources[port], slot.outputs[port], root, alias.byte_offset));
    HYLE_RETURN_IF_ERROR(cd::LeaseAccess::set_retained_backing(result,
        artifact.output_resources[port], facts.authority, facts.canonical_owner,
        facts.full_bytes, facts.backing_offset));
  }
  return result;
}

cd::CompletionEndpoint endpoint(void *, void *invocation) noexcept {
  const auto &slot = *static_cast<const Slot *>(invocation);
  const auto &state = *vn::Access::invocation_state(slot.invocation);
  if (state.run->async || state.run->needs_drain ||
      std::ranges::any_of(state.run->stages, [](const auto &stage) { return bool(stage.iree); }))
    return {&observed_endpoint, state.run.get()};
  return region_ordered_endpoint(state.program->runtime);
}
bool imports(void *prepared, void *invocation, cd::CompletionEndpoint source) noexcept {
  const auto &slot = *static_cast<const Slot *>(invocation);
  const auto &run = *vn::Access::invocation_state(slot.invocation)->run;
  if (run.async || run.needs_drain ||
      std::ranges::any_of(run.stages, [](const auto &stage) { return bool(stage.iree); }))
    return false;
  return imports_region_ordered_endpoint(
      runtime(*static_cast<Artifact *>(prepared)), source);
}
Result<compute::Completion> submit(void *prepared, void *invocation,
    std::span<const cd::CompletionDependency> dependencies,
    std::span<const compute::ConstTensorView>,
    std::span<const compute::TensorView>) noexcept {
  const auto &artifact = *static_cast<Artifact *>(prepared);
  auto &slot = *static_cast<Slot *>(invocation);
  for (const auto dependency : dependencies)
    HYLE_RETURN_IF_ERROR(vn::qualify_copy_dependency(runtime(artifact), dependency));
  slot.result.reset(); // Never pre-retain the original storage before begin().
  const bool pending = std::ranges::any_of(slot.pending_frames,
      [](const auto &frame) { return frame.valid(); });
  auto submitted = [&]() -> Result<compute::OwnedResult<cv::Results>> {
    if (!pending) return slot.invocation.submit();
    if (slot.pending_consumed || !slot.pending_source || !slot.pending_uses.valid())
      return std::unexpected(Status::FailedPrecondition("Vulkan row pending source/episode is unavailable"));
    HYLE_RETURN_IF_ERROR(vn::PagedBatchAccess::reuse_gate(slot.invocation));
    const auto &update = slot.pending_source->source.update();
    HYLE_RETURN_IF_ERROR(qualify_native_pending_source(*slot.pending_source, update));
    HYLE_ASSIGN_OR_RETURN(auto uses, slot.pending_uses.begin());
    slot.pending_reads.reset();
    for (const auto &frame : slot.pending_frames)
      if (frame.valid())
        HYLE_RETURN_IF_ERROR((cd::PagedArenaAccess::collect_reads<cv::CopyBatchUpdate,
            vn::CopyStreamAccess>(slot.pending_reads, frame, update)));
    HYLE_RETURN_IF_ERROR(slot.pending_reads.admit(uses));
    return vn::PagedBatchAccess::submit(slot.invocation, std::move(uses));
  }();
  if (!submitted) return std::unexpected(submitted.error());
  if (pending) slot.pending_consumed = true;
  auto accepted = std::move(*submitted);
  auto transfer = cd::ResultAccess::take_storage(std::move(accepted));
  slot.result = std::move(transfer.owner);
  return std::move(transfer.completion);
}

extern const ProviderRegionPreparedVTable prepared_table;

bool empty_safe_row_reader(const RowAccess &access) noexcept {
  return access.native.kind == vn::RowCacheAccessKind::kPaddedCheckedDecode ||
         access.native.kind == vn::RowCacheAccessKind::kPagedIndexerScores;
}
bool empty_row_layout(const Artifact &artifact, const Slot *slot,
                      std::size_t index) noexcept {
  return slot && index < slot->row_layouts.size() &&
      empty_safe_row_reader(artifact.row_accesses[index]) &&
      slot->row_layouts[index].configured && slot->row_layouts[index].empty;
}
Status inspect_row_port(const ProviderRegionPrepared &prepared, void *invocation,
    std::uint32_t port, cd::OperationPreparedAccess *output) noexcept {
  if (prepared.vtable != &prepared_table || !prepared.state || !output)
    return Status::InvalidArgument("Vulkan projected row artifact differs");
  const auto &artifact = *static_cast<const Artifact *>(prepared.state);
  for (std::size_t i = 0; i < artifact.row_accesses.size(); ++i) {
    const auto &access = artifact.row_accesses[i];
    if (access.input_port == cd::OperationPreparedAccess::kNoInput ||
        access.input_port != port)
      continue;
    *output = {.projected_input = port,
        .allows_empty_footprint = empty_row_layout(
            artifact, static_cast<const Slot *>(invocation), i)};
    return Status::Ok();
  }
  return Status::Unimplemented("Vulkan port has no checked projected row reader");
}
Status inspect_row_node(const ProviderRegionPrepared &prepared, void *invocation,
    std::uint32_t node, compute::ProviderExtensionId capability,
    graph::detail::RegionNodeAccess *output) noexcept {
  if (prepared.vtable != &prepared_table || !prepared.state || !output)
    return Status::InvalidArgument("Vulkan projected row artifact differs");
  if (capability != cd::kRowCacheColdPlanningId)
    return Status::Unimplemented("Vulkan logical access capability differs");
  const auto &artifact = *static_cast<const Artifact *>(prepared.state);
  for (std::size_t i = 0; i < artifact.row_accesses.size(); ++i) {
    const auto &access = artifact.row_accesses[i];
    if (access.node != node) continue;
    *output = {.access = {.projected_input = access.input_port,
        .allows_empty_footprint = empty_row_layout(
            artifact, static_cast<const Slot *>(invocation), i),
        .requires_configuration = true},
        .allows_pending = access.input_port != cd::OperationPreparedAccess::kNoInput &&
            std::ranges::find(artifact.pending_ports, access.input_port) != artifact.pending_ports.end()};
    return Status::Ok();
  }
  return Status::Unimplemented("Vulkan node has no selected row access proof");
}
Result<bool> configure_row_node(const ProviderRegionPrepared &prepared,
    void *invocation, std::uint32_t node, compute::ProviderExtensionId capability,
    const void *configuration) noexcept {
  if (prepared.vtable != &prepared_table || !prepared.state || !invocation ||
      !configuration)
    return std::unexpected(Status::InvalidArgument("Vulkan row configuration is incomplete"));
  if (capability != cd::kRowCacheColdPlanningId)
    return std::unexpected(Status::Unimplemented("Vulkan row configuration capability differs"));
  const auto &artifact = *static_cast<const Artifact *>(prepared.state);
  auto &slot = *static_cast<Slot *>(invocation);
  if (slot.row_layouts.size() != artifact.row_accesses.size() ||
      vn::Access::invocation_state(slot.invocation)->program.get() !=
          vn::Access::program(artifact.program).get())
    return std::unexpected(Status::InvalidArgument("Vulkan row slot differs from its artifact"));
  if (std::ranges::any_of(slot.pending_frames, [](const auto &frame) { return frame.valid(); }))
    return std::unexpected(Status::FailedPrecondition("reset Vulkan pending row frames before changing their layout"));
  const auto position = std::ranges::find(artifact.row_accesses, node, &RowAccess::node);
  if (position == artifact.row_accesses.end())
    return std::unexpected(Status::Unimplemented("Vulkan node has no selected row configuration"));
  if (slot.sealed) {
    slot.result.reset();
    HYLE_RETURN_IF_ERROR(vn::PagedBatchAccess::reuse_gate(slot.invocation));
  }
  auto &layout = slot.row_layouts[static_cast<std::size_t>(position - artifact.row_accesses.begin())];
  layout = {};
  const auto &fact = position->native;
  const auto &config = *static_cast<const cd::RowCacheColdLayout *>(configuration);
  const bool mapping = fact.kind == vn::RowCacheAccessKind::kMap ||
                       fact.kind == vn::RowCacheAccessKind::kPaddedMap ||
                       fact.kind == vn::RowCacheAccessKind::kPagedIndexerScores;
  if (config.query_indptr.size() <= config.requests ||
      config.kv_indptr.size() <= config.requests ||
      config.last_page_lengths.size() < config.requests ||
      config.query_indptr[0] != 0 || config.kv_indptr[0] != 0 ||
      config.query_indptr[config.requests] != std::int64_t(fact.query_rows) ||
      config.kv_indptr[config.requests] < 0 ||
      (mapping && (config.requests > fact.max_requests ||
                   std::uint32_t(config.kv_indptr[config.requests]) > fact.max_refs)))
    return std::unexpected(Status::InvalidArgument("Vulkan row cold layout extent differs"));
  bool empty = true;
  for (std::uint32_t request = 0; request < config.requests; ++request) {
    const auto begin = config.query_indptr[request], end = config.query_indptr[request + 1];
    const auto first = config.kv_indptr[request], last = config.kv_indptr[request + 1];
    const auto tail = config.last_page_lengths[request];
    if (begin < 0 || end < begin || std::uint64_t(end) > fact.query_rows ||
        first < 0 || last < first ||
        !((last == first && tail == 0) ||
          (last > first && tail > 0 && (!mapping || std::uint32_t(tail) <= fact.page_size))))
      return std::unexpected(Status::InvalidArgument("Vulkan row cold request span differs"));
    if (fact.kind == vn::RowCacheAccessKind::kPagedIndexerScores && last > first &&
        (std::uint64_t(last - first - 1) * fact.page_size + std::uint32_t(tail) >
         fact.score_columns))
      return std::unexpected(Status::OutOfRange("Vulkan row scorer history exceeds its logical columns"));
    empty = empty && last == first;
  }
  // Only copied facts survive this call. The original GPU metadata remains
  // the sole runtime map input; no caller layout span or alternate map is kept.
  layout = {.configured = true, .empty = empty,
      .requests = config.requests,
      .page_refs = static_cast<std::uint32_t>(config.kv_indptr[config.requests])};
  return true;
}
const graph::detail::RegionProjectedReadVTable row_projected_table{
    .inspect = &inspect_row_port, .inspect_node = &inspect_row_node,
    .configure_node = &configure_row_node};

Status reserve_pending_reads(const Artifact &artifact, Slot &slot) noexcept {
  std::uint64_t resources = 0;
  for (const auto access : artifact.pending_accesses)
    resources += slot.row_layouts[access].page_refs;
  if (resources > UINT32_MAX)
    return Status::ResourceExhausted("Vulkan pending row page guard capacity overflows");
  const auto capacity = static_cast<std::uint32_t>(std::max(std::uint64_t{1}, resources));
  if (slot.pending_uses.valid() && capacity <= slot.pending_capacity)
    return Status::Ok();
  if (slot.pending_uses.active_batches())
    return Status::Unavailable("Vulkan pending row guards have not retired");
  HYLE_ASSIGN_OR_RETURN(auto uses, compute::LeaseUsePool::create({
      .max_batches = 1, .max_resources_per_batch = capacity,
      .allocator = slot.allocator, .resource_budget = artifact.allocation.resource_budget}));
  HYLE_RETURN_IF_ERROR(slot.pending_reads.reserve(capacity));
  slot.pending_uses = std::move(uses);
  slot.pending_capacity = capacity;
  return Status::Ok();
}
const bd::NativePendingRegionVTable row_pending_table{
    .inspect = [](void *prepared, void *) noexcept -> Result<bd::NativePendingRegionFacts> {
      const auto &artifact = *static_cast<const Artifact *>(prepared);
      return bd::NativePendingRegionFacts{artifact.pending_inputs, artifact.pending_ports,
          {&native_pending_source_table, runtime(artifact).get(),
              [](void *opaque, std::size_t capacity, mem::AllocatorRef allocator,
                  compute::ResourceBudget budget) noexcept -> Result<void *> {
                return create_native_pending_source(runtime(*static_cast<Artifact *>(opaque)),
                    capacity, allocator, std::move(budget));
              }}};
    },
    .validate_frame = [](void *prepared, void *invocation, ValueId value,
        const compute::PagedArenaPendingRead &frame,
        const bd::NativePendingSourceOwner &owner) noexcept -> Status {
      const auto &artifact = *static_cast<const Artifact *>(prepared);
      auto &slot = *static_cast<Slot *>(invocation);
      HYLE_ASSIGN_OR_RETURN(auto source, native_pending_source(owner, runtime(artifact)));
      if (slot.pending_source && slot.pending_source != source)
        return Status::InvalidArgument("Vulkan row pending source differs within this episode");
      const auto position = std::ranges::find(artifact.pending_inputs, value);
      if (position == artifact.pending_inputs.end() || !frame.valid() || slot.pending_consumed)
        return Status::InvalidArgument("Vulkan row pending input/frame/episode differs");
      const auto bank = static_cast<std::size_t>(position - artifact.pending_inputs.begin());
      if (slot.pending_frames[bank].valid())
        return Status::FailedPrecondition("reset Vulkan pending row input before rebinding");
      const auto access = artifact.pending_accesses[bank];
      const auto &layout = slot.row_layouts[access];
      const auto facts = frame.layout().facts();
      if (!layout.configured || facts.query_rows != artifact.row_accesses[access].native.query_rows ||
          facts.requests != layout.requests || facts.page_refs != layout.page_refs)
        return Status::InvalidArgument("Vulkan row pending frame differs from its cold layout");
      HYLE_ASSIGN_OR_RETURN(auto tensor, cd::PagedArenaAccess::tensor(frame));
      HYLE_RETURN_IF_ERROR(vn::MemoryAccess::validate_resource(runtime(artifact), tensor.storage().resource()));
      if (tensor.storage().resource().properties().has(mem::MemoryProperty::kHostAccessible))
        return Status::InvalidArgument("Vulkan pending row bank requires actual device storage");
      const auto &program = *vn::Access::program(artifact.program);
      const auto &description = program.values[artifact.row_accesses[access].native.bank_value].description;
      // The typed arena exposes [pages, page_size, row_bytes]; the actual
      // decoder sees its original dense [physical_rows, row_bytes] carrier.
      // This is only a descriptor projection of the exact same byte range.
      HYLE_ASSIGN_OR_RETURN(auto frame_bytes, tensor.description().required_bytes());
      HYLE_ASSIGN_OR_RETURN(auto bank_bytes, description.required_bytes());
      if (!tensor.description().is_plain() || !tensor.description().is_dense() ||
          tensor.description().data_type() != compute::DataType::kUInt8 ||
          tensor.description().rank() != 3 || description.rank() != 2 ||
          tensor.description().dimensions()[2] != description.dimensions()[1] ||
          frame_bytes != bank_bytes)
        return Status::InvalidArgument("Vulkan pending row physical bank description differs");
      return reserve_pending_reads(artifact, slot);
    },
    .commit_frame = [](void *prepared, void *invocation, ValueId value,
        compute::PagedArenaPendingRead frame, const bd::NativePendingSourceOwner &owner) noexcept {
      const auto &artifact = *static_cast<const Artifact *>(prepared);
      auto &slot = *static_cast<Slot *>(invocation);
      const auto source = native_pending_source(owner, runtime(artifact));
      const auto position = std::ranges::find(artifact.pending_inputs, value);
      HYLE_FATAL_ASSERT(source.has_value() && position != artifact.pending_inputs.end());
      slot.pending_source = *source;
      slot.pending_frames[static_cast<std::size_t>(position - artifact.pending_inputs.begin())] = std::move(frame);
    },
    .reset = [](void *, void *invocation) noexcept {
      auto &slot = *static_cast<Slot *>(invocation);
      for (auto &frame : slot.pending_frames) frame = {};
      slot.pending_source = nullptr;
      slot.pending_consumed = false;
      slot.pending_reads.reset();
    }};

const graph::detail::RegionBindingReleaseVTable binding_release_table{
    .validate = [](void *, void *invocation) noexcept -> Status {
      const auto &slot = *static_cast<const Slot *>(invocation);
      if (!slot.invocation.valid())
        return slot.result ? Status::FailedPrecondition("native result has no invocation") : Status::Ok();
      return vn::PagedBatchAccess::can_detach_inputs(slot.invocation, slot.result);
    },
    .release = [](void *, void *invocation) noexcept {
      auto &slot = *static_cast<Slot *>(invocation);
      slot.result.reset();
      if (slot.invocation.valid()) vn::PagedBatchAccess::detach_inputs(slot.invocation);
      for (auto &input : slot.inputs) input = {};
      slot.needs_seal = false;
    }};

const ProviderRegionPreparedVTable prepared_table{
    .destroy_prepared = [](void *state, mem::AllocatorRef allocator) noexcept {
      allocator.destroy(static_cast<Artifact *>(state));
    },
    .alias_support = [](void *, std::size_t, std::size_t) noexcept {
      return compute::AliasSupport{};
    },
    .create_invocation = &create_slot,
    .destroy_invocation = [](void *state, mem::AllocatorRef allocator) noexcept {
      allocator.destroy(static_cast<Slot *>(state));
    },
    .delegated_submit = &submit, .endpoint = &endpoint,
    .can_import_endpoint = &imports,
    .output_contract = [](void *prepared, std::size_t port) noexcept {
      return static_cast<Artifact *>(prepared)->output_contracts[port];
    },
    .output_value = &value, .prepare_bindings = &bind,
    .seal_bindings = &seal, .output_lease = &lease,
    .query_extension = [](void *prepared, compute::ProviderExtensionId id) noexcept -> const void * {
      if (id == graph::detail::kRegionBindingReleaseExtensionId) return &binding_release_table;
      if (id == graph::detail::kRegionProjectedReadExtensionId) return &row_projected_table;
      if (id == bd::kNativePendingRegionExtensionId &&
          !static_cast<const Artifact *>(prepared)->pending_inputs.empty()) return &row_pending_table;
      return nullptr;
    }};

compute::ProviderPrepareOutcome compile(void *context,
    const RegionCompileRequest &request, mem::AllocatorRef allocator,
    ProviderRegionPrepared *result) noexcept {
  using Outcome = compute::ProviderPrepareOutcome;
  *result = {};
  if (request.route == kNativeAppendRoute)
    return compile_native_append(context, request, allocator, result);
  if (request.route == 2)
    return compile_native_paged(context, request, allocator, result);
  if (request.route == kNativeRaggedRoute)
    return compile_native_ragged(context, request, allocator, result);
  auto status = qualify(request);
  if (!status.ok()) return Outcome::unsupported(std::move(status));
  auto *artifact = allocator.construct<Artifact>(allocator);
  if (!artifact) return Outcome::error(Status::ResourceExhausted("Vulkan region artifact metadata"));
  result->vtable = &prepared_table;
  result->state = artifact;
  try {
    const auto build = [&]() -> Status {
      const auto &graph = *request.graph;
      HYLE_ASSIGN_OR_RETURN(auto analysis, graph::detail::analyze_native_fragment(graph, request.nodes, allocator));
      vn::Plan plan;
      plan.workload = request.workload;
      Vector<std::uint32_t> ids(graph.value_count(), UINT32_MAX,
                                mem::StdAllocator<std::uint32_t>(allocator));
      Vector<CtPattern::Pattern> packed{mem::StdAllocator<CtPattern::Pattern>(allocator)};
      Vector<StaticPattern::Pattern> static_packed{mem::StdAllocator<StaticPattern::Pattern>(allocator)};
      if (request.route == kCtPackedRoute) {
        CtPattern::Uses ct_uses(allocator);
        HYLE_RETURN_IF_ERROR(ct_uses.initialize(graph));
        Vector<std::uint8_t> ct_allowed(graph.node_count(), 0,
            mem::StdAllocator<std::uint8_t>(allocator));
        HYLE_ASSIGN_OR_RETURN(packed, packed_patterns<CtPattern>(graph, request.nodes, ct_uses, ct_allowed, allocator));
        if (packed.empty()) return Status::Unimplemented("Vulkan CT route has no closed public recipe");
      }
      if (request.route == kStaticFp8Route) {
        StaticPattern::Uses uses(allocator);
        HYLE_RETURN_IF_ERROR(uses.initialize(graph));
        Vector<std::uint8_t> allowed(graph.node_count(), 0, mem::StdAllocator<std::uint8_t>(allocator));
        HYLE_ASSIGN_OR_RETURN(static_packed,
            packed_patterns<StaticPattern>(graph, request.nodes, uses, allowed, allocator));
        if (static_packed.empty()) return Status::Unimplemented("Vulkan static FP8 route has no closed public recipe");
      }
      Vector<std::uint32_t> step_nodes{mem::StdAllocator<std::uint32_t>(allocator)};
      const auto add = [&](ValueId value) -> Result<std::uint32_t> {
        auto &index = ids[value.ordinal()];
        if (index == UINT32_MAX) {
          HYLE_ASSIGN_OR_RETURN(auto description, graph.value_description(value));
          index = static_cast<std::uint32_t>(plan.values.size());
          plan.values.push_back({description, {}});
        }
        return index;
      };
      for (std::size_t port = 0; port < request.inputs.size(); ++port) {
        const auto input = request.inputs[port];
        HYLE_ASSIGN_OR_RETURN(auto index, add(input));
        const auto constant = std::ranges::find_if(request.constants,
            [&](const ConstantBinding &entry) { return entry.value == input; });
        if (constant != request.constants.end()) {
          HYLE_ASSIGN_OR_RETURN(plan.values[index].constant,
              vn::MemoryAccess::tensor_view(runtime(context), constant->tensor, allocator,
                  request.resource_budget));
        } else {
          plan.inputs.push_back(index);
          artifact->input_ports.push_back(port);
        }
      }
      for (const auto node : request.nodes) {
        const CtPattern::Pattern *contraction = nullptr;
        const StaticPattern::Pattern *static_contraction = nullptr;
        bool internal = false;
        for (const auto &pattern : packed) {
          if (!pattern.contains(node)) continue;
          if (CtPattern::producer(graph, pattern.output) == node) contraction = &pattern;
          else internal = true;
          break;
        }
        for (const auto &pattern : static_packed) {
          if (!pattern.contains(node)) continue;
          if (StaticPattern::producer(graph, pattern.output) == node) static_contraction = &pattern;
          else internal = true;
          break;
        }
        if (internal) continue;
        vn::PlanStep step;
        if (contraction) step.operation = contraction->composite();
        else if (static_contraction) step.operation = static_contraction->composite();
        else { HYLE_ASSIGN_OR_RETURN(step.operation, graph.operation(node)); }
        step.control_boundary = analysis.boundary[node.ordinal()] != 0;
        step.preserve_outputs = analysis.control_predecessor[node.ordinal()] != 0;
        const auto input_count = static_contraction ? Result<std::size_t>(static_contraction->input_count)
                                            : contraction ? Result<std::size_t>(contraction->input_count)
                                            : graph.node_input_count(node);
        HYLE_ASSIGN_OR_RETURN(auto count, input_count);
        for (std::size_t port = 0; port < count; ++port) {
          const auto original_input = static_contraction ? Result<ValueId>(static_contraction->inputs[port])
                                                 : contraction ? Result<ValueId>(contraction->inputs[port])
                                                 : graph.node_input(node, port);
          HYLE_ASSIGN_OR_RETURN(auto input, original_input);
          HYLE_ASSIGN_OR_RETURN(auto index, add(input));
          step.inputs.push_back(index);
        }
        HYLE_ASSIGN_OR_RETURN(count, graph.node_output_count(node));
        for (std::size_t port = 0; port < count; ++port) {
          HYLE_ASSIGN_OR_RETURN(auto output, graph.result(node, port));
          HYLE_ASSIGN_OR_RETURN(auto index, add(output));
          step.outputs.push_back(index);
        }
        plan.steps.push_back(std::move(step));
        step_nodes.push_back(node.ordinal());
      }
      for (const auto output : request.outputs) plan.outputs.push_back(ids[output.ordinal()]);
      // A control-only fragment still evaluates real terminal arrays and its
      // checked algorithm status. These outputs never become Graph ports.
      if (plan.outputs.empty()) {
        const auto &terminal = plan.steps.back().outputs;
        if (terminal.empty())
          return Status::Unimplemented("Vulkan control fragment has no native terminal result");
        plan.outputs.assign(terminal.begin(), terminal.end());
      }
      vn::PlanFacts facts;
      HYLE_ASSIGN_OR_RETURN(artifact->program, vn::prepare_plan(runtime(context), plan, &facts,
          custom_resolver(static_cast<const SourceContext *>(context)->custom)));
      artifact->output_resources.assign(request.output_resources.begin(), request.output_resources.end());
      const auto &native = *vn::Access::program(artifact->program);
      for (const auto &fact : facts.row_cache_accesses) {
        if (fact.step >= step_nodes.size())
          return Status::Internal("Vulkan row lowering step exceeds selected region");
        RowAccess access{.node = step_nodes[fact.step], .native = fact};
        if (fact.bank_value != UINT32_MAX) {
          const auto input = std::ranges::find(native.inputs, fact.bank_value);
          if (input == native.inputs.end())
            return Status::Internal("Vulkan projected bank is not a native boundary input");
          const auto native_port = static_cast<std::size_t>(input - native.inputs.begin());
          if (native_port >= artifact->input_ports.size() ||
              artifact->input_ports[native_port] > UINT32_MAX)
            return Status::Internal("Vulkan projected bank boundary port differs");
          access.input_port = static_cast<std::uint32_t>(artifact->input_ports[native_port]);
        }
        if (access.input_port != cd::OperationPreparedAccess::kNoInput &&
            request.inputs[access.input_port].kind() == ValueKind::kInput &&
            std::ranges::find(artifact->pending_inputs, request.inputs[access.input_port]) == artifact->pending_inputs.end()) {
          artifact->pending_inputs.push_back(request.inputs[access.input_port]);
          artifact->pending_ports.push_back(access.input_port);
          artifact->pending_accesses.push_back(artifact->row_accesses.size());
        }
        artifact->row_accesses.push_back(access);
      }
      if (request.resource_budget) {
        artifact->allocation.resource_budget = request.resource_budget;
        const auto matches = [&](mem::MemoryResourceRef resource, bool staging) {
          return resource.valid() &&
              resource.properties().has(mem::MemoryProperty::kHostAccessible) == staging &&
              vn::MemoryAccess::validate_resource(runtime(context), resource).ok();
        };
        if (matches(request.workspace_resource, false))
          artifact->allocation.device = request.workspace_resource;
        for (const auto &domain : request.available_storage_domains) {
          if (!artifact->allocation.device.valid() && matches(domain.resource, false))
            artifact->allocation.device = domain.resource;
          if (!artifact->allocation.staging.valid() && matches(domain.resource, true))
            artifact->allocation.staging = domain.resource;
        }
        if (!artifact->allocation.device.valid() ||
            (vn::requires_staging(native) && !artifact->allocation.staging.valid()))
          return Status::InvalidArgument(
              "Vulkan native allocation budget requires its actual device/status resources");
      }
      for (std::size_t port = 0; port < request.outputs.size(); ++port) {
        const auto output = native.outputs[port];
        const auto root = native.aliases[output].root;
        const bool external = std::ranges::find(native.inputs, root) != native.inputs.end();
        artifact->output_contracts.push_back({.kind = external
            ? RegionOutputKind::kBoundGenerated : RegionOutputKind::kStableGenerated,
            .projection = RegionOutputProjection::kTensorView});
      }
      result->properties = {.persistent_bytes = std::nullopt,
          .runtime = runtime_properties, .invocation_private_bytes = std::nullopt};
      return Status::Ok();
    };
    status = build();
    if (!status.ok()) {
      if (status.IsUnimplemented()) return Outcome::unsupported(std::move(status));
      return Outcome::error(std::move(status));
    }
    return Outcome::prepared();
  } catch (const std::bad_alloc &) {
    return Outcome::error(Status::ResourceExhausted("Vulkan region preparation metadata"));
  }
}
const RegionProviderVTable region_table{&propose, &resources, &compile};
} // namespace

cd::CompletionEndpoint region_ordered_endpoint(
    const std::shared_ptr<vn::Runtime> &owner) noexcept {
  return {&ordered_endpoint, owner.get()};
}
bool imports_region_ordered_endpoint(const std::shared_ptr<vn::Runtime> &owner,
                                     cd::CompletionEndpoint source) noexcept {
  return source.authority == &ordered_endpoint && source.identity == owner.get();
}

const RegionProviderVTable *native_region_provider() noexcept { return &region_table; }
} // namespace hyle::graph::vulkan::detail
